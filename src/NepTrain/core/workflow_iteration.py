"""Production workflow Adapter for the deterministic generation controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from .candidate_pool import (
    CandidatePoolError,
    regular_batch_minimum,
    validate_candidate_pool,
    write_candidate_pool,
)
from .content_addressing import file_sha256
from .descriptor_features import (
    ELEMENTWISE_MEAN_STD,
    GLOBAL_MEAN,
    descriptor_elements,
    reduce_atomic_descriptors,
)
from .fps import (
    adaptive_novelty_threshold,
    hierarchical_farthest_point_sampling,
)
from .labeling import LabelRequest, LabelResult, label
from .iteration import (
    StageContext,
    StageOutcome,
)
from .md import MdError, MdRequest, MdResult, run_md
from .nep.calculator import DescriptorCalculator, Nep3Calculator
from .reporting import (
    ParitySeries,
    build_evaluation_report,
    build_parity_report,
)
from .persistence import atomic_write_json
from .scenario import ScenarioLadder
from .sampling_route import (
    SamplingRoute,
    SamplingRouteError,
    load_sampling_routes,
)
from .scientific_data import (
    ScientificDataError,
    labeled_input_structure_ids,
    reference_forces,
    structure_id,
    validate_labeled_frames,
)
from .spin import validate_spin_dataset
from .training import TrainingRequest, TrainingResult, train


class WorkflowIterationError(RuntimeError):
    """Raised when a real workflow stage cannot satisfy the iteration contract."""


TrainRunner = Callable[[TrainingRequest, str], TrainingResult]
MdRunner = Callable[[MdRequest, str], MdResult]
LabelRunner = Callable[[LabelRequest, str], LabelResult]
DescriptorRunner = Callable[[Path, Sequence[Atoms]], np.ndarray]
AtomicDescriptorRunner = Callable[[Path, Sequence[Atoms]], np.ndarray]


@dataclass(frozen=True)
class PredictionEvaluation:
    """Metrics and paired values from one model inference pass."""

    metrics: Mapping[str, float]
    comparisons: Mapping[str, ParitySeries] = field(default_factory=dict)


PredictionRunner = Callable[
    [Path, Sequence[Atoms], str],
    PredictionEvaluation,
]
_DESCRIPTOR_BATCH_SIZE = 4096
_CANDIDATE_VALIDATION_REGRESSION_FACTOR = 1.02
_PREDICTION_METRIC_BASIS = "per_atom_v1"


def _nep_descriptors(model: Path, frames: Sequence[Atoms]) -> np.ndarray:
    calculator = DescriptorCalculator("nep", model_file=model)
    try:
        return np.asarray(calculator.get_structures_descriptors(frames), dtype=np.float64)
    finally:
        calculator.calculator.close()


def _nep_atomic_descriptors(
    model: Path, frames: Sequence[Atoms]
) -> np.ndarray:
    calculator = DescriptorCalculator("nep", model_file=model)
    try:
        return np.asarray(
            calculator.get_structures_atomic_descriptors(frames),
            dtype=np.float64,
        )
    finally:
        calculator.calculator.close()


def _batched_descriptors(
    runner: DescriptorRunner,
    model: Path,
    frames: Sequence[Atoms],
) -> np.ndarray:
    """Describe every frame while bounding one backend call's working set."""

    chunks = []
    width = None
    for start in range(0, len(frames), _DESCRIPTOR_BATCH_SIZE):
        stop = min(start + _DESCRIPTOR_BATCH_SIZE, len(frames))
        values = np.asarray(runner(model, frames[start:stop]), dtype=np.float64)
        if values.ndim != 2 or len(values) != stop - start:
            raise WorkflowIterationError(
                "descriptor backend must return one feature row per frame"
            )
        if width is None:
            width = values.shape[1]
        elif values.shape[1] != width:
            raise WorkflowIterationError(
                "descriptor feature width changed between batches"
            )
        chunks.append(values)
    if not chunks:
        raise WorkflowIterationError("descriptor input cannot be empty")
    return np.vstack(chunks)


def _batched_elementwise_descriptors(
    runner: AtomicDescriptorRunner,
    model: Path,
    frames: Sequence[Atoms],
    *,
    elements: Sequence[str],
) -> np.ndarray:
    """Describe and reduce each frame batch without retaining all atomic rows."""

    chunks = []
    width = None
    for start in range(0, len(frames), _DESCRIPTOR_BATCH_SIZE):
        stop = min(start + _DESCRIPTOR_BATCH_SIZE, len(frames))
        batch = frames[start:stop]
        values = np.asarray(runner(model, batch), dtype=np.float64)
        expected_atoms = sum(len(frame) for frame in batch)
        if values.ndim != 2 or len(values) != expected_atoms:
            raise WorkflowIterationError(
                "atomic descriptor backend must return one feature row per atom"
            )
        if width is None:
            width = values.shape[1]
        elif values.shape[1] != width:
            raise WorkflowIterationError(
                "atomic descriptor feature width changed between batches"
            )
        chunks.append(
            reduce_atomic_descriptors(
                batch,
                values,
                reduction=ELEMENTWISE_MEAN_STD,
                elements=elements,
            )
        )
    if not chunks:
        raise WorkflowIterationError("descriptor input cannot be empty")
    return np.vstack(chunks)


def _structure_descriptors(
    runtime: "WorkflowRuntime",
    model: Path,
    frames: Sequence[Atoms],
    *,
    reduction: str,
    elements: Sequence[str],
) -> np.ndarray:
    if reduction == GLOBAL_MEAN:
        return _batched_descriptors(runtime.descriptors, model, frames)
    if reduction == ELEMENTWISE_MEAN_STD:
        return _batched_elementwise_descriptors(
            runtime.atomic_descriptors,
            model,
            frames,
            elements=elements,
        )
    raise WorkflowIterationError(
        f"unsupported sampling descriptor reduction: {reduction}"
    )


def _rmse(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if reference.shape != prediction.shape:
        raise WorkflowIterationError(
            f"prediction shape {prediction.shape} does not match labels {reference.shape}"
        )
    return float(np.sqrt(np.mean(np.square(prediction - reference))))


def _r2(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Return a deterministic R2, including constant-reference slices."""

    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if reference.shape != prediction.shape or not len(reference):
        raise WorkflowIterationError("R2 requires non-empty paired values")
    residual = float(np.sum(np.square(prediction - reference)))
    total = float(np.sum(np.square(reference - np.mean(reference))))
    if total <= np.finfo(np.float64).eps * max(1, len(reference)):
        return 1.0 if residual <= np.finfo(np.float64).eps else 0.0
    return float(1.0 - residual / total)


def _comparison_quality(series: ParitySeries) -> dict[str, float]:
    reference = np.asarray(series.reference, dtype=np.float64).reshape(-1)
    prediction = np.asarray(series.predicted, dtype=np.float64).reshape(-1)
    scale = float(np.std(reference))
    tolerance = max(3.0 * scale, np.finfo(np.float64).eps)
    return {
        "r2": _r2(reference, prediction),
        "outlier_fraction": float(
            np.mean(np.abs(prediction - reference) > tolerance)
        ),
    }


def _evaluation_quality(
    evaluation: PredictionEvaluation,
    frames: Sequence[Atoms],
) -> dict[str, Any]:
    quality: dict[str, Any] = {
        "r2": {},
        "outlier_fraction": {},
        "element_force_r2": {},
    }
    for name, series in evaluation.comparisons.items():
        values = _comparison_quality(series)
        quality["r2"][f"{name}_r2"] = values["r2"]
        quality["outlier_fraction"][name] = values["outlier_fraction"]
    force = evaluation.comparisons.get("force")
    if force is not None:
        symbols = np.asarray(
            [symbol for frame in frames for symbol in frame.get_chemical_symbols()]
        )
        component_symbols = np.repeat(symbols, 3)
        if len(component_symbols) != len(force.reference):
            raise WorkflowIterationError(
                "force parity values do not match the labeled atom count"
            )
        for symbol in sorted(set(symbols)):
            mask = component_symbols == symbol
            quality["element_force_r2"][symbol] = _r2(
                force.reference[mask], force.predicted[mask]
            )
    return quality


def _within_thresholds(
    metrics: Mapping[str, float], thresholds: Mapping[str, Any]
) -> bool:
    return bool(metrics) and all(
        name in metrics
        and np.isfinite(float(metrics[name]))
        and float(metrics[name]) <= float(limit)
        for name, limit in thresholds.items()
    )


def _threshold_score(
    metrics: Mapping[str, float], thresholds: Mapping[str, Any]
) -> float:
    missing = sorted(name for name in thresholds if name not in metrics)
    if missing:
        raise WorkflowIterationError(
            "model evaluation is missing metrics: " + ", ".join(missing)
        )
    return max(
        (
            float(metrics[name]) / float(limit)
            for name, limit in thresholds.items()
            if name in metrics
        ),
        default=float("inf"),
    )


def _acquisition_convergence_status(
    diagnostic: Mapping[str, Any],
    policy: Mapping[str, Any],
    previous_signals: Mapping[str, Any],
) -> dict[str, Any]:
    """Judge convergence from errors made before the new labels are trained."""

    if not policy:
        return {
            "acquisition_convergence_configured": False,
            "acquisition_converged": False,
            "acquisition_convergence_streak": 0,
        }
    thresholds = dict(policy.get("acquisition_max_rmse", {}))
    r2_thresholds = dict(policy.get("acquisition_min_r2", {}))
    metrics = {name: diagnostic.get(f"current_model_{name}") for name in thresholds}
    r2_metrics = {name: diagnostic.get(f"current_model_{name}") for name in r2_thresholds}

    def metrics_accepted(values: Mapping[str, Any]) -> bool:
        return all(values.get(name) is not None for name in thresholds) and (
            _within_thresholds(values, thresholds)
        )

    aggregate_rmse_accepted = metrics_accepted(metrics) if thresholds else True
    aggregate_r2_accepted = all(
        r2_metrics.get(name) is not None
        and np.isfinite(float(r2_metrics[name]))
        and float(r2_metrics[name]) >= float(limit)
        for name, limit in r2_thresholds.items()
    )
    attempt_metrics = diagnostic.get("attempt_metrics", {})
    attempts_accepted = all(
        metrics_accepted(values)
        for values in attempt_metrics.values()
    ) if thresholds else True
    group_min = policy.get("group_min_force_r2")
    element_r2 = dict(diagnostic.get("element_force_r2", {}))
    condition_r2 = dict(diagnostic.get("condition_force_r2", {}))
    groups_accepted = bool(
        group_min is None
        or (
            element_r2
            and condition_r2
            and all(float(value) >= float(group_min) for value in element_r2.values())
            and all(float(value) >= float(group_min) for value in condition_r2.values())
        )
    )
    max_outliers = policy.get("max_outlier_fraction")
    outliers = dict(diagnostic.get("outlier_fraction", {}))
    outliers_accepted = bool(
        max_outliers is None
        or (
            outliers
            and all(float(value) <= float(max_outliers) for value in outliers.values())
        )
    )
    min_selected = int(policy.get("min_selected", 0))
    enough_evidence = int(diagnostic.get("evaluated_count", 0)) >= min_selected
    acquisition_accepted = bool(
        aggregate_rmse_accepted
        and aggregate_r2_accepted
        and attempts_accepted
        and groups_accepted
        and outliers_accepted
        and enough_evidence
    )
    previous_streak = int(
        previous_signals.get("acquisition_convergence_streak", 0)
    )
    streak = previous_streak + 1 if acquisition_accepted else 0
    required = int(policy.get("consecutive_generations", 1))
    return {
        "acquisition_convergence_configured": True,
        "acquisition_metrics": metrics,
        "acquisition_r2": r2_metrics,
        "acquisition_max_rmse": thresholds,
        "acquisition_min_r2": r2_thresholds,
        "acquisition_groups_accepted": groups_accepted,
        "acquisition_outliers_accepted": outliers_accepted,
        "acquisition_evidence_count": int(diagnostic.get("evaluated_count", 0)),
        "acquisition_min_selected": min_selected,
        "acquisition_accepted": acquisition_accepted,
        "acquisition_convergence_streak": streak,
        "acquisition_convergence_required": required,
        "acquisition_converged": streak >= required,
    }


def _nep_prediction_evaluation(
    model: Path, frames: Sequence[Atoms], backend: str
) -> PredictionEvaluation:
    if not frames:
        raise WorkflowIterationError("evaluation requires at least one labeled frame")
    spin = "spin" in frames[0].arrays
    with Nep3Calculator(model, backend=backend) as calculator:
        if spin:
            energy, forces, virials, mforces = calculator.calculate_spin(
                frames, mean_virial=True
            )
        else:
            energy, forces, virials = calculator.calculate(
                frames, mean_virial=True
            )
            mforces = None
    atom_counts = np.asarray([len(frame) for frame in frames], dtype=float)
    reference_energy = np.asarray(
        [frame.get_potential_energy() for frame in frames]
    ) / atom_counts
    predicted_energy = np.asarray(energy).reshape(-1) / atom_counts
    reference_force = np.concatenate(
        [reference_forces(frame) for frame in frames]
    )
    predicted_force = np.concatenate(forces)
    reference_virial = np.asarray(
        [frame.info["virial"] for frame in frames]
    )
    reference_virial = reference_virial / atom_counts.reshape(
        (-1,) + (1,) * (reference_virial.ndim - 1)
    )
    predicted_virial = np.asarray(virials)
    metrics = {
        "energy_rmse": _rmse(reference_energy, predicted_energy),
        "force_rmse": _rmse(reference_force, predicted_force),
        "virial_rmse": _rmse(reference_virial, predicted_virial),
    }
    comparisons = {
        "energy": ParitySeries(
            reference_energy,
            predicted_energy,
            "eV/atom",
        ),
        "force": ParitySeries(
            reference_force,
            predicted_force,
            "eV/Å",
        ),
        "virial": ParitySeries(
            reference_virial,
            predicted_virial,
            "eV/atom",
        ),
    }
    if spin:
        reference_mforce = np.concatenate(
            [frame.arrays["mforce"] for frame in frames]
        )
        predicted_mforce = np.concatenate(mforces)
        metrics["mforce_rmse"] = _rmse(
            reference_mforce,
            predicted_mforce,
        )
        comparisons["mforce"] = ParitySeries(
            reference_mforce,
            predicted_mforce,
            "eV/μB",
        )
    return PredictionEvaluation(metrics, comparisons)


@dataclass(frozen=True)
class WorkflowRuntime:
    """Narrow dependency seam used by tests; defaults are the real backends."""

    train: TrainRunner = train
    md: MdRunner = run_md
    label: LabelRunner = label
    descriptors: DescriptorRunner = _nep_descriptors
    atomic_descriptors: AtomicDescriptorRunner = _nep_atomic_descriptors
    predict: PredictionRunner = _nep_prediction_evaluation


def _read_frames(path: Path, *, allow_empty: bool = False) -> list[Atoms]:
    paths = (
        sorted(
            item
            for pattern in ("*.xyz", "*.extxyz", "*.vasp", "POSCAR*")
            for item in path.glob(pattern)
        )
        if path.is_dir()
        else [path]
    )
    frames: list[Atoms] = []
    for item in paths:
        if allow_empty and item.is_file() and item.stat().st_size == 0:
            continue
        loaded = ase_read(item, index=":", format=None)
        frames.extend(loaded if isinstance(loaded, list) else [loaded])
    if not frames and not allow_empty:
        raise WorkflowIterationError(f"no structures found in {path}")
    return frames


def _condition_stratum(stratum: str) -> str:
    """Drop transient trajectory-window identity from a physical condition."""

    return "|".join(
        part for part in str(stratum).split("|") if not part.startswith("W=")
    )


class WorkflowIterationAdapter:
    """Connect real training, MD, NEP descriptor, and labeling Interfaces.

    The controller owns stage ordering and resumability. This class only maps
    one stage to the existing production Interfaces, so the caller can run
    ``train`` on a GPU job and ``explore`` on a separate CPU job.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        initial_training: str | Path | None,
        base_dir: str | Path = ".",
        runtime: WorkflowRuntime | None = None,
        active_stage: str | None = None,
        active_generation_kind: str = "legacy",
    ):
        self.config = dict(config)
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.runtime = runtime or WorkflowRuntime()
        requires_routes = active_stage is None or active_stage == "explore" or (
            active_stage == "evaluate" and active_generation_kind == "legacy"
        )
        if requires_routes:
            try:
                self.routes = load_sampling_routes(
                    self.config.get("sampling", {}),
                    base_dir=self.base_dir,
                )
            except SamplingRouteError as error:
                raise WorkflowIterationError(str(error)) from error
        else:
            self.routes = ()
        self.scenario_ladders = {
            route.route_id: ScenarioLadder.from_sampling(
                {
                    "conditions": route.conditions,
                    "progression": route.progression,
                }
            )
            for route in self.routes
        }
        requires_initial = active_stage is None or (
            active_stage == "train" and initial_training is not None
        )
        self.initial_training = (
            self._path(initial_training) if requires_initial else None
        )
        if requires_initial and not self.initial_training.is_file():
            raise WorkflowIterationError(
                f"initial training set does not exist: {self.initial_training}"
            )
        self.evaluation_configured = "evaluation" in self.config
        evaluation = self.config.get("evaluation", {})
        validation_value = (
            evaluation.get("validation_path")
            if self.evaluation_configured
            else None
        )
        requires_validation = active_stage is None or active_stage == "evaluate"
        if (
            requires_validation
            and self.evaluation_configured
            and not validation_value
        ):
            raise WorkflowIterationError(
                "configured evaluation requires evaluation.validation_path"
            )
        self.validation = (
            self._path(validation_value)
            if requires_validation and validation_value
            else None
        )
        if self.validation is not None and not self.validation.is_file():
            raise WorkflowIterationError(
                f"validation dataset does not exist: {self.validation}"
            )
        thresholds = evaluation.get("max_rmse") if self.evaluation_configured else None
        required_thresholds = {"energy_rmse", "force_rmse"}
        if self.config.get("md", {}).get("spin", False):
            required_thresholds.add("mforce_rmse")
        missing_thresholds = sorted(required_thresholds - set(thresholds or {}))
        requires_thresholds = active_stage is None or active_stage in {
            "diagnose",
            "evaluate",
        }
        if (
            requires_thresholds
            and self.evaluation_configured
            and missing_thresholds
        ):
            raise WorkflowIterationError(
                "post-retrain acceptance requires evaluation.max_rmse for "
                + ", ".join(missing_thresholds)
            )
        requires_training_config = (
            active_stage is None
            or active_stage == "retrain"
            or (
                active_stage == "train"
                and (
                    initial_training is not None
                    or active_generation_kind in {"acquisition", "finalization"}
                )
            )
        )
        if (
            requires_training_config
            and self.config.get("md", {}).get("spin", False)
        ):
            config_path = self._path(self.config.get("training", {}).get("config_path"))
            text = config_path.read_text(encoding="utf-8")
            if not re.search(r"^\s*spin_descriptor\s+spin_nep_lite\s*$", text, re.MULTILINE):
                raise WorkflowIterationError(
                    "spin workflow training config must use spin_descriptor spin_nep_lite"
                )

    def _path(self, value: str | Path | None) -> Path:
        if value is None:
            raise WorkflowIterationError("required workflow path is missing")
        path = Path(value).expanduser()
        return (self.base_dir / path).resolve() if not path.is_absolute() else path.resolve()

    def _optional_path(self, value: str | Path | None) -> Path | None:
        if value in {None, "", "auto"}:
            return None
        return self._path(value)

    @staticmethod
    def _labeling_kpoints(
        options: Mapping[str, Any],
    ) -> tuple[int, int, int]:
        if options.get("kpoint_mode", "auto") != "kpoints":
            return (1, 1, 1)
        raw = options.get("kpoints", (1, 1, 1))
        values = [int(raw)] if isinstance(raw, int | float | str) else [int(value) for value in raw]
        if len(values) == 1:
            values *= 3
        if len(values) != 3 or any(value < 1 for value in values):
            raise WorkflowIterationError(
                "labeling.kpoints must contain one or three positive integers"
            )
        return tuple(values)

    def run_stage(self, stage: str, context: StageContext) -> StageOutcome:
        if context.generation_kind in {"acquisition", "finalization"}:
            method = getattr(self, f"_{context.generation_kind}_{stage}", None)
            if method is None:
                raise WorkflowIterationError(
                    f"stage {stage} is not valid for a {context.generation_kind} generation"
                )
            return method(context)
        method = getattr(self, f"_{stage}", None)
        if method is None:
            raise WorkflowIterationError(f"unsupported workflow stage: {stage}")
        outcome = method(context)
        if (
            stage == "evaluate"
            and context.stage_input.get("generation_protocol") == "adaptive_v2"
        ):
            return self._defer_legacy_convergence(context, outcome)
        return outcome

    def _execute_training(
        self,
        context: StageContext,
        *,
        training_input: Path,
        role: str,
        warm_start: Path | None,
    ) -> tuple[TrainingResult, int, Path]:
        options = self.config.get("training", {})
        backend = str(options.get("backend", "gpumd"))
        config_file = self._path(options.get("config_path"))
        output_dir = (
            context.work_dir
            if context.flat_output
            else context.work_dir / role
        )
        if backend == "torchnep" and warm_start is not None:
            config_file = self._torchnep_finetune_config(
                config_file, output_dir, options
            )
        request = TrainingRequest(
            config_file=config_file,
            train_file=training_input,
            output_dir=output_dir,
            test_file=self._path(options["test_path"])
            if backend != "torchnep" and options.get("test_path")
            else None,
            restart_file=warm_start
            if backend == "gpumd" and options.get("restart", True)
            else None,
            finetune_file=warm_start
            if backend == "torchnep" and options.get("restart", True)
            else None,
            continue_steps=int(options.get("restart_steps", 10_000)),
            device=str(options.get("device", "cuda")),
            torch_backend=str(options.get("torch_backend", "auto")),
            precision=str(options.get("precision", "float32")),
            use_compile=bool(options.get("use_compile", False)),
            seed=int(
                self.config.get("workflow", {}).get("seed", 20260723)
            ),
        )
        result = self.runtime.train(request, backend)
        return result, len(_read_frames(training_input)), config_file

    @staticmethod
    def _torchnep_finetune_config(
        source: Path,
        generation_dir: Path,
        options: Mapping[str, Any],
    ) -> Path:
        """Lower only the incremental TorchNEP learning rate.

        A newly added high-gradient minibatch can destroy an otherwise good
        checkpoint before TorchNEP records its first best model. Keep initial
        training unchanged and make the safer fine-tune rate explicit in the
        generated artifact for reproducibility.
        """

        scale = float(options.get("finetune_lr_scale", 0.1))
        explicit = options.get("finetune_lr")
        text = source.read_text(encoding="utf-8")
        output = []
        replaced = False
        for line in text.splitlines(keepends=True):
            newline = "\n" if line.endswith("\n") else ""
            body = line[:-1] if newline else line
            match = re.match(r"^(\s*lr\s+)(\S+)(.*)$", body)
            if match is None:
                output.append(line)
                continue
            current = float(match.group(2))
            value = float(explicit) if explicit is not None else current * scale
            output.append(
                f"{match.group(1)}{value:.12g}{match.group(3)}{newline}"
            )
            replaced = True
        if not replaced:
            return source
        path = generation_dir / "torchnep-finetune.in"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(output), encoding="utf-8")
        return path

    def _adaptive_train(self, context: StageContext) -> StageOutcome:
        """Train exactly one model on the dataset entering this generation."""

        if context.generation == 1:
            if self.initial_training is None:
                raise WorkflowIterationError(
                    "the first adaptive generation requires an initial training set"
                )
            training_input = self.initial_training
            warm_start = None
            parent_model_sha256 = None
        else:
            training_input = context.previous_artifacts.get("training_set")
            if training_input is None:
                raise WorkflowIterationError(
                    "the previous generation did not publish a merged training set"
                )
            parent_model = context.previous_artifacts.get("activated_model")
            parent_model_sha256 = (
                file_sha256(parent_model) if parent_model is not None else None
            )
            warm_start = context.previous_artifacts.get("activated_checkpoint")
        result, frame_count, _ = self._execute_training(
            context,
            training_input=training_input,
            role="training",
            warm_start=warm_start,
        )
        candidate_sha256 = file_sha256(result.best_model)
        lineage = {
            "version": 2,
            "generation": context.generation,
            "generation_kind": context.generation_kind,
            "parent_model_sha256": parent_model_sha256,
            "candidate_model_sha256": candidate_sha256,
            "training_dataset_sha256": file_sha256(training_input),
            "training_count": frame_count,
            "trained_on_generation_input": True,
        }
        artifacts: dict[str, Path] = {
            "training_input": training_input,
            "model_training_set": training_input,
            "candidate_model": result.best_model,
            "model_lineage": atomic_write_json(
                context.work_dir / "model-lineage.json", lineage
            ),
        }
        if context.generation > 1 and parent_model is not None:
            artifacts["parent_model"] = parent_model
        if result.final_model is not None:
            artifacts["candidate_final_model"] = result.final_model
        if result.checkpoint is not None:
            artifacts["candidate_checkpoint"] = result.checkpoint
        artifacts.update(
            {
                f"training_output_{name.replace('.', '_')}": path
                for name, path in result.outputs.items()
            }
        )
        return StageOutcome(
            artifacts=artifacts,
            metrics={
                "backend": result.backend,
                "training_count": frame_count,
                "training_dataset_sha256": file_sha256(training_input),
                "parent_model_sha256": parent_model_sha256,
                "candidate_model_sha256": candidate_sha256,
                "generation_kind": context.generation_kind,
            },
        )

    def _adaptive_evaluate(self, context: StageContext) -> StageOutcome:
        """Qualify the generation model before it can drive acquisition."""

        candidate = context.artifacts["candidate_model"]
        candidate_sha256 = file_sha256(candidate)
        training_input = context.artifacts["model_training_set"]
        lineage = json.loads(
            context.artifacts["model_lineage"].read_text(encoding="utf-8")
        )
        if (
            lineage.get("candidate_model_sha256") != candidate_sha256
            or lineage.get("training_dataset_sha256")
            != file_sha256(training_input)
        ):
            raise WorkflowIterationError(
                "adaptive evaluate received inconsistent model lineage"
            )

        options = self.config.get("evaluation", {})
        thresholds = dict(options.get("max_rmse", {}))
        evaluation: PredictionEvaluation | None = None
        metrics: dict[str, float] = {}
        validation_accepted: bool | None = None
        parent_metrics: dict[str, float] | None = None
        candidate_score: float | None = None
        parent_score: float | None = None
        candidate_limit: float | None = None
        evaluated_count = 0
        spin_count = 0
        if self.evaluation_configured:
            assert self.validation is not None
            frames = _read_frames(self.validation)
            _, spin_count = validate_spin_dataset(frames, require_mforce=True)
            training_ids = {
                structure_id(frame) for frame in _read_frames(training_input)
            }
            overlap = sum(structure_id(frame) in training_ids for frame in frames)
            if overlap:
                raise WorkflowIterationError(
                    f"validation dataset overlaps the model training set by {overlap} frames"
                )
            evaluation = self.runtime.predict(
                candidate,
                frames,
                str(options.get("inference_backend", "auto")),
            )
            metrics = {name: float(value) for name, value in evaluation.metrics.items()}
            evaluated_count = len(frames)
            validation_accepted = _within_thresholds(metrics, thresholds)
            candidate_score = _threshold_score(metrics, thresholds)
            parent = context.artifacts.get("parent_model")
            if parent is not None:
                parent_evaluation = self.runtime.predict(
                    parent,
                    frames,
                    str(options.get("inference_backend", "auto")),
                )
                parent_metrics = {
                    name: float(value)
                    for name, value in parent_evaluation.metrics.items()
                }
                parent_score = _threshold_score(parent_metrics, thresholds)
                parent_accepted = _within_thresholds(parent_metrics, thresholds)
                candidate_limit = (
                    1.0
                    if parent_accepted
                    else parent_score * _CANDIDATE_VALIDATION_REGRESSION_FACTOR
                )
                if candidate_score > candidate_limit:
                    raise WorkflowIterationError(
                        "trained candidate regressed on the independent validation set"
                    )
        finite = bool(candidate.is_file() and candidate.stat().st_size > 0) and all(
            np.isfinite(float(value)) for value in metrics.values()
        )
        if not finite:
            raise WorkflowIterationError(
                "trained candidate is empty or produced non-finite evaluation metrics"
            )
        accepted = bool(
            finite
            and (
                context.generation_kind != "finalization"
                or validation_accepted is not False
            )
        )
        active_lineage = {
            **lineage,
            "active_model_sha256": candidate_sha256,
            "candidate_activation_accepted": True,
            "activation_basis": (
                "finite independent validation metrics"
                if self.evaluation_configured
                else "successful training and non-empty model artifact"
            ),
            "validation_accepted": validation_accepted,
        }
        evaluation_record = {
            "version": 1,
            "generation": context.generation,
            "generation_kind": context.generation_kind,
            "model_sha256": candidate_sha256,
            "training_dataset_sha256": file_sha256(training_input),
            "metrics": metrics,
            "model_improved": bool(
                lineage.get("parent_model_sha256") != candidate_sha256
            ),
            "validation_accepted": validation_accepted,
            "parent_validation_metrics": parent_metrics,
            "candidate_validation_score": candidate_score,
            "parent_validation_score": parent_score,
            "candidate_validation_limit": candidate_limit,
            "evaluated_count": evaluated_count,
        }
        artifacts: dict[str, Path] = {
            "model": candidate,
            "activated_model": candidate,
            "active_model_lineage": atomic_write_json(
                context.work_dir / "active-model-lineage.json", active_lineage
            ),
            "model_evaluation": atomic_write_json(
                context.work_dir / "model-evaluation.json", evaluation_record
            ),
        }
        if "candidate_checkpoint" in context.artifacts:
            artifacts["activated_checkpoint"] = context.artifacts[
                "candidate_checkpoint"
            ]
        signals = {
            **metrics,
            "prediction_metric_basis": _PREDICTION_METRIC_BASIS,
            "accepted": accepted,
            "generation_kind": context.generation_kind,
            "evaluation_configured": self.evaluation_configured,
            "validation_accepted": validation_accepted,
            "evaluated_count": evaluated_count,
            "spin_frame_count": spin_count,
            "active_model_sha256": candidate_sha256,
            "model_training_set_sha256": file_sha256(training_input),
            "workflow_converged": context.generation_kind == "finalization" and accepted,
            "generation_disposition": (
                "finalize" if context.generation_kind == "finalization" else None
            ),
            "finalization_pending": False,
        }
        if context.generation_kind == "finalization":
            artifacts["training_set"] = training_input
            artifacts["signals"] = atomic_write_json(
                context.work_dir / "signals.json", signals
            )
        report = build_evaluation_report(
            context.work_dir,
            metrics=metrics,
            thresholds=thresholds,
        )
        artifacts["evaluation_report"] = report.report
        if report.chart is not None:
            artifacts["evaluation_chart"] = report.chart
        if evaluation is not None and evaluation.comparisons:
            parity = build_parity_report(
                context.work_dir,
                series=evaluation.comparisons,
                source={
                    "validation_name": self.validation.name if self.validation else None,
                    "candidate_model_sha256": candidate_sha256,
                    "evaluated_count": evaluated_count,
                },
            )
            artifacts["evaluation_parity_report"] = parity.report
            if parity.chart is not None:
                artifacts["evaluation_parity"] = parity.chart
        return StageOutcome(artifacts=artifacts, metrics=signals)

    def _acquisition_train(self, context: StageContext) -> StageOutcome:
        return self._adaptive_train(context)

    def _finalization_train(self, context: StageContext) -> StageOutcome:
        return self._adaptive_train(context)

    def _acquisition_evaluate(self, context: StageContext) -> StageOutcome:
        return self._adaptive_evaluate(context)

    def _finalization_evaluate(self, context: StageContext) -> StageOutcome:
        return self._adaptive_evaluate(context)

    def _acquisition_explore(self, context: StageContext) -> StageOutcome:
        return self._explore(context)

    def _acquisition_select(self, context: StageContext) -> StageOutcome:
        return self._select(context)

    def _acquisition_label(self, context: StageContext) -> StageOutcome:
        return self._label(context)

    def _acquisition_diagnose(self, context: StageContext) -> StageOutcome:
        return self._diagnose(context)

    def _train(self, context: StageContext) -> StageOutcome:
        if context.generation > 1:
            lineage_path = context.previous_artifacts.get(
                "active_model_lineage"
            )
            if lineage_path is None:
                raise WorkflowIterationError(
                    "the previous generation has no activated model lineage"
                )
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            previous_model = context.previous_artifacts["activated_model"]
            if (
                lineage.get("generation") != context.generation - 1
                or lineage.get("active_model_sha256")
                != file_sha256(previous_model)
            ):
                raise WorkflowIterationError(
                    "the previous round did not publish a valid active model lineage"
                )
            artifacts = {
                "training_input": context.previous_artifacts["training_set"],
                "model": previous_model,
            }
            if "activated_checkpoint" in context.previous_artifacts:
                artifacts["checkpoint"] = context.previous_artifacts[
                    "activated_checkpoint"
                ]
            frame_count = len(_read_frames(artifacts["training_input"]))
            return StageOutcome(
                artifacts=artifacts,
                metrics={
                    "backend": "reuse",
                    "training_count": frame_count,
                    "reused_previous_model": True,
                },
            )

        if self.initial_training is None:
            raise WorkflowIterationError(
                "train stage requires an initial training set"
            )
        training_input = self.initial_training
        result, frame_count, _ = self._execute_training(
            context,
            training_input=training_input,
            role="training",
            warm_start=None,
        )
        artifacts: dict[str, Path] = {
            "training_input": training_input,
            "model": result.best_model,
        }
        if result.final_model is not None:
            artifacts["final_model"] = result.final_model
        if result.checkpoint is not None:
            artifacts["checkpoint"] = result.checkpoint
        artifacts.update(
            {
                f"training_output_{name.replace('.', '_')}": path
                for name, path in result.outputs.items()
            }
        )
        return StageOutcome(
            artifacts=artifacts,
            metrics={
                "backend": result.backend,
                "training_count": frame_count,
                "reused_previous_model": False,
            },
        )

    def plan_explore_attempts(
        self, context: StageContext
    ) -> tuple[dict[str, Any], ...]:
        """Enumerate the complete MD frontier for the active model.

        The persistent controller uses these immutable identities to create
        one scheduler task per attempt.  Route progression remains the sole
        source of scientific conditions; execution concurrency is not part of
        this plan.
        """

        previous_route_histories: dict[str, Any] = {}
        if "scenario_maturity" in context.previous_artifacts:
            previous = json.loads(
                context.previous_artifacts["scenario_maturity"].read_text(
                    encoding="utf-8"
                )
            )
            if previous.get("version") != 3 or not isinstance(
                previous.get("routes"), dict
            ):
                raise WorkflowIterationError(
                    "unsupported multi-route scenario maturity history"
                )
            previous_route_histories = previous["routes"]
        model_id = file_sha256(context.artifacts["model"])
        planned: list[dict[str, Any]] = []
        for route_index, route in enumerate(self.routes):
            ladder = self.scenario_ladders[route.route_id]
            route_history_record = previous_route_histories.get(route.route_id)
            route_history = None
            if route_history_record is not None:
                if (
                    route_history_record.get("route_fingerprint")
                    != route.fingerprint
                ):
                    raise WorkflowIterationError(
                        f"sampling route {route.route_id!r} fingerprint changed"
                    )
                route_history = route_history_record.get("history")
            structure_ids = sorted(
                {
                    structure_id(frame)
                    for path in route.structure_paths
                    for frame in _read_frames(path)
                }
            )
            attempts = ladder.schedule(
                structure_ids,
                route_id=route.route_id,
                route_fingerprint=route.fingerprint,
                template_sha256=route.template_sha256,
                pressure=float(route.conditions["pressure"]),
                generation=context.generation,
                seed=context.plan.seed + route_index,
                limit=(
                    len(structure_ids)
                    * len(ladder.temperature_path)
                    * max(ladder.replicas.values())
                ),
                model_id=model_id,
                history=route_history,
            )
            planned.extend(
                {
                    "route_id": route.route_id,
                    "route_fingerprint": route.fingerprint,
                    "attempt_id": attempt.attempt_id,
                    "temperature": attempt.temperature,
                    "steps": attempt.steps,
                    "target_level": attempt.target_level,
                    "replica": attempt.replica,
                }
                for attempt in attempts
            )
        return tuple(
            sorted(
                planned,
                key=lambda item: (
                    item["route_id"],
                    item["attempt_id"],
                ),
            )
        )

    def _explore(self, context: StageContext) -> StageOutcome:
        options = self.config.get("md", {})
        sampling = self.config.get("sampling", {})
        candidate_pool = sampling.get("candidate_pool", {})
        backend = str(options.get("backend", "lammps"))
        regular_minimum = regular_batch_minimum(context.plan.max_selected)
        previous_route_histories: dict[str, Any] = {}
        if "scenario_maturity" in context.previous_artifacts:
            previous = json.loads(
                context.previous_artifacts["scenario_maturity"].read_text(
                    encoding="utf-8"
                )
            )
            if previous.get("version") != 3 or not isinstance(
                previous.get("routes"), dict
            ):
                raise WorkflowIterationError(
                    "unsupported multi-route scenario maturity history"
                )
            previous_route_histories = previous["routes"]

        model_id = file_sha256(context.artifacts["model"])
        requested_md_runs = 0
        available_md_runs = 0
        route_plans: list[dict[str, Any]] = []
        groups: list[tuple[str, list[Atoms]]] = []
        backend_details: set[str] = set()
        source_metadata: dict[str, dict[str, Any]] = {}
        attempt_results: list[dict[str, Any]] = []
        unique_candidate_ids: set[str] = set()
        route_metrics: list[dict[str, Any]] = []
        requested_attempt_ids = {
            str(item)
            for item in context.stage_input.get("attempt_ids", [])
        }

        for route_index, route in enumerate(self.routes):
            ladder = self.scenario_ladders[route.route_id]
            route_history_record = previous_route_histories.get(route.route_id)
            route_history = None
            if route_history_record is not None:
                if (
                    route_history_record.get("route_fingerprint")
                    != route.fingerprint
                ):
                    raise WorkflowIterationError(
                        f"sampling route {route.route_id!r} fingerprint changed"
                    )
                route_history = route_history_record.get("history")

            route_structures = [
                frame
                for path in route.structure_paths
                for frame in _read_frames(path)
            ]
            structure_by_id = {
                structure_id(atoms): atoms for atoms in route_structures
            }
            ordered_structure_ids = sorted(structure_by_id)
            pressure = float(route.conditions["pressure"])
            maximum_frontier = (
                len(ordered_structure_ids)
                * len(ladder.temperature_path)
                * max(ladder.replicas.values())
            )
            available_attempts = ladder.schedule(
                ordered_structure_ids,
                route_id=route.route_id,
                route_fingerprint=route.fingerprint,
                template_sha256=route.template_sha256,
                pressure=pressure,
                generation=context.generation,
                seed=context.plan.seed + route_index,
                limit=maximum_frontier,
                model_id=model_id,
                history=route_history,
            )
            if requested_attempt_ids:
                available_attempts = [
                    attempt
                    for attempt in available_attempts
                    if attempt.attempt_id in requested_attempt_ids
                ]
            requested_md_runs += len(available_attempts)
            available_md_runs += len(available_attempts)
            executed_attempts = []
            for attempt in available_attempts:
                executed_attempts.append(attempt)
                atoms = structure_by_id[attempt.structure_id]
                source = (
                    f"g{context.generation}-{route.route_id}-"
                    f"s{attempt.structure_id[:8]}-T{attempt.temperature:g}-"
                    f"P{pressure:g}-r{attempt.replica}-"
                    f"{attempt.attempt_id[:8]}"
                )
                run_dir = (
                    context.work_dir
                    if context.flat_output and len(available_attempts) == 1
                    else context.work_dir / "calculations" / source
                )
                request = MdRequest(
                    atoms=atoms.copy(),
                    model_file=context.artifacts["model"],
                    output_dir=run_dir,
                    output_file=run_dir / "trajectory.xyz",
                    temperature=attempt.temperature,
                    steps=attempt.steps,
                    seed=attempt.seed,
                    pressure=pressure,
                    replica=attempt.replica,
                    route_id=route.route_id,
                    route_fingerprint=route.fingerprint,
                    spin=bool(options.get("spin", False)),
                    spin_temperature=float(attempt.temperature)
                    if options.get("spin", False)
                    else None,
                    template_path=route.template_path,
                    inference_backend=str(
                        options.get("inference_backend", "auto")
                    ),
                    lmp_command=os.environ.get(
                        "NEPTRAIN_LMP_COMMAND", "lmp"
                    ),
                    mpiexec=os.environ.get(
                        "NEPTRAIN_MPIEXEC", "mpirun"
                    ),
                    mpi_ranks=int(
                        os.environ.get("SLURM_CPUS_PER_TASK", "1")
                    ),
                    pre_failure_frames=int(
                        candidate_pool.get("pre_failure_frames", 2)
                    ),
                    bad_tail_frames=int(
                        candidate_pool.get("bad_tail_frames", 1)
                    ),
                    health=dict(candidate_pool.get("health") or {}),
                )
                base_result = {
                    "source_id": source,
                    "route_id": route.route_id,
                    "route_fingerprint": route.fingerprint,
                    "template_sha256": route.template_sha256,
                    "structure_hash": attempt.structure_hash,
                    "scenario_id": attempt.scenario_id,
                    "scenario_attempt_id": attempt.attempt_id,
                    "seed": attempt.seed,
                    "replica": attempt.replica,
                    "temperature": attempt.temperature,
                    "pressure": pressure,
                    "sampling_model_sha256": model_id,
                }
                try:
                    result = self.runtime.md(request, backend)
                except MdError as error:
                    attempt_results.append(
                        {
                            **base_result,
                            "completed": False,
                            "usable_frames": 0,
                            "window_counts": {},
                            "last_step": None,
                            "failure_code": "md_error",
                            "failure_reason": str(error),
                        }
                    )
                    continue
                run_frames = _read_frames(result.trajectory)
                window_counts = {
                    window: sum(
                        str(frame.info.get("md_window", "stable_prefix"))
                        == window
                        for frame in run_frames
                    )
                    for window in (
                        "stable_prefix",
                        "pre_failure",
                        "bad_tail",
                    )
                }
                window_counts = {
                    key: count
                    for key, count in window_counts.items()
                    if count
                }
                usable_frames = [
                    frame
                    for frame in run_frames
                    if frame.info.get("md_window", "stable_prefix")
                    != "bad_tail"
                ]
                usable_frames.sort(
                    key=lambda frame: (
                        0
                        if frame.info.get("md_window") == "pre_failure"
                        else 1,
                        int(
                            frame.info.get(
                                "md_step",
                                frame.info.get("lammps_step", 0),
                            )
                        ),
                    )
                )
                if usable_frames:
                    groups.append((source, usable_frames))
                    unique_candidate_ids.update(
                        f"{route.fingerprint}:{structure_id(frame)}"
                        for frame in usable_frames
                    )
                source_metadata[source] = {
                    **base_result,
                    "structure_id": attempt.structure_id,
                    "maturity_target": attempt.target_level,
                    "md_steps": attempt.steps,
                    "completed": bool(result.completed),
                    "failure_code": result.failure_code,
                }
                backend_details.add(
                    result.inference_backend or result.backend
                )
                health_summary = (
                    json.loads(
                        result.health_report.read_text(encoding="utf-8")
                    )
                    if result.health_report is not None
                    else None
                )
                attempt_results.append(
                    {
                        **base_result,
                        "completed": bool(result.completed),
                        "usable_frames": len(usable_frames),
                        "window_counts": window_counts,
                        "last_step": result.last_step,
                        "failure_code": result.failure_code,
                        "failure_reason": result.failure_reason,
                        "health_report": str(result.health_report)
                        if result.health_report is not None
                        else None,
                        "health": health_summary,
                    }
                )
            route_plans.append(
                {
                    "route_id": route.route_id,
                    "route_fingerprint": route.fingerprint,
                    "template_sha256": route.template_sha256,
                    "structure_ids": ordered_structure_ids,
                    "pressure": pressure,
                    "attempts": ladder.serialize(executed_attempts),
                    "completed": {
                        item["scenario_attempt_id"]: bool(item["completed"])
                        for item in attempt_results
                        if item["route_id"] == route.route_id
                    },
                }
            )
            route_metrics.append(
                {
                    "route_id": route.route_id,
                    "route_fingerprint": route.fingerprint,
                    "template_sha256": route.template_sha256,
                    "structure_count": len(ordered_structure_ids),
                    "available_attempts": len(available_attempts),
                    "scheduled_attempts": len(executed_attempts),
                    "temperatures": list(ladder.temperature_path),
                    "pressure": pressure,
                }
            )

        candidates = [
            (source, frame_index, frame)
            for source, frames in groups
            for frame_index, frame in enumerate(frames)
        ]
        if not candidates:
            failed = sum(not item["completed"] for item in attempt_results)
            if context.stage_input.get("allow_empty", False):
                return StageOutcome(
                    artifacts={
                        "md_attempts": atomic_write_json(
                            context.work_dir / "md-attempts.json",
                            {
                                "version": 2,
                                "model_sha256": model_id,
                                "routes": route_metrics,
                                "attempts": attempt_results,
                            },
                        ),
                        "scenario_plan": atomic_write_json(
                            context.work_dir / "scenario-plan.json",
                            {
                                "version": 3,
                                "model_id": model_id,
                                "routes": route_plans,
                            },
                        ),
                    },
                    metrics={
                        "candidate_count": 0,
                        "scheduled_source_count": len(attempt_results),
                        "completed_source_count": 0,
                        "failed_source_count": failed,
                        "routes": route_metrics,
                    },
                )
            raise WorkflowIterationError(
                "MD exploration produced no safe candidate frames "
                f"across {len(attempt_results)} attempts ({failed} failed)"
            )
        output_frames = []
        for source, frame_index, frame in candidates:
            copied = frame.copy()
            metadata = source_metadata[source]
            copied.info.update(
                generation=context.generation,
                source_id=source,
                temperature=metadata["temperature"],
                pressure=metadata["pressure"],
                frame_step=int(
                    frame.info.get(
                        "md_step",
                        frame.info.get("lammps_step", frame_index),
                    )
                ),
                scenario_structure_id=metadata["structure_id"],
                structure_hash=metadata["structure_hash"],
                md_steps=metadata["md_steps"],
                md_seed=metadata["seed"],
                md_completed=metadata["completed"],
                route_id=metadata["route_id"],
                route_fingerprint=metadata["route_fingerprint"],
                template_sha256=metadata["template_sha256"],
                sampling_model_sha256=model_id,
            )
            if metadata["failure_code"] is not None:
                copied.info["md_failure_code"] = metadata["failure_code"]
            if metadata["scenario_id"] is not None:
                copied.info.update(
                    scenario_id=metadata["scenario_id"],
                    scenario_attempt_id=metadata["scenario_attempt_id"],
                    maturity_target=metadata["maturity_target"],
                    scenario_replica=metadata["replica"],
                )
            output_frames.append(copied)
        output = context.work_dir / "candidates.xyz"
        pool_manifest_path = context.work_dir / "candidate-pool.json"
        failed_md_runs = sum(
            not bool(item["completed"]) for item in attempt_results
        )
        try:
            pool_manifest = write_candidate_pool(
                output,
                pool_manifest_path,
                output_frames,
                generation=context.generation,
                model_path=context.artifacts["model"],
                requested_md_runs=requested_md_runs,
                available_md_runs=available_md_runs,
                scheduled_md_runs=len(attempt_results),
                failed_md_runs=failed_md_runs,
            )
        except CandidatePoolError as error:
            raise WorkflowIterationError(str(error)) from error
        artifacts = {
            "candidates": output,
            "candidate_pool_manifest": pool_manifest_path,
            "md_attempts": atomic_write_json(
                context.work_dir / "md-attempts.json",
                {
                    "version": 2,
                    "model_sha256": model_id,
                    "routes": route_metrics,
                    "attempts": attempt_results,
                },
            ),
        }
        scenario_plan = atomic_write_json(
            context.work_dir / "scenario-plan.json",
            {
                "version": 3,
                "model_id": model_id,
                "routes": route_plans,
            },
        )
        artifacts["scenario_plan"] = scenario_plan
        return StageOutcome(
            artifacts=artifacts,
            metrics={
                "backend": backend,
                "inference_backends": sorted(backend_details),
                "candidate_count": len(output_frames),
                "unique_candidate_count": len(unique_candidate_ids),
                "regular_batch_minimum": regular_minimum,
                "sampling_model_sha256": pool_manifest.model_sha256,
                "route_fingerprints": dict(
                    pool_manifest.route_fingerprints
                ),
                "source_count": len(groups),
                "scheduled_source_count": len(attempt_results),
                "completed_source_count": sum(
                    bool(item["completed"]) for item in attempt_results
                ),
                "failed_source_count": sum(
                    not bool(item["completed"]) for item in attempt_results
                ),
                "routes": route_metrics,
                "candidate_counts_by_window": {
                    window: sum(
                        frame.info.get("md_window", "stable_prefix") == window
                        for frame in output_frames
                    )
                    for window in ("stable_prefix", "pre_failure")
                    if any(
                        frame.info.get("md_window", "stable_prefix") == window
                        for frame in output_frames
                    )
                },
                "scenario_steps": sorted(
                    {
                        int(attempt["steps"])
                        for route_plan in route_plans
                        for attempt in route_plan["attempts"]
                    }
                ),
                "scenario_temperatures": sorted(
                    {
                        float(attempt["temperature"])
                        for route_plan in route_plans
                        for attempt in route_plan["attempts"]
                    }
                ),
                "scenario_temperatures_by_route": {
                    str(route_plan["route_id"]): sorted(
                        {
                            float(attempt["temperature"])
                            for attempt in route_plan["attempts"]
                        }
                    )
                    for route_plan in route_plans
                },
                "scenario_targets": sorted(
                    {
                        str(attempt["target_level"])
                        for route_plan in route_plans
                        for attempt in route_plan["attempts"]
                    }
                ),
            },
        )

    def merge_explore_outcomes(
        self,
        context: StageContext,
        outcomes: Sequence[StageOutcome],
    ) -> StageOutcome:
        """Merge independently scheduled MD attempts into one explore stage."""

        frames: list[Atoms] = []
        attempts: list[dict[str, Any]] = []
        route_metric_parts: dict[str, list[dict[str, Any]]] = {}
        route_plan_parts: dict[str, list[dict[str, Any]]] = {}
        model_id = file_sha256(context.artifacts["model"])
        for outcome in outcomes:
            candidate_path = outcome.artifacts.get("candidates")
            if candidate_path is not None:
                frames.extend(_read_frames(candidate_path))
            attempts_path = outcome.artifacts.get("md_attempts")
            if attempts_path is None:
                raise WorkflowIterationError(
                    "MD attempt result is missing md_attempts provenance"
                )
            attempt_value = json.loads(
                attempts_path.read_text(encoding="utf-8")
            )
            if attempt_value.get("model_sha256") != model_id:
                raise WorkflowIterationError(
                    "MD wave mixed results from different sampling models"
                )
            attempts.extend(attempt_value.get("attempts", []))
            for item in attempt_value.get("routes", []):
                route_metric_parts.setdefault(
                    str(item["route_id"]), []
                ).append(item)
            scenario_path = outcome.artifacts.get("scenario_plan")
            if scenario_path is None:
                raise WorkflowIterationError(
                    "MD attempt result is missing scenario provenance"
                )
            scenario_value = json.loads(
                scenario_path.read_text(encoding="utf-8")
            )
            if scenario_value.get("model_id") != model_id:
                raise WorkflowIterationError(
                    "MD wave scenario plan belongs to another model"
                )
            for item in scenario_value.get("routes", []):
                route_plan_parts.setdefault(
                    str(item["route_id"]), []
                ).append(item)
        if not frames:
            failed = sum(not bool(item.get("completed")) for item in attempts)
            raise WorkflowIterationError(
                "MD exploration produced no safe candidate frames "
                f"across {len(attempts)} attempts ({failed} failed)"
            )

        output = context.work_dir / "candidates.xyz"
        pool_manifest_path = context.work_dir / "candidate-pool.json"
        failed_md_runs = sum(
            not bool(item.get("completed")) for item in attempts
        )
        pool_manifest = write_candidate_pool(
            output,
            pool_manifest_path,
            frames,
            generation=context.generation,
            model_path=context.artifacts["model"],
            requested_md_runs=len(outcomes),
            available_md_runs=len(outcomes),
            scheduled_md_runs=len(attempts),
            failed_md_runs=failed_md_runs,
        )

        route_metrics: list[dict[str, Any]] = []
        for route_id, parts in sorted(route_metric_parts.items()):
            first = parts[0]
            route_metrics.append(
                {
                    **first,
                    "available_attempts": sum(
                        int(item.get("available_attempts", 0))
                        for item in parts
                    ),
                    "scheduled_attempts": sum(
                        int(item.get("scheduled_attempts", 0))
                        for item in parts
                    ),
                }
            )
        route_plans: list[dict[str, Any]] = []
        for route_id, parts in sorted(route_plan_parts.items()):
            first = parts[0]
            route_plans.append(
                {
                    **first,
                    "structure_ids": sorted(
                        {
                            value
                            for item in parts
                            for value in item.get("structure_ids", [])
                        }
                    ),
                    "attempts": [
                        value
                        for item in parts
                        for value in item.get("attempts", [])
                    ],
                    "completed": {
                        key: value
                        for item in parts
                        for key, value in item.get("completed", {}).items()
                    },
                }
            )
        artifacts = {
            "candidates": output,
            "candidate_pool_manifest": pool_manifest_path,
            "md_attempts": atomic_write_json(
                context.work_dir / "md-attempts.json",
                {
                    "version": 2,
                    "model_sha256": model_id,
                    "routes": route_metrics,
                    "attempts": attempts,
                },
            ),
            "scenario_plan": atomic_write_json(
                context.work_dir / "scenario-plan.json",
                {
                    "version": 3,
                    "model_id": model_id,
                    "routes": route_plans,
                },
            ),
        }
        return StageOutcome(
            artifacts=artifacts,
            metrics={
                "backend": str(self.config.get("md", {}).get("backend", "lammps")),
                "candidate_count": len(frames),
                "unique_candidate_count": len(
                    {
                        (
                            str(frame.info.get("route_fingerprint", "")),
                            structure_id(frame),
                        )
                        for frame in frames
                    }
                ),
                "regular_batch_minimum": regular_batch_minimum(
                    context.plan.max_selected
                ),
                "sampling_model_sha256": pool_manifest.model_sha256,
                "route_fingerprints": dict(
                    pool_manifest.route_fingerprints
                ),
                "scheduled_source_count": len(attempts),
                "completed_source_count": sum(
                    bool(item.get("completed")) for item in attempts
                ),
                "failed_source_count": failed_md_runs,
                "routes": route_metrics,
                "md_wave_task_count": len(outcomes),
                "scenario_steps": sorted(
                    {
                        int(attempt["steps"])
                        for route_plan in route_plans
                        for attempt in route_plan["attempts"]
                    }
                ),
                "scenario_temperatures": sorted(
                    {
                        float(attempt["temperature"])
                        for route_plan in route_plans
                        for attempt in route_plan["attempts"]
                    }
                ),
                "scenario_temperatures_by_route": {
                    str(route_plan["route_id"]): sorted(
                        {
                            float(attempt["temperature"])
                            for attempt in route_plan["attempts"]
                        }
                    )
                    for route_plan in route_plans
                },
                "scenario_targets": sorted(
                    {
                        str(attempt["target_level"])
                        for route_plan in route_plans
                        for attempt in route_plan["attempts"]
                    }
                ),
            },
        )

    def _select(self, context: StageContext) -> StageOutcome:
        all_candidates = _read_frames(context.artifacts["candidates"])
        try:
            pool_manifest = validate_candidate_pool(
                all_candidates,
                context.artifacts["candidate_pool_manifest"],
                generation=context.generation,
                model_path=context.artifacts["model"],
            )
        except CandidatePoolError as error:
            raise WorkflowIterationError(str(error)) from error
        if "md_attempts" in context.artifacts:
            attempts = json.loads(
                context.artifacts["md_attempts"].read_text(encoding="utf-8")
            )["attempts"]
            failed_count = sum(
                not bool(item["completed"]) for item in attempts
            )
        else:
            failed_count = int(
                any(not bool(frame.info.get("md_completed", True)) for frame in all_candidates)
            )
        candidates = list(all_candidates)
        training = _read_frames(context.artifacts["training_input"])
        descriptor_reduction = self.config["sampling"]["selection"].get(
            "descriptor_reduction", GLOBAL_MEAN
        )
        training_elements = descriptor_elements(training)
        known_ids = {structure_id(frame) for frame in training}
        candidates = [
            frame for frame in candidates if structure_id(frame) not in known_ids
        ]
        if not candidates:
            selected_path = context.work_dir / "selected-input.xyz"
            selected_path.touch()
            result_path = atomic_write_json(
                context.work_dir / "selection-result.json",
                {
                    "selected_indices": [],
                    "selected_ids": [],
                    "selected_novelty": [],
                    "counts_by_stratum": {},
                    "remaining_novelty_by_stratum": {},
                    "remaining_novelty_by_condition": {},
                    "remaining_novelty": 0.0,
                    "groups": [],
                    "generation": context.generation,
                    "sampling_model_sha256": pool_manifest.model_sha256,
                    "route_fingerprints": dict(
                        pool_manifest.route_fingerprints
                    ),
                    "batch_ready": False,
                    "batch_kind": "coverage_complete",
                    "regular_minimum": regular_batch_minimum(
                        context.plan.max_selected
                    ),
                    "novelty_mode": "auto"
                    if self.config["sampling"]["selection"].get(
                        "novelty", "auto"
                    )
                    == "auto"
                    else "explicit",
                    "descriptor_reduction": descriptor_reduction,
                    "descriptor_elements": list(training_elements),
                    "resolved_selection_novelty_threshold": 0.0,
                    "resolved_completion_coverage_threshold": 0.0,
                },
            )
            return StageOutcome(
                artifacts={
                    "selected_input": selected_path,
                    "selection_result": result_path,
                },
                metrics={
                    "candidate_count_before_deduplication": len(
                        all_candidates
                    ),
                    "candidate_count_after_deduplication": 0,
                    "duplicate_candidate_count": 0,
                    "selected_count": 0,
                    "remaining_novelty": 0.0,
                    "configured_max_selected": context.plan.max_selected,
                    "regular_batch_minimum": regular_batch_minimum(
                        context.plan.max_selected
                    ),
                    "batch_ready": False,
                    "batch_kind": "coverage_complete",
                    "sampling_model_sha256": pool_manifest.model_sha256,
                    "route_fingerprints": dict(
                        pool_manifest.route_fingerprints
                    ),
                    "failed_md_attempt_count": failed_count,
                    "selection_novelty_threshold": 0.0,
                    "completion_coverage_threshold": 0.0,
                    "descriptor_reduction": descriptor_reduction,
                    "descriptor_elements": list(training_elements),
                    "novel_selected_count": 0,
                    "counts_by_stratum": {},
                    "fps_groups": [],
                },
            )
        def preference(frame: Atoms) -> tuple[str, ...]:
            return (
                "0" if frame.info.get("md_window") == "pre_failure" else "1",
                str(frame.info.get("route_fingerprint", "")),
                str(frame.info.get("route_id", "")),
                str(frame.info.get("temperature", "")),
                str(frame.info.get("pressure", "")),
                str(frame.info.get("source_id", "")),
                str(frame.info.get("scenario_attempt_id", "")),
            )

        unique_candidates: dict[str, Atoms] = {}
        for frame in candidates:
            identifier = structure_id(frame)
            route_fingerprint = str(frame.info.get("route_fingerprint", ""))
            if not route_fingerprint:
                raise WorkflowIterationError(
                    "candidate is missing its sampling route fingerprint"
                )
            current = unique_candidates.get(identifier)
            if current is None or preference(frame) < preference(current):
                # Labels depend on the physical input state, not on which
                # sampling route happened to encounter it.  Keep one stable
                # provenance representative so the label stage never receives
                # duplicate structures.
                unique_candidates[identifier] = frame
        duplicate_candidate_count = len(candidates) - len(unique_candidates)
        candidates = list(unique_candidates.values())
        candidate_ids = [
            f"{frame.info['route_fingerprint']}:{structure_id(frame)}"
            for frame in candidates
        ]
        strata = []
        for frame in candidates:
            base = (
                f"R={frame.info['route_id']}|"
                f"T={float(frame.info['temperature']):g}|"
                f"P={float(frame.info['pressure']):g}"
            )
            if "md_window" in frame.info:
                base = f"W={frame.info['md_window']}|{base}"
            strata.append(base)
        elements = descriptor_elements([*candidates, *training])
        candidate_descriptors = _structure_descriptors(
            self.runtime,
            context.artifacts["model"],
            candidates,
            reduction=descriptor_reduction,
            elements=elements,
        )
        reference_descriptors = _structure_descriptors(
            self.runtime,
            context.artifacts["model"],
            training,
            reduction=descriptor_reduction,
            elements=elements,
        )
        novelty = self.config["sampling"]["selection"].get(
            "novelty", "auto"
        )
        adaptive = novelty == "auto"
        selection_threshold = (
            adaptive_novelty_threshold(
                candidates,
                candidate_descriptors,
                training,
                reference_descriptors,
            )
            if adaptive
            else context.plan.selection_novelty_threshold
        )
        completion_threshold = (
            selection_threshold
            if adaptive
            else context.plan.completion_coverage_threshold
        )
        result = hierarchical_farthest_point_sampling(
            candidates,
            candidate_descriptors,
            candidate_ids,
            strata,
            budget=context.plan.max_selected,
            min_novelty=selection_threshold,
            reference_structures=training,
            reference_descriptors=reference_descriptors,
        )
        selected = [candidates[index] for index in result.selected_indices]
        selected_path = context.work_dir / "selected-input.xyz"
        if selected:
            ase_write(selected_path, selected, format="extxyz")
        else:
            selected_path.touch()
        regular_minimum = regular_batch_minimum(context.plan.max_selected)
        emergency_minimum = min(8, context.plan.max_selected)
        emergency = failed_count > 0
        batch_kind = (
            "coverage_complete"
            if not selected
            else (
                "regular"
                if len(selected) >= regular_minimum
                else (
                    "emergency"
                    if emergency and len(selected) >= emergency_minimum
                    else "novelty_flush"
                )
            )
        )
        remaining_by_condition: dict[str, float] = {}
        for stratum, value in result.remaining_novelty_by_stratum.items():
            condition = _condition_stratum(stratum)
            remaining_by_condition[condition] = max(
                remaining_by_condition.get(condition, 0.0),
                float(value),
            )
        result_path = atomic_write_json(
            context.work_dir / "selection-result.json",
            {
                "selected_indices": list(result.selected_indices),
                "selected_ids": list(result.selected_ids),
                "selected_novelty": list(result.selected_novelty),
                "counts_by_stratum": dict(result.counts_by_stratum),
                "remaining_novelty_by_stratum": dict(
                    result.remaining_novelty_by_stratum
                ),
                "remaining_novelty_by_condition": dict(
                    sorted(remaining_by_condition.items())
                ),
                "remaining_novelty": result.remaining_novelty,
                "groups": [
                    {
                        **asdict(report),
                        "element_set": list(report.element_set),
                    }
                    for _, report in sorted(result.groups.items())
                ],
                "generation": context.generation,
                "sampling_model_sha256": pool_manifest.model_sha256,
                "route_fingerprints": dict(
                    pool_manifest.route_fingerprints
                ),
                "batch_ready": bool(selected),
                "batch_kind": batch_kind,
                "regular_minimum": regular_minimum,
                "novelty_mode": "auto" if adaptive else "explicit",
                "descriptor_reduction": descriptor_reduction,
                "descriptor_elements": list(elements),
                "resolved_selection_novelty_threshold": (
                    selection_threshold
                ),
                "resolved_completion_coverage_threshold": (
                    completion_threshold
                ),
            },
        )
        return StageOutcome(
            artifacts={"selected_input": selected_path, "selection_result": result_path},
            metrics={
                "candidate_count_before_deduplication": len(all_candidates),
                "candidate_count_after_deduplication": len(candidates),
                "duplicate_candidate_count": duplicate_candidate_count,
                "selected_count": len(selected),
                "remaining_novelty": result.remaining_novelty,
                "configured_max_selected": context.plan.max_selected,
                "regular_batch_minimum": regular_minimum,
                "batch_ready": bool(selected),
                "batch_kind": batch_kind,
                "sampling_model_sha256": pool_manifest.model_sha256,
                "route_fingerprints": dict(
                    pool_manifest.route_fingerprints
                ),
                "failed_md_attempt_count": failed_count,
                "selection_novelty_threshold": selection_threshold,
                "completion_coverage_threshold": completion_threshold,
                "novelty_mode": "auto" if adaptive else "explicit",
                "descriptor_reduction": descriptor_reduction,
                "descriptor_elements": list(elements),
                "novel_selected_count": sum(
                    value > selection_threshold
                    for value in result.selected_novelty
                ),
                "counts_by_stratum": dict(result.counts_by_stratum),
                "remaining_novelty_by_condition": dict(
                    sorted(remaining_by_condition.items())
                ),
                "fps_groups": [
                    {
                        "element_set": list(report.element_set),
                        "candidate_count": report.candidate_count,
                        "reference_count": report.reference_count,
                        "initial_quota": report.initial_quota,
                        "selected_count": report.selected_count,
                        "remaining_novelty": report.remaining_novelty,
                        "remaining_novelty_by_stratum": dict(
                            report.remaining_novelty_by_stratum
                        ),
                    }
                    for _, report in sorted(result.groups.items())
                ],
            },
        )

    def _label(self, context: StageContext) -> StageOutcome:
        options = self.config.get("labeling", {})
        backend = str(options.get("backend", "toy"))
        kpoint_mode = str(options.get("kpoint_mode", "auto"))
        selected = _read_frames(
            context.artifacts["selected_input"], allow_empty=True
        )
        if not selected:
            output = context.work_dir / "selected-labels.xyz"
            output.touch()
            provenance_path = atomic_write_json(
                context.work_dir / "label-provenance.json",
                {
                    "backend": backend,
                    "origin": "no_novel_structures",
                    "input_structure_ids": [],
                    "labeled_count": 0,
                    "labels_sha256": file_sha256(output),
                },
            )
            return StageOutcome(
                artifacts={
                    "labeled": output,
                    "label_provenance": provenance_path,
                },
                metrics={
                    "backend": backend,
                    "origin": "no_novel_structures",
                    "labeled_count": 0,
                    "skipped": True,
                },
            )
        frame_indices = context.stage_input.get("frame_indices", ())
        flat_single_case = (
            context.flat_output
            and isinstance(frame_indices, Sequence)
            and not isinstance(frame_indices, str)
            and len(frame_indices) == 1
        )
        output = context.work_dir / "selected-labels.xyz"
        result = self.runtime.label(
            LabelRequest(
                source=context.artifacts["selected_input"],
                output_file=output,
                work_dir=(
                    context.work_dir
                    if flat_single_case
                    else context.work_dir / "calculation"
                ),
                settings={
                    "input_file": self._optional_path(
                        options.get("input_path")
                    ),
                    "resource_dir": self._optional_path(
                        options.get("resource_path")
                    ),
                    "resource_manifest": self._optional_path(
                        options.get("potcar_manifest_path")
                        or options.get("resource_manifest_path")
                    ),
                    "n_cpu": int(
                        os.environ.get("SLURM_CPUS_PER_TASK", "1")
                    ),
                    "use_gamma": bool(
                        options.get("gamma_centered", False)
                    ),
                    "kpoint_mode": kpoint_mode,
                    "kspacing": (
                        float(options["kspacing"])
                        if kpoint_mode == "kspacing"
                        else None
                    ),
                    "ka": self._labeling_kpoints(options),
                    "model_file": self._optional_path(
                        options.get("model_path")
                    ),
                    "model_name": options.get("model_name"),
                    "runner": options.get("runner"),
                    "device": options.get("device", "cuda"),
                    "precision": options.get("precision", "float32"),
                    "profile": (
                        "spin"
                        if self.config.get("md", {}).get("spin", False)
                        else "ordinary"
                    ),
                    "flat_single_case": flat_single_case,
                },
            ),
            backend,
        )
        expected_ids = [structure_id(frame) for frame in selected]
        try:
            validate_labeled_frames(result.frames)
            actual_ids = labeled_input_structure_ids(result.frames)
        except ScientificDataError as error:
            raise WorkflowIterationError(
                f"labeling backend produced invalid scientific labels: {error}"
            ) from error
        if actual_ids != expected_ids:
            raise WorkflowIterationError(
                "labeling backend changed or reordered selected structures"
            )
        provenance_path = context.work_dir / "label-provenance.json"
        atomic_write_json(
            provenance_path,
            {
                **dict(result.provenance),
                "input_structure_ids": expected_ids,
                "labeled_count": len(result.frames),
                "labels_sha256": file_sha256(result.output_file),
            },
        )
        return StageOutcome(
            artifacts={
                "labeled": result.output_file,
                "label_provenance": provenance_path,
            },
            metrics={
                "backend": result.backend,
                "origin": result.provenance.get("origin"),
                "labeled_count": len(result.frames),
            },
        )

    def merge_label_outcomes(
        self,
        context: StageContext,
        outcomes: Sequence[StageOutcome],
        *,
        successful_frame_indices: Sequence[int] | None = None,
        failures: Sequence[Mapping[str, Any]] = (),
    ) -> StageOutcome:
        """Merge independently labeled batches in their original frame order."""

        if not outcomes:
            raise WorkflowIterationError("label stage produced no batch results")
        requested = _read_frames(context.artifacts["selected_input"])
        if successful_frame_indices is None:
            successful_frame_indices = tuple(range(len(requested)))
        indices = [int(index) for index in successful_frame_indices]
        if (
            indices != sorted(set(indices))
            or any(index < 0 or index >= len(requested) for index in indices)
        ):
            raise WorkflowIterationError(
                "successful label frame indices must be unique, ordered, and in range"
            )
        expected = [requested[index] for index in indices]
        frames: list[Atoms] = []
        backends = set()
        provenances = []
        for outcome in outcomes:
            path = outcome.artifacts.get("labeled")
            if path is None:
                raise WorkflowIterationError(
                    "label batch result is missing the labeled artifact"
                )
            frames.extend(_read_frames(path))
            backend = outcome.metrics.get("backend")
            if backend:
                backends.add(str(backend))
            provenance_path = outcome.artifacts.get("label_provenance")
            if provenance_path is None:
                raise WorkflowIterationError(
                    "label batch result is missing label provenance"
                )
            provenances.append(
                json.loads(provenance_path.read_text(encoding="utf-8"))
            )
        if len(frames) != len(expected):
            raise WorkflowIterationError(
                f"label batches returned {len(frames)} structures for "
                f"{len(expected)} selected structures"
            )
        expected_ids = [structure_id(frame) for frame in expected]
        try:
            validate_labeled_frames(frames)
            actual_ids = labeled_input_structure_ids(frames)
        except ScientificDataError as error:
            raise WorkflowIterationError(
                f"label batch result violates the scientific data contract: {error}"
            ) from error
        if actual_ids != expected_ids:
            raise WorkflowIterationError(
                "label batch results do not match the selected structure order"
            )
        if len(backends) > 1:
            raise WorkflowIterationError(
                "label batches reported inconsistent labeling backends"
            )
        output = context.work_dir / "selected-labels.xyz"
        ase_write(output, frames, format="extxyz")
        backend = next(
            iter(backends),
            str(self.config.get("labeling", {}).get("backend", "toy")),
        )
        provenance_keys = {
            (
                value.get("backend"),
                value.get("origin"),
                value.get("engine"),
                value.get("model_sha256"),
            )
            for value in provenances
        }
        if len(provenance_keys) != 1:
            raise WorkflowIterationError(
                "label batches reported inconsistent provenance"
            )
        combined_provenance = {
            **{
                key: value
                for key, value in provenances[0].items()
                if key
                not in {
                    "input_structure_ids",
                    "labeled_count",
                    "labels_sha256",
                }
            },
            "input_structure_ids": actual_ids,
            "labeled_count": len(frames),
            "labels_sha256": file_sha256(output),
        }
        provenance_output = context.work_dir / "label-provenance.json"
        atomic_write_json(provenance_output, combined_provenance)
        artifacts = {
            "labeled": output,
            "label_provenance": provenance_output,
        }
        failed_frame_indices = sorted(
            {
                int(index)
                for failure in failures
                for index in failure.get("frame_indices", ())
            }
        )
        if failures:
            failure_output = context.work_dir / "label-failures.json"
            atomic_write_json(
                failure_output,
                {
                    "version": 1,
                    "requested_count": len(requested),
                    "labeled_count": len(frames),
                    "failures": list(failures),
                },
            )
            artifacts["label_failures"] = failure_output
        return StageOutcome(
            artifacts=artifacts,
            metrics={
                "backend": backend,
                "requested_count": len(requested),
                "labeled_count": len(frames),
                "batch_count": len(outcomes),
                "failed_batch_count": len(failures),
                "failed_frame_count": len(failed_frame_indices),
                "failed_frame_indices": failed_frame_indices,
                "partial": bool(failures),
            },
        )

    def _diagnose(self, context: StageContext) -> StageOutcome:
        options = self.config.get("evaluation", {})
        convergence_policy = self.config.get("workflow", {}).get(
            "convergence", {}
        )
        thresholds = dict(
            convergence_policy.get("acquisition_max_rmse", {})
            or options.get("max_rmse", {})
        )
        quality_gate_configured = bool(thresholds)
        frames = _read_frames(context.artifacts["labeled"], allow_empty=True)
        if not frames:
            signals = {
                "diagnostic_only": True,
                "quality_gate_configured": quality_gate_configured,
                "diagnostic_accepted": None,
                "attempt_accepted": {},
                "attempt_metrics": {},
                "evaluated_count": 0,
                "spin_frame_count": 0,
                "skipped": True,
                "reason": "no novel structures required labeling",
            }
            output = atomic_write_json(
                context.work_dir / "acquisition-signals.json", signals
            )
            return StageOutcome(
                artifacts={"acquisition_signals": output}, metrics=signals
            )
        _, spin_count = validate_spin_dataset(frames, require_mforce=True)
        evaluation = self.runtime.predict(
            context.artifacts["model"],
            frames,
            str(options.get("inference_backend", "auto")),
        )
        raw_metrics = evaluation.metrics
        metrics = {
            f"current_model_{name}": float(value)
            for name, value in raw_metrics.items()
        }
        r2_policy = dict(convergence_policy.get("acquisition_min_r2", {}))
        quality = (
            _evaluation_quality(evaluation, frames)
            if r2_policy
            else {"r2": {}, "outlier_fraction": {}, "element_force_r2": {}}
        )
        metrics.update(
            {
                f"current_model_{name}": float(value)
                for name, value in quality["r2"].items()
            }
        )
        grouped: dict[str, list[Atoms]] = {}
        for frame in frames:
            attempt_id = frame.info.get("scenario_attempt_id")
            if attempt_id is not None:
                grouped.setdefault(str(attempt_id), []).append(frame)
        attempt_metrics = {}
        attempt_accepted = {}
        for attempt_id, attempt_frames in sorted(grouped.items()):
            values = {
                name: float(value)
                for name, value in self.runtime.predict(
                    context.artifacts["model"],
                    attempt_frames,
                    str(options.get("inference_backend", "auto")),
                ).metrics.items()
            }
            attempt_metrics[attempt_id] = values
            attempt_accepted[attempt_id] = (
                _within_thresholds(values, thresholds)
                if quality_gate_configured
                else None
            )
        condition_force_r2: dict[str, float] = {}
        if r2_policy:
            condition_groups: dict[str, list[Atoms]] = {}
            for frame in frames:
                condition = (
                    f"R={frame.info.get('route_id', 'unknown')}|"
                    f"T={float(frame.info.get('temperature', 0.0)):g}|"
                    f"P={float(frame.info.get('pressure', 0.0)):g}"
                )
                condition_groups.setdefault(condition, []).append(frame)
            for condition, condition_frames in sorted(condition_groups.items()):
                condition_evaluation = self.runtime.predict(
                    context.artifacts["model"],
                    condition_frames,
                    str(options.get("inference_backend", "auto")),
                )
                force = condition_evaluation.comparisons.get("force")
                if force is not None:
                    condition_force_r2[condition] = _r2(
                        force.reference, force.predicted
                    )
        diagnostic_accepted = (
            _within_thresholds(raw_metrics, thresholds)
            if quality_gate_configured
            else None
        )
        signals = {
            **metrics,
            "prediction_metric_basis": _PREDICTION_METRIC_BASIS,
            "diagnostic_only": True,
            "quality_gate_configured": quality_gate_configured,
            "diagnostic_accepted": diagnostic_accepted,
            "attempt_accepted": attempt_accepted,
            "attempt_metrics": attempt_metrics,
            "element_force_r2": quality["element_force_r2"],
            "condition_force_r2": condition_force_r2,
            "outlier_fraction": quality["outlier_fraction"],
            "evaluated_count": len(frames),
            "spin_frame_count": spin_count,
        }
        output = atomic_write_json(
            context.work_dir / "acquisition-signals.json", signals
        )
        return StageOutcome(
            artifacts={"acquisition_signals": output}, metrics=signals
        )

    def _merge(self, context: StageContext) -> StageOutcome:
        original = _read_frames(context.artifacts["training_input"])
        labeled = _read_frames(context.artifacts["labeled"], allow_empty=True)
        try:
            original_ids = validate_labeled_frames(original)
            labeled_ids = (
                validate_labeled_frames(labeled) if labeled else []
            )
        except ScientificDataError as error:
            raise WorkflowIterationError(
                f"training merge input violates the scientific data contract: {error}"
            ) from error
        if len(set(original_ids)) != len(original_ids):
            raise WorkflowIterationError(
                "training input contains duplicate physical structures; "
                "deduplicate it explicitly before recovery"
            )
        if len(set(labeled_ids)) != len(labeled_ids):
            raise WorkflowIterationError(
                "new labels contain duplicate physical structures"
            )
        overlap = set(original_ids) & set(labeled_ids)
        if overlap:
            raise WorkflowIterationError(
                "new labels overlap the existing training set; refusing "
                "to silently keep one of two labels"
            )
        merged = [*original, *labeled]
        try:
            validate_labeled_frames(merged)
        except ScientificDataError as error:
            raise WorkflowIterationError(
                f"merged training set violates the scientific data contract: {error}"
            ) from error
        output = context.work_dir / "train.xyz"
        ase_write(output, merged, format="extxyz")
        return StageOutcome(
            artifacts={"training_set": output},
            metrics={
                "training_count": len(merged),
                "added_count": len(labeled),
                "duplicate_labeled_count": 0,
            },
        )

    def _acquisition_merge(self, context: StageContext) -> StageOutcome:
        """Commit labels and decide only what kind of generation comes next."""

        merged = self._merge(context)
        diagnostic = json.loads(
            context.artifacts["acquisition_signals"].read_text(encoding="utf-8")
        )
        evaluation = json.loads(
            context.artifacts["model_evaluation"].read_text(encoding="utf-8")
        )
        selection = json.loads(
            context.artifacts["selection_result"].read_text(encoding="utf-8")
        )
        novelty_threshold = float(
            selection.get(
                "resolved_completion_coverage_threshold",
                context.plan.completion_coverage_threshold,
            )
        )
        remaining_novelty = float(selection.get("remaining_novelty", 0.0))
        novelty_converged = bool(remaining_novelty <= novelty_threshold)
        previous_signals = (
            json.loads(
                context.previous_artifacts["signals"].read_text(encoding="utf-8")
            )
            if "signals" in context.previous_artifacts
            else {}
        )
        convergence = _acquisition_convergence_status(
            diagnostic,
            self.config.get("workflow", {}).get("convergence", {}),
            previous_signals,
        )
        convergence_evidence = (
            convergence["acquisition_converged"]
            if convergence["acquisition_convergence_configured"]
            else novelty_converged
        )
        validation_accepted = evaluation.get("validation_accepted")
        history, production_ready = self._record_route_histories(
            context,
            diagnostic=diagnostic,
            validation_metrics=evaluation.get("metrics") or None,
            evidence_validation=evaluation.get("metrics") or None,
            validation_accepted=validation_accepted,
            model_improved=bool(evaluation.get("model_improved", True)),
            novelty_converged=novelty_converged,
            final_model_id=file_sha256(context.artifacts["model"]),
        )
        sampling_complete = bool(production_ready and convergence_evidence)
        disposition = "finalize" if sampling_complete else "continue"
        history.update(convergence)
        history.update(
            {
                "workflow_converged": False,
                "finalization_pending": sampling_complete,
                "generation_disposition": disposition,
            }
        )
        signals = {
            **diagnostic,
            **convergence,
            "accepted": True,
            "generation_kind": "acquisition",
            "generation_disposition": disposition,
            "finalization_pending": sampling_complete,
            "workflow_converged": False,
            "workflow_stalled": False,
            "production_ready": production_ready,
            "model_validation_accepted": validation_accepted,
            "remaining_novelty": remaining_novelty,
            "novelty_threshold": novelty_threshold,
            "novelty_converged": novelty_converged,
            "scenario_counts_by_maturity": history["counts_by_maturity"],
            "active_model_sha256": file_sha256(context.artifacts["model"]),
            "merged_training_set_sha256": file_sha256(
                merged.artifacts["training_set"]
            ),
        }
        artifacts = dict(merged.artifacts)
        artifacts["scenario_maturity"] = atomic_write_json(
            context.work_dir / "scenario-maturity.json", history
        )
        artifacts["signals"] = atomic_write_json(
            context.work_dir / "signals.json", signals
        )
        return StageOutcome(
            artifacts=artifacts,
            metrics={
                **merged.metrics,
                **convergence,
                "accepted": True,
                "generation_kind": "acquisition",
                "generation_disposition": disposition,
                "finalization_pending": sampling_complete,
                "workflow_converged": False,
                "production_ready": production_ready,
            },
        )

    @staticmethod
    def _defer_legacy_convergence(
        context: StageContext,
        outcome: StageOutcome,
    ) -> StageOutcome:
        """Turn a migrated legacy stop into an explicit finalization request."""

        metrics = dict(outcome.metrics)
        sampling_complete = metrics.get("workflow_converged") is True
        metrics.update(
            {
                "workflow_converged": False,
                "generation_disposition": (
                    "finalize" if sampling_complete else "continue"
                ),
                "finalization_pending": sampling_complete,
            }
        )
        for name in ("signals", "scenario_maturity"):
            path = outcome.artifacts.get(name)
            if path is None:
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            record.update(
                {
                    "workflow_converged": False,
                    "generation_disposition": metrics[
                        "generation_disposition"
                    ],
                    "finalization_pending": sampling_complete,
                }
            )
            atomic_write_json(path, record)
        return StageOutcome(artifacts=outcome.artifacts, metrics=metrics)

    def _retrain(self, context: StageContext) -> StageOutcome:
        training_input = context.artifacts["training_set"]
        diagnostic = json.loads(
            context.artifacts["acquisition_signals"].read_text(encoding="utf-8")
        )
        attempts = json.loads(
            context.artifacts["md_attempts"].read_text(encoding="utf-8")
        )["attempts"]
        failed_md = any(not bool(item["completed"]) for item in attempts)
        previous_signals = (
            json.loads(
                context.previous_artifacts["signals"].read_text(
                    encoding="utf-8"
                )
            )
            if "signals" in context.previous_artifacts
            else {}
        )
        continue_training = bool(
            previous_signals.get("production_ready") is True
            and previous_signals.get("validation_accepted") is False
        )
        parent_model_sha256 = file_sha256(context.artifacts["model"])
        training_count = len(_read_frames(training_input))
        previous_training_count = len(
            _read_frames(context.artifacts["training_input"])
        )
        added_count = training_count - previous_training_count
        retrain_required = bool(
            added_count > 0
            and (
                failed_md
                or not diagnostic.get("diagnostic_accepted", False)
                or continue_training
            )
        )
        if not retrain_required:
            decision = {
                "retrained": False,
                "reason": (
                    "no novel structures required labeling; "
                    "advance coverage without changing the model"
                    if added_count == 0
                    else (
                        "current model passed new-label diagnostics; "
                        "continue certification without changing the model"
                    )
                ),
            }
            lineage = {
                "version": 1,
                "generation": context.generation,
                "parent_model_sha256": parent_model_sha256,
                "candidate_model_sha256": parent_model_sha256,
                "model_updated": False,
                "training_dataset_sha256": file_sha256(training_input),
                "training_count": training_count,
                "pending_label_count": added_count,
                "trained_on_current_labels": False,
                "label_provenance_sha256": file_sha256(
                    context.artifacts["label_provenance"]
                ),
            }
            artifacts = {
                "retrained_model": context.artifacts["model"],
                "retraining_decision": atomic_write_json(
                    context.work_dir / "retraining-decision.json", decision
                ),
                "model_lineage": atomic_write_json(
                    context.work_dir / "model-lineage.json", lineage
                ),
            }
            if "checkpoint" in context.artifacts:
                artifacts["retrained_checkpoint"] = context.artifacts[
                    "checkpoint"
                ]
            return StageOutcome(
                artifacts=artifacts,
                metrics={
                    "backend": "reuse",
                    "training_count": training_count,
                    "parent_model_sha256": parent_model_sha256,
                    "candidate_model_sha256": parent_model_sha256,
                    "model_updated": False,
                    **decision,
                },
            )

        result, frame_count, config_file = self._execute_training(
            context,
            training_input=training_input,
            role="retraining",
            warm_start=context.artifacts.get("checkpoint"),
        )
        artifacts: dict[str, Path] = {"retrained_model": result.best_model}
        if result.final_model is not None:
            artifacts["retrained_final_model"] = result.final_model
        if result.checkpoint is not None:
            artifacts["retrained_checkpoint"] = result.checkpoint
        artifacts.update(
            {
                f"retraining_output_{name.replace('.', '_')}": path
                for name, path in result.outputs.items()
            }
        )
        original_config = self._path(
            self.config.get("training", {}).get("config_path")
        )
        if config_file != original_config:
            artifacts["retraining_config"] = config_file
        decision = {
            "retrained": True,
            "reason": (
                "MD recovery required"
                if failed_md
                else (
                    "trajectory label diagnostics exceeded thresholds"
                    if not diagnostic.get("diagnostic_accepted", False)
                    else "new labels require a new model generation"
                )
            ),
        }
        artifacts["retraining_decision"] = atomic_write_json(
            context.work_dir / "retraining-decision.json", decision
        )
        lineage = {
            "version": 1,
            "generation": context.generation,
            "parent_model_sha256": parent_model_sha256,
            "candidate_model_sha256": file_sha256(result.best_model),
            "model_updated": (
                file_sha256(result.best_model) != parent_model_sha256
            ),
            "training_dataset_sha256": file_sha256(training_input),
            "training_count": frame_count,
            "pending_label_count": 0,
            "trained_on_current_labels": True,
            "label_provenance_sha256": file_sha256(
                context.artifacts["label_provenance"]
            ),
        }
        artifacts["model_lineage"] = atomic_write_json(
            context.work_dir / "model-lineage.json", lineage
        )
        return StageOutcome(
            artifacts=artifacts,
            metrics={
                "backend": result.backend,
                "training_count": frame_count,
                "parent_model_sha256": lineage["parent_model_sha256"],
                "candidate_model_sha256": lineage["candidate_model_sha256"],
                "model_updated": lineage["model_updated"],
                **decision,
            },
        )

    def _record_route_histories(
        self,
        context: StageContext,
        *,
        diagnostic: Mapping[str, Any],
        validation_metrics: Mapping[str, Any] | None,
        evidence_validation: Mapping[str, Any] | None,
        validation_accepted: bool | None,
        model_improved: bool,
        novelty_converged: bool,
        final_model_id: str,
    ) -> tuple[dict[str, Any], bool]:
        scenario_plan = json.loads(
            context.artifacts["scenario_plan"].read_text(encoding="utf-8")
        )
        if scenario_plan.get("version") != 3:
            raise WorkflowIterationError(
                "unsupported multi-route scenario plan"
            )
        previous_routes: dict[str, Any] = {}
        if "scenario_maturity" in context.previous_artifacts:
            previous = json.loads(
                context.previous_artifacts["scenario_maturity"].read_text(
                    encoding="utf-8"
                )
            )
            if previous.get("version") != 3:
                raise WorkflowIterationError(
                    "unsupported multi-route scenario maturity history"
                )
            previous_routes = previous.get("routes", {})

        route_by_id = {route.route_id: route for route in self.routes}
        selection = json.loads(
            context.artifacts["selection_result"].read_text(encoding="utf-8")
        )
        completion_threshold = float(
            selection.get(
                "resolved_completion_coverage_threshold",
                context.plan.completion_coverage_threshold,
            )
        )
        remaining_by_condition = {
            str(key): float(value)
            for key, value in selection.get(
                "remaining_novelty_by_condition", {}
            ).items()
        }
        histories: dict[str, Any] = {}
        all_ready = True
        total_counts: dict[str, int] = {}
        no_progress = []
        for route_plan in scenario_plan["routes"]:
            route_id = str(route_plan["route_id"])
            route = route_by_id.get(route_id)
            if route is None or route.fingerprint != route_plan.get(
                "route_fingerprint"
            ):
                raise WorkflowIterationError(
                    f"scenario plan route identity changed for {route_id!r}"
                )
            previous_record = previous_routes.get(route_id)
            previous_history = None
            if previous_record is not None:
                if previous_record.get("route_fingerprint") != route.fingerprint:
                    raise WorkflowIterationError(
                        f"scenario history route identity changed for {route_id!r}"
                    )
                previous_history = previous_record.get("history")
            attempt_accepted = {
                str(attempt["attempt_id"]): diagnostic.get(
                    "attempt_accepted", {}
                ).get(str(attempt["attempt_id"]))
                for attempt in route_plan["attempts"]
            }
            novelty_by_attempt = {}
            for attempt in route_plan["attempts"]:
                condition = (
                    f"R={route_id}|"
                    f"T={float(attempt['temperature']):g}|"
                    f"P={float(attempt['pressure']):g}"
                )
                novelty_by_attempt[str(attempt["attempt_id"])] = (
                    bool(
                        remaining_by_condition[condition]
                        <= completion_threshold
                    )
                    if condition in remaining_by_condition
                    else novelty_converged
                )
            ladder = self.scenario_ladders[route_id]
            route_history = ladder.record(
                route_plan["attempts"],
                completed=route_plan["completed"],
                diagnostic_accepted=attempt_accepted,
                history=previous_history,
                diagnostic=diagnostic,
                validation=validation_metrics,
                evidence_validation=evidence_validation,
                validation_accepted=validation_accepted,
                model_improved=model_improved,
                novelty_converged=novelty_converged,
                novelty_converged_by_attempt=novelty_by_attempt,
                final_model_id=final_model_id,
            )
            ready = ladder.production_ready(
                route_plan["structure_ids"],
                route_id=route_id,
                route_fingerprint=route.fingerprint,
                pressure=float(route_plan["pressure"]),
                model_id=final_model_id,
                history=route_history,
            )
            all_ready = all_ready and ready
            for level, count in route_history["counts_by_maturity"].items():
                total_counts[level] = total_counts.get(level, 0) + int(count)
            no_progress.append(int(route_history.get("no_progress_rounds", 0)))
            histories[route_id] = {
                "route_id": route_id,
                "route_fingerprint": route.fingerprint,
                "template_sha256": route.template_sha256,
                "production_ready": ready,
                "history": route_history,
            }
        return (
            {
                "version": 3,
                "routes": histories,
                "counts_by_maturity": total_counts,
                "no_progress_rounds": min(no_progress, default=0),
                "last_model_id": final_model_id,
                "validation_accepted": validation_accepted,
            },
            all_ready,
        )

    def _evaluate_without_validation(
        self, context: StageContext
    ) -> StageOutcome:
        options = self.config.get("evaluation", {})
        parent_model = context.artifacts["model"]
        candidate_model = context.artifacts["retrained_model"]
        parent_model_sha256 = file_sha256(parent_model)
        candidate_model_sha256 = file_sha256(candidate_model)
        retraining = json.loads(
            context.artifacts["retraining_decision"].read_text(
                encoding="utf-8"
            )
        )
        lineage = json.loads(
            context.artifacts["model_lineage"].read_text(encoding="utf-8")
        )
        candidate_trained = retraining.get("retrained") is True
        labeled_frames = _read_frames(
            context.artifacts["labeled"], allow_empty=True
        )
        label_metrics = (
            {
                name: float(value)
                for name, value in self.runtime.predict(
                    candidate_model,
                    labeled_frames,
                    str(options.get("inference_backend", "auto")),
                ).metrics.items()
            }
            if candidate_trained
            else {}
        )
        candidate_finite = all(
            np.isfinite(float(value)) for value in label_metrics.values()
        )
        lineage_valid = bool(
            lineage.get("parent_model_sha256") == parent_model_sha256
            and lineage.get("candidate_model_sha256")
            == candidate_model_sha256
        )
        if not lineage_valid:
            raise WorkflowIterationError(
                "evaluate received an inconsistent candidate model lineage"
            )
        candidate_activation_accepted = bool(
            not candidate_trained or candidate_finite
        )
        active_model = (
            candidate_model
            if candidate_activation_accepted
            else parent_model
        )
        active_model_sha256 = file_sha256(active_model)
        active_checkpoint = (
            context.artifacts.get("retrained_checkpoint")
            if candidate_activation_accepted and candidate_trained
            else context.artifacts.get("checkpoint")
        )
        diagnostic = json.loads(
            context.artifacts["acquisition_signals"].read_text(
                encoding="utf-8"
            )
        )
        selection = json.loads(
            context.artifacts["selection_result"].read_text(encoding="utf-8")
        )
        novelty_threshold = float(
            selection.get(
                "resolved_completion_coverage_threshold",
                context.plan.completion_coverage_threshold,
            )
        )
        remaining_novelty = float(selection.get("remaining_novelty", 0.0))
        novelty_converged = bool(remaining_novelty <= novelty_threshold)
        previous_signals = (
            json.loads(
                context.previous_artifacts["signals"].read_text(
                    encoding="utf-8"
                )
            )
            if "signals" in context.previous_artifacts
            else {}
        )
        convergence = _acquisition_convergence_status(
            diagnostic,
            self.config.get("workflow", {}).get("convergence", {}),
            previous_signals,
        )
        history, production_ready = self._record_route_histories(
            context,
            diagnostic=diagnostic,
            validation_metrics=None,
            evidence_validation=None,
            validation_accepted=None,
            model_improved=active_model_sha256 != parent_model_sha256,
            novelty_converged=novelty_converged,
            final_model_id=active_model_sha256,
        )
        workflow_converged = bool(
            production_ready and convergence["acquisition_converged"]
        )
        history.update(convergence)
        history["workflow_converged"] = workflow_converged
        history["workflow_stalled"] = False
        maturity_path = atomic_write_json(
            context.work_dir / "scenario-maturity.json", history
        )
        active_lineage = {
            "version": 1,
            "generation": context.generation,
            "parent_model_sha256": parent_model_sha256,
            "candidate_model_sha256": candidate_model_sha256,
            "active_model_sha256": active_model_sha256,
            "candidate_activation_accepted": candidate_activation_accepted,
            "activation_basis": "finite_new_label_metrics_without_independent_validation",
            "model_updated": active_model_sha256 != parent_model_sha256,
            "trained_on_current_labels": bool(
                candidate_activation_accepted
                and lineage.get("trained_on_current_labels")
            ),
            "training_dataset_sha256": lineage.get(
                "training_dataset_sha256"
            ),
            "training_count": lineage.get("training_count"),
        }
        signals = {
            **convergence,
            "prediction_metric_basis": diagnostic.get(
                "prediction_metric_basis"
            ),
            "accepted": True,
            "evaluation_configured": False,
            "validation_accepted": None,
            "candidate_activation_accepted": (
                candidate_activation_accepted
            ),
            "candidate_activation_reason": (
                "current model reused; no candidate model update"
                if not candidate_trained
                else (
                    "candidate has finite metrics on newly labeled structures"
                    if candidate_activation_accepted
                    else "candidate produced non-finite new-label metrics"
                )
            ),
            "candidate_label_metrics": label_metrics,
            "candidate_validation_metrics": None,
            "active_model_sha256": active_model_sha256,
            "parent_model_sha256": parent_model_sha256,
            "candidate_model_sha256": candidate_model_sha256,
            "model_updated": active_model_sha256 != parent_model_sha256,
            "remaining_novelty": remaining_novelty,
            "novelty_threshold": novelty_threshold,
            "novelty_converged": novelty_converged,
            "scenario_counts_by_maturity": history[
                "counts_by_maturity"
            ],
            "sampling_routes": [
                {
                    "route_id": route.route_id,
                    "route_fingerprint": route.fingerprint,
                }
                for route in self.routes
            ],
            "production_ready": production_ready,
            "workflow_converged": workflow_converged,
            "workflow_stalled": False,
            "no_progress_rounds": int(
                history.get("no_progress_rounds", 0)
            ),
        }
        artifacts = {
            "activated_model": active_model,
            "active_model_lineage": atomic_write_json(
                context.work_dir / "active-model-lineage.json",
                active_lineage,
            ),
            "scenario_maturity": maturity_path,
            "signals": atomic_write_json(
                context.work_dir / "signals.json", signals
            ),
        }
        report = build_evaluation_report(
            context.work_dir,
            metrics=label_metrics,
            thresholds={},
        )
        artifacts["evaluation_report"] = report.report
        if active_checkpoint is not None:
            artifacts["activated_checkpoint"] = active_checkpoint
        return StageOutcome(artifacts=artifacts, metrics=signals)

    def _evaluate(self, context: StageContext) -> StageOutcome:
        if not self.evaluation_configured:
            return self._evaluate_without_validation(context)
        options = self.config.get("evaluation", {})
        assert self.validation is not None
        frames = _read_frames(self.validation)
        _, spin_count = validate_spin_dataset(frames, require_mforce=True)
        training_ids = {
            structure_id(frame)
            for frame in _read_frames(context.artifacts["training_set"])
        }
        overlap = sum(structure_id(frame) in training_ids for frame in frames)
        if overlap:
            raise WorkflowIterationError(
                f"validation dataset overlaps the merged training set by {overlap} frames"
            )
        inference_backend = str(options.get("inference_backend", "auto"))
        thresholds = dict(options.get("max_rmse", {}))
        parent_model = context.artifacts["model"]
        candidate_model = context.artifacts["retrained_model"]
        parent_model_sha256 = file_sha256(parent_model)
        candidate_model_sha256 = file_sha256(candidate_model)
        parent_evaluation = self.runtime.predict(
            parent_model,
            frames,
            inference_backend,
        )
        parent_metrics = dict(parent_evaluation.metrics)
        parent_finite = all(
            np.isfinite(float(value)) for value in parent_metrics.values()
        )
        parent_validation_accepted = _within_thresholds(
            parent_metrics, thresholds
        )
        parent_validation_score = _threshold_score(
            parent_metrics, thresholds
        )
        previous_signals = (
            json.loads(
                context.previous_artifacts["signals"].read_text(
                    encoding="utf-8"
                )
            )
            if "signals" in context.previous_artifacts
            else {}
        )
        previous_validation_score = previous_signals.get("validation_score")
        training_before = _read_frames(context.artifacts["training_input"])
        training_after = _read_frames(context.artifacts["training_set"])
        added_count = len(training_after) - len(training_before)
        retraining = json.loads(
            context.artifacts["retraining_decision"].read_text(encoding="utf-8")
        )
        lineage = json.loads(
            context.artifacts["model_lineage"].read_text(encoding="utf-8")
        )
        lineage_valid = bool(
            lineage.get("parent_model_sha256") == parent_model_sha256
            and lineage.get("candidate_model_sha256")
            == candidate_model_sha256
            and (
                (
                    retraining.get("retrained") is True
                    and lineage.get("trained_on_current_labels") is True
                )
                or (
                    retraining.get("retrained") is False
                    and lineage.get("model_updated") is False
                    and lineage.get("parent_model_sha256")
                    == candidate_model_sha256
                )
            )
        )
        if not lineage_valid:
            raise WorkflowIterationError(
                "evaluate received an inconsistent candidate model lineage"
            )

        candidate_updated = bool(
            lineage.get("model_updated")
            and candidate_model_sha256 != parent_model_sha256
        )
        candidate_trained = retraining.get("retrained") is True
        candidate_evaluation = (
            self.runtime.predict(
                candidate_model,
                frames,
                inference_backend,
            )
            if candidate_trained
            else parent_evaluation
        )
        candidate_metrics = dict(candidate_evaluation.metrics)
        candidate_finite = all(
            np.isfinite(float(value)) for value in candidate_metrics.values()
        )
        candidate_validation_accepted = _within_thresholds(
            candidate_metrics, thresholds
        )
        candidate_validation_score = _threshold_score(
            candidate_metrics, thresholds
        )
        labeled_frames = _read_frames(
            context.artifacts["labeled"], allow_empty=True
        )
        candidate_label_metrics: dict[str, float] = {}
        candidate_label_attempt_metrics: dict[
            str, dict[str, float]
        ] = {}
        candidate_label_attempt_accepted: dict[str, bool] = {}
        candidate_label_accepted = True
        if candidate_trained:
            candidate_label_metrics = {
                name: float(value)
                for name, value in self.runtime.predict(
                    candidate_model,
                    labeled_frames,
                    inference_backend,
                ).metrics.items()
            }
            candidate_label_accepted = _within_thresholds(
                candidate_label_metrics, thresholds
            )
            grouped_labels: dict[str, list[Atoms]] = {}
            for frame in labeled_frames:
                attempt_id = frame.info.get("scenario_attempt_id")
                if attempt_id is not None:
                    grouped_labels.setdefault(str(attempt_id), []).append(
                        frame
                    )
            for attempt_id, attempt_frames in sorted(
                grouped_labels.items()
            ):
                attempt_metrics = {
                    name: float(value)
                    for name, value in self.runtime.predict(
                        candidate_model,
                        attempt_frames,
                        inference_backend,
                    ).metrics.items()
                }
                candidate_label_attempt_metrics[
                    attempt_id
                ] = attempt_metrics
                candidate_label_attempt_accepted[
                    attempt_id
                ] = _within_thresholds(attempt_metrics, thresholds)
            candidate_label_accepted = bool(
                candidate_label_accepted
                and all(candidate_label_attempt_accepted.values())
            )

        if not candidate_trained:
            candidate_activation_accepted = True
            activation_reason = "current model reused; no candidate model update"
            candidate_validation_limit = parent_validation_score
        else:
            candidate_validation_limit = (
                1.0
                if parent_validation_accepted
                or not parent_finite
                else (
                    parent_validation_score
                    * _CANDIDATE_VALIDATION_REGRESSION_FACTOR
                )
            )
            validation_gate = bool(
                candidate_finite
                and candidate_validation_score
                <= candidate_validation_limit
            )
            candidate_activation_accepted = bool(
                candidate_finite
                and candidate_label_accepted
                and validation_gate
            )
            if not candidate_finite:
                activation_reason = (
                    "candidate validation produced non-finite metrics"
                )
            elif not candidate_label_accepted:
                activation_reason = (
                    "candidate failed the newly labeled structure canary"
                )
            elif not validation_gate:
                activation_reason = (
                    "candidate regressed on the independent validation set"
                )
            else:
                activation_reason = (
                    "candidate passed new-label and independent validation gates"
                )

        if candidate_activation_accepted:
            active_model = candidate_model
            metrics = candidate_metrics
            active_model_sha256 = candidate_model_sha256
            active_checkpoint = (
                context.artifacts.get("retrained_checkpoint")
                if candidate_trained
                else context.artifacts.get("checkpoint")
            )
        else:
            active_model = parent_model
            metrics = parent_metrics
            active_model_sha256 = parent_model_sha256
            active_checkpoint = context.artifacts.get("checkpoint")
        finite = all(np.isfinite(float(value)) for value in metrics.values())
        validation_accepted = _within_thresholds(metrics, thresholds)
        validation_score = _threshold_score(metrics, thresholds)
        accepted = bool(frames and finite)

        diagnostic = json.loads(
            context.artifacts["acquisition_signals"].read_text(encoding="utf-8")
        )
        selection = json.loads(
            context.artifacts["selection_result"].read_text(encoding="utf-8")
        )
        novelty_threshold = float(
            selection.get(
                "resolved_completion_coverage_threshold",
                context.plan.completion_coverage_threshold,
            )
        )
        remaining_novelty = float(selection.get("remaining_novelty", 0.0))
        novelty_converged = bool(remaining_novelty <= novelty_threshold)
        comparison_validation_score = (
            float(previous_validation_score)
            if previous_validation_score is not None
            else parent_validation_score
        )
        validation_improved = bool(
            active_model_sha256 != parent_model_sha256
            and validation_score < comparison_validation_score * 0.99
        )
        signals = {
            **metrics,
            "prediction_metric_basis": _PREDICTION_METRIC_BASIS,
            "accepted": accepted,
            "evaluation_configured": True,
            "validation_accepted": validation_accepted,
            "validation_score": validation_score,
            "previous_validation_score": previous_validation_score,
            "validation_improved": validation_improved,
            "evaluated_count": len(frames),
            "spin_frame_count": spin_count,
            "added_training_count": added_count,
            "model_trained_on_current_labels": bool(
                candidate_activation_accepted
                and lineage.get("trained_on_current_labels")
            ),
            "active_model_sha256": active_model_sha256,
            "parent_model_sha256": parent_model_sha256,
            "candidate_model_sha256": candidate_model_sha256,
            "candidate_model_updated": candidate_updated,
            "candidate_activation_accepted": candidate_activation_accepted,
            "candidate_activation_reason": activation_reason,
            "candidate_validation_metrics": candidate_metrics,
            "candidate_validation_score": candidate_validation_score,
            "candidate_validation_limit": candidate_validation_limit,
            "candidate_validation_regression_factor": (
                _CANDIDATE_VALIDATION_REGRESSION_FACTOR
            ),
            "candidate_validation_accepted": candidate_validation_accepted,
            "candidate_label_metrics": candidate_label_metrics,
            "candidate_label_attempt_metrics": (
                candidate_label_attempt_metrics
            ),
            "candidate_label_attempt_accepted": (
                candidate_label_attempt_accepted
            ),
            "candidate_label_accepted": candidate_label_accepted,
            "parent_validation_metrics": parent_metrics,
            "parent_validation_score": parent_validation_score,
            "parent_validation_accepted": parent_validation_accepted,
            "parent_validation_finite": parent_finite,
            "model_updated": active_model_sha256 != parent_model_sha256,
            "remaining_novelty": remaining_novelty,
            "novelty_threshold": novelty_threshold,
            "novelty_converged": novelty_converged,
            "validation_path": str(self.validation),
        }
        active_lineage = {
            "version": 1,
            "generation": context.generation,
            "parent_model_sha256": parent_model_sha256,
            "candidate_model_sha256": candidate_model_sha256,
            "active_model_sha256": active_model_sha256,
            "candidate_activation_accepted": candidate_activation_accepted,
            "activation_reason": activation_reason,
            "model_updated": active_model_sha256 != parent_model_sha256,
            "trained_on_current_labels": bool(
                candidate_activation_accepted
                and lineage.get("trained_on_current_labels")
            ),
            "training_dataset_sha256": lineage.get(
                "training_dataset_sha256"
            ),
            "training_count": lineage.get("training_count"),
            "pending_label_count": (
                0
                if candidate_activation_accepted
                and lineage.get("trained_on_current_labels")
                else added_count
            ),
        }
        artifacts = {
            "activated_model": active_model,
            "active_model_lineage": atomic_write_json(
                context.work_dir / "active-model-lineage.json",
                active_lineage,
            ),
        }
        if active_checkpoint is not None:
            artifacts["activated_checkpoint"] = active_checkpoint
        attempt = 1
        while (context.work_dir / f"signals-attempt-{attempt}.json").exists():
            attempt += 1
        recovering = (context.work_dir / "signals.json").exists()
        suffix = f"-attempt-{attempt}" if recovering else ""
        final_model_id = active_model_sha256
        history, production_ready = self._record_route_histories(
            context,
            diagnostic=diagnostic,
            validation_metrics=metrics,
            evidence_validation=parent_metrics,
            validation_accepted=validation_accepted,
            model_improved=validation_improved,
            novelty_converged=novelty_converged,
            final_model_id=final_model_id,
        )
        convergence = _acquisition_convergence_status(
            diagnostic,
            self.config.get("workflow", {}).get("convergence", {}),
            previous_signals,
        )
        convergence_evidence = (
            convergence["acquisition_converged"]
            if convergence["acquisition_convergence_configured"]
            else novelty_converged
        )
        workflow_converged = bool(
            validation_accepted and production_ready and convergence_evidence
        )
        workflow_stalled = bool(
            not workflow_converged
            and int(history.get("no_progress_rounds", 0)) >= 2
        )
        history.update(convergence)
        history["workflow_converged"] = workflow_converged
        history["workflow_stalled"] = workflow_stalled
        maturity_path = atomic_write_json(
            context.work_dir / f"scenario-maturity{suffix}.json", history
        )
        artifacts["scenario_maturity"] = maturity_path
        signals.update(
            **convergence,
            scenario_counts_by_maturity=history["counts_by_maturity"],
            sampling_routes=[
                {
                    "route_id": route.route_id,
                    "route_fingerprint": route.fingerprint,
                }
                for route in self.routes
            ],
            production_ready=production_ready,
            workflow_converged=workflow_converged,
            workflow_stalled=workflow_stalled,
            no_progress_rounds=int(history.get("no_progress_rounds", 0)),
        )
        output = atomic_write_json(context.work_dir / f"signals{suffix}.json", signals)
        artifacts["signals"] = output
        report = build_evaluation_report(
            context.work_dir,
            metrics=candidate_metrics,
            thresholds=thresholds,
            parent_metrics=parent_metrics,
            suffix=suffix,
        )
        artifacts["evaluation_report"] = report.report
        if report.chart is not None:
            artifacts["evaluation_chart"] = report.chart
        if candidate_evaluation.comparisons:
            parity = build_parity_report(
                context.work_dir,
                series=candidate_evaluation.comparisons,
                source={
                    "validation_name": self.validation.name,
                    "validation_sha256": file_sha256(self.validation),
                    "candidate_model_sha256": candidate_model_sha256,
                    "evaluated_count": len(frames),
                },
                suffix=suffix,
            )
            artifacts["evaluation_parity_report"] = parity.report
            if parity.chart is not None:
                artifacts["evaluation_parity"] = parity.chart
        return StageOutcome(artifacts=artifacts, metrics=signals)


__all__ = [
    "PredictionEvaluation",
    "WorkflowIterationAdapter",
    "WorkflowIterationError",
    "WorkflowRuntime",
]
