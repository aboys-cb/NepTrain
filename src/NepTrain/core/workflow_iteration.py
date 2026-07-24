"""Production workflow Adapter for the deterministic generation controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from .dft import LabelRequest, LabelResult, label
from .iteration import (
    StageContext,
    StageOutcome,
    stratified_farthest_point_sampling,
)
from .md import MdError, MdRequest, MdResult, run_md
from .nep.calculator import DescriptorCalculator, Nep3Calculator
from .scenario import ScenarioLadder
from .spin import validate_spin_dataset
from .toy_workflow import structure_id
from .training import TrainingRequest, TrainingResult, train


class WorkflowIterationError(RuntimeError):
    """Raised when a real workflow stage cannot satisfy the iteration contract."""


TrainRunner = Callable[[TrainingRequest, str], TrainingResult]
MdRunner = Callable[[MdRequest, str], MdResult]
LabelRunner = Callable[[LabelRequest, str], LabelResult]
DescriptorRunner = Callable[[Path, Sequence[Atoms]], np.ndarray]
PredictionRunner = Callable[[Path, Sequence[Atoms], str], Mapping[str, float]]


def _nep_descriptors(model: Path, frames: Sequence[Atoms]) -> np.ndarray:
    calculator = DescriptorCalculator("nep", model_file=model)
    try:
        return np.asarray(calculator.get_structures_descriptors(frames), dtype=np.float64)
    finally:
        calculator.calculator.close()


def _rmse(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if reference.shape != prediction.shape:
        raise WorkflowIterationError(
            f"prediction shape {prediction.shape} does not match labels {reference.shape}"
        )
    return float(np.sqrt(np.mean(np.square(prediction - reference))))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _nep_prediction_metrics(
    model: Path, frames: Sequence[Atoms], backend: str
) -> Mapping[str, float]:
    if not frames:
        raise WorkflowIterationError("evaluation requires at least one labeled frame")
    spin = "spin" in frames[0].arrays
    with Nep3Calculator(model, backend=backend) as calculator:
        if spin:
            energy, forces, virials, mforces = calculator.calculate_spin(frames)
        else:
            energy, forces, virials = calculator.calculate(frames)
            mforces = None
    result = {
        "energy_rmse": _rmse(
            np.asarray([frame.get_potential_energy() for frame in frames]), energy
        ),
        "force_rmse": _rmse(
            np.concatenate([frame.get_forces() for frame in frames]),
            np.concatenate(forces),
        ),
        "virial_rmse": _rmse(
            np.asarray([frame.info["virial"] for frame in frames]), virials
        ),
    }
    if spin:
        result["mforce_rmse"] = _rmse(
            np.concatenate([frame.arrays["mforce"] for frame in frames]),
            np.concatenate(mforces),
        )
    return result


@dataclass(frozen=True)
class WorkflowRuntime:
    """Narrow dependency seam used by tests; defaults are the real backends."""

    train: TrainRunner = train
    md: MdRunner = run_md
    label: LabelRunner = label
    descriptors: DescriptorRunner = _nep_descriptors
    predict: PredictionRunner = _nep_prediction_metrics


def _read_frames(path: Path) -> list[Atoms]:
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
        loaded = ase_read(item, index=":", format=None)
        frames.extend(loaded if isinstance(loaded, list) else [loaded])
    if not frames:
        raise WorkflowIterationError(f"no structures found in {path}")
    return frames


def _write_json(path: Path, value: Any) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


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
        initial_training: str | Path,
        base_dir: str | Path = ".",
        runtime: WorkflowRuntime | None = None,
    ):
        self.config = dict(config)
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.initial_training = self._path(initial_training)
        self.runtime = runtime or WorkflowRuntime()
        self.scenario_ladder = ScenarioLadder.from_sampling(
            self.config.get("sampling", {})
        )
        if not self.initial_training.is_file():
            raise WorkflowIterationError(
                f"initial training set does not exist: {self.initial_training}"
            )
        evaluation = self.config.get("evaluation", {})
        validation_value = evaluation.get("validation_path") or self.config.get(
            "training", {}
        ).get("test_path")
        if not validation_value:
            raise WorkflowIterationError(
                "post-retrain acceptance requires evaluation.validation_path "
                "or training.test_path"
            )
        self.validation = self._path(validation_value)
        if not self.validation.is_file():
            raise WorkflowIterationError(
                f"validation dataset does not exist: {self.validation}"
            )
        thresholds = evaluation.get("max_rmse")
        required_thresholds = {"energy_rmse", "force_rmse"}
        if self.config.get("md", {}).get("spin", False):
            required_thresholds.add("mforce_rmse")
        missing_thresholds = sorted(
            required_thresholds - set(thresholds or {})
        )
        if missing_thresholds:
            raise WorkflowIterationError(
                "post-retrain acceptance requires evaluation.max_rmse for "
                + ", ".join(missing_thresholds)
            )
        if self.config.get("md", {}).get("spin", False):
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
    def _dft_kpoints(options: Mapping[str, Any]) -> tuple[int, int, int]:
        if options.get("use_k_stype", "kspacing") != "kpoints":
            return (1, 1, 1)
        raw = options.get("kpoints", (1, 1, 1))
        values = [int(raw)] if isinstance(raw, int | float | str) else [int(value) for value in raw]
        if len(values) == 1:
            values *= 3
        if len(values) != 3 or any(value < 1 for value in values):
            raise WorkflowIterationError("dft.kpoints must contain one or three positive integers")
        return tuple(values)

    def run_stage(self, stage: str, context: StageContext) -> StageOutcome:
        method = getattr(self, f"_{stage}", None)
        if method is None:
            raise WorkflowIterationError(f"unsupported workflow stage: {stage}")
        return method(context)

    def _execute_training(
        self,
        context: StageContext,
        *,
        training_input: Path,
        output_name: str,
        warm_start: Path | None,
    ) -> tuple[TrainingResult, int, Path]:
        options = self.config.get("training", {})
        backend = str(options.get("backend", "gpumd"))
        config_file = self._path(options.get("config_path"))
        if backend == "torchnep" and warm_start is not None:
            config_file = self._torchnep_finetune_config(
                config_file, context.work_dir, output_name, options
            )
        request = TrainingRequest(
            config_file=config_file,
            train_file=training_input,
            output_dir=context.work_dir / output_name,
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
                options.get(
                    "seed",
                    self.config.get("workflow", {}).get("seed", 20260723),
                )
            ),
        )
        result = self.runtime.train(request, backend)
        return result, len(_read_frames(training_input)), config_file

    @staticmethod
    def _torchnep_finetune_config(
        source: Path,
        generation_dir: Path,
        output_name: str,
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
        suffix = output_name.removeprefix("retraining")
        path = generation_dir / f"torchnep-finetune{suffix}.in"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(output), encoding="utf-8")
        return path

    @staticmethod
    def _attempt_output_name(generation_dir: Path, base: str) -> str:
        if not (generation_dir / base).exists():
            return base
        attempt = 2
        while (generation_dir / f"{base}-attempt-{attempt}").exists():
            attempt += 1
        return f"{base}-attempt-{attempt}"

    def _train(self, context: StageContext) -> StageOutcome:
        if context.generation > 1:
            artifacts = {
                "training_input": context.previous_artifacts["training_set"],
                "model": context.previous_artifacts["retrained_model"],
            }
            if "retrained_checkpoint" in context.previous_artifacts:
                artifacts["checkpoint"] = context.previous_artifacts[
                    "retrained_checkpoint"
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

        training_input = self.initial_training
        result, frame_count, _ = self._execute_training(
            context,
            training_input=training_input,
            output_name="training",
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
        return StageOutcome(
            artifacts=artifacts,
            metrics={
                "backend": result.backend,
                "training_count": frame_count,
                "reused_previous_model": False,
            },
        )

    @staticmethod
    def _balanced_cap(
        groups: Sequence[tuple[str, list[Atoms]]], limit: int
    ) -> list[tuple[str, int, Atoms]]:
        """Cap every source fairly while retaining its full time span.

        ``groups`` may contain trajectories much longer than the candidate
        budget. Taking their first frames would make a longer MD run pay its
        full cost while FPS only sees the beginning. Allocate the budget in a
        round-robin manner, keep pre-failure frames first, and spread the
        remaining quota over each stable trajectory from start to finish.
        """

        if limit < 1:
            return []
        nonempty = [(source, frames) for source, frames in groups if frames]
        quotas = [0] * len(nonempty)
        remaining = min(limit, sum(len(frames) for _, frames in nonempty))
        while remaining:
            changed = False
            for index, (_, frames) in enumerate(nonempty):
                if quotas[index] >= len(frames):
                    continue
                quotas[index] += 1
                remaining -= 1
                changed = True
                if not remaining:
                    break
            if not changed:
                break

        spread_groups: list[tuple[str, list[tuple[int, Atoms]]]] = []
        for (source, frames), quota in zip(nonempty, quotas):
            pre_failure = [
                (index, frame)
                for index, frame in enumerate(frames)
                if frame.info.get("md_window") == "pre_failure"
            ]
            stable = [
                (index, frame)
                for index, frame in enumerate(frames)
                if frame.info.get("md_window") != "pre_failure"
            ]
            if len(pre_failure) >= quota:
                chosen = pre_failure[-quota:]
            else:
                chosen = list(pre_failure)
                stable_quota = quota - len(chosen)
                if stable_quota >= len(stable):
                    chosen.extend(stable)
                elif stable_quota == 1:
                    chosen.append(stable[-1])
                elif stable_quota > 1:
                    indices = np.linspace(
                        0, len(stable) - 1, stable_quota
                    ).round().astype(int)
                    chosen.extend(stable[int(index)] for index in indices)
            spread_groups.append((source, chosen))

        selected: list[tuple[str, int, Atoms]] = []
        for position in range(
            max((len(frames) for _, frames in spread_groups), default=0)
        ):
            for source, frames in spread_groups:
                if position < len(frames):
                    frame_index, frame = frames[position]
                    selected.append((source, frame_index, frame))
        return selected

    def _explore(self, context: StageContext) -> StageOutcome:
        options = self.config.get("md", {})
        sampling = self.config.get("sampling", {})
        conditions = sampling.get("conditions", {})
        progression = sampling.get("progression", {})
        candidate_pool = sampling.get("candidate_pool", {})
        structures = _read_frames(self._path(options.get("structures")))
        backend = str(options.get("backend", "lammps"))
        structure_by_id = {structure_id(atoms): atoms for atoms in structures}
        ordered_structure_ids = sorted(structure_by_id)
        target_sources = int(progression.get("md_runs_per_iteration", 1))
        scenario_history = None
        if "scenario_maturity" in context.previous_artifacts:
            scenario_history = json.loads(
                context.previous_artifacts["scenario_maturity"].read_text(
                    encoding="utf-8"
                )
            )
        scenario_attempts = self.scenario_ladder.schedule(
            ordered_structure_ids,
            pressure=context.plan.pressure,
            generation=context.generation,
            seed=context.plan.seed,
            limit=min(target_sources, context.plan.dft_budget),
            model_id=_file_sha256(context.artifacts["model"]),
            history=scenario_history,
        )
        run_specs = [
            (
                attempt.structure_id,
                attempt.temperature,
                attempt.steps,
                attempt.scenario_id,
                attempt.attempt_id,
                attempt.target_level,
                attempt.replica,
                attempt.seed,
            )
            for attempt in scenario_attempts
        ]
        groups: list[tuple[str, list[Atoms]]] = []
        backend_details: set[str] = set()
        source_metadata: dict[str, dict[str, Any]] = {}
        attempt_results: list[dict[str, Any]] = []
        for (
            structure_identifier,
            temperature,
            steps,
            scenario_identifier,
            scenario_attempt_identifier,
            target_level,
            replica,
            md_seed,
        ) in run_specs:
            atoms = structure_by_id[structure_identifier]
            source = (
                f"g{context.generation}-s{structure_identifier[:8]}-"
                f"T{temperature:g}-P{context.plan.pressure:g}-"
                f"r{replica}-{scenario_attempt_identifier[:8]}"
            )
            run_dir = context.work_dir / "md" / source
            trajectory = run_dir / "trajectory.xyz"
            spin_temperature = conditions.get("spin_temperature")
            if spin_temperature in {None, "auto"}:
                spin_temperature = temperature
            request = MdRequest(
                atoms=atoms.copy(),
                model_file=context.artifacts["model"],
                output_dir=run_dir,
                output_file=trajectory,
                temperature=float(temperature),
                steps=int(steps),
                seed=md_seed,
                pressure=context.plan.pressure,
                spin=bool(options.get("spin", False)),
                spin_temperature=float(spin_temperature)
                if spin_temperature is not None
                else None,
                template_path=self._path(options["template_path"])
                if options.get("template_path")
                else None,
                inference_backend=str(options.get("inference_backend", "auto")),
                lmp_command=str(options.get("lmp", "lmp")),
                mpiexec=str(options.get("mpiexec", "mpirun")),
                mpi_ranks=int(options.get("mpi_ranks", 1)),
                pre_failure_frames=int(
                    candidate_pool.get("pre_failure_frames", 2)
                ),
                bad_tail_frames=int(candidate_pool.get("bad_tail_frames", 1)),
                health=dict(candidate_pool.get("health") or {}),
            )
            try:
                result = self.runtime.md(request, backend)
            except MdError as error:
                if backend != "lammps":
                    raise
                attempt_results.append(
                    {
                        "source_id": source,
                        "scenario_id": scenario_identifier,
                        "scenario_attempt_id": scenario_attempt_identifier,
                        "seed": md_seed,
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
                    str(frame.info.get("md_window", "stable_prefix")) == window
                    for frame in run_frames
                )
                for window in ("stable_prefix", "pre_failure", "bad_tail")
            }
            window_counts = {
                window: count for window, count in window_counts.items() if count
            }
            usable_frames = [
                frame
                for frame in run_frames
                if frame.info.get("md_window", "stable_prefix") != "bad_tail"
            ]
            usable_frames.sort(
                key=lambda frame: (
                    0
                    if frame.info.get("md_window") == "pre_failure"
                    else 1,
                    int(frame.info.get("lammps_step", 0)),
                )
            )
            stride = int(candidate_pool.get("frame_stride", 1))
            stable_index = 0
            thinned_frames = []
            for frame in usable_frames:
                if frame.info.get("md_window") == "pre_failure":
                    thinned_frames.append(frame)
                    continue
                if stable_index % stride == 0:
                    thinned_frames.append(frame)
                stable_index += 1
            usable_frames = thinned_frames
            if usable_frames:
                groups.append((source, usable_frames))
            source_metadata[source] = {
                "structure_id": structure_identifier,
                "scenario_id": scenario_identifier,
                "scenario_attempt_id": scenario_attempt_identifier,
                "maturity_target": target_level,
                "replica": replica,
                "seed": md_seed,
                "md_steps": int(steps),
                "temperature": float(temperature),
                "completed": bool(result.completed),
                "failure_code": result.failure_code,
            }
            backend_details.add(result.inference_backend or result.backend)
            health_summary = (
                json.loads(result.health_report.read_text(encoding="utf-8"))
                if result.health_report is not None
                else None
            )
            attempt_results.append(
                {
                    "source_id": source,
                    "scenario_id": scenario_identifier,
                    "scenario_attempt_id": scenario_attempt_identifier,
                    "seed": md_seed,
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

        candidates = [
            (source, frame_index, frame)
            for source, frames in groups
            for frame_index, frame in enumerate(frames)
        ]
        if not candidates:
            failed = sum(not item["completed"] for item in attempt_results)
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
                pressure=context.plan.pressure,
                frame_step=int(frame.info.get("lammps_step", frame_index)),
                scenario_structure_id=metadata["structure_id"],
                md_steps=metadata["md_steps"],
                md_seed=metadata["seed"],
                md_completed=metadata["completed"],
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
        ase_write(output, output_frames, format="extxyz")
        artifacts = {
            "candidates": output,
            "md_attempts": _write_json(
                context.work_dir / "md-attempts.json",
                {"version": 1, "attempts": attempt_results},
            ),
        }
        scenario_plan = _write_json(
            context.work_dir / "scenario-plan.json",
            {
                "version": 2,
                "model_id": _file_sha256(context.artifacts["model"]),
                "structure_ids": ordered_structure_ids,
                "attempts": self.scenario_ladder.serialize(scenario_attempts),
                "completed": {
                    str(item["scenario_attempt_id"]): bool(item["completed"])
                    for item in attempt_results
                    if item.get("scenario_attempt_id") is not None
                },
            },
        )
        artifacts["scenario_plan"] = scenario_plan
        return StageOutcome(
            artifacts=artifacts,
            metrics={
                "backend": backend,
                "inference_backends": sorted(backend_details),
                "candidate_count": len(output_frames),
                "source_count": len(groups),
                "scheduled_source_count": len(attempt_results),
                "completed_source_count": sum(
                    bool(item["completed"]) for item in attempt_results
                ),
                "failed_source_count": sum(
                    not bool(item["completed"]) for item in attempt_results
                ),
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
                "temperatures": list(context.plan.temperatures),
                "pressure": context.plan.pressure,
                "steps": context.plan.steps,
                "scenario_steps": sorted({int(spec[2]) for spec in run_specs}),
                "scenario_targets": sorted(
                    {str(spec[5]) for spec in run_specs if spec[5] is not None}
                ),
            },
        )

    def _select(self, context: StageContext) -> StageOutcome:
        all_candidates = _read_frames(context.artifacts["candidates"])
        sampling = self.config.get("sampling", {})
        selection = sampling.get("selection", {})
        if "md_attempts" in context.artifacts:
            attempts = json.loads(
                context.artifacts["md_attempts"].read_text(encoding="utf-8")
            )["attempts"]
            scheduled_count = len(attempts)
            failed_count = sum(
                not bool(item["completed"]) for item in attempts
            )
        else:
            scheduled_count = len(
                {
                    str(frame.info.get("scenario_attempt_id"))
                    if frame.info.get("scenario_attempt_id") is not None
                    else str(frame.info.get("source_id"))
                    for frame in all_candidates
                }
            )
            failed_count = int(
                any(not bool(frame.info.get("md_completed", True)) for frame in all_candidates)
            )
        minimum_budget = int(selection.get("minimum_dft_budget", 1))
        effective_budget = (
            context.plan.dft_budget
            if failed_count
            else min(
                context.plan.dft_budget,
                max(scheduled_count, minimum_budget),
            )
        )
        candidates = list(all_candidates)
        training = _read_frames(context.artifacts["training_input"])
        known_ids = {structure_id(frame) for frame in training}
        candidates = [
            frame for frame in candidates if structure_id(frame) not in known_ids
        ]
        if not candidates:
            raise WorkflowIterationError(
                "all MD candidates already exist in the training set; "
                "increase MD steps or temperature"
            )
        unique_candidates: dict[str, Atoms] = {}
        for frame in candidates:
            identifier = structure_id(frame)
            current = unique_candidates.get(identifier)
            if current is None or (
                frame.info.get("md_window") == "pre_failure"
                and current.info.get("md_window") != "pre_failure"
            ):
                unique_candidates[identifier] = frame
        duplicate_candidate_count = len(candidates) - len(unique_candidates)
        candidates = list(unique_candidates.values())
        grouped: dict[str, list[Atoms]] = {}
        for frame in candidates:
            grouped.setdefault(str(frame.info["source_id"]), []).append(frame)
        capped = self._balanced_cap(
            sorted(grouped.items()), context.plan.candidate_count
        )
        candidates = [frame for _, _, frame in capped]
        if not candidates:
            raise WorkflowIterationError(
                "sampling.candidate_pool.target removed every MD candidate"
            )
        candidate_ids = [structure_id(frame) for frame in candidates]
        strata = []
        for frame in candidates:
            base = (
                f"T={float(frame.info['temperature']):g}|"
                f"P={float(frame.info['pressure']):g}"
            )
            if "scenario_attempt_id" in frame.info:
                base = (
                    f"A={str(frame.info['scenario_attempt_id'])[:8]}|{base}"
                )
            if "scenario_id" in frame.info:
                base = f"S={str(frame.info['scenario_id'])[:8]}|{base}"
            if "md_window" in frame.info:
                base = f"W={frame.info['md_window']}|{base}"
            strata.append(base)
        audit_plan = replace(
            context.plan,
            dft_budget=effective_budget,
            # Every frontier point needs at least a small DFT audit. Novelty is
            # recorded as evidence and decides convergence, not whether the
            # audit is allowed to happen.
            min_novelty=0.0,
        )
        result = stratified_farthest_point_sampling(
            self.runtime.descriptors(context.artifacts["model"], candidates),
            self.runtime.descriptors(context.artifacts["model"], training),
            candidate_ids,
            strata,
            audit_plan,
        )
        selected = [candidates[index] for index in result.selected_indices]
        if not selected:
            raise WorkflowIterationError(
                "FPS selected no structures; lower sampling.selection.min_novelty"
            )
        selected_path = context.work_dir / "selected-input.xyz"
        ase_write(selected_path, selected, format="extxyz")
        result_path = _write_json(
            context.work_dir / "selection-result.json", asdict(result)
        )
        return StageOutcome(
            artifacts={"selected_input": selected_path, "selection_result": result_path},
            metrics={
                "candidate_count_before_thinning": len(all_candidates),
                "candidate_count_after_thinning": len(candidates),
                "duplicate_candidate_count": duplicate_candidate_count,
                "selected_count": len(selected),
                "remaining_novelty": result.remaining_novelty,
                "configured_dft_budget": context.plan.dft_budget,
                "effective_dft_budget": effective_budget,
                "failed_md_attempt_count": failed_count,
                "novelty_threshold": context.plan.min_novelty,
                "novel_selected_count": sum(
                    value > context.plan.min_novelty
                    for value in result.selected_novelty
                ),
                "counts_by_stratum": dict(result.counts_by_stratum),
            },
        )

    def _label(self, context: StageContext) -> StageOutcome:
        options = self.config.get("dft", {})
        backend = str(options.get("backend", "toy"))
        use_k_stype = str(options.get("use_k_stype", "kspacing"))
        output = context.work_dir / "selected-labels.xyz"
        result = self.runtime.label(
            LabelRequest(
                source=context.artifacts["selected_input"],
                output_file=output,
                work_dir=context.work_dir / "teacher",
                input_file=self._optional_path(
                    options.get("input_path")
                ),
                resource_dir=self._optional_path(options.get("resource_path")),
                n_cpu=int(options.get("n_cpu", 1)),
                use_gamma=bool(options.get("use_gamma", options.get("kpoints_use_gamma", False))),
                kpoint_mode=use_k_stype,
                kspacing=float(options["kspacing"])
                if use_k_stype != "kpoints" and options.get("kspacing") is not None
                else None,
                ka=self._dft_kpoints(options),
                options={"profile": options.get("teacher_profile", "ordinary")},
            ),
            backend,
        )
        validate_spin_dataset(result.frames, require_mforce=True)
        return StageOutcome(
            artifacts={"labeled": result.output_file},
            metrics={"backend": result.backend, "labeled_count": len(result.frames)},
        )

    def _diagnose(self, context: StageContext) -> StageOutcome:
        options = self.config.get("evaluation", {})
        frames = _read_frames(context.artifacts["labeled"])
        _, spin_count = validate_spin_dataset(frames, require_mforce=True)
        thresholds = dict(options.get("max_rmse", {}))
        raw_metrics = self.runtime.predict(
            context.artifacts["model"],
            frames,
            str(options.get("inference_backend", "auto")),
        )
        metrics = {
            f"current_model_{name}": float(value)
            for name, value in raw_metrics.items()
        }
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
                ).items()
            }
            attempt_metrics[attempt_id] = values
            attempt_accepted[attempt_id] = _within_thresholds(
                values, thresholds
            )
        signals = {
            **metrics,
            "diagnostic_only": True,
            "diagnostic_accepted": _within_thresholds(
                raw_metrics, thresholds
            ),
            "attempt_accepted": attempt_accepted,
            "attempt_metrics": attempt_metrics,
            "evaluated_count": len(frames),
            "spin_frame_count": spin_count,
        }
        output = _write_json(
            context.work_dir / "acquisition-signals.json", signals
        )
        return StageOutcome(
            artifacts={"acquisition_signals": output}, metrics=signals
        )

    def _merge(self, context: StageContext) -> StageOutcome:
        original = _read_frames(context.artifacts["training_input"])
        labeled = _read_frames(context.artifacts["labeled"])
        merged = []
        seen = set()
        for frame in [
            *original,
            *labeled,
        ]:
            identifier = structure_id(frame)
            if identifier not in seen:
                seen.add(identifier)
                merged.append(frame)
        validate_spin_dataset(merged, require_mforce=True)
        output = context.work_dir / "train.xyz"
        ase_write(output, merged, format="extxyz")
        return StageOutcome(
            artifacts={"training_set": output},
            metrics={
                "training_count": len(merged),
                "added_count": len(merged) - len(original),
                "duplicate_labeled_count": len(original) + len(labeled) - len(merged),
            },
        )

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
        retrain_required = bool(
            failed_md
            or not diagnostic.get("diagnostic_accepted", False)
            or continue_training
        )
        if not retrain_required:
            decision = {
                "retrained": False,
                "reason": "current model passed trajectory DFT diagnostics",
            }
            artifacts = {
                "retrained_model": context.artifacts["model"],
                "retraining_decision": _write_json(
                    context.work_dir / "retraining-decision.json", decision
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
                    "training_count": len(_read_frames(training_input)),
                    **decision,
                },
            )

        output_name = self._attempt_output_name(
            context.work_dir, "retraining"
        )
        result, frame_count, config_file = self._execute_training(
            context,
            training_input=training_input,
            output_name=output_name,
            warm_start=context.artifacts.get("checkpoint"),
        )
        artifacts: dict[str, Path] = {"retrained_model": result.best_model}
        if result.final_model is not None:
            artifacts["retrained_final_model"] = result.final_model
        if result.checkpoint is not None:
            artifacts["retrained_checkpoint"] = result.checkpoint
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
                    "trajectory DFT diagnostics exceeded thresholds"
                    if not diagnostic.get("diagnostic_accepted", False)
                    else "continued checkpoint training for global validation"
                )
            ),
        }
        artifacts["retraining_decision"] = _write_json(
            context.work_dir / "retraining-decision.json", decision
        )
        return StageOutcome(
            artifacts=artifacts,
            metrics={
                "backend": result.backend,
                "training_count": frame_count,
                **decision,
            },
        )

    def _evaluate(self, context: StageContext) -> StageOutcome:
        options = self.config.get("evaluation", {})
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
        metrics = dict(
            self.runtime.predict(
                context.artifacts["retrained_model"],
                frames,
                str(options.get("inference_backend", "auto")),
            )
        )
        finite = all(np.isfinite(float(value)) for value in metrics.values())
        thresholds = dict(options.get("max_rmse", {}))
        validation_accepted = _within_thresholds(metrics, thresholds)
        validation_score = _threshold_score(metrics, thresholds)
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
        accepted = bool(frames and finite)
        retraining = json.loads(
            context.artifacts["retraining_decision"].read_text(encoding="utf-8")
        )
        diagnostic = json.loads(
            context.artifacts["acquisition_signals"].read_text(encoding="utf-8")
        )
        selection = json.loads(
            context.artifacts["selection_result"].read_text(encoding="utf-8")
        )
        novelty_threshold = float(context.plan.min_novelty)
        remaining_novelty = float(selection.get("remaining_novelty", 0.0))
        novelty_converged = bool(
            novelty_threshold <= 0.0
            or remaining_novelty <= novelty_threshold
        )
        validation_improved = bool(
            retraining["retrained"]
            and (
                previous_validation_score is None
                or validation_score
                < float(previous_validation_score) * 0.99
            )
        )
        signals = {
            **metrics,
            "accepted": accepted,
            "validation_accepted": validation_accepted,
            "validation_score": validation_score,
            "previous_validation_score": previous_validation_score,
            "validation_improved": validation_improved,
            "evaluated_count": len(frames),
            "spin_frame_count": spin_count,
            "added_training_count": added_count,
            "model_trained_on_current_labels": bool(
                retraining["retrained"]
            ),
            "remaining_novelty": remaining_novelty,
            "novelty_threshold": novelty_threshold,
            "novelty_converged": novelty_converged,
            "validation_path": str(self.validation),
        }
        artifacts = {}
        attempt = 1
        while (context.work_dir / f"signals-attempt-{attempt}.json").exists():
            attempt += 1
        recovering = (context.work_dir / "signals.json").exists()
        suffix = f"-attempt-{attempt}" if recovering else ""
        scenario_plan = json.loads(
            context.artifacts["scenario_plan"].read_text(encoding="utf-8")
        )
        previous = None
        if "scenario_maturity" in context.previous_artifacts:
            previous = json.loads(
                context.previous_artifacts["scenario_maturity"].read_text(
                    encoding="utf-8"
                )
            )
        attempt_accepted = {
            str(attempt["attempt_id"]): bool(
                diagnostic.get("attempt_accepted", {}).get(
                    str(attempt["attempt_id"]), False
                )
            )
            for attempt in scenario_plan["attempts"]
        }
        final_model_id = _file_sha256(context.artifacts["retrained_model"])
        history = self.scenario_ladder.record(
            scenario_plan["attempts"],
            completed=scenario_plan["completed"],
            diagnostic_accepted=attempt_accepted,
            history=previous,
            diagnostic=diagnostic,
            validation=metrics,
            validation_accepted=validation_accepted,
            model_improved=validation_improved,
            novelty_converged=novelty_converged,
            final_model_id=final_model_id,
        )
        production_ready = self.scenario_ladder.production_ready(
            scenario_plan["structure_ids"],
            pressure=context.plan.pressure,
            model_id=final_model_id,
            history=history,
        )
        workflow_converged = bool(
            validation_accepted and production_ready and novelty_converged
        )
        workflow_stalled = bool(
            not workflow_converged
            and int(history.get("no_progress_rounds", 0)) >= 2
        )
        history["workflow_converged"] = workflow_converged
        history["workflow_stalled"] = workflow_stalled
        maturity_path = _write_json(
            context.work_dir / f"scenario-maturity{suffix}.json", history
        )
        artifacts["scenario_maturity"] = maturity_path
        signals.update(
            scenario_counts_by_maturity=history["counts_by_maturity"],
            production_ready=production_ready,
            workflow_converged=workflow_converged,
            workflow_stalled=workflow_stalled,
            no_progress_rounds=int(history.get("no_progress_rounds", 0)),
        )
        output = _write_json(context.work_dir / f"signals{suffix}.json", signals)
        artifacts["signals"] = output
        return StageOutcome(artifacts=artifacts, metrics=signals)


__all__ = [
    "WorkflowIterationAdapter",
    "WorkflowIterationError",
    "WorkflowRuntime",
]
