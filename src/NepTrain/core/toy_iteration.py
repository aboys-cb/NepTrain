"""Two-generation Toy Teacher workflow for testing real loop control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from .content_addressing import file_sha256
from .labeling import LabelRequest, label
from .dft.toy import ToyTeacher
from .iteration import (
    GenerationController,
    GenerationPlan,
    StageContext,
    StageOutcome,
    stratified_farthest_point_sampling,
)
from .persistence import atomic_write_json
from .spin import validate_spin_dataset
from .scenario import ScenarioLadder
from .scientific_data import structure_id
from .toy_workflow import (
    toy_base_frame,
    toy_candidate_frames,
    toy_raw_features,
)


class ToyIterationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToyGenerationPlan(GenerationPlan):
    """Toy-only sampling physics used by the deterministic smoke workflow."""

    steps: int = 100
    temperatures: tuple[float, ...] = (300.0, 500.0, 700.0)
    pressure: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.steps < 1 or not self.temperatures:
            raise ValueError("toy steps and temperatures must be non-empty")


@dataclass(frozen=True)
class ToyIterationReport:
    profile: str
    generations_requested: int
    generations_completed: int
    max_selected: tuple[int, ...]
    steps: tuple[int, ...]
    candidate_counts: tuple[int, ...]
    selected_counts: tuple[int, ...]
    strata_counts: tuple[Mapping[str, int], ...]
    remaining_novelty: tuple[float, ...]
    training_counts: tuple[int, ...]
    coverage_p95: tuple[float, ...]
    scenario_steps: tuple[tuple[int, ...], ...]
    maturity_counts: tuple[Mapping[str, int], ...]
    deterministic_selection: bool
    resume_reused_artifacts: bool
    passed: bool


def _frames(path: Path) -> list[Atoms]:
    loaded = ase_read(path, index=":", format="extxyz")
    return loaded if isinstance(loaded, list) else [loaded]


def _prepare_inputs(root: Path, profile: str, seed: int) -> tuple[Path, Path]:
    teacher = ToyTeacher(profile)
    initial_raw = [toy_base_frame(profile == "spin")]
    initial_raw.extend(
        toy_candidate_frames(profile, seed - 11, 3, generation=1)[:3]
    )
    initial = [teacher.label(frame) for frame in initial_raw]
    initial_path = root / "initial-train.xyz"
    ase_write(initial_path, initial, format="extxyz")

    validation_raw = toy_candidate_frames(
        profile,
        seed + 10_000,
        18,
        generation=2,
        temperatures=(300.0, 500.0),
    )
    validation = [teacher.label(frame) for frame in validation_raw]
    validation_path = root / "validation.xyz"
    ase_write(validation_path, validation, format="extxyz")
    return initial_path, validation_path


class ToyIterationAdapter:
    """Local Adapter that exercises every controller stage without remote work."""

    def __init__(
        self,
        *,
        profile: str,
        initial_training: Path,
        validation: Path,
        candidate_count: int = 24,
    ):
        self.profile = profile
        self.initial_training = initial_training.resolve()
        self.validation = validation.resolve()
        self.candidate_count = int(candidate_count)
        self.scenario_ladder = ScenarioLadder(
            {
                "smoke_passed": 100,
                "short_stable": 400,
                "long_stable": 1600,
                "production_ready": 6400,
            },
            temperature_path=(300.0, 500.0, 700.0),
            replicas={
                "smoke_passed": 1,
                "short_stable": 1,
                "long_stable": 1,
                "production_ready": 1,
            },
        )
        self.stage_calls: list[tuple[int, str]] = []

    def run_stage(self, stage: str, context: StageContext) -> StageOutcome:
        self.stage_calls.append((context.generation, stage))
        method = getattr(self, f"_{stage}", None)
        if method is None:
            raise ToyIterationError(f"unsupported Toy iteration stage: {stage}")
        return method(context)

    def _train(self, context: StageContext) -> StageOutcome:
        if context.generation > 1:
            training_input = context.previous_artifacts["training_set"]
            model = context.previous_artifacts["activated_model"]
            frames = _frames(training_input)
            validate_spin_dataset(frames, require_mforce=True)
            return StageOutcome(
                artifacts={"training_input": training_input, "model": model},
                metrics={"training_count": len(frames), "reused_previous_model": True},
            )

        training_input = self.initial_training
        frames = _frames(training_input)
        validate_spin_dataset(frames, require_mforce=True)
        model = context.generation_dir / "toy-model.json"
        atomic_write_json(
            model,
            {
                "kind": "coverage-surrogate",
                "profile": self.profile,
                "training_count": len(frames),
                "training_sha256": file_sha256(training_input),
            },
        )
        return StageOutcome(
            artifacts={"training_input": training_input, "model": model},
            metrics={"training_count": len(frames), "reused_previous_model": False},
        )

    def _explore(self, context: StageContext) -> StageOutcome:
        base_identifier = structure_id(toy_base_frame(self.profile == "spin"))
        history = None
        if "scenario_maturity" in context.previous_artifacts:
            history = json.loads(
                context.previous_artifacts["scenario_maturity"].read_text(
                    encoding="utf-8"
                )
            )
        attempts = self.scenario_ladder.schedule(
            [base_identifier],
            pressure=context.plan.pressure,
            generation=context.generation,
            seed=context.plan.seed,
            limit=len(context.plan.temperatures),
            model_id=file_sha256(context.artifacts["model"]),
            history=history,
        )
        by_temperature = {attempt.temperature: attempt for attempt in attempts}
        candidates = toy_candidate_frames(
            self.profile,
            context.plan.seed,
            self.candidate_count,
            generation=context.generation,
            temperatures=tuple(by_temperature),
            pressure=context.plan.pressure,
        )
        for frame in candidates:
            attempt = by_temperature[float(frame.info["temperature"])]
            frame.info.update(
                scenario_id=attempt.scenario_id,
                scenario_structure_id=attempt.structure_id,
                maturity_target=attempt.target_level,
                md_steps=attempt.steps,
            )
        output = context.generation_dir / "candidates.xyz"
        ase_write(output, candidates, format="extxyz")
        scenario_plan = context.generation_dir / "scenario-plan.json"
        atomic_write_json(
            scenario_plan,
            {
                "version": 2,
                "structure_ids": [base_identifier],
                "attempts": self.scenario_ladder.serialize(attempts),
                "completed": {
                    attempt.attempt_id: True for attempt in attempts
                },
            },
        )
        return StageOutcome(
            artifacts={"candidates": output, "scenario_plan": scenario_plan},
            metrics={
                "candidate_count": len(candidates),
                "temperatures": list(context.plan.temperatures),
                "pressure": context.plan.pressure,
                "steps": context.plan.steps,
                "scenario_temperatures": sorted(
                    {attempt.temperature for attempt in attempts}
                ),
                "scenario_temperatures_by_route": {
                    "default": sorted(
                        {attempt.temperature for attempt in attempts}
                    )
                },
                "scenario_steps": sorted({attempt.steps for attempt in attempts}),
                "scenario_targets": sorted(
                    {attempt.target_level for attempt in attempts}
                ),
            },
        )

    def _select(self, context: StageContext) -> StageOutcome:
        all_candidates = _frames(context.artifacts["candidates"])
        candidates = list(all_candidates)
        training = _frames(context.artifacts["training_input"])
        candidate_ids = [structure_id(frame) for frame in candidates]
        strata = [
            f"T={float(frame.info['temperature']):g}|P={float(frame.info['pressure']):g}"
            for frame in candidates
        ]
        result = stratified_farthest_point_sampling(
            toy_raw_features(candidates, self.profile),
            toy_raw_features(training, self.profile),
            candidate_ids,
            strata,
            context.plan,
        )
        selected = [candidates[index] for index in result.selected_indices]
        selected_path = context.generation_dir / "selected-input.xyz"
        ase_write(selected_path, selected, format="extxyz")
        result_path = context.generation_dir / "selection-result.json"
        atomic_write_json(result_path, asdict(result))
        return StageOutcome(
            artifacts={"selected_input": selected_path, "selection_result": result_path},
            metrics={
                "candidate_count_before_deduplication": len(all_candidates),
                "candidate_count_after_deduplication": len(candidates),
                "selected_count": len(selected),
                "remaining_novelty": result.remaining_novelty,
                "counts_by_stratum": dict(result.counts_by_stratum),
            },
        )

    def _label(self, context: StageContext) -> StageOutcome:
        output = context.generation_dir / "selected-labels.xyz"
        result = label(
            LabelRequest(
                context.artifacts["selected_input"],
                output,
                context.generation_dir / "teacher",
                settings={"profile": self.profile},
            ),
            "toy",
        )
        validate_spin_dataset(result.frames, require_mforce=True)
        return StageOutcome(
            artifacts={"labeled": output},
            metrics={"labeled_count": len(result.frames)},
        )

    def _diagnose(self, context: StageContext) -> StageOutcome:
        labeled = _frames(context.artifacts["labeled"])
        training = _frames(context.artifacts["training_input"])
        raw_labeled = toy_raw_features(labeled, self.profile)
        raw_training = toy_raw_features(training, self.profile)
        combined = np.vstack([raw_training, raw_labeled])
        center = np.median(combined, axis=0)
        scale = np.std(combined, axis=0)
        scale[scale < 1.0e-12] = 1.0
        distances = np.linalg.norm(
            ((raw_labeled - center) / scale)[:, None, :]
            - ((raw_training - center) / scale)[None, :, :],
            axis=2,
        ).min(axis=1)
        signals = {
            "diagnostic_only": True,
            "evaluated_count": len(labeled),
            "current_model_coverage_max": float(distances.max(initial=0.0)),
            "current_model_coverage_p95": float(np.quantile(distances, 0.95)),
        }
        output = context.generation_dir / "acquisition-signals.json"
        atomic_write_json(output, signals)
        return StageOutcome(
            artifacts={"acquisition_signals": output}, metrics=signals
        )

    def _merge(self, context: StageContext) -> StageOutcome:
        merged = []
        seen = set()
        for frame in [
            *_frames(context.artifacts["training_input"]),
            *_frames(context.artifacts["labeled"]),
        ]:
            identifier = structure_id(frame)
            if identifier not in seen:
                seen.add(identifier)
                merged.append(frame)
        validate_spin_dataset(merged, require_mforce=True)
        output = context.generation_dir / "train.xyz"
        ase_write(output, merged, format="extxyz")
        return StageOutcome(
            artifacts={"training_set": output},
            metrics={"training_count": len(merged)},
        )

    def _retrain(self, context: StageContext) -> StageOutcome:
        training_input = context.artifacts["training_set"]
        frames = _frames(training_input)
        validate_spin_dataset(frames, require_mforce=True)
        model = context.generation_dir / "toy-retrained-model.json"
        atomic_write_json(
            model,
            {
                "kind": "coverage-surrogate",
                "profile": self.profile,
                "training_count": len(frames),
                "training_sha256": file_sha256(training_input),
            },
        )
        return StageOutcome(
            artifacts={"retrained_model": model},
            metrics={"training_count": len(frames)},
        )

    def _evaluate(self, context: StageContext) -> StageOutcome:
        training = _frames(context.artifacts["training_set"])
        validation = _frames(self.validation)
        validate_spin_dataset(training, require_mforce=True)
        model = json.loads(
            context.artifacts["retrained_model"].read_text(encoding="utf-8")
        )
        if int(model["training_count"]) != len(training):
            raise ToyIterationError("retrained Toy model does not match merged training set")
        raw_training = toy_raw_features(training, self.profile)
        raw_validation = toy_raw_features(validation, self.profile)
        combined = np.vstack([raw_training, raw_validation])
        center = np.median(combined, axis=0)
        scale = np.std(combined, axis=0)
        scale[scale < 1.0e-12] = 1.0
        normalized_training = (raw_training - center) / scale
        normalized_validation = (raw_validation - center) / scale
        distances = np.linalg.norm(
            normalized_validation[:, None, :] - normalized_training[None, :, :],
            axis=2,
        ).min(axis=1)
        coverage_p95 = float(np.quantile(distances, 0.95))
        previous_p95 = None
        if "signals" in context.previous_artifacts:
            previous_p95 = json.loads(
                context.previous_artifacts["signals"].read_text(encoding="utf-8")
            )["coverage_p95"]
        selected_count = len(
            json.loads(context.artifacts["selection_result"].read_text(encoding="utf-8"))[
                "selected_indices"
            ]
        )
        accepted = (
            selected_count > 0
            and np.isfinite(coverage_p95)
            and (previous_p95 is None or coverage_p95 <= previous_p95 * 1.05)
        )
        signals = {
            "accepted": bool(accepted),
            "model_trained_on_current_labels": True,
            "coverage_max": float(distances.max(initial=0.0)),
            "coverage_p95": coverage_p95,
            "previous_coverage_p95": previous_p95,
            "selected_count": selected_count,
            "training_count": len(training),
        }
        previous = None
        if "scenario_maturity" in context.previous_artifacts:
            previous = json.loads(
                context.previous_artifacts["scenario_maturity"].read_text(
                    encoding="utf-8"
                )
            )
        scenario_plan = json.loads(
            context.artifacts["scenario_plan"].read_text(encoding="utf-8")
        )
        maturity = self.scenario_ladder.record(
            scenario_plan["attempts"],
            completed=scenario_plan["completed"],
            diagnostic_accepted={
                attempt["attempt_id"]: bool(accepted)
                for attempt in scenario_plan["attempts"]
            },
            history=previous,
            diagnostic=json.loads(
                context.artifacts["acquisition_signals"].read_text(encoding="utf-8")
            ),
            validation=signals,
            validation_accepted=bool(accepted),
            model_improved=True,
            novelty_converged=True,
            final_model_id=file_sha256(context.artifacts["retrained_model"]),
        )
        signals["scenario_counts_by_maturity"] = maturity["counts_by_maturity"]
        output = context.generation_dir / "signals.json"
        atomic_write_json(output, signals)
        maturity_path = context.generation_dir / "scenario-maturity.json"
        atomic_write_json(maturity_path, maturity)
        return StageOutcome(
            artifacts={
                "signals": output,
                "scenario_maturity": maturity_path,
                "activated_model": context.artifacts["retrained_model"],
            },
            metrics=signals,
        )


def _selected_ids(summaries) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(
            json.loads(summary.artifacts["selection_result"].read_text(encoding="utf-8"))[
                "selected_ids"
            ]
        )
        for summary in summaries
    )


def run_toy_iteration_smoke(
    output_dir: str | Path,
    *,
    profile: str = "spin",
    generations: int = 2,
    seed: int = 20260721,
    max_selected: int = 8,
    force: bool = False,
) -> ToyIterationReport:
    if profile not in {"ordinary", "spin"}:
        raise ToyIterationError("Toy iteration profile must be ordinary or spin")
    root = Path(output_dir).expanduser().resolve()
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
        Path(tempfile.gettempdir()).resolve(),
    }
    if root in protected or len(root.parts) < 3:
        raise ToyIterationError(f"refusing unsafe Toy iteration output: {root}")
    if root.exists():
        if not force:
            raise ToyIterationError(f"Toy iteration output already exists: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    plans = tuple(
        ToyGenerationPlan(
            generation=offset + 1,
            seed=seed + offset,
            max_selected=max_selected,
        )
        for offset in range(generations)
    )

    initial, validation = _prepare_inputs(root, profile, seed)
    workflow = root / "workflow"
    adapter = ToyIterationAdapter(
        profile=profile, initial_training=initial, validation=validation
    )
    controller = GenerationController(workflow, f"toy-{profile}-{seed}")
    summaries = controller.run_workflow(plans, adapter)
    calls_after_first_run = len(adapter.stage_calls)
    resumed = controller.run_workflow(plans, adapter)
    resume_reused = len(adapter.stage_calls) == calls_after_first_run

    replay_root = root / "replay"
    replay_root.mkdir()
    replay_initial, replay_validation = _prepare_inputs(replay_root, profile, seed)
    replay_adapter = ToyIterationAdapter(
        profile=profile,
        initial_training=replay_initial,
        validation=replay_validation,
    )
    replay = GenerationController(
        replay_root / "workflow", f"toy-{profile}-{seed}"
    ).run_workflow(plans, replay_adapter)
    deterministic = _selected_ids(summaries) == _selected_ids(replay)

    selected_counts = tuple(
        int(summary.metrics["select"]["selected_count"]) for summary in summaries
    )
    strata_counts = tuple(
        dict(summary.metrics["select"]["counts_by_stratum"])
        for summary in summaries
    )
    remaining_novelty = tuple(
        float(summary.metrics["select"]["remaining_novelty"])
        for summary in summaries
    )
    training_counts = tuple(
        int(summary.metrics["merge"]["training_count"]) for summary in summaries
    )
    coverage = tuple(
        float(summary.metrics["evaluate"]["coverage_p95"]) for summary in summaries
    )
    scenario_steps = tuple(
        tuple(int(value) for value in summary.metrics["explore"]["scenario_steps"])
        for summary in summaries
    )
    maturity_counts = tuple(
        dict(summary.metrics["evaluate"]["scenario_counts_by_maturity"])
        for summary in summaries
    )
    passed = (
        len(summaries) == generations
        and len(resumed) == generations
        and len(replay) == generations
        and all(summary.accepted for summary in summaries)
        and deterministic
        and resume_reused
        and all(count > 0 for count in selected_counts)
        and all(plan.max_selected == max_selected for plan in plans)
        and all(b > a for a, b in zip(training_counts, training_counts[1:]))
    )
    report = ToyIterationReport(
        profile=profile,
        generations_requested=generations,
        generations_completed=len(summaries),
        max_selected=tuple(plan.max_selected for plan in plans),
        steps=tuple(plan.steps for plan in plans),
        candidate_counts=tuple(
            int(summary.metrics["explore"]["candidate_count"])
            for summary in summaries
        ),
        selected_counts=selected_counts,
        strata_counts=strata_counts,
        remaining_novelty=remaining_novelty,
        training_counts=training_counts,
        coverage_p95=coverage,
        scenario_steps=scenario_steps,
        maturity_counts=maturity_counts,
        deterministic_selection=deterministic,
        resume_reused_artifacts=resume_reused,
        passed=passed,
    )
    atomic_write_json(root / "toy-iteration-report.json", asdict(report))
    if not passed:
        raise ToyIterationError(
            f"Toy iteration smoke failed; see {root / 'toy-iteration-report.json'}"
        )
    return report


__all__ = [
    "ToyIterationAdapter",
    "ToyIterationError",
    "ToyIterationReport",
    "run_toy_iteration_smoke",
]
