"""Deterministic generation control and stratified FPS for long-running loops."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


STAGES = (
    "train",
    "explore",
    "select",
    "label",
    "diagnose",
    "merge",
    "retrain",
    "evaluate",
)


class IterationError(RuntimeError):
    """Raised when a workflow plan, ledger, or artifact is inconsistent."""


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class GenerationPlan:
    generation: int
    seed: int
    max_selected: int
    selection_novelty_threshold: float = 0.0
    completion_coverage_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.generation < 1:
            raise ValueError("generation must be at least 1")
        if self.seed < 0 or self.max_selected < 1:
            raise ValueError("seed and max_selected must be positive")
        if self.selection_novelty_threshold < 0:
            raise ValueError("selection_novelty_threshold must be non-negative")
        if self.completion_coverage_threshold < 0:
            raise ValueError("completion_coverage_threshold must be non-negative")

    @property
    def sha256(self) -> str:
        return _canonical_hash(asdict(self))


def progressive_plans(
    generations: int,
    *,
    seed: int = 20260721,
    max_selected: int = 8,
    selection_novelty_threshold: float = 0.0,
    completion_coverage_threshold: float = 0.0,
) -> tuple[GenerationPlan, ...]:
    """Build model-generation plans without duplicating route physics."""

    if generations < 1 or max_selected < 1:
        raise ValueError("invalid automatic sampling progression")
    plans = []
    for offset in range(generations):
        generation = offset + 1
        plans.append(
            GenerationPlan(
                generation=generation,
                seed=seed + offset,
                max_selected=int(max_selected),
                selection_novelty_threshold=float(selection_novelty_threshold),
                completion_coverage_threshold=float(completion_coverage_threshold),
            )
        )
    return tuple(plans)


@dataclass(frozen=True)
class SelectionResult:
    plan_sha256: str
    selected_indices: tuple[int, ...]
    selected_ids: tuple[str, ...]
    selected_novelty: tuple[float, ...]
    counts_by_stratum: Mapping[str, int]
    remaining_novelty: float


def _nearest_reference_distances(
    points: np.ndarray,
    reference: np.ndarray,
    *,
    point_batch_size: int = 4096,
    reference_batch_size: int = 512,
) -> np.ndarray:
    """Return exact nearest-reference distances with bounded peak memory."""

    result = np.full(len(points), np.inf, dtype=np.float64)
    for point_start in range(0, len(points), point_batch_size):
        point_stop = min(point_start + point_batch_size, len(points))
        point_chunk = points[point_start:point_stop]
        point_norm = np.einsum("ij,ij->i", point_chunk, point_chunk)
        chunk_min = np.full(len(point_chunk), np.inf, dtype=np.float64)
        for reference_start in range(0, len(reference), reference_batch_size):
            reference_stop = min(
                reference_start + reference_batch_size, len(reference)
            )
            reference_chunk = reference[reference_start:reference_stop]
            reference_norm = np.einsum(
                "ij,ij->i", reference_chunk, reference_chunk
            )
            squared = (
                point_norm[:, None]
                + reference_norm[None, :]
                - 2.0 * point_chunk @ reference_chunk.T
            )
            chunk_min = np.minimum(
                chunk_min,
                np.maximum(squared, 0.0).min(axis=1),
            )
        result[point_start:point_stop] = np.sqrt(chunk_min)
    return result


def stratified_farthest_point_sampling(
    points: np.ndarray,
    reference: np.ndarray,
    candidate_ids: Sequence[str],
    strata: Sequence[str],
    plan: GenerationPlan,
) -> SelectionResult:
    """Select a balanced deterministic subset in normalized feature space."""

    points = np.asarray(points, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if points.ndim != 2 or reference.ndim != 2:
        raise ValueError("points and reference must be two-dimensional")
    if points.shape[1] != reference.shape[1]:
        raise ValueError("points and reference feature dimensions must match")
    if len(points) != len(candidate_ids) or len(points) != len(strata):
        raise ValueError("candidate metadata must match points")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids must be unique")
    if len(points) == 0:
        return SelectionResult(plan.sha256, (), (), (), {}, 0.0)

    original_indices = np.arange(len(points))
    order = np.asarray(sorted(range(len(points)), key=lambda index: candidate_ids[index]))
    points = points[order]
    original_indices = original_indices[order]
    ids = [str(candidate_ids[index]) for index in order]
    groups = [str(strata[index]) for index in order]

    combined = np.vstack([reference, points]) if len(reference) else points
    center = np.median(combined, axis=0)
    scale = np.std(combined, axis=0)
    scale[scale < 1.0e-12] = 1.0
    normalized_points = (points - center) / scale
    normalized_reference = (reference - center) / scale
    if len(reference):
        distances = _nearest_reference_distances(
            normalized_points, normalized_reference
        )
    else:
        distances = np.linalg.norm(
            normalized_points - normalized_points.mean(axis=0), axis=1
        )

    budget = min(plan.max_selected, len(points))
    selected: list[int] = []
    novelty: list[float] = []
    counts = {group: 0 for group in sorted(set(groups))}
    available = np.ones(len(points), dtype=bool)
    while len(selected) < budget:
        novelty_gate = (
            np.ones(len(points), dtype=bool)
            if plan.selection_novelty_threshold == 0.0
            else distances > plan.selection_novelty_threshold
        )
        eligible = np.flatnonzero(available & novelty_gate)
        if not len(eligible):
            break
        active_groups = {groups[index] for index in eligible}
        minimum_count = min(counts[group] for group in active_groups)
        balanced = [
            index
            for index in eligible
            if counts[groups[index]] == minimum_count
        ]
        best = min(balanced, key=lambda index: (-distances[index], ids[index]))
        selected.append(best)
        novelty.append(float(distances[best]))
        counts[groups[best]] += 1
        available[best] = False
        new_distance = np.linalg.norm(
            normalized_points - normalized_points[best], axis=1
        )
        distances = np.minimum(distances, new_distance)

    remaining = float(distances[available].max(initial=0.0))
    return SelectionResult(
        plan_sha256=plan.sha256,
        selected_indices=tuple(int(original_indices[index]) for index in selected),
        selected_ids=tuple(ids[index] for index in selected),
        selected_novelty=tuple(novelty),
        counts_by_stratum={key: value for key, value in counts.items() if value},
        remaining_novelty=remaining,
    )


@dataclass(frozen=True)
class StageOutcome:
    artifacts: Mapping[str, Path]
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageContext:
    generation: int
    generation_dir: Path
    plan: GenerationPlan
    artifacts: Mapping[str, Path]
    previous_artifacts: Mapping[str, Path]
    stage_dir: Path | None = None
    stage_input: Mapping[str, Any] = field(default_factory=dict)
    flat_output: bool = False

    @property
    def work_dir(self) -> Path:
        """Directory owned by the current stage.

        Direct controller users keep writing to ``generation_dir``; workflow
        layout v4 gives each stage a focused subdirectory.
        """

        return self.stage_dir or self.generation_dir


class IterationAdapter(Protocol):
    def run_stage(self, stage: str, context: StageContext) -> StageOutcome: ...


@dataclass(frozen=True)
class GenerationSummary:
    generation: int
    plan_sha256: str
    artifacts: Mapping[str, Path]
    metrics: Mapping[str, Mapping[str, Any]]
    accepted: bool


@dataclass(frozen=True)
class StageSummary:
    generation: int
    stage: str
    artifacts: Mapping[str, Path]
    metrics: Mapping[str, Any]
    generation_complete: bool
    accepted: bool | None


class GenerationController:
    """Execute the fixed generation sequence with an atomic, hash-checked ledger."""

    def __init__(self, root: str | Path, workflow_id: str):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspace = None
        try:
            from .workflow_workspace import WorkflowWorkspace

            workspace = WorkflowWorkspace.locate(self.root)
        except FileNotFoundError:
            workspace = None
        if workspace is not None and workspace.version == 4:
            self.workspace = workspace
            self.ledger_path = workspace.ledger
            self.lock_path = workspace.ledger_lock
        else:
            self.ledger_path = self.root / "workflow-ledger.json"
            self.lock_path = self.root / ".workflow-ledger.lock"
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.workflow_id = workflow_id
        with self._lock():
            self._ledger = self._load()

    @contextmanager
    def _lock(self):
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load(self) -> dict[str, Any]:
        if not self.ledger_path.exists():
            ledger = {"version": 1, "workflow_id": self.workflow_id, "generations": {}}
            self._write(ledger)
            return ledger
        ledger = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        if ledger.get("workflow_id") != self.workflow_id:
            raise IterationError("workflow_id does not match the existing ledger")
        return ledger

    def _write(self, ledger: Mapping[str, Any]) -> None:
        temporary = self.ledger_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.ledger_path)

    @staticmethod
    def _restore_artifacts(stage_record: Mapping[str, Any]) -> dict[str, Path]:
        restored = {}
        for name, record in stage_record.get("artifacts", {}).items():
            path = Path(record["path"])
            if not path.is_file() or _file_hash(path) != record["sha256"]:
                raise IterationError(f"completed artifact drifted: {path}")
            restored[name] = path
        return restored

    def _generation_state(
        self, plan: GenerationPlan
    ) -> tuple[
        dict[str, Any],
        Path,
        dict[str, Path],
        dict[str, Mapping[str, Any]],
        dict[str, Path],
    ]:
        key = str(plan.generation)
        generation_record = self._ledger["generations"].setdefault(
            key, {"plan_sha256": plan.sha256, "stages": {}}
        )
        if generation_record["plan_sha256"] != plan.sha256:
            raise IterationError(
                f"generation {plan.generation} plan changed after it entered the ledger"
            )

        previous_artifacts: dict[str, Path] = {}
        if plan.generation > 1:
            previous = self._ledger["generations"].get(str(plan.generation - 1))
            if not previous or not previous.get("complete") or not previous.get("accepted"):
                raise IterationError("previous generation is not complete and accepted")
            for stage in STAGES:
                previous_stage = previous["stages"].get(stage)
                if previous_stage is None:
                    raise IterationError("previous generation has an incomplete stage ledger")
                previous_artifacts.update(self._restore_artifacts(previous_stage))

        generation_dir = (
            self.workspace.generation_dir(plan.generation)
            if self.workspace is not None
            else self.root / f"Generation-{plan.generation}"
        )
        artifacts: dict[str, Path] = {}
        metrics: dict[str, Mapping[str, Any]] = {}
        missing_seen = False
        for stage in STAGES:
            stage_record = generation_record["stages"].get(stage)
            if stage_record is None:
                missing_seen = True
                continue
            if missing_seen:
                raise IterationError("generation stage ledger is not a contiguous prefix")
            restored = self._restore_artifacts(stage_record)
            overlap = set(artifacts).intersection(restored)
            if overlap:
                raise IterationError(f"duplicate artifact names: {sorted(overlap)}")
            artifacts.update(restored)
            metrics[stage] = stage_record.get("metrics", {})
        return generation_record, generation_dir, artifacts, metrics, previous_artifacts

    def _next_stage(self, plan: GenerationPlan) -> str | None:
        generation_record, _, _, metrics, _ = self._generation_state(plan)
        completed = len(metrics)
        if completed < len(STAGES):
            if generation_record.get("complete"):
                raise IterationError("generation is marked complete before all stages finished")
            return STAGES[completed]
        accepted = metrics["evaluate"].get("accepted")
        if not isinstance(accepted, bool):
            raise IterationError("evaluate stage must report an accepted boolean")
        changed = False
        if generation_record.get("complete"):
            if generation_record.get("accepted") is not accepted:
                raise IterationError("generation acceptance does not match evaluate stage")
        else:
            generation_record["accepted"] = accepted
            generation_record["complete"] = True
            changed = True
        if self.workspace is not None and accepted:
            try:
                publication = (
                    self.workspace.prepare_generation_publication(
                        plan.generation,
                        generation_record,
                    )
                )
            except (OSError, ValueError) as error:
                raise IterationError(
                    f"cannot prepare generation {plan.generation} publication: "
                    f"{error}"
                ) from error
            if generation_record.get("publication") != publication:
                generation_record["publication"] = publication
                changed = True
        if changed:
            self._write(self._ledger)
        if self.workspace is not None:
            try:
                self.workspace.activate_generation(
                    plan.generation,
                    generation_record,
                )
            except (OSError, ValueError) as error:
                raise IterationError(
                    f"generation {plan.generation} is committed but its results "
                    f"projection needs repair: {error}"
                ) from error
        return None

    def next_stage(self, plan: GenerationPlan) -> str | None:
        """Return the only stage that may run next for this generation."""

        with self._lock():
            self._ledger = self._load()
            return self._next_stage(plan)

    def stage_context(
        self, plan: GenerationPlan, stage: str | None = None
    ) -> tuple[str, StageContext]:
        """Describe the next stage without executing or mutating it.

        Persistent workflow controllers use this method to build an immutable
        task for another machine.  The returned paths remain protected by the
        ledger hashes and are checked again when the result is committed.
        """

        with self._lock():
            self._ledger = self._load()
            expected = self._next_stage(plan)
            if expected is None:
                raise IterationError(f"generation {plan.generation} is already complete")
            requested = expected if stage is None else stage
            if requested != expected:
                raise IterationError(
                    f"generation {plan.generation} expects stage {expected}, not {requested}"
                )
            (
                _,
                generation_dir,
                artifacts,
                _,
                previous_artifacts,
            ) = self._generation_state(plan)
            generation_dir.mkdir(parents=True, exist_ok=True)
            context = StageContext(
                generation=plan.generation,
                generation_dir=generation_dir,
                plan=plan,
                artifacts=dict(artifacts),
                previous_artifacts=dict(previous_artifacts),
                stage_dir=(
                    self.workspace.stage_dir(plan.generation, requested)
                    if self.workspace is not None
                    else generation_dir
                ),
                flat_output=self.workspace is not None,
            )
            context.work_dir.mkdir(parents=True, exist_ok=True)
            return requested, context

    def _commit_outcome(
        self,
        plan: GenerationPlan,
        stage: str,
        outcome: StageOutcome,
    ) -> StageSummary:
        expected = self._next_stage(plan)
        if expected is None:
            raise IterationError(f"generation {plan.generation} is already complete")
        if stage != expected:
            raise IterationError(
                f"generation {plan.generation} expects stage {expected}, not {stage}"
            )
        generation_record, _, artifacts, _, _ = self._generation_state(plan)
        overlap = set(artifacts).intersection(outcome.artifacts)
        if overlap:
            raise IterationError(f"duplicate artifact names: {sorted(overlap)}")
        artifact_records = {}
        resolved_artifacts = {}
        for name, raw_path in outcome.artifacts.items():
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise IterationError(
                    f"stage {stage} did not produce artifact {path}"
                )
            artifact_records[name] = {
                "path": str(path),
                "sha256": _file_hash(path),
            }
            resolved_artifacts[name] = path
        stage_record = {
            "artifacts": artifact_records,
            "metrics": dict(outcome.metrics),
        }
        generation_record["stages"][stage] = stage_record
        complete = stage == STAGES[-1]
        accepted = None
        if complete:
            accepted = stage_record["metrics"].get("accepted")
            if not isinstance(accepted, bool):
                raise IterationError(
                    "evaluate stage must report an accepted boolean"
                )
            generation_record["accepted"] = accepted
            generation_record["complete"] = True
            if self.workspace is not None and accepted:
                try:
                    generation_record["publication"] = (
                        self.workspace.prepare_generation_publication(
                            plan.generation,
                            generation_record,
                        )
                    )
                except (OSError, ValueError) as error:
                    raise IterationError(
                        f"cannot prepare generation {plan.generation} "
                        f"publication: {error}"
                    ) from error

        # The ledger is the commit point.  Human-facing links are derived
        # projections and are switched only after this atomic write.
        self._write(self._ledger)
        if complete and self.workspace is not None:
            try:
                self.workspace.activate_generation(
                    plan.generation,
                    generation_record,
                )
            except (OSError, ValueError) as error:
                raise IterationError(
                    f"generation {plan.generation} is committed but its results "
                    f"projection needs repair: {error}"
                ) from error
        return StageSummary(
            generation=plan.generation,
            stage=stage,
            artifacts=resolved_artifacts,
            metrics=stage_record["metrics"],
            generation_complete=complete,
            accepted=accepted,
        )

    def commit_stage(
        self, plan: GenerationPlan, stage: str, outcome: StageOutcome
    ) -> StageSummary:
        """Atomically accept one externally executed stage result."""

        with self._lock():
            self._ledger = self._load()
            return self._commit_outcome(plan, stage, outcome)

    def run_stage(
        self,
        plan: GenerationPlan,
        adapter: IterationAdapter,
        stage: str | None = None,
    ) -> StageSummary:
        """Run exactly one ledger-authorized stage.

        This is the resource-boundary API: training and MD may be submitted as
        separate Slurm jobs while sharing one deterministic workflow ledger.
        """

        with self._lock():
            # A job can wait on this lock for a previous resource stage. Reload
            # after acquiring it so a stale process cannot repeat that stage.
            self._ledger = self._load()
            expected = self._next_stage(plan)
            if expected is None:
                raise IterationError(f"generation {plan.generation} is already complete")
            requested = expected if stage is None else stage
            if requested != expected:
                raise IterationError(
                    f"generation {plan.generation} expects stage {expected}, not {requested}"
                )
            (
                generation_record,
                generation_dir,
                artifacts,
                _,
                previous_artifacts,
            ) = self._generation_state(plan)
            generation_dir.mkdir(parents=True, exist_ok=True)
            context = StageContext(
                generation=plan.generation,
                generation_dir=generation_dir,
                plan=plan,
                artifacts=dict(artifacts),
                previous_artifacts=dict(previous_artifacts),
                stage_dir=(
                    self.workspace.stage_dir(plan.generation, requested)
                    if self.workspace is not None
                    else generation_dir
                ),
                flat_output=self.workspace is not None,
            )
            context.work_dir.mkdir(parents=True, exist_ok=True)
            outcome = adapter.run_stage(requested, context)
            # Direct execution keeps the lock for its whole stage.  External
            # controllers use ``stage_context`` + ``commit_stage`` and share
            # the same commit primitive below.
            return self._commit_outcome(plan, requested, outcome)

    def reopen_rejected(
        self, plan: GenerationPlan, *, from_stage: str = "retrain"
    ) -> None:
        """Reopen a rejected generation without discarding its failed attempt."""

        if from_stage not in STAGES:
            raise IterationError(f"unknown recovery stage: {from_stage}")
        with self._lock():
            self._ledger = self._load()
            generation_record, _, _, _, _ = self._generation_state(plan)
            if not generation_record.get("complete") or generation_record.get(
                "accepted"
            ) is not False:
                raise IterationError(
                    f"generation {plan.generation} is not complete and rejected"
                )
            start = STAGES.index(from_stage)
            archived = {
                stage: generation_record["stages"].pop(stage)
                for stage in STAGES[start:]
            }
            recoveries = generation_record.setdefault("recovery_attempts", [])
            recoveries.append(
                {
                    "attempt": len(recoveries) + 1,
                    "from_stage": from_stage,
                    "accepted": False,
                    "stages": archived,
                }
            )
            generation_record.pop("complete", None)
            generation_record.pop("accepted", None)
            self._write(self._ledger)

    def _summary(self, plan: GenerationPlan) -> GenerationSummary:
        generation_record, _, artifacts, metrics, _ = self._generation_state(plan)
        if not generation_record.get("complete"):
            raise IterationError(f"generation {plan.generation} is not complete")
        accepted = generation_record.get("accepted")
        if not isinstance(accepted, bool):
            raise IterationError("completed generation is missing acceptance")
        return GenerationSummary(
            generation=plan.generation,
            plan_sha256=plan.sha256,
            artifacts=artifacts,
            metrics=metrics,
            accepted=accepted,
        )

    def run_generation(
        self, plan: GenerationPlan, adapter: IterationAdapter
    ) -> GenerationSummary:
        while self.next_stage(plan) is not None:
            self.run_stage(plan, adapter)
        return self._summary(plan)

    def run_workflow(
        self, plans: Sequence[GenerationPlan], adapter: IterationAdapter
    ) -> tuple[GenerationSummary, ...]:
        expected = list(range(1, len(plans) + 1))
        observed = [plan.generation for plan in plans]
        if observed != expected:
            raise IterationError("workflow plans must be contiguous and start at generation 1")
        summaries = []
        for plan in plans:
            summary = self.run_generation(plan, adapter)
            summaries.append(summary)
            if not summary.accepted:
                break
        return tuple(summaries)


__all__ = [
    "GenerationController",
    "GenerationPlan",
    "GenerationSummary",
    "IterationAdapter",
    "IterationError",
    "SelectionResult",
    "StageContext",
    "StageOutcome",
    "StageSummary",
    "progressive_plans",
    "stratified_farthest_point_sampling",
]
