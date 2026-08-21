"""Persistent, scheduler-independent controller for NepTrain workflows."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping

from ase.io import read as ase_read

from .content_addressing import canonical_sha256, file_sha256
from .persistence import atomic_write_json
from .workflow_workspace import WorkflowWorkspace
from .config import (
    ConfigError,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_STRUCTURES_PER_LABEL_JOB,
    load_config,
)
from .execution import (
    ExecutionError,
    ExecutionHandle,
    ExecutionStatus,
    ExecutionTarget,
    PermanentExecutionError,
    StageExecutor,
    SubmissionDeferred,
    build_stage_task,
    executor_for,
    load_stage_result,
)
from .generation_policy import LEGACY_GENERATION_PROTOCOL
from .generation_policy import (
    generation_disposition,
    generation_stage_sequence,
    stage_for_role,
)
from .iteration import GenerationController, GenerationPlan, IterationError, StageOutcome


class ControllerError(RuntimeError):
    """Raised when a workflow controller cannot safely make progress."""


_RESOURCE_FOR_STAGE = {
    "train": "training",
    "validate": "analysis",
    "explore": "sampling",
    "select": "analysis",
    "label": "labeling",
    "diagnose": "analysis",
    "merge": "analysis",
    "retrain": "training",
    "evaluate": "analysis",
    "update": "analysis",
}
_EXECUTION_UNKNOWN_GRACE_SECONDS = 300.0
_MAX_PARALLEL_OBSERVATIONS = 4
_MAX_PARALLEL_LAUNCHES = 4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(
    path: Path,
    default: Mapping[str, Any] | None = None,
    *,
    role: str = "controller state",
) -> dict[str, Any]:
    if not path.is_file():
        if default is not None:
            return dict(default)
        raise ControllerError(f"required controller state file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControllerError(
            f"cannot read {role} JSON at {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise ControllerError(
            f"{role} JSON at {path} must contain an object"
        )
    return dict(value)


def _record_matches(record: Mapping[str, Any]) -> bool:
    path = Path(str(record["path"]))
    if record.get("kind", "file") == "file":
        return path.is_file() and file_sha256(path) == record.get("sha256")
    if not path.is_dir():
        return False
    entries = [
        {"path": str(item.relative_to(path)), "sha256": file_sha256(item)}
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    digest = canonical_sha256(entries)
    return digest == record.get("sha256")


def _plan(path: Path) -> GenerationPlan:
    value = _read_json(path, role="generation plan")
    return GenerationPlan(**value)


def _controller_command() -> list[str]:
    return [sys.executable, "-m", "NepTrain.cli.cli"]


def _process_matches(pid: int, project: Path) -> bool:
    environment = os.environ.copy()
    environment["COLUMNS"] = "10000"
    try:
        completed = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
    except OSError:
        return False
    command = completed.stdout.strip()
    return bool(command and "controller" in command and str(project) in command)


def controller_running(project: str | Path) -> bool:
    workspace = WorkflowWorkspace.locate(project)
    workspace.controller_lock.parent.mkdir(parents=True, exist_ok=True)
    with workspace.controller_lock.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    return False


ExecutorFactory = Callable[[ExecutionTarget], StageExecutor]


@dataclass(frozen=True)
class ControllerTick:
    state: str
    generation: int | None = None
    stage: str | None = None
    target: str | None = None
    execution_id: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class RestartPlan:
    generation: int
    from_stage: str
    task_scope: str
    reused_stages: tuple[str, ...]
    restarted_stages: tuple[str, ...]
    preserved_tasks: int = 0
    retried_tasks: int = 0


class PersistentController:
    """Advance one workflow through portable stage tasks.

    The ledger remains the scientific source of truth.  This module owns only
    execution intent and remote handles, so restarting it cannot repeat an
    already committed scientific stage.
    """

    def __init__(
        self,
        project: str | Path,
        *,
        executor_factory: ExecutorFactory = executor_for,
    ):
        self.workspace = WorkflowWorkspace.locate(project)
        if self.workspace.version != 4:
            raise ControllerError(
                "persistent controllers require workflow layout v4"
            )
        from .workflow import WorkflowError, _normalise_manifest

        try:
            self.manifest = _normalise_manifest(
                _read_json(self.workspace.manifest, role="workflow manifest"),
                self.workspace.manifest,
            )
        except WorkflowError as error:
            raise ControllerError(str(error)) from error
        if self.manifest.get("orchestration") != "controller":
            raise ControllerError(
                "workflow was not prepared for the current controller"
            )
        for record in [
            self.manifest["config"],
            self.manifest["initial_training"],
            *self.manifest.get("plans", []),
            *self.manifest.get("dependencies", []),
        ]:
            if not _record_matches(record):
                raise ControllerError(
                    f"prepared workflow artifact drifted: {record['path']}"
                )
        self.workflow_id = str(self.manifest["workflow_id"])
        try:
            self.config, _ = load_config(self.workspace.project_file)
        except ConfigError as error:
            raise ControllerError(f"invalid project configuration: {error}") from error
        self.plans = tuple(_plan(Path(item["path"])) for item in self.manifest["plans"])
        self.initial_training = Path(self.manifest["initial_training"]["path"])
        self.generation_controller = GenerationController(
            self.workspace.root,
            self.workflow_id,
            generation_protocol=str(
                self.manifest.get(
                    "generation_protocol", LEGACY_GENERATION_PROTOCOL
                )
            ),
        )
        self.executor_factory = executor_factory
        (
            self.targets,
            self.stage_targets,
            self.sampling_route_targets,
        ) = self._execution_config(
            self.config, self.workspace.root
        )
        self.state = _read_json(
            self.workspace.controller_file,
            {
                "protocol": "neptrain.controller.v1",
                "workflow_id": self.workflow_id,
                "state": "idle",
                "history": [],
                "current": None,
            },
        )
        if self.state.get("workflow_id") != self.workflow_id:
            raise ControllerError("controller state belongs to a different workflow")

    @staticmethod
    def _execution_config(
        config: Mapping[str, Any], base_dir: Path
    ) -> tuple[dict[str, ExecutionTarget], dict[str, str], dict[str, str]]:
        execution = config.get("execution", {})
        raw_targets = execution.get("targets", {})
        routes = {
            str(key): str(value)
            for key, value in dict(execution.get("stage_targets", {})).items()
        }
        sampling_route_targets = {
            str(key): str(value)
            for key, value in dict(
                execution.get("sampling_route_targets", {})
            ).items()
        }
        required = {"training", "sampling", "labeling", "analysis"}
        missing = sorted(required - set(routes))
        if missing:
            raise ControllerError(
                "execution.stage_targets is missing " + ", ".join(missing)
            )
        targets = {}
        for name, value in dict(raw_targets).items():
            normalized = dict(value)
            setup = normalized.get("setup_script")
            if setup:
                candidate = Path(setup).expanduser()
                local = (base_dir / candidate).resolve() if not candidate.is_absolute() else candidate
                if local.is_file():
                    normalized["setup_script"] = str(local)
            targets[str(name)] = ExecutionTarget.from_mapping(str(name), normalized)
        unknown = sorted(
            (set(routes.values()) | set(sampling_route_targets.values()))
            - set(targets)
        )
        if unknown:
            raise ControllerError(
                "execution target mapping refers to unknown targets: "
                + ", ".join(unknown)
            )
        return targets, routes, sampling_route_targets

    def _save(self) -> None:
        self.state["heartbeat_at"] = _now()
        atomic_write_json(self.workspace.controller_file, self.state)

    def _ledger(self) -> dict[str, Any]:
        return _read_json(
            self.workspace.ledger,
            {"version": 1, "workflow_id": self.workflow_id, "generations": {}},
            role="scientific ledger",
        )

    def _stage_dir(self, generation: int, stage: str) -> Path:
        if stage != "explore":
            return self.workspace.stage_dir(generation, stage)
        record = self._ledger().get("generations", {}).get(str(generation))
        sequence = (
            generation_stage_sequence(record)
            if isinstance(record, Mapping)
            else None
        )
        return self.workspace.stage_dir(
            generation,
            stage,
            stage_sequence=sequence,
        )

    def _next(self) -> tuple[GenerationPlan, str, Any] | None:
        ledger = self._ledger()
        sampling_budget = int(
            self.manifest.get("sampling_generation_budget", len(self.plans))
        )
        for plan in self.plans:
            record = ledger.get("generations", {}).get(str(plan.generation), {})
            if plan.generation > sampling_budget and not record:
                previous = ledger.get("generations", {}).get(
                    str(plan.generation - 1), {}
                )
                if generation_disposition(previous) != "finalize":
                    break
            if record.get("complete"):
                if record.get("accepted") is False:
                    self.state["state"] = "rejected"
                    self.state["reason"] = f"generation {plan.generation} failed evaluation"
                    self._save()
                    return None
                validation_stage = stage_for_role(record, "validate")
                evaluate = (
                    record.get("stages", {})
                    .get(validation_stage or "evaluate", {})
                    .get("metrics", {})
                )
                if evaluate.get("workflow_converged") is True:
                    self.state["state"] = "complete"
                    self.state["reason"] = (
                        f"workflow converged after model generation {plan.generation}"
                    )
                    self.state["current"] = None
                    self._save()
                    return None
                if evaluate.get("workflow_stalled") is True:
                    self.state["state"] = "stalled"
                    self.state["reason"] = (
                        "validation is still outside target and repeated "
                        "production probes found no useful model update"
                    )
                    self.state["current"] = None
                    self._save()
                    return None
                continue
            stage, context = self.generation_controller.stage_context(plan)
            return plan, stage, context
        self.state["state"] = "budget_exhausted"
        self.state["reason"] = (
            "maximum model generations reached before the production trust envelope "
            "and validation targets both converged"
        )
        self.state["current"] = None
        self._save()
        return None

    def _attempt(self, generation: int, stage: str) -> int:
        count = sum(
            int(item.get("generation", -1)) == generation
            and item.get("stage") == stage
            for item in self.state.get("history", [])
        )
        current = self.state.get("current")
        if current and int(current.get("generation", -1)) == generation and current.get("stage") == stage:
            return int(current.get("attempt", count + 1))
        return count + 1

    def _install_result(
        self,
        *,
        plan: GenerationPlan,
        stage: str,
        attempt: int,
        bundle: Path,
    ) -> Any:
        try:
            _, outcome = load_stage_result(bundle)
        except ExecutionError as error:
            raise ControllerError(f"completed stage returned an invalid result: {error}") from error
        return self._install_outcome(
            plan=plan,
            stage=stage,
            attempt=attempt,
            outcome=outcome,
        )

    def _retry_selection_with_more_memory(
        self,
        current: dict[str, Any],
        target: ExecutionTarget,
    ) -> None:
        generation = int(current["generation"])
        plan = next(item for item in self.plans if item.generation == generation)
        _, context = self.generation_controller.stage_context(plan, "select")
        new_attempt = int(current.get("attempt", 1)) + 1
        memory_tier = int(current.get("memory_tier", 0)) + 1
        archived = dict(current)
        archived["failed_at"] = _now()
        archived["automatic_retry"] = "oom_memory_ladder"
        self.state.setdefault("history", []).append(archived)
        task = build_stage_task(
            self.workspace.tasks_dir,
            workflow_root=self.workspace.root,
            workflow_id=self.workflow_id,
            workflow_instance_id=self.manifest["instance_id"],
            generation=generation,
            stage="select",
            attempt=new_attempt,
            target=target,
            plan=plan,
            config=self.config,
            initial_training=self.initial_training,
            context=context,
            stage_input={"memory_tier": memory_tier},
        )
        self.state["current"] = {
            "task_id": task.task_id,
            "generation": generation,
            "stage": "select",
            "resource": str(current["resource"]),
            "target": target.name,
            "attempt": new_attempt,
            "memory_tier": memory_tier,
            "bundle": str(task.bundle),
            "handle": None,
            "created_at": _now(),
        }
        self.state["state"] = "launching"
        self.state.pop("reason", None)
        self._save()

    def _install_outcome(
        self,
        *,
        plan: GenerationPlan,
        stage: str,
        attempt: int,
        outcome: StageOutcome,
    ) -> Any:
        root = self._stage_dir(plan.generation, stage)
        installed = {}
        for name, source in outcome.artifacts.items():
            destination = root / source.name
            existing_name = next(
                (
                    artifact_name
                    for artifact_name, artifact_path in installed.items()
                    if artifact_path == destination
                ),
                None,
            )
            if existing_name is not None:
                if file_sha256(source) != file_sha256(destination):
                    raise ControllerError(
                        f"stage artifacts {existing_name} and {name} both "
                        f"publish as {destination.name}"
                    )
                installed[name] = destination
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                shutil.copy2(source, temporary)
                temporary.replace(destination)
            installed[name] = destination
        return self.generation_controller.commit_stage(
            plan,
            stage,
            StageOutcome(artifacts=installed, metrics=outcome.metrics),
        )

    def _publish_calculation_link(
        self,
        *,
        generation: int,
        stage: str,
        bundle: Path,
        grouped: bool = False,
        link_name_override: str | None = None,
    ) -> Path | None:
        work = bundle / "output"
        if not work.is_dir():
            work = bundle / "work"
        if not work.is_dir():
            return None
        target = work
        link_name = target.name
        preferred = {
            "train": "training",
            "retrain": "retraining",
            "label": "calculation",
        }.get(stage)
        if preferred is not None:
            candidates = sorted(
                path
                for path in work.iterdir()
                if path.is_dir() and path.name.startswith(preferred)
            )
            if len(candidates) == 1:
                target = candidates[0]
                link_name = target.name
        elif stage == "explore":
            candidates = sorted(
                path for path in (work / "md").glob("*") if path.is_dir()
            )
            if len(candidates) == 1:
                target = candidates[0]
            attempts = work / "md-attempts.json"
            if attempts.is_file():
                values = _read_json(attempts, {}).get("attempts", [])
                if len(values) == 1 and values[0].get("source_id"):
                    link_name = str(values[0]["source_id"])
            if link_name == work.name:
                link_name = bundle.name

        stage_root = self._stage_dir(generation, stage)
        if link_name_override is not None:
            if not link_name_override or Path(link_name_override).name != link_name_override:
                raise ControllerError("calculation link name must be a simple name")
            link_name = link_name_override
        if grouped and stage == "label":
            link = stage_root / link_name
        elif grouped:
            link = stage_root / "calculations" / link_name
        else:
            link = stage_root / "calculation"
        link.parent.mkdir(parents=True, exist_ok=True)
        relative_target = Path(os.path.relpath(target, start=link.parent))
        if link.is_symlink():
            if Path(os.readlink(link)) == relative_target:
                return link
            link.unlink()
        elif link.exists():
            raise ControllerError(
                f"cannot publish calculation link over existing path: {link}"
            )
        temporary = link.with_name(link.name + ".tmp")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(relative_target, target_is_directory=True)
        temporary.replace(link)
        return link

    @staticmethod
    def _unknown_became_lost(
        record: dict[str, Any],
        state: str,
    ) -> bool:
        if state != "unknown":
            record.pop("unknown_since", None)
            record.pop("unknown_observations", None)
            return False
        since = record.get("unknown_since")
        if since is None:
            since = time.time()
            record["unknown_since"] = since
            record["unknown_observations"] = 1
        else:
            record["unknown_observations"] = int(
                record.get("unknown_observations", 0)
            ) + 1
        return (
            time.time() - float(since)
            >= _EXECUTION_UNKNOWN_GRACE_SECONDS
        )

    @staticmethod
    def _mark_group_task_failure(
        item: dict[str, Any],
        *,
        stage: str,
        detail: str,
        failure_kind: str,
    ) -> None:
        item["terminal_failure"] = True
        item["failure_kind"] = failure_kind
        item["retryable"] = stage != "label"
        item["failure"] = detail

    def _tick_task_group(
        self,
        current: dict[str, Any],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> ControllerTick:
        from .execution import StageTask

        stage = str(current["stage"])
        maximum = int(current.get("max_concurrent", len(current["tasks"])))
        inflight = sum(
            item.get("handle") is not None
            and not item.get("collected_bundle")
            and not item.get("terminal_failure")
            for item in current["tasks"]
        )
        group_failed = any(
            item.get("terminal_failure") and item.get("retryable", True)
            for item in current["tasks"]
        )
        running = 0
        unknown = 0
        submission_deferred = False

        observed_items = [
            item
            for item in current["tasks"]
            if item.get("handle") is not None
            and not item.get("collected_bundle")
            and not item.get("terminal_failure")
        ]

        def observe(item):
            target = self.targets[str(item["target"])]
            executor = self.executor_factory(target)
            handle = ExecutionHandle.from_mapping(item["handle"])
            status = executor.inspect(handle)
            return item, executor, handle, status

        def apply_observation(observation):
            nonlocal group_failed, running, unknown
            (
                item,
                executor,
                handle,
                status,
                collected,
                collection_error,
            ) = observation
            item["observed_state"] = status.state
            item["observed_at"] = _now()
            item["detail"] = status.detail
            lost = self._unknown_became_lost(item, status.state)
            if status.state in {"failed", "cancelled"} or lost:
                failure_kind = status.failure_kind
                if failure_kind is None:
                    failure_kind = (
                        "cancelled"
                        if status.state == "cancelled"
                        else "execution_failure"
                    )
                self._mark_group_task_failure(
                    item,
                    stage=stage,
                    detail=(
                        "execution remained unknown beyond the recovery grace "
                        f"period: {status.detail}"
                        if lost
                        else status.detail
                    )
                    or f"{stage} task {item['task_id']} failed",
                    failure_kind=failure_kind,
                )
                if stage != "label":
                    group_failed = True
                else:
                    collect_failure = getattr(executor, "collect_failure", None)
                    if collect_failure is not None:
                        try:
                            item["failure_bundle"] = str(
                                collect_failure(handle)
                            )
                        except ExecutionError as error:
                            item["failure_collection_error"] = str(error)
                self._save()
                return
            if status.state in {"running", "unknown"}:
                if status.state == "unknown":
                    unknown += 1
                else:
                    running += 1
                return
            if collection_error is not None:
                self._mark_group_task_failure(
                    item,
                    stage=stage,
                    detail=str(collection_error),
                    failure_kind="result_validation_failure",
                )
                if stage != "label":
                    group_failed = True
                elif hasattr(executor, "collect_failure"):
                    try:
                        item["failure_bundle"] = str(
                            executor.collect_failure(handle)
                        )
                    except ExecutionError as collection_error:
                        item["failure_collection_error"] = str(
                            collection_error
                        )
                self._save()
                return
            if collected is None:
                raise ControllerError(
                    f"{stage} task {item['task_id']} completed without a result"
                )
            item["collected_bundle"] = str(collected)
            self._publish_calculation_link(
                generation=int(current["generation"]),
                stage=str(current["stage"]),
                bundle=Path(collected),
                grouped=True,
                link_name_override=item.get("display_name"),
            )
            item["completed_at"] = _now()
            self._save()

        if observed_items:
            observations = []
            observed_by_target: dict[str, list[dict[str, Any]]] = {}
            for item in observed_items:
                observed_by_target.setdefault(str(item["target"]), []).append(
                    item
                )

            def observe_target(target_items):
                target = self.targets[str(target_items[0]["target"])]
                executor = self.executor_factory(target)
                inspect_many = getattr(executor, "inspect_many", None)
                if callable(inspect_many):
                    handles = [
                        ExecutionHandle.from_mapping(item["handle"])
                        for item in target_items
                    ]
                    statuses = inspect_many(handles)
                    if len(statuses) != len(target_items):
                        raise ControllerError(
                            "batch executor returned the wrong number of "
                            "inspection results"
                        )
                    return [
                        (item, executor, handle, status)
                        for item, handle, status in zip(
                            target_items,
                            handles,
                            statuses,
                            strict=True,
                        )
                    ]
                with ThreadPoolExecutor(
                    max_workers=min(
                        _MAX_PARALLEL_OBSERVATIONS,
                        len(target_items),
                    )
                ) as pool:
                    return list(pool.map(observe, target_items))

            target_groups = list(observed_by_target.values())
            with ThreadPoolExecutor(max_workers=len(target_groups)) as pool:
                for target_observations in pool.map(
                    observe_target,
                    target_groups,
                ):
                    observations.extend(target_observations)

            completed = [
                observation
                for observation in observations
                if observation[3].state
                not in {
                    "failed",
                    "cancelled",
                    "running",
                    "unknown",
                }
            ]
            collections: dict[
                int,
                tuple[Path | None, PermanentExecutionError | None],
            ] = {}
            grouped: dict[str, list[tuple[Any, ...]]] = {}
            for observation in completed:
                grouped.setdefault(
                    str(observation[0]["target"]),
                    [],
                ).append(observation)
            for target_observations in grouped.values():
                executor = target_observations[0][1]
                collect_many = getattr(executor, "collect_many", None)
                if callable(collect_many):
                    outputs = collect_many(
                        [
                            observation[2]
                            for observation in target_observations
                        ]
                    )
                    if len(outputs) != len(target_observations):
                        raise ControllerError(
                            "batch executor returned the wrong number of "
                            "collection results"
                        )
                    for observation, output in zip(
                        target_observations,
                        outputs,
                        strict=True,
                    ):
                        if isinstance(output, Path):
                            collections[id(observation[0])] = (
                                output,
                                None,
                            )
                        elif isinstance(
                            output,
                            PermanentExecutionError,
                        ):
                            collections[id(observation[0])] = (
                                None,
                                output,
                            )
                        else:
                            raise ControllerError(
                                "batch executor returned an invalid "
                                "collection result"
                            )
                    continue

                def collect_one(observation):
                    try:
                        return (
                            observation,
                            observation[1].collect(observation[2]),
                            None,
                        )
                    except PermanentExecutionError as error:
                        return observation, None, error

                with ThreadPoolExecutor(
                    max_workers=min(
                        _MAX_PARALLEL_OBSERVATIONS,
                        len(target_observations),
                    )
                ) as pool:
                    for observation, collected, error in pool.map(
                        collect_one,
                        target_observations,
                    ):
                        collections[id(observation[0])] = (
                            collected,
                            error,
                        )

            for item, executor, handle, status in observations:
                collected, collection_error = collections.get(
                    id(item),
                    (None, None),
                )
                apply_observation(
                    (
                        item,
                        executor,
                        handle,
                        status,
                        collected,
                        collection_error,
                    )
                )

        launch_candidates = [
            item
            for item in current["tasks"]
            if item.get("handle") is None
            and not item.get("terminal_failure")
        ][: max(0, maximum - inflight)]

        def launch(item):
            target = self.targets[str(item["target"])]
            executor = self.executor_factory(target)
            task = StageTask(
                str(item["task_id"]),
                self.workflow_id,
                int(current["generation"]),
                str(current["stage"]),
                str(item["target"]),
                Path(item["bundle"]),
            )
            try:
                return item, executor.launch(task), None
            except SubmissionDeferred as error:
                return item, None, error
            except PermanentExecutionError as error:
                return item, None, error
            except Exception as error:
                return item, None, error

        def record_launch_results(results):
            nonlocal inflight, running, submission_deferred, group_failed
            transport_error: Exception | None = None
            for item, handle, error in results:
                if handle is None and error is None:
                    continue
                if error is None:
                    item["handle"] = asdict(handle)
                    item["submitted_at"] = _now()
                    self.state.pop("last_submission_error", None)
                    self.state.pop("reason", None)
                    inflight += 1
                    running += 1
                elif isinstance(error, SubmissionDeferred):
                    submission_deferred = True
                    self.state["state"] = "waiting"
                    self.state["reason"] = (
                        "Slurm submission capacity is temporarily exhausted; "
                        "the same task will be submitted again"
                    )
                    self.state["last_submission_error"] = str(error)
                elif isinstance(error, PermanentExecutionError):
                    self._mark_group_task_failure(
                        item,
                        stage=stage,
                        detail=str(error),
                        failure_kind="submission_failure",
                    )
                    item["retryable"] = True
                    group_failed = True
                elif transport_error is None:
                    transport_error = error
            self._save()
            if transport_error is not None:
                raise transport_error

        used_bulk_launch = False
        if (
            launch_candidates
            and not group_failed
            and not submission_deferred
            and not (should_stop is not None and should_stop())
            and len(
                {str(item["target"]) for item in launch_candidates}
            )
            == 1
        ):
            target = self.targets[str(launch_candidates[0]["target"])]
            executor = self.executor_factory(target)
            launch_many = getattr(executor, "launch_many", None)
            if callable(launch_many):
                tasks = [
                    StageTask(
                        str(item["task_id"]),
                        self.workflow_id,
                        int(current["generation"]),
                        str(current["stage"]),
                        str(item["target"]),
                        Path(item["bundle"]),
                    )
                    for item in launch_candidates
                ]
                outputs = launch_many(tasks)
                if len(outputs) != len(tasks):
                    raise ControllerError(
                        "batch executor returned the wrong number of launch results"
                    )
                results = []
                for item, output in zip(
                    launch_candidates,
                    outputs,
                    strict=True,
                ):
                    if isinstance(output, ExecutionHandle):
                        results.append((item, output, None))
                    elif isinstance(output, Exception):
                        results.append((item, None, output))
                    elif output is None:
                        results.append((item, None, None))
                    else:
                        raise ControllerError(
                            "batch executor returned an invalid launch result"
                        )
                record_launch_results(results)
                used_bulk_launch = True

        for start in range(
            0,
            0 if used_bulk_launch else len(launch_candidates),
            _MAX_PARALLEL_LAUNCHES,
        ):
            if (
                group_failed
                or submission_deferred
                or (should_stop is not None and should_stop())
            ):
                break
            batch = launch_candidates[
                start : start + _MAX_PARALLEL_LAUNCHES
            ]
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                results = list(pool.map(launch, batch))
            record_launch_results(results)

        failures = [
            item
            for item in current["tasks"]
            if item.get("terminal_failure")
            and item.get("retryable", True)
        ]
        skipped_failures = [
            item
            for item in current["tasks"]
            if item.get("terminal_failure")
            and not item.get("retryable", True)
        ]
        active = any(
            item.get("handle") is not None
            and not item.get("collected_bundle")
            and not item.get("terminal_failure")
            for item in current["tasks"]
        )
        if failures and not active:
            self.state["state"] = "failed"
            self.state["reason"] = (
                f"{len(failures)} of {len(current['tasks'])} {stage} tasks failed; "
                "retry will preserve completed task results"
            )
            self._save()
            return ControllerTick(
                "failed",
                int(current["generation"]),
                stage,
                detail=self.state["reason"],
            )
        if any(
            not item.get("collected_bundle")
            and not item.get("terminal_failure")
            for item in current["tasks"]
        ):
            self.state["state"] = (
                "waiting"
                if submission_deferred
                else ("degraded" if unknown else "running")
            )
            if not submission_deferred:
                self.state.pop("last_submission_error", None)
                self.state.pop("reason", None)
            self._save()
            return ControllerTick(
                str(self.state["state"]),
                int(current["generation"]),
                str(current["stage"]),
                detail=(
                    f"{sum(bool(item.get('collected_bundle')) for item in current['tasks'])}/"
                    f"{len(current['tasks'])} {stage} tasks complete; "
                    f"{running} running"
                ),
            )

        plan = next(
            item
            for item in self.plans
            if item.generation == int(current["generation"])
        )
        _, context = self.generation_controller.stage_context(plan, stage)
        collected_items = [
            item for item in current["tasks"] if item.get("collected_bundle")
        ]
        outcomes: list[StageOutcome] = []
        successful_items = []
        invalid_results = []
        for item in collected_items:
            try:
                _, outcome = load_stage_result(item["collected_bundle"])
            except ExecutionError as error:
                invalid_bundle = item.pop("collected_bundle")
                item.pop("completed_at", None)
                item["invalid_bundle"] = invalid_bundle
                self._mark_group_task_failure(
                    item,
                    stage=stage,
                    detail=str(error),
                    failure_kind="result_validation_failure",
                )
                invalid_results.append(item)
                continue
            outcomes.append(outcome)
            successful_items.append(item)
        if invalid_results:
            self._save()
            if stage != "label":
                self.state["state"] = "failed"
                self.state["reason"] = (
                    f"{len(invalid_results)} collected {stage} task results "
                    "failed validation"
                )
                self._save()
                return ControllerTick(
                    "failed",
                    int(current["generation"]),
                    stage,
                    detail=self.state["reason"],
                )
            skipped_failures = [
                item
                for item in current["tasks"]
                if item.get("terminal_failure")
                and not item.get("retryable", True)
            ]
        if not successful_items:
            self.state["state"] = "stalled"
            self.state["reason"] = (
                f"all {len(skipped_failures)} label tasks failed, were "
                "cancelled, or returned invalid labels; failure evidence "
                "was preserved"
            )
            self._save()
            return ControllerTick(
                "stalled",
                int(current["generation"]),
                stage,
                detail=self.state["reason"],
            )
        from .workflow_iteration import (
            WorkflowIterationAdapter,
            WorkflowIterationError,
        )

        adapter = WorkflowIterationAdapter(
            self.config,
            initial_training=self.initial_training,
            base_dir=self.workspace.root,
        )
        try:
            if stage == "explore":
                merged = adapter.merge_explore_outcomes(context, outcomes)
            elif stage == "label":
                merged = adapter.merge_label_outcomes(
                    context,
                    outcomes,
                    successful_frame_indices=[
                        int(index)
                        for item in successful_items
                        for index in item["frame_indices"]
                    ],
                    failures=[
                        {
                            "task_id": str(item["task_id"]),
                            "batch_index": int(item["batch_index"]),
                            "frame_indices": list(item["frame_indices"]),
                            "failure_kind": str(
                                item.get("failure_kind", "unknown")
                            ),
                            "detail": str(item.get("failure", "")),
                            "failure_bundle": item.get("failure_bundle")
                            or item.get("invalid_bundle"),
                            "failure_collection_error": item.get(
                                "failure_collection_error"
                            ),
                        }
                        for item in skipped_failures
                    ],
                )
            else:
                raise ControllerError(f"unsupported grouped stage: {stage}")
        except WorkflowIterationError as error:
            current["merge_failure"] = str(error)
            if stage == "explore" and str(error).startswith(
                "MD exploration produced no safe candidate frames"
            ):
                self.state["state"] = "stalled"
                self.state["reason"] = str(error)
            elif stage == "label":
                for item in successful_items:
                    item["invalid_bundle"] = item.pop("collected_bundle")
                    item.pop("completed_at", None)
                    self._mark_group_task_failure(
                        item,
                        stage=stage,
                        detail=str(error),
                        failure_kind="label_validation_failure",
                    )
                self.state["state"] = "stalled"
                self.state["reason"] = (
                    f"label batch results could not be validated: {error}; "
                    "failure evidence was preserved"
                )
            else:
                self.state["state"] = "failed"
                self.state["reason"] = (
                    f"{stage} results could not be merged: {error}"
                )
            self._save()
            return ControllerTick(
                str(self.state["state"]),
                int(current["generation"]),
                stage,
                detail=self.state["reason"],
            )
        summary = self._install_outcome(
            plan=plan,
            stage=stage,
            attempt=int(current["attempt"]),
            outcome=merged,
        )
        archived = dict(current)
        archived["completed_at"] = _now()
        archived["metrics"] = dict(summary.metrics)
        self.state.setdefault("history", []).append(archived)
        self.state["current"] = None
        self.state["state"] = "idle"
        self.state.pop("reason", None)
        self._save()
        return ControllerTick(
            "idle",
            plan.generation,
            stage,
            detail=(
                f"merged {len(outcomes)} {stage} tasks; "
                f"{len(skipped_failures)} failed tasks skipped"
            ),
        )

    def tick(
        self,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> ControllerTick:
        self.state = _read_json(self.workspace.controller_file, self.state)
        current = self.state.get("current")
        if current is not None:
            if current.get("kind") == "task_group":
                result = self._tick_task_group(
                    current,
                    should_stop=should_stop,
                )
                if self.state.get("current") is not None:
                    return result
                current = None
        if current is not None:
            target = self.targets[str(current["target"])]
            executor = self.executor_factory(target)
            bundle = Path(current["bundle"])
            handle_value = current.get("handle")
            if handle_value is None:
                from .execution import StageTask

                task = StageTask(
                    str(current["task_id"]),
                    self.workflow_id,
                    int(current["generation"]),
                    str(current["stage"]),
                    str(current["target"]),
                    bundle,
                )
                try:
                    handle = executor.launch(task)
                except SubmissionDeferred as error:
                    self.state["state"] = "waiting"
                    self.state["reason"] = (
                        "Slurm submission capacity is temporarily exhausted; "
                        "the same task will be submitted again"
                    )
                    self.state["last_submission_error"] = str(error)
                    self._save()
                    return ControllerTick(
                        "waiting",
                        task.generation,
                        task.stage,
                        task.target,
                        detail=self.state["reason"],
                    )
                except PermanentExecutionError as error:
                    current["terminal_failure"] = True
                    current["failure_kind"] = "submission_failure"
                    current["failure"] = str(error)
                    self.state["state"] = "failed"
                    self.state["reason"] = str(error)
                    self._save()
                    return ControllerTick(
                        "failed",
                        task.generation,
                        task.stage,
                        task.target,
                        detail=str(error),
                    )
                current["handle"] = asdict(handle)
                current["submitted_at"] = _now()
                self.state["state"] = "running"
                self.state.pop("last_transport_error", None)
                self.state.pop("last_submission_error", None)
                self.state.pop("reason", None)
                self._save()
                return ControllerTick(
                    "running",
                    task.generation,
                    task.stage,
                    task.target,
                    handle.execution_id,
                    "submitted",
                )
            handle = ExecutionHandle.from_mapping(handle_value)
            status = executor.inspect(handle)
            current["observed_state"] = status.state
            current["observed_at"] = _now()
            current["detail"] = status.detail
            lost = self._unknown_became_lost(current, status.state)
            if lost:
                status = type(status)(
                    "failed",
                    "execution remained unknown beyond the recovery grace "
                    f"period: {status.detail}",
                )
            if status.state in {"running", "unknown"}:
                self.state["state"] = (
                    "degraded" if status.state == "unknown" else "running"
                )
                if status.state == "unknown":
                    self.state["last_transport_error"] = (
                        status.detail or "execution state is temporarily unknown"
                    )
                self._save()
                return ControllerTick(
                    str(self.state["state"]),
                    int(current["generation"]),
                    str(current["stage"]),
                    str(current["target"]),
                    handle.execution_id,
                    status.detail,
                )
            if status.state in {"failed", "cancelled"}:
                failure_kind = status.failure_kind or (
                    "cancelled" if status.state == "cancelled" else "unknown"
                )
                current["failure_kind"] = failure_kind
                current["failure"] = status.detail or "stage execution failed"
                if (
                    str(current["stage"]) == "select"
                    and failure_kind == "out_of_memory"
                    and int(current.get("memory_tier", 0)) + 1
                    < len(target.memory_ladder)
                ):
                    self.state["reason"] = (
                        f"selection exceeded memory tier "
                        f"{target.memory_ladder[int(current.get('memory_tier', 0))]}; "
                        "retrying with the next configured tier"
                    )
                    self._retry_selection_with_more_memory(current, target)
                    return self.tick(should_stop=should_stop)
                self.state["state"] = "failed"
                self.state["reason"] = current["failure"]
                self._save()
                return ControllerTick(
                    "failed",
                    int(current["generation"]),
                    str(current["stage"]),
                    str(current["target"]),
                    handle.execution_id,
                    self.state["reason"],
                )
            try:
                collected = executor.collect(handle)
            except PermanentExecutionError as error:
                current["terminal_failure"] = True
                current["failure_kind"] = "result_validation_failure"
                current["failure"] = str(error)
                self.state["state"] = "failed"
                self.state["reason"] = str(error)
                self._save()
                return ControllerTick(
                    "failed",
                    int(current["generation"]),
                    str(current["stage"]),
                    str(current["target"]),
                    handle.execution_id,
                    str(error),
                )
            plan = next(
                item for item in self.plans if item.generation == int(current["generation"])
            )
            self._publish_calculation_link(
                generation=plan.generation,
                stage=str(current["stage"]),
                bundle=Path(collected),
            )
            summary = self._install_result(
                plan=plan,
                stage=str(current["stage"]),
                attempt=int(current["attempt"]),
                bundle=collected,
            )
            archived = dict(current)
            archived["completed_at"] = _now()
            archived["metrics"] = dict(summary.metrics)
            archived["handle"] = dict(handle_value)
            self.state.setdefault("history", []).append(archived)
            self.state["current"] = None
            self.state["state"] = "idle"
            self.state.pop("reason", None)
            self._save()
            if summary.generation_complete and summary.accepted is False:
                self.state["state"] = "rejected"
                self.state["reason"] = f"generation {plan.generation} failed evaluation"
                self._save()
                return ControllerTick("rejected", plan.generation, summary.stage)

        next_value = self._next()
        if next_value is None:
            return ControllerTick(
                str(self.state["state"]), detail=str(self.state.get("reason", ""))
            )
        plan, stage, context = next_value
        if stage == "explore":
            from .workflow_iteration import WorkflowIterationAdapter

            adapter = WorkflowIterationAdapter(
                self.config,
                initial_training=self.initial_training,
                base_dir=self.workspace.root,
            )
            attempt_specs = adapter.plan_explore_attempts(context)
            if not attempt_specs:
                self.state["state"] = "coverage_exhausted"
                self.state["reason"] = (
                    "sampling coverage is exhausted for the active model, "
                    "but independent validation has not established workflow "
                    "convergence"
                )
                self.state["current"] = None
                self._save()
                return ControllerTick(
                    "coverage_exhausted",
                    plan.generation,
                    stage,
                    detail=self.state["reason"],
                )
            attempt = self._attempt(plan.generation, stage)
            tasks = []
            for spec in attempt_specs:
                target_name = self.sampling_route_targets.get(
                    str(spec["route_id"]),
                    self.stage_targets["sampling"],
                )
                target = self.targets[target_name]
                task = build_stage_task(
                    self.workspace.tasks_dir,
                    workflow_root=self.workspace.root,
                    workflow_id=self.workflow_id,
                    generation=plan.generation,
                    stage=stage,
                    attempt=attempt,
                    target=target,
                    plan=plan,
                    config=self.config,
                    initial_training=self.initial_training,
                    context=context,
                    workflow_instance_id=self.manifest["instance_id"],
                    stage_input={
                        "route_id": spec["route_id"],
                        "attempt_ids": [spec["attempt_id"]],
                        "allow_empty": True,
                    },
                )
                tasks.append(
                    {
                        "task_id": task.task_id,
                        "route_id": spec["route_id"],
                        "route_fingerprint": spec["route_fingerprint"],
                        "scenario_attempt_id": spec["attempt_id"],
                        "temperature": spec["temperature"],
                        "steps": spec["steps"],
                        "target_level": spec["target_level"],
                        "replica": spec["replica"],
                        "target": target_name,
                        "bundle": str(task.bundle),
                        "handle": None,
                    }
                )
            self.state["current"] = {
                "kind": "task_group",
                "generation": plan.generation,
                "stage": stage,
                "resource": "sampling",
                "attempt": attempt,
                "tasks": tasks,
                "created_at": _now(),
            }
            self.state["state"] = "launching"
            self._save()
            return self.tick(should_stop=should_stop)
        if (
            stage == "label"
            and self.config.get("labeling", {}).get("backend", "vasp")
            in {"vasp", "abacus"}
            and context.artifacts["selected_input"].stat().st_size > 0
        ):
            frames = ase_read(context.artifacts["selected_input"], index=":")
            if not isinstance(frames, list):
                frames = [frames]
            structures_per_job = int(
                self.config.get("labeling", {}).get(
                    "structures_per_job",
                    DEFAULT_STRUCTURES_PER_LABEL_JOB,
                )
            )
            maximum = int(
                self.config.get("labeling", {}).get(
                    "max_concurrent",
                    DEFAULT_MAX_CONCURRENT,
                )
            )
            target_name = self.stage_targets["labeling"]
            target = self.targets[target_name]
            attempt = self._attempt(plan.generation, stage)
            tasks = []
            for start in range(0, len(frames), structures_per_job):
                stop = min(start + structures_per_job, len(frames))
                batch_index = start // structures_per_job + 1
                task = build_stage_task(
                    self.workspace.tasks_dir,
                    workflow_root=self.workspace.root,
                    workflow_id=self.workflow_id,
                    generation=plan.generation,
                    stage=stage,
                    attempt=attempt,
                    target=target,
                    plan=plan,
                    config=self.config,
                    initial_training=self.initial_training,
                    context=context,
                    workflow_instance_id=self.manifest["instance_id"],
                    stage_input={
                        "batch_index": batch_index,
                        "frame_indices": list(range(start, stop)),
                    },
                )
                display_name = (
                    f"{start + 1:06d}-{frames[start].get_chemical_formula()}"
                    if stop - start == 1
                    else f"{start + 1:06d}-{stop:06d}"
                )
                tasks.append(
                    {
                        "task_id": task.task_id,
                        "batch_index": batch_index,
                        "frame_indices": list(range(start, stop)),
                        "display_name": display_name,
                        "target": target_name,
                        "bundle": str(task.bundle),
                        "handle": None,
                    }
                )
            self.state["current"] = {
                "kind": "task_group",
                "generation": plan.generation,
                "stage": stage,
                "resource": "labeling",
                "attempt": attempt,
                "max_concurrent": maximum,
                "tasks": tasks,
                "created_at": _now(),
            }
            self.state["state"] = "launching"
            self._save()
            return self.tick(should_stop=should_stop)
        empty_label = bool(
            stage == "label"
            and context.artifacts["selected_input"].stat().st_size == 0
        )
        resource = (
            "analysis"
            if empty_label
            else _RESOURCE_FOR_STAGE[stage]
        )
        target_name = self.stage_targets[resource]
        target = self.targets[target_name]
        attempt = self._attempt(plan.generation, stage)
        memory_tier = 0
        if stage == "select" and target.memory_ladder:
            memory_tier = max(
                (
                    int(item.get("memory_tier", 0))
                    for item in self.state.get("history", [])
                    if int(item.get("generation", -1)) == plan.generation
                    and item.get("stage") == "select"
                ),
                default=0,
            )
        task = build_stage_task(
            self.workspace.tasks_dir,
            workflow_root=self.workspace.root,
            workflow_id=self.workflow_id,
            generation=plan.generation,
            stage=stage,
            attempt=attempt,
            target=target,
            plan=plan,
            config=self.config,
            initial_training=self.initial_training,
            context=context,
            workflow_instance_id=self.manifest["instance_id"],
            stage_input=(
                {"empty_selection": True}
                if empty_label
                else ({"memory_tier": memory_tier} if stage == "select" and target.memory_ladder else None)
            ),
        )
        self.state["current"] = {
            "task_id": task.task_id,
            "generation": plan.generation,
            "stage": stage,
            "resource": resource,
            "target": target_name,
            "attempt": attempt,
            **(
                {"memory_tier": memory_tier}
                if stage == "select" and target.memory_ladder
                else {}
            ),
            "bundle": str(task.bundle),
            "handle": None,
            "created_at": _now(),
        }
        self.state["state"] = "launching"
        self._save()
        # Launch in the same tick; the persisted intent makes this retry-safe.
        return self.tick(should_stop=should_stop)

    def retry(self, *, recover_rejected: bool = False) -> None:
        state = str(self.state.get("state", "idle"))
        if state == "complete":
            raise ControllerError("workflow is already complete")
        if state == "failed" and self.state.pop(
            "preserve_current_on_retry", False
        ):
            self.state["state"] = "launching"
            self.state.pop("reason", None)
            self._save()
            return
        if state == "rejected":
            if not recover_rejected:
                raise ControllerError("workflow is rejected; explicit recovery is required")
            ledger = self._ledger()
            rejected = [
                plan
                for plan in self.plans
                if ledger.get("generations", {})
                .get(str(plan.generation), {})
                .get("accepted")
                is False
            ]
            if not rejected:
                raise ControllerError("controller says rejected but ledger has no rejected generation")
            self.generation_controller.reopen_rejected(rejected[0])
        current = self.state.get("current")
        if current is not None:
            if current.get("kind") == "task_group":
                plan = next(
                    plan
                    for plan in self.plans
                    if plan.generation == int(current["generation"])
                )
                _, context = self.generation_controller.stage_context(
                    plan,
                    str(current["stage"]),
                )
                new_attempt = int(current.get("attempt", 1)) + 1
                retried = 0
                for item in current["tasks"]:
                    if item.get("collected_bundle") or not item.get(
                        "retryable", True
                    ):
                        continue
                    handle = item.get("handle")
                    item.setdefault("retry_history", []).append(
                        {
                            "task_id": item["task_id"],
                            "bundle": item["bundle"],
                            "handle": handle,
                            "failed_at": _now(),
                            "failure": item.get("detail")
                            or self.state.get("reason", "manual retry"),
                            "failure_kind": item.get("failure_kind"),
                            "failure_bundle": item.get("failure_bundle"),
                            "invalid_bundle": item.get("invalid_bundle"),
                            "failure_collection_error": item.get(
                                "failure_collection_error"
                            ),
                        }
                    )
                    target_name = str(item["target"])
                    target = self.targets[target_name]
                    if current["stage"] == "explore":
                        stage_input = {
                            "route_id": item["route_id"],
                            "attempt_ids": [
                                item["scenario_attempt_id"]
                            ],
                            "allow_empty": True,
                        }
                    elif current["stage"] == "label":
                        stage_input = {
                            "batch_index": item["batch_index"],
                            "frame_indices": list(item["frame_indices"]),
                        }
                    else:
                        raise ControllerError(
                            f"unsupported grouped retry stage: "
                            f"{current['stage']}"
                        )
                    task = build_stage_task(
                        self.workspace.tasks_dir,
                        workflow_root=self.workspace.root,
                        workflow_id=self.workflow_id,
                        workflow_instance_id=self.manifest["instance_id"],
                        generation=plan.generation,
                        stage=str(current["stage"]),
                        attempt=new_attempt,
                        target=target,
                        plan=plan,
                        config=self.config,
                        initial_training=self.initial_training,
                        context=context,
                        stage_input=stage_input,
                    )
                    item["task_id"] = task.task_id
                    item["bundle"] = str(task.bundle)
                    item["handle"] = None
                    item.pop("terminal_failure", None)
                    item.pop("failure", None)
                    item.pop("observed_state", None)
                    item.pop("observed_at", None)
                    item.pop("detail", None)
                    item.pop("failure_kind", None)
                    item.pop("retryable", None)
                    item.pop("failure_bundle", None)
                    item.pop("invalid_bundle", None)
                    item.pop("failure_collection_error", None)
                    item.pop("completed_at", None)
                    item.pop("cancellation", None)
                    item.pop("cancellation_requested_at", None)
                    retried += 1
                if not retried:
                    if any(
                        item.get("collected_bundle")
                        for item in current["tasks"]
                    ):
                        current["merge_retry_count"] = int(
                            current.get("merge_retry_count", 0)
                        ) + 1
                        self.state["state"] = "launching"
                        self.state.pop("reason", None)
                        self._save()
                        return
                    raise ControllerError(
                        f"{current['stage']} task group has no failed or "
                        "unfinished tasks to retry"
                    )
                current["attempt"] = new_attempt
                current.pop("cancellations", None)
                self.state["state"] = "launching"
                self.state.pop("reason", None)
                self._save()
                return
            archived = dict(current)
            archived["failed_at"] = _now()
            archived["failure"] = self.state.get("reason", "manual retry")
            self.state.setdefault("history", []).append(archived)
        self.state["current"] = None
        self.state["state"] = "idle"
        self.state.pop("reason", None)
        self._save()

    @staticmethod
    def _current_execution_is_terminal(current: Mapping[str, Any]) -> bool:
        if current.get("kind") == "task_group":
            return all(
                item.get("collected_bundle")
                or item.get("terminal_failure")
                or item.get("observed_state")
                in {"completed", "failed", "cancelled"}
                for item in current.get("tasks", [])
            )
        if current.get("handle") is None:
            return True
        return current.get("observed_state") in {
            "completed",
            "failed",
            "cancelled",
        }

    def plan_restart(
        self,
        *,
        generation: int,
        from_stage: str,
        task_scope: str = "failed",
    ) -> RestartPlan:
        """Validate and describe an explicit restart without changing state."""

        if task_scope not in {"failed", "all"}:
            raise ControllerError("restart task scope must be 'failed' or 'all'")
        ledger = self._ledger()
        records = ledger.get("generations", {})
        entered = sorted(
            int(key)
            for key, value in records.items()
            if isinstance(value, Mapping) and value.get("stages") is not None
        )
        if generation not in entered:
            raise ControllerError(f"generation {generation} has not started")
        if generation != entered[-1]:
            raise ControllerError(
                "restart only supports the latest entered generation; "
                f"generation {entered[-1]} already depends on generation {generation}"
            )
        record = records[str(generation)]
        if record.get("complete") and record.get("accepted") is True:
            raise ControllerError(
                "restart does not rewrite an accepted generation; start or "
                "extend the workflow with a new generation instead"
            )
        try:
            sequence = generation_stage_sequence(record)
        except ValueError as error:
            raise ControllerError(str(error)) from error
        if from_stage not in sequence:
            raise ControllerError(
                f"stage {from_stage!r} is not part of generation {generation}; "
                f"choose one of: {', '.join(sequence)}"
            )
        stage_records = record.get("stages", {})
        completed_count = 0
        while (
            completed_count < len(sequence)
            and sequence[completed_count] in stage_records
        ):
            completed_count += 1
        if any(stage in stage_records for stage in sequence[completed_count + 1 :]):
            raise ControllerError(
                f"generation {generation} has a non-contiguous stage ledger"
            )
        start = sequence.index(from_stage)
        if start > completed_count:
            expected = sequence[completed_count]
            raise ControllerError(
                f"generation {generation} has not reached {from_stage}; "
                f"its next stage is {expected}"
            )

        current = self.state.get("current")
        preserved_tasks = 0
        retried_tasks = 0
        if current is not None:
            current_generation = int(current.get("generation", -1))
            current_stage = str(current.get("stage", ""))
            if current_generation != generation:
                raise ControllerError(
                    "controller execution does not belong to the requested generation"
                )
            if not self._current_execution_is_terminal(current):
                raise ControllerError(
                    "the current execution is not terminal; stop and reconcile its "
                    "scheduler jobs before restarting"
                )
            if (
                task_scope == "failed"
                and current_stage == from_stage
                and current.get("kind") == "task_group"
            ):
                tasks = list(current.get("tasks", []))
                preserved_tasks = sum(
                    bool(item.get("collected_bundle")) for item in tasks
                )
                retried_tasks = len(tasks) - preserved_tasks
                if retried_tasks == 0:
                    raise ControllerError(
                        f"{from_stage} has no failed or unfinished tasks to retry"
                    )

        return RestartPlan(
            generation=generation,
            from_stage=from_stage,
            task_scope=task_scope,
            reused_stages=tuple(sequence[:start]),
            restarted_stages=tuple(sequence[start:]),
            preserved_tasks=preserved_tasks,
            retried_tasks=retried_tasks,
        )

    def restart_from(
        self,
        *,
        generation: int,
        from_stage: str,
        task_scope: str = "failed",
    ) -> RestartPlan:
        """Restart the latest unfinished generation from a chosen stage."""

        plan = self.plan_restart(
            generation=generation,
            from_stage=from_stage,
            task_scope=task_scope,
        )
        generation_plan = next(
            item for item in self.plans if item.generation == generation
        )
        current = self.state.get("current")
        retry_current_group = bool(
            task_scope == "failed"
            and current is not None
            and int(current.get("generation", -1)) == generation
            and current.get("stage") == from_stage
            and current.get("kind") == "task_group"
        )

        try:
            self.generation_controller.reopen_from(
                generation_plan,
                from_stage=from_stage,
            )
        except IterationError as error:
            raise ControllerError(str(error)) from error
        if retry_current_group:
            assert current is not None
            for item in current["tasks"]:
                if not item.get("collected_bundle"):
                    item["retryable"] = True
            self.state["state"] = "failed"
            self.state["reason"] = "explicit stage restart"
            self.retry()
            return plan

        if current is not None:
            archived = dict(current)
            archived["superseded_at"] = _now()
            archived["superseded_by"] = {
                "generation": generation,
                "from_stage": from_stage,
                "task_scope": task_scope,
            }
            self.state.setdefault("history", []).append(archived)
        self.state["current"] = None
        self.state["state"] = "idle"
        self.state.pop("reason", None)
        self.state.pop("preserve_current_on_retry", None)
        self._save()
        return plan

    def resume_stopped(self) -> None:
        """Reconcile stop/cancel state before a controller is restarted."""

        if self.state.get("state") != "stopped":
            return
        current = self.state.get("current")
        if current is None:
            self.state["state"] = "idle"
            self.state.pop("reason", None)
            self._save()
            return
        if current.get("kind") == "task_group":
            stage = str(current["stage"])
            cancellation_seen = False
            for item in current["tasks"]:
                if item.get("collected_bundle"):
                    continue
                cancellation = item.get("cancellation")
                if cancellation is None:
                    continue
                cancellation_seen = True
                handle_value = item.get("handle")
                if handle_value is None:
                    self._mark_group_task_failure(
                        item,
                        stage=stage,
                        detail="task was stopped before launch",
                        failure_kind="cancelled",
                    )
                    continue
                executor = self.executor_factory(
                    self.targets[str(item["target"])]
                )
                handle = ExecutionHandle.from_mapping(handle_value)
                recorded_state = item.get("observed_state")
                if recorded_state in {"completed", "failed", "cancelled"}:
                    status = ExecutionStatus(
                        str(recorded_state),
                        str(item.get("detail") or ""),
                        item.get("failure_kind"),
                    )
                else:
                    status = executor.inspect(handle)
                item["observed_state"] = status.state
                item["observed_at"] = _now()
                item["detail"] = status.detail
                if status.state == "completed":
                    try:
                        collected = executor.collect(handle)
                    except PermanentExecutionError as error:
                        self._mark_group_task_failure(
                            item,
                            stage=stage,
                            detail=str(error),
                            failure_kind="result_validation_failure",
                        )
                        continue
                    item["collected_bundle"] = str(collected)
                    item["completed_at"] = _now()
                    self._publish_calculation_link(
                        generation=int(current["generation"]),
                        stage=str(current["stage"]),
                        bundle=Path(collected),
                        grouped=True,
                        link_name_override=item.get("display_name"),
                    )
                elif status.state in {"failed", "cancelled"}:
                    self._mark_group_task_failure(
                        item,
                        stage=stage,
                        detail=(
                            status.detail
                            or "cancelled execution is terminal"
                        ),
                        failure_kind=status.failure_kind or "cancelled",
                    )
                else:
                    self._save()
                    raise ControllerError(
                        "workflow cancellation is not terminal yet; run resume "
                        "again after the scheduler finishes cancelling current jobs"
                    )
            if cancellation_seen and any(
                item.get("terminal_failure") for item in current["tasks"]
            ):
                if stage == "label":
                    if not any(
                        item.get("collected_bundle")
                        for item in current["tasks"]
                    ):
                        for item in current["tasks"]:
                            if item.get("cancellation") is not None:
                                item["retryable"] = True
                        self.state["state"] = "failed"
                        self.state["reason"] = (
                            "stopped label task group has no collected labels; "
                            "retrying cancelled tasks with a new attempt"
                        )
                        self._save()
                        self.retry()
                        return
                    self.state["state"] = "launching"
                    self.state.pop("reason", None)
                    self._save()
                    return
                self.state["state"] = "failed"
                self.state["reason"] = (
                    "stopped task group is terminal; retrying only unfinished "
                    "tasks with a new attempt"
                )
                self._save()
                self.retry()
                return
            self.state["state"] = "launching"
            self.state.pop("reason", None)
            self._save()
            return

        cancellation = current.get("cancellation")
        if cancellation is None:
            self.state["state"] = "running"
            self.state.pop("reason", None)
            self._save()
            return
        handle_value = current.get("handle")
        if handle_value is None:
            self.state["state"] = "failed"
            self.state["reason"] = "stopped task has no execution handle"
            self._save()
            self.retry()
            return
        executor = self.executor_factory(
            self.targets[str(current["target"])]
        )
        handle = ExecutionHandle.from_mapping(handle_value)
        status = executor.inspect(handle)
        current["observed_state"] = status.state
        current["observed_at"] = _now()
        current["detail"] = status.detail
        if status.state == "completed":
            self.state["state"] = "running"
            self.state.pop("reason", None)
            self._save()
            return
        if status.state in {"failed", "cancelled"}:
            self.state["state"] = "failed"
            self.state["reason"] = (
                status.detail or "cancelled execution is terminal"
            )
            self._save()
            self.retry()
            return
        self._save()
        raise ControllerError(
            "workflow cancellation is not terminal yet; run resume again after "
            "the scheduler finishes cancelling the current job"
        )


def run_controller(project: str | Path, *, poll_interval: float | None = None) -> int:
    """Hold the workflow controller lock and supervise until a terminal state."""

    workspace = WorkflowWorkspace.locate(project)
    interval = poll_interval
    if interval is None:
        config, _ = load_config(workspace.project_file)
        interval = float(config.get("execution", {}).get("poll_interval", 30.0))
    if interval < 0.2:
        raise ControllerError("execution.poll_interval must be at least 0.2 seconds")
    stop_event = threading.Event()

    def request_stop(_signum, _frame) -> None:
        stop_event.set()

    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        for signum in previous_handlers:
            signal.signal(signum, request_stop)
        with workspace.controller_lock.open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ControllerError(
                    "workflow controller is already running"
                ) from error
            workspace.controller_pid.write_text(
                f"{os.getpid()}\n", encoding="utf-8"
            )
            controller = PersistentController(workspace.root)
            controller.state["pid"] = os.getpid()
            controller.state["started_at"] = _now()
            controller.state["state"] = "running"
            controller.state.pop("reason", None)
            controller._save()
            try:
                from .notifications import build_workflow_notifier

                notifier = build_workflow_notifier(
                    controller.config,
                    workspace,
                )
            except Exception as error:
                notifier = None
                print(
                    f"NepTrain notification warning: cannot initialize "
                    f"notifications: {error}",
                    file=sys.stderr,
                    flush=True,
                )

            def report_progress(tick: ControllerTick) -> None:
                if notifier is None:
                    return
                try:
                    notifier.observe(
                        workflow_id=controller.workflow_id,
                        plans=controller.plans,
                        controller_state=controller.state,
                        tick=tick,
                    )
                except Exception as error:
                    print(
                        f"NepTrain notification warning: {error}",
                        file=sys.stderr,
                        flush=True,
                    )

            try:
                while not stop_event.is_set():
                    try:
                        tick = controller.tick(
                            should_stop=stop_event.is_set,
                        )
                        if tick.state != "degraded":
                            recovered = controller.state.pop(
                                "last_transport_error", None
                            )
                            if recovered is not None or controller.state.get(
                                "transport_failures", 0
                            ):
                                controller.state["transport_failures"] = 0
                                controller._save()
                    except ExecutionError as error:
                        failures = (
                            int(controller.state.get("transport_failures", 0))
                            + 1
                        )
                        controller.state["transport_failures"] = failures
                        controller.state["last_transport_error"] = str(error)
                        # A scheduler/SSH timeout says nothing about the
                        # scientific task's state. Keep the immutable handle
                        # and continue observing it until transport recovers
                        # or the scheduler reports a real terminal result.
                        controller.state["state"] = "degraded"
                        controller.state.pop("reason", None)
                        controller._save()
                        tick = ControllerTick(
                            "degraded",
                            detail=str(error),
                        )
                    report_progress(tick)
                    if tick.state in {
                        "complete",
                        "rejected",
                        "failed",
                        "stalled",
                        "budget_exhausted",
                        "coverage_exhausted",
                    }:
                        return 0 if tick.state == "complete" else 2
                    stop_event.wait(interval)
                controller.state["state"] = "stopped"
                controller.state["reason"] = "controller stopped by user"
                controller._save()
                return 0
            except Exception as error:
                # An internal controller failure does not say that a submitted
                # scheduler task failed.  Preserve immutable handles so resume
                # reconciles the existing work instead of submitting duplicates.
                if controller.state.get("current") is not None:
                    controller.state["preserve_current_on_retry"] = True
                controller.state["state"] = "failed"
                controller.state["reason"] = str(error)
                controller._save()
                report_progress(
                    ControllerTick(
                        "failed",
                        detail=str(error),
                    )
                )
                raise
            finally:
                if notifier is not None:
                    try:
                        notifier.close()
                    except Exception as error:
                        print(
                            "NepTrain notification warning: cannot stop "
                            f"notification thread: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                try:
                    if (
                        workspace.controller_pid.read_text(
                            encoding="utf-8"
                        ).strip()
                        == str(os.getpid())
                    ):
                        workspace.controller_pid.unlink()
                except FileNotFoundError:
                    pass
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def start_controller(
    project: str | Path,
    *,
    foreground: bool = False,
    poll_interval: float | None = None,
) -> int:
    workspace = WorkflowWorkspace.locate(project)
    if controller_running(workspace.root):
        raise ControllerError("workflow controller is already running")
    if foreground:
        return run_controller(workspace.root, poll_interval=poll_interval)
    command = [*_controller_command(), "controller", str(workspace.root)]
    if poll_interval is not None:
        command.extend(["--poll-interval", str(poll_interval)])
    workspace.controller_log.parent.mkdir(parents=True, exist_ok=True)
    log = workspace.controller_log.open("ab")
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    python_path = [
        str(Path(item).expanduser().resolve())
        for item in environment.get("PYTHONPATH", "").split(os.pathsep)
        if item
    ]
    if source_root not in python_path:
        python_path.insert(0, source_root)
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    try:
        process = subprocess.Popen(
            command,
            cwd=workspace.root,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
    finally:
        log.close()
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if workspace.controller_pid.is_file() and controller_running(workspace.root):
            try:
                state = _read_json(workspace.controller_file)
            except ControllerError:
                state = {}
            # The controller can finish its first tick before the parent gets
            # scheduled again.  In that case a live, lock-owning controller is
            # already ``launching`` or ``waiting`` rather than ``running``.
            # Matching the recorded pid is the durable readiness signal; an
            # already-terminal state still belongs on the startup error path.
            if (
                int(state.get("pid", -1)) == process.pid
                and state.get("state")
                not in {
                    "failed",
                    "stalled",
                    "rejected",
                    "budget_exhausted",
                    "coverage_exhausted",
                    "complete",
                    "stopped",
                }
            ):
                return process.pid
        if process.poll() is not None:
            detail = workspace.controller_log.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise ControllerError(f"controller failed to start: {detail.strip()}")
        time.sleep(0.05)
    process.terminate()
    raise ControllerError("controller did not acquire its lock within 8 seconds")


def stop_controller(project: str | Path) -> None:
    workspace = WorkflowWorkspace.locate(project)
    if not controller_running(workspace.root):
        raise ControllerError("workflow controller is not running")
    try:
        pid = int(workspace.controller_pid.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError) as error:
        raise ControllerError("running controller has no valid pid record") from error
    if not _process_matches(pid, workspace.root):
        raise ControllerError("refusing to signal a pid that is not this workflow controller")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not controller_running(workspace.root):
            return
        time.sleep(0.05)
    if not _process_matches(pid, workspace.root):
        raise ControllerError(
            "controller did not stop gracefully and its pid identity changed"
        )
    os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not controller_running(workspace.root):
            return
        time.sleep(0.05)
    raise ControllerError("controller did not stop after SIGTERM and SIGKILL")


def _append_execution_event(
    workspace: WorkflowWorkspace,
    event: Mapping[str, Any],
) -> None:
    manifest = _read_json(
        workspace.manifest,
        role="workflow manifest",
    )
    workflow_id = str(manifest.get("workflow_id", ""))
    if not workflow_id:
        raise ControllerError("workflow manifest has no workflow_id")
    with workspace.ledger_lock.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger = _read_json(
            workspace.ledger,
            {
                "version": 1,
                "workflow_id": workflow_id,
                "generations": {},
            },
            role="scientific ledger",
        )
        if ledger.get("workflow_id") != workflow_id:
            raise ControllerError(
                "cannot record stop: scientific ledger belongs to another workflow"
            )
        records = ledger.setdefault("execution_events", [])
        if not isinstance(records, list):
            raise ControllerError(
                "scientific ledger execution_events must be a list"
            )
        records.append(
            {
                "sequence": len(records) + 1,
                "recorded_at": _now(),
                **dict(event),
            }
        )
        atomic_write_json(workspace.ledger, ledger)


def stop_workflow(
    project: str | Path,
    *,
    cancel_jobs: bool = True,
    executor_factory: ExecutorFactory = executor_for,
) -> dict[str, Any]:
    """Stop the controller and cancel its current execution by default.

    ``cancel_jobs=False`` keeps the current execution for a later resume.
    A successfully cancelled task is archived and removed from ``current`` so
    a later resume creates a new, traceable attempt.
    """

    workspace = WorkflowWorkspace.locate(project)
    was_running = controller_running(workspace.root)
    if was_running:
        stop_controller(workspace.root)

    result: dict[str, Any] = {
        "project": str(workspace.root),
        "controller": "stopped" if was_running else "already_stopped",
        "current_execution": None,
    }

    if not was_running:
        prior_state = (
            _read_json(workspace.controller_file, role="controller state")
            if workspace.controller_file.is_file()
            else {}
        )
        prior_current = prior_state.get("current")
        prior_name = str(prior_state.get("state", "prepared"))
        no_op_states = {
            "prepared",
            "idle",
            "stopped",
            "complete",
            "failed",
            "rejected",
            "stalled",
            "budget_exhausted",
            "coverage_exhausted",
        }
        if prior_current is None and prior_name in no_op_states:
            result["current_execution"] = {
                "action": "none",
                "detail": (
                    "workflow has no active execution; its existing state "
                    f"{prior_name!r} was left unchanged"
                ),
            }
            return result
        if not cancel_jobs:
            result["current_execution"] = {
                "action": "preserved",
                "detail": (
                    "controller was already stopped; the recorded current "
                    "execution was left unchanged"
                ),
            }
            return result

    def finish() -> dict[str, Any]:
        _append_execution_event(
            workspace,
            {
                "event": "workflow_stop",
                "cancel_jobs": bool(cancel_jobs),
                "controller": result["controller"],
                "current_execution": result["current_execution"],
            },
        )
        return result

    if not cancel_jobs:
        return finish()

    with workspace.controller_lock.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:  # pragma: no cover - race protection
            raise ControllerError(
                "workflow controller restarted before jobs could be cancelled"
            ) from error
        controller = PersistentController(
            workspace.root, executor_factory=executor_factory
        )
        current = controller.state.get("current")
        if current is None:
            controller.state["state"] = "stopped"
            controller.state["reason"] = (
                "controller stopped by user; no current execution to cancel"
            )
            controller._save()
            result["current_execution"] = {"action": "none"}
            return finish()
        if current.get("kind") == "task_group":
            records_by_task: dict[str, dict[str, Any]] = {}
            cancellable_by_target: dict[
                str,
                list[tuple[dict[str, Any], ExecutionHandle]],
            ] = {}
            for item in current["tasks"]:
                handle_value = item.get("handle")
                if item.get("collected_bundle"):
                    records_by_task[str(item["task_id"])] = {
                        "target": item["target"],
                        "execution_id": (
                            (handle_value or {}).get("execution_id")
                        ),
                        "action": "completed",
                        "detail": "result was already collected and preserved",
                    }
                    continue
                if handle_value is None:
                    record = {
                        "target": item["target"],
                        "execution_id": None,
                        "action": "cancelled_before_launch",
                        "detail": "task had not been submitted",
                    }
                    item["cancellation"] = record
                    item["cancellation_requested_at"] = _now()
                    item["terminal_failure"] = True
                    item["failure"] = record["detail"]
                    records_by_task[str(item["task_id"])] = record
                    continue
                target_name = str(item["target"])
                handle = ExecutionHandle.from_mapping(handle_value)
                cancellable_by_target.setdefault(target_name, []).append(
                    (item, handle)
                )

            for target_name, target_items in cancellable_by_target.items():
                executor = controller.executor_factory(
                    controller.targets[target_name]
                )
                try:
                    cancel_many = getattr(executor, "cancel_many", None)
                    handles = tuple(handle for _, handle in target_items)
                    statuses = (
                        tuple(cancel_many(handles))
                        if callable(cancel_many)
                        else tuple(executor.cancel(handle) for handle in handles)
                    )
                except ExecutionError as error:
                    raise ControllerError(
                        f"failed to cancel {target_name} task group: {error}"
                    ) from error
                if len(statuses) != len(target_items):
                    raise ControllerError(
                        "batch executor returned the wrong number of "
                        "cancellation results"
                    )
                for (item, handle), status in zip(
                    target_items,
                    statuses,
                    strict=True,
                ):
                    record = {
                        "target": target_name,
                        "executor": handle.executor,
                        "execution_id": handle.execution_id,
                        "action": status.state,
                        "detail": status.detail,
                    }
                    item["cancellation"] = record
                    item["cancellation_requested_at"] = _now()
                    item["observed_state"] = status.state
                    item["observed_at"] = _now()
                    item["detail"] = status.detail
                    records_by_task[str(item["task_id"])] = record
                    if status.state not in {
                        "cancelled",
                        "failed",
                        "completed",
                        "cancelling",
                    }:
                        raise ControllerError(
                            f"{current['stage']} task cancellation returned "
                            f"unsafe state {status.state}"
                        )
                    if status.state in {"cancelled", "failed"}:
                        item["terminal_failure"] = True
                        item["failure"] = (
                            status.detail or "execution cancelled by user"
                        )
                    elif status.state == "completed":
                        item.pop("cancellation_requested_at", None)
            records = [
                records_by_task[str(item["task_id"])]
                for item in current["tasks"]
            ]
            current["cancellations"] = records
            current["stopped_at"] = _now()
            controller.state["state"] = "stopped"
            controller.state["reason"] = (
                f"controller stopped; cancellation state for current "
                f"{current['stage']} task group is preserved"
            )
            controller._save()
            result["current_execution"] = {
                "action": "group_cancellation_requested",
                "tasks": records,
            }
            return finish()
        handle_value = current.get("handle")
        if handle_value is None:
            raise ControllerError(
                "current task has no execution handle; cancellation cannot "
                "safely determine whether submission completed"
            )
        target_name = str(current["target"])
        target = controller.targets[target_name]
        handle = ExecutionHandle.from_mapping(handle_value)
        executor = controller.executor_factory(target)
        try:
            status = executor.cancel(handle)
        except ExecutionError as error:
            raise ControllerError(
                f"failed to cancel {target_name} execution "
                f"{handle.execution_id}: {error}"
            ) from error

        record = {
            "target": target_name,
            "executor": handle.executor,
            "execution_id": handle.execution_id,
            "action": status.state,
            "detail": status.detail,
        }
        result["current_execution"] = record
        if status.state == "completed":
            controller.state["state"] = "stopped"
            controller.state["reason"] = (
                "controller stopped by user; current execution completed "
                "before cancellation and remains available for collection"
            )
            controller._save()
            return finish()
        if status.state == "cancelling":
            current["cancellation_requested_at"] = _now()
            current["cancellation"] = record
            controller.state["state"] = "stopped"
            controller.state["reason"] = (
                "controller stopped; current execution cancellation was "
                "accepted but is not terminal yet"
            )
            controller._save()
            return finish()
        if status.state not in {"cancelled", "failed"}:
            raise ControllerError(
                f"execution cancellation returned unsafe state {status.state}"
            )
        archived = dict(current)
        archived["cancelled_at"] = _now()
        archived["cancellation"] = record
        controller.state.setdefault("history", []).append(archived)
        controller.state["current"] = None
        controller.state["state"] = "stopped"
        controller.state["reason"] = (
            "controller stopped and current execution cancelled by user"
            if status.state == "cancelled"
            else "controller stopped; current execution had already failed"
        )
        controller._save()
        return finish()


__all__ = [
    "ControllerError",
    "ControllerTick",
    "PersistentController",
    "controller_running",
    "run_controller",
    "start_controller",
    "stop_controller",
    "stop_workflow",
]
