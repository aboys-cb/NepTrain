"""Persistent, scheduler-independent controller for NepTrain workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
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

from .workflow_workspace import WorkflowWorkspace
from .config import ConfigError, load_config
from .execution import (
    ExecutionError,
    ExecutionHandle,
    ExecutionTarget,
    StageExecutor,
    build_stage_task,
    executor_for,
    load_stage_result,
)
from .iteration import GenerationController, GenerationPlan, IterationError, StageOutcome


class ControllerError(RuntimeError):
    """Raised when a workflow controller cannot safely make progress."""


_RESOURCE_FOR_STAGE = {
    "train": "training",
    "explore": "sampling",
    "select": "analysis",
    "label": "labeling",
    "diagnose": "analysis",
    "merge": "analysis",
    "retrain": "training",
    "evaluate": "analysis",
}
_STOP = False
_STOP_EVENT = threading.Event()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path, default: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_file():
        return dict(default or {})
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_matches(record: Mapping[str, Any]) -> bool:
    path = Path(str(record["path"]))
    if record.get("kind", "file") == "file":
        return path.is_file() and _sha256(path) == record.get("sha256")
    if not path.is_dir():
        return False
    entries = [
        {"path": str(item.relative_to(path)), "sha256": _sha256(item)}
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    digest = hashlib.sha256(
        json.dumps(
            entries, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return digest == record.get("sha256")


def _plan(path: Path) -> GenerationPlan:
    value = json.loads(path.read_text(encoding="utf-8"))
    value["temperatures"] = tuple(value["temperatures"])
    return GenerationPlan(**value)


def _controller_command() -> list[str]:
    return [sys.executable, "-m", "NepTrain.cli.cli"]


def _process_matches(pid: int, project: Path) -> bool:
    try:
        completed = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
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
        if self.workspace.version != 2:
            raise ControllerError(
                "persistent controllers require workflow layout v2; migrate the project first"
            )
        self.manifest = _read_json(self.workspace.manifest)
        if self.manifest.get("orchestration") != "controller-v1":
            raise ControllerError(
                "workflow was prepared for the legacy Slurm dependency chain"
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
            self.workspace.root, self.workflow_id
        )
        self.executor_factory = executor_factory
        self.targets, self.routes = self._execution_config(
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
    ) -> tuple[dict[str, ExecutionTarget], dict[str, str]]:
        execution = config.get("execution", {})
        raw_targets = execution.get("targets", {})
        routes = {
            str(key): str(value)
            for key, value in dict(execution.get("routes", {})).items()
        }
        required = {"training", "sampling", "labeling", "analysis"}
        missing = sorted(required - set(routes))
        if missing:
            raise ControllerError(
                "execution.routes is missing " + ", ".join(missing)
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
        unknown = sorted(set(routes.values()) - set(targets))
        if unknown:
            raise ControllerError(
                "execution.routes refers to unknown targets: " + ", ".join(unknown)
            )
        return targets, routes

    def _save(self) -> None:
        self.state["heartbeat_at"] = _now()
        _write_json(self.workspace.controller_file, self.state)

    def _ledger(self) -> dict[str, Any]:
        return _read_json(
            self.workspace.ledger,
            {"version": 1, "workflow_id": self.workflow_id, "generations": {}},
        )

    def _next(self) -> tuple[GenerationPlan, str, Any] | None:
        ledger = self._ledger()
        for plan in self.plans:
            record = ledger.get("generations", {}).get(str(plan.generation), {})
            if record.get("complete"):
                if record.get("accepted") is False:
                    self.state["state"] = "rejected"
                    self.state["reason"] = f"generation {plan.generation} failed evaluation"
                    self._save()
                    return None
                continue
            stage, context = self.generation_controller.stage_context(plan)
            return plan, stage, context
        self.state["state"] = "complete"
        self.state["reason"] = "all prepared generations passed evaluation"
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
            value, outcome = load_stage_result(bundle)
        except ExecutionError as error:
            raise ControllerError(f"completed stage returned an invalid result: {error}") from error
        current = self.state["current"]
        expected = {
            "task_id": current["task_id"],
            "workflow_id": self.workflow_id,
            "generation": plan.generation,
            "stage": stage,
            "plan_sha256": plan.sha256,
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise ControllerError(
                    f"stage result {key} does not match controller intent"
                )
        root = self.workspace.stage_dir(plan.generation, stage) / "artifacts" / f"attempt-{attempt}"
        installed = {}
        for name, source in outcome.artifacts.items():
            destination = root / name / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
            installed[name] = destination
        return self.generation_controller.commit_stage(
            plan,
            stage,
            StageOutcome(artifacts=installed, metrics=outcome.metrics),
        )

    def tick(self) -> ControllerTick:
        self.state = _read_json(self.workspace.controller_file, self.state)
        current = self.state.get("current")
        if current is not None:
            target = self.targets[str(current["target"])]
            executor = self.executor_factory(target)
            bundle = Path(current["bundle"])
            handle_value = current.get("handle")
            if handle_value is None:
                task_value = json.loads((bundle / "task.json").read_text(encoding="utf-8"))
                from .execution import StageTask

                task = StageTask(
                    str(current["task_id"]),
                    self.workflow_id,
                    int(current["generation"]),
                    str(current["stage"]),
                    str(current["target"]),
                    bundle,
                )
                handle = executor.launch(task)
                current["handle"] = asdict(handle)
                current["submitted_at"] = _now()
                self.state["state"] = "running"
                self.state.pop("last_transport_error", None)
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
            if status.state == "failed":
                self.state["state"] = "failed"
                self.state["reason"] = status.detail or "stage execution failed"
                self._save()
                return ControllerTick(
                    "failed",
                    int(current["generation"]),
                    str(current["stage"]),
                    str(current["target"]),
                    handle.execution_id,
                    self.state["reason"],
                )
            collected = executor.collect(handle)
            plan = next(
                item for item in self.plans if item.generation == int(current["generation"])
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
        resource = _RESOURCE_FOR_STAGE[stage]
        target_name = self.routes[resource]
        target = self.targets[target_name]
        attempt = self._attempt(plan.generation, stage)
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
        )
        self.state["current"] = {
            "task_id": task.task_id,
            "generation": plan.generation,
            "stage": stage,
            "resource": resource,
            "target": target_name,
            "attempt": attempt,
            "bundle": str(task.bundle),
            "handle": None,
            "created_at": _now(),
        }
        self.state["state"] = "launching"
        self._save()
        # Launch in the same tick; the persisted intent makes this retry-safe.
        return self.tick()

    def retry(self, *, recover_rejected: bool = False) -> None:
        state = str(self.state.get("state", "idle"))
        if state == "complete":
            raise ControllerError("workflow is already complete")
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
            self.generation_controller.reopen_rejected(rejected[0], from_stage="retrain")
        current = self.state.get("current")
        if current is not None:
            archived = dict(current)
            archived["failed_at"] = _now()
            archived["failure"] = self.state.get("reason", "manual retry")
            self.state.setdefault("history", []).append(archived)
        self.state["current"] = None
        self.state["state"] = "idle"
        self.state.pop("reason", None)
        self._save()


def _signal_stop(_signum, _frame) -> None:
    global _STOP
    _STOP = True
    _STOP_EVENT.set()


def run_controller(project: str | Path, *, poll_interval: float | None = None) -> int:
    """Hold the workflow controller lock and supervise until a terminal state."""

    global _STOP
    _STOP = False
    _STOP_EVENT.clear()
    workspace = WorkflowWorkspace.locate(project)
    interval = poll_interval
    if interval is None:
        config, _ = load_config(workspace.project_file)
        interval = float(config.get("execution", {}).get("poll_interval", 30.0))
    if interval < 0.2:
        raise ControllerError("execution.poll_interval must be at least 0.2 seconds")
    signal.signal(signal.SIGTERM, _signal_stop)
    signal.signal(signal.SIGINT, _signal_stop)
    with workspace.controller_lock.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ControllerError("workflow controller is already running") from error
        workspace.controller_pid.write_text(f"{os.getpid()}\n", encoding="utf-8")
        controller = PersistentController(workspace.root)
        controller.state["pid"] = os.getpid()
        controller.state["started_at"] = _now()
        controller.state["state"] = "running"
        controller.state.pop("reason", None)
        controller._save()
        try:
            while not _STOP:
                try:
                    tick = controller.tick()
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
                    failures = int(controller.state.get("transport_failures", 0)) + 1
                    controller.state["transport_failures"] = failures
                    controller.state["last_transport_error"] = str(error)
                    controller.state["state"] = "degraded"
                    controller._save()
                    tick = ControllerTick("degraded", detail=str(error))
                if tick.state in {"complete", "rejected", "failed"}:
                    return 0 if tick.state == "complete" else 2
                _STOP_EVENT.wait(interval)
            controller.state["state"] = "stopped"
            controller.state["reason"] = "controller stopped by user"
            controller._save()
            return 0
        except Exception as error:
            controller.state["state"] = "failed"
            controller.state["reason"] = str(error)
            controller._save()
            raise
        finally:
            try:
                if workspace.controller_pid.read_text(encoding="utf-8").strip() == str(os.getpid()):
                    workspace.controller_pid.unlink()
            except FileNotFoundError:
                pass


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
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not controller_running(workspace.root):
            return
        time.sleep(0.05)
    raise ControllerError("controller did not stop within 5 seconds")


def stop_workflow(
    project: str | Path,
    *,
    cancel_jobs: bool = False,
    executor_factory: ExecutorFactory = executor_for,
) -> dict[str, Any]:
    """Stop the controller and optionally cancel its current execution.

    Cancellation is intentionally explicit. A successfully cancelled task is
    archived and removed from ``current`` so a later resume creates a new,
    traceable attempt instead of inspecting the cancelled handle again.
    """

    workspace = WorkflowWorkspace.locate(project)
    was_running = controller_running(workspace.root)
    if was_running:
        stop_controller(workspace.root)
    elif not cancel_jobs:
        raise ControllerError("workflow controller is not running")

    result: dict[str, Any] = {
        "project": str(workspace.root),
        "controller": "stopped" if was_running else "already_stopped",
        "current_execution": None,
    }
    if not cancel_jobs:
        return result

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
            return result
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
            return result
        if status.state == "cancelling":
            current["cancellation_requested_at"] = _now()
            current["cancellation"] = record
            controller.state["state"] = "stopped"
            controller.state["reason"] = (
                "controller stopped; current execution cancellation was "
                "accepted but is not terminal yet"
            )
            controller._save()
            return result
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
        return result


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
