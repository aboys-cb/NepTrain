"""Portable stage tasks and execution adapters for persistent workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from typing import Any, Callable, Mapping, Protocol, Sequence

from ase.io import read as ase_read
from ase.io import write as ase_write

from .iteration import GenerationPlan, StageContext, StageOutcome
from .sampling_route import load_sampling_routes


class ExecutionError(RuntimeError):
    """Raised when a stage cannot be transported, launched, or collected."""


_TERMINAL_SUCCESS = {"COMPLETED"}
_TERMINAL_FAILURE = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}
_ACTIVE = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "STAGE_OUT",
    "SUSPENDED",
}


def _slurm_state(value: str) -> str:
    """Normalize states such as ``COMPLETED+`` and ``CANCELLED by <uid>``."""

    return re.split(r"[+\s]", value.strip(), maxsplit=1)[0].upper()


_STAGE_CONFIG_PATH_FIELDS = {
    "train": ("training.config_path", "training.test_path"),
    "explore": (),
    "select": (),
    "label": ("dft.input_path", "dft.resource_path"),
    "diagnose": (),
    "merge": (),
    "retrain": ("training.config_path", "training.test_path"),
    "evaluate": (),
}
_STAGE_ARTIFACTS = {
    "train": (),
    "explore": ("model",),
    "select": (
        "candidates",
        "candidate_pool_manifest",
        "training_input",
        "model",
        "md_attempts",
    ),
    "label": ("selected_input",),
    "diagnose": ("labeled", "model"),
    "merge": ("training_input", "labeled"),
    "retrain": (
        "training_input",
        "training_set",
        "model",
        "checkpoint",
        "acquisition_signals",
        "md_attempts",
    ),
    "evaluate": (
        "training_input",
        "training_set",
        "model",
        "checkpoint",
        "labeled",
        "retrained_model",
        "retrained_checkpoint",
        "retraining_decision",
        "model_lineage",
        "scenario_plan",
        "acquisition_signals",
        "selection_result",
        "md_attempts",
    ),
}
_STAGE_PREVIOUS_ARTIFACTS = {
    "train": (
        "training_set",
        "activated_model",
        "activated_checkpoint",
        "active_model_lineage",
    ),
    "explore": ("scenario_maturity",),
    "select": (),
    "label": (),
    "diagnose": (),
    "merge": (),
    "retrain": ("signals", "scenario_maturity"),
    "evaluate": ("signals", "scenario_maturity"),
}
_STAGE_DIRECTORY_LABELS = {
    "train": "train",
    "explore": "md",
    "select": "select",
    "label": "dft",
    "diagnose": "diagnose",
    "merge": "merge",
    "retrain": "retrain",
    "evaluate": "evaluate",
}


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> str:
    if path.is_file():
        return _sha256(path)
    records = [
        (str(item.relative_to(path)), _sha256(item))
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    return hashlib.sha256(
        json.dumps(records, separators=(",", ":")).encode()
    ).hexdigest()


def _link_or_copy(source: str, target: str) -> str:
    try:
        os.link(source, target)
        return target
    except OSError:
        return shutil.copy2(source, target)


def _copy_input(source: Path, target: Path) -> None:
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(str(source), str(target))
        return
    if source.is_dir():
        shutil.copytree(source, target, copy_function=_link_or_copy)
        return
    raise ExecutionError(f"task input does not exist: {source}")


def _get_dotted(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for key in dotted.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _set_dotted(value: dict[str, Any], dotted: str, replacement: Any) -> None:
    keys = dotted.split(".")
    current = value
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = replacement


def _delete_dotted(value: dict[str, Any], dotted: str) -> None:
    keys = dotted.split(".")
    current: Any = value
    for key in keys[:-1]:
        if not isinstance(current, dict) or key not in current:
            return
        current = current[key]
    if isinstance(current, dict):
        current.pop(keys[-1], None)


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _safe_relative(base: Path, value: str | Path, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ExecutionError(f"{label} must stay inside the task bundle")
    root = base.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ExecutionError(f"{label} escapes the task bundle") from error
    return path


def _validate_artifact_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ExecutionError(f"invalid stage artifact name: {name}")


def _safe_name(value: str, fallback: str = "task") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return cleaned or fallback


def _task_directory_name(
    *,
    generation: int,
    stage: str,
    attempt: int,
    task_id: str,
    stage_input: Mapping[str, Any],
) -> str:
    parts = [f"g{generation:04d}", _STAGE_DIRECTORY_LABELS[stage]]
    route_id = str(stage_input.get("route_id", "")).strip()
    if route_id:
        parts.append(_safe_name(route_id, "route"))
    attempt_ids = stage_input.get("attempt_ids", ())
    if isinstance(attempt_ids, Sequence) and not isinstance(attempt_ids, str):
        first_attempt_id = next(iter(attempt_ids), None)
        if first_attempt_id:
            parts.append(_safe_name(str(first_attempt_id), "sample")[:8])
    batch_index = stage_input.get("batch_index")
    if batch_index is not None:
        parts.append(f"{int(batch_index):06d}")
    parts.extend([f"a{attempt}", task_id[:8]])
    return "-".join(parts)


@dataclass(frozen=True)
class ExecutionTarget:
    """One named place where a stage can run."""

    name: str
    executor: str
    host: str | None = None
    work_root: str | None = None
    command: str = "neptrain"
    setup_script: str | None = None
    partition: str | None = None
    qos: str | None = None
    time: str = "01:00:00"
    cpus_per_task: int | None = None
    gpus_per_node: int | None = None
    directives: tuple[str, ...] = ()
    dft_resource_path: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> "ExecutionTarget":
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ExecutionError(f"invalid execution target name: {name}")
        executor = str(value.get("executor", value.get("type", "process")))
        if executor not in {"process", "slurm"}:
            raise ExecutionError(
                f"execution target {name}.executor must be process or slurm"
            )
        command = str(value.get("command", "neptrain"))
        command_tokens = shlex.split(command)
        if not command_tokens or "=" in command_tokens[0]:
            raise ExecutionError(
                f"execution target {name}.command must start with an executable; "
                "put environment variables under environment"
            )
        host = value.get("host")
        work_root = value.get("work_root")
        if host and not re.fullmatch(r"[A-Za-z0-9_.@-]+", str(host)):
            raise ExecutionError(f"execution target {name} has an unsafe SSH host")
        if host and not work_root:
            raise ExecutionError(
                f"remote execution target {name} requires work_root"
            )
        if work_root and not re.fullmatch(r"(?:/|~/)[A-Za-z0-9_./-]+", str(work_root)):
            raise ExecutionError(
                f"execution target {name}.work_root must be a safe absolute path or start with ~/"
            )
        if work_root and ".." in Path(str(work_root).removeprefix("~/")).parts:
            raise ExecutionError(
                f"execution target {name}.work_root cannot contain parent traversal"
            )
        for field_name in ("partition", "qos", "time", "setup_script"):
            raw = value.get(field_name)
            if raw is not None and any(character in str(raw) for character in "\r\n"):
                raise ExecutionError(
                    f"execution target {name}.{field_name} cannot contain newlines"
                )
        for field_name in ("partition", "qos"):
            raw = value.get(field_name)
            if raw is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", str(raw)):
                raise ExecutionError(
                    f"execution target {name}.{field_name} contains unsafe characters"
                )
        if not re.fullmatch(r"[0-9:-]+", str(value.get("time", "01:00:00"))):
            raise ExecutionError(f"execution target {name}.time is invalid")
        if executor == "slurm" and not value.get("partition"):
            raise ExecutionError(
                f"Slurm execution target {name} requires partition"
            )
        cpus = value.get("cpus_per_task")
        gpus = value.get("gpus_per_node")
        if cpus is not None and int(cpus) < 1:
            raise ExecutionError(f"execution target {name} has invalid cpus_per_task")
        if gpus is not None and int(gpus) < 0:
            raise ExecutionError(f"execution target {name} has invalid gpus_per_node")
        dft_resource_path = value.get("dft_resource_path")
        if dft_resource_path is not None and any(
            character in str(dft_resource_path) for character in "\r\n"
        ):
            raise ExecutionError(
                f"execution target {name}.dft_resource_path cannot contain newlines"
            )
        if dft_resource_path is not None:
            resource_text = str(dft_resource_path)
            if not (
                Path(resource_text).is_absolute()
                or resource_text.startswith("~/")
            ):
                raise ExecutionError(
                    f"execution target {name}.dft_resource_path must be absolute "
                    "or start with ~/"
                )
            if ".." in Path(resource_text.removeprefix("~/")).parts:
                raise ExecutionError(
                    f"execution target {name}.dft_resource_path cannot contain "
                    "parent traversal"
                )
        environment = dict(value.get("environment", {}))
        invalid_environment = [
            str(key)
            for key in environment
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key))
        ]
        if invalid_environment:
            raise ExecutionError(
                f"execution target {name} has invalid environment keys: "
                + ", ".join(invalid_environment)
            )
        directives = tuple(str(item) for item in value.get("directives", []))
        if any(
            "\n" in item
            or "\r" in item
            or not item.removeprefix("#SBATCH ").startswith("--")
            for item in directives
        ):
            raise ExecutionError(
                f"execution target {name} directives must be single-line Slurm --options"
            )
        return cls(
            name=name,
            executor=executor,
            host=str(host) if host else None,
            work_root=str(work_root) if work_root else None,
            command=command,
            setup_script=str(value["setup_script"])
            if value.get("setup_script")
            else None,
            partition=str(value["partition"]) if value.get("partition") else None,
            qos=str(value["qos"]) if value.get("qos") else None,
            time=str(value.get("time", "01:00:00")),
            cpus_per_task=int(cpus) if cpus is not None else None,
            gpus_per_node=int(gpus) if gpus is not None else None,
            directives=directives,
            dft_resource_path=(
                str(dft_resource_path) if dft_resource_path is not None else None
            ),
            environment={
                str(key): str(item)
                for key, item in environment.items()
            },
        )


@dataclass(frozen=True)
class StageTask:
    task_id: str
    workflow_id: str
    generation: int
    stage: str
    target: str
    bundle: Path

    @property
    def descriptor(self) -> Path:
        return self.bundle / "task.json"


@dataclass(frozen=True)
class ExecutionHandle:
    task_id: str
    target: str
    executor: str
    execution_id: str
    local_bundle: str
    remote_bundle: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExecutionHandle":
        return cls(
            task_id=str(value["task_id"]),
            target=str(value["target"]),
            executor=str(value["executor"]),
            execution_id=str(value["execution_id"]),
            local_bundle=str(value["local_bundle"]),
            remote_bundle=str(value["remote_bundle"])
            if value.get("remote_bundle")
            else None,
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class ExecutionStatus:
    state: str
    detail: str = ""

    @property
    def terminal(self) -> bool:
        return self.state in {"completed", "failed", "cancelled"}


class StageExecutor(Protocol):
    """Small execution seam used by the workflow controller."""

    def launch(self, task: StageTask) -> ExecutionHandle: ...

    def inspect(self, handle: ExecutionHandle) -> ExecutionStatus: ...

    def collect(self, handle: ExecutionHandle) -> Path: ...

    def cancel(self, handle: ExecutionHandle) -> ExecutionStatus: ...


def build_stage_task(
    tasks_dir: Path,
    *,
    workflow_root: Path,
    workflow_id: str,
    generation: int,
    stage: str,
    attempt: int,
    target: ExecutionTarget,
    plan: GenerationPlan,
    config: Mapping[str, Any],
    initial_training: Path,
    context: StageContext,
    stage_input: Mapping[str, Any] | None = None,
) -> StageTask:
    """Build an immutable, self-contained task directory."""

    if stage not in _STAGE_CONFIG_PATH_FIELDS:
        raise ExecutionError(f"unsupported stage task: {stage}")

    requested_route_id = str((stage_input or {}).get("route_id", ""))
    route_identities = (
        [
            {
                "route_id": route.route_id,
                "route_fingerprint": route.fingerprint,
            }
            for route in load_sampling_routes(
                config["sampling"],
                base_dir=workflow_root,
            )
            if not requested_route_id or route.route_id == requested_route_id
        ]
        if stage in {"explore", "evaluate"}
        and config.get("sampling", {}).get("routes")
        else []
    )
    identity = {
        "workflow_id": workflow_id,
        "generation": generation,
        "stage": stage,
        "attempt": attempt,
        "plan_sha256": plan.sha256,
        "target": target.name,
        "sampling_routes": route_identities,
        "stage_input": dict(stage_input or {}),
    }
    task_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    directory_name = _task_directory_name(
        generation=generation,
        stage=stage,
        attempt=attempt,
        task_id=task_id,
        stage_input=dict(stage_input or {}),
    )
    bundle = tasks_dir / directory_name
    if bundle.exists():
        descriptor = bundle / "task.json"
        if not descriptor.is_file():
            raise ExecutionError(f"incomplete existing task bundle: {bundle}")
        existing = json.loads(descriptor.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise ExecutionError(f"task id collision at {bundle}")
        return StageTask(task_id, workflow_id, generation, stage, target.name, bundle)

    tasks_dir.mkdir(parents=True, exist_ok=True)
    temporary = tasks_dir / f".{directory_name}.building"
    if temporary.exists():
        shutil.rmtree(temporary)
    inputs = temporary / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    portable_config = json.loads(json.dumps(config))
    if stage == "explore" and requested_route_id:
        portable_config["sampling"]["routes"] = [
            route
            for route in portable_config["sampling"]["routes"]
            if str(route["id"]) == requested_route_id
        ]
        if len(portable_config["sampling"]["routes"]) != 1:
            raise ExecutionError(
                f"explore task refers to unknown route {requested_route_id}"
            )
    if stage == "label" and target.dft_resource_path:
        portable_config.setdefault("dft", {})["resource_path"] = (
            target.dft_resource_path
        )

    path_fields = set(_STAGE_CONFIG_PATH_FIELDS[stage])
    if stage == "train" and generation > 1:
        path_fields.clear()
    evaluation = portable_config.get("evaluation", {})
    if stage == "evaluate":
        validation_field = (
            "evaluation.validation_path"
            if evaluation.get("validation_path")
            else "training.test_path"
        )
        path_fields.add(validation_field)
    all_path_fields = {
        dotted
        for fields in _STAGE_CONFIG_PATH_FIELDS.values()
        for dotted in fields
    } | {"evaluation.validation_path", "training.initial_path"}
    for dotted in sorted(all_path_fields - path_fields):
        _delete_dotted(portable_config, dotted)
    if stage not in {"explore", "evaluate"}:
        portable_config.get("sampling", {}).pop("routes", None)
    for dotted in sorted(path_fields):
        value = _get_dotted(portable_config, dotted)
        if value in {None, "", "auto"}:
            continue
        if dotted == "dft.resource_path" and target.dft_resource_path:
            continue
        source = _resolve_path(value, workflow_root)
        suffix = source.suffix if source.is_file() else ""
        destination = inputs / "config" / dotted.replace(".", "/")
        if suffix:
            destination = destination.with_suffix(suffix)
        _copy_input(source, destination)
        _set_dotted(
            portable_config,
            dotted,
            str(destination.relative_to(temporary)),
        )

    routes = (
        portable_config.get("sampling", {}).get("routes", [])
        if stage in {"explore", "evaluate"}
        else []
    )
    for route_index, route in enumerate(routes):
        route_root = (
            inputs
            / "sampling"
            / "routes"
            / f"{route_index:03d}-{route['id']}"
        )
        source = _resolve_path(route["template_path"], workflow_root)
        destination = route_root / f"template{source.suffix}"
        _copy_input(source, destination)
        route["template_path"] = str(destination.relative_to(temporary))
        copied_structures = []
        for structure_index, value in enumerate(route["structures"]):
            source = _resolve_path(value, workflow_root)
            destination = route_root / "structures" / str(structure_index)
            if source.is_file():
                destination = destination.with_suffix(source.suffix)
            _copy_input(source, destination)
            copied_structures.append(str(destination.relative_to(temporary)))
        route["structures"] = copied_structures

    initial_value = None
    if stage == "train" and generation == 1:
        initial_destination = inputs / "initial" / initial_training.name
        if not initial_destination.exists():
            _copy_input(initial_training, initial_destination)
        initial_value = str(initial_destination.relative_to(temporary))
        portable_config.setdefault("training", {})["initial_path"] = initial_value
    if target.setup_script:
        setup_source = Path(target.setup_script).expanduser()
        if setup_source.is_file():
            _copy_input(setup_source.resolve(), temporary / "target-setup.sh")

    def copy_artifacts(
        values: Mapping[str, Path], category: str, required: Sequence[str]
    ) -> dict[str, str]:
        copied = {}
        for name in required:
            if name not in values:
                continue
            _validate_artifact_name(name)
            source = values[name]
            source = Path(source).resolve()
            destination = (
                inputs / "artifacts" / f"{category}-{name}--{source.name}"
            )
            if stage == "label" and name == "selected_input":
                raw_indices = (stage_input or {}).get("frame_indices")
                if raw_indices is None:
                    _copy_input(source, destination)
                else:
                    if (
                        not isinstance(raw_indices, Sequence)
                        or isinstance(raw_indices, str)
                        or not raw_indices
                    ):
                        raise ExecutionError(
                            "label frame_indices must be a non-empty sequence"
                        )
                    indices = [int(value) for value in raw_indices]
                    if indices != sorted(set(indices)) or indices[0] < 0:
                        raise ExecutionError(
                            "label frame_indices must be unique sorted "
                            "non-negative integers"
                        )
                    frames = ase_read(source, index=":")
                    if not isinstance(frames, list):
                        frames = [frames]
                    if indices[-1] >= len(frames):
                        raise ExecutionError(
                            "label frame_indices exceed selected input size"
                        )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    ase_write(
                        destination,
                        [frames[index] for index in indices],
                        format="extxyz",
                    )
            else:
                _copy_input(source, destination)
            copied[name] = str(destination.relative_to(temporary))
        return copied

    descriptor = {
        "protocol": "neptrain.stage-task.v2",
        "task_id": task_id,
        "identity": identity,
        "created_at": _now(),
        "config": portable_config,
        "initial_training": initial_value,
        "plan": asdict(plan),
        "stage_input": dict(stage_input or {}),
        "artifacts": copy_artifacts(
            context.artifacts, "current", _STAGE_ARTIFACTS[stage]
        ),
        "previous_artifacts": copy_artifacts(
            context.previous_artifacts,
            "previous",
            _STAGE_PREVIOUS_ARTIFACTS[stage],
        ),
    }
    records = [
        {
            "path": str(path.relative_to(temporary)),
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(temporary.rglob("*"))
        if path.is_file() and path.name != "task.json"
    ]
    descriptor["files"] = records
    _write_json(temporary / "task.json", descriptor)
    temporary.replace(bundle)
    return StageTask(task_id, workflow_id, generation, stage, target.name, bundle)


def _verify_task_bundle(bundle: Path) -> dict[str, Any]:
    descriptor_path = bundle / "task.json"
    if not descriptor_path.is_file():
        raise ExecutionError("stage task is missing task.json")
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if descriptor.get("protocol") != "neptrain.stage-task.v2":
        raise ExecutionError("unsupported stage task descriptor")
    for record in descriptor.get("files", []):
        path = _safe_relative(bundle, record["path"], "task manifest path")
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise ExecutionError(f"stage task input drifted: {path}")
    return descriptor


def run_stage_worker(bundle_path: str | Path) -> int:
    """Execute exactly one portable task and write a hash-checked result."""

    bundle = Path(bundle_path).expanduser().resolve()
    bundle.mkdir(parents=True, exist_ok=True)
    lock_path = bundle / ".worker.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        execution_path = bundle / "execution.json"
        try:
            descriptor = _verify_task_bundle(bundle)
            _write_json(
                execution_path,
                {
                    "task_id": descriptor["task_id"],
                    "state": "RUNNING",
                    "pid": os.getpid(),
                    "started_at": _now(),
                },
            )
            config = descriptor["config"]
            plan_value = dict(descriptor["plan"])
            plan = GenerationPlan(**plan_value)
            resolve = lambda value: _safe_relative(bundle, value, "task input path")
            artifacts = {
                name: resolve(path)
                for name, path in descriptor.get("artifacts", {}).items()
            }
            previous = {
                name: resolve(path)
                for name, path in descriptor.get("previous_artifacts", {}).items()
            }
            from .workflow_iteration import WorkflowIterationAdapter

            work_dir = bundle / "output"
            work_dir.mkdir(parents=True, exist_ok=True)
            initial_value = descriptor.get("initial_training")
            adapter = WorkflowIterationAdapter(
                config,
                initial_training=resolve(initial_value) if initial_value else None,
                base_dir=bundle,
                active_stage=str(descriptor["identity"]["stage"]),
            )
            outcome = adapter.run_stage(
                descriptor["identity"]["stage"],
                StageContext(
                    generation=int(descriptor["identity"]["generation"]),
                    generation_dir=work_dir,
                    plan=plan,
                    artifacts=artifacts,
                    previous_artifacts=previous,
                    stage_dir=work_dir,
                    stage_input=dict(descriptor.get("stage_input", {})),
                    flat_output=True,
                ),
            )
            result_artifacts = {}
            for name, raw_path in sorted(outcome.artifacts.items()):
                _validate_artifact_name(str(name))
                source = Path(raw_path).expanduser().resolve()
                if not source.is_file():
                    raise ExecutionError(
                        f"stage worker did not produce artifact {source}"
                    )
                try:
                    source.relative_to(work_dir.resolve())
                    destination = source
                except ValueError:
                    destination = work_dir / f"{name}--{source.name}"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                result_artifacts[name] = {
                    "path": str(destination.relative_to(bundle)),
                    "sha256": _sha256(destination),
                    "size": destination.stat().st_size,
                }
            result = {
                "protocol": "neptrain.stage-result.v2",
                "task_id": descriptor["task_id"],
                "workflow_id": descriptor["identity"]["workflow_id"],
                "generation": descriptor["identity"]["generation"],
                "stage": descriptor["identity"]["stage"],
                "plan_sha256": descriptor["identity"]["plan_sha256"],
                "completed_at": _now(),
                "artifacts": result_artifacts,
                "metrics": dict(outcome.metrics),
            }
            _write_json(bundle / "result.json", result)
            _write_json(
                execution_path,
                {
                    "task_id": descriptor["task_id"],
                    "state": "COMPLETED",
                    "pid": os.getpid(),
                    "completed_at": _now(),
                },
            )
            return 0
        except Exception as error:
            _write_json(
                execution_path,
                {
                    "state": "FAILED",
                    "pid": os.getpid(),
                    "failed_at": _now(),
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            return 1


def load_stage_result(bundle_path: str | Path) -> tuple[dict[str, Any], StageOutcome]:
    bundle = Path(bundle_path).expanduser().resolve()
    path = bundle / "result.json"
    if not path.is_file():
        raise ExecutionError(f"stage result does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol") != "neptrain.stage-result.v2":
        raise ExecutionError("unsupported stage result protocol")
    artifacts = {}
    for name, record in value.get("artifacts", {}).items():
        _validate_artifact_name(str(name))
        artifact = _safe_relative(bundle, record["path"], "result artifact path")
        if not artifact.is_file() or _sha256(artifact) != record["sha256"]:
            raise ExecutionError(f"stage result artifact drifted: {artifact}")
        artifacts[name] = artifact
    return value, StageOutcome(artifacts=artifacts, metrics=value.get("metrics", {}))


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class ExecutionTransport:
    """Shared local/SSH command and file transport for all execution paths."""

    def __init__(self, target: ExecutionTarget, runner: CommandRunner = subprocess.run):
        self.target = target
        self.runner = runner

    @property
    def remote(self) -> bool:
        return self.target.host is not None

    def run_script(
        self,
        script: str,
        *arguments: str | Path,
        check: bool = False,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess[str]:
        if not self.remote:
            raise ExecutionError("remote script requires an SSH execution target")
        command = [
            "ssh",
            str(self.target.host),
            "bash",
            "-s",
            "--",
            *(str(value) for value in arguments),
        ]
        try:
            completed = self.runner(
                command,
                input=script if script.endswith("\n") else script + "\n",
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise ExecutionError(
                f"remote command timed out after {timeout}s on {self.target.name}"
            ) from error
        if check and completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise ExecutionError(f"execution command failed: {detail}")
        return completed

    def copy(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        recursive: bool = False,
        check: bool = False,
        timeout: int = 300,
    ) -> subprocess.CompletedProcess[str]:
        if not self.remote:
            raise ExecutionError("remote copy requires an SSH execution target")
        command = ["scp"]
        if recursive:
            command.append("-r")
        command.extend([str(source), str(destination)])
        try:
            completed = self.runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise ExecutionError(
                f"remote copy timed out after {timeout}s for {self.target.name}"
            ) from error
        if check and completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise ExecutionError(f"execution copy failed: {detail}")
        return completed

    def fetch_paths(
        self,
        remote_root: str | Path,
        members: Sequence[str | Path],
        destination_root: str | Path,
    ) -> tuple[str, ...]:
        """Fetch existing remote paths with one archive transfer."""

        if not self.remote:
            raise ExecutionError("remote fetch requires an SSH execution target")
        relative_members = []
        for value in members:
            relative = Path(value)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
            ):
                raise ExecutionError(
                    "remote fetch paths must stay inside the requested root"
                )
            relative_members.append(relative.as_posix())
        if not relative_members:
            return ()

        destination = Path(destination_root).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        token = f"{os.getpid()}-{time.time_ns()}"
        archive_name = f".neptrain-fetch-{token}.tar.gz"
        remote_base = str(remote_root).rstrip("/")
        remote_archive = f"{remote_base}/{archive_name}"
        local_archive = destination / archive_name
        prepared = self.run_script(
            """set -eo pipefail
root=$1
archive=$2
shift 2
cd "$root"
existing=()
for member in "$@"; do
  if test -e "$member"; then
    existing+=("$member")
  fi
done
if [ "${#existing[@]}" -eq 0 ]; then
  tar -czf "$archive" --files-from /dev/null
else
  tar -czf "$archive" -- "${existing[@]}"
fi
""",
            remote_base,
            archive_name,
            *relative_members,
        )
        if prepared.returncode != 0:
            raise ExecutionError(
                (
                    prepared.stderr
                    or prepared.stdout
                    or "cannot prepare remote result archive"
                ).strip()
            )
        try:
            self.copy(
                f"{self.target.host}:{remote_archive}",
                local_archive,
                check=True,
            )
            with tarfile.open(local_archive, "r:gz") as archive:
                archived = archive.getmembers()
                for member in archived:
                    target = _safe_relative(
                        destination,
                        member.name,
                        "remote archive member",
                    )
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        raise ExecutionError(
                            "remote archive contains an unsupported member"
                        )
                    source = archive.extractfile(member)
                    if source is None:
                        raise ExecutionError(
                            "remote archive contains an unreadable file"
                        )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(target.suffix + ".tmp")
                    with source, temporary.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    temporary.replace(target)
            return tuple(member.name for member in archived)
        finally:
            local_archive.unlink(missing_ok=True)
            try:
                self.run_script('rm -f "$1"\n', remote_archive)
            except ExecutionError:
                # A cleanup timeout must not hide the transfer or validation
                # error that brought us here.
                pass

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        input_text: str | None = None,
        check: bool = False,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess[str]:
        command = list(args)
        actual_cwd = cwd
        if self.remote:
            command = [
                "ssh",
                str(self.target.host),
                "bash",
                "-s",
                "--",
                *command,
            ]
            actual_cwd = None
            script = """set -eo pipefail
root=$1
shift
case "$root" in
  '~/'*) root="$HOME/${root:2}" ;;
  /*) ;;
  *) echo "remote path must be absolute or start with ~/" >&2; exit 2 ;;
esac
cd "$root"
exec "$@"
"""
            input_text = script if input_text is None else input_text
        try:
            completed = self.runner(
                command,
                cwd=actual_cwd,
                input=input_text,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise ExecutionError(
                f"execution command timed out after {timeout}s on "
                f"{self.target.name}"
            ) from error
        if check and completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise ExecutionError(f"execution command failed: {detail}")
        return completed

    def deploy(self, task: StageTask) -> str:
        if not self.remote:
            return str(task.bundle)
        root = str(self.target.work_root)
        remote_name = task.bundle.name
        remote_bundle = (
            f"{root.rstrip('/')}/{task.workflow_id}/jobs/{remote_name}"
        )
        remote_archive = (
            f"{root.rstrip('/')}/{task.workflow_id}/incoming/{remote_name}.tar.gz"
        )
        setup = """set -eo pipefail
root=$1
workflow=$2
name=$3
case "$root" in
  '~/'*) root="$HOME/${root:2}" ;;
  /*) ;;
  *) exit 2 ;;
esac
mkdir -p "$root/$workflow/incoming" "$root/$workflow/jobs"
if [ -f "$root/$workflow/jobs/$name/task.json" ]; then exit 0; fi
"""
        completed = self.run_script(
            setup,
            root,
            task.workflow_id,
            remote_name,
        )
        if completed.returncode != 0:
            raise ExecutionError((completed.stderr or completed.stdout).strip())
        remote_exists = (
            self.run_script(
                'test -f "$1"\n',
                f"{remote_bundle}/task.json",
            ).returncode
            == 0
        )
        if remote_exists:
            return remote_bundle
        archive = task.bundle.parent / f".{remote_name}.upload.tar.gz"
        try:
            with tarfile.open(archive, "w:gz") as handle:
                handle.add(task.bundle, arcname=remote_name)
            digest = _sha256(archive)
            upload = self.copy(
                archive,
                f"{self.target.host}:{remote_archive}",
            )
            if upload.returncode != 0:
                raise ExecutionError((upload.stderr or upload.stdout).strip())
            extract = """set -eo pipefail
root=$1
workflow=$2
name=$3
expected=$4
case "$root" in
  '~/'*) root="$HOME/${root:2}" ;;
esac
archive="$root/$workflow/incoming/$name.tar.gz"
actual=$(sha256sum "$archive" | cut -d' ' -f1)
[ "$actual" = "$expected" ]
temporary="$root/$workflow/jobs/.$name.extracting"
rm -rf "$temporary"
mkdir -p "$temporary"
tar -xzf "$archive" -C "$temporary" --strip-components=1
mv "$temporary" "$root/$workflow/jobs/$name"
rm -f "$archive"
"""
            completed = self.run_script(
                extract,
                root,
                task.workflow_id,
                remote_name,
                digest,
            )
            if completed.returncode != 0:
                raise ExecutionError((completed.stderr or completed.stdout).strip())
        finally:
            archive.unlink(missing_ok=True)
        return remote_bundle

    def collect(self, handle: ExecutionHandle) -> Path:
        local = Path(handle.local_bundle)
        if not self.remote:
            return local
        remote = str(handle.remote_bundle)
        self.fetch_paths(
            remote,
            ("result.json", "execution.json", "result"),
            local,
        )
        for name in ("result.json", "execution.json"):
            if not (local / name).is_file():
                raise ExecutionError(
                    f"remote stage result is missing required file: {name}"
                )
        result = json.loads((local / "result.json").read_text(encoding="utf-8"))
        for record in result.get("artifacts", {}).values():
            destination = _safe_relative(
                local, record["path"], "result artifact path"
            )
            if not destination.is_file():
                raise ExecutionError(
                    f"remote stage result is missing artifact: {record['path']}"
                )
        return local


def _prepare_setup(target: ExecutionTarget, task: StageTask) -> None:
    if not target.setup_script:
        return
    candidate = Path(target.setup_script).expanduser()
    if candidate.is_file() and not (task.bundle / "target-setup.sh").is_file():
        shutil.copy2(candidate, task.bundle / "target-setup.sh")


def _setup_line(target: ExecutionTarget, bundle: str) -> str | None:
    if not target.setup_script:
        return None
    path = target.setup_script
    candidate = Path(path).expanduser()
    if candidate.is_file():
        source = str(candidate.resolve()) if target.host is None else f"{bundle}/target-setup.sh"
        return f"source {shlex.quote(source)}"
    return f"source {shlex.quote(path)}"


def _worker_command(target: ExecutionTarget, bundle: str) -> str:
    return shlex.join([*shlex.split(target.command), "stage-worker", bundle])


def _pid_matches_bundle(pid: int, bundle: str) -> bool:
    completed = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and bundle in completed.stdout


def _wait_for_process_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            waited, _ = os.waitpid(pid, os.WNOHANG)
            if waited == pid:
                return True
        except ChildProcessError:
            pass
        completed = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return True
        if completed.stdout.strip().startswith("Z"):
            return True
        time.sleep(0.05)
    return False


def _local_execution_status(path: Path) -> ExecutionStatus | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    state = value.get("state")
    if state == "COMPLETED":
        return ExecutionStatus("completed")
    if state == "FAILED":
        return ExecutionStatus(
            "failed", str(value.get("error", "worker failed"))
        )
    return None


def _wait_for_local_execution_status(
    path: Path, timeout: float = 0.25
) -> ExecutionStatus | None:
    deadline = time.monotonic() + timeout
    while True:
        status = _local_execution_status(path)
        if status is not None or time.monotonic() >= deadline:
            return status
        time.sleep(0.02)


class ProcessExecutor:
    """Run one stage as a detached process, locally or through SSH."""

    def __init__(self, target: ExecutionTarget, runner: CommandRunner = subprocess.run):
        self.target = target
        self.transport = ExecutionTransport(target, runner)

    def launch(self, task: StageTask) -> ExecutionHandle:
        _prepare_setup(self.target, task)
        bundle = self.transport.deploy(task)
        local_execution = task.bundle / "execution.json"
        if self.target.host is None and local_execution.is_file():
            value = json.loads(local_execution.read_text(encoding="utf-8"))
            if value.get("state") in {"RUNNING", "COMPLETED"}:
                return ExecutionHandle(
                    task.task_id,
                    self.target.name,
                    "process",
                    str(value.get("pid", "unknown")),
                    str(task.bundle),
                )
        lines = ["#!/bin/bash", "set -eo pipefail"]
        setup = _setup_line(self.target, bundle)
        if setup:
            lines.append(setup)
        for key, value in self.target.environment.items():
            lines.append(f"export {key}={shlex.quote(value)}")
        lines.extend([f"cd {shlex.quote(bundle)}", f"exec {_worker_command(self.target, bundle)}"])
        script_name = "submit.sh"
        log_name = "stdout.log"
        script = task.bundle / script_name
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script.chmod(0o755)
        if self.target.host is None:
            log = (task.bundle / log_name).open("ab")
            try:
                process = subprocess.Popen(
                    ["bash", str(script)],
                    cwd=task.bundle,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            finally:
                log.close()
            return ExecutionHandle(
                task.task_id,
                self.target.name,
                "process",
                str(process.pid),
                str(task.bundle),
            )
        # The script was created after the initial deployment; upload it alone.
        uploaded = self.transport.copy(
            script,
            f"{self.target.host}:{bundle}/{script_name}",
        )
        if uploaded.returncode != 0:
            raise ExecutionError((uploaded.stderr or uploaded.stdout).strip())
        launch = """set -eo pipefail
bundle=$1
cd "$bundle"
if [ -f execution.json ] && grep -q '"state": "COMPLETED"' execution.json; then
  cat worker.pid 2>/dev/null || echo completed
  exit 0
fi
if [ -f worker.pid ] && kill -0 "$(cat worker.pid)" 2>/dev/null && \
   ps -p "$(cat worker.pid)" -o args= | grep -F -- "$bundle" >/dev/null; then
  cat worker.pid
  exit 0
fi
nohup setsid bash "$2" > "$3" 2>&1 < /dev/null &
echo $! > worker.pid
cat worker.pid
"""
        completed = self.transport.run_script(
            launch,
            bundle,
            script_name,
            log_name,
        )
        if completed.returncode != 0:
            raise ExecutionError((completed.stderr or completed.stdout).strip())
        pid = completed.stdout.strip().splitlines()[-1]
        return ExecutionHandle(
            task.task_id,
            self.target.name,
            "process",
            pid,
            str(task.bundle),
            remote_bundle=bundle,
        )

    def inspect(self, handle: ExecutionHandle) -> ExecutionStatus:
        if self.target.host is None:
            path = Path(handle.local_bundle) / "execution.json"
            file_status = _local_execution_status(path)
            if file_status is not None:
                return file_status
            try:
                pid = int(handle.execution_id)
                owned_child = True
                try:
                    waited, _ = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    owned_child = False
                    waited = 0
                if waited == pid:
                    file_status = _wait_for_local_execution_status(path)
                    if file_status is not None:
                        return file_status
                    return ExecutionStatus("failed", "worker exited without a result")
                os.kill(pid, 0)
                if owned_child:
                    return ExecutionStatus("running")
                if not _pid_matches_bundle(pid, handle.local_bundle):
                    file_status = _wait_for_local_execution_status(path)
                    if file_status is not None:
                        return file_status
                    return ExecutionStatus(
                        "failed", "worker pid no longer belongs to this task"
                    )
                return ExecutionStatus("running")
            except (OSError, ValueError):
                file_status = _wait_for_local_execution_status(path)
                if file_status is not None:
                    return file_status
                return ExecutionStatus("failed", "worker exited without a result")
        script = """set -eo pipefail
bundle=$1
if [ -f "$bundle/execution.json" ]; then cat "$bundle/execution.json"; exit 0; fi
if [ -f "$bundle/worker.pid" ] && \
   kill -0 "$(cat "$bundle/worker.pid")" 2>/dev/null && \
   ps -p "$(cat "$bundle/worker.pid")" -o args= | grep -F -- "$bundle" >/dev/null; then
  echo '{"state":"RUNNING"}'
else
  echo '{"state":"FAILED","error":"remote worker exited without a result"}'
fi
"""
        completed = self.transport.run_script(
            script,
            str(handle.remote_bundle),
        )
        if completed.returncode != 0:
            return ExecutionStatus("unknown", (completed.stderr or completed.stdout).strip())
        value = json.loads(completed.stdout)
        if value.get("state") == "COMPLETED":
            return ExecutionStatus("completed")
        if value.get("state") == "FAILED":
            return ExecutionStatus("failed", str(value.get("error", "worker failed")))
        return ExecutionStatus("running")

    def collect(self, handle: ExecutionHandle) -> Path:
        return self.transport.collect(handle)

    def cancel(self, handle: ExecutionHandle) -> ExecutionStatus:
        status = self.inspect(handle)
        if status.terminal:
            return status
        try:
            pid = int(handle.execution_id)
        except ValueError as error:
            raise ExecutionError(
                f"process execution has an invalid pid: {handle.execution_id}"
            ) from error
        if self.target.host is None:
            if not _pid_matches_bundle(pid, handle.local_bundle):
                raise ExecutionError(
                    "refusing to cancel a process that no longer belongs to "
                    "this workflow task"
                )
            try:
                process_group = os.getpgid(pid)
                if process_group != pid:
                    raise ExecutionError(
                        "refusing to cancel a process outside its own process group"
                    )
                os.killpg(process_group, signal.SIGTERM)
            except ProcessLookupError:
                return ExecutionStatus("failed", "process already exited")
            if not _wait_for_process_exit(pid, 3.0):
                if not _pid_matches_bundle(pid, handle.local_bundle):
                    raise ExecutionError(
                        "process identity changed while waiting for cancellation"
                    )
                os.killpg(process_group, signal.SIGKILL)
                if not _wait_for_process_exit(pid, 2.0):
                    raise ExecutionError(
                        f"process group {pid} did not exit after SIGKILL"
                    )
            return ExecutionStatus("cancelled", f"sent SIGTERM to process group {pid}")
        bundle = str(handle.remote_bundle)
        script = """set -eo pipefail
bundle=$1
pid=$2
kill -0 "$pid" 2>/dev/null || exit 3
ps -p "$pid" -o args= | grep -F -- "$bundle" >/dev/null
pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
[ "$pgid" = "$pid" ]
kill -TERM -- "-$pgid"
for _ in $(seq 1 30); do
  kill -0 "$pid" 2>/dev/null || exit 0
  sleep 0.1
done
ps -p "$pid" -o args= | grep -F -- "$bundle" >/dev/null
kill -KILL -- "-$pgid"
for _ in $(seq 1 20); do
  kill -0 "$pid" 2>/dev/null || exit 0
  sleep 0.1
done
exit 4
"""
        completed = self.transport.run_script(
            script,
            bundle,
            str(pid),
            timeout=15,
        )
        if completed.returncode == 3:
            return ExecutionStatus("failed", "remote process already exited")
        if completed.returncode == 4:
            raise ExecutionError(
                f"remote process group {pid} did not exit after SIGKILL"
            )
        if completed.returncode != 0:
            raise ExecutionError((completed.stderr or completed.stdout).strip())
        return ExecutionStatus(
            "cancelled", f"sent SIGTERM to remote process group {pid}"
        )


class SlurmExecutor:
    """Submit one stage to a local or SSH-accessed Slurm controller."""

    def __init__(self, target: ExecutionTarget, runner: CommandRunner = subprocess.run):
        self.target = target
        self.transport = ExecutionTransport(target, runner)

    @staticmethod
    def _job_name(task_id: str) -> str:
        return f"nt-{task_id}"

    def _worker_status(self, bundle: str) -> ExecutionStatus | None:
        if self.target.host is None:
            path = Path(bundle) / "execution.json"
            if not path.is_file():
                return None
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        else:
            script = """set -eo pipefail
bundle=$1
test -f "$bundle/execution.json" || exit 3
cat "$bundle/execution.json"
"""
            completed = self.transport.run_script(
                script,
                bundle,
            )
            if completed.returncode != 0:
                return None
            try:
                value = json.loads(completed.stdout)
            except json.JSONDecodeError:
                return None
        state = value.get("state")
        if state == "COMPLETED":
            return ExecutionStatus("completed", "worker result complete")
        if state == "FAILED":
            return ExecutionStatus("failed", str(value.get("error", "worker failed")))
        if state == "RUNNING":
            return ExecutionStatus("running", "worker running")
        return None

    def _find_job(self, name: str, bundle: str) -> str | None:
        # Only active jobs are safe to recover by deterministic job name.
        # Historical sacct rows can outlive a deleted/reprepared workflow and
        # therefore cannot prove ownership of the current task bundle.
        args = ["squeue", "--noheader", "--name", name, "--format", "%A|%j"]
        completed = self.transport.run(
            [bundle, *args] if self.target.host else args,
            cwd=Path(bundle) if not self.target.host else None,
        )
        if completed.returncode != 0:
            return None
        for line in completed.stdout.splitlines():
            columns = line.strip().split("|")
            if len(columns) >= 2 and columns[0].isdigit() and columns[1] == name:
                return columns[0]
        return None

    def launch(self, task: StageTask) -> ExecutionHandle:
        if os.environ.get("SLURM_JOB_ID") and self.target.host is None:
            raise ExecutionError("workflow controller must run on the Slurm login node")
        _prepare_setup(self.target, task)
        script_name = "submit.sh"
        output_name = "stdout-%j.log"
        script = task.bundle / script_name
        bundle = self.transport.deploy(task)
        recovered_worker = self._worker_status(bundle)
        if recovered_worker is not None and recovered_worker.terminal:
            return ExecutionHandle(
                task.task_id,
                self.target.name,
                "slurm",
                f"recovered-{recovered_worker.state}",
                str(task.bundle),
                remote_bundle=bundle if self.target.host else None,
                metadata={
                    "job_name": self._job_name(task.task_id),
                    "recovered_from_worker_result": True,
                },
            )
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self._job_name(task.task_id)}",
            f"#SBATCH --output={bundle}/{output_name}",
            f"#SBATCH --time={self.target.time}",
            f"#SBATCH --partition={self.target.partition}",
        ]
        if self.target.qos:
            lines.append(f"#SBATCH --qos={self.target.qos}")
        if self.target.cpus_per_task is not None:
            lines.append(f"#SBATCH --cpus-per-task={self.target.cpus_per_task}")
        if self.target.gpus_per_node is not None:
            lines.append(f"#SBATCH --gpus-per-node={self.target.gpus_per_node}")
        for directive in self.target.directives:
            lines.append(
                directive if directive.startswith("#SBATCH ") else f"#SBATCH {directive}"
            )
        lines.extend(["", "set -eo pipefail"])
        setup = _setup_line(self.target, bundle)
        if setup:
            lines.append(setup)
        if self.target.cpus_per_task is not None:
            lines.append('export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"')
        for key, value in self.target.environment.items():
            lines.append(f"export {key}={shlex.quote(value)}")
        lines.extend([f"cd {shlex.quote(bundle)}", _worker_command(self.target, bundle), ""])
        script.write_text("\n".join(lines), encoding="utf-8")
        if self.target.host:
            uploaded = self.transport.copy(
                script,
                f"{self.target.host}:{bundle}/{script_name}",
            )
            if uploaded.returncode != 0:
                raise ExecutionError((uploaded.stderr or uploaded.stdout).strip())
        name = self._job_name(task.task_id)
        existing = self._find_job(name, bundle)
        if existing is not None:
            return ExecutionHandle(
                task.task_id,
                self.target.name,
                "slurm",
                existing,
                str(task.bundle),
                remote_bundle=bundle if self.target.host else None,
                metadata={"job_name": name},
            )
        args = ["sbatch", "--parsable", script_name]
        completed = self.transport.run(
            [bundle, *args] if self.target.host else args,
            cwd=task.bundle if not self.target.host else None,
        )
        if completed.returncode != 0:
            raise ExecutionError((completed.stderr or completed.stdout).strip())
        job_id = completed.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            recovered = self._find_job(name, bundle)
            if recovered is None:
                raise ExecutionError(
                    f"Slurm accepted an unparseable submission for {name}: {completed.stdout.strip()}"
                )
            job_id = recovered
        return ExecutionHandle(
            task.task_id,
            self.target.name,
            "slurm",
            job_id,
            str(task.bundle),
            remote_bundle=bundle if self.target.host else None,
            metadata={"job_name": name},
        )

    def inspect(self, handle: ExecutionHandle) -> ExecutionStatus:
        bundle = handle.remote_bundle or handle.local_bundle
        worker_status = self._worker_status(bundle)
        if worker_status is not None and worker_status.terminal:
            return worker_status
        squeue = ["squeue", "--noheader", "--jobs", handle.execution_id, "--format", "%T"]
        completed = self.transport.run(
            [bundle, *squeue] if self.target.host else squeue,
            cwd=Path(bundle) if not self.target.host else None,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            state = _slurm_state(completed.stdout.strip().splitlines()[0])
            if state in _ACTIVE:
                return ExecutionStatus("running", state)
        sacct = [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            handle.execution_id,
            "--format",
            "State,ExitCode",
        ]
        completed = self.transport.run(
            [bundle, *sacct] if self.target.host else sacct,
            cwd=Path(bundle) if not self.target.host else None,
        )
        if completed.returncode != 0:
            return ExecutionStatus("unknown", (completed.stderr or completed.stdout).strip())
        rows = [line.split("|") for line in completed.stdout.splitlines() if line.strip()]
        if not rows:
            return ExecutionStatus("unknown", "Slurm has no accounting record yet")
        state = _slurm_state(rows[0][0])
        exit_code = rows[0][1] if len(rows[0]) > 1 else ""
        if state in _TERMINAL_SUCCESS and exit_code.startswith("0:0"):
            return ExecutionStatus("completed", state)
        if state in _TERMINAL_FAILURE or state in _TERMINAL_SUCCESS:
            return ExecutionStatus("failed", f"Slurm {state} exit={exit_code}")
        return ExecutionStatus("running", state)

    def collect(self, handle: ExecutionHandle) -> Path:
        return self.transport.collect(handle)

    def cancel(self, handle: ExecutionHandle) -> ExecutionStatus:
        status = self.inspect(handle)
        if status.terminal:
            return status
        if not handle.execution_id.isdigit():
            raise ExecutionError(
                f"Slurm execution has an invalid job id: {handle.execution_id}"
            )
        bundle = handle.remote_bundle or handle.local_bundle
        args = ["scancel", handle.execution_id]
        completed = self.transport.run(
            [bundle, *args] if self.target.host else args,
            cwd=Path(bundle) if not self.target.host else None,
        )
        if completed.returncode != 0:
            refreshed = self.inspect(handle)
            if refreshed.terminal:
                return refreshed
            raise ExecutionError((completed.stderr or completed.stdout).strip())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            refreshed = self.inspect(handle)
            if refreshed.state == "completed":
                return refreshed
            if refreshed.state == "failed":
                if "CANCELLED" in refreshed.detail.upper():
                    return ExecutionStatus(
                        "cancelled",
                        f"Slurm job {handle.execution_id} is cancelled",
                    )
                return refreshed
            time.sleep(0.2)
        return ExecutionStatus(
            "cancelling",
            f"Slurm accepted cancellation for job {handle.execution_id}; "
            "terminal state is not confirmed yet",
        )


def executor_for(
    target: ExecutionTarget, runner: CommandRunner = subprocess.run
) -> StageExecutor:
    if target.executor == "process":
        return ProcessExecutor(target, runner)
    if target.executor == "slurm":
        return SlurmExecutor(target, runner)
    raise ExecutionError(f"unsupported executor: {target.executor}")


__all__ = [
    "ExecutionError",
    "ExecutionHandle",
    "ExecutionStatus",
    "ExecutionTarget",
    "ExecutionTransport",
    "ProcessExecutor",
    "SlurmExecutor",
    "StageExecutor",
    "StageTask",
    "build_stage_task",
    "executor_for",
    "load_stage_result",
    "run_stage_worker",
]
