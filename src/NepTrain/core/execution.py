"""Portable stage tasks and execution adapters for persistent workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import fcntl
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
import tempfile
import time
import traceback
from typing import Any, Callable, Mapping, Protocol, Sequence

from ase.io import read as ase_read
from ase.io import write as ase_write

from .content_addressing import canonical_sha256, file_sha256
from .iteration import GenerationPlan, StageContext, StageOutcome
from .persistence import atomic_write_json
from .sampling_route import load_sampling_routes
from .slurm import (
    ACTIVE_STATES as _ACTIVE,
    FAILURE_STATES as _TERMINAL_FAILURE,
    SUCCESS_STATES as _TERMINAL_SUCCESS,
    SlurmScript,
    SlurmSubmissionError,
    SlurmSubmissionThrottled,
    query_job,
    render_script,
    setup_line,
    submit_job,
)


class ExecutionError(RuntimeError):
    """Raised when a stage cannot be transported, launched, or collected."""


class PermanentExecutionError(ExecutionError):
    """Raised when retrying the same execution operation cannot succeed."""


class SubmissionDeferred(ExecutionError):
    """Raised when a scheduler temporarily refuses additional submissions."""


def _failure_kind(detail: str) -> str | None:
    normalized = detail.lower()
    if any(
        marker in normalized
        for marker in (
            "out of memory",
            "out_of_memory",
            "oom-kill",
            "oom kill",
            "cannot allocate memory",
            "std::bad_alloc",
            "cuda error: out of memory",
        )
    ):
        return "out_of_memory"
    if "scf did not converge" in normalized:
        return "non_convergence"
    return None


_STAGE_CONFIG_PATH_FIELDS = {
    "train": ("training.config_path", "training.test_path"),
    "explore": (),
    "select": (),
    "label": (
        "labeling.input_path",
        "labeling.potcar_manifest_path",
        "labeling.resource_manifest_path",
        "labeling.model_path",
    ),
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
        "label_provenance",
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
_PORTABLE_ARTIFACT_STEMS = {
    "model": "nep",
    "activated_model": "nep",
    "retrained_model": "candidate-nep",
    "training_input": "base-train",
    "training_set": "train",
    "selected_input": "selected",
    "labeled": "labels",
    "checkpoint": "checkpoint",
    "activated_checkpoint": "checkpoint",
    "retrained_checkpoint": "candidate-checkpoint",
}
_STAGE_DIRECTORY_LABELS = {
    "train": "train",
    "explore": "md",
    "select": "select",
    "label": "label",
    "diagnose": "diagnose",
    "merge": "merge",
    "retrain": "retrain",
    "evaluate": "evaluate",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tree_hash(path: Path) -> str:
    if path.is_file():
        return file_sha256(path)
    records = [
        (str(item.relative_to(path)), file_sha256(item))
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    return canonical_sha256(records)


def _copy_input(source: Path, target: Path) -> None:
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return
    if source.is_dir():
        shutil.copytree(source, target, copy_function=shutil.copy2)
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
    labeling_resource_path: str | None = None
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
        labeling_resource_path = value.get("labeling_resource_path")
        if labeling_resource_path is not None and any(
            character in str(labeling_resource_path) for character in "\r\n"
        ):
            raise ExecutionError(
                f"execution target {name}.labeling_resource_path cannot "
                "contain newlines"
            )
        if labeling_resource_path is not None:
            resource_text = str(labeling_resource_path)
            if not (
                Path(resource_text).is_absolute()
                or resource_text.startswith("~/")
            ):
                raise ExecutionError(
                    f"execution target {name}.labeling_resource_path must be "
                    "absolute or start with ~/"
                )
            if ".." in Path(resource_text.removeprefix("~/")).parts:
                raise ExecutionError(
                    f"execution target {name}.labeling_resource_path cannot "
                    "contain parent traversal"
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
            labeling_resource_path=(
                str(labeling_resource_path)
                if labeling_resource_path is not None
                else None
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
    failure_kind: str | None = None

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
    workflow_instance_id: str | None = None,
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
        "workflow_instance_id": workflow_instance_id or workflow_id,
        "generation": generation,
        "stage": stage,
        "attempt": attempt,
        "plan_sha256": plan.sha256,
        "target": target.name,
        "sampling_routes": route_identities,
        "stage_input": dict(stage_input or {}),
    }
    tasks_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".stage-task-building-", dir=tasks_dir)
    )
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
    if stage == "label" and target.labeling_resource_path:
        portable_config.setdefault("labeling", {})["resource_path"] = (
            target.labeling_resource_path
        )
    elif stage == "label":
        resource_path = portable_config.get("labeling", {}).get(
            "resource_path"
        )
        if resource_path:
            portable_config["labeling"]["resource_path"] = str(
                _resolve_path(resource_path, workflow_root)
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
    } | {
        "labeling.resource_path",
        "evaluation.validation_path",
        "training.initial_path",
    }
    retained_path_fields = set(path_fields)
    if stage == "label":
        retained_path_fields.add("labeling.resource_path")
    for dotted in sorted(all_path_fields - retained_path_fields):
        _delete_dotted(portable_config, dotted)
    if stage not in {"explore", "evaluate"}:
        portable_config.get("sampling", {}).pop("routes", None)
    for dotted in sorted(path_fields):
        value = _get_dotted(portable_config, dotted)
        if value in {None, "", "auto"}:
            continue
        if (
            dotted == "labeling.resource_path"
            and target.labeling_resource_path
        ):
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

    artifact_destinations: dict[Path, str] = {}

    def copy_artifacts(
        values: Mapping[str, Path], required: Sequence[str]
    ) -> dict[str, str]:
        copied = {}
        for name in required:
            if name not in values:
                continue
            _validate_artifact_name(name)
            source = values[name]
            source = Path(source).resolve()
            stem = _PORTABLE_ARTIFACT_STEMS.get(
                name,
                name.replace("_", "-"),
            )
            destination = inputs / f"{stem}{source.suffix}"
            previous_name = artifact_destinations.get(destination)
            if previous_name is not None:
                raise ExecutionError(
                    f"portable task artifacts {previous_name} and {name} "
                    f"both map to {destination.name}"
                )
            artifact_destinations[destination] = name
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

    descriptor: dict[str, Any] = {
        "protocol": "neptrain.stage-task.v3",
        "identity": identity,
        "config": portable_config,
        "target": {
            **asdict(target),
            "setup_script": (
                "target-setup.sh"
                if (temporary / "target-setup.sh").is_file()
                else target.setup_script
            ),
        },
        "initial_training": initial_value,
        "plan": asdict(plan),
        "stage_input": dict(stage_input or {}),
        "artifacts": copy_artifacts(
            context.artifacts, _STAGE_ARTIFACTS[stage]
        ),
        "previous_artifacts": copy_artifacts(
            context.previous_artifacts,
            _STAGE_PREVIOUS_ARTIFACTS[stage],
        ),
    }
    records = [
        {
            "path": str(path.relative_to(temporary)),
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(temporary.rglob("*"))
        if path.is_file() and path.name != "task.json"
    ]
    descriptor["files"] = records
    spec_sha256 = canonical_sha256(_task_content_spec(descriptor))
    task_id = spec_sha256[:24]
    descriptor["task_id"] = task_id
    descriptor["spec_sha256"] = spec_sha256
    descriptor["created_at"] = _now()
    directory_name = _task_directory_name(
        generation=generation,
        stage=stage,
        attempt=attempt,
        task_id=task_id,
        stage_input=dict(stage_input or {}),
    )
    bundle = tasks_dir / directory_name
    atomic_write_json(temporary / "task.json", descriptor)
    if bundle.exists():
        try:
            existing = _verify_task_bundle(bundle)
            if existing["spec_sha256"] != spec_sha256:
                raise ExecutionError(f"task id collision at {bundle}")
        finally:
            shutil.rmtree(temporary)
        return StageTask(
            task_id, workflow_id, generation, stage, target.name, bundle
        )
    temporary.replace(bundle)
    return StageTask(task_id, workflow_id, generation, stage, target.name, bundle)


def _task_content_spec(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: descriptor.get(key)
        for key in (
            "identity",
            "config",
            "target",
            "initial_training",
            "plan",
            "stage_input",
            "artifacts",
            "previous_artifacts",
            "files",
        )
    }


def _verify_task_bundle(bundle: Path) -> dict[str, Any]:
    descriptor_path = bundle / "task.json"
    if not descriptor_path.is_file():
        raise ExecutionError("stage task is missing task.json")
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionError(f"stage task descriptor is unreadable: {descriptor_path}") from error
    if descriptor.get("protocol") != "neptrain.stage-task.v3":
        raise ExecutionError(
            "unsupported stage task descriptor; retry the stage to create a "
            "content-addressed v3 task"
        )
    for record in descriptor.get("files", []):
        path = _safe_relative(bundle, record["path"], "task manifest path")
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size"])
            or file_sha256(path) != record["sha256"]
        ):
            raise ExecutionError(f"stage task input drifted: {path}")
    expected = canonical_sha256(_task_content_spec(descriptor))
    if descriptor.get("spec_sha256") != expected:
        raise ExecutionError("stage task content identity does not match its manifest")
    if descriptor.get("task_id") != expected[:24]:
        raise ExecutionError("stage task id does not match its content identity")
    return descriptor


def verify_stage_task(bundle_path: str | Path) -> None:
    """Verify a complete stage bundle without executing it."""

    _verify_task_bundle(Path(bundle_path).expanduser().resolve())


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
            try:
                load_stage_result(bundle)
            except ExecutionError:
                pass
            else:
                atomic_write_json(
                    execution_path,
                    {
                        "task_id": descriptor["task_id"],
                        "task_spec_sha256": descriptor["spec_sha256"],
                        "state": "COMPLETED",
                        "pid": os.getpid(),
                        "completed_at": _now(),
                        "recovered": True,
                    },
                )
                return 0
            atomic_write_json(
                execution_path,
                {
                    "task_id": descriptor["task_id"],
                    "task_spec_sha256": descriptor["spec_sha256"],
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

            work_dir = bundle / f".output-building-{os.getpid()}"
            if work_dir.exists():
                shutil.rmtree(work_dir)
            work_dir.mkdir()
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
                relative_output = destination.relative_to(work_dir)
                result_artifacts[name] = {
                    "path": str(Path("output") / relative_output),
                    "sha256": file_sha256(destination),
                    "size": destination.stat().st_size,
                }
            published_output = bundle / "output"
            if published_output.exists():
                archive = (
                    bundle
                    / "attempts"
                    / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
                )
                archive.mkdir(parents=True)
                published_output.replace(archive / "output")
                for name in ("result.json", "execution.json"):
                    previous = bundle / name
                    if previous.exists():
                        previous.replace(archive / name)
            work_dir.replace(published_output)
            result = {
                "protocol": "neptrain.stage-result.v3",
                "task_id": descriptor["task_id"],
                "task_spec_sha256": descriptor["spec_sha256"],
                "workflow_id": descriptor["identity"]["workflow_id"],
                "workflow_instance_id": descriptor["identity"][
                    "workflow_instance_id"
                ],
                "generation": descriptor["identity"]["generation"],
                "stage": descriptor["identity"]["stage"],
                "plan_sha256": descriptor["identity"]["plan_sha256"],
                "completed_at": _now(),
                "artifacts": result_artifacts,
                "metrics": dict(outcome.metrics),
            }
            atomic_write_json(bundle / "result.json", result)
            atomic_write_json(
                execution_path,
                {
                    "task_id": descriptor["task_id"],
                    "task_spec_sha256": descriptor["spec_sha256"],
                    "state": "COMPLETED",
                    "pid": os.getpid(),
                    "completed_at": _now(),
                },
            )
            return 0
        except Exception as error:
            atomic_write_json(
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


def _load_stage_result_payload(
    result_root: Path,
    descriptor: Mapping[str, Any],
) -> tuple[dict[str, Any], StageOutcome]:
    path = result_root / "result.json"
    if not path.is_file():
        raise ExecutionError(f"stage result does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionError(f"stage result descriptor is unreadable: {path}") from error
    if value.get("protocol") != "neptrain.stage-result.v3":
        raise ExecutionError("unsupported stage result protocol")
    expected = {
        "task_id": descriptor["task_id"],
        "task_spec_sha256": descriptor["spec_sha256"],
        "workflow_id": descriptor["identity"]["workflow_id"],
        "workflow_instance_id": descriptor["identity"]["workflow_instance_id"],
        "generation": descriptor["identity"]["generation"],
        "stage": descriptor["identity"]["stage"],
        "plan_sha256": descriptor["identity"]["plan_sha256"],
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ExecutionError(
                f"stage result {name} does not match the task descriptor"
            )
    artifacts = {}
    for name, record in value.get("artifacts", {}).items():
        _validate_artifact_name(str(name))
        artifact = _safe_relative(
            result_root, record["path"], "result artifact path"
        )
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(record["size"])
            or file_sha256(artifact) != record["sha256"]
        ):
            raise ExecutionError(f"stage result artifact drifted: {artifact}")
        artifacts[name] = artifact
    return value, StageOutcome(artifacts=artifacts, metrics=value.get("metrics", {}))


def load_stage_result(bundle_path: str | Path) -> tuple[dict[str, Any], StageOutcome]:
    bundle = Path(bundle_path).expanduser().resolve()
    descriptor = _verify_task_bundle(bundle)
    return _load_stage_result_payload(bundle, descriptor)


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
            "-o",
            "BatchMode=yes",
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

    def resolve_remote_path(self, path: str | Path) -> str:
        """Return an absolute remote path, expanding a leading ``~/`` safely."""

        raw = str(path)
        if raw.startswith("/"):
            return raw
        if not raw.startswith("~/"):
            raise ExecutionError(
                "remote path must be absolute or start with ~/"
            )
        completed = self.run_script(
            """set -eo pipefail
path=$1
case "$path" in
  '~/'*) path="$HOME/${path:2}" ;;
  *) exit 2 ;;
esac
case "$path" in
  /*) printf '%s\n' "$path" ;;
  *) exit 2 ;;
esac
""",
            raw,
            check=True,
        )
        resolved = completed.stdout.strip()
        if (
            not resolved.startswith("/")
            or "\n" in resolved
            or "\r" in resolved
        ):
            raise ExecutionError(
                f"remote target {self.target.name} returned an invalid path"
            )
        return resolved

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
        command = ["scp", "-o", "BatchMode=yes"]
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
        remote_base = self.resolve_remote_path(remote_root).rstrip("/")
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
                "-o",
                "BatchMode=yes",
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
        descriptor = _verify_task_bundle(task.bundle)
        if descriptor["task_id"] != task.task_id:
            raise ExecutionError("stage task object does not match its bundle")
        root = self.resolve_remote_path(str(self.target.work_root))
        remote_name = task.bundle.name
        remote_bundle = (
            f"{root.rstrip('/')}/{task.workflow_id}/jobs/{remote_name}"
        )
        token = f"{os.getpid()}-{time.time_ns()}"
        remote_archive = (
            f"{root.rstrip('/')}/{task.workflow_id}/incoming/"
            f"{remote_name}-{token}.tar.gz"
        )
        descriptor_sha256 = file_sha256(task.descriptor)
        verify_command = shlex.split(self.target.command)
        setup = """set -eo pipefail
root=$1
workflow=$2
name=$3
expected=$4
shift 4
command=("$@")
verify_bundle() {
  bundle=$1
  if [ -f "$bundle/target-setup.sh" ]; then
    (source "$bundle/target-setup.sh"; "${command[@]}" stage-verify "$bundle")
  else
    "${command[@]}" stage-verify "$bundle"
  fi
}
case "$root" in
  '~/'*) root="$HOME/${root:2}" ;;
  /*) ;;
  *) exit 2 ;;
esac
mkdir -p "$root/$workflow/incoming" "$root/$workflow/jobs"
destination="$root/$workflow/jobs/$name"
if [ -e "$destination" ]; then
  if [ ! -f "$destination/task.json" ]; then
    printf '%s\\n' INCOMPLETE
    exit 0
  fi
  actual=$(sha256sum "$destination/task.json" | cut -d' ' -f1)
  if [ "$actual" = "$expected" ]; then
    if verify_bundle "$destination" >/dev/null 2>&1; then
      printf '%s\\n' READY
    else
      printf '%s\\n' INCOMPLETE
    fi
    exit 0
  fi
  echo "remote task descriptor conflicts with local content identity: $destination" >&2
  exit 17
fi
printf '%s\\n' MISSING
"""
        completed = self.run_script(
            setup,
            root,
            task.workflow_id,
            remote_name,
            descriptor_sha256,
            *verify_command,
            timeout=300,
        )
        if completed.returncode != 0:
            raise ExecutionError((completed.stderr or completed.stdout).strip())
        if completed.stdout.strip().splitlines()[-1:] == ["READY"]:
            return remote_bundle
        archive = task.bundle.parent / f".{remote_name}.{token}.upload.tar.gz"
        try:
            with tarfile.open(archive, "w:gz") as handle:
                handle.add(task.bundle, arcname=remote_name)
            digest = file_sha256(archive)
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
archive_path=$4
expected_archive=$5
expected_descriptor=$6
token=$7
shift 7
command=("$@")
verify_bundle() {
  bundle=$1
  if [ -f "$bundle/target-setup.sh" ]; then
    (source "$bundle/target-setup.sh"; "${command[@]}" stage-verify "$bundle")
  else
    "${command[@]}" stage-verify "$bundle"
  fi
}
case "$root" in
  '~/'*) root="$HOME/${root:2}" ;;
  /*) ;;
  *) exit 2 ;;
esac
jobs="$root/$workflow/jobs"
destination="$jobs/$name"
lock="$jobs/.$name.deploy.lock"
if ! mkdir "$lock" 2>/dev/null; then
  if [ -f "$destination/task.json" ]; then
    actual=$(sha256sum "$destination/task.json" | cut -d' ' -f1)
    if [ "$actual" = "$expected_descriptor" ] &&
       verify_bundle "$destination" >/dev/null 2>&1; then
      exit 0
    fi
  fi
  echo "another deployment owns $destination; retry later" >&2
  exit 75
fi
temporary="$jobs/.$name.extracting-$token"
cleanup() {
  rm -f -- "$archive_path"
  rm -rf -- "$temporary"
  rmdir "$lock" 2>/dev/null || true
}
trap cleanup EXIT
actual=$(sha256sum "$archive_path" | cut -d' ' -f1)
[ "$actual" = "$expected_archive" ]
mkdir -p "$temporary"
tar -xzf "$archive_path" -C "$temporary" --strip-components=1
[ -f "$temporary/task.json" ]
actual=$(sha256sum "$temporary/task.json" | cut -d' ' -f1)
[ "$actual" = "$expected_descriptor" ]
verify_bundle "$temporary"
if [ -e "$destination" ]; then
  if [ -f "$destination/task.json" ]; then
    actual=$(sha256sum "$destination/task.json" | cut -d' ' -f1)
    if [ "$actual" = "$expected_descriptor" ] &&
       verify_bundle "$destination" >/dev/null 2>&1; then
      exit 0
    fi
    if [ "$actual" != "$expected_descriptor" ]; then
      echo "remote task descriptor conflicts with local content identity: $destination" >&2
      exit 17
    fi
  fi
  quarantine="$destination.incomplete.$(date +%Y%m%d-%H%M%S).$$"
  mv -- "$destination" "$quarantine"
fi
mv -- "$temporary" "$destination"
actual=$(sha256sum "$destination/task.json" | cut -d' ' -f1)
[ "$actual" = "$expected_descriptor" ]
verify_bundle "$destination"
"""
            completed = self.run_script(
                extract,
                root,
                task.workflow_id,
                remote_name,
                remote_archive,
                digest,
                descriptor_sha256,
                token,
                *verify_command,
                timeout=300,
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
        try:
            descriptor = _verify_task_bundle(local)
        except ExecutionError as error:
            raise PermanentExecutionError(str(error)) from error
        if handle.task_id != descriptor["task_id"]:
            raise PermanentExecutionError(
                "execution handle does not belong to the local task bundle"
            )
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{local.name}.collecting-",
                dir=local.parent,
            )
        )
        try:
            self.fetch_paths(
                remote,
                ("result.json", "execution.json", "output"),
                temporary,
            )
            for name in ("result.json", "execution.json"):
                if not (temporary / name).is_file():
                    raise PermanentExecutionError(
                        f"remote stage result is missing required file: {name}"
                    )
            try:
                _load_stage_result_payload(temporary, descriptor)
            except ExecutionError as error:
                raise PermanentExecutionError(str(error)) from error
            try:
                execution = json.loads(
                    (temporary / "execution.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                raise PermanentExecutionError(
                    "remote stage execution descriptor is unreadable"
                ) from error
            if (
                execution.get("state") != "COMPLETED"
                or execution.get("task_id") != descriptor["task_id"]
                or execution.get("task_spec_sha256")
                != descriptor["spec_sha256"]
            ):
                raise PermanentExecutionError(
                    "remote execution state does not belong to the requested task"
                )

            archive = (
                local
                / "attempts"
                / (
                    "remote-collection-"
                    + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
                )
            )
            previous = [
                local / name
                for name in ("output", "result.json", "execution.json")
                if (local / name).exists()
            ]
            if previous:
                archive.mkdir(parents=True)
                for path in previous:
                    path.replace(archive / path.name)
            (temporary / "output").replace(local / "output")
            (temporary / "result.json").replace(local / "result.json")
            (temporary / "execution.json").replace(local / "execution.json")
            try:
                load_stage_result(local)
            except ExecutionError as error:
                raise PermanentExecutionError(str(error)) from error
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return local


def _prepare_setup(target: ExecutionTarget, task: StageTask) -> None:
    if not target.setup_script:
        return
    candidate = Path(target.setup_script).expanduser()
    if candidate.is_file() and not (task.bundle / "target-setup.sh").is_file():
        shutil.copy2(candidate, task.bundle / "target-setup.sh")


def _worker_command(target: ExecutionTarget, bundle: str) -> str:
    return shlex.join([*shlex.split(target.command), "stage-worker", bundle])


def _pid_matches_bundle(pid: int, bundle: str) -> bool:
    completed = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "command="],
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
    except (OSError, json.JSONDecodeError) as error:
        return ExecutionStatus(
            "failed",
            f"execution descriptor is unreadable: {error}",
        )
    if not isinstance(value, Mapping):
        return ExecutionStatus(
            "failed",
            "execution descriptor must contain a JSON object",
        )
    state = value.get("state")
    if state == "COMPLETED":
        return ExecutionStatus("completed")
    if state == "FAILED":
        detail = str(value.get("error", "worker failed"))
        return ExecutionStatus("failed", detail, _failure_kind(detail))
    if state == "RUNNING":
        return None
    return ExecutionStatus(
        "failed",
        f"execution descriptor has unsupported state {state!r}",
    )


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
            try:
                value = json.loads(local_execution.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                value = {}
            state = value.get("state") if isinstance(value, Mapping) else None
            pid = str(value.get("pid", "")) if isinstance(value, Mapping) else ""
            reusable = state == "COMPLETED" or (
                state == "RUNNING"
                and pid.isdigit()
                and _pid_matches_bundle(int(pid), str(task.bundle))
            )
            if reusable:
                return ExecutionHandle(
                    task.task_id,
                    self.target.name,
                    "process",
                    pid or "completed",
                    str(task.bundle),
                )
            archive = (
                task.bundle
                / "attempts"
                / (
                    "process-launch-"
                    + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
                )
            )
            archive.mkdir(parents=True, exist_ok=False)
            local_execution.replace(archive / "execution.json")
        lines = ["#!/bin/bash", "set -eo pipefail"]
        setup = setup_line(
            self.target.setup_script,
            local=self.target.host is None,
            packaged_remote_path=f"{bundle}/target-setup.sh",
        )
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
   ps -ww -p "$(cat worker.pid)" -o args= | grep -F -- "$bundle" >/dev/null; then
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
   ps -ww -p "$(cat "$bundle/worker.pid")" -o args= | grep -F -- "$bundle" >/dev/null; then
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
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            return ExecutionStatus(
                "failed",
                f"remote execution descriptor is unreadable: {error}",
            )
        if not isinstance(value, Mapping):
            return ExecutionStatus(
                "failed",
                "remote execution descriptor must contain a JSON object",
            )
        if value.get("state") == "COMPLETED":
            return ExecutionStatus("completed")
        if value.get("state") == "FAILED":
            detail = str(value.get("error", "worker failed"))
            return ExecutionStatus("failed", detail, _failure_kind(detail))
        if value.get("state") == "RUNNING":
            return ExecutionStatus("running")
        return ExecutionStatus(
            "failed",
            "remote execution descriptor has unsupported state "
            f"{value.get('state')!r}",
        )

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
ps -ww -p "$pid" -o args= | grep -F -- "$bundle" >/dev/null
pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
[ "$pgid" = "$pid" ]
kill -TERM -- "-$pgid"
for _ in $(seq 1 30); do
  kill -0 "$pid" 2>/dev/null || exit 0
  sleep 0.1
done
ps -ww -p "$pid" -o args= | grep -F -- "$bundle" >/dev/null
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
            detail = str(value.get("error", "worker failed"))
            return ExecutionStatus("failed", detail, _failure_kind(detail))
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
        setup = setup_line(
            self.target.setup_script,
            local=self.target.host is None,
            packaged_remote_path=f"{bundle}/target-setup.sh",
        )
        script.write_text(
            render_script(
                SlurmScript(
                    job_name=self._job_name(task.task_id),
                    output_path=f"{bundle}/{output_name}",
                    workdir=bundle,
                    command=_worker_command(self.target, bundle),
                    partition=str(self.target.partition),
                    time_limit=self.target.time,
                    qos=self.target.qos,
                    cpus_per_task=self.target.cpus_per_task,
                    gpus_per_node=self.target.gpus_per_node,
                    directives=self.target.directives,
                    environment=self.target.environment,
                    setup_line=setup,
                )
            ),
            encoding="utf-8",
        )
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
        try:
            job_id = submit_job(
                lambda command: self.transport.run(
                    [bundle, *command] if self.target.host else command,
                    cwd=task.bundle if not self.target.host else None,
                ),
                script_name,
            )
        except SlurmSubmissionThrottled as error:
            raise SubmissionDeferred(str(error)) from error
        except SlurmSubmissionError as error:
            if not error.accepted:
                raise PermanentExecutionError(str(error)) from error
            recovered = self._find_job(name, bundle)
            if recovered is None:
                raise ExecutionError(
                    f"{error}; active job {name} could not be recovered"
                ) from error
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
        if worker_status is not None and worker_status.state == "completed":
            return worker_status
        observation = query_job(
            lambda command: self.transport.run(
                [bundle, *command] if self.target.host else command,
                cwd=Path(bundle) if not self.target.host else None,
            ),
            handle.execution_id,
        )
        if observation.state is None:
            return worker_status or ExecutionStatus(
                "unknown",
                observation.error or "Slurm has no accounting record yet",
            )
        state = observation.state
        exit_code = observation.exit_code
        if state in _ACTIVE:
            return ExecutionStatus("running", state)
        if worker_status is not None and worker_status.state == "failed":
            scheduler_detail = f"Slurm {state} exit={exit_code}"
            detail = worker_status.detail
            if scheduler_detail not in detail:
                detail = f"{detail}; {scheduler_detail}"
            return ExecutionStatus(
                "failed",
                detail,
                worker_status.failure_kind
                or ("out_of_memory" if state == "OUT_OF_MEMORY" else None),
            )
        if state in _TERMINAL_SUCCESS and exit_code.startswith("0:0"):
            return ExecutionStatus("completed", state)
        if state in _TERMINAL_FAILURE or state in _TERMINAL_SUCCESS:
            detail = f"Slurm {state} exit={exit_code}"
            return ExecutionStatus(
                "failed",
                detail,
                "out_of_memory" if state == "OUT_OF_MEMORY" else None,
            )
        return ExecutionStatus("running", state)

    def collect(self, handle: ExecutionHandle) -> Path:
        return self.transport.collect(handle)

    def collect_failure(self, handle: ExecutionHandle) -> Path:
        """Collect best-effort diagnostics without requiring a stage result."""

        local = Path(handle.local_bundle)
        if self.target.host is None:
            return local
        remote = str(handle.remote_bundle)
        discovered = self.transport.run_script(
            """set -eo pipefail
bundle=$1
cd "$bundle"
for path in .output-building-*; do
  test -e "$path" && printf '%s\\n' "$path"
done
""",
            remote,
        )
        building = []
        if discovered.returncode == 0:
            building = [
                line.strip()
                for line in discovered.stdout.splitlines()
                if line.strip().startswith(".output-building-")
                and "/" not in line.strip()
            ]
        evidence = local / "failure-evidence" / f"slurm-{handle.execution_id}"
        self.transport.fetch_paths(
            remote,
            (
                "execution.json",
                "submit.sh",
                f"stdout-{handle.execution_id}.log",
                *building,
            ),
            evidence,
        )
        return evidence

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
    "PermanentExecutionError",
    "ExecutionStatus",
    "ExecutionTarget",
    "ExecutionTransport",
    "ProcessExecutor",
    "SlurmExecutor",
    "SubmissionDeferred",
    "StageExecutor",
    "StageTask",
    "build_stage_task",
    "executor_for",
    "load_stage_result",
    "run_stage_worker",
    "verify_stage_task",
]
