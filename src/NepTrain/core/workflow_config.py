"""Classify project edits and adopt operational changes at controller restart."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any

from .config import ConfigError, load_config
from .persistence import atomic_write_json


class WorkflowConfigError(ValueError):
    pass


_RESOURCE_FIELDS = {
    "partition", "qos", "time", "cpus_per_task", "gpus_per_node",
    "memory_ladder", "directives",
}
_LAUNCH_FIELDS = {"command", "environment", "setup_script"}
_LAUNCH_SCOPE = "new task bundles; prepared and submitted tasks keep their launch settings"
_LAUNCH_WARNING = (
    "Launch settings can change the software or numerical environment. "
    "Existing results retain their original provenance and are not recomputed."
)
# Directives that only choose resources, without changing the worker command,
# environment, working directory, or output/collection contract.
_RESOURCE_DIRECTIVES = {
    "--nodes", "--ntasks", "--ntasks-per-node", "--cpus-per-task",
    "--mem", "--mem-per-cpu", "--mem-per-gpu", "--account", "--constraint",
    "--nodelist", "--exclude", "--reservation", "--hint", "--exclusive",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot_config(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep a hash-verified baseline without changing the editable project file."""
    path = Path(record["path"])
    data = path.read_bytes()
    if _sha(data) != record["sha256"]:
        raise WorkflowConfigError("Project changed while saving its configuration baseline; retry.")
    relative = Path(".neptrain/configurations") / f"{record['sha256']}.yaml"
    snapshot = path.parent / relative
    snapshot.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not snapshot.exists():
        # The caller holds the manifest lock, or is preparing a new workflow.
        with snapshot.open("xb") as stream:
            snapshot.chmod(0o600)
            stream.write(data)
    if _sha(snapshot.read_bytes()) != record["sha256"]:
        raise WorkflowConfigError(f"Configuration baseline is damaged: {snapshot}")
    return {**record, "snapshot": str(relative)}


def _differences(before: Any, after: Any, path: tuple[str, ...] = ()):
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        for key in sorted(before.keys() | after.keys()):
            yield from _differences(before.get(key), after.get(key), (*path, str(key)))
    elif (
        isinstance(before, list) and isinstance(after, list)
        and len(before) == len(after)
        and all(isinstance(item, Mapping) for item in [*before, *after])
    ):
        for index, (old, new) in enumerate(zip(before, after)):
            yield from _differences(old, new, (*path, str(index)))
    elif before != after:
        # Descend into additions/removals too, so adding a target cannot evade
        # the executor/host restrictions.
        if isinstance(before, Mapping) and after is None:
            yield from _differences(before, {}, path)
        elif before is None and isinstance(after, Mapping):
            yield from _differences({}, after, path)
        else:
            yield path, before, after


def _scope(path: tuple[str, ...], before: Any, after: Any) -> str | None:
    if path[:1] == ("notifications",):
        return "controller restart"
    if path == ("execution", "poll_interval"):
        return "controller restart"
    if path == ("labeling", "max_concurrent"):
        return "new submissions, including the current labeling group"
    if path == ("labeling", "structures_per_job"):
        return "new labeling groups"
    if len(path) >= 4 and path[:2] == ("execution", "targets") and path[3] in _LAUNCH_FIELDS:
        return _LAUNCH_SCOPE
    if len(path) == 4 and path[:2] == ("execution", "targets") and path[3] in _RESOURCE_FIELDS:
        if path[3] == "directives":
            for value in (before, after):
                for item in value or []:
                    try:
                        tokens = shlex.split(str(item).removeprefix("#SBATCH "))
                    except ValueError:
                        return None
                    if not tokens or tokens[0].split("=", 1)[0] not in _RESOURCE_DIRECTIVES:
                        return None
                    if any(
                        token.startswith("-") and token.split("=", 1)[0] not in _RESOURCE_DIRECTIVES
                        for token in tokens[1:]
                    ):
                        return None
        return "newly submitted jobs"
    return None


def inspect_config(record: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Read-only classification; never accept changes or rewrite hashes here."""
    path = Path(record["path"])
    try:
        current_hash = _sha(path.read_bytes())
    except OSError as error:
        raise WorkflowConfigError(f"Cannot read workflow project: {path}: {error}") from error
    snapshot_name = record.get("snapshot")
    if not snapshot_name:
        if current_hash == record["sha256"]:
            return ()
        raise WorkflowConfigError(
            "This workflow has no saved configuration baseline. Restore its original "
            "project.yaml, run workflow stop --keep-jobs with the updated NepTrain, "
            "then edit project.yaml and resume. Existing results need not be deleted."
        )
    snapshot = path.parent / snapshot_name
    if not snapshot.is_file() or _sha(snapshot.read_bytes()) != record["sha256"]:
        raise WorkflowConfigError(f"Configuration baseline is missing or damaged: {snapshot}")
    if current_hash == record["sha256"]:
        return ()
    try:
        before, _ = load_config(snapshot)
        after, _ = load_config(path)
    except ConfigError as error:
        raise WorkflowConfigError(f"Invalid edited project configuration: {error}") from error
    changes = []
    old_targets = before.get("execution", {}).get("targets", {})
    new_targets = after.get("execution", {}).get("targets", {})
    blocked = [
        f"execution.targets.{name}"
        for name in sorted(old_targets.keys() ^ new_targets.keys())
    ]
    for parts, old, new in _differences(before, after):
        scope = _scope(parts, old, new)
        field = ".".join(parts)
        if scope is None:
            blocked.append(field)
        else:
            secret = (
                parts[:1] == ("notifications",) and parts[-1] in {"secret", "webhook"}
            ) or (parts[:2] == ("execution", "targets") and parts[3:4] == ("environment",))
            changes.append({
                "field": field,
                "before": "[redacted]" if secret else old,
                "after": "[redacted]" if secret else new,
                "applies_to": scope,
                **({"warning": _LAUNCH_WARNING} if scope == _LAUNCH_SCOPE else {}),
            })
    if blocked:
        hint = (
            "Use workflow extend to change the generation budget. "
            if "workflow.max_model_generations" in blocked else ""
        )
        raise WorkflowConfigError(
            "These configuration changes cannot be applied to an existing workflow: "
            + ", ".join(blocked)
            + ". They may change scientific evidence or task execution identity. "
            + hint
            + "Restore these fields to continue this run, or prepare a new workflow "
            "with the revised settings and the existing merged training set. "
            "No edits were accepted; existing results are retained."
        )
    return tuple(changes)


def setup_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Describe the launch scripts referenced by the current editable project."""
    project = Path(manifest["config"]["path"])
    config, _ = load_config(project)
    records = []
    for name, target in config.get("execution", {}).get("targets", {}).items():
        value = target.get("setup_script")
        if not value or value == "auto":
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = project.parent / path
        try:
            data = path.read_bytes()
        except OSError as error:
            raise WorkflowConfigError(f"Cannot read launch script for target {name}: {path}: {error}") from error
        records.append({"role": f"execution_setup_{name}", "kind": "file", "path": str(path.resolve()), "sha256": _sha(data)})
    return records


def inspect_setup_changes(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    before = {r["role"]: r for r in manifest.get("dependencies", []) if r.get("role", "").startswith("execution_setup_")}
    after = {r["role"]: r for r in setup_records(manifest)}
    changes = []
    for role in sorted(before.keys() | after.keys()):
        old, new = before.get(role), after.get(role)
        if old and new and all(old[k] == new[k] for k in ("path", "sha256")):
            continue
        name = role.removeprefix("execution_setup_")
        changes.append({
            "field": f"execution.targets.{name}.setup_script.contents",
            "before": old["sha256"][:12] if old else None,
            "after": new["sha256"][:12] if new else None,
            "applies_to": _LAUNCH_SCOPE,
            "warning": _LAUNCH_WARNING,
        })
        # Older submissions may still source the editable project path. We
        # cannot rewrite Slurm's stored batch script or recover an overwritten
        # file on behalf of that job; surface this limitation without blocking.
        state_path = Path(manifest["config"]["path"]).parent / ".neptrain/controller.json"
        if old and state_path.is_file():
            current = json.loads(state_path.read_text(encoding="utf-8")).get("current") or {}
            tasks = current.get("tasks", [current])
            for task in tasks:
                if task.get("target") != name or not task.get("handle") or task.get("collected_bundle"):
                    continue
                script = Path(task["bundle"]) / "submit.sh"
                if script.is_file() and old["path"] in script.read_text(encoding="utf-8"):
                    changes[-1]["warning"] += (
                        " A previously submitted job still references the editable script. "
                        "Its launch environment may change if it has not started yet."
                    )
                    break
    return tuple(changes)


def snapshot_setup_records(root: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        if not record.get("role", "").startswith("execution_setup_"):
            result.append(record)
            continue
        data = Path(record["path"]).read_bytes()
        if _sha(data) != record["sha256"]:
            raise WorkflowConfigError("Launch script changed while saving its version; retry.")
        relative = Path(".neptrain/launch-inputs") / f"{record['sha256']}.sh"
        snapshot = root / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not snapshot.exists():
            with snapshot.open("xb") as stream:
                snapshot.chmod(0o600)
                stream.write(data)
        if _sha(snapshot.read_bytes()) != record["sha256"]:
            raise WorkflowConfigError(f"Saved launch script is damaged: {snapshot}")
        result.append({**record, "snapshot": str(relative)})
    return result


def accept_config_changes(root: Path) -> tuple[dict[str, Any], ...]:
    """Accept only supported edits while no controller can submit work."""
    from .workflow import _coerce_preparation, _validated_manifest
    from .workflow_workspace import WorkflowWorkspace

    workspace = WorkflowWorkspace.locate(root)
    with ExitStack() as stack:
        for path in (workspace.manifest_lock, workspace.controller_lock):
            handle = stack.enter_context(path.open("a+"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise WorkflowConfigError(
                    "Workflow is running or busy. Stop the controller with "
                    "workflow stop --keep-jobs before applying configuration edits."
                ) from error
        manifest = _validated_manifest(_coerce_preparation(root))
        record = manifest["config"]
        current_hash = _sha(workspace.project_file.read_bytes())
        launch_records = setup_records(manifest)
        changes = (*inspect_config(record), *inspect_setup_changes(manifest))
        if launch_records != setup_records(manifest):
            raise WorkflowConfigError("Launch script changed during configuration review; retry.")
        if _sha(workspace.project_file.read_bytes()) != current_hash:
            raise WorkflowConfigError("Project changed during configuration review; retry.")
        old_launch = [r for r in manifest.get("dependencies", []) if r.get("role", "").startswith("execution_setup_")]
        if (
            current_hash == record["sha256"] and record.get("snapshot") and not changes
            and all(r.get("snapshot") for r in old_launch)
        ):
            return ()
        updated = snapshot_config({**record, "sha256": current_hash})
        new_launch = snapshot_setup_records(root, launch_records)
        if current_hash != record["sha256"] or changes:
            manifest.setdefault("config_updates", []).append({
                "created_at": datetime.now(timezone.utc).isoformat(),
                "before": dict(record),
                "after": dict(updated),
                "changes": list(changes),
                "format_only": not changes,
                "launch_inputs_before": old_launch,
                "launch_inputs_after": new_launch,
            })
        manifest["config"] = updated
        manifest["dependencies"] = [
            r for r in manifest.get("dependencies", []) if not r.get("role", "").startswith("execution_setup_")
        ] + new_launch
        atomic_write_json(workspace.manifest, manifest)
        return changes


def remember_config(root: Path) -> None:
    """Give unchanged legacy workflows a baseline before their first edit."""
    from .workflow import _workflow_lock
    from .workflow_workspace import WorkflowWorkspace

    workspace = WorkflowWorkspace.locate(root)
    with _workflow_lock(root):
        manifest = json.loads(workspace.manifest.read_text(encoding="utf-8"))
        record = manifest["config"]
        if record.get("snapshot") or _sha(workspace.project_file.read_bytes()) != record["sha256"]:
            return
        manifest["config"] = snapshot_config(record)
        atomic_write_json(workspace.manifest, manifest)
