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


def _before_first_labeling(project: Path) -> bool:
    """Fail closed when any label preparation/execution/evidence already exists."""
    state = project.parent / ".neptrain"
    try:
        ledger = json.loads((state / "ledger.json").read_text())
        controller = json.loads((state / "controller.json").read_text())
        if controller.get("state") != "stopped":
            return False
        for generation in ledger["generations"].values():
            if "label" in generation.get("stages", {}):
                return False
        records = [controller.get("current"), *controller.get("history", [])]
        if any(r and r.get("stage") == "label" for r in records):
            return False
        # Bundles are durable even if a controller crashed before recording submission.
        jobs = state / "jobs"
        if not jobs.is_dir():
            return False
        for bundle in jobs.iterdir():
            if "-label-" in bundle.name:
                return False
            task = bundle / "task.json"
            if task.is_file() and json.loads(task.read_text()).get("stage") == "label":
                return False
        if any((project.parent / "generations").glob("*/label*")):
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def _before_sampling(project: Path) -> bool:
    """No current execution or current-generation sampling evidence may survive."""
    from .generation_policy import generation_stage_sequence

    state = project.parent / ".neptrain"
    try:
        controller = json.loads((state / "controller.json").read_text())
        if controller.get("current") is not None or controller.get("state") not in {"idle", "stopped"}:
            return False
        records = json.loads((state / "ledger.json").read_text())["generations"]
        entered = [int(k) for k, v in records.items() if v.get("stages") is not None]
        if not entered:
            return False
        record = records[str(max(entered))]
        if record.get("complete"):
            return False
        sequence = generation_stage_sequence(record)
        preceding = set(sequence[:sequence.index("explore")])
        return bool(record.get("stages")) and set(record["stages"]) <= preceding
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _sampling_policy_boundary(project: Path) -> bool:
    """A stopped controller may change future sampling, never an existing pool.

    A retained training job owns its packaged input and can finish unchanged.
    New route fingerprints invalidate maturity evidence from the old protocol.
    """
    state = project.parent / ".neptrain"
    try:
        controller = json.loads((state / "controller.json").read_text())
        if controller.get("state") != "stopped":
            return False
        records = json.loads((state / "ledger.json").read_text())["generations"]
        if not records:
            return False
        generation = max(map(int, records))
        record = records[str(generation)]
        current = controller.get("current")
        if current is not None and (
            current.get("generation") != generation or current.get("stage") != "train"
        ):
            return False
        if not record.get("complete") and set(record.get("stages", {})) - {"train"}:
            return False
        # A crash may leave a submitted sampling bundle without a current intent.
        for bundle in (state / "jobs").glob("*"):
            task = bundle / "task.json"
            if not task.is_file():
                continue
            spec = json.loads(task.read_text())
            if (not record.get("complete") and spec.get("generation") == generation
                    and spec.get("stage") not in {None, "train"}):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _reopened_sampling_boundary(project: Path) -> bool:
    """Accept a new template after an explicit explore restart archived its pool."""
    if not _before_sampling(project):
        return False
    try:
        records = json.loads((project.parent / ".neptrain/ledger.json").read_text())["generations"]
        entered = [int(k) for k, v in records.items() if v.get("stages") is not None]
        record = records[str(max(entered))]
        recoveries = record.get("recovery_attempts", [])
        return bool(recoveries) and recoveries[-1].get("from_stage") == "explore"
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _convergence_policy_boundary(project: Path) -> bool:
    """A stopped workflow may change criteria before its next accuracy stage."""
    from .generation_policy import stage_for_role

    state = project.parent / ".neptrain"
    try:
        controller_file = state / "controller.json"
        controller = (
            json.loads(controller_file.read_text())
            if controller_file.is_file()
            else {"state": "prepared", "current": None}
        )
        if controller.get("state") not in {"prepared", "stopped", "idle"}:
            return False
        current = controller.get("current")
        ledger_file = state / "ledger.json"
        if not ledger_file.is_file():
            return controller.get("state") == "prepared" and current is None
        records = json.loads(ledger_file.read_text())["generations"]
        if not records:
            return current is None
        generation = max(map(int, records))
        record = records[str(generation)]
        if record.get("complete"):
            return current is None
        evaluation_stage = stage_for_role(record, "evaluate")
        if evaluation_stage is None or evaluation_stage in record.get("stages", {}):
            return False
        if current is not None and (
            int(current.get("generation", -1)) != generation
            or current.get("stage") == evaluation_stage
        ):
            return False
        for bundle in (state / "jobs").glob("*"):
            task = bundle / "task.json"
            if not task.is_file():
                continue
            identity = json.loads(task.read_text()).get("identity", {})
            if (
                int(identity.get("generation", -1)) == generation
                and identity.get("stage") == evaluation_stage
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _invalidates_prior_accuracy(change: Mapping[str, Any]) -> bool:
    """Only a stricter per-generation test can invalidate an earlier pass."""
    field = str(change["field"])
    before, after = change["before"], change["after"]
    if field.startswith("workflow.convergence.acquisition_max_rmse.") or field == (
        "workflow.convergence.max_outlier_fraction"
    ):
        return after is not None and (before is None or float(after) < float(before))
    if field.startswith("workflow.convergence.acquisition_min_r2.") or field == (
        "workflow.convergence.group_min_force_r2"
    ):
        return after is not None and (before is None or float(after) > float(before))
    if field == "workflow.convergence.min_selected":
        return int(after if after is not None else 1) > int(
            before if before is not None else 1
        )
    # Consecutive generations and production coverage are aggregate requirements;
    # changing them does not turn an earlier passing accuracy generation into a fail.
    return False


def inspect_config(
    record: Mapping[str, Any], *, sampling_restart: bool = False,
) -> tuple[dict[str, Any], ...]:
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
    sampling_boundary = (
        sampling_restart or _sampling_policy_boundary(path)
        or _reopened_sampling_boundary(path)
    )
    differences = tuple(_differences(before, after))
    convergence_boundary = any(
        parts[:2] == ("workflow", "convergence")
        for parts, _, _ in differences
    ) and _convergence_policy_boundary(path)
    for parts, old, new in differences:
        scope = _scope(parts, old, new)
        if parts[:2] == ("workflow", "convergence") and convergence_boundary:
            scope = "next accuracy evaluation; prior results retained"
        if (len(parts) == 4 and parts[:2] == ("sampling", "routes")
                and parts[3] == "template_path" and sampling_boundary):
            scope = "future sampling only; existing training and labels retained"
        if (len(parts) == 6 and parts[:2] == ("sampling", "routes")
                and parts[3:5] == ("progression", "steps") and sampling_boundary):
            scope = "future sampling only; route maturity restarts under the new fingerprint"
        if parts[:3] == ("sampling", "candidate_pool", "health") and (
            sampling_boundary or _before_sampling(path)
        ):
            scope = "sampling and downstream stages rebuilt; earlier training retained"
        if (parts[:2] == ("labeling", "hubbard") or parts == ("labeling", "input_path")) and after.get("labeling", {}).get("hubbard") is not None and _before_first_labeling(path):
            scope = "first labeling only; existing training jobs and results are retained"
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
        if all(field.startswith("workflow.convergence.") for field in blocked):
            raise WorkflowConfigError(
                "These convergence changes cannot be applied at the current "
                "workflow boundary: " + ", ".join(blocked)
                + ". Stop the controller with --keep-jobs before the next "
                "accuracy evaluation, then edit "
                "project.yaml. Existing results are retained; no edits were accepted."
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
        new_label_inputs = None
        old_label_inputs = [r for r in manifest.get("dependencies", []) if r.get("role") == "labeling_input"]
        if any(c["field"].startswith("labeling.hubbard") or c["field"] == "labeling.input_path" for c in changes):
            from .workflow import _dependencies, _resolved_config
            from .dft.vasp.native import validate_vasp_input_file
            config, _ = load_config(workspace.project_file)
            config = _resolved_config(config, workspace.project_file.parent)
            label = config["labeling"]
            validate_vasp_input_file(label["input_path"], hubbard=label.get("hubbard"))
            new_label_inputs = [r for r in _dependencies(config) if r.get("role") == "labeling_input"]
        old_sampling = [r for r in manifest.get("dependencies", [])
                        if r.get("role", "").startswith("sampling_route_")]
        new_sampling = None
        new_routes = None
        if any(c["field"].startswith("sampling.routes.") for c in changes):
            from .workflow import _dependencies, _resolved_config
            from .sampling_route import load_sampling_routes
            config, _ = load_config(workspace.project_file)
            config = _resolved_config(config, workspace.project_file.parent)
            new_sampling = [r for r in _dependencies(config)
                            if r.get("role", "").startswith("sampling_route_")]
            new_routes = [{"route_id": route.route_id,
                           "route_fingerprint": route.fingerprint,
                           "template_sha256": route.template_sha256,
                           "structure_source_sha256": list(route.structure_source_sha256)}
                          for route in load_sampling_routes(config["sampling"], base_dir=root)]
        updated = snapshot_config({**record, "sha256": current_hash})
        new_launch = snapshot_setup_records(root, launch_records)
        if current_hash != record["sha256"] or changes:
            manifest.setdefault("config_updates", []).append({
                "created_at": datetime.now(timezone.utc).isoformat(),
                "before": dict(record),
                "after": dict(updated),
                "changes": list(changes),
                "format_only": not changes,
                "sampling_inputs_before": old_sampling,
                "sampling_inputs_after": new_sampling if new_sampling is not None else old_sampling,
                "launch_inputs_before": old_launch,
                "launch_inputs_after": new_launch,
                "label_inputs_before": old_label_inputs,
                "label_inputs_after": new_label_inputs if new_label_inputs is not None else old_label_inputs,
            })
        convergence_changes = [
            change for change in changes
            if change["field"].startswith("workflow.convergence.")
        ]
        if convergence_changes:
            manifest["convergence_revision"] = int(
                manifest.get("convergence_revision", 0)
            ) + 1
            if any(_invalidates_prior_accuracy(change) for change in convergence_changes):
                manifest["convergence_evidence_revision"] = int(
                    manifest.get("convergence_evidence_revision", 0)
                ) + 1
        if new_label_inputs is not None:
            manifest["dependencies"] = [r for r in manifest.get("dependencies", []) if r.get("role") != "labeling_input"] + new_label_inputs
        if new_sampling is not None:
            manifest["dependencies"] = [r for r in manifest.get("dependencies", [])
                                        if not r.get("role", "").startswith("sampling_route_")] + new_sampling
            manifest["sampling_routes"] = new_routes
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
