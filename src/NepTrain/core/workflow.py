"""High-level preparation and control for persistent iteration workflows."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any, Mapping

from .workflow_workspace import WorkflowWorkspace
from .sampling_route import load_sampling_routes


class WorkflowError(RuntimeError):
    """Raised when workflow preparation or submission is inconsistent."""


@dataclass(frozen=True)
class WorkflowPreparation:
    workflow_id: str
    output_dir: Path
    config_file: Path
    initial_training: Path
    plans: tuple[Path, ...]
    manifest: Path


@dataclass(frozen=True)
class WorkflowResume:
    workflow_id: str
    action: str
    manifest: Path
    controller_pid: int | None = None
    controller_exit_code: int | None = None


@dataclass(frozen=True)
class WorkflowStatus:
    workflow_id: str
    state: str
    completed_generations: int
    total_generations: int
    generation: int | None
    stage: str | None
    reason: str
    next_action: str | None
    generations: tuple[Mapping[str, Any], ...]
    jobs: tuple[Mapping[str, Any], ...]


_STAGES = (
    "train",
    "explore",
    "select",
    "label",
    "diagnose",
    "merge",
    "retrain",
    "evaluate",
)
_LEDGER_UNSET = object()


@contextmanager
def _workflow_lock(output_dir: Path):
    """Serialize scheduler side effects and manifest updates per workflow."""

    try:
        lock_path = WorkflowWorkspace.locate(output_dir).manifest_lock
    except FileNotFoundError:
        lock_path = output_dir / ".workflow-manifest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _path_record(role: str, path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "role": role,
            "kind": "file",
            "path": str(path),
            "sha256": _sha256(path),
        }
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        entries = [
            {"path": str(item.relative_to(path)), "sha256": _sha256(item)}
            for item in files
        ]
        return {
            "role": role,
            "kind": "directory",
            "path": str(path),
            "sha256": _canonical_hash(entries),
            "file_count": len(entries),
        }
    raise WorkflowError(f"workflow dependency does not exist: {path}")


def _record_matches(record: Mapping[str, Any]) -> bool:
    path = Path(record["path"])
    if record.get("kind", "file") == "file":
        return path.is_file() and _sha256(path) == record["sha256"]
    if not path.is_dir():
        return False
    current = _path_record(str(record.get("role", "dependency")), path)
    return current["sha256"] == record["sha256"]


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def _write_json(path: Path, value: Any) -> Path:
    return _write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _absolute_path(value: Any, base_dir: Path) -> str | None:
    if value in {None, ""}:
        return None
    path = Path(value).expanduser()
    return str((base_dir / path).resolve() if not path.is_absolute() else path.resolve())


def _resolved_config(config: Mapping[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = json.loads(json.dumps(config))
    training = resolved.get("training", {})
    for key in ("config_path", "test_path"):
        if training.get(key):
            training[key] = _absolute_path(training[key], base_dir)
    for route in resolved.get("sampling", {}).get("routes", []):
        route["structures"] = [
            _absolute_path(value, base_dir)
            for value in route.get("structures", [])
        ]
        if route.get("template_path"):
            route["template_path"] = _absolute_path(
                route["template_path"], base_dir
            )
    dft = resolved.get("dft", {})
    for key in ("input_path", "resource_path"):
        if dft.get(key):
            dft[key] = _absolute_path(dft[key], base_dir)
    evaluation = resolved.get("evaluation", {})
    if evaluation.get("validation_path"):
        evaluation["validation_path"] = _absolute_path(
            evaluation["validation_path"], base_dir
        )
    for profile in resolved.get("execution", {}).get("targets", {}).values():
        if profile.get("setup_script"):
            profile["setup_script"] = _absolute_path(
                profile["setup_script"], base_dir
            )
    return resolved


def _plans(
    settings: Mapping[str, Any], sampling: Mapping[str, Any]
) -> tuple[Any, ...]:
    from .iteration import progressive_plans

    selection = dict(sampling.get("selection") or {})
    novelty = selection.get("novelty", "auto")
    if novelty == "auto":
        selection_threshold = 0.0
        completion_threshold = 0.0
    else:
        selection_threshold = float(novelty["selection_threshold"])
        completion_threshold = float(novelty["completion_threshold"])
    plans = progressive_plans(
        int(settings.get("max_model_generations", 1)),
        seed=int(settings.get("seed", 20260721)),
        max_selected=int(selection.get("max_selected", 100)),
        selection_novelty_threshold=selection_threshold,
        completion_coverage_threshold=completion_threshold,
    )
    return plans


def _dependencies(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths: list[tuple[str, Path]] = []
    training = config.get("training", {})
    sampling = config.get("sampling", {})
    dft = config.get("dft", {})
    evaluation = config.get("evaluation", {})
    for role, value in (
        ("training_config", training.get("config_path")),
        ("training_test", training.get("test_path")),
        ("dft_input", dft.get("input_path")),
        ("dft_resources", dft.get("resource_path")),
        ("evaluation_validation", evaluation.get("validation_path")),
    ):
        if value and value != "auto":
            paths.append((role, Path(value)))
    for route in sampling.get("routes", []):
        route_id = str(route["id"])
        paths.append(
            (
                f"sampling_route_{route_id}_template",
                Path(route["template_path"]),
            )
        )
        paths.extend(
            (
                f"sampling_route_{route_id}_structure_{index}",
                Path(value),
            )
            for index, value in enumerate(route["structures"])
        )
    for name, target in config.get("execution", {}).get("targets", {}).items():
        setup_script = target.get("setup_script")
        if setup_script and setup_script != "auto":
            paths.append((f"execution_setup_{name}", Path(setup_script)))
    return [_path_record(role, path) for role, path in paths]


def _preparation_from_manifest(path: Path) -> WorkflowPreparation:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    workspace = WorkflowWorkspace.locate(path)
    return WorkflowPreparation(
        workflow_id=manifest["workflow_id"],
        output_dir=workspace.root,
        config_file=Path(manifest["config"]["path"]),
        initial_training=Path(manifest["initial_training"]["path"]),
        plans=tuple(Path(record["path"]) for record in manifest["plans"]),
        manifest=path,
    )


def _coerce_preparation(
    preparation: WorkflowPreparation | str | Path,
) -> WorkflowPreparation:
    if isinstance(preparation, WorkflowPreparation):
        return preparation
    path = Path(preparation).expanduser().resolve()
    try:
        manifest = WorkflowWorkspace.locate(path).manifest
    except FileNotFoundError:
        manifest = path if path.is_file() else path / ".neptrain" / "manifest.json"
    if not manifest.is_file():
        raise WorkflowError(f"prepared workflow manifest does not exist: {manifest}")
    return _preparation_from_manifest(manifest)


def prepare_workflow(
    config_path: str | Path,
    initial_training: str | Path,
    output_dir: str | Path,
    *,
    workflow_id: str | None = None,
) -> WorkflowPreparation:
    """Prepare immutable plans for the persistent workflow controller."""

    from .config import ConfigError, load_config, save_config

    source_config = Path(config_path).expanduser().resolve()
    initial = Path(initial_training).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not initial.is_file():
        raise WorkflowError(f"initial training set does not exist: {initial}")
    try:
        config, _ = load_config(source_config)
    except ConfigError as error:
        raise WorkflowError(f"invalid project configuration: {error}") from error
    config = _resolved_config(config, source_config.parent)
    labeling_target_name = config["execution"]["stage_targets"]["labeling"]
    labeling_target = config["execution"]["targets"][labeling_target_name]
    dft_backend = config["dft"]["backend"]
    dft_resource = labeling_target.get(
        "dft_resource_path", config["dft"].get("resource_path")
    )
    if dft_backend in {"vasp", "abacus"} and not dft_resource:
        raise WorkflowError(
            f"{dft_backend} workflows require dft.resource_path or "
            "execution.targets.<labeling>.dft_resource_path"
        )
    settings = config.get("workflow", {})
    selected_id = workflow_id or str(settings.get("id", output.name))
    if not selected_id.strip():
        raise WorkflowError("workflow id cannot be empty")
    plans = _plans(settings, config["sampling"])
    command = "neptrain"
    source_dependencies = _dependencies(config)
    spec = {
        "workflow_id": selected_id,
        "config": config,
        "initial_training": str(initial),
        "initial_training_sha256": _sha256(initial),
        "plans": [asdict(plan) for plan in plans],
        "command": command,
        "dependencies": source_dependencies,
    }
    spec_sha256 = _canonical_hash(spec)
    if (output / "workflow-manifest.json").is_file():
        workspace = WorkflowWorkspace.locate(output)
    elif (output / ".neptrain" / "layout.json").is_file():
        workspace = WorkflowWorkspace.locate(output)
    else:
        try:
            workspace = WorkflowWorkspace.create(output)
        except ValueError as error:
            raise WorkflowError(str(error)) from error
    manifest_path = workspace.manifest
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("version") != 5
            or existing.get("orchestration") != "controller"
        ):
            raise WorkflowError(
                "existing workflow uses an unsupported test-phase manifest; "
                "choose a new output directory"
            )
        if existing.get("spec_sha256") != spec_sha256:
            raise WorkflowError(
                "workflow preparation changed; choose a new output directory or "
                "restore the original inputs"
            )
        for record in [
            existing["config"],
            existing["initial_training"],
            *existing["plans"],
            *existing.get("dependencies", []),
        ]:
            if not _record_matches(record):
                raise WorkflowError(
                    f"prepared workflow artifact drifted: {record['path']}"
                )
        return _preparation_from_manifest(manifest_path)

    resolved_config = workspace.project_file
    portable_config, initial_snapshot = workspace.snapshot_inputs(config, initial)
    save_config(portable_config, resolved_config)
    runtime_config = _resolved_config(portable_config, workspace.root)
    dependencies = _dependencies(runtime_config)
    sampling_routes = [
        {
            "route_id": route.route_id,
            "route_fingerprint": route.fingerprint,
            "template_sha256": route.template_sha256,
            "structure_source_sha256": list(
                route.structure_source_sha256
            ),
        }
        for route in load_sampling_routes(
            runtime_config["sampling"], base_dir=workspace.root
        )
    ]
    plan_paths: list[Path] = []
    for plan in plans:
        plan_path = workspace.plans_dir / f"generation-{plan.generation}.json"
        _write_json(plan_path, asdict(plan))
        plan_paths.append(plan_path)
    manifest = {
        "version": 5,
        "layout_version": workspace.version,
        "orchestration": "controller",
        "workflow_id": selected_id,
        "spec_sha256": spec_sha256,
        "config": {"path": str(resolved_config), "sha256": _sha256(resolved_config)},
        "initial_training": {
            "path": str(initial_snapshot),
            "sha256": _sha256(initial_snapshot),
        },
        "plans": [
            {"path": str(path), "sha256": _sha256(path)} for path in plan_paths
        ],
        "dependencies": dependencies,
        "sampling_routes": sampling_routes,
    }
    _write_json(manifest_path, manifest)
    return _preparation_from_manifest(manifest_path)


def extend_workflow(
    preparation: WorkflowPreparation | str | Path,
    total_generations: int,
) -> WorkflowPreparation:
    """Append immutable generations to a completed, accepted workflow."""

    from .config import load_config, save_config

    preparation = _coerce_preparation(preparation)
    with _workflow_lock(preparation.output_dir):
        manifest = _validated_manifest(preparation)
        progress = _workflow_progress(preparation, manifest)
        if progress.state != "complete":
            raise WorkflowError(
                "workflow can only be extended after all prepared generations "
                "completed and passed evaluation"
            )
        current_total = len(preparation.plans)
        if total_generations <= current_total:
            raise WorkflowError(
                f"extension target must exceed current total {current_total}"
            )

        portable_config, _ = load_config(preparation.config_file)
        portable_config.setdefault("workflow", {})[
            "max_model_generations"
        ] = int(total_generations)
        config = _resolved_config(
            portable_config, preparation.config_file.parent
        )
        settings = dict(config.get("workflow", {}))
        all_plans = _plans(settings, config["sampling"])
        existing_values = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in preparation.plans
        ]
        if [
            _canonical_hash(asdict(plan)) for plan in all_plans[:current_total]
        ] != [_canonical_hash(value) for value in existing_values]:
            raise WorkflowError("workflow extension changed an existing generation plan")

        new_plans = all_plans[current_total:]
        new_plan_paths: list[Path] = []
        workspace = WorkflowWorkspace.locate(preparation.output_dir)
        for plan in new_plans:
            path = workspace.plans_dir / f"generation-{plan.generation}.json"
            _write_json(path, asdict(plan))
            new_plan_paths.append(path)

        manifest["plans"].extend(
            {"path": str(path), "sha256": _sha256(path)} for path in new_plan_paths
        )
        extension = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "from_generations": current_total,
            "to_generations": total_generations,
        }
        manifest.setdefault("extensions", []).append(extension)
        save_config(portable_config, preparation.config_file)
        manifest["config"]["sha256"] = _sha256(preparation.config_file)
        _write_json(preparation.manifest, manifest)
        if workspace.controller_file.is_file():
            controller_state = json.loads(
                workspace.controller_file.read_text(encoding="utf-8")
            )
            if controller_state.get("state") == "budget_exhausted":
                controller_state["state"] = "idle"
                controller_state["current"] = None
                controller_state.pop("reason", None)
                _write_json(workspace.controller_file, controller_state)
        return _preparation_from_manifest(preparation.manifest)


def _validated_manifest(preparation: WorkflowPreparation) -> dict[str, Any]:
    manifest = json.loads(preparation.manifest.read_text(encoding="utf-8"))
    if manifest.get("version") != 5 or manifest.get("orchestration") != "controller":
        raise WorkflowError(
            "unsupported workflow manifest; create a new workflow with the current NepTrain"
        )
    for record in [
        manifest["config"],
        manifest["initial_training"],
        *manifest["plans"],
        *manifest.get("dependencies", []),
    ]:
        if not _record_matches(record):
            raise WorkflowError(
                f"prepared workflow artifact drifted: {record['path']}"
            )
    return manifest


@dataclass(frozen=True)
class _WorkflowProgress:
    state: str
    completed_generations: int
    generation: int | None
    stage: str | None
    reason: str


def _read_workflow_ledger(
    preparation: WorkflowPreparation,
) -> Mapping[str, Any] | None:
    ledger_path = WorkflowWorkspace.locate(preparation.output_dir).ledger
    if not ledger_path.exists():
        return None
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("workflow_id") != preparation.workflow_id:
        raise WorkflowError("workflow_id does not match the existing ledger")
    if not isinstance(ledger.get("generations", {}), Mapping):
        raise WorkflowError("workflow generations must be a mapping")
    return ledger


def _workflow_progress(
    preparation: WorkflowPreparation,
    manifest: Mapping[str, Any],
    *,
    ledger: Mapping[str, Any] | None | object = _LEDGER_UNSET,
) -> _WorkflowProgress:
    plans = [
        json.loads(Path(record["path"]).read_text(encoding="utf-8"))
        for record in manifest["plans"]
    ]
    if ledger is _LEDGER_UNSET:
        ledger = _read_workflow_ledger(preparation)
    if ledger is None:
        return _WorkflowProgress(
            "prepared", 0, int(plans[0]["generation"]), "train", "ledger not started"
        )
    assert isinstance(ledger, Mapping)
    generations = ledger.get("generations", {})
    for plan in plans:
        generation = int(plan["generation"])
        record = generations.get(str(generation))
        if record is None:
            return _WorkflowProgress(
                "incomplete",
                generation - 1,
                generation,
                "train",
                f"generation {generation} has not started",
            )
        if record.get("plan_sha256") != _canonical_hash(plan):
            raise WorkflowError(
                f"generation {generation} plan changed after it entered the ledger"
            )
        stages = record.get("stages", {})
        completed = []
        missing_seen = False
        for stage in _STAGES:
            if stage not in stages:
                missing_seen = True
            elif missing_seen:
                raise WorkflowError(
                    "generation stage ledger is not a contiguous prefix"
                )
            else:
                completed.append(stage)
        if len(completed) < len(_STAGES):
            if record.get("complete"):
                raise WorkflowError(
                    "generation is marked complete before all stages finished"
                )
            stage = _STAGES[len(completed)]
            return _WorkflowProgress(
                "incomplete",
                generation - 1,
                generation,
                stage,
                f"generation {generation} is waiting for stage {stage}",
            )
        if not record.get("complete"):
            raise WorkflowError(
                f"generation {generation} has all stages but is not marked complete"
            )
        if record.get("accepted") is False:
            return _WorkflowProgress(
                "rejected",
                generation - 1,
                generation,
                None,
                f"generation {generation} failed its evaluation acceptance gate",
            )
        if record.get("accepted") is not True:
            raise WorkflowError(
                f"generation {generation} completion is missing accepted=true/false"
            )
    return _WorkflowProgress(
        "complete",
        len(plans),
        None,
        None,
        "all generations completed and passed evaluation",
    )


def _generation_science(
    plan: Mapping[str, Any], record: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Normalize one ledger generation into a stable scientific summary."""

    stages = {} if record is None else record.get("stages", {})
    if not isinstance(stages, Mapping):
        raise WorkflowError("generation stages must be a mapping")

    def metrics(stage: str) -> Mapping[str, Any]:
        stage_record = stages.get(stage, {})
        if not isinstance(stage_record, Mapping):
            raise WorkflowError(f"generation stage {stage} must be a mapping")
        value = stage_record.get("metrics", {})
        if not isinstance(value, Mapping):
            raise WorkflowError(f"generation stage {stage} metrics must be a mapping")
        return value

    train = metrics("train")
    explore = metrics("explore")
    select = metrics("select")
    label = metrics("label")
    diagnose = metrics("diagnose")
    merge = metrics("merge")
    retrain = metrics("retrain")
    evaluate = metrics("evaluate")

    if record is None:
        state = "not_started"
    elif record.get("complete") and record.get("accepted") is True:
        state = "accepted"
    elif record.get("complete") and record.get("accepted") is False:
        state = "rejected"
    else:
        state = "in_progress"

    acquisition_rmse = {
        name: diagnose.get(f"current_model_{name}")
        for name in ("energy_rmse", "force_rmse", "virial_rmse", "mforce_rmse")
    }
    validation_rmse = {
        name: evaluate.get(name)
        for name in ("energy_rmse", "force_rmse", "virial_rmse", "mforce_rmse")
    }
    return {
        "generation": int(plan["generation"]),
        "state": state,
        "completed_stages": tuple(stage for stage in _STAGES if stage in stages),
        "plan": {
            "max_selected": int(plan["max_selected"]),
            "selection_novelty_threshold": float(
                plan.get("selection_novelty_threshold", 0.0)
            ),
            "completion_coverage_threshold": float(
                plan.get("completion_coverage_threshold", 0.0)
            ),
        },
        "sampling": {
            "candidate_count": explore.get("candidate_count"),
            "candidate_counts_by_window": dict(
                explore.get("candidate_counts_by_window", {})
            ),
            "scheduled_source_count": explore.get("scheduled_source_count"),
            "completed_source_count": explore.get("completed_source_count"),
            "failed_source_count": explore.get("failed_source_count"),
            "candidate_count_before_deduplication": select.get(
                "candidate_count_before_deduplication"
            ),
            "candidate_count_after_deduplication": select.get(
                "candidate_count_after_deduplication"
            ),
            "duplicate_candidate_count": select.get("duplicate_candidate_count"),
            "selected_count": select.get("selected_count"),
            "regular_batch_minimum": select.get("regular_batch_minimum"),
            "batch_kind": select.get("batch_kind"),
            "sampling_model_sha256": explore.get(
                "sampling_model_sha256"
            ),
            "counts_by_stratum": dict(select.get("counts_by_stratum", {})),
            "labeled_count": label.get("labeled_count"),
        },
        "training": {
            "before_count": train.get("training_count"),
            "merged_count": merge.get("training_count"),
            "after_count": retrain.get("training_count"),
            "added_count": evaluate.get("added_training_count", merge.get("added_count")),
            "model_updated": retrain.get("model_updated"),
            "active_model_sha256": evaluate.get("active_model_sha256"),
        },
        "quality": {
            "acquisition_rmse": acquisition_rmse,
            "validation_rmse": validation_rmse,
            "accepted": evaluate.get("accepted"),
            "validation_count": evaluate.get("evaluated_count"),
            "spin_validation_count": evaluate.get("spin_frame_count"),
        },
        "scenarios": {
            "target_maturities": tuple(explore.get("scenario_targets", ())),
            "attempted_steps": tuple(explore.get("scenario_steps", ())),
            "counts_by_maturity": dict(
                evaluate.get("scenario_counts_by_maturity", {})
            ),
        },
    }


def _scientific_progress(
    preparation: WorkflowPreparation,
    manifest: Mapping[str, Any],
    *,
    ledger: Mapping[str, Any] | None | object = _LEDGER_UNSET,
) -> tuple[Mapping[str, Any], ...]:
    plans = [
        json.loads(Path(record["path"]).read_text(encoding="utf-8"))
        for record in manifest["plans"]
    ]
    generations: Mapping[str, Any] = {}
    if ledger is _LEDGER_UNSET:
        ledger = _read_workflow_ledger(preparation)
    if ledger is not None:
        assert isinstance(ledger, Mapping)
        generations = ledger.get("generations", {})
    return tuple(
        _generation_science(plan, generations.get(str(plan["generation"])))
        for plan in plans
    )


def workflow_status(output_dir: str | Path) -> WorkflowStatus:
    """Return the controller ledger and execution summary for one workflow."""

    preparation = _coerce_preparation(output_dir)
    manifest = _validated_manifest(preparation)
    ledger = _read_workflow_ledger(preparation)
    progress = _workflow_progress(preparation, manifest, ledger=ledger)
    generations = _scientific_progress(preparation, manifest, ledger=ledger)
    from .controller import controller_running

    workspace = WorkflowWorkspace.locate(preparation.output_dir)
    controller = (
        json.loads(workspace.controller_file.read_text(encoding="utf-8"))
        if workspace.controller_file.is_file()
        else {}
    )
    active = controller_running(workspace.root)
    controller_state = str(controller.get("state", "prepared"))
    current = controller.get("current")
    jobs = []
    for item in controller.get("history", []):
        if item.get("completed_at"):
            execution_state = "COMPLETED"
        elif item.get("cancelled_at"):
            execution_state = "CANCELLED"
        else:
            execution_state = "FAILED"
        task_records = (
            item.get("tasks", []) if item.get("kind") == "task_group" else [item]
        )
        for task_record in task_records:
            handle = task_record.get("handle") or {}
            jobs.append(
                {
                    "attempt": f"attempt-{item.get('attempt', 1)}",
                    "script": (
                        f"{task_record.get('target', '-')}/"
                        f"{item.get('stage', '-')}"
                    ),
                    "job_id": handle.get("execution_id"),
                    "dependency": None,
                    "state": execution_state,
                    "current": False,
                    "detail": item.get("failure")
                    or (item.get("cancellation") or {}).get("detail"),
                }
            )
    if current:
        task_records = (
            current.get("tasks", [])
            if current.get("kind") == "task_group"
            else [current]
        )
        for task_record in task_records:
            handle = task_record.get("handle") or {}
            observed = str(
                task_record.get("observed_state", controller_state)
            ).upper()
            jobs.append(
                {
                    "attempt": f"attempt-{current.get('attempt', 1)}",
                    "script": (
                        f"{task_record.get('target', '-')}/"
                        f"{current.get('stage', '-')}"
                    ),
                    "job_id": handle.get("execution_id"),
                    "dependency": None,
                    "state": observed,
                    "current": True,
                    "detail": task_record.get("detail"),
                }
            )

    next_action = None
    state = progress.state
    reason = progress.reason
    workflow_path = shlex.quote(str(preparation.output_dir))
    if controller_state == "complete":
        state = "complete"
        reason = str(controller.get("reason", "workflow trust envelope converged"))
    elif controller_state == "budget_exhausted":
        state = "budget_exhausted"
        reason = str(
            controller.get("reason", "model-generation budget exhausted")
        )
        next_action = (
            f"neptrain workflow extend {workflow_path} "
            f"{len(preparation.plans) + 1}"
        )
    elif controller_state == "stalled":
        state = "stalled"
        reason = str(controller.get("reason", "workflow made no progress"))
    elif progress.state == "rejected" or controller_state == "rejected":
        state = "rejected"
        reason = str(controller.get("reason", progress.reason))
        next_action = f"neptrain workflow resume {workflow_path}"
    elif active:
        state = "degraded" if controller_state == "degraded" else "running"
        reason = str(
            controller.get("last_transport_error")
            or controller.get("reason")
            or "persistent controller is active"
        )
        next_action = f"neptrain workflow status {workflow_path}"
    elif controller_state == "failed":
        state = "failed"
        reason = str(controller.get("reason", "controller failed"))
        next_action = f"neptrain workflow resume {workflow_path}"
    elif controller_state == "stopped":
        state = "paused"
        reason = str(controller.get("reason", "controller is stopped"))
        next_action = f"neptrain workflow resume {workflow_path}"
    elif controller_state in {"running", "launching", "degraded"}:
        state = "paused"
        reason = "controller process is not running; remote work is preserved"
        next_action = f"neptrain workflow resume {workflow_path}"
    elif progress.state == "complete":
        state = "complete"
        reason = progress.reason
    else:
        state = "prepared"
        reason = "workflow is prepared and controller has not started"
        next_action = f"neptrain workflow run {workflow_path}"

    return WorkflowStatus(
        workflow_id=preparation.workflow_id,
        state=state,
        completed_generations=progress.completed_generations,
        total_generations=len(preparation.plans),
        generation=int(current["generation"]) if current else progress.generation,
        stage=str(current["stage"]) if current else progress.stage,
        reason=reason,
        next_action=next_action,
        generations=generations,
        jobs=tuple(jobs),
    )


def resume_workflow(
    output_dir: str | Path,
    *,
    foreground: bool = False,
    poll_interval: float | None = None,
) -> WorkflowResume:
    """Resume one prepared controller workflow."""

    output = Path(output_dir).expanduser().resolve()
    preparation = _coerce_preparation(output)
    from .controller import (
        PersistentController,
        controller_running,
        start_controller,
    )

    status = workflow_status(output)
    if status.state == "complete":
        return WorkflowResume(
            preparation.workflow_id,
            "complete",
            preparation.manifest,
        )
    if controller_running(output):
        raise WorkflowError("workflow is already running")
    controller = PersistentController(output)
    action = "start" if status.state == "prepared" else "resume"
    if status.state == "failed":
        controller.retry()
        action = "retry"
    elif status.state == "rejected":
        controller.retry(recover_rejected=True)
        action = "recover_rejected"
    try:
        controller_result = start_controller(
            output,
            foreground=foreground,
            poll_interval=poll_interval,
        )
    except Exception as error:
        raise WorkflowError(str(error)) from error
    return WorkflowResume(
        preparation.workflow_id,
        action,
        preparation.manifest,
        controller_pid=None if foreground else controller_result,
        controller_exit_code=controller_result if foreground else None,
    )


__all__ = [
    "WorkflowError",
    "WorkflowPreparation",
    "WorkflowResume",
    "WorkflowStatus",
    "workflow_status",
    "extend_workflow",
    "prepare_workflow",
    "resume_workflow",
]
