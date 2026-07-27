"""High-level preparation and control for persistent iteration workflows."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
from typing import Any, Mapping
import uuid

from .content_addressing import canonical_sha256, file_sha256
from .persistence import atomic_write_json
from .workflow_workspace import WorkflowWorkspace
from .sampling_route import load_sampling_routes
from .scientific_data import STRUCTURE_ID_VERSION


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


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowError(
            f"cannot read {role} JSON at {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise WorkflowError(f"{role} JSON at {path} must contain an object")
    return dict(value)


def _manifest_instance_id(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> str:
    value = str(manifest.get("instance_id", "")).strip()
    if value:
        return value
    # One release of read compatibility for v5 workflows.  The path prevents a
    # historical task from another workflow directory being treated as the
    # same instance, while all newly prepared workflows receive a random id.
    legacy = f"{manifest_path.resolve()}:{manifest.get('spec_sha256', '')}"
    return "legacy-" + hashlib.sha256(legacy.encode()).hexdigest()[:32]


def _normalise_manifest(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    value = dict(manifest)
    if (
        value.get("version") != 7
        or value.get("structure_id_version") != STRUCTURE_ID_VERSION
    ):
        raise WorkflowError(
            "unsupported workflow manifest or structure identity version; "
            "create a new workflow with the current NepTrain"
        )
    required_records = ("config", "initial_training")
    if (
        value.get("orchestration") != "controller"
        or not isinstance(value.get("workflow_id"), str)
        or not value["workflow_id"]
        or any(not isinstance(value.get(name), Mapping) for name in required_records)
        or not isinstance(value.get("plans"), list)
        or not value["plans"]
        or any(not isinstance(item, Mapping) for item in value["plans"])
        or not isinstance(value.get("dependencies", []), list)
    ):
        raise WorkflowError(
            f"workflow manifest is incomplete or malformed: {manifest_path}"
        )
    value["instance_id"] = _manifest_instance_id(value, manifest_path)
    return value


def _path_record(role: str, path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "role": role,
            "kind": "file",
            "path": str(path),
            "sha256": file_sha256(path),
        }
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        entries = [
            {"path": str(item.relative_to(path)), "sha256": file_sha256(item)}
            for item in files
        ]
        return {
            "role": role,
            "kind": "directory",
            "path": str(path),
            "sha256": canonical_sha256(entries),
            "file_count": len(entries),
        }
    raise WorkflowError(f"workflow dependency does not exist: {path}")


def _record_matches(record: Mapping[str, Any]) -> bool:
    path = Path(record["path"])
    if record.get("kind", "file") == "file":
        return path.is_file() and file_sha256(path) == record["sha256"]
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
    labeling = resolved.get("labeling", {})
    for key in (
        "input_path",
        "resource_path",
        "potcar_manifest_path",
        "resource_manifest_path",
        "model_path",
    ):
        if labeling.get(key):
            labeling[key] = _absolute_path(labeling[key], base_dir)
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


def _sampling_frames(config: Mapping[str, Any]) -> list[Any]:
    from ase.io import read as ase_read

    frames = []
    for route in config["sampling"]["routes"]:
        for raw_source in route["structures"]:
            source = Path(raw_source)
            paths = [source]
            if source.is_dir():
                paths = sorted(
                    {
                        path
                        for pattern in ("*.xyz", "*.extxyz", "*.vasp", "POSCAR*")
                        for path in source.glob(pattern)
                        if path.is_file()
                    }
                )
            for path in paths:
                try:
                    loaded = ase_read(path, index=":")
                except Exception as error:
                    raise WorkflowError(
                        f"cannot read sampling structure while validating DFT "
                        f"resources: {path}: {error}"
                    ) from error
                frames.extend(loaded if isinstance(loaded, list) else [loaded])
    if not frames:
        raise WorkflowError(
            "DFT resource validation found no sampling structures"
        )
    return frames


def _sampling_frames_by_order(
    frames: list[Any],
    order_for,
) -> dict[tuple[str, ...], Any]:
    frames_by_order: dict[tuple[str, ...], Any] = {}
    for frame in frames:
        frames_by_order.setdefault(order_for(frame), frame)
    return frames_by_order


def _validate_labeled_dataset_for_preparation(
    path: Path,
    *,
    role: str,
    expect_spin: bool,
) -> None:
    from ase.io import read as ase_read

    from .scientific_data import ScientificDataError, validate_labeled_frames
    from .spin import SpinDataError, validate_spin_dataset

    try:
        loaded = ase_read(path, index=":")
        frames = loaded if isinstance(loaded, list) else [loaded]
        validate_labeled_frames(frames)
        total, spin_frames = validate_spin_dataset(
            frames,
            require_mforce=True,
        )
    except (OSError, ScientificDataError, SpinDataError, ValueError) as error:
        raise WorkflowError(f"{role} dataset is invalid at {path}: {error}") from error
    if (spin_frames == total) != expect_spin:
        expected = "spin/mforce" if expect_spin else "ordinary"
        raise WorkflowError(
            f"{role} dataset must contain {expected} labels to match md.spin"
        )


def _validate_vasp_preparation(
    config: Mapping[str, Any],
    *,
    labeling_target: Mapping[str, Any],
    sampling_frames: list[Any],
) -> None:
    if config.get("labeling", {}).get("backend") != "vasp":
        return
    from .dft.vasp.resources import (
        VaspResourceError,
        validate_vasp_manifest_elements,
        validate_vasp_resources,
        vasp_element_order,
    )
    from .dft.vasp.native import (
        NativeVaspError,
        validate_vasp_input_file,
        validate_vasp_structure,
    )

    labeling = config["labeling"]
    manifest_path = Path(str(labeling["potcar_manifest_path"]))
    frames_by_order = _sampling_frames_by_order(
        sampling_frames,
        vasp_element_order,
    )
    try:
        input_path = labeling.get("input_path")
        electronic_mode = (
            validate_vasp_input_file(input_path)
            if input_path
            else "non_spin_polarized"
        )
        for frame in frames_by_order.values():
            validate_vasp_structure(
                frame,
                electronic_mode=electronic_mode,
            )
        validate_vasp_manifest_elements(manifest_path, frames_by_order)
        if not labeling_target.get("host"):
            resource_root = (
                labeling_target.get("labeling_resource_path")
                or labeling.get("resource_path")
            )
            for frame in frames_by_order.values():
                validate_vasp_resources(
                    str(resource_root),
                    manifest_path,
                    frame,
                )
    except (NativeVaspError, VaspResourceError) as error:
        raise WorkflowError(str(error)) from error


def _validate_abacus_preparation(
    config: Mapping[str, Any],
    *,
    labeling_target: Mapping[str, Any],
    sampling_frames: list[Any],
) -> None:
    if config.get("labeling", {}).get("backend") != "abacus":
        return
    from .dft.abacus.io import read_input_file
    from .dft.abacus.native import (
        NativeAbacusError,
        validate_abacus_spin_contract,
    )
    from .dft.abacus.resources import (
        AbacusResourceError,
        validate_abacus_manifest_elements,
        validate_abacus_resources,
    )

    labeling = config["labeling"]
    manifest_path = Path(str(labeling["resource_manifest_path"]))
    frames_by_order = _sampling_frames_by_order(
        sampling_frames,
        lambda frame: tuple(dict.fromkeys(frame.get_chemical_symbols())),
    )
    input_path = labeling.get("input_path")
    parameters = read_input_file(str(input_path)) if input_path else {}
    require_orbitals = (
        str(parameters.get("basis_type", "pw")).strip().lower() == "lcao"
    )
    try:
        validate_abacus_spin_contract(
            dict(parameters),
            spin_frame=bool(config.get("md", {}).get("spin", False)),
        )
        validate_abacus_manifest_elements(manifest_path, frames_by_order)
        if not labeling_target.get("host"):
            resource_root = (
                labeling_target.get("labeling_resource_path")
                or labeling.get("resource_path")
            )
            for frame in frames_by_order.values():
                validate_abacus_resources(
                    str(resource_root),
                    manifest_path,
                    frame,
                    require_orbitals=require_orbitals,
                )
    except (AbacusResourceError, NativeAbacusError) as error:
        raise WorkflowError(str(error)) from error


def _dependencies(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths: list[tuple[str, Path]] = []
    training = config.get("training", {})
    sampling = config.get("sampling", {})
    labeling = config.get("labeling", {})
    evaluation = config.get("evaluation", {})
    for role, value in (
        ("training_config", training.get("config_path")),
        ("training_test", training.get("test_path")),
        ("labeling_input", labeling.get("input_path")),
        (
            "labeling_potcar_manifest",
            labeling.get("potcar_manifest_path"),
        ),
        (
            "labeling_resource_manifest",
            labeling.get("resource_manifest_path"),
        ),
        ("labeling_model", labeling.get("model_path")),
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
    manifest = _normalise_manifest(
        _read_json(path, role="workflow manifest"),
        path,
    )
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
    expect_spin = bool(config.get("md", {}).get("spin", False))
    _validate_labeled_dataset_for_preparation(
        initial,
        role="initial training",
        expect_spin=expect_spin,
    )
    for role, raw_path in (
        ("training test", config.get("training", {}).get("test_path")),
        ("evaluation validation", config.get("evaluation", {}).get("validation_path")),
    ):
        if raw_path:
            _validate_labeled_dataset_for_preparation(
                Path(str(raw_path)),
                role=role,
                expect_spin=expect_spin,
            )
    labeling_target_name = config["execution"]["stage_targets"]["labeling"]
    labeling_target = config["execution"]["targets"][labeling_target_name]
    labeling_backend = config["labeling"]["backend"]
    labeling_resource = labeling_target.get(
        "labeling_resource_path",
        config["labeling"].get("resource_path"),
    )
    if (
        labeling_backend in {"vasp", "abacus"}
        and not labeling_resource
    ):
        raise WorkflowError(
            f"{labeling_backend} workflows require labeling.resource_path or "
            "execution.targets.<labeling>.labeling_resource_path"
        )
    sampling_frames = _sampling_frames(config)
    from .spin import SpinDataError, validate_spin_dataset

    try:
        total_frames, spin_frames = validate_spin_dataset(
            sampling_frames,
            require_mforce=False,
        )
    except SpinDataError as error:
        raise WorkflowError(
            f"sampling structure spin contract is invalid: {error}"
        ) from error
    expected_spin = expect_spin
    if (spin_frames == total_frames) != expected_spin:
        expected = "canonical spin:R:3" if expected_spin else "ordinary"
        raise WorkflowError(
            f"sampling structures must all be {expected} frames to match md.spin"
        )
    _validate_vasp_preparation(
        config,
        labeling_target=labeling_target,
        sampling_frames=sampling_frames,
    )
    _validate_abacus_preparation(
        config,
        labeling_target=labeling_target,
        sampling_frames=sampling_frames,
    )
    settings = config.get("workflow", {})
    selected_id = workflow_id or str(settings.get("id", output.name))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", selected_id):
        raise WorkflowError(
            "workflow id must be 1-64 safe characters: letters, numbers, "
            "dot, underscore, or hyphen, and must start with a letter or number"
        )
    plans = _plans(settings, config["sampling"])
    command = "neptrain"
    source_dependencies = _dependencies(config)
    spec = {
        "workflow_id": selected_id,
        "structure_id_version": STRUCTURE_ID_VERSION,
        "config": config,
        "initial_training": str(initial),
        "initial_training_sha256": file_sha256(initial),
        "plans": [asdict(plan) for plan in plans],
        "command": command,
        "dependencies": source_dependencies,
    }
    spec_sha256 = canonical_sha256(spec)
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
        existing = _normalise_manifest(
            _read_json(manifest_path, role="workflow manifest"),
            manifest_path,
        )
        if (
            existing.get("orchestration") != "controller"
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
        atomic_write_json(plan_path, asdict(plan))
        plan_paths.append(plan_path)
    manifest = {
        "version": 7,
        "layout_version": workspace.version,
        "structure_id_version": STRUCTURE_ID_VERSION,
        "orchestration": "controller",
        "workflow_id": selected_id,
        "instance_id": uuid.uuid4().hex,
        "spec_sha256": spec_sha256,
        "config": {"path": str(resolved_config), "sha256": file_sha256(resolved_config)},
        "initial_training": {
            "path": str(initial_snapshot),
            "sha256": file_sha256(initial_snapshot),
        },
        "plans": [
            {"path": str(path), "sha256": file_sha256(path)} for path in plan_paths
        ],
        "dependencies": dependencies,
        "sampling_routes": sampling_routes,
    }
    atomic_write_json(manifest_path, manifest)
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
            _read_json(path, role="generation plan")
            for path in preparation.plans
        ]
        if [
            canonical_sha256(asdict(plan)) for plan in all_plans[:current_total]
        ] != [canonical_sha256(value) for value in existing_values]:
            raise WorkflowError("workflow extension changed an existing generation plan")

        new_plans = all_plans[current_total:]
        new_plan_paths: list[Path] = []
        workspace = WorkflowWorkspace.locate(preparation.output_dir)
        for plan in new_plans:
            path = workspace.plans_dir / f"generation-{plan.generation}.json"
            atomic_write_json(path, asdict(plan))
            new_plan_paths.append(path)

        manifest["plans"].extend(
            {"path": str(path), "sha256": file_sha256(path)} for path in new_plan_paths
        )
        extension = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "from_generations": current_total,
            "to_generations": total_generations,
        }
        manifest.setdefault("extensions", []).append(extension)
        save_config(portable_config, preparation.config_file)
        manifest["config"]["sha256"] = file_sha256(preparation.config_file)
        atomic_write_json(preparation.manifest, manifest)
        if workspace.controller_file.is_file():
            controller_state = _read_json(
                workspace.controller_file,
                role="controller state",
            )
            if controller_state.get("state") == "budget_exhausted":
                controller_state["state"] = "idle"
                controller_state["current"] = None
                controller_state.pop("reason", None)
                atomic_write_json(workspace.controller_file, controller_state)
        return _preparation_from_manifest(preparation.manifest)


def _validated_manifest(preparation: WorkflowPreparation) -> dict[str, Any]:
    manifest = _normalise_manifest(
        _read_json(preparation.manifest, role="workflow manifest"),
        preparation.manifest,
    )
    if manifest.get("orchestration") != "controller":
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
    ledger = _read_json(ledger_path, role="scientific ledger")
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
        _read_json(Path(record["path"]), role="generation plan")
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
    workspace = WorkflowWorkspace.locate(preparation.output_dir)
    latest_accepted: tuple[int, Mapping[str, Any]] | None = None

    def damaged(
        generation: int,
        stage: str | None,
        reason: str,
    ) -> _WorkflowProgress:
        return _WorkflowProgress(
            "damaged",
            max(0, generation - 1),
            generation,
            stage,
            reason,
        )

    def projection_damage() -> _WorkflowProgress | None:
        if latest_accepted is None:
            return None
        generation, record = latest_accepted
        publication = record.get("publication")
        if not isinstance(publication, Mapping):
            return damaged(
                generation,
                None,
                f"accepted generation {generation} uses the legacy result "
                "projection and must be migrated by resume",
            )
        issues = workspace.publication_issues(
            publication,
            check_projection=True,
        )
        if issues:
            return damaged(generation, None, "; ".join(issues))
        return None

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
        if record.get("plan_sha256") != canonical_sha256(plan):
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
                stage_record = stages[stage]
                if not isinstance(stage_record, Mapping):
                    return damaged(
                        generation,
                        stage,
                        f"generation {generation} stage {stage} metadata is invalid",
                    )
                artifacts = stage_record.get("artifacts", {})
                if not isinstance(artifacts, Mapping):
                    return damaged(
                        generation,
                        stage,
                        f"generation {generation} stage {stage} artifacts are invalid",
                    )
                for name, artifact_record in artifacts.items():
                    if not isinstance(artifact_record, Mapping):
                        return damaged(
                            generation,
                            stage,
                            f"artifact {name} has invalid ledger metadata",
                        )
                    path = Path(str(artifact_record.get("path", "")))
                    try:
                        path.resolve().relative_to(workspace.root)
                    except ValueError:
                        return damaged(
                            generation,
                            stage,
                            f"committed artifact path escapes the workflow: {path}",
                        )
                    if (
                        not path.is_file()
                        or file_sha256(path) != artifact_record.get("sha256")
                    ):
                        return damaged(
                            generation,
                            stage,
                            f"committed artifact drifted or is missing: {path}",
                        )
        if len(completed) < len(_STAGES):
            if record.get("complete"):
                raise WorkflowError(
                    "generation is marked complete before all stages finished"
                )
            stage = _STAGES[len(completed)]
            issue = projection_damage()
            if issue is not None:
                return issue
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
            issue = projection_damage()
            if issue is not None:
                return issue
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
        publication = record.get("publication")
        if isinstance(publication, Mapping):
            issues = workspace.publication_issues(
                publication,
                check_projection=False,
            )
            if issues:
                return damaged(generation, None, "; ".join(issues))
        latest_accepted = (generation, record)
    issue = projection_damage()
    if issue is not None:
        return issue
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
        _read_json(Path(record["path"]), role="generation plan")
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
        _read_json(workspace.controller_file, role="controller state")
        if workspace.controller_file.is_file()
        else {}
    )
    history = controller.get("history", [])
    current = controller.get("current")
    if (
        not isinstance(history, list)
        or any(not isinstance(item, Mapping) for item in history)
        or (current is not None and not isinstance(current, Mapping))
        or not isinstance(controller.get("state", "prepared"), str)
    ):
        raise WorkflowError(
            f"controller state is malformed: {workspace.controller_file}"
        )
    if ledger is None:
        generation_evidence = any(
            item.is_file() or item.is_symlink()
            for item in workspace.generations_dir.rglob("*")
        )
        accepted_evidence = any(
            (workspace.results_dir / "accepted").iterdir()
        )
        if history or generation_evidence or accepted_evidence:
            progress = _WorkflowProgress(
                "damaged",
                0,
                1,
                None,
                "scientific ledger is missing while workflow result or "
                "execution evidence still exists",
            )
    active = controller_running(workspace.root)
    controller_state = str(controller.get("state", "prepared"))
    jobs = []
    for item in history:
        task_records = (
            item.get("tasks", []) if item.get("kind") == "task_group" else [item]
        )
        for task_record in task_records:
            if task_record.get("collected_bundle"):
                execution_state = "COMPLETED"
            elif (
                task_record.get("terminal_failure")
                and not task_record.get("retryable", True)
                and task_record.get("failure_kind") == "out_of_memory"
            ):
                execution_state = "SKIPPED_OOM"
            elif item.get("cancelled_at"):
                execution_state = "CANCELLED"
            elif item.get("completed_at"):
                execution_state = "COMPLETED"
            else:
                execution_state = "FAILED"
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
                    "detail": task_record.get("failure")
                    or task_record.get("detail")
                    or item.get("failure")
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
            cancellation = task_record.get("cancellation") or {}
            if task_record.get("collected_bundle"):
                observed = "COMPLETED"
            elif cancellation.get("action") in {
                "cancelled",
                "failed",
                "cancelled_before_launch",
            }:
                observed = "CANCELLED"
            elif cancellation:
                observed = "CANCELLING"
            elif (
                task_record.get("terminal_failure")
                and not task_record.get("retryable", True)
                and task_record.get("failure_kind") == "out_of_memory"
            ):
                observed = "SKIPPED_OOM"
            elif task_record.get("terminal_failure"):
                observed = "FAILED"
            else:
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
                    "detail": task_record.get("detail")
                    or cancellation.get("detail"),
                }
            )

    next_action = None
    state = progress.state
    reason = progress.reason
    workflow_path = shlex.quote(str(preparation.output_dir))
    if progress.state == "damaged":
        state = "damaged"
        reason = progress.reason
        next_action = f"neptrain workflow resume {workflow_path}"
    elif controller_state == "complete":
        if progress.state != "complete":
            state = "damaged"
            reason = (
                "controller says complete but the scientific ledger is "
                f"{progress.state}: {progress.reason}"
            )
            next_action = f"neptrain workflow resume {workflow_path}"
        else:
            state = "complete"
            reason = str(
                controller.get("reason", "workflow trust envelope converged")
            )
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
        state = (
            controller_state
            if controller_state in {"degraded", "waiting"}
            else "running"
        )
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
    elif controller_state in {"running", "launching", "degraded", "waiting"}:
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


def _repair_damaged_workflow(preparation: WorkflowPreparation) -> tuple[str, ...]:
    """Restore hash-identical committed files and rebuild derived projections."""

    workspace = WorkflowWorkspace.locate(preparation.output_dir)
    ledger = _read_workflow_ledger(preparation)
    if ledger is None:
        return ()

    candidates: dict[str, list[Path]] = {}

    def add_candidate(path: Path, expected: Any) -> None:
        sha256 = str(expected)
        try:
            resolved = path.resolve()
            resolved.relative_to(workspace.root)
        except (OSError, ValueError):
            return
        if resolved.is_file():
            candidates.setdefault(sha256, []).append(resolved)

    for result_path in workspace.tasks_dir.glob("*/result.json"):
        try:
            result = _read_json(result_path, role="stage result")
        except WorkflowError:
            continue
        artifacts = result.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            continue
        for record in artifacts.values():
            if not isinstance(record, Mapping):
                continue
            relative = Path(str(record.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                continue
            add_candidate(result_path.parent / relative, record.get("sha256"))

    accepted_root = workspace.results_dir / "accepted"
    for publication_path in accepted_root.glob("*/publication.json"):
        try:
            publication = _read_json(
                publication_path,
                role="accepted publication",
            )
        except WorkflowError:
            continue
        files = publication.get("files", {})
        if not isinstance(files, Mapping):
            continue
        for record in files.values():
            if not isinstance(record, Mapping):
                continue
            relative = Path(str(record.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                continue
            add_candidate(
                publication_path.parent / relative,
                record.get("sha256"),
            )

    repaired = []
    damaged_root = (
        workspace.internal_dir
        / "damaged"
        / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    )
    generations = ledger.get("generations", {})
    for generation in generations.values():
        if not isinstance(generation, Mapping):
            continue
        stages = generation.get("stages", {})
        if not isinstance(stages, Mapping):
            continue
        for stage in stages.values():
            if not isinstance(stage, Mapping):
                continue
            artifacts = stage.get("artifacts", {})
            if not isinstance(artifacts, Mapping):
                continue
            for record in artifacts.values():
                if not isinstance(record, Mapping):
                    continue
                destination = Path(str(record.get("path", "")))
                expected = str(record.get("sha256", ""))
                try:
                    destination.parent.resolve().relative_to(workspace.root)
                    relative_destination = destination.absolute().relative_to(
                        workspace.root
                    )
                except (OSError, ValueError):
                    continue
                if destination.is_file() and file_sha256(destination) == expected:
                    continue
                source = next(
                    (
                        item
                        for item in candidates.get(expected, [])
                        if item.is_file() and file_sha256(item) == expected
                    ),
                    None,
                )
                if source is None:
                    continue
                if destination.exists() or destination.is_symlink():
                    quarantine = damaged_root / relative_destination
                    quarantine.parent.mkdir(parents=True, exist_ok=True)
                    destination.replace(quarantine)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".repair")
                shutil.copy2(source, temporary)
                temporary.replace(destination)
                if file_sha256(destination) != expected:
                    raise WorkflowError(
                        f"restored artifact failed its ledger hash: {destination}"
                    )
                repaired.append(str(destination))

    # GenerationController owns the ledger lock and the publication protocol.
    # Calling next_stage on completed generations is an idempotent projection
    # repair; it cannot accept a new scientific result.
    from .controller import PersistentController

    controller = PersistentController(preparation.output_dir)
    for plan in controller.plans:
        record = ledger.get("generations", {}).get(str(plan.generation), {})
        if isinstance(record, Mapping) and record.get("complete"):
            controller.generation_controller.next_stage(plan)
    return tuple(repaired)


def resume_workflow(
    output_dir: str | Path,
    *,
    foreground: bool = False,
    poll_interval: float | None = None,
) -> WorkflowResume:
    """Recover one workflow that has already been started."""

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
    if status.state == "prepared":
        raise WorkflowError(
            "workflow has not been started; use "
            f"neptrain workflow run {shlex.quote(str(output))}"
        )
    if status.state == "budget_exhausted":
        raise WorkflowError(
            "workflow exhausted its model-generation budget; use "
            f"neptrain workflow extend {shlex.quote(str(output))} "
            f"{len(preparation.plans) + 1}"
        )
    if status.state == "stalled":
        raise WorkflowError(
            "workflow is scientifically stalled; unchanged execution cannot "
            "make progress. Inspect the accepted generation and revise the "
            "sampling or evaluation policy before creating a new workflow"
        )
    if controller_running(output):
        raise WorkflowError("workflow is already running")
    if status.state == "damaged":
        repair_error = None
        try:
            _repair_damaged_workflow(preparation)
        except Exception as error:
            repair_error = str(error)
        repaired_status = workflow_status(output)
        if repaired_status.state == "damaged":
            detail = repaired_status.reason
            if repair_error:
                detail += f"; repair attempt failed: {repair_error}"
            raise WorkflowError(
                "workflow has authoritative damage and resume was refused "
                f"before launching work: {detail}. Restore the missing files "
                "from backup or create a new workflow"
            )
        status = repaired_status
        if status.state == "complete":
            return WorkflowResume(
                preparation.workflow_id,
                "repair",
                preparation.manifest,
            )
    controller = PersistentController(output)
    action = "resume"
    if status.state == "failed":
        controller.retry()
        action = "retry"
    elif status.state == "rejected":
        controller.retry(recover_rejected=True)
        action = "recover_rejected"
    elif status.state == "paused":
        controller.resume_stopped()
        action = "resume"
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


def start_workflow(
    output_dir: str | Path,
    *,
    foreground: bool = False,
    poll_interval: float | None = None,
) -> WorkflowResume:
    """Start exactly one prepared workflow and reject recovery states."""

    output = Path(output_dir).expanduser().resolve()
    preparation = _coerce_preparation(output)
    status = workflow_status(output)
    if status.state != "prepared":
        if status.state == "complete":
            next_action = f"neptrain workflow status {shlex.quote(str(output))}"
        elif status.state == "budget_exhausted":
            next_action = (
                f"neptrain workflow extend {shlex.quote(str(output))} "
                f"{len(preparation.plans) + 1}"
            )
        else:
            next_action = f"neptrain workflow resume {shlex.quote(str(output))}"
        raise WorkflowError(
            f"workflow run only starts a prepared workflow; current state is "
            f"{status.state}. Use {next_action}"
        )
    from .controller import controller_running, start_controller

    if controller_running(output):
        raise WorkflowError("workflow is already running")
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
        "start",
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
    "start_workflow",
]
