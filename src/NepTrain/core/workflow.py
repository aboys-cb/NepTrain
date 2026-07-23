"""High-level preparation and control for persistent iteration workflows."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Callable, Mapping, Sequence
import uuid

from .workflow_workspace import WorkflowWorkspace


class WorkflowError(RuntimeError):
    """Raised when workflow preparation or submission is inconsistent."""


class _SubmissionRejected(WorkflowError):
    """Raised when Slurm definitively rejected a submission."""


class _SubmissionUncertain(WorkflowError):
    """Raised when Slurm may have accepted a submission without returning its id."""


@dataclass(frozen=True)
class WorkflowPreparation:
    workflow_id: str
    output_dir: Path
    config_file: Path
    initial_training: Path
    plans: tuple[Path, ...]
    scripts: tuple[Path, ...]
    manifest: Path


@dataclass(frozen=True)
class WorkflowSubmission:
    workflow_id: str
    job_ids: tuple[str, ...]
    manifest: Path


@dataclass(frozen=True)
class WorkflowRetry:
    workflow_id: str
    retry_number: int
    from_generation: int
    from_stage: str
    job_ids: tuple[str, ...]
    manifest: Path


@dataclass(frozen=True)
class WorkflowResume:
    workflow_id: str
    action: str
    job_ids: tuple[str, ...]
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


SubmitRunner = Callable[[Sequence[str], Path], str]
JobStateRunner = Callable[[str, Path], str]
CancelRunner = Callable[[str, Path], None]
SubmissionResolver = Callable[[Mapping[str, Any], Path], str | None]


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
_ACTIVE_JOB_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "REQUEUED",
    "RESIZING",
    "STAGE_OUT",
    "SUSPENDED",
}
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
    md = resolved.get("md", {})
    for key in ("structures", "template_path", "plugin_path"):
        if md.get(key):
            md[key] = _absolute_path(md[key], base_dir)
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


def _slurm_profile(workflow: Mapping[str, Any], name: str) -> dict[str, Any]:
    profiles = workflow.get("slurm", {})
    selected = profiles.get(name)
    if name == "dft" and not selected:
        selected = profiles.get("cpu", {})
    profile = dict(selected or {})
    if not profile.get("partition"):
        raise WorkflowError(f"workflow.slurm.{name}.partition is required")
    profile.setdefault("time", "01:00:00")
    if name == "training":
        profile.setdefault("gpus_per_node", 1)
    elif name == "cpu" or (
        profile.get("gpus_per_node") is None
        and profile.get("cpus_per_task") is None
    ):
        profile.setdefault("cpus_per_task", 4)
    return profile


def _slurm_header(
    profile: Mapping[str, Any], *, job_name: str, output: Path
) -> list[str]:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output}",
        f"#SBATCH --time={profile['time']}",
        f"#SBATCH --partition={profile['partition']}",
    ]
    if profile.get("qos"):
        lines.append(f"#SBATCH --qos={profile['qos']}")
    if profile.get("gpus_per_node") is not None:
        lines.append(f"#SBATCH --gpus-per-node={int(profile['gpus_per_node'])}")
    if profile.get("cpus_per_task") is not None:
        lines.append(f"#SBATCH --cpus-per-task={int(profile['cpus_per_task'])}")
    for directive in profile.get("directives", []):
        directive = str(directive).strip()
        lines.append(
            directive if directive.startswith("#SBATCH ") else f"#SBATCH {directive}"
        )
    return lines


def _resource_command(
    command: str,
    *,
    config_file: Path,
    plan_file: Path,
    initial_training: Path,
    workflow_dir: Path,
    workflow_id: str,
    resource: str,
) -> str:
    tokens = [
        *shlex.split(command),
        "iteration-resource",
        str(config_file),
        "--plan",
        str(plan_file),
        "--initial-training",
        str(initial_training),
        "--workflow-dir",
        str(workflow_dir),
        "--workflow-id",
        workflow_id,
        "--resource",
        resource,
    ]
    if not shlex.split(command):
        raise WorkflowError("workflow.command cannot be empty")
    return shlex.join(tokens)


def _render_script(
    profile: Mapping[str, Any],
    *,
    job_name: str,
    log_file: Path,
    commands: Sequence[str],
    cpu: bool,
) -> str:
    lines = _slurm_header(profile, job_name=job_name, output=log_file)
    lines.extend(["", "set -euo pipefail"])
    setup_script = profile.get("setup_script")
    if setup_script:
        lines.append(f"source {shlex.quote(str(setup_script))}")
    if cpu:
        lines.append('export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"')
    lines.extend(["", *commands, ""])
    return "\n".join(lines)


def _plans(
    settings: Mapping[str, Any], md: Mapping[str, Any]
) -> tuple[Any, ...]:
    from .iteration import progressive_plans

    plans = progressive_plans(
        int(settings.get("generations", 1)),
        seed=int(settings.get("seed", 20260721)),
        initial_candidates=int(settings.get("initial_candidates", 24)),
        initial_budget=int(settings.get("dft_budget", 8)),
        minimum_budget=int(settings.get("minimum_dft_budget", 4)),
        initial_steps=int(md["initial_steps"]),
        temperatures=tuple(float(value) for value in md["temperatures"]),
        pressure=float(md.get("pressure", 0.0)),
        min_distance=float(settings.get("min_distance", 0.0)),
        frame_stride=int(settings.get("frame_stride", 2)),
    )
    return plans


def _dependencies(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths: list[tuple[str, Path]] = []
    training = config.get("training", {})
    md = config.get("md", {})
    dft = config.get("dft", {})
    evaluation = config.get("evaluation", {})
    for role, value in (
        ("training_config", training.get("config_path")),
        ("training_test", training.get("test_path")),
        ("md_structures", md.get("structures")),
        ("md_template", md.get("template_path")),
        ("md_plugin", md.get("plugin_path")),
        ("dft_input", dft.get("input_path")),
        ("dft_resources", dft.get("resource_path")),
        ("evaluation_validation", evaluation.get("validation_path")),
    ):
        if value and value != "auto":
            paths.append((role, Path(value)))
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
        scripts=tuple(Path(record["path"]) for record in manifest["scripts"]),
        manifest=path,
    )


def _write_workflow_scripts(
    *,
    output: Path,
    config_file: Path,
    initial_training: Path,
    workflow_id: str,
    plans: Sequence[Any],
    plan_paths: Sequence[Path],
    command: str,
    training_profile: Mapping[str, Any],
    cpu_profile: Mapping[str, Any],
    dft_profile: Mapping[str, Any],
) -> list[Path]:
    """Write one contiguous workflow segment, including its train bootstrap."""

    if not plans or len(plans) != len(plan_paths):
        raise WorkflowError("workflow script segment must contain matching plans")
    workspace = WorkflowWorkspace.locate(output)
    script_paths: list[Path] = []
    workflow_dir = workspace.controller_root
    logs = workspace.logs_dir
    logs.mkdir(parents=True, exist_ok=True)

    first_generation = plans[0].generation
    bootstrap_script = (
        workspace.jobs_dir / f"generation-{first_generation}-bootstrap.sbatch"
    )
    bootstrap_command = _resource_command(
        command,
        config_file=config_file,
        plan_file=plan_paths[0],
        initial_training=initial_training,
        workflow_dir=workflow_dir,
        workflow_id=workflow_id,
        resource="training",
    )
    _write_text(
        bootstrap_script,
        _render_script(
            training_profile,
            job_name=f"neptrain-g{first_generation}-bootstrap",
            log_file=logs / f"generation-{first_generation}-bootstrap-%j.out",
            commands=[bootstrap_command],
            cpu=False,
        ),
    )
    script_paths.append(bootstrap_script)

    for index, (plan, plan_path) in enumerate(zip(plans, plan_paths)):
        generation = plan.generation
        stage_profiles = (
            ("sample", "cpu", cpu_profile, True),
            (
                "label",
                "dft",
                dft_profile,
                dft_profile.get("cpus_per_task") is not None,
            ),
            ("merge", "cpu", cpu_profile, True),
            ("retrain", "training", training_profile, False),
        )
        for stage, resource, profile, cpu in stage_profiles:
            script = workspace.jobs_dir / f"generation-{generation}-{stage}.sbatch"
            stage_command = _resource_command(
                command,
                config_file=config_file,
                plan_file=plan_path,
                initial_training=initial_training,
                workflow_dir=workflow_dir,
                workflow_id=workflow_id,
                resource=resource,
            )
            _write_text(
                script,
                _render_script(
                    profile,
                    job_name=f"neptrain-g{generation}-{stage}",
                    log_file=logs / f"generation-{generation}-{stage}-%j.out",
                    commands=[stage_command],
                    cpu=cpu,
                ),
            )
            script_paths.append(script)

        evaluate_script = workspace.jobs_dir / f"generation-{generation}-evaluate.sbatch"
        evaluate_commands = [
            _resource_command(
                command,
                config_file=config_file,
                plan_file=plan_path,
                initial_training=initial_training,
                workflow_dir=workflow_dir,
                workflow_id=workflow_id,
                resource="cpu",
            )
        ]
        if index + 1 < len(plan_paths):
            evaluate_commands.append(
                _resource_command(
                    command,
                    config_file=config_file,
                    plan_file=plan_paths[index + 1],
                    initial_training=initial_training,
                    workflow_dir=workflow_dir,
                    workflow_id=workflow_id,
                    resource="training",
                )
            )
        _write_text(
            evaluate_script,
            _render_script(
                cpu_profile,
                job_name=f"neptrain-g{generation}-evaluate",
                log_file=logs / f"generation-{generation}-evaluate-%j.out",
                commands=evaluate_commands,
                cpu=True,
            ),
        )
        script_paths.append(evaluate_script)
    return script_paths


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
    labeling_target_name = config["execution"]["routes"]["labeling"]
    labeling_target = config["execution"]["targets"][labeling_target_name]
    target_overrides = labeling_target.get("overrides", {})
    dft_backend = target_overrides.get("dft.backend", config["dft"]["backend"])
    dft_resource = target_overrides.get(
        "dft.resource_path", config["dft"].get("resource_path")
    )
    if dft_backend in {"vasp", "abacus"} and not dft_resource:
        raise WorkflowError(
            f"{dft_backend} workflows require dft.resource_path or the "
            "labeling target override dft.resource_path"
        )
    settings = config.get("workflow", {})
    selected_id = workflow_id or str(settings.get("id", output.name))
    if not selected_id.strip():
        raise WorkflowError("workflow id cannot be empty")
    plans = _plans(settings, config["md"])
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
        if existing.get("spec_sha256") != spec_sha256:
            raise WorkflowError(
                "workflow preparation changed; choose a new output directory or "
                "restore the original inputs"
            )
        for record in [
            existing["config"],
            existing["initial_training"],
            *existing["plans"],
            *existing["scripts"],
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
    plan_paths: list[Path] = []
    for plan in plans:
        plan_path = workspace.plans_dir / f"generation-{plan.generation}.json"
        _write_json(plan_path, asdict(plan))
        plan_paths.append(plan_path)
    script_paths: list[Path] = []

    manifest = {
        "version": 3,
        "layout_version": workspace.version,
        "orchestration": "controller-v1",
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
        "scripts": [
            {"path": str(path), "sha256": _sha256(path)} for path in script_paths
        ],
        "dependencies": dependencies,
        "jobs": [],
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
        portable_config.setdefault("workflow", {})["generations"] = int(
            total_generations
        )
        config = _resolved_config(
            portable_config, preparation.config_file.parent
        )
        settings = dict(config.get("workflow", {}))
        all_plans = _plans(settings, config["md"])
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

        if manifest.get("orchestration") == "controller-v1":
            new_scripts = []
        else:
            training_profile = _slurm_profile(settings, "training")
            cpu_profile = _slurm_profile(settings, "cpu")
            dft_profile = _slurm_profile(settings, "dft")
            new_scripts = _write_workflow_scripts(
                output=preparation.output_dir,
                config_file=preparation.config_file,
                initial_training=preparation.initial_training,
                workflow_id=preparation.workflow_id,
                plans=new_plans,
                plan_paths=new_plan_paths,
                command=str(settings.get("command", "neptrain")),
                training_profile=training_profile,
                cpu_profile=cpu_profile,
                dft_profile=dft_profile,
            )
        manifest["plans"].extend(
            {"path": str(path), "sha256": _sha256(path)} for path in new_plan_paths
        )
        manifest["scripts"].extend(
            {"path": str(path), "sha256": _sha256(path)} for path in new_scripts
        )
        extension = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "from_generations": current_total,
            "to_generations": total_generations,
        }
        if new_scripts:
            extension["script_start_index"] = len(manifest["scripts"]) - len(new_scripts)
        manifest.setdefault("extensions", []).append(extension)
        save_config(portable_config, preparation.config_file)
        manifest["config"]["sha256"] = _sha256(preparation.config_file)
        _write_json(preparation.manifest, manifest)
        return _preparation_from_manifest(preparation.manifest)


def _submit(args: Sequence[str], cwd: Path) -> str:
    if os.environ.get("SLURM_JOB_ID"):
        raise _SubmissionRejected(
            "NepTrain run must execute on the Slurm login node, not inside a batch job"
        )
    completed = subprocess.run(
        list(args), cwd=cwd, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise _SubmissionRejected(f"Slurm submission failed: {detail}")
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise _SubmissionUncertain(
            f"cannot parse Slurm job id: {completed.stdout.strip()}"
        )
    return job_id


def _submission_token(workflow_id: str, script: Path) -> str:
    workflow = hashlib.sha256(workflow_id.encode()).hexdigest()[:8]
    stage = "".join(character for character in script.stem if character.isalnum())[:24]
    return f"nt-{workflow}-{stage}-{uuid.uuid4().hex[:12]}"


def _resolve_submission(record: Mapping[str, Any], cwd: Path) -> str | None:
    """Find a write-ahead submission intent in Slurm by its unique job name."""

    if os.environ.get("SLURM_JOB_ID"):
        raise WorkflowError(
            "workflow submission reconciliation must run on the Slurm login node"
        )
    token = str(record.get("submission_token", ""))
    if not token:
        raise WorkflowError("pending workflow job is missing its submission token")
    submitted_at = datetime.fromisoformat(str(record["submitted_at"]))
    accounting_start = (submitted_at - timedelta(days=1)).date().isoformat()
    commands = [
        ["squeue", "--noheader", "--name", token, "--format", "%A|%j"],
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--starttime",
            accounting_start,
            "--name",
            token,
            "--format",
            "JobIDRaw,JobName",
        ],
    ]
    job_ids = set()
    for command in commands:
        completed = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise WorkflowError(
                f"cannot reconcile Slurm submission {token}: {detail}"
            )
        for line in completed.stdout.splitlines():
            columns = line.strip().split("|")
            if len(columns) >= 2 and columns[0].isdigit() and columns[1] == token:
                job_ids.add(columns[0])
    if len(job_ids) > 1:
        raise WorkflowError(
            f"submission token {token} matched multiple Slurm jobs: {sorted(job_ids)}"
        )
    return next(iter(job_ids), None)


def _reconcile_submission_records(
    records: list[dict[str, Any]],
    scripts: Sequence[Path],
    *,
    output: Path,
    resolver: SubmissionResolver,
    persist: Callable[[], None],
    chain_starts: set[int] | None = None,
) -> str | None:
    expected = [str(path) for path in scripts]
    if [record.get("script") for record in records] != expected[: len(records)]:
        raise WorkflowError("workflow job history is not a valid script prefix")
    starts = {0, *(chain_starts or set())}
    dependency = None
    for index, record in enumerate(records):
        if index in starts:
            dependency = None
        if record.get("dependency") != dependency:
            raise WorkflowError("workflow job history has an invalid dependency chain")
        job_id = record.get("job_id")
        if job_id is None:
            job_id = resolver(record, output)
            if job_id is None:
                raise WorkflowError(
                    "submission outcome is still uncertain; Slurm has no job for "
                    f"token {record.get('submission_token')}. Retry after accounting updates."
                )
            record["job_id"] = str(job_id)
            record["submission_state"] = "submitted"
            record["reconciled_at"] = datetime.now(timezone.utc).isoformat()
            persist()
        if not str(job_id).isdigit():
            raise WorkflowError(f"invalid Slurm job id in workflow history: {job_id}")
        dependency = str(job_id)
    return dependency


def _submit_missing_records(
    records: list[dict[str, Any]],
    scripts: Sequence[Path],
    *,
    workflow_id: str,
    output: Path,
    runner: SubmitRunner,
    resolver: SubmissionResolver,
    persist: Callable[[], None],
    chain_starts: set[int] | None = None,
) -> list[dict[str, Any]]:
    starts = {0, *(chain_starts or set())}
    dependency = _reconcile_submission_records(
        records,
        scripts,
        output=output,
        resolver=resolver,
        persist=persist,
        chain_starts=starts,
    )
    for index, script in enumerate(scripts[len(records) :], start=len(records)):
        if index in starts:
            dependency = None
        token = _submission_token(workflow_id, script)
        record = {
            "script": str(script),
            "dependency": dependency,
            "submission_token": token,
            "submission_state": "intent",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        records.append(record)
        persist()
        args = ["sbatch", "--parsable", f"--job-name={token}"]
        if dependency is not None:
            args.append(f"--dependency=afterok:{dependency}")
        args.append(str(script))
        try:
            job_id = runner(args, output)
        except _SubmissionRejected:
            records.pop()
            persist()
            raise
        if not str(job_id).isdigit():
            raise _SubmissionUncertain(
                f"submission {token} returned invalid Slurm job id {job_id}"
            )
        record["job_id"] = str(job_id)
        record["submission_state"] = "submitted"
        persist()
        dependency = str(job_id)
    return records


def _submission_chain_starts(
    manifest: Mapping[str, Any], scripts: Sequence[Path]
) -> set[int]:
    """Return scheduler-chain reset points created by accepted extensions."""

    names = [path.name for path in scripts]
    starts = {0}
    for extension in manifest.get("extensions", []):
        index = extension.get("script_start_index")
        if index is None:
            generation = int(extension["from_generations"]) + 1
            bootstrap = f"generation-{generation}-bootstrap.sbatch"
            try:
                index = names.index(bootstrap)
            except ValueError as error:
                raise WorkflowError(
                    f"workflow extension is missing scheduler bootstrap {bootstrap}"
                ) from error
        starts.add(int(index))
    return starts


def _validated_manifest(preparation: WorkflowPreparation) -> dict[str, Any]:
    manifest = json.loads(preparation.manifest.read_text(encoding="utf-8"))
    for record in [
        manifest["config"],
        manifest["initial_training"],
        *manifest["plans"],
        *manifest["scripts"],
        *manifest.get("dependencies", []),
    ]:
        if not _record_matches(record):
            raise WorkflowError(
                f"prepared workflow artifact drifted: {record['path']}"
            )
    return manifest


def submit_workflow(
    preparation: WorkflowPreparation | str | Path,
    *,
    runner: SubmitRunner = _submit,
) -> WorkflowSubmission:
    """Submit the prepared scripts as one strict afterok dependency chain."""

    preparation = _coerce_preparation(preparation)
    manifest = json.loads(preparation.manifest.read_text(encoding="utf-8"))
    if manifest.get("orchestration") == "controller-v1":
        raise WorkflowError(
            "controller workflows do not submit a Slurm dependency chain; "
            "start the persistent controller instead"
        )
    with _workflow_lock(preparation.output_dir):
        return _submit_workflow_locked(
            preparation,
            runner=runner,
            submission_resolver=_resolve_submission,
        )


def _submit_workflow_locked(
    preparation: WorkflowPreparation,
    *,
    runner: SubmitRunner,
    submission_resolver: SubmissionResolver,
) -> WorkflowSubmission:
    manifest = _validated_manifest(preparation)
    jobs = list(manifest.get("jobs", []))

    def persist() -> None:
        manifest["jobs"] = jobs
        _write_json(preparation.manifest, manifest)

    _submit_missing_records(
        jobs,
        preparation.scripts,
        workflow_id=preparation.workflow_id,
        output=preparation.output_dir,
        runner=runner,
        resolver=submission_resolver,
        persist=persist,
        chain_starts=_submission_chain_starts(manifest, preparation.scripts),
    )
    return WorkflowSubmission(
        workflow_id=preparation.workflow_id,
        job_ids=tuple(record["job_id"] for record in jobs),
        manifest=preparation.manifest,
    )


def _normalise_job_state(value: str) -> str:
    return value.strip().split("+", 1)[0].split(maxsplit=1)[0].upper()


def _job_state(job_id: str, cwd: Path) -> str:
    if os.environ.get("SLURM_JOB_ID"):
        raise WorkflowError(
            "NepTrain resume must execute on the Slurm login node, not inside a batch job"
        )
    completed = subprocess.run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            job_id,
            "--format",
            "JobIDRaw,State",
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise WorkflowError(f"cannot query Slurm job {job_id}: {detail}")
    fallback = None
    for line in completed.stdout.splitlines():
        columns = line.strip().split("|")
        if len(columns) < 2:
            continue
        fallback = fallback or columns[1]
        if columns[0] == job_id:
            return _normalise_job_state(columns[1])
    if fallback:
        return _normalise_job_state(fallback)
    raise WorkflowError(f"Slurm has no accounting record for job {job_id}")


def _cancel(job_id: str, cwd: Path) -> None:
    if os.environ.get("SLURM_JOB_ID"):
        raise WorkflowError(
            "NepTrain resume must execute on the Slurm login node, not inside a batch job"
        )
    completed = subprocess.run(
        ["scancel", job_id],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise WorkflowError(f"cannot cancel stale Slurm job {job_id}: {detail}")


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
            "candidate_target": int(plan["candidate_count"]),
            "dft_budget": int(plan["dft_budget"]),
            "steps": int(plan["steps"]),
            "temperatures": tuple(float(value) for value in plan["temperatures"]),
            "pressure": float(plan.get("pressure", 0.0)),
            "frame_stride": int(plan.get("frame_stride", 1)),
        },
        "sampling": {
            "candidate_count": explore.get("candidate_count"),
            "candidate_counts_by_window": dict(
                explore.get("candidate_counts_by_window", {})
            ),
            "scheduled_source_count": explore.get("scheduled_source_count"),
            "completed_source_count": explore.get("completed_source_count"),
            "failed_source_count": explore.get("failed_source_count"),
            "candidate_count_before_thinning": select.get(
                "candidate_count_before_thinning"
            ),
            "candidate_count_after_thinning": select.get(
                "candidate_count_after_thinning"
            ),
            "duplicate_candidate_count": select.get("duplicate_candidate_count"),
            "selected_count": select.get("selected_count"),
            "counts_by_stratum": dict(select.get("counts_by_stratum", {})),
            "labeled_count": label.get("labeled_count"),
        },
        "training": {
            "before_count": train.get("training_count"),
            "merged_count": merge.get("training_count"),
            "after_count": retrain.get("training_count"),
            "added_count": evaluate.get("added_training_count", merge.get("added_count")),
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


def _workflow_breakpoint(
    preparation: WorkflowPreparation, manifest: Mapping[str, Any]
) -> tuple[int, str]:
    progress = _workflow_progress(preparation, manifest)
    if progress.state in {"prepared", "incomplete"}:
        assert progress.generation is not None and progress.stage is not None
        return progress.generation, progress.stage
    if progress.state == "rejected":
        raise WorkflowError(
            f"{progress.reason}; retry cannot bypass the acceptance gate"
        )
    raise WorkflowError("workflow is already complete; nothing to retry")


def _retry_script_index(
    preparation: WorkflowPreparation, generation: int, stage: str
) -> int:
    if stage == "train":
        bootstrap = f"generation-{generation}-bootstrap.sbatch"
        filename = (
            bootstrap
            if any(script.name == bootstrap for script in preparation.scripts)
            else f"generation-{generation - 1}-evaluate.sbatch"
        )
    elif stage in {"explore", "select", "label", "diagnose", "merge"}:
        if stage in {"explore", "select"}:
            filename = f"generation-{generation}-sample.sbatch"
        elif stage == "label":
            filename = f"generation-{generation}-label.sbatch"
        else:
            filename = f"generation-{generation}-merge.sbatch"
    else:
        filename = f"generation-{generation}-{stage}.sbatch"
    for index, script in enumerate(preparation.scripts):
        if script.name == filename:
            return index
    raise WorkflowError(
        f"prepared workflow has no job script for generation {generation} stage {stage}"
    )


def _job_records(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = list(manifest.get("jobs", []))
    for retry in manifest.get("retries", []):
        records.extend(retry.get("jobs", []))
    return records


def _retry_result(
    preparation: WorkflowPreparation, retry: Mapping[str, Any]
) -> WorkflowRetry:
    return WorkflowRetry(
        workflow_id=preparation.workflow_id,
        retry_number=int(retry["retry"]),
        from_generation=int(retry["from_generation"]),
        from_stage=str(retry["from_stage"]),
        job_ids=tuple(str(record["job_id"]) for record in retry.get("jobs", [])),
        manifest=preparation.manifest,
    )


def _status_job_records(
    preparation: WorkflowPreparation, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    records: list[tuple[str, Mapping[str, Any]]] = [
        ("initial", record) for record in manifest.get("jobs", [])
    ]
    for retry in manifest.get("retries", []):
        attempt = f"retry-{retry['retry']}"
        records.extend((attempt, record) for record in retry.get("jobs", []))
    latest_index = {
        str(record["script"]): index
        for index, (_, record) in enumerate(records)
    }
    statuses = []
    seen_scripts = set()
    for index, (attempt, record) in enumerate(records):
        script = str(record["script"])
        seen_scripts.add(script)
        job_id = record.get("job_id")
        detail = None
        if job_id is None:
            try:
                job_id = _resolve_submission(record, preparation.output_dir)
            except WorkflowError as error:
                state = "UNKNOWN"
                detail = str(error)
            else:
                state = "SUBMISSION_UNCERTAIN" if job_id is None else "UNKNOWN"
        else:
            state = "UNKNOWN"
        if job_id is not None:
            try:
                state = _normalise_job_state(
                    _job_state(str(job_id), preparation.output_dir)
                )
            except WorkflowError as error:
                detail = str(error)
        statuses.append(
            {
                "attempt": attempt,
                "script": script,
                "job_id": str(job_id) if job_id is not None else None,
                "dependency": record.get("dependency"),
                "state": state,
                "current": latest_index[script] == index,
                "detail": detail,
            }
        )
    for script in preparation.scripts:
        if str(script) not in seen_scripts:
            statuses.append(
                {
                    "attempt": None,
                    "script": str(script),
                    "job_id": None,
                    "dependency": None,
                    "state": "NOT_SUBMITTED",
                    "current": True,
                    "detail": None,
                }
            )
    return tuple(statuses)


def workflow_status(output_dir: str | Path) -> WorkflowStatus:
    """Return a read-only ledger and Slurm summary for one prepared workflow."""

    preparation = _coerce_preparation(output_dir)
    manifest = _validated_manifest(preparation)
    ledger = _read_workflow_ledger(preparation)
    progress = _workflow_progress(preparation, manifest, ledger=ledger)
    generations = _scientific_progress(preparation, manifest, ledger=ledger)
    if manifest.get("orchestration") == "controller-v1":
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
            handle = item.get("handle") or {}
            if item.get("completed_at"):
                execution_state = "COMPLETED"
            elif item.get("cancelled_at"):
                execution_state = "CANCELLED"
            else:
                execution_state = "FAILED"
            jobs.append(
                {
                    "attempt": f"attempt-{item.get('attempt', 1)}",
                    "script": f"{item.get('target', '-')}/{item.get('stage', '-')}",
                    "job_id": handle.get("execution_id"),
                    "dependency": None,
                    "state": execution_state,
                    "current": False,
                    "detail": item.get("failure")
                    or (item.get("cancellation") or {}).get("detail"),
                }
            )
        if current:
            handle = current.get("handle") or {}
            observed = str(current.get("observed_state", controller_state)).upper()
            jobs.append(
                {
                    "attempt": f"attempt-{current.get('attempt', 1)}",
                    "script": f"{current.get('target', '-')}/{current.get('stage', '-')}",
                    "job_id": handle.get("execution_id"),
                    "dependency": None,
                    "state": observed,
                    "current": True,
                    "detail": current.get("detail"),
                }
            )
        state = progress.state
        reason = progress.reason
        next_action = None
        if progress.state == "complete":
            state = "complete"
        elif progress.state == "rejected" or controller_state == "rejected":
            state = "rejected"
            reason = str(controller.get("reason", progress.reason))
            next_action = f"neptrain workflow resume {shlex.quote(str(preparation.output_dir))}"
        elif active:
            state = "degraded" if controller_state == "degraded" else "running"
            reason = str(
                controller.get("last_transport_error")
                or controller.get("reason")
                or "persistent controller is active"
            )
            next_action = f"neptrain workflow status {shlex.quote(str(preparation.output_dir))}"
        elif controller_state == "failed":
            state = "failed"
            reason = str(controller.get("reason", "controller failed"))
            next_action = f"neptrain workflow resume {shlex.quote(str(preparation.output_dir))}"
        elif controller_state == "stopped":
            state = "paused"
            reason = str(controller.get("reason", "controller is stopped"))
            next_action = f"neptrain workflow resume {shlex.quote(str(preparation.output_dir))}"
        elif controller_state in {"running", "launching", "degraded"}:
            state = "paused"
            reason = "controller process is not running; remote work is preserved"
            next_action = f"neptrain workflow resume {shlex.quote(str(preparation.output_dir))}"
        else:
            state = "prepared"
            reason = "workflow is prepared and controller has not started"
            next_action = f"neptrain workflow run {shlex.quote(str(preparation.output_dir))}"
        return WorkflowStatus(
            workflow_id=preparation.workflow_id,
            state=state,
            completed_generations=progress.completed_generations,
            total_generations=len(preparation.plans),
            generation=int(current["generation"])
            if current
            else progress.generation,
            stage=str(current["stage"]) if current else progress.stage,
            reason=reason,
            next_action=next_action,
            generations=generations,
            jobs=tuple(jobs),
        )
    jobs = _status_job_records(preparation, manifest)
    next_action = None
    state = progress.state
    reason = progress.reason
    output = preparation.output_dir

    if progress.state in {"prepared", "incomplete"}:
        assert progress.generation is not None and progress.stage is not None
        target_index = _retry_script_index(
            preparation, progress.generation, progress.stage
        )
        target_script = str(preparation.scripts[target_index])
        target = next(
            job
            for job in reversed(jobs)
            if job["script"] == target_script and job["current"]
        )
        current_by_id = {
            job["job_id"]: job
            for job in jobs
            if job["current"] and job["job_id"] is not None
        }
        active = [
            job
            for job in jobs
            if job["current"] and job["state"] in _ACTIVE_JOB_STATES
        ]
        target_state = str(target["state"])
        dependency = current_by_id.get(target.get("dependency"))
        dependency_failed = (
            target_state == "PENDING"
            and dependency is not None
            and dependency["state"] not in _ACTIVE_JOB_STATES
            and dependency["state"] != "COMPLETED"
        )
        if dependency_failed:
            state = "blocked"
            reason = (
                f"{Path(target_script).name} is blocked by failed job "
                f"{dependency['job_id']}"
            )
            next_action = (
                f"NepTrain resume {shlex.quote(str(output))}"
            )
        elif target_state in _ACTIVE_JOB_STATES or active:
            state = "running"
            reason = f"{len(active)} current Slurm job(s) are active"
            next_action = (
                f"NepTrain status {shlex.quote(str(output))}"
            )
        elif target_state in {"NOT_SUBMITTED"}:
            state = "prepared" if not manifest.get("jobs") else "paused"
            reason = f"{Path(target_script).name} has not been submitted"
            next_action = (
                f"NepTrain run {shlex.quote(str(output))}"
            )
        elif target_state in {"SUBMISSION_UNCERTAIN", "UNKNOWN"}:
            state = "uncertain"
            reason = target.get("detail") or (
                f"Slurm state for {Path(target_script).name} is not yet known"
            )
            next_action = (
                f"NepTrain status {shlex.quote(str(output))}"
            )
        elif target_state == "COMPLETED":
            state = "inconsistent"
            reason = (
                f"{Path(target_script).name} completed but ledger still expects "
                f"stage {progress.stage}"
            )
        else:
            state = "failed"
            reason = f"job {target.get('job_id')} ended in state {target_state}"
            next_action = (
                f"NepTrain resume {shlex.quote(str(output))}"
            )
    elif progress.state == "rejected":
        state = "rejected"
        reason = progress.reason
        next_action = (
            f"NepTrain resume {shlex.quote(str(output))}"
        )

    return WorkflowStatus(
        workflow_id=preparation.workflow_id,
        state=state,
        completed_generations=progress.completed_generations,
        total_generations=len(preparation.plans),
        generation=progress.generation,
        stage=progress.stage,
        reason=reason,
        next_action=next_action,
        generations=generations,
        jobs=jobs,
    )


def resume_workflow(
    output_dir: str | Path,
    *,
    foreground: bool = False,
    poll_interval: float | None = None,
) -> WorkflowResume:
    """Take the only safe continuation without exposing scheduler failure modes."""

    output = Path(output_dir).expanduser().resolve()
    preparation = _coerce_preparation(output)
    manifest = json.loads(preparation.manifest.read_text(encoding="utf-8"))
    if manifest.get("orchestration") == "controller-v1":
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
                (),
                preparation.manifest,
            )
        if controller_running(output):
            raise WorkflowError("workflow controller is already running")
        controller = PersistentController(output)
        action = "resume"
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
            (),
            preparation.manifest,
            controller_pid=None if foreground else controller_result,
            controller_exit_code=controller_result if foreground else None,
        )
    status = workflow_status(output)
    if status.state in {"prepared", "paused"}:
        submission = submit_workflow(output)
        return WorkflowResume(
            submission.workflow_id,
            "submit",
            submission.job_ids,
            submission.manifest,
        )
    if status.state in {"failed", "blocked"}:
        retry = retry_failed_workflow(output)
        return WorkflowResume(
            retry.workflow_id,
            "retry",
            retry.job_ids,
            retry.manifest,
        )
    if status.state == "rejected":
        retry = retry_failed_workflow(output, recover_rejected=True)
        return WorkflowResume(
            retry.workflow_id,
            "recover_rejected",
            retry.job_ids,
            retry.manifest,
        )
    if status.state == "complete":
        return WorkflowResume(
            preparation.workflow_id,
            "complete",
            (),
            preparation.manifest,
        )
    if status.state == "running":
        raise WorkflowError("workflow is already running")
    raise WorkflowError(
        f"workflow cannot resume safely while state is {status.state}: {status.reason}"
    )


def retry_failed_workflow(
    output_dir: str | Path,
    *,
    recover_rejected: bool = False,
    runner: SubmitRunner = _submit,
    state_runner: JobStateRunner = _job_state,
    cancel_runner: CancelRunner = _cancel,
) -> WorkflowRetry:
    """Resume a prepared workflow from its first unfinished ledger stage.

    Scheduler history remains append-only. Pending jobs from the obsolete
    dependency tail are canceled before a fresh strict afterok chain is made.
    """

    output = Path(output_dir).expanduser().resolve()
    try:
        workspace = WorkflowWorkspace.locate(output)
    except FileNotFoundError as error:
        raise WorkflowError(
            f"prepared workflow does not exist: {output}"
        ) from error
    if not workspace.manifest.is_file():
        raise WorkflowError(
            f"prepared workflow manifest does not exist: {workspace.manifest}"
        )
    with _workflow_lock(output):
        return _retry_failed_workflow_locked(
            output,
            runner=runner,
            state_runner=state_runner,
            cancel_runner=cancel_runner,
            submission_resolver=_resolve_submission,
            recover_rejected=recover_rejected,
        )


def _retry_failed_workflow_locked(
    output: Path,
    *,
    runner: SubmitRunner,
    state_runner: JobStateRunner,
    cancel_runner: CancelRunner,
    submission_resolver: SubmissionResolver,
    recover_rejected: bool = False,
) -> WorkflowRetry:
    manifest_path = WorkflowWorkspace.locate(output).manifest
    preparation = _preparation_from_manifest(manifest_path)
    manifest = _validated_manifest(preparation)
    original_jobs = list(manifest.get("jobs", []))

    def persist_original() -> None:
        manifest["jobs"] = original_jobs
        _write_json(manifest_path, manifest)

    _reconcile_submission_records(
        original_jobs,
        preparation.scripts,
        output=output,
        resolver=submission_resolver,
        persist=persist_original,
        chain_starts=_submission_chain_starts(manifest, preparation.scripts),
    )
    recovery_started = False
    progress = _workflow_progress(preparation, manifest)
    if recover_rejected:
        if progress.state != "rejected" or progress.generation is None:
            raise WorkflowError(
                "rejected recovery requires a completed rejected generation"
            )
        from .iteration import GenerationController, GenerationPlan

        value = json.loads(
            preparation.plans[progress.generation - 1].read_text(encoding="utf-8")
        )
        value["temperatures"] = tuple(value["temperatures"])
        controller = GenerationController(
            WorkflowWorkspace.locate(output).controller_root,
            preparation.workflow_id,
        )
        try:
            controller.reopen_rejected(
                GenerationPlan(**value), from_stage="retrain"
            )
        except Exception as error:
            raise WorkflowError(f"cannot reopen rejected generation: {error}") from error
        recovery_started = True
    generation, stage = _workflow_breakpoint(preparation, manifest)
    start = _retry_script_index(preparation, generation, stage)
    remaining = preparation.scripts[start:]
    target = str(remaining[0])

    retries = list(manifest.get("retries", []))
    if retries and retries[-1].get("from_script") == target:
        latest_retry = retries[-1]
        retry_jobs = list(latest_retry.get("jobs", []))

        def persist_retry() -> None:
            latest_retry["jobs"] = retry_jobs
            manifest["retries"] = retries
            _write_json(manifest_path, manifest)

        _reconcile_submission_records(
            retry_jobs,
            remaining,
            output=output,
            resolver=submission_resolver,
            persist=persist_retry,
        )
        states = [
            _normalise_job_state(state_runner(str(record["job_id"]), output))
            for record in retry_jobs
        ]
        failed = any(
            state not in _ACTIVE_JOB_STATES and state != "COMPLETED"
            for state in states
        )
        if not failed:
            if len(retry_jobs) == len(remaining):
                if states and all(state == "COMPLETED" for state in states):
                    raise WorkflowError(
                        "retry jobs completed but the workflow ledger did not "
                        "advance; inspect the job logs"
                    )
                return _retry_result(preparation, latest_retry)
            _submit_missing_records(
                retry_jobs,
                remaining,
                workflow_id=preparation.workflow_id,
                output=output,
                runner=runner,
                resolver=submission_resolver,
                persist=persist_retry,
            )
            return _retry_result(preparation, latest_retry)

    latest_by_script: dict[str, Mapping[str, Any]] = {}
    for record in _job_records(manifest):
        latest_by_script[str(record["script"])] = record
    if target not in latest_by_script:
        raise WorkflowError(
            "the unfinished stage has no submitted job history; use NepTrain run"
        )
    replaced = []
    for script in remaining:
        record = latest_by_script.get(str(script))
        if record is None:
            continue
        job_id = str(record["job_id"])
        state = _normalise_job_state(state_runner(job_id, output))
        replaced.append({"script": str(script), "job_id": job_id, "state": state})
        if script == remaining[0]:
            dependency = record.get("dependency")
            blocked_by_failed_dependency = False
            if state == "PENDING" and dependency is not None:
                dependency_state = _normalise_job_state(
                    state_runner(str(dependency), output)
                )
                blocked_by_failed_dependency = (
                    dependency_state not in _ACTIVE_JOB_STATES
                    and dependency_state != "COMPLETED"
                )
                replaced[-1]["dependency_state"] = dependency_state
            if blocked_by_failed_dependency:
                cancel_runner(job_id, output)
            elif state in _ACTIVE_JOB_STATES:
                raise WorkflowError(
                    f"job {job_id} for the unfinished stage is still {state}; "
                    "retry is not needed"
                )
            if state == "COMPLETED":
                if not recovery_started:
                    raise WorkflowError(
                        f"job {job_id} completed but the workflow ledger did not "
                        "advance; inspect the job log"
                    )
        elif state == "PENDING":
            cancel_runner(job_id, output)
        elif state in _ACTIVE_JOB_STATES:
            raise WorkflowError(
                f"downstream job {job_id} is already {state}; refusing concurrent recovery"
            )

    retry = {
        "retry": len(retries) + 1,
        "from_generation": generation,
        "from_stage": stage,
        "from_script": target,
        "replaced_jobs": replaced,
        "jobs": [],
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "recovery_of_rejected_generation": recovery_started,
    }
    retries.append(retry)
    manifest["retries"] = retries
    _write_json(manifest_path, manifest)

    retry_jobs = []

    def persist_retry() -> None:
        retry["jobs"] = retry_jobs
        _write_json(manifest_path, manifest)

    _submit_missing_records(
        retry_jobs,
        remaining,
        workflow_id=preparation.workflow_id,
        output=output,
        runner=runner,
        resolver=submission_resolver,
        persist=persist_retry,
    )
    return _retry_result(preparation, retry)


__all__ = [
    "WorkflowError",
    "WorkflowPreparation",
    "WorkflowResume",
    "WorkflowRetry",
    "WorkflowStatus",
    "WorkflowSubmission",
    "workflow_status",
    "extend_workflow",
    "prepare_workflow",
    "resume_workflow",
    "retry_failed_workflow",
    "submit_workflow",
]
