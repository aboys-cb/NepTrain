"""Durable manual-step runs with local and Slurm execution.

The public Interface is one operation directory. Scientific Adapters do not
know whether their request was run in the foreground, as a Slurm job, or as a
Slurm array.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from typing import Any, Mapping, Sequence

from ase.io import read as ase_read
from ase.io import write as ase_write

from .execution import ExecutionTarget, ExecutionTransport


class ManualTaskError(RuntimeError):
    """Raised when a manual step cannot be prepared, run, or collected."""


_SLURM_ACTIVE = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "STAGE_OUT",
    "SUSPENDED",
}
_SLURM_FAILURE = {
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


@dataclass(frozen=True)
class ManualOperation:
    root: Path
    operation_id: str
    kind: str
    target: ExecutionTarget
    shard_count: int
    jobs_directory: str = "jobs"

    @property
    def descriptor(self) -> Path:
        return self.root / "operation.json"

    @property
    def jobs_root(self) -> Path:
        return self.root / self.jobs_directory


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(message: str) -> None:
    print(f"[NepTrain] {message}", file=sys.stderr, flush=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManualTaskError(f"cannot read task metadata: {path}") from error


def _copy(source: Path, destination: Path) -> str:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise ManualTaskError(f"input file does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def _frames(source: Path) -> list:
    source = source.expanduser().resolve()
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
    frames = []
    for path in paths:
        loaded = ase_read(path, index=":")
        frames.extend(loaded if isinstance(loaded, list) else [loaded])
    if not frames:
        raise ManualTaskError(f"no readable structures found in {source}")
    return frames


def target_from_project(
    project: str | Path | None,
    target_name: str | None,
    *,
    route: str,
) -> ExecutionTarget:
    if project is None:
        if target_name not in {None, "local"}:
            raise ManualTaskError("--target requires --project")
        return ExecutionTarget("local", "process")
    from .config import load_config

    project_path = Path(project).expanduser().resolve()
    config, _ = load_config(project_path)
    execution = config["execution"]
    name = target_name or execution["stage_targets"][route]
    try:
        raw = dict(execution["targets"][name])
    except KeyError as error:
        raise ManualTaskError(f"unknown execution target: {name}") from error
    setup = raw.get("setup_script")
    if setup and not Path(str(setup)).expanduser().is_absolute():
        candidate = (project_path.parent / str(setup)).resolve()
        if candidate.is_file():
            raw["setup_script"] = str(candidate)
    return ExecutionTarget.from_mapping(name, raw)


def _operation_id(kind: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"{kind}-{stamp}-{digest}"


def _prepare_root(
    kind: str,
    workdir: str | Path | None,
    payload: Mapping[str, Any],
) -> tuple[Path, str]:
    operation_id = _operation_id(kind, payload)
    root = (
        Path(workdir).expanduser().resolve()
        if workdir
        else (Path.cwd() / "runs" / operation_id).resolve()
    )
    if root.exists():
        raise ManualTaskError(f"run directory already exists: {root}")
    (root / "jobs").mkdir(parents=True)
    (root / "logs").mkdir()
    return root, operation_id


def _output_path(value: str | Path, *, force: bool) -> Path:
    output = Path(value).expanduser().resolve()
    if output.exists() and not force:
        raise ManualTaskError(
            f"output already exists: {output}; pass --force to replace it"
        )
    if output.is_dir():
        raise ManualTaskError(f"output must be a file path: {output}")
    return output


def _write_operation(
    root: Path,
    *,
    operation_id: str,
    kind: str,
    target: ExecutionTarget,
    output: Path,
    jobs: Sequence[Mapping[str, Any]],
    max_concurrent: int,
) -> ManualOperation:
    if max_concurrent < 1:
        raise ManualTaskError("max_concurrent must be at least 1")
    setup = Path(target.setup_script).expanduser() if target.setup_script else None
    if target.host and setup and setup.is_file():
        packaged = root / "inputs" / "setup.sh"
        _copy(setup, packaged)
        target = replace(target, setup_script="./inputs/setup.sh")
    value = {
        "protocol": "neptrain.manual-operation.v2",
        "operation_id": operation_id,
        "kind": kind,
        "created_at": _now(),
        "state": "prepared",
        "target": asdict(target),
        "output": str(output.expanduser().resolve()),
        "max_concurrent": int(max_concurrent),
        "jobs": [dict(item) for item in jobs],
        "attempts": [],
    }
    _write_json(root / "operation.json", value)
    _progress(
        f"{kind}: prepared {len(jobs)} job(s) in {root}"
    )
    return ManualOperation(root, operation_id, kind, target, len(jobs))


def prepare_training(
    train_file: str | Path,
    *,
    backend: str,
    config_file: str | Path,
    output: str | Path,
    workdir: str | Path | None,
    target: ExecutionTarget,
    test_file: str | Path | None = None,
    restart_file: str | Path | None = None,
    device: str = "cuda",
    torch_backend: str = "auto",
    precision: str = "float32",
    use_compile: bool = False,
    seed: int = 20260723,
    force: bool = False,
) -> ManualOperation:
    _progress(f"train: reading inputs from {Path(train_file).expanduser()}")
    payload = {
        "backend": backend,
        "train": str(Path(train_file).expanduser().resolve()),
        "config": str(Path(config_file).expanduser().resolve()),
        "target": target.name,
    }
    output_path = _output_path(output, force=force)
    root, operation_id = _prepare_root("train", workdir, payload)
    job = root / "jobs" / "000000"
    job.mkdir()
    request = {
        "backend": backend,
        "train_file": str(Path(_copy(Path(train_file), job / "train.xyz")).relative_to(root)),
        "config_file": str(Path(_copy(Path(config_file), job / "nep.in")).relative_to(root)),
        "test_file": None,
        "restart_file": None,
        "device": device,
        "torch_backend": torch_backend,
        "precision": precision,
        "use_compile": bool(use_compile),
        "seed": int(seed),
    }
    if test_file:
        request["test_file"] = str(
            Path(_copy(Path(test_file), job / "test.xyz")).relative_to(root)
        )
    if restart_file:
        request["restart_file"] = str(
            Path(_copy(Path(restart_file), job / "restart")).relative_to(root)
        )
    _write_json(job / "request.json", request)
    return _write_operation(
        root,
        operation_id=operation_id,
        kind="train",
        target=target,
        output=output_path,
        jobs=[{"index": 0, "request": "jobs/000000/request.json"}],
        max_concurrent=1,
    )


def prepare_dft(
    source: str | Path,
    *,
    backend: str,
    output: str | Path,
    workdir: str | Path | None,
    target: ExecutionTarget,
    input_file: str | Path | None = None,
    resource_dir: str | Path | None = None,
    n_cpu: int | None = None,
    use_gamma: bool = False,
    kpoint_mode: str = "auto",
    kspacing: float | None = None,
    ka: Sequence[int] = (1, 1, 1),
    structures_per_job: int = 1,
    max_concurrent: int = 20,
    teacher_profile: str = "ordinary",
    force: bool = False,
) -> ManualOperation:
    if structures_per_job < 1:
        raise ManualTaskError("structures_per_job must be at least 1")
    effective_resource = (
        target.dft_resource_path
        or (
            str(Path(resource_dir).expanduser().resolve())
            if resource_dir
            else None
        )
    )
    if backend in {"vasp", "abacus"} and not effective_resource:
        raise ManualTaskError(
            f"{backend} labeling requires --resources, dft.resource_path, "
            "or execution.targets.<name>.dft_resource_path"
        )
    if (
        backend in {"vasp", "abacus"}
        and target.host is None
        and not Path(str(effective_resource)).expanduser().is_dir()
    ):
        raise ManualTaskError(
            f"{backend} resource directory does not exist: {effective_resource}"
        )
    effective_n_cpu = int(
        n_cpu
        if n_cpu is not None
        else target.cpus_per_task or 1
    )
    if target.host and resource_dir and not target.dft_resource_path:
        raise ManualTaskError(
            "remote DFT targets require dft_resource_path; "
            "large resource directories are not copied"
        )
    _progress(f"dft: reading structures from {Path(source).expanduser()}")
    output_path = _output_path(output, force=force)
    frames = _frames(Path(source))
    _progress(
        f"dft: read {len(frames)} structure(s); "
        f"{structures_per_job} structure(s) per Slurm task"
    )
    payload = {
        "backend": backend,
        "source": str(Path(source).expanduser().resolve()),
        "frames": len(frames),
        "target": target.name,
    }
    root, operation_id = _prepare_root("dft", workdir, payload)
    common = root / "inputs"
    copied_input = (
        str(Path(_copy(Path(input_file), common / Path(input_file).name)).relative_to(root))
        if input_file
        else None
    )
    jobs = []
    for index, start in enumerate(range(0, len(frames), structures_per_job)):
        job = root / "jobs" / f"{index:06d}"
        job.mkdir()
        ase_write(
            job / "input.xyz",
            frames[start : start + structures_per_job],
            format="extxyz",
        )
        request = {
            "backend": backend,
            "source": str((job / "input.xyz").relative_to(root)),
            "input_file": copied_input,
            "resource_dir": effective_resource,
            "n_cpu": effective_n_cpu,
            "use_gamma": bool(use_gamma),
            "kpoint_mode": kpoint_mode,
            "kspacing": kspacing,
            "ka": [int(value) for value in ka],
            "teacher_profile": teacher_profile,
        }
        _write_json(job / "request.json", request)
        jobs.append(
            {
                "index": index,
                "first_frame": start,
                "frame_count": min(structures_per_job, len(frames) - start),
                "request": str((job / "request.json").relative_to(root)),
            }
        )
    return _write_operation(
        root,
        operation_id=operation_id,
        kind="dft",
        target=target,
        output=output_path,
        jobs=jobs,
        max_concurrent=max_concurrent,
    )


def prepare_md(
    source: str | Path,
    *,
    backend: str,
    model_file: str | Path,
    temperatures: Sequence[float],
    output: str | Path,
    workdir: str | Path | None,
    target: ExecutionTarget,
    steps: int,
    pressure: float = 0.0,
    ensemble: str = "nvt",
    template_path: str | Path | None = None,
    spin: bool = False,
    spin_temperature: float | None = None,
    inference_backend: str = "auto",
    lmp: str = "lmp",
    mpiexec: str = "mpirun",
    mpi_ranks: int = 1,
    pre_failure_frames: int = 2,
    bad_tail_frames: int = 1,
    health: Mapping[str, Any] | None = None,
    max_concurrent: int = 20,
    force: bool = False,
) -> ManualOperation:
    _progress(f"md: reading structures from {Path(source).expanduser()}")
    output_path = _output_path(output, force=force)
    frames = _frames(Path(source))
    _progress(
        f"md: read {len(frames)} structure(s) across "
        f"{len(temperatures)} temperature(s)"
    )
    if not temperatures:
        raise ManualTaskError("at least one temperature is required")
    payload = {
        "backend": backend,
        "source": str(Path(source).expanduser().resolve()),
        "frames": len(frames),
        "temperatures": [float(value) for value in temperatures],
        "target": target.name,
    }
    root, operation_id = _prepare_root("md", workdir, payload)
    common = root / "inputs"
    model = str(
        Path(_copy(Path(model_file), common / "nep.txt")).relative_to(root)
    )
    template = (
        str(
            Path(
                _copy(Path(template_path), common / Path(template_path).name)
            ).relative_to(root)
        )
        if template_path
        else None
    )
    jobs = []
    index = 0
    for frame_index, frame in enumerate(frames):
        for temperature in temperatures:
            job = root / "jobs" / f"{index:06d}"
            job.mkdir()
            ase_write(job / "input.xyz", frame, format="extxyz")
            request = {
                "backend": backend,
                "source": str((job / "input.xyz").relative_to(root)),
                "model_file": model,
                "temperature": float(temperature),
                "steps": int(steps),
                "pressure": float(pressure),
                "ensemble": ensemble,
                "template_path": template,
                "spin": bool(spin),
                "spin_temperature": spin_temperature,
                "inference_backend": inference_backend,
                "lmp": lmp,
                "mpiexec": mpiexec,
                "mpi_ranks": int(mpi_ranks),
                "pre_failure_frames": int(pre_failure_frames),
                "bad_tail_frames": int(bad_tail_frames),
                "health": dict(health or {}),
            }
            _write_json(job / "request.json", request)
            jobs.append(
                {
                    "index": index,
                    "frame": frame_index,
                    "temperature": float(temperature),
                    "request": str((job / "request.json").relative_to(root)),
                }
            )
            index += 1
    return _write_operation(
        root,
        operation_id=operation_id,
        kind="md",
        target=target,
        output=output_path,
        jobs=jobs,
        max_concurrent=max_concurrent,
    )


def load_operation(path: str | Path) -> ManualOperation:
    root = Path(path).expanduser().resolve()
    descriptor = _read_json(root / "operation.json")
    protocol = descriptor.get("protocol")
    if protocol not in {
        "neptrain.manual-operation.v1",
        "neptrain.manual-operation.v2",
    }:
        raise ManualTaskError(f"not a NepTrain manual run: {root}")
    jobs_directory = (
        "jobs"
        if protocol == "neptrain.manual-operation.v2"
        else "shards"
    )
    jobs = descriptor.get("jobs", descriptor.get("shards", []))
    return ManualOperation(
        root=root,
        operation_id=str(descriptor["operation_id"]),
        kind=str(descriptor["kind"]),
        target=ExecutionTarget.from_mapping(
            str(descriptor["target"]["name"]), descriptor["target"]
        ),
        shard_count=len(jobs),
        jobs_directory=jobs_directory,
    )


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def run_manual_worker(root_path: str | Path, index: int) -> int:
    operation = load_operation(root_path)
    descriptor = _read_json(operation.descriptor)
    jobs = descriptor.get("jobs", descriptor.get("shards", []))
    if index < 0 or index >= len(jobs):
        raise ManualTaskError(f"job index out of range: {index}")
    job_meta = jobs[index]
    job = operation.jobs_root / f"{index:06d}"
    request = _read_json(operation.root / job_meta["request"])
    execution = job / "execution.json"
    lock_path = job / ".worker.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        try:
            _write_json(
                execution,
                {"state": "RUNNING", "pid": os.getpid(), "started_at": _now()},
            )
            result_file = _expected_job_result(operation, job)
            result_file.unlink(missing_ok=True)
            calculation_dir = job / "calculation"
            if calculation_dir.exists():
                shutil.rmtree(calculation_dir)
            calculation_dir.mkdir()
            if operation.kind == "train":
                from .training import TrainingRequest, train

                result = train(
                    TrainingRequest(
                        config_file=_resolve(operation.root, request["config_file"]),
                        train_file=_resolve(operation.root, request["train_file"]),
                        output_dir=calculation_dir,
                        test_file=_resolve(operation.root, request.get("test_file")),
                        restart_file=_resolve(
                            operation.root, request.get("restart_file")
                        ),
                        device=request["device"],
                        torch_backend=request["torch_backend"],
                        precision=request["precision"],
                        use_compile=bool(request["use_compile"]),
                        seed=int(request["seed"]),
                    ),
                    request["backend"],
                )
                shutil.copy2(result.best_model, result_file)
                metrics = {"backend": result.backend}
            elif operation.kind == "dft":
                from .dft import LabelRequest, label

                result = label(
                    LabelRequest(
                        source=_resolve(operation.root, request["source"]),
                        output_file=result_file,
                        work_dir=calculation_dir,
                        input_file=_resolve(
                            operation.root, request.get("input_file")
                        ),
                        resource_dir=_resolve(
                            operation.root, request.get("resource_dir")
                        ),
                        n_cpu=int(request["n_cpu"]),
                        use_gamma=bool(request["use_gamma"]),
                        kpoint_mode=request["kpoint_mode"],
                        kspacing=request.get("kspacing"),
                        ka=tuple(request["ka"]),
                        options={"profile": request["teacher_profile"]},
                    ),
                    request["backend"],
                )
                metrics = {"backend": result.backend, "frames": len(result.frames)}
            elif operation.kind == "md":
                from .md import MdRequest, run_md

                atoms = ase_read(
                    _resolve(operation.root, request["source"]), index=0
                )
                result = run_md(
                    MdRequest(
                        atoms=atoms,
                        model_file=_resolve(
                            operation.root, request["model_file"]
                        ),
                        output_dir=calculation_dir,
                        output_file=result_file,
                        temperature=float(request["temperature"]),
                        steps=int(request["steps"]),
                        pressure=float(request["pressure"]),
                        ensemble=request["ensemble"],
                        template_path=_resolve(
                            operation.root, request.get("template_path")
                        ),
                        spin=bool(request["spin"]),
                        spin_temperature=request.get("spin_temperature"),
                        inference_backend=request["inference_backend"],
                        lmp_command=request["lmp"],
                        mpiexec=request["mpiexec"],
                        mpi_ranks=int(request["mpi_ranks"]),
                        pre_failure_frames=int(request["pre_failure_frames"]),
                        bad_tail_frames=int(request["bad_tail_frames"]),
                        health=dict(request.get("health") or {}),
                    ),
                    request["backend"],
                )
                metrics = {
                    "backend": result.backend,
                    "temperature": request["temperature"],
                }
            else:
                raise ManualTaskError(f"unsupported manual task: {operation.kind}")
            _write_json(
                job / "result.json",
                {"completed_at": _now(), "metrics": metrics},
            )
            _write_json(
                execution,
                {"state": "COMPLETED", "pid": os.getpid(), "completed_at": _now()},
            )
            if operation.target.host is None:
                try:
                    _publish_if_ready(operation)
                except Exception as error:
                    descriptor = _read_json(operation.descriptor)
                    descriptor["collection_error"] = str(error)
                    _write_json(operation.descriptor, descriptor)
            return 0
        except Exception as error:
            _write_json(
                execution,
                {
                    "state": "FAILED",
                    "pid": os.getpid(),
                    "failed_at": _now(),
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            return 1


def _setup_line(target: ExecutionTarget, root: str) -> str | None:
    if not target.setup_script:
        return None
    candidate = Path(target.setup_script).expanduser()
    if candidate.is_file() and target.host is None:
        return f"source {shlex.quote(str(candidate.resolve()))}"
    return f"source {shlex.quote(str(target.setup_script))}"


def _remote_root(operation: ManualOperation) -> str:
    assert operation.target.work_root
    return (
        operation.target.work_root.rstrip("/")
        + "/manual/"
        + operation.operation_id
    )


def _deploy_remote(operation: ManualOperation) -> str:
    target = operation.target
    transport = ExecutionTransport(target)
    remote = _remote_root(operation)
    archive = operation.root.parent / f".{operation.operation_id}.tar.gz"
    try:
        if remote.startswith("~/"):
            _progress(f"{operation.kind}: resolving remote home on {target.name}")
            home_result = transport.run_script(
                'printf %s "$HOME"',
                check=True,
            )
            remote_home = home_result.stdout.strip()
            if not remote_home.startswith("/"):
                raise ManualTaskError(
                    f"remote target {target.name} returned an invalid home directory"
                )
            remote = remote_home.rstrip("/") + "/" + remote[2:]
        _progress(f"{operation.kind}: packing portable task inputs")
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(operation.root, arcname=operation.operation_id)
        remote_parent = remote.rsplit("/", 1)[0]
        _progress(f"{operation.kind}: creating remote run directory on {target.name}")
        transport.run_script(
            'mkdir -p -- "$1"',
            remote_parent,
            check=True,
        )
        _progress(
            f"{operation.kind}: uploading {archive.stat().st_size} bytes to {target.name}"
        )
        transport.copy(
            archive,
            f"{target.host}:{remote_parent}/",
            check=True,
        )
        _progress(f"{operation.kind}: extracting remote task bundle")
        transport.run_script(
            """set -eo pipefail
cd "$1"
tar -xzf "$2"
rm -f -- "$2"
""",
            remote_parent,
            archive.name,
            check=True,
        )
    finally:
        archive.unlink(missing_ok=True)
    _progress(f"{operation.kind}: remote task ready at {remote}")
    return remote


def _slurm_script(
    operation: ManualOperation, root: str, indices: Sequence[int]
) -> str:
    target = operation.target
    if not indices:
        raise ManualTaskError("no jobs selected for submission")
    array = ",".join(str(value) for value in indices)
    if len(indices) > 1:
        array += f"%{_read_json(operation.descriptor)['max_concurrent']}"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=nt-{operation.operation_id}",
        f"#SBATCH --output={root}/logs/scheduler-%A_%a.out",
        f"#SBATCH --time={target.time}",
        f"#SBATCH --partition={target.partition}",
        f"#SBATCH --array={array}",
    ]
    if target.qos:
        lines.append(f"#SBATCH --qos={target.qos}")
    if target.cpus_per_task is not None:
        lines.append(f"#SBATCH --cpus-per-task={target.cpus_per_task}")
    if target.gpus_per_node is not None:
        lines.append(f"#SBATCH --gpus-per-node={target.gpus_per_node}")
    lines.extend(
        directive
        if directive.startswith("#SBATCH ")
        else f"#SBATCH {directive}"
        for directive in target.directives
    )
    lines.extend(["", "set -eo pipefail", f"cd {shlex.quote(root)}"])
    setup = _setup_line(target, root)
    if setup:
        lines.append(setup)
    if target.cpus_per_task is not None:
        lines.append('export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"')
    for key, value in target.environment.items():
        lines.append(f"export {key}={shlex.quote(value)}")
    worker = shlex.join(
        [
            *shlex.split(target.command),
            "manual-worker",
            root,
            "${SLURM_ARRAY_TASK_ID}",
        ]
    ).replace("'${SLURM_ARRAY_TASK_ID}'", '"${SLURM_ARRAY_TASK_ID}"')
    lines.extend([worker, ""])
    return "\n".join(lines)


def submit_operation(
    operation: ManualOperation,
    *,
    wait: bool = False,
    poll_interval: float = 10.0,
) -> dict[str, Any]:
    if operation.target.executor == "process":
        if operation.target.host:
            raise ManualTaskError(
                "remote process targets are not supported for manual steps; "
                "use a Slurm target or run the command on that platform"
            )
        _progress(
            f"{operation.kind}: running {operation.shard_count} local job(s)"
        )
        for index in range(operation.shard_count):
            _progress(
                f"{operation.kind}: local job {index + 1}/{operation.shard_count}"
            )
            if run_manual_worker(operation.root, index) != 0:
                break
        status = refresh_operation(operation)
        if status["state"] != "complete":
            raise ManualTaskError(status["reason"])
        return status
    if os.environ.get("SLURM_JOB_ID") and operation.target.host is None:
        raise ManualTaskError("submit manual Slurm tasks from a login node")
    remote = (
        _deploy_remote(operation) if operation.target.host else str(operation.root)
    )
    indices = list(range(operation.shard_count))
    script = operation.root / "job.sbatch"
    script.write_text(_slurm_script(operation, remote, indices), encoding="utf-8")
    if operation.target.host:
        transport = ExecutionTransport(operation.target)
        _progress(f"{operation.kind}: uploading Slurm submission script")
        transport.copy(
            script,
            f"{operation.target.host}:{remote}/job.sbatch",
            check=True,
        )
        _progress(
            f"{operation.kind}: submitting {operation.shard_count} Slurm task(s) "
            f"with max concurrency "
            f"{_read_json(operation.descriptor)['max_concurrent']}"
        )
        completed = transport.run_script(
            """set -eo pipefail
cd "$1"
sbatch --parsable "$2"
""",
            remote,
            "job.sbatch",
        )
    else:
        _progress(
            f"{operation.kind}: submitting {operation.shard_count} Slurm task(s)"
        )
        completed = subprocess.run(
            ["sbatch", "--parsable", "job.sbatch"],
            cwd=operation.root,
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise ManualTaskError((completed.stderr or completed.stdout).strip())
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ManualTaskError(f"Slurm returned an invalid job id: {job_id}")
    _progress(f"{operation.kind}: submitted Slurm job {job_id}")
    descriptor = _read_json(operation.descriptor)
    descriptor["state"] = "submitted"
    descriptor["job_id"] = job_id
    descriptor["remote_root"] = remote if operation.target.host else None
    if operation.target.host:
        (operation.root / "remote.txt").write_text(
            f"host: {operation.target.host}\npath: {remote}\n",
            encoding="utf-8",
        )
    descriptor["attempts"].append(
        {"submitted_at": _now(), "job_id": job_id, "indices": indices}
    )
    _write_json(operation.descriptor, descriptor)
    if wait:
        return wait_operation(operation, poll_interval=poll_interval)
    _progress(f"{operation.kind}: checking initial scheduler state")
    return refresh_operation(operation)


def _result_filename(kind: str) -> str:
    return {
        "train": "nep.txt",
        "dft": "labeled.xyz",
        "md": "trajectory.xyz",
    }[kind]


def _expected_job_result(operation: ManualOperation, job: Path) -> Path:
    if operation.jobs_directory == "shards":
        return job / "result" / _result_filename(operation.kind)
    return job / _result_filename(operation.kind)


def _completed_job_is_collectable(
    operation: ManualOperation, job: Path
) -> bool:
    execution = job / "execution.json"
    return (
        execution.is_file()
        and _read_json(execution).get("state") == "COMPLETED"
        and (job / "result.json").is_file()
        and _expected_job_result(operation, job).is_file()
    )


def _sync_remote_results(operation: ManualOperation) -> None:
    descriptor = _read_json(operation.descriptor)
    remote = descriptor.get("remote_root")
    if not remote:
        return
    lock_path = operation.root / ".remote-sync.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        members = []
        incomplete = []
        for index in range(operation.shard_count):
            relative = f"{index:06d}"
            job = operation.jobs_root / relative
            execution = job / "execution.json"
            state = (
                _read_json(execution).get("state")
                if execution.is_file()
                else None
            )
            synchronized = state == "FAILED" or (
                state == "COMPLETED"
                and _completed_job_is_collectable(operation, job)
            )
            if synchronized:
                continue
            incomplete.append(index)
            members.extend(
                (
                    f"{relative}/execution.json",
                    f"{relative}/result.json",
                    (
                        f"{relative}/result"
                        if operation.jobs_directory == "shards"
                        else f"{relative}/{_result_filename(operation.kind)}"
                    ),
                )
            )
        if not incomplete:
            return
        _progress(
            f"{operation.kind}: synchronizing {len(incomplete)} incomplete "
            "remote job(s) in one archive"
        )
        transport = ExecutionTransport(operation.target)
        archived = transport.fetch_paths(
            f"{remote.rstrip('/')}/{operation.jobs_directory}",
            members,
            operation.jobs_root,
        )
        _progress(
            f"{operation.kind}: received {len(archived)} remote result "
            "path(s)"
        )


def _all_jobs_completed(operation: ManualOperation) -> bool:
    for index in range(operation.shard_count):
        job = operation.jobs_root / f"{index:06d}"
        if not _completed_job_is_collectable(operation, job):
            return False
    return True


def _ensure_visible_result(
    operation: ManualOperation, output: Path
) -> None:
    visible_result = operation.root / _result_filename(operation.kind)
    if visible_result.resolve() == output.resolve():
        return
    temporary_link = visible_result.with_name(f".{visible_result.name}.link")
    temporary_link.unlink(missing_ok=True)
    temporary_link.symlink_to(os.path.relpath(output, operation.root))
    temporary_link.replace(visible_result)


def _ensure_remote_pointer(
    operation: ManualOperation, remote: str | None
) -> None:
    if not operation.target.host or not remote:
        return
    pointer = operation.root / "remote.txt"
    value = f"host: {operation.target.host}\npath: {remote}\n"
    if not pointer.is_file() or pointer.read_text(encoding="utf-8") != value:
        pointer.write_text(value, encoding="utf-8")


def _collect_unlocked(operation: ManualOperation) -> Path:
    descriptor = _read_json(operation.descriptor)
    _ensure_remote_pointer(operation, descriptor.get("remote_root"))
    output = Path(descriptor["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if operation.kind == "train":
        shutil.copy2(
            _expected_job_result(
                operation,
                operation.jobs_root / "000000",
            ),
            temporary,
        )
    else:
        frames = []
        for index in range(operation.shard_count):
            path = _expected_job_result(
                operation,
                operation.jobs_root / f"{index:06d}",
            )
            loaded = ase_read(path, index=":")
            frames.extend(loaded if isinstance(loaded, list) else [loaded])
        ase_write(temporary, frames, format="extxyz")
    temporary.replace(output)
    _ensure_visible_result(operation, output)
    descriptor["state"] = "complete"
    descriptor["completed_at"] = _now()
    descriptor["result"] = str(output)
    _write_json(operation.descriptor, descriptor)
    return output


def _collect(operation: ManualOperation) -> Path:
    lock_path = operation.root / ".collection.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        descriptor = _read_json(operation.descriptor)
        if descriptor.get("state") == "complete":
            return Path(descriptor["result"])
        if not _all_jobs_completed(operation):
            raise ManualTaskError("cannot publish an incomplete manual operation")
        return _collect_unlocked(operation)


def _publish_if_ready(operation: ManualOperation) -> None:
    if _all_jobs_completed(operation):
        _collect(operation)


def _normalise_slurm_state(value: str) -> str:
    state = value.strip().split("|", 1)[0]
    return re.split(r"[+\s]", state, maxsplit=1)[0].upper()


def _aggregate_slurm_states(states: Sequence[str]) -> str | None:
    values = [_normalise_slurm_state(value) for value in states if value.strip()]
    if not values:
        return None
    if any(value in {"RUNNING", "COMPLETING"} for value in values):
        return "RUNNING"
    if any(value in _SLURM_ACTIVE for value in values):
        return "PENDING"
    if any(value in _SLURM_FAILURE for value in values):
        if all(value == "CANCELLED" for value in values):
            return "CANCELLED"
        return next(value for value in values if value in _SLURM_FAILURE)
    if all(value == "COMPLETED" for value in values):
        return "COMPLETED"
    return values[0]


def _run_on_target(
    operation: ManualOperation, command: Sequence[str]
) -> subprocess.CompletedProcess:
    if operation.target.host:
        return ExecutionTransport(operation.target).run_script(
            'exec "$@"',
            *command,
        )
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )


def _slurm_job_state(
    operation: ManualOperation, job_id: str | None
) -> tuple[str | None, str | None]:
    if not job_id or not str(job_id).isdigit():
        return None, None
    queue = _run_on_target(
        operation, ["squeue", "-h", "-j", str(job_id), "-o", "%T"]
    )
    if queue.returncode == 0:
        state = _aggregate_slurm_states(queue.stdout.splitlines())
        if state:
            return state, None
    account = _run_on_target(
        operation,
        [
            "sacct",
            "-n",
            "-X",
            "-j",
            str(job_id),
            "--format=State",
            "--parsable2",
        ],
    )
    if account.returncode == 0:
        state = _aggregate_slurm_states(account.stdout.splitlines())
        if state:
            return state, None
    detail = (
        account.stderr.strip()
        or queue.stderr.strip()
        or "job is absent from squeue and sacct"
    )
    return None, detail


def refresh_operation(operation: ManualOperation) -> dict[str, Any]:
    _sync_remote_results(operation)
    descriptor = _read_json(operation.descriptor)
    _ensure_remote_pointer(operation, descriptor.get("remote_root"))
    if descriptor.get("state") == "complete" and descriptor.get("result"):
        output = Path(descriptor["result"])
        if output.is_file():
            _ensure_visible_result(operation, output)
    scheduler_state = None
    scheduler_error = None
    if (
        operation.target.executor == "slurm"
        and descriptor.get("state") not in {"complete", "cancelled"}
    ):
        scheduler_state, scheduler_error = _slurm_job_state(
            operation, descriptor.get("job_id")
        )
    states = []
    errors = []
    for index in range(operation.shard_count):
        path = operation.jobs_root / f"{index:06d}" / "execution.json"
        if not path.is_file():
            states.append("PENDING")
            continue
        value = _read_json(path)
        state = str(value.get("state", "UNKNOWN"))
        states.append(state)
        if state == "FAILED":
            errors.append({"index": index, "error": value.get("error", "")})
    terminal_scheduler_failure = scheduler_state in _SLURM_FAILURE
    if terminal_scheduler_failure:
        failed_indices = {item["index"] for item in errors}
        for index, shard_state in enumerate(states):
            if shard_state != "COMPLETED" and index not in failed_indices:
                errors.append(
                    {
                        "index": index,
                        "error": f"Slurm job ended in {scheduler_state}",
                    }
                )
    if descriptor.get("state") == "cancelled" or scheduler_state == "CANCELLED":
        state = "cancelled"
        reason = f"{states.count('COMPLETED')}/{operation.shard_count} jobs completed before cancellation"
    elif errors:
        state = "failed"
        reason = f"{len(errors)} of {operation.shard_count} jobs failed"
    elif states and all(value == "COMPLETED" for value in states):
        if descriptor.get("state") != "complete":
            try:
                _collect(operation)
                descriptor = _read_json(operation.descriptor)
            except Exception as error:
                descriptor["collection_error"] = str(error)
                descriptor["state"] = "failed"
                _write_json(operation.descriptor, descriptor)
                errors.append({"index": None, "error": str(error)})
        if errors:
            state = "failed"
            reason = "all jobs completed but the final result could not be published"
        else:
            state = "complete"
            reason = "all jobs completed and the final result was published"
    elif any(value == "RUNNING" for value in states) or scheduler_state == "RUNNING":
        state = "running"
        reason = f"{states.count('COMPLETED')}/{operation.shard_count} jobs completed"
    elif scheduler_state == "COMPLETED":
        missing = [
            index for index, shard_state in enumerate(states)
            if shard_state != "COMPLETED"
        ]
        errors.extend(
            {
                "index": index,
                "error": "Slurm completed without a job result",
            }
            for index in missing
        )
        state = "failed"
        reason = f"{len(missing)} jobs produced no result"
    else:
        state = descriptor.get("state", "prepared")
        reason = f"{states.count('COMPLETED')}/{operation.shard_count} jobs completed"
    return {
        "operation_id": operation.operation_id,
        "kind": operation.kind,
        "state": state,
        "reason": reason,
        "job_id": descriptor.get("job_id"),
        "scheduler_state": scheduler_state,
        "scheduler_error": scheduler_error,
        "completed": states.count("COMPLETED"),
        "failed": len(errors),
        "total": operation.shard_count,
        "result": descriptor.get("result"),
        "remote_directory": descriptor.get("remote_root"),
        "run_directory": str(operation.root),
        "errors": errors,
    }


def wait_operation(
    operation: ManualOperation, *, poll_interval: float = 10.0
) -> dict[str, Any]:
    previous: tuple[Any, ...] | None = None
    while True:
        status = refresh_operation(operation)
        snapshot = (
            status["state"],
            status["scheduler_state"],
            status["completed"],
            status["failed"],
        )
        if snapshot != previous:
            scheduler = status["scheduler_state"] or "UNKNOWN"
            _progress(
                f"{operation.kind}: state={status['state']}, "
                f"scheduler={scheduler}, "
                f"completed={status['completed']}/{status['total']}, "
                f"failed={status['failed']}"
            )
            previous = snapshot
        if status["state"] in {"complete", "failed", "cancelled"}:
            return status
        time.sleep(max(0.2, poll_interval))


def cancel_operation(operation: ManualOperation) -> dict[str, Any]:
    status = refresh_operation(operation)
    if status["state"] == "complete":
        raise ManualTaskError("operation is already complete")
    if status["state"] == "cancelled":
        return status
    if (
        status["state"] == "failed"
        and status.get("scheduler_state") not in _SLURM_ACTIVE
    ):
        raise ManualTaskError("operation has already failed; use task retry")
    descriptor = _read_json(operation.descriptor)
    job_id = descriptor.get("job_id")
    if not job_id:
        raise ManualTaskError("operation has no submitted Slurm job")
    _progress(f"{operation.kind}: cancelling Slurm job {job_id}")
    completed = _run_on_target(operation, ["scancel", str(job_id)])
    if completed.returncode != 0:
        raise ManualTaskError((completed.stderr or completed.stdout).strip())
    descriptor["state"] = "cancelled"
    descriptor["cancelled_at"] = _now()
    _write_json(operation.descriptor, descriptor)
    return refresh_operation(operation)


def retry_failed(operation: ManualOperation) -> dict[str, Any]:
    if operation.target.executor != "slurm":
        raise ManualTaskError("retry is only needed for submitted Slurm operations")
    status = refresh_operation(operation)
    failed = [
        int(item["index"])
        for item in status["errors"]
        if isinstance(item.get("index"), int)
    ]
    if not failed:
        raise ManualTaskError(
            "operation has no failed jobs; fix the reported collection error "
            "and run task status again"
        )
    descriptor = _read_json(operation.descriptor)
    remote = descriptor.get("remote_root") or str(operation.root)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_entry = (
        "result"
        if operation.jobs_directory == "shards"
        else _result_filename(operation.kind)
    )
    retry_entries = ["execution.json", "result.json", result_entry]
    if operation.jobs_directory == "jobs":
        retry_entries.append("calculation")
    if operation.target.host:
        transport = ExecutionTransport(operation.target)
        _progress(
            f"{operation.kind}: archiving {len(failed)} failed remote job(s)"
        )
        commands = ["set -eo pipefail"]
        for index in failed:
            job = f"{remote}/{operation.jobs_directory}/{index:06d}"
            attempt = f"{job}/attempts/{stamp}"
            commands.append(f"mkdir -p {shlex.quote(attempt)}")
            for name in retry_entries:
                source = f"{job}/{name}"
                commands.append(
                    f"if test -e {shlex.quote(source)}; then "
                    f"mv {shlex.quote(source)} {shlex.quote(attempt + '/' + name)}; fi"
                )
        archived = transport.run_script(
            "\n".join(commands),
        )
        if archived.returncode != 0:
            raise ManualTaskError(
                (archived.stderr or archived.stdout or "remote retry archival failed").strip()
            )
    for index in failed:
        job = operation.jobs_root / f"{index:06d}"
        attempt = job / "attempts" / stamp
        attempt.mkdir(parents=True)
        for name in retry_entries:
            path = job / name
            if path.exists():
                shutil.move(str(path), attempt / name)
    script = operation.root / "retry.sbatch"
    script.write_text(_slurm_script(operation, remote, failed), encoding="utf-8")
    if operation.target.host:
        _progress(f"{operation.kind}: uploading retry submission script")
        transport.copy(
            script,
            f"{operation.target.host}:{remote}/retry.sbatch",
            check=True,
        )
        _progress(f"{operation.kind}: submitting failed jobs for retry")
        completed = transport.run_script(
            """set -eo pipefail
cd "$1"
sbatch --parsable "$2"
""",
            remote,
            "retry.sbatch",
        )
    else:
        completed = subprocess.run(
            ["sbatch", "--parsable", "retry.sbatch"],
            cwd=operation.root,
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise ManualTaskError((completed.stderr or completed.stdout).strip())
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ManualTaskError(f"Slurm returned an invalid job id: {job_id}")
    _progress(f"{operation.kind}: submitted retry job {job_id}")
    descriptor["state"] = "submitted"
    descriptor["job_id"] = job_id
    descriptor["attempts"].append(
        {"submitted_at": _now(), "job_id": job_id, "indices": failed}
    )
    _write_json(operation.descriptor, descriptor)
    return refresh_operation(operation)


def operation_logs(operation: ManualOperation) -> list[str]:
    descriptor = _read_json(operation.descriptor)
    logs_root = operation.root / "logs"
    logs_root.mkdir(exist_ok=True)
    for legacy in operation.root.glob("scheduler-*.out"):
        legacy.replace(logs_root / legacy.name)
    local_logs = sorted(logs_root.glob("scheduler-*.out"))
    if descriptor.get("state") == "complete":
        expected_logs = {
            f"scheduler-{attempt['job_id']}_{index}.out"
            for attempt in descriptor.get("attempts", [])
            if str(attempt.get("job_id", "")).isdigit()
            for index in attempt.get("indices", [])
        }
        if expected_logs and expected_logs.issubset(
            {path.name for path in local_logs}
        ):
            return [str(path) for path in local_logs]
    remote = descriptor.get("remote_root")
    if operation.target.host and remote:
        transport = ExecutionTransport(operation.target)
        _progress(f"{operation.kind}: listing remote scheduler logs")
        remote_logs = (
            f"{remote.rstrip('/')}/logs"
            if operation.jobs_directory == "jobs"
            else remote
        )
        listed = transport.run_script(
            """find "$1" -maxdepth 1 -type f \
-name 'scheduler-*.out' -print
""",
            remote_logs,
        )
        if listed.returncode != 0:
            raise ManualTaskError(
                (listed.stderr or listed.stdout or "cannot list remote logs").strip()
            )
        names = []
        for value in listed.stdout.splitlines():
            name = Path(value).name
            if not re.fullmatch(r"scheduler-[A-Za-z0-9_.-]+\.out", name):
                continue
            names.append(name)
        if names:
            _progress(
                f"{operation.kind}: synchronizing {len(names)} scheduler "
                "log(s) in one archive"
            )
            transport.fetch_paths(remote_logs, names, logs_root)
    return [
        str(path)
        for path in sorted(logs_root.glob("scheduler-*.out"))
    ]


__all__ = [
    "ManualOperation",
    "ManualTaskError",
    "cancel_operation",
    "load_operation",
    "operation_logs",
    "prepare_dft",
    "prepare_md",
    "prepare_training",
    "refresh_operation",
    "retry_failed",
    "run_manual_worker",
    "submit_operation",
    "target_from_project",
    "wait_operation",
]
