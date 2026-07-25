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
import tarfile
import time
import traceback
from typing import Any, Mapping, Sequence

from ase.io import read as ase_read
from ase.io import write as ase_write

from .execution import ExecutionTarget


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

    @property
    def descriptor(self) -> Path:
        return self.root / "operation.json"


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
    (root / "shards").mkdir(parents=True)
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
    shards: Sequence[Mapping[str, Any]],
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
        "protocol": "neptrain.manual-operation.v1",
        "operation_id": operation_id,
        "kind": kind,
        "created_at": _now(),
        "state": "prepared",
        "target": asdict(target),
        "output": str(output.expanduser().resolve()),
        "max_concurrent": int(max_concurrent),
        "shards": [dict(item) for item in shards],
        "attempts": [],
    }
    _write_json(root / "operation.json", value)
    return ManualOperation(root, operation_id, kind, target, len(shards))


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
    payload = {
        "backend": backend,
        "train": str(Path(train_file).expanduser().resolve()),
        "config": str(Path(config_file).expanduser().resolve()),
        "target": target.name,
    }
    output_path = _output_path(output, force=force)
    root, operation_id = _prepare_root("train", workdir, payload)
    shard = root / "shards" / "000000"
    shard.mkdir()
    request = {
        "backend": backend,
        "train_file": str(Path(_copy(Path(train_file), shard / "train.xyz")).relative_to(root)),
        "config_file": str(Path(_copy(Path(config_file), shard / "nep.in")).relative_to(root)),
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
            Path(_copy(Path(test_file), shard / "test.xyz")).relative_to(root)
        )
    if restart_file:
        request["restart_file"] = str(
            Path(_copy(Path(restart_file), shard / "restart")).relative_to(root)
        )
    _write_json(shard / "request.json", request)
    return _write_operation(
        root,
        operation_id=operation_id,
        kind="train",
        target=target,
        output=output_path,
        shards=[{"index": 0, "request": "shards/000000/request.json"}],
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
    output_path = _output_path(output, force=force)
    frames = _frames(Path(source))
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
    shards = []
    for index, start in enumerate(range(0, len(frames), structures_per_job)):
        shard = root / "shards" / f"{index:06d}"
        shard.mkdir()
        ase_write(
            shard / "input.xyz",
            frames[start : start + structures_per_job],
            format="extxyz",
        )
        request = {
            "backend": backend,
            "source": str((shard / "input.xyz").relative_to(root)),
            "input_file": copied_input,
            "resource_dir": effective_resource,
            "n_cpu": effective_n_cpu,
            "use_gamma": bool(use_gamma),
            "kpoint_mode": kpoint_mode,
            "kspacing": kspacing,
            "ka": [int(value) for value in ka],
            "teacher_profile": teacher_profile,
        }
        _write_json(shard / "request.json", request)
        shards.append(
            {
                "index": index,
                "first_frame": start,
                "frame_count": min(structures_per_job, len(frames) - start),
                "request": str((shard / "request.json").relative_to(root)),
            }
        )
    return _write_operation(
        root,
        operation_id=operation_id,
        kind="dft",
        target=target,
        output=output_path,
        shards=shards,
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
    output_path = _output_path(output, force=force)
    frames = _frames(Path(source))
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
    shards = []
    index = 0
    for frame_index, frame in enumerate(frames):
        for temperature in temperatures:
            shard = root / "shards" / f"{index:06d}"
            shard.mkdir()
            ase_write(shard / "input.xyz", frame, format="extxyz")
            request = {
                "backend": backend,
                "source": str((shard / "input.xyz").relative_to(root)),
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
            _write_json(shard / "request.json", request)
            shards.append(
                {
                    "index": index,
                    "frame": frame_index,
                    "temperature": float(temperature),
                    "request": str((shard / "request.json").relative_to(root)),
                }
            )
            index += 1
    return _write_operation(
        root,
        operation_id=operation_id,
        kind="md",
        target=target,
        output=output_path,
        shards=shards,
        max_concurrent=max_concurrent,
    )


def load_operation(path: str | Path) -> ManualOperation:
    root = Path(path).expanduser().resolve()
    descriptor = _read_json(root / "operation.json")
    if descriptor.get("protocol") != "neptrain.manual-operation.v1":
        raise ManualTaskError(f"not a NepTrain manual run: {root}")
    return ManualOperation(
        root=root,
        operation_id=str(descriptor["operation_id"]),
        kind=str(descriptor["kind"]),
        target=ExecutionTarget.from_mapping(
            str(descriptor["target"]["name"]), descriptor["target"]
        ),
        shard_count=len(descriptor["shards"]),
    )


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def run_manual_worker(root_path: str | Path, index: int) -> int:
    operation = load_operation(root_path)
    descriptor = _read_json(operation.descriptor)
    if index < 0 or index >= len(descriptor["shards"]):
        raise ManualTaskError(f"shard index out of range: {index}")
    shard_meta = descriptor["shards"][index]
    shard = operation.root / "shards" / f"{index:06d}"
    request = _read_json(operation.root / shard_meta["request"])
    execution = shard / "execution.json"
    lock_path = shard / "worker.lock"
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
            result_dir = shard / "result"
            if result_dir.exists():
                shutil.rmtree(result_dir)
            result_dir.mkdir()
            if operation.kind == "train":
                from .training import TrainingRequest, train

                result = train(
                    TrainingRequest(
                        config_file=_resolve(operation.root, request["config_file"]),
                        train_file=_resolve(operation.root, request["train_file"]),
                        output_dir=shard / "work" / "training",
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
                shutil.copy2(result.best_model, result_dir / "nep.txt")
                metrics = {"backend": result.backend}
            elif operation.kind == "dft":
                from .dft import LabelRequest, label

                result = label(
                    LabelRequest(
                        source=_resolve(operation.root, request["source"]),
                        output_file=result_dir / "labeled.xyz",
                        work_dir=shard / "work" / "dft",
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
                        output_dir=shard / "work" / "md",
                        output_file=result_dir / "trajectory.xyz",
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
                shard / "result.json",
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
    remote = _remote_root(operation)
    archive = operation.root.parent / f".{operation.operation_id}.tar.gz"
    try:
        if remote.startswith("~/"):
            home_result = subprocess.run(
                ["ssh", str(target.host), "bash", "-lc", 'printf %s "$HOME"'],
                check=True,
                capture_output=True,
                text=True,
            )
            remote_home = home_result.stdout.strip()
            if not remote_home.startswith("/"):
                raise ManualTaskError(
                    f"remote target {target.name} returned an invalid home directory"
                )
            remote = remote_home.rstrip("/") + "/" + remote[2:]
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(operation.root, arcname=operation.operation_id)
        remote_parent = remote.rsplit("/", 1)[0]
        subprocess.run(
            ["ssh", str(target.host), "mkdir", "-p", remote_parent],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["scp", str(archive), f"{target.host}:{remote_parent}/"],
            check=True,
            capture_output=True,
            text=True,
        )
        command = (
            f"cd {shlex.quote(remote_parent)} && "
            f"tar -xzf {shlex.quote(archive.name)} && rm -f {shlex.quote(archive.name)}"
        )
        subprocess.run(
            ["ssh", str(target.host), "bash", "-lc", command],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise ManualTaskError(
            (error.stderr or error.stdout or "remote deployment failed").strip()
        ) from error
    finally:
        archive.unlink(missing_ok=True)
    return remote


def _slurm_script(
    operation: ManualOperation, root: str, indices: Sequence[int]
) -> str:
    target = operation.target
    if not indices:
        raise ManualTaskError("no shards selected for submission")
    array = ",".join(str(value) for value in indices)
    if len(indices) > 1:
        array += f"%{_read_json(operation.descriptor)['max_concurrent']}"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=nt-{operation.operation_id}",
        f"#SBATCH --output={root}/scheduler-%A_%a.out",
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
        for index in range(operation.shard_count):
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
        subprocess.run(
            ["scp", str(script), f"{operation.target.host}:{remote}/job.sbatch"],
            check=True,
            capture_output=True,
            text=True,
        )
        completed = subprocess.run(
            [
                "ssh",
                str(operation.target.host),
                "bash",
                "-lc",
                f"cd {shlex.quote(remote)} && sbatch --parsable job.sbatch",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
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
    descriptor = _read_json(operation.descriptor)
    descriptor["state"] = "submitted"
    descriptor["job_id"] = job_id
    descriptor["remote_root"] = remote if operation.target.host else None
    descriptor["attempts"].append(
        {"submitted_at": _now(), "job_id": job_id, "indices": indices}
    )
    _write_json(operation.descriptor, descriptor)
    if wait:
        return wait_operation(operation, poll_interval=poll_interval)
    return refresh_operation(operation)


def _sync_remote_results(operation: ManualOperation) -> None:
    descriptor = _read_json(operation.descriptor)
    remote = descriptor.get("remote_root")
    if not remote:
        return
    listed = subprocess.run(
        [
            "ssh",
            str(operation.target.host),
            "bash",
            "-lc",
            f"find {shlex.quote(remote + '/shards')} -mindepth 2 -maxdepth 2 "
            "-name execution.json -type f -print",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        return
    prefix = remote.rstrip("/") + "/shards/"
    for remote_execution in listed.stdout.splitlines():
        if not remote_execution.startswith(prefix):
            continue
        relative = remote_execution[len(prefix) :]
        match = re.fullmatch(r"([0-9]{6})/execution\.json", relative)
        if not match:
            continue
        index = int(match.group(1))
        if index >= operation.shard_count:
            continue
        shard = operation.root / "shards" / f"{index:06d}"
        execution = shard / "execution.json"
        if execution.is_file():
            local_state = _read_json(execution).get("state")
            if local_state in {"COMPLETED", "FAILED"}:
                continue
        copied = subprocess.run(
            [
                "scp",
                f"{operation.target.host}:{remote_execution}",
                str(execution),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if copied.returncode != 0 or not execution.is_file():
            continue
        if _read_json(execution).get("state") != "COMPLETED":
            continue
        remote_shard = remote_execution.rsplit("/", 1)[0]
        subprocess.run(
            [
                "scp",
                f"{operation.target.host}:{remote_shard}/result.json",
                str(shard / "result.json"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        result = shard / "result"
        if not result.exists():
            subprocess.run(
                [
                    "scp",
                    "-r",
                    f"{operation.target.host}:{remote_shard}/result",
                    str(result),
                ],
                capture_output=True,
                text=True,
                check=False,
            )


def _all_shards_completed(operation: ManualOperation) -> bool:
    for index in range(operation.shard_count):
        execution = (
            operation.root / "shards" / f"{index:06d}" / "execution.json"
        )
        if not execution.is_file():
            return False
        if _read_json(execution).get("state") != "COMPLETED":
            return False
    return True


def _collect_unlocked(operation: ManualOperation) -> Path:
    descriptor = _read_json(operation.descriptor)
    output = Path(descriptor["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if operation.kind == "train":
        shutil.copy2(
            operation.root / "shards" / "000000" / "result" / "nep.txt",
            temporary,
        )
    else:
        frames = []
        filename = "labeled.xyz" if operation.kind == "dft" else "trajectory.xyz"
        for index in range(operation.shard_count):
            path = (
                operation.root
                / "shards"
                / f"{index:06d}"
                / "result"
                / filename
            )
            loaded = ase_read(path, index=":")
            frames.extend(loaded if isinstance(loaded, list) else [loaded])
        ase_write(temporary, frames, format="extxyz")
    temporary.replace(output)
    descriptor["state"] = "complete"
    descriptor["completed_at"] = _now()
    descriptor["result"] = str(output)
    _write_json(operation.descriptor, descriptor)
    return output


def _collect(operation: ManualOperation) -> Path:
    lock_path = operation.root / "collection.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        descriptor = _read_json(operation.descriptor)
        if descriptor.get("state") == "complete":
            return Path(descriptor["result"])
        if not _all_shards_completed(operation):
            raise ManualTaskError("cannot publish an incomplete manual operation")
        return _collect_unlocked(operation)


def _publish_if_ready(operation: ManualOperation) -> None:
    if _all_shards_completed(operation):
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
    arguments = list(command)
    if operation.target.host:
        arguments = ["ssh", str(operation.target.host), *arguments]
    return subprocess.run(
        arguments,
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
        path = operation.root / "shards" / f"{index:06d}" / "execution.json"
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
        reason = f"{states.count('COMPLETED')}/{operation.shard_count} shards completed before cancellation"
    elif errors:
        state = "failed"
        reason = f"{len(errors)} of {operation.shard_count} shards failed"
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
            reason = "all shards completed but the final result could not be published"
        else:
            state = "complete"
            reason = "all shards completed and the final result was published"
    elif any(value == "RUNNING" for value in states) or scheduler_state == "RUNNING":
        state = "running"
        reason = f"{states.count('COMPLETED')}/{operation.shard_count} shards completed"
    elif scheduler_state == "COMPLETED":
        missing = [
            index for index, shard_state in enumerate(states)
            if shard_state != "COMPLETED"
        ]
        errors.extend(
            {
                "index": index,
                "error": "Slurm completed without a shard result",
            }
            for index in missing
        )
        state = "failed"
        reason = f"{len(missing)} shards produced no result"
    else:
        state = descriptor.get("state", "prepared")
        reason = f"{states.count('COMPLETED')}/{operation.shard_count} shards completed"
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
        "run_directory": str(operation.root),
        "errors": errors,
    }


def wait_operation(
    operation: ManualOperation, *, poll_interval: float = 10.0
) -> dict[str, Any]:
    while True:
        status = refresh_operation(operation)
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
    command = ["scancel", str(job_id)]
    if operation.target.host:
        command = ["ssh", str(operation.target.host), *command]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
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
            "operation has no failed shards; fix the reported collection error "
            "and run task status again"
        )
    descriptor = _read_json(operation.descriptor)
    remote = descriptor.get("remote_root") or str(operation.root)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if operation.target.host:
        commands = ["set -eo pipefail"]
        for index in failed:
            shard = f"{remote}/shards/{index:06d}"
            attempt = f"{shard}/attempts/{stamp}"
            commands.append(f"mkdir -p {shlex.quote(attempt)}")
            for name in ("execution.json", "result.json", "result"):
                source = f"{shard}/{name}"
                commands.append(
                    f"if test -e {shlex.quote(source)}; then "
                    f"mv {shlex.quote(source)} {shlex.quote(attempt + '/' + name)}; fi"
                )
        archived = subprocess.run(
            [
                "ssh",
                str(operation.target.host),
                "bash",
                "-lc",
                "\n".join(commands),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if archived.returncode != 0:
            raise ManualTaskError(
                (archived.stderr or archived.stdout or "remote retry archival failed").strip()
            )
    for index in failed:
        shard = operation.root / "shards" / f"{index:06d}"
        attempt = shard / "attempts" / stamp
        attempt.mkdir(parents=True)
        for name in ("execution.json", "result.json", "result"):
            path = shard / name
            if path.exists():
                shutil.move(str(path), attempt / name)
    script = operation.root / "retry.sbatch"
    script.write_text(_slurm_script(operation, remote, failed), encoding="utf-8")
    if operation.target.host:
        subprocess.run(
            ["scp", str(script), f"{operation.target.host}:{remote}/retry.sbatch"],
            check=True,
            capture_output=True,
            text=True,
        )
        command = [
            "ssh",
            str(operation.target.host),
            "bash",
            "-lc",
            f"cd {shlex.quote(remote)} && sbatch --parsable retry.sbatch",
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False
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
    descriptor["state"] = "submitted"
    descriptor["job_id"] = job_id
    descriptor["attempts"].append(
        {"submitted_at": _now(), "job_id": job_id, "indices": failed}
    )
    _write_json(operation.descriptor, descriptor)
    return refresh_operation(operation)


def operation_logs(operation: ManualOperation) -> list[str]:
    descriptor = _read_json(operation.descriptor)
    remote = descriptor.get("remote_root")
    if operation.target.host and remote:
        listed = subprocess.run(
            [
                "ssh",
                str(operation.target.host),
                "bash",
                "-lc",
                f"find {shlex.quote(remote)} -maxdepth 1 -type f "
                "-name 'scheduler-*.out' -print",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if listed.returncode != 0:
            raise ManualTaskError(
                (listed.stderr or listed.stdout or "cannot list remote logs").strip()
            )
        for value in listed.stdout.splitlines():
            name = Path(value).name
            if not re.fullmatch(r"scheduler-[A-Za-z0-9_.-]+\.out", name):
                continue
            subprocess.run(
                [
                    "scp",
                    f"{operation.target.host}:{value}",
                    str(operation.root / name),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
    return [
        str(path)
        for path in sorted(operation.root.glob("scheduler-*.out"))
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
