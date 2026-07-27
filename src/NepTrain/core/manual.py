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

from .execution import ExecutionError, ExecutionTarget, ExecutionTransport
from .config import DEFAULT_MAX_CONCURRENT, DEFAULT_STRUCTURES_PER_LABEL_JOB
from .scientific_data import (
    ScientificDataError,
    labeled_input_structure_ids,
    structure_id,
    validate_labeled_frames,
)
from .spin import SpinDataError, validate_spin_dataset


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
_SCHEDULER_MISSING_GRACE_SECONDS = 300.0


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

    @property
    def jobs_root(self) -> Path:
        return self.root / "jobs"


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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManualTaskError(
            f"cannot read task metadata {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise ManualTaskError(f"task metadata {path} must contain an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _manual_spec(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: descriptor.get(key)
        for key in (
            "operation_id",
            "kind",
            "target",
            "output",
            "max_concurrent",
            "jobs",
            "scientific_input",
            "files",
            "input_manifest",
        )
    }


def _verify_operation_bundle(root: Path) -> dict[str, Any]:
    descriptor = _read_json(root / "operation.json")
    protocol = descriptor.get("protocol")
    if protocol != "neptrain.manual-operation.v3":
        if protocol in {
            "neptrain.manual-operation.v1",
            "neptrain.manual-operation.v2",
        }:
            raise ManualTaskError(
                "manual run uses an unsafe legacy protocol and cannot be "
                "collected or retried automatically; keep its raw job outputs "
                "and prepare a new v3 run"
            )
        raise ManualTaskError(f"not a NepTrain manual run: {root}")
    for record in descriptor.get("files", []):
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ManualTaskError("manual input manifest escapes the run directory")
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("size", -1))
            or _sha256(path) != record.get("sha256")
        ):
            raise ManualTaskError(f"manual task input drifted: {path}")
    input_manifest = descriptor.get("input_manifest", {})
    manifest_path = root / str(input_manifest.get("path", ""))
    if (
        not manifest_path.is_file()
        or _sha256(manifest_path) != input_manifest.get("sha256")
    ):
        raise ManualTaskError("manual input checksum manifest drifted or is missing")
    expected = _canonical_hash(_manual_spec(descriptor))
    if descriptor.get("spec_sha256") != expected:
        raise ManualTaskError("manual operation content identity is invalid")
    return descriptor


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
    sampling_route_id: str | None = None,
) -> ExecutionTarget:
    if project is None:
        if target_name not in {None, "local"}:
            raise ManualTaskError("--target requires --project")
        return ExecutionTarget("local", "process")
    from .config import load_config

    project_path = Path(project).expanduser().resolve()
    config, _ = load_config(project_path)
    execution = config["execution"]
    route_target = None
    if route == "sampling" and sampling_route_id is not None:
        route_target = dict(execution.get("sampling_route_targets") or {}).get(
            sampling_route_id
        )
    name = target_name or route_target or execution["stage_targets"][route]
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
    scientific_input: Mapping[str, Any] | None = None,
) -> ManualOperation:
    if max_concurrent < 1:
        raise ManualTaskError("max_concurrent must be at least 1")
    setup = Path(target.setup_script).expanduser() if target.setup_script else None
    if target.host and setup and setup.is_file():
        packaged = root / "inputs" / "setup.sh"
        _copy(setup, packaged)
        target = replace(target, setup_script="./inputs/setup.sh")
    value = {
        "protocol": "neptrain.manual-operation.v3",
        "operation_id": operation_id,
        "kind": kind,
        "created_at": _now(),
        "state": "prepared",
        "target": asdict(target),
        "output": str(output.expanduser().resolve()),
        "max_concurrent": int(max_concurrent),
        "jobs": [dict(item) for item in jobs],
        "scientific_input": dict(scientific_input or {}),
        "attempts": [],
    }
    value["files"] = [
        {
            "path": str(path.relative_to(root)),
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != "operation.json"
        and "calculation" not in path.parts
        and "logs" not in path.parts
    ]
    checksum_manifest = root / "input-manifest.sha256"
    checksum_manifest.write_text(
        "".join(
            f"{record['sha256']}  {record['path']}\n"
            for record in value["files"]
        ),
        encoding="utf-8",
    )
    value["input_manifest"] = {
        "path": checksum_manifest.name,
        "sha256": _sha256(checksum_manifest),
    }
    value["spec_sha256"] = _canonical_hash(_manual_spec(value))
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


def prepare_labeling(
    source: str | Path,
    *,
    backend: str,
    output: str | Path,
    workdir: str | Path | None,
    target: ExecutionTarget,
    input_file: str | Path | None = None,
    resource_dir: str | Path | None = None,
    resource_manifest: str | Path | None = None,
    n_cpu: int | None = None,
    use_gamma: bool = False,
    kpoint_mode: str = "auto",
    kspacing: float | None = None,
    ka: Sequence[int] = (1, 1, 1),
    structures_per_job: int = DEFAULT_STRUCTURES_PER_LABEL_JOB,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    teacher_profile: str = "ordinary",
    model_file: str | Path | None = None,
    model_name: str | None = None,
    runner: str | None = None,
    device: str = "cuda",
    precision: str = "float32",
    force: bool = False,
) -> ManualOperation:
    if structures_per_job < 1:
        raise ManualTaskError("structures_per_job must be at least 1")
    selected_resource = (
        resource_dir
        if resource_dir is not None
        else target.labeling_resource_path
    )
    effective_resource = None
    if selected_resource is not None:
        resource_path = Path(selected_resource).expanduser()
        if target.host:
            if not resource_path.is_absolute():
                raise ManualTaskError(
                    "remote labeling resource paths must be absolute on the target"
                )
            effective_resource = str(resource_path)
        else:
            effective_resource = str(resource_path.resolve())
    if backend in {"vasp", "abacus"} and not effective_resource:
        raise ManualTaskError(
            f"{backend} labeling requires --resources, "
            "labeling.resource_path, "
            "or execution.targets.<name>.labeling_resource_path"
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
    if backend == "model":
        if model_file is None or not Path(model_file).expanduser().is_file():
            raise ManualTaskError(
                "model labeling requires an existing --model"
            )
        if not model_name or not str(model_name).strip():
            raise ManualTaskError("model labeling requires --model-name")
        if not runner or not shlex.split(str(runner)):
            raise ManualTaskError("model labeling requires --runner")
    _progress(f"label: reading structures from {Path(source).expanduser()}")
    output_path = _output_path(output, force=force)
    frames = _frames(Path(source))
    try:
        _, spin_frames = validate_spin_dataset(
            frames,
            require_mforce=False,
        )
    except SpinDataError as error:
        raise ManualTaskError(
            f"labeling input spin contract is invalid: {error}"
        ) from error
    frame_ids = [structure_id(frame) for frame in frames]
    resource_provenance = None
    resource_manifest_source = None
    if backend == "vasp":
        from .dft.vasp.native import (
            NativeVaspError,
            validate_vasp_input_file,
            validate_vasp_structure,
        )
        from .dft.vasp.resources import (
            VaspResourceError,
            validate_vasp_manifest_elements,
            validate_vasp_resources,
            vasp_element_order,
        )

        if resource_manifest is None:
            raise ManualTaskError(
                "VASP labeling requires --potcar-manifest or "
                "labeling.potcar_manifest_path"
            )
        resource_manifest_source = Path(resource_manifest).expanduser().resolve()
        orders = sorted({vasp_element_order(frame) for frame in frames})
        try:
            electronic_mode = (
                validate_vasp_input_file(input_file)
                if input_file is not None
                else "non_spin_polarized"
            )
            for frame in frames:
                validate_vasp_structure(
                    frame,
                    electronic_mode=electronic_mode,
                )
            resource_provenance = validate_vasp_manifest_elements(
                resource_manifest_source,
                orders,
            )
            if target.host is None:
                resource_provenance["validated_orders"] = [
                    validate_vasp_resources(
                        str(effective_resource),
                        resource_manifest_source,
                        next(
                            frame
                            for frame in frames
                            if vasp_element_order(frame) == order
                        ),
                    )
                    for order in orders
                ]
        except (NativeVaspError, VaspResourceError) as error:
            raise ManualTaskError(str(error)) from error
    elif backend == "abacus":
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

        if resource_manifest is None:
            raise ManualTaskError(
                "ABACUS labeling requires --resource-manifest or "
                "labeling.resource_manifest_path"
            )
        resource_manifest_source = Path(resource_manifest).expanduser().resolve()
        orders = sorted(
            {tuple(dict.fromkeys(frame.get_chemical_symbols())) for frame in frames}
        )
        parameters = (
            read_input_file(str(input_file))
            if input_file is not None
            else {}
        )
        require_orbitals = (
            str(parameters.get("basis_type", "pw")).strip().lower() == "lcao"
        )
        try:
            validate_abacus_spin_contract(
                dict(parameters),
                spin_frame=bool(spin_frames),
            )
            resource_provenance = validate_abacus_manifest_elements(
                resource_manifest_source,
                orders,
            )
            if target.host is None:
                resource_provenance["validated_orders"] = [
                    validate_abacus_resources(
                        str(effective_resource),
                        resource_manifest_source,
                        next(
                            frame
                            for frame in frames
                            if tuple(
                                dict.fromkeys(frame.get_chemical_symbols())
                            )
                            == order
                        ),
                        require_orbitals=require_orbitals,
                    )[0]
                    for order in orders
                ]
        except (AbacusResourceError, NativeAbacusError) as error:
            raise ManualTaskError(str(error)) from error
    first_by_id: dict[str, int] = {}
    duplicates = []
    for index, identifier in enumerate(frame_ids):
        if identifier in first_by_id:
            duplicates.append((first_by_id[identifier], index))
        else:
            first_by_id[identifier] = index
    if duplicates:
        first, duplicate = duplicates[0]
        raise ManualTaskError(
            "labeling input contains duplicate physical structures at indices "
            f"{first} and {duplicate}; deduplicate before submission"
        )
    _progress(
        f"label: read {len(frames)} structure(s); "
        f"{structures_per_job} structure(s) per Slurm task"
    )
    payload = {
        "backend": backend,
        "source": str(Path(source).expanduser().resolve()),
        "frames": len(frames),
        "target": target.name,
    }
    root, operation_id = _prepare_root("label", workdir, payload)
    common = root / "inputs"
    copied_input = (
        str(Path(_copy(Path(input_file), common / Path(input_file).name)).relative_to(root))
        if input_file
        else None
    )
    copied_resource_manifest = (
        str(
            Path(
                _copy(
                    resource_manifest_source,
                    common / f"{backend}-resources.json",
                )
            ).relative_to(root)
        )
        if resource_manifest_source is not None
        else None
    )
    copied_model = (
        str(
            Path(
                _copy(
                    Path(model_file),
                    common / Path(model_file).name,
                )
            ).relative_to(root)
        )
        if model_file is not None
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
            "resource_manifest": copied_resource_manifest,
            "n_cpu": effective_n_cpu,
            "use_gamma": bool(use_gamma),
            "kpoint_mode": kpoint_mode,
            "kspacing": kspacing,
            "ka": [int(value) for value in ka],
            "teacher_profile": teacher_profile,
            "model_file": copied_model,
            "model_name": model_name,
            "runner": runner,
            "device": device,
            "precision": precision,
        }
        _write_json(job / "request.json", request)
        jobs.append(
            {
                "index": index,
                "first_frame": start,
                "frame_count": min(structures_per_job, len(frames) - start),
                "frame_ids": frame_ids[
                    start : start + structures_per_job
                ],
                "request": str((job / "request.json").relative_to(root)),
            }
        )
    return _write_operation(
        root,
        operation_id=operation_id,
        kind="label",
        target=target,
        output=output_path,
        jobs=jobs,
        max_concurrent=max_concurrent,
        scientific_input={
            "structure_id_version": "neptrain.structure-id.v2",
            "frame_count": len(frame_ids),
            "frame_ids": frame_ids,
            "labeling_resources": resource_provenance,
        },
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
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
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
                    "input_structure_id": structure_id(frame),
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
    descriptor = _verify_operation_bundle(root)
    jobs = descriptor.get("jobs", [])
    return ManualOperation(
        root=root,
        operation_id=str(descriptor["operation_id"]),
        kind=str(descriptor["kind"]),
        target=ExecutionTarget.from_mapping(
            str(descriptor["target"]["name"]), descriptor["target"]
        ),
        shard_count=len(jobs),
    )


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _archive_worker_state(job: Path, result_file: Path) -> None:
    existing = [
        path
        for path in (
            job / "execution.json",
            job / "result.json",
            result_file,
            job / "calculation",
        )
        if path.exists()
    ]
    if not existing:
        return
    archive = (
        job
        / "attempts"
        / datetime.now().strftime("worker-%Y%m%d-%H%M%S-%f")
    )
    archive.mkdir(parents=True)
    for path in existing:
        shutil.move(str(path), archive / path.name)


def run_manual_worker(root_path: str | Path, index: int) -> int:
    operation = load_operation(root_path)
    descriptor = _verify_operation_bundle(operation.root)
    jobs = descriptor["jobs"]
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
            descriptor = _verify_operation_bundle(operation.root)
            try:
                _validate_job_result(operation, index, descriptor)
            except (ManualTaskError, OSError, ValueError):
                pass
            else:
                if operation.target.host is None:
                    _publish_if_ready(operation)
                return 0
            result_file = _expected_job_result(operation, job)
            _archive_worker_state(job, result_file)
            _write_json(
                execution,
                {
                    "state": "RUNNING",
                    "operation_id": operation.operation_id,
                    "operation_spec_sha256": descriptor["spec_sha256"],
                    "job_index": index,
                    "pid": os.getpid(),
                    "started_at": _now(),
                },
            )
            calculation_dir = job / "calculation"
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
            elif operation.kind == "label":
                from .labeling import LabelRequest, label

                result = label(
                    LabelRequest(
                        source=_resolve(operation.root, request["source"]),
                        output_file=result_file,
                        work_dir=calculation_dir,
                        settings={
                            "input_file": _resolve(
                                operation.root,
                                request.get("input_file"),
                            ),
                            "resource_dir": _resolve(
                                operation.root,
                                request.get("resource_dir"),
                            ),
                            "resource_manifest": _resolve(
                                operation.root,
                                request.get("resource_manifest"),
                            ),
                            "n_cpu": int(request["n_cpu"]),
                            "use_gamma": bool(request["use_gamma"]),
                            "kpoint_mode": request["kpoint_mode"],
                            "kspacing": request.get("kspacing"),
                            "ka": tuple(request["ka"]),
                            "profile": request["teacher_profile"],
                            "model_file": _resolve(
                                operation.root,
                                request.get("model_file"),
                            ),
                            "model_name": request.get("model_name"),
                            "runner": request.get("runner"),
                            "device": request.get("device", "cuda"),
                            "precision": request.get(
                                "precision",
                                "float32",
                            ),
                        },
                    ),
                    request["backend"],
                )
                written = ase_read(result_file, index=":")
                written_frames = (
                    written if isinstance(written, list) else [written]
                )
                try:
                    validate_labeled_frames(written_frames)
                    frame_ids = labeled_input_structure_ids(written_frames)
                except ScientificDataError as error:
                    raise ManualTaskError(
                        f"label job {index} produced invalid labels: {error}"
                    ) from error
                expected_ids = [str(value) for value in job_meta["frame_ids"]]
                if frame_ids != expected_ids:
                    raise ManualTaskError(
                        f"label job {index} result structures do not match its "
                        "input frame identities and order"
                    )
                metrics = {
                    "backend": result.backend,
                    "origin": result.provenance.get("origin"),
                    "provenance": dict(result.provenance),
                    "frames": len(written_frames),
                }
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
            artifact = {
                "path": str(result_file.relative_to(job)),
                "sha256": _sha256(result_file),
                "size": result_file.stat().st_size,
            }
            result_descriptor = {
                "protocol": "neptrain.manual-job-result.v1",
                "operation_id": operation.operation_id,
                "operation_spec_sha256": descriptor["spec_sha256"],
                "job_index": index,
                "kind": operation.kind,
                "completed_at": _now(),
                "artifact": artifact,
                "metrics": metrics,
            }
            if operation.kind == "label":
                result_descriptor["frame_ids"] = frame_ids
                result_descriptor["frame_count"] = len(frame_ids)
            elif operation.kind == "md":
                trajectory = ase_read(result_file, index=":")
                trajectory_frames = (
                    trajectory if isinstance(trajectory, list) else [trajectory]
                )
                if not trajectory_frames:
                    raise ManualTaskError(
                        f"MD job {index} produced an empty trajectory"
                    )
                result_descriptor["frame_count"] = len(trajectory_frames)
            _write_json(job / "result.json", result_descriptor)
            _write_json(
                execution,
                {
                    "state": "COMPLETED",
                    "operation_id": operation.operation_id,
                    "operation_spec_sha256": descriptor["spec_sha256"],
                    "job_index": index,
                    "pid": os.getpid(),
                    "completed_at": _now(),
                },
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
                    "operation_id": operation.operation_id,
                    "operation_spec_sha256": descriptor.get("spec_sha256"),
                    "job_index": index,
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
    descriptor = _verify_operation_bundle(operation.root)
    token = f"{os.getpid()}-{time.time_ns()}"
    archive = operation.root.parent / (
        f".{operation.operation_id}-{token}.tar.gz"
    )
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
        remote_archive = f"{remote_parent}/.incoming/{archive.name}"
        descriptor_sha256 = _sha256(operation.descriptor)
        _progress(f"{operation.kind}: creating remote run directory on {target.name}")
        prepared = transport.run_script(
            """set -eo pipefail
parent=$1
destination=$2
expected=$3
mkdir -p -- "$parent/.incoming"
if [ -e "$destination" ]; then
  if [ ! -f "$destination/operation.json" ]; then
    printf '%s\\n' INCOMPLETE
    exit 0
  fi
  actual=$(sha256sum "$destination/operation.json" | cut -d' ' -f1)
  if [ "$actual" != "$expected" ]; then
    echo "remote manual task conflicts with local operation identity: $destination" >&2
    exit 17
  fi
  if (cd "$destination" && sha256sum -c input-manifest.sha256 >/dev/null); then
    printf '%s\\n' READY
  else
    printf '%s\\n' INCOMPLETE
  fi
  exit 0
fi
printf '%s\\n' MISSING
""",
            remote_parent,
            remote,
            descriptor_sha256,
        )
        if prepared.returncode != 0:
            raise ManualTaskError(
                (prepared.stderr or prepared.stdout).strip()
            )
        if prepared.stdout.strip().splitlines()[-1:] == ["READY"]:
            _progress(f"{operation.kind}: remote task already verified at {remote}")
            return remote
        _progress(
            f"{operation.kind}: uploading {archive.stat().st_size} bytes to {target.name}"
        )
        transport.copy(
            archive,
            f"{target.host}:{remote_archive}",
            check=True,
        )
        _progress(f"{operation.kind}: extracting remote task bundle")
        published = transport.run_script(
            """set -eo pipefail
parent=$1
destination=$2
archive=$3
expected_archive=$4
expected_descriptor=$5
token=$6
name=$7
lock="$parent/.$name.deploy.lock"
if ! mkdir "$lock" 2>/dev/null; then
  if [ -f "$destination/operation.json" ]; then
    actual=$(sha256sum "$destination/operation.json" | cut -d' ' -f1)
    if [ "$actual" = "$expected_descriptor" ] && \
       (cd "$destination" && sha256sum -c input-manifest.sha256 >/dev/null); then
      exit 0
    fi
  fi
  echo "another deployment owns $destination; retry later" >&2
  exit 75
fi
temporary="$parent/.$name.extracting-$token"
cleanup() {
  rm -f -- "$archive"
  rm -rf -- "$temporary"
  rmdir "$lock" 2>/dev/null || true
}
trap cleanup EXIT
actual=$(sha256sum "$archive" | cut -d' ' -f1)
[ "$actual" = "$expected_archive" ]
mkdir -p "$temporary"
tar -xzf "$archive" -C "$temporary" --strip-components=1
[ -f "$temporary/operation.json" ]
actual=$(sha256sum "$temporary/operation.json" | cut -d' ' -f1)
[ "$actual" = "$expected_descriptor" ]
(cd "$temporary" && sha256sum -c input-manifest.sha256 >/dev/null)
if [ -e "$destination" ]; then
  if [ -f "$destination/operation.json" ]; then
    actual=$(sha256sum "$destination/operation.json" | cut -d' ' -f1)
    if [ "$actual" != "$expected_descriptor" ]; then
      echo "remote manual task conflicts with local operation identity: $destination" >&2
      exit 17
    fi
  fi
  quarantine="$destination.incomplete.$(date +%Y%m%d-%H%M%S).$$"
  mv -- "$destination" "$quarantine"
fi
mv -- "$temporary" "$destination"
(cd "$destination" && sha256sum -c input-manifest.sha256 >/dev/null)
""",
            remote_parent,
            remote,
            remote_archive,
            _sha256(archive),
            descriptor_sha256,
            token,
            operation.operation_id,
        )
        if published.returncode != 0:
            raise ManualTaskError(
                (published.stderr or published.stdout).strip()
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
        "label": "labeled.xyz",
        "md": "trajectory.xyz",
    }[kind]


def _expected_job_result(operation: ManualOperation, job: Path) -> Path:
    return job / _result_filename(operation.kind)


def _validate_job_result(
    operation: ManualOperation,
    index: int,
    descriptor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    descriptor = (
        dict(descriptor)
        if descriptor is not None
        else _verify_operation_bundle(operation.root)
    )
    job_meta = descriptor["jobs"][index]
    job = operation.jobs_root / f"{index:06d}"
    execution = _read_json(job / "execution.json")
    if (
        execution.get("state") != "COMPLETED"
        or execution.get("operation_id") != operation.operation_id
        or execution.get("operation_spec_sha256")
        != descriptor["spec_sha256"]
        or int(execution.get("job_index", -1)) != index
    ):
        raise ManualTaskError(f"job {index} execution is not a matching completion")
    result = _read_json(job / "result.json")
    if (
        result.get("protocol") != "neptrain.manual-job-result.v1"
        or result.get("operation_id") != operation.operation_id
        or result.get("operation_spec_sha256") != descriptor["spec_sha256"]
        or int(result.get("job_index", -1)) != index
        or result.get("kind") != operation.kind
    ):
        raise ManualTaskError(f"job {index} result metadata does not match its task")
    record = result.get("artifact")
    if not isinstance(record, Mapping):
        raise ManualTaskError(f"job {index} result has no artifact manifest")
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ManualTaskError(f"job {index} result path escapes its job directory")
    artifact = (job / relative).resolve()
    if (
        not artifact.is_file()
        or artifact != _expected_job_result(operation, job).resolve()
        or artifact.stat().st_size != int(record.get("size", -1))
        or _sha256(artifact) != record.get("sha256")
    ):
        raise ManualTaskError(f"job {index} result artifact drifted or is missing")
    if operation.kind == "label":
        loaded = ase_read(artifact, index=":")
        frames = loaded if isinstance(loaded, list) else [loaded]
        try:
            validate_labeled_frames(frames)
            actual_ids = labeled_input_structure_ids(frames)
        except ScientificDataError as error:
            raise ManualTaskError(f"job {index} labels are invalid: {error}") from error
        expected_ids = [str(value) for value in job_meta.get("frame_ids", [])]
        if (
            int(result.get("frame_count", -1)) != len(expected_ids)
            or list(result.get("frame_ids", [])) != expected_ids
            or actual_ids != expected_ids
        ):
            raise ManualTaskError(
                f"job {index} has missing, duplicate, or reordered label frames"
            )
    return result


def _completed_job_is_collectable(
    operation: ManualOperation,
    job: Path,
    descriptor: Mapping[str, Any] | None = None,
) -> bool:
    try:
        index = int(job.name)
        _validate_job_result(operation, index, descriptor)
    except (ManualTaskError, OSError, ValueError):
        return False
    return True


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
            try:
                state = (
                    _read_json(execution).get("state")
                    if execution.is_file()
                    else None
                )
            except ManualTaskError:
                state = None
            synchronized = state == "FAILED" or (
                state == "COMPLETED"
                and _completed_job_is_collectable(operation, job, descriptor)
            )
            if synchronized:
                continue
            incomplete.append(index)
            members.extend(
                (
                    f"{relative}/execution.json",
                    f"{relative}/result.json",
                    f"{relative}/{_result_filename(operation.kind)}",
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
            f"{remote.rstrip('/')}/jobs",
            members,
            operation.jobs_root,
        )
        _progress(
            f"{operation.kind}: received {len(archived)} remote result "
            "path(s)"
        )


def _all_jobs_completed(operation: ManualOperation) -> bool:
    descriptor = _verify_operation_bundle(operation.root)
    for index in range(operation.shard_count):
        job = operation.jobs_root / f"{index:06d}"
        if not _completed_job_is_collectable(operation, job, descriptor):
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


def _final_result_error(
    operation: ManualOperation,
    descriptor: Mapping[str, Any],
) -> str | None:
    record = descriptor.get("result")
    if not isinstance(record, Mapping):
        return "completed operation has no final result manifest"
    output = Path(str(record.get("path", "")))
    if (
        not output.is_file()
        or output.stat().st_size != int(record.get("size", -1))
        or _sha256(output) != record.get("sha256")
    ):
        return f"final result drifted or is missing: {output}"
    if operation.kind == "label":
        try:
            loaded = ase_read(output, index=":")
            frames = loaded if isinstance(loaded, list) else [loaded]
            validate_labeled_frames(frames)
            actual_ids = labeled_input_structure_ids(frames)
        except (OSError, ScientificDataError, ValueError) as error:
            return f"final labeling result is invalid: {error}"
        expected_ids = list(
            descriptor.get("scientific_input", {}).get("frame_ids", [])
        )
        if (
            actual_ids != expected_ids
            or list(record.get("frame_ids", [])) != expected_ids
            or int(record.get("frame_count", -1)) != len(expected_ids)
        ):
            return (
                "final labeling result has missing, duplicate, or reordered "
                "frames"
            )
    return None


def _collect_unlocked(operation: ManualOperation) -> Path:
    descriptor = _verify_operation_bundle(operation.root)
    _ensure_remote_pointer(operation, descriptor.get("remote_root"))
    output = Path(descriptor["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    if operation.kind == "train":
        _validate_job_result(operation, 0, descriptor)
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
            _validate_job_result(operation, index, descriptor)
            path = _expected_job_result(
                operation,
                operation.jobs_root / f"{index:06d}",
            )
            loaded = ase_read(path, index=":")
            frames.extend(loaded if isinstance(loaded, list) else [loaded])
        if operation.kind == "label":
            try:
                validate_labeled_frames(frames)
                actual_ids = labeled_input_structure_ids(frames)
            except ScientificDataError as error:
                raise ManualTaskError(
                    f"cannot merge invalid labels: {error}"
                ) from error
            expected_ids = list(
                descriptor.get("scientific_input", {}).get("frame_ids", [])
            )
            if actual_ids != expected_ids:
                raise ManualTaskError(
                    "label merge detected missing, duplicate, or reordered frames"
                )
        elif not frames:
            raise ManualTaskError("cannot merge an empty MD trajectory")
        ase_write(temporary, frames, format="extxyz")
        if operation.kind == "label":
            restored = ase_read(temporary, index=":", format="extxyz")
            restored_frames = (
                restored if isinstance(restored, list) else [restored]
            )
            validate_labeled_frames(restored_frames)
            restored_ids = labeled_input_structure_ids(restored_frames)
            if restored_ids != expected_ids:
                raise ManualTaskError(
                    "label merge did not survive the final extxyz round trip"
                )
    if output.exists() or output.is_symlink():
        archive = (
            operation.root
            / "repairs"
            / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        )
        archive.mkdir(parents=True)
        shutil.move(str(output), archive / output.name)
    temporary.replace(output)
    _ensure_visible_result(operation, output)
    descriptor["state"] = "complete"
    descriptor["completed_at"] = _now()
    descriptor["result"] = {
        "path": str(output),
        "sha256": _sha256(output),
        "size": output.stat().st_size,
    }
    if operation.kind == "label":
        descriptor["result"]["frame_count"] = len(expected_ids)
        descriptor["result"]["frame_ids"] = expected_ids
    descriptor.pop("collection_error", None)
    _write_json(operation.descriptor, descriptor)
    return output


def _collect(operation: ManualOperation) -> Path:
    lock_path = operation.root / ".collection.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        descriptor = _read_json(operation.descriptor)
        if descriptor.get("state") == "complete":
            error = _final_result_error(operation, descriptor)
            if error is None:
                return Path(descriptor["result"]["path"])
        if not _all_jobs_completed(operation):
            raise ManualTaskError(
                "cannot publish or repair an incomplete manual operation"
            )
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
        try:
            return ExecutionTransport(operation.target).run_script(
                'exec "$@"',
                *command,
                timeout=20,
            )
        except ExecutionError as error:
            return subprocess.CompletedProcess(
                list(command),
                124,
                stdout="",
                stderr=str(error),
            )
    try:
        return subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            list(command),
            124,
            stdout="",
            stderr="scheduler command timed out after 20s",
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
    descriptor = _verify_operation_bundle(operation.root)
    _ensure_remote_pointer(operation, descriptor.get("remote_root"))
    repair_error = None
    if descriptor.get("state") == "complete" and descriptor.get("result"):
        result_record = descriptor["result"]
        output = Path(str(result_record.get("path", "")))
        final_error = _final_result_error(operation, descriptor)
        if final_error is None:
            _ensure_visible_result(operation, output)
        else:
            try:
                _collect(operation)
                descriptor = _verify_operation_bundle(operation.root)
            except Exception as error:
                repair_error = f"{final_error}; automatic repair failed: {error}"
    scheduler_state = None
    scheduler_error = None
    if (
        operation.target.executor == "slurm"
        and descriptor.get("state") not in {"complete", "cancelled", "failed"}
    ):
        scheduler_state, scheduler_error = _slurm_job_state(
            operation, descriptor.get("job_id")
        )
    changed = False
    if (
        operation.target.executor == "slurm"
        and descriptor.get("state") in {"submitted", "cancelling"}
        and scheduler_state is None
    ):
        missing_since = descriptor.get("scheduler_missing_since")
        if missing_since is None:
            missing_since = time.time()
            descriptor["scheduler_missing_since"] = missing_since
            descriptor["scheduler_missing_observations"] = 1
            changed = True
        else:
            descriptor["scheduler_missing_observations"] = int(
                descriptor.get("scheduler_missing_observations", 0)
            ) + 1
            changed = True
        if time.time() - float(missing_since) >= _SCHEDULER_MISSING_GRACE_SECONDS:
            scheduler_state = "LOST"
            scheduler_error = (
                scheduler_error
                or "job remained absent from squeue and sacct beyond the "
                "scheduler accounting grace period"
            )
    elif scheduler_state is not None:
        if descriptor.pop("scheduler_missing_since", None) is not None:
            changed = True
        if descriptor.pop("scheduler_missing_observations", None) is not None:
            changed = True

    states = []
    errors = []
    if repair_error:
        errors.append({"index": None, "error": repair_error})
    for index in range(operation.shard_count):
        path = operation.jobs_root / f"{index:06d}" / "execution.json"
        if not path.is_file():
            states.append("PENDING")
            continue
        try:
            value = _read_json(path)
        except ManualTaskError as error:
            states.append("INVALID")
            errors.append({"index": index, "error": str(error)})
            continue
        state = str(value.get("state", "UNKNOWN"))
        if state == "COMPLETED":
            try:
                _validate_job_result(operation, index, descriptor)
            except (ManualTaskError, OSError, ValueError) as error:
                state = "INVALID"
                errors.append({"index": index, "error": str(error)})
        states.append(state)
        if state == "FAILED":
            errors.append({"index": index, "error": value.get("error", "")})
    terminal_scheduler_failure = (
        (
            scheduler_state in _SLURM_FAILURE
            and scheduler_state != "CANCELLED"
        )
        or scheduler_state == "LOST"
    )
    if terminal_scheduler_failure:
        failed_indices = {item["index"] for item in errors}
        for index, shard_state in enumerate(states):
            if shard_state != "COMPLETED" and index not in failed_indices:
                errors.append(
                    {
                        "index": index,
                        "error": (
                            f"Slurm job ended in {scheduler_state}"
                            if scheduler_state != "LOST"
                            else "Slurm job disappeared from queue and accounting"
                        ),
                    }
                )
    cancellation_requested = descriptor.get("state") == "cancelling"
    if scheduler_state == "CANCELLED":
        failed_indices = {item["index"] for item in errors}
        for index, shard_state in enumerate(states):
            if shard_state != "COMPLETED" and index not in failed_indices:
                errors.append(
                    {
                        "index": index,
                        "error": "job was cancelled before producing a valid result",
                    }
                )
        descriptor["state"] = "cancelled"
        descriptor["cancelled_at"] = _now()
        changed = True
    elif descriptor.get("state") == "cancelled":
        failed_indices = {item["index"] for item in errors}
        for index, shard_state in enumerate(states):
            if shard_state != "COMPLETED" and index not in failed_indices:
                errors.append(
                    {
                        "index": index,
                        "error": "job was cancelled before producing a valid result",
                    }
                )
    if descriptor.get("state") == "cancelled":
        state = "cancelled"
        reason = f"{states.count('COMPLETED')}/{operation.shard_count} jobs completed before cancellation"
    elif cancellation_requested:
        state = "cancelling"
        reason = (
            f"cancellation requested; {states.count('COMPLETED')}/"
            f"{operation.shard_count} jobs completed"
        )
    elif repair_error:
        state = "damaged"
        reason = repair_error
    elif errors:
        state = "failed"
        reason = f"{len(errors)} of {operation.shard_count} jobs failed"
    elif states and all(value == "COMPLETED" for value in states):
        if descriptor.get("state") != "complete":
            try:
                _collect(operation)
                descriptor = _verify_operation_bundle(operation.root)
            except Exception as error:
                descriptor["collection_error"] = str(error)
                descriptor["state"] = "failed"
                changed = True
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
    if state in {"failed", "cancelled"} and descriptor.get("state") != state:
        descriptor["state"] = state
        changed = True
    if changed:
        _write_json(operation.descriptor, descriptor)
    next_action = None
    if state in {"prepared", "submitted", "running", "cancelling"}:
        next_action = f"neptrain task wait {shlex.quote(str(operation.root))}"
    elif state in {"failed", "cancelled", "damaged"}:
        next_action = f"neptrain task retry {shlex.quote(str(operation.root))}"
    return {
        "protocol": "neptrain.manual-status.v1",
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
        "result": (
            descriptor.get("result", {}).get("path")
            if isinstance(descriptor.get("result"), Mapping)
            else None
        ),
        "remote_directory": descriptor.get("remote_root"),
        "run_directory": str(operation.root),
        "next_action": next_action,
        "jobs": [
            {"index": index, "state": shard_state}
            for index, shard_state in enumerate(states)
        ],
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
        if status["state"] in {
            "complete",
            "failed",
            "cancelled",
            "damaged",
        }:
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
        refreshed = refresh_operation(operation)
        if refreshed["state"] in {"complete", "failed", "cancelled"}:
            return refreshed
        if refreshed.get("scheduler_state") is not None:
            raise ManualTaskError((completed.stderr or completed.stdout).strip())
    descriptor["state"] = "cancelling"
    cancellation = {
        "requested_at": _now(),
        "job_id": str(job_id),
        "completed_jobs": status["completed"],
    }
    if completed.returncode != 0:
        cancellation["scheduler_response"] = (
            completed.stderr or completed.stdout
        ).strip()
    descriptor.setdefault("cancellations", []).append(cancellation)
    _write_json(operation.descriptor, descriptor)
    return refresh_operation(operation)


def retry_failed(operation: ManualOperation) -> dict[str, Any]:
    if operation.target.executor != "slurm":
        raise ManualTaskError("retry is only needed for submitted Slurm operations")
    status = refresh_operation(operation)
    if status["state"] == "cancelling":
        raise ManualTaskError(
            "cancellation is not terminal yet; wait for task status to report "
            "cancelled before retrying"
        )
    descriptor = _verify_operation_bundle(operation.root)
    failed = [
        index
        for index in range(operation.shard_count)
        if not _completed_job_is_collectable(
            operation,
            operation.jobs_root / f"{index:06d}",
            descriptor,
        )
    ]
    if not failed:
        raise ManualTaskError(
            "operation has no failed jobs; fix the reported collection error "
            "and run task status again"
        )
    remote = descriptor.get("remote_root") or str(operation.root)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_entry = _result_filename(operation.kind)
    retry_entries = ["execution.json", "result.json", result_entry]
    retry_entries.append("calculation")
    if operation.target.host:
        transport = ExecutionTransport(operation.target)
        _progress(
            f"{operation.kind}: archiving {len(failed)} failed remote job(s)"
        )
        commands = ["set -eo pipefail"]
        for index in failed:
            job = f"{remote}/jobs/{index:06d}"
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
    descriptor.pop("cancelled_at", None)
    descriptor.pop("scheduler_missing_since", None)
    descriptor.pop("scheduler_missing_observations", None)
    descriptor.pop("collection_error", None)
    descriptor["attempts"].append(
        {"submitted_at": _now(), "job_id": job_id, "indices": failed}
    )
    _write_json(operation.descriptor, descriptor)
    return refresh_operation(operation)


def operation_logs(operation: ManualOperation) -> list[str]:
    descriptor = _read_json(operation.descriptor)
    logs_root = operation.root / "logs"
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
        remote_logs = f"{remote.rstrip('/')}/logs"
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
    "prepare_labeling",
    "prepare_md",
    "prepare_training",
    "refresh_operation",
    "retry_failed",
    "run_manual_worker",
    "submit_operation",
    "target_from_project",
    "wait_operation",
]
