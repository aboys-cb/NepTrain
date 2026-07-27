"""Small molecular-dynamics Interface shared by both MD Adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Mapping

import numpy as np
from ase import Atoms
from ase.io import iread as ase_iread
from ase.io import write as ase_write

from ..persistence import atomic_write_json
from .lammps import LammpsError, run_lammps
from .health import (
    TrajectoryHealthError,
    TrajectoryHealthPolicy,
    classify_trajectory,
)


class MdError(RuntimeError):
    pass


@dataclass(frozen=True)
class MdRequest:
    atoms: Atoms
    model_file: Path
    output_dir: Path
    output_file: Path
    temperature: float
    steps: int
    seed: int = 12345
    replica: int = 1
    timestep: float = 0.001
    ensemble: str = "nvt"
    pressure: float = 0.0
    spin: bool = False
    spin_temperature: float | None = None
    template_path: Path | None = None
    inference_backend: str = "auto"
    lmp_command: str = "lmp"
    mpiexec: str = "mpirun"
    mpi_ranks: int = 1
    pre_failure_frames: int = 2
    bad_tail_frames: int = 1
    health: Mapping[str, object] = field(default_factory=dict)
    route_id: str = ""
    route_fingerprint: str = ""


@dataclass(frozen=True)
class MdResult:
    backend: str
    trajectory: Path
    run_directory: Path
    inference_backend: str | None = None
    pair_style: str | None = None
    completed: bool = True
    last_step: int | None = None
    failure_code: str | None = None
    failure_reason: str | None = None
    health_report: Path | None = None


def _template(request: MdRequest) -> str:
    if request.template_path is not None:
        return request.template_path.read_text(encoding="utf-8")
    mode = "spin-" if request.spin else ""
    resource = files("NepTrain.core.md").joinpath(
        f"templates/{mode}{request.ensemble}.in"
    )
    return resource.read_text(encoding="utf-8")


def _gpumd_failure_reason(returncode: int, stderr: Path, stdout: Path) -> str:
    lines: list[str] = []
    for path in (stderr, stdout):
        if path.is_file():
            lines.extend(
                line.strip()
                for line in path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if line.strip()
            )
    detail = lines[-1] if lines else "no output detail"
    return f"exit code {returncode}: {detail}"


def _read_gpumd_frames(
    dump: Path,
    *,
    timestep_fs: float,
    dump_interval: int,
    allow_incomplete_tail: bool,
) -> list[Atoms]:
    frames: list[Atoms] = []
    try:
        for index, frame in enumerate(
            ase_iread(dump, index=":", format="extxyz")
        ):
            time_value = frame.info.get("Time")
            if isinstance(time_value, int | float | np.number):
                step = int(round(float(time_value) / timestep_fs))
            else:
                step = (index + 1) * dump_interval
            frame.info.update(
                Config_type=f"gpumd-step-{step}",
                gpumd_step=step,
                md_step=step,
            )
            if (
                "nep_force" not in frame.arrays
                and frame.calc is not None
                and "forces" in frame.calc.results
            ):
                frame.set_array(
                    "nep_force",
                    np.asarray(frame.calc.results["forces"], dtype=np.float64),
                )
            frames.append(frame)
    except (EOFError, OSError, ValueError):
        if not allow_incomplete_tail or not frames:
            raise
    return frames


def _run_gpumd(
    request: MdRequest,
    health_policy: TrajectoryHealthPolicy,
) -> MdResult:
    if request.spin:
        raise MdError("spin evolution uses the LAMMPS DynSpin backend")
    from ..gpumd.io import GpumdInputError, RunInput

    timestep_fs = request.timestep * 1000.0
    try:
        run = RunInput(request.model_file)
        if request.template_path is None:
            run.use_default(
                ensemble=request.ensemble,
                temperature=request.temperature,
                pressure=request.pressure,
                steps=request.steps,
                timestep_fs=timestep_fs,
                seed=request.seed,
            )
        else:
            run.read_run(request.template_path)
            run.configure(
                temperature=request.temperature,
                pressure=request.pressure,
                steps=request.steps,
                timestep_fs=timestep_fs,
                seed=request.seed,
            )
        dump_interval = run.dump_interval()
        effective_timestep_fs = run.timestep_fs()
        process = run.calculate(request.atoms, request.output_dir)
    except (GpumdInputError, OSError, ValueError) as error:
        raise MdError(str(error)) from error

    process_completed = process.returncode == 0
    process_failure = (
        None
        if process_completed
        else _gpumd_failure_reason(
            process.returncode, process.stderr, process.stdout
        )
    )
    dump = request.output_dir / "dump.xyz"
    if not dump.is_file():
        if process_failure is None:
            raise MdError("GPUMD completed without dump.xyz")
        raise MdError(
            f"GPUMD failed ({process_failure}) and produced no recoverable trajectory"
        )
    try:
        frames = _read_gpumd_frames(
            dump,
            timestep_fs=effective_timestep_fs,
            dump_interval=dump_interval,
            allow_incomplete_tail=not process_completed,
        )
    except (EOFError, OSError, ValueError) as error:
        if process_failure is None:
            raise MdError(f"GPUMD produced an unreadable dump.xyz: {error}") from error
        raise MdError(
            f"GPUMD failed ({process_failure}) and produced no recoverable "
            f"trajectory: {error}"
        ) from error
    if not frames:
        raise MdError("GPUMD produced no readable frames in dump.xyz")

    try:
        health = classify_trajectory(
            frames,
            request.atoms,
            process_completed=process_completed,
            policy=health_policy,
            pre_failure_frames=request.pre_failure_frames,
            bad_tail_frames=request.bad_tail_frames,
        )
    except TrajectoryHealthError as error:
        raise MdError(str(error)) from error
    for frame, window in zip(frames, health.windows):
        frame.info.update(
            md_window=window,
            md_completed=health.trajectory_completed,
        )
    health_payload = health.to_dict()
    health_payload["process_failure_reason"] = process_failure
    health_path = request.output_dir / "trajectory-health.json"
    atomic_write_json(health_path, health_payload)
    failure_code = None
    failure_reason = None
    if health.first_bad_frame is not None:
        failure_code = "trajectory_health"
        failure_reason = (
            f"trajectory health failed at step {health.first_bad_step}: "
            + ", ".join(health.reason_codes)
        )
        if process_failure is not None:
            failure_reason += f"; GPUMD also failed ({process_failure})"
    elif not process_completed:
        failure_code = "gpumd_nonzero_exit"
        failure_reason = process_failure
    request.output_file.parent.mkdir(parents=True, exist_ok=True)
    ase_write(request.output_file, frames, format="extxyz")
    return MdResult(
        backend="gpumd",
        trajectory=request.output_file,
        run_directory=request.output_dir,
        completed=health.trajectory_completed,
        last_step=int(frames[-1].info["md_step"]),
        failure_code=failure_code,
        failure_reason=failure_reason,
        health_report=health_path,
    )


def run_md(request: MdRequest, backend: str) -> MdResult:
    allowed_ensembles = {"nvt", "npt"}
    if backend == "gpumd":
        allowed_ensembles.add("nve")
    if (
        request.template_path is None
        and request.ensemble not in allowed_ensembles
    ):
        choices = ", ".join(sorted(allowed_ensembles))
        raise MdError(f"ensemble must be one of: {choices}")
    if (
        request.steps <= 0
        or request.timestep <= 0
        or request.seed < 1
        or request.replica < 1
    ):
        raise MdError("steps, timestep, seed, and replica must be positive")
    if request.mpi_ranks < 1:
        raise MdError("mpi_ranks must be at least 1")
    if request.pre_failure_frames < 0 or request.bad_tail_frames < 1:
        raise MdError(
            "pre_failure_frames must be non-negative and bad_tail_frames at least 1"
        )
    if request.spin and request.spin_temperature is None:
        raise MdError("spin MD requires spin_temperature")
    try:
        health_policy = TrajectoryHealthPolicy.from_mapping(request.health)
    except TrajectoryHealthError as error:
        raise MdError(str(error)) from error
    if backend == "gpumd":
        return _run_gpumd(request, health_policy)
    if backend != "lammps":
        raise MdError("MD backend must be gpumd or lammps")
    variables = {
        "temperature": request.temperature,
        "spin_temperature": request.spin_temperature,
        "pressure": request.pressure,
        "steps": request.steps,
        "seed": request.seed,
        "replica": request.replica,
        "route_id": request.route_id,
        "route_fingerprint": request.route_fingerprint,
    }
    try:
        result = run_lammps(
            atoms=request.atoms,
            model_file=request.model_file,
            output_dir=request.output_dir,
            output_file=request.output_file,
            template=_template(request),
            variables=variables,
            inference_backend=request.inference_backend,
            lmp_command=request.lmp_command,
            mpiexec=request.mpiexec,
            mpi_ranks=request.mpi_ranks,
            spin=request.spin,
            pre_failure_frames=request.pre_failure_frames,
            bad_tail_frames=request.bad_tail_frames,
            health_policy=health_policy,
        )
    except (LammpsError, TrajectoryHealthError) as error:
        raise MdError(str(error)) from error
    return MdResult(
        backend="lammps",
        trajectory=result.trajectory,
        run_directory=request.output_dir,
        inference_backend=result.backend,
        pair_style=result.pair_style,
        completed=result.completed,
        last_step=result.last_step,
        failure_code=result.failure_code,
        failure_reason=result.failure_reason,
        health_report=result.health_report,
    )
