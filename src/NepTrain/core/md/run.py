"""Small molecular-dynamics Interface shared by both MD Adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Mapping

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from .lammps import LammpsError, run_lammps
from .health import TrajectoryHealthError, TrajectoryHealthPolicy


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


def _run_gpumd(request: MdRequest) -> MdResult:
    if request.spin:
        raise MdError("spin evolution uses the LAMMPS DynSpin backend")
    from ..gpumd.io import RunInput

    run = RunInput(str(request.model_file))
    if request.template_path is None:
        resource = files("NepTrain.core.gpumd").joinpath("run.in")
        run.read_run(str(resource))
    else:
        run.read_run(str(request.template_path))
    duration_ps = request.steps * request.timestep
    run.set_time_temp(duration_ps, request.temperature)
    run.calculate(request.atoms, str(request.output_dir))
    dump = request.output_dir / "dump.xyz"
    if not dump.is_file():
        raise MdError("GPUMD completed without dump.xyz")
    frames = ase_read(dump, index=":", format="extxyz")
    request.output_file.parent.mkdir(parents=True, exist_ok=True)
    ase_write(
        request.output_file,
        frames,
        format="extxyz",
    )
    return MdResult("gpumd", request.output_file, request.output_dir)


def run_md(request: MdRequest, backend: str) -> MdResult:
    if request.template_path is None and request.ensemble not in {"nvt", "npt"}:
        raise MdError("ensemble must be nvt or npt")
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
        return _run_gpumd(request)
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
