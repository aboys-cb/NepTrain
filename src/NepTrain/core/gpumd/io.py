"""GPUMD input preparation and process execution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from ase import Atoms
from ase.io import write as ase_write

from ..md.dump import adaptive_dump_interval
from ..md.template import render_template


class GpumdInputError(ValueError):
    """Raised when a GPUMD template cannot be adapted safely."""


@dataclass(frozen=True)
class GpumdProcessResult:
    returncode: int
    stdout: Path
    stderr: Path


class RunInput:
    """Render user-owned GPUMD inputs and read trajectory metadata."""

    def __init__(self, nep_txt_path: str | Path):
        self.nep_txt_path = Path(nep_txt_path).expanduser()
        self.command = os.environ.get("NEPTRAIN_GPUMD_COMMAND", "gpumd")
        self._template = ""
        self._text = ""

    def read_run(self, file_name: str | Path) -> None:
        self._template = Path(file_name).read_text(encoding="utf-8")
        self._text = self._template

    def _arguments(self, command: str) -> list[list[str]]:
        arguments = []
        for line in self._text.splitlines():
            tokens = line.split("#", 1)[0].split()
            if tokens and tokens[0] == command:
                arguments.append(tokens[1:])
        return arguments

    def use_default(
        self,
        *,
        ensemble: str,
        temperature: float,
        pressure: float,
        steps: int,
        timestep_fs: float,
        seed: int,
    ) -> None:
        if ensemble == "nve":
            ensemble_values = ["nve"]
        elif ensemble == "nvt":
            ensemble_values = ["nvt_nhc", temperature, temperature, 100]
        elif ensemble == "npt":
            ensemble_values = [
                "npt_scr",
                temperature,
                temperature,
                100,
                pressure,
                pressure,
                pressure,
                0,
                0,
                0,
                100,
                100,
                100,
                100,
                100,
                100,
                1000,
            ]
        else:
            raise GpumdInputError("GPUMD ensemble must be nve, nvt, or npt")
        dump_interval = adaptive_dump_interval(steps)
        commands = [
            ["potential", ["nep.txt"]],
            ["velocity", [temperature, "seed", seed]],
            ["ensemble", ensemble_values],
            ["time_step", [timestep_fs]],
            ["dump_thermo", [dump_interval]],
            ["dump_exyz", [dump_interval, 0, 1]],
            ["run", [steps]],
        ]
        self._text = "".join(
            f"{key} {' '.join(str(value) for value in values)}\n"
            for key, values in commands
        )

    def configure(
        self,
        *,
        temperature: float,
        pressure: float,
        steps: int,
        timestep_fs: float,
        seed: int,
        replica: int = 1,
        route_id: str = "",
        route_fingerprint: str = "",
    ) -> None:
        """Replace explicit placeholders; leave all physical commands untouched."""
        variables = {
            "temperature": temperature,
            "pressure": pressure,
            "steps": steps,
            "timestep_fs": timestep_fs,
            "timestep_ps": timestep_fs / 1000.0,
            "seed": seed,
            "replica": replica,
            "route_id": route_id,
            "route_fingerprint": route_fingerprint,
            "model_file": "nep.txt",
            "structure_file": "model.xyz",
            "trajectory_file": "dump.xyz",
            "dump_interval": adaptive_dump_interval(steps),
        }
        try:
            self._text = render_template(self._template, variables, backend="GPUMD")
        except ValueError as error:
            raise GpumdInputError(str(error)) from error

    def dump_interval(self) -> int:
        dumps = self._arguments("dump_exyz")
        if not dumps:
            raise GpumdInputError("GPUMD input must define dump_exyz for dump.xyz")
        intervals = []
        for values in dumps:
            try:
                interval = int(values[0])
            except (IndexError, ValueError) as error:
                raise GpumdInputError("dump_exyz requires an integer interval") from error
            if interval < 1:
                raise GpumdInputError("dump_exyz interval must be positive")
            if len(values) >= 5 and values[4] == "1":
                raise GpumdInputError(
                    "separated dump_exyz output is not supported by the workflow adapter"
                )
            intervals.append(interval)
        return intervals[-1]

    def timestep_fs(self) -> float:
        try:
            values = [float(args[0]) for args in self._arguments("time_step")]
        except (IndexError, ValueError) as error:
            raise GpumdInputError("GPUMD time_step must be positive") from error
        if not values or not all(value > 0 for value in values):
            raise GpumdInputError("GPUMD time_step must be positive")
        return values[-1]

    def write_run(self, file_name: str | Path) -> None:
        Path(file_name).write_text(self._text, encoding="utf-8")

    def calculate(self, atoms: Atoms, directory: str | Path) -> GpumdProcessResult:
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.write_run(directory / "run.in")
        ase_write(directory / "model.xyz", atoms, format="extxyz")
        model_source = self.nep_txt_path.resolve()
        if not model_source.is_file():
            raise GpumdInputError(f"GPUMD model does not exist: {model_source}")
        model_target = directory / "nep.txt"
        if model_source != model_target:
            shutil.copy2(model_source, model_target)

        for stale in directory.glob("dump*.xyz"):
            stale.unlink()
        stdout = directory / "gpumd.out"
        stderr = directory / "gpumd.err"
        with stdout.open("w", encoding="utf-8") as f_std, stderr.open(
            "w", encoding="utf-8", buffering=1
        ) as f_err:
            completed = subprocess.run(
                shlex.split(self.command),
                stdout=f_std,
                stderr=f_err,
                cwd=directory,
                check=False,
            )
        return GpumdProcessResult(completed.returncode, stdout, stderr)


__all__ = [
    "GpumdInputError",
    "GpumdProcessResult",
    "RunInput",
]
