"""GPUMD input preparation and process execution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

from ase import Atoms
from ase.io import write as ase_write


class GpumdInputError(ValueError):
    """Raised when a GPUMD template cannot be adapted safely."""


@dataclass(frozen=True)
class GpumdProcessResult:
    returncode: int
    stdout: Path
    stderr: Path


class RunInput:
    """A narrow adapter for the subset of ``run.in`` owned by NepTrain."""

    _NVE_ENSEMBLES = {"nve"}
    _NVT_ENSEMBLES = {"nvt_ber", "nvt_nhc", "nvt_bdp", "nvt_lan", "nvt_bao"}
    _NPT_ENSEMBLES = {"npt_ber", "npt_scr"}

    def __init__(self, nep_txt_path: str | Path):
        self.nep_txt_path = Path(nep_txt_path).expanduser()
        self.command = os.environ.get("NEPTRAIN_GPUMD_COMMAND", "gpumd")
        self.run_in: list[list[object]] = []

    def read_run(self, file_name: str | Path) -> None:
        text = Path(file_name).read_text(encoding="utf-8")
        self.run_in.clear()
        for key, value in re.findall(
            r"^\s*([A-Za-z_]+)\s+([^#\n]*?)(?:\s*#.*)?$",
            text,
            re.MULTILINE,
        ):
            values = value.split()
            if values:
                self.run_in.append([key, values])
        if not self.run_in:
            raise GpumdInputError("GPUMD template contains no input commands")

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
        dump_interval = self._default_dump_interval(steps)
        self.run_in = [
            ["potential", ["nep.txt"]],
            ["velocity", [temperature, "seed", seed]],
            ["ensemble", ensemble_values],
            ["time_step", [timestep_fs]],
            ["dump_thermo", [dump_interval]],
            ["dump_exyz", [dump_interval, 0, 1]],
            ["run", [steps]],
        ]

    def configure(
        self,
        *,
        temperature: float,
        pressure: float,
        steps: int,
        timestep_fs: float,
        seed: int,
    ) -> None:
        """Apply route conditions without guessing unsupported ensemble syntax."""

        ensembles = [
            values
            for key, values in self.run_in
            if key == "ensemble"
        ]
        if not ensembles:
            raise GpumdInputError("GPUMD template must define an ensemble")
        for values in ensembles:
            method = str(values[0])
            supported = (
                self._NVE_ENSEMBLES
                | self._NVT_ENSEMBLES
                | self._NPT_ENSEMBLES
            )
            if method not in supported:
                raise GpumdInputError(
                    f"cannot safely adapt unsupported GPUMD ensemble {method!r}"
                )
            if method in self._NVE_ENSEMBLES:
                if len(values) != 1:
                    raise GpumdInputError(
                        "GPUMD ensemble 'nve' does not accept parameters"
                    )
                continue
            if len(values) < 4:
                raise GpumdInputError(
                    f"GPUMD ensemble {method!r} is missing temperature parameters"
                )
            values[1:3] = [temperature, temperature]
            if method in self._NPT_ENSEMBLES:
                self._set_pressure(values, pressure)

        run_commands = 0
        dump_commands = 0
        first_run = len(self.run_in)
        for index, (key, values) in enumerate(self.run_in):
            if key == "potential":
                values[:] = ["nep.txt"]
            elif key == "velocity":
                values[:] = [temperature, "seed", seed]
            elif key == "run":
                values[:] = [steps]
                run_commands += 1
                first_run = min(first_run, index)
            elif key == "dump_exyz":
                self._configure_dump(values, steps)
                dump_commands += 1
        if run_commands == 0:
            raise GpumdInputError("GPUMD template must contain at least one run command")
        if not any(key == "potential" for key, _ in self.run_in):
            self.run_in.insert(0, ["potential", ["nep.txt"]])
            first_run += 1
        if not any(key == "velocity" for key, _ in self.run_in):
            velocity_index = (
                1 if self.run_in and self.run_in[0][0] == "potential" else 0
            )
            self.run_in.insert(
                velocity_index,
                ["velocity", [temperature, "seed", seed]],
            )
            first_run += 1
        if not any(key == "time_step" for key, _ in self.run_in):
            self.run_in.insert(first_run, ["time_step", [timestep_fs]])
            first_run += 1
        if dump_commands == 0:
            interval = self._default_dump_interval(steps)
            self.run_in.insert(first_run, ["dump_exyz", [interval, 0, 1]])

    @staticmethod
    def _default_dump_interval(steps: int) -> int:
        return max(1, min(1000, steps // 100))

    @staticmethod
    def _set_pressure(values: list[object], pressure: float) -> None:
        pressure_count = len(values) - 4
        if pressure_count == 3:
            values[4] = pressure
        elif pressure_count == 7:
            values[4:7] = [pressure, pressure, pressure]
        elif pressure_count == 13:
            values[4:10] = [pressure, pressure, pressure, 0, 0, 0]
        else:
            raise GpumdInputError(
                "GPUMD npt_ber/npt_scr pressure controls must use the "
                "isotropic, orthorhombic, or triclinic form"
            )

    @staticmethod
    def _configure_dump(values: list[object], steps: int) -> None:
        if not values:
            raise GpumdInputError("dump_exyz requires an interval")
        try:
            interval = int(str(values[0]))
        except ValueError as error:
            raise GpumdInputError("dump_exyz interval must be an integer") from error
        if interval < 1:
            raise GpumdInputError("dump_exyz interval must be positive")
        values[0] = min(interval, steps)
        while len(values) < 3:
            values.append(0)
        values[2] = 1
        if len(values) >= 5 and str(values[4]) == "1":
            raise GpumdInputError(
                "separated dump_exyz output is not supported by the workflow adapter"
            )

    def dump_interval(self) -> int:
        intervals = [
            int(str(values[0]))
            for key, values in self.run_in
            if key == "dump_exyz"
        ]
        if not intervals:
            raise GpumdInputError("GPUMD input has no dump_exyz command")
        return intervals[-1]

    def timestep_fs(self) -> float:
        values = [
            float(str(arguments[0]))
            for key, arguments in self.run_in
            if key == "time_step" and arguments
        ]
        if not values or not all(value > 0 for value in values):
            raise GpumdInputError("GPUMD time_step must be positive")
        return values[-1]

    def write_run(self, file_name: str | Path) -> None:
        lines = [
            f"{key} {' '.join(str(value) for value in values)}"
            for key, values in self.run_in
        ]
        Path(file_name).write_text("\n".join(lines) + "\n", encoding="utf-8")

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
