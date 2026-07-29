"""Run a local TACE model through NepTrain's model-label protocol."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.stress import voigt_6_to_full_3x3_stress


class TaceRunnerError(RuntimeError):
    """Raised when TACE cannot produce the required training labels."""


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _matrix(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (6,):
        return voigt_6_to_full_3x3_stress(array)
    if array.shape == (9,):
        return array.reshape(3, 3)
    if array.shape == (3, 3):
        return array
    raise TaceRunnerError(
        f"TACE {name} must have shape (6,), (9,), or (3, 3), "
        f"got {array.shape}"
    )


def _first(mapping: dict, names: tuple[str, ...]) -> object | None:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def label_frames(
    model: str | Path,
    source: str | Path,
    output: str | Path,
    *,
    device: str,
    precision: str,
    fidelity_index: int | None = None,
    command_runner: CommandRunner = subprocess.run,
) -> list[Atoms]:
    """Label structures with ``tace-eval`` and write canonical extxyz."""

    model_path = Path(model).expanduser().resolve()
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not model_path.is_file():
        raise TaceRunnerError(f"TACE model does not exist: {model_path}")
    if device not in {"cpu", "cuda"}:
        raise TaceRunnerError("device must be cpu or cuda")
    if precision not in {"float32", "float64"}:
        raise TaceRunnerError("precision must be float32 or float64")
    if fidelity_index is not None and fidelity_index < 0:
        raise TaceRunnerError("fidelity index must be non-negative")

    loaded = ase_read(source_path, index=":")
    frames = loaded if isinstance(loaded, list) else [loaded]
    if not frames:
        raise TaceRunnerError(f"input contains no structures: {source_path}")
    spin_input = any("spin" in frame.arrays for frame in frames)
    if spin_input and not all("spin" in frame.arrays for frame in frames):
        raise TaceRunnerError(
            "ordinary and spin frames cannot be mixed in one TACE input"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".neptrain-tace-",
        dir=output_path.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        tace_input = temporary_root / "input.xyz"
        tace_output = temporary_root / "prediction.xyz"
        prepared = []
        for source_frame in frames:
            frame = source_frame.copy()
            frame.calc = None
            if fidelity_index is not None:
                frame.info["fidelity_idx"] = fidelity_index
            prepared.append(frame)
        ase_write(tace_input, prepared, format="extxyz")

        command = [
            "tace-eval",
            "--input",
            str(tace_input),
            "--model",
            str(model_path),
            "--output",
            str(tace_output),
            "--device",
            device,
            "--dtype",
            precision,
        ]
        if spin_input:
            command.extend(
                ["--initial_noncollinear_magmoms_key", "spin"]
            )
        try:
            completed = command_runner(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as error:
            raise TaceRunnerError(
                "tace-eval is not installed; install TACE from "
                "https://github.com/xvzemin/tace"
            ) from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise TaceRunnerError(
                f"tace-eval failed: {detail or 'no diagnostic output'}"
            )
        if not tace_output.is_file():
            raise TaceRunnerError(
                "tace-eval completed without writing its prediction file"
            )
        predicted = ase_read(tace_output, index=":")
        predictions = (
            predicted if isinstance(predicted, list) else [predicted]
        )

    if len(predictions) != len(frames):
        raise TaceRunnerError(
            "tace-eval changed the number of structures: "
            f"expected {len(frames)}, got {len(predictions)}"
        )

    labeled_frames = []
    for index, (source_frame, prediction) in enumerate(
        zip(frames, predictions, strict=True)
    ):
        energy_value = _first(prediction.info, ("TACE_energy",))
        forces_value = _first(
            prediction.arrays,
            ("TACE_forces", "TACE_direct_forces"),
        )
        if energy_value is None:
            raise TaceRunnerError(
                f"frame {index}: TACE model did not return energy"
            )
        if forces_value is None:
            raise TaceRunnerError(
                f"frame {index}: TACE model did not return forces"
            )
        energy = float(energy_value)
        forces = np.asarray(forces_value, dtype=np.float64)
        if forces.shape != (len(source_frame), 3):
            raise TaceRunnerError(
                f"frame {index}: TACE forces have shape {forces.shape}"
            )

        virial_value = _first(
            prediction.info,
            ("TACE_virials", "TACE_direct_virials"),
        )
        if virial_value is not None:
            virial = _matrix(virial_value, name="virial")
        else:
            stress_value = _first(
                prediction.info,
                ("TACE_stress", "TACE_direct_stress"),
            )
            if stress_value is None:
                raise TaceRunnerError(
                    f"frame {index}: TACE model returned neither virial "
                    "nor stress"
                )
            volume = float(source_frame.get_volume())
            if not np.isfinite(volume) or volume <= 0.0:
                raise TaceRunnerError(
                    f"frame {index}: positive cell volume is required "
                    "to convert TACE stress to virial"
                )
            virial = -_matrix(stress_value, name="stress") * volume

        mforce = None
        if spin_input:
            mforce_value = _first(
                prediction.arrays,
                ("TACE_noncollinear_magnetic_forces",),
            )
            if mforce_value is None:
                raise TaceRunnerError(
                    f"frame {index}: spin input requires TACE "
                    "noncollinear magnetic forces"
                )
            mforce = np.asarray(mforce_value, dtype=np.float64)
            if mforce.shape != (len(source_frame), 3):
                raise TaceRunnerError(
                    f"frame {index}: TACE magnetic forces have shape "
                    f"{mforce.shape}"
                )

        if (
            not np.isfinite(energy)
            or not np.isfinite(forces).all()
            or not np.isfinite(virial).all()
            or (mforce is not None and not np.isfinite(mforce).all())
        ):
            raise TaceRunnerError(
                f"frame {index}: TACE returned non-finite labels"
            )

        frame = source_frame.copy()
        frame.calc = SinglePointCalculator(
            frame,
            energy=energy,
            forces=forces,
        )
        frame.info["virial"] = virial
        if mforce is not None:
            frame.arrays["mforce"] = mforce
        labeled_frames.append(frame)

    ase_write(output_path, labeled_frames, format="extxyz")
    return labeled_frames
