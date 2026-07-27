"""Run a local MACE checkpoint through NepTrain's model-label protocol."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.stress import voigt_6_to_full_3x3_stress


class MaceRunnerError(RuntimeError):
    """Raised when MACE cannot produce the required training labels."""


CalculatorFactory = Callable[[Path, str, str], Calculator]


def _mace_calculator(model: Path, device: str, precision: str) -> Calculator:
    try:
        from mace.calculators import MACECalculator
    except ImportError as error:
        raise MaceRunnerError(
            "MACE is not installed; install NepTrain with the 'mace' extra"
        ) from error
    return MACECalculator(
        model_paths=str(model),
        device=device,
        default_dtype=precision,
    )


def _stress_matrix(stress: object) -> np.ndarray:
    array = np.asarray(stress, dtype=np.float64)
    if array.shape == (6,):
        return voigt_6_to_full_3x3_stress(array)
    if array.shape == (3, 3):
        return array
    raise MaceRunnerError(
        f"MACE stress must have shape (6,) or (3, 3), got {array.shape}"
    )


def label_frames(
    model: str | Path,
    source: str | Path,
    output: str | Path,
    *,
    device: str,
    precision: str,
    calculator_factory: CalculatorFactory = _mace_calculator,
) -> list[Atoms]:
    """Label ordinary periodic structures and write canonical extxyz."""

    model_path = Path(model).expanduser().resolve()
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not model_path.is_file():
        raise MaceRunnerError(f"MACE model does not exist: {model_path}")
    if device not in {"cpu", "cuda"}:
        raise MaceRunnerError("device must be cpu or cuda")
    if precision not in {"float32", "float64"}:
        raise MaceRunnerError("precision must be float32 or float64")

    loaded = ase_read(source_path, index=":")
    frames = loaded if isinstance(loaded, list) else [loaded]
    if not frames:
        raise MaceRunnerError(f"input contains no structures: {source_path}")
    if any("spin" in frame.arrays for frame in frames):
        raise MaceRunnerError(
            "the MACE runner does not emit magnetic forces for spin inputs"
        )

    calculator = calculator_factory(model_path, device, precision)
    labeled_frames = []
    for index, source_frame in enumerate(frames):
        frame = source_frame.copy()
        volume = float(frame.get_volume())
        if not np.isfinite(volume) or volume <= 0.0:
            raise MaceRunnerError(
                f"frame {index}: a positive periodic cell volume is required"
            )
        frame.calc = calculator
        try:
            energy = float(frame.get_potential_energy())
            forces = np.asarray(frame.get_forces(), dtype=np.float64)
            stress = _stress_matrix(frame.get_stress())
        except Exception as error:
            raise MaceRunnerError(
                f"frame {index}: MACE inference failed: {error}"
            ) from error
        if forces.shape != (len(frame), 3):
            raise MaceRunnerError(
                f"frame {index}: MACE forces have shape {forces.shape}"
            )
        if (
            not np.isfinite(energy)
            or not np.isfinite(forces).all()
            or not np.isfinite(stress).all()
        ):
            raise MaceRunnerError(
                f"frame {index}: MACE returned non-finite labels"
            )
        frame.calc = SinglePointCalculator(
            frame,
            energy=energy,
            forces=forces,
        )
        frame.info["virial"] = -stress * volume
        labeled_frames.append(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(output_path, labeled_frames, format="extxyz")
    return labeled_frames


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label extxyz structures with a local MACE checkpoint."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument(
        "--precision",
        choices=["float32", "float64"],
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        label_frames(
            args.model,
            args.input,
            args.output,
            device=args.device,
            precision=args.precision,
        )
    except (MaceRunnerError, OSError, ValueError) as error:
        parser.exit(1, f"neptrain-label-mace: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
