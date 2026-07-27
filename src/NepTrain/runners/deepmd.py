"""Run a local DeePMD model through NepTrain's model-label protocol."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.stress import voigt_6_to_full_3x3_stress


class DeepmdRunnerError(RuntimeError):
    """Raised when DeePMD cannot produce the required training labels."""


CalculatorFactory = Callable[[Path, str, str, str | None], Calculator]


def _deepmd_calculator(
    model: Path,
    device: str,
    precision: str,
    head: str | None,
) -> Calculator:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        try:
            import torch
        except ImportError as error:
            raise DeepmdRunnerError(
                "CUDA inference for DPA models requires PyTorch"
            ) from error
        if not torch.cuda.is_available():
            raise DeepmdRunnerError(
                "device is cuda but PyTorch cannot access a CUDA device"
            )
    os.environ["DP_INTERFACE_PREC"] = (
        "low" if precision == "float32" else "high"
    )
    try:
        from deepmd.calculator import DP
    except ImportError as error:
        raise DeepmdRunnerError(
            "DeePMD-kit is not installed; install NepTrain with the "
            "'deepmd' extra"
        ) from error
    return DP(model=str(model), head=head)


def _stress_matrix(stress: object) -> np.ndarray:
    array = np.asarray(stress, dtype=np.float64)
    if array.shape == (6,):
        return voigt_6_to_full_3x3_stress(array)
    if array.shape == (3, 3):
        return array
    raise DeepmdRunnerError(
        f"DeePMD stress must have shape (6,) or (3, 3), got {array.shape}"
    )


def label_frames(
    model: str | Path,
    source: str | Path,
    output: str | Path,
    *,
    device: str,
    precision: str,
    head: str | None = None,
    calculator_factory: CalculatorFactory = _deepmd_calculator,
) -> list[Atoms]:
    """Label structures with one local model supported by DeePMD-kit."""

    model_path = Path(model).expanduser().resolve()
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not model_path.is_file():
        raise DeepmdRunnerError(f"DeePMD model does not exist: {model_path}")
    if device not in {"cpu", "cuda"}:
        raise DeepmdRunnerError("device must be cpu or cuda")
    if precision not in {"float32", "float64"}:
        raise DeepmdRunnerError("precision must be float32 or float64")
    if head is not None and not head.strip():
        raise DeepmdRunnerError("head must not be empty")

    loaded = ase_read(source_path, index=":")
    frames = loaded if isinstance(loaded, list) else [loaded]
    if not frames:
        raise DeepmdRunnerError(f"input contains no structures: {source_path}")
    if any("spin" in frame.arrays for frame in frames):
        raise DeepmdRunnerError(
            "the DeePMD runner does not emit magnetic forces for spin inputs"
        )

    calculator = calculator_factory(model_path, device, precision, head)
    labeled_frames = []
    for index, source_frame in enumerate(frames):
        frame = source_frame.copy()
        volume = float(frame.get_volume())
        if not np.isfinite(volume) or volume <= 0.0:
            raise DeepmdRunnerError(
                f"frame {index}: a positive cell volume is required"
            )
        frame.calc = calculator
        try:
            energy = float(frame.get_potential_energy())
            forces = np.asarray(frame.get_forces(), dtype=np.float64)
            stress = _stress_matrix(frame.get_stress())
        except Exception as error:
            raise DeepmdRunnerError(
                f"frame {index}: DeePMD inference failed: {error}"
            ) from error
        if forces.shape != (len(frame), 3):
            raise DeepmdRunnerError(
                f"frame {index}: DeePMD forces have shape {forces.shape}"
            )
        if (
            not np.isfinite(energy)
            or not np.isfinite(forces).all()
            or not np.isfinite(stress).all()
        ):
            raise DeepmdRunnerError(
                f"frame {index}: DeePMD returned non-finite labels"
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
