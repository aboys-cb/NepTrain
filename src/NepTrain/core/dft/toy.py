"""Deterministic analytic teacher used only for workflow development smoke."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read as ase_read
from ase.io import write as ase_write

from ..spin import validate_spin_dataset, validate_spin_structure

if TYPE_CHECKING:
    from ..labeling import LabelRequest


@dataclass(frozen=True)
class ToyTeacherParameters:
    pair_depth: float = 0.35
    pair_width: float = 1.4
    equilibrium_distance: float = 2.45
    exchange: float = 0.08
    exchange_decay: float = 0.8
    longitudinal: float = 0.025
    target_magnitude: float = 1.7
    anisotropy: float = 0.01
    field_z: float = 0.015


class ToyTeacher:
    """Small spin-lattice oracle with exact energy derivatives."""

    def __init__(self, profile: str = "ordinary", parameters: ToyTeacherParameters | None = None):
        if profile not in {"ordinary", "spin"}:
            raise ValueError("toy teacher profile must be ordinary or spin")
        self.profile = profile
        self.parameters = parameters or ToyTeacherParameters()

    def calculate(self, atoms: Atoms) -> tuple[float, np.ndarray, np.ndarray, np.ndarray | None]:
        spin_frame = validate_spin_structure(atoms, require_mforce=False)
        if self.profile == "spin" and not spin_frame:
            raise ValueError("spin toy teacher requires spin:R:3")
        if self.profile == "ordinary" and spin_frame:
            raise ValueError("ordinary toy teacher does not accept spin frames")

        p = self.parameters
        energy = 0.0
        forces = np.zeros((len(atoms), 3), dtype=np.float64)
        virial = np.zeros((3, 3), dtype=np.float64)
        spins = np.asarray(atoms.arrays["spin"], dtype=np.float64) if spin_frame else None
        mforce = np.zeros_like(spins) if spin_frame else None

        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                vector = np.asarray(atoms.get_distance(i, j, mic=True, vector=True), dtype=np.float64)
                distance = float(np.linalg.norm(vector))
                if distance <= 1.0e-12:
                    raise ValueError("toy teacher does not accept overlapping atoms")
                unit = vector / distance
                x = distance - p.equilibrium_distance
                exp1 = np.exp(-p.pair_width * x)
                exp2 = exp1 * exp1
                energy += p.pair_depth * (exp2 - 2.0 * exp1)
                derivative = 2.0 * p.pair_depth * p.pair_width * (exp1 - exp2)

                if spins is not None and mforce is not None:
                    exchange = p.exchange * np.exp(-p.exchange_decay * x)
                    spin_dot = float(np.dot(spins[i], spins[j]))
                    energy -= exchange * spin_dot
                    derivative += p.exchange_decay * exchange * spin_dot
                    mforce[i] += exchange * spins[j]
                    mforce[j] += exchange * spins[i]

                pair_force = derivative * unit
                forces[i] += pair_force
                forces[j] -= pair_force
                virial -= np.outer(vector, pair_force)

        if spins is not None and mforce is not None:
            magnitude2 = np.einsum("ij,ij->i", spins, spins)
            delta = magnitude2 - p.target_magnitude**2
            energy += float(p.longitudinal * np.dot(delta, delta))
            energy += float(p.anisotropy * np.dot(spins[:, 2], spins[:, 2]))
            energy -= float(p.field_z * spins[:, 2].sum())
            mforce -= 4.0 * p.longitudinal * delta[:, None] * spins
            mforce[:, 2] -= 2.0 * p.anisotropy * spins[:, 2]
            mforce[:, 2] += p.field_z

        return energy, forces, virial, mforce

    def label(self, atoms: Atoms) -> Atoms:
        labeled = atoms.copy()
        labeled.arrays.pop("mforce", None)
        labeled.arrays.pop("mforces", None)
        energy, forces, virial, mforce = self.calculate(labeled)
        labeled.calc = SinglePointCalculator(labeled, energy=energy, forces=forces)
        labeled.info["virial"] = virial
        labeled.info["Config_type"] = f"NepTrain toy-{self.profile}"
        labeled.info["Weight"] = 1.0
        labeled.info["toy_teacher"] = self.profile
        if mforce is not None:
            labeled.set_array("mforce", mforce)
        return labeled


def _read_frames(source: Path) -> list[Atoms]:
    paths = (
        sorted(path for pattern in ("*.xyz", "*.vasp", "POSCAR*") for path in source.glob(pattern))
        if source.is_dir()
        else [source]
    )
    frames: list[Atoms] = []
    for path in paths:
        loaded = ase_read(path, index=":", format=None)
        frames.extend(loaded if isinstance(loaded, list) else [loaded])
    return frames


def run_toy_teacher(request: "LabelRequest") -> list[Atoms]:
    profile = str(request.settings.get("profile", "ordinary"))
    teacher = ToyTeacher(profile)
    frames = _read_frames(request.source)
    if not frames:
        raise ValueError(f"no readable structures found in {request.source}")
    labeled = [teacher.label(frame) for frame in frames]
    validate_spin_dataset(labeled, require_mforce=True)
    request.work_dir.mkdir(parents=True, exist_ok=True)
    request.output_file.parent.mkdir(parents=True, exist_ok=True)
    ase_write(request.output_file, labeled, format="extxyz", append=request.append)
    return labeled


__all__ = ["ToyTeacher", "ToyTeacherParameters", "run_toy_teacher"]
