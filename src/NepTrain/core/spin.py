"""Canonical spin and magnetic-force data contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from ase import Atoms


SPIN_KEY = "spin"
MFORCE_KEY = "mforce"
_SPIN_ALIASES = ("spins",)
_MFORCE_ALIASES = (
    "mforces",
    "force_mag",
    "forces_mag",
    "magnetic_force",
    "magnetic_forces",
)


class SpinDataError(ValueError):
    """Raised when a structure violates the spin data contract."""


def _vector_array(atoms: Atoms, key: str) -> np.ndarray | None:
    if key not in atoms.arrays:
        return None
    array = np.asarray(atoms.arrays[key], dtype=np.float64)
    if array.shape != (len(atoms), 3):
        raise SpinDataError(f"{key} must have shape (natoms, 3)")
    if not np.isfinite(array).all():
        raise SpinDataError(f"{key} contains non-finite values")
    return array


def canonicalize_spin_arrays(atoms: Atoms) -> Atoms:
    """Migrate accepted legacy aliases to singular canonical array names."""

    for canonical, aliases in (
        (SPIN_KEY, _SPIN_ALIASES),
        (MFORCE_KEY, _MFORCE_ALIASES),
    ):
        names = [name for name in (canonical, *aliases) if name in atoms.arrays]
        if not names:
            continue
        source = names[0]
        reference = np.asarray(atoms.arrays[source])
        for name in names[1:]:
            if not np.array_equal(reference, atoms.arrays[name]):
                raise SpinDataError(
                    f"ambiguous {canonical}: {source} and {name} differ"
                )
        if canonical not in atoms.arrays:
            atoms.set_array(canonical, reference.copy())
        for alias in aliases:
            atoms.arrays.pop(alias, None)
    return atoms


def validate_spin_structure(atoms: Atoms, *, require_mforce: bool) -> bool:
    """Validate one frame and return whether it is a spin frame."""

    canonicalize_spin_arrays(atoms)
    spin = _vector_array(atoms, SPIN_KEY)
    mforce = _vector_array(atoms, MFORCE_KEY)
    if spin is None:
        if mforce is not None:
            raise SpinDataError("mforce is present but spin is missing")
        return False
    if require_mforce and mforce is None:
        raise SpinDataError("spin frame is missing mandatory mforce labels")
    return True


def prepare_spin_for_dft(atoms: Atoms) -> bool:
    """Expose canonical vector spin through ASE's DFT input convention."""

    is_spin = validate_spin_structure(atoms, require_mforce=False)
    if is_spin:
        # A recalculation must not pass validation with a stale reference label.
        atoms.arrays.pop(MFORCE_KEY, None)
        atoms.arrays.pop("mforces", None)
        atoms.set_initial_magnetic_moments(np.asarray(atoms.arrays[SPIN_KEY]).copy())
    return is_spin


def collect_mforce_from_results(atoms: Atoms, results: dict) -> None:
    """Copy a calculator magnetic-force result into canonical extxyz storage."""

    for key in (MFORCE_KEY, *_MFORCE_ALIASES):
        if key in results:
            atoms.set_array(MFORCE_KEY, np.asarray(results[key], dtype=np.float64))
            return


def validate_spin_dataset(
    frames: Iterable[Atoms], *, require_mforce: bool
) -> tuple[int, int]:
    total = 0
    spin_frames = 0
    mode: bool | None = None
    for index, atoms in enumerate(frames):
        total += 1
        try:
            is_spin = validate_spin_structure(atoms, require_mforce=require_mforce)
        except SpinDataError as error:
            raise SpinDataError(f"frame {index}: {error}") from error
        if mode is None:
            mode = is_spin
        elif mode != is_spin:
            raise SpinDataError("ordinary and spin frames cannot be mixed in one dataset")
        spin_frames += int(is_spin)
    return total, spin_frames


@dataclass(frozen=True)
class LammpsSpin:
    direction: np.ndarray
    magnitude: np.ndarray


def spin_to_lammps(spin: np.ndarray) -> LammpsSpin:
    vector = np.asarray(spin, dtype=np.float64)
    if vector.ndim != 2 or vector.shape[1] != 3:
        raise SpinDataError("spin must have shape (natoms, 3)")
    magnitude = np.linalg.norm(vector, axis=1)
    if np.any(magnitude <= 0.0):
        raise SpinDataError("LAMMPS DynSpin does not accept zero-length spins")
    return LammpsSpin(direction=vector / magnitude[:, None], magnitude=magnitude)


def spin_from_lammps(direction: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    direction_array = np.asarray(direction, dtype=np.float64)
    magnitude_array = np.asarray(magnitude, dtype=np.float64)
    if direction_array.ndim != 2 or direction_array.shape[1] != 3:
        raise SpinDataError("LAMMPS spin direction must have shape (natoms, 3)")
    if magnitude_array.shape != (len(direction_array),):
        raise SpinDataError("LAMMPS spin magnitude must have shape (natoms,)")
    return direction_array * magnitude_array[:, None]
