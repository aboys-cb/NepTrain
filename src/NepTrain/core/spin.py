"""Canonical spin and magnetic-force data contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write


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
    """Explicitly migrate legacy aliases to singular canonical array names.

    Normal validation deliberately does not call this function.  A migration
    must be written back to disk before a trainer or DFT task can consume it;
    otherwise the validated in-memory object and the file actually submitted
    can disagree.
    """

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


def _reject_legacy_aliases(atoms: Atoms) -> None:
    aliases = [
        name
        for name in (*_SPIN_ALIASES, *_MFORCE_ALIASES)
        if name in atoms.arrays
    ]
    if aliases:
        names = ", ".join(sorted(aliases))
        raise SpinDataError(
            f"legacy spin field(s) {names} require explicit migration to "
            "the canonical spin/mforce fields"
        )


def validate_spin_structure(atoms: Atoms, *, require_mforce: bool) -> bool:
    """Validate one frame and return whether it is a spin frame."""

    _reject_legacy_aliases(atoms)
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


def migrate_spin_dataset(
    source: str | Path,
    output: str | Path,
    *,
    force: bool = False,
) -> dict:
    """Atomically rewrite legacy spin aliases to the canonical disk contract."""

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_path.is_file():
        raise SpinDataError(f"spin migration input does not exist: {source_path}")
    if output_path.exists() and not force:
        raise SpinDataError(
            f"spin migration output exists: {output_path}; use --force to replace it"
        )
    loaded = ase_read(source_path, index=":")
    frames = loaded if isinstance(loaded, list) else [loaded]
    migrated_fields = 0
    for index, frame in enumerate(frames):
        before = set(frame.arrays)
        try:
            canonicalize_spin_arrays(frame)
        except SpinDataError as error:
            raise SpinDataError(f"frame {index}: {error}") from error
        after = set(frame.arrays)
        migrated_fields += len(before - after)
    total, spin_frames = validate_spin_dataset(frames, require_mforce=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        ase_write(temporary, frames, format="extxyz")
        restored = ase_read(temporary, index=":", format="extxyz")
        restored_frames = restored if isinstance(restored, list) else [restored]
        restored_total, restored_spin = validate_spin_dataset(
            restored_frames,
            require_mforce=False,
        )
        if (restored_total, restored_spin) != (total, spin_frames):
            raise SpinDataError(
                "spin migration roundtrip changed frame or spin-frame counts"
            )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "protocol": "neptrain.spin-migration.v1",
        "input": str(source_path),
        "input_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "output": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "frames": total,
        "spin_frames": spin_frames,
        "legacy_fields_removed": migrated_fields,
    }


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
