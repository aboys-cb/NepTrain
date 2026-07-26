"""Versioned scientific-data identity and label validation.

This module is the authority for deciding whether two input structures are the
same and whether a DFT result is safe to publish.  Execution metadata must not
invent weaker, backend-specific versions of these checks.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
from ase import Atoms

from .spin import SpinDataError, validate_spin_structure


STRUCTURE_ID_VERSION = "neptrain.structure-id.v2"
GEOMETRY_ID_VERSION = "neptrain.geometry-id.v1"
INPUT_STRUCTURE_ID_KEY = "neptrain_input_structure_id"


class ScientificDataError(ValueError):
    """Raised when scientific input or output violates the data contract."""


def _canonical_float64(value: object) -> bytes:
    array = np.asarray(value, dtype="<f8")
    return array.tobytes(order="C")


def structure_id(atoms: Atoms) -> str:
    """Return a stable identity for the physical structure sent to a model.

    Labels and incidental ``Atoms.info`` metadata are intentionally excluded.
    Boundary conditions are part of the physical structure and therefore part
    of the identity.
    """

    digest = hashlib.sha256()
    digest.update(STRUCTURE_ID_VERSION.encode("ascii"))
    digest.update(np.asarray(atoms.numbers, dtype="<i8").tobytes(order="C"))
    digest.update(_canonical_float64(atoms.cell))
    digest.update(_canonical_float64(atoms.positions))
    digest.update(np.asarray(atoms.pbc, dtype=np.uint8).tobytes(order="C"))
    if "spin" in atoms.arrays:
        digest.update(b"spin")
        digest.update(_canonical_float64(atoms.arrays["spin"]))
    else:
        digest.update(b"ordinary")
        initial_magmoms = atoms.arrays.get("initial_magmoms")
        if initial_magmoms is not None and np.any(
            np.asarray(initial_magmoms, dtype=np.float64) != 0.0
        ):
            digest.update(b"collinear-initial-magmoms")
            digest.update(_canonical_float64(initial_magmoms))
    return digest.hexdigest()


def geometry_id(atoms: Atoms) -> str:
    """Identify atom order and geometry while excluding electronic state."""

    digest = hashlib.sha256()
    digest.update(GEOMETRY_ID_VERSION.encode("ascii"))
    digest.update(np.asarray(atoms.numbers, dtype="<i8").tobytes(order="C"))
    digest.update(_canonical_float64(atoms.cell))
    digest.update(_canonical_float64(atoms.positions))
    digest.update(np.asarray(atoms.pbc, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def bind_labeled_frames_to_inputs(
    inputs: Iterable[Atoms],
    outputs: Iterable[Atoms],
) -> list[str]:
    """Bind output labels to exact input states without equating final spin."""

    input_frames = list(inputs)
    output_frames = list(outputs)
    if len(input_frames) != len(output_frames):
        raise ScientificDataError(
            f"DFT returned {len(output_frames)} frames for "
            f"{len(input_frames)} inputs"
        )
    identifiers = []
    for index, (source, result) in enumerate(zip(input_frames, output_frames)):
        if geometry_id(source) != geometry_id(result):
            raise ScientificDataError(
                f"frame {index}: DFT changed atom order, geometry, cell, or PBC"
            )
        identifier = structure_id(source)
        result.info[INPUT_STRUCTURE_ID_KEY] = identifier
        identifiers.append(identifier)
    return identifiers


def labeled_input_structure_ids(frames: Iterable[Atoms]) -> list[str]:
    """Read and validate the persisted input-state ownership of labels."""

    identifiers = []
    for index, frame in enumerate(frames):
        value = frame.info.get(INPUT_STRUCTURE_ID_KEY)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ScientificDataError(
                f"frame {index}: labeled result is missing a valid "
                f"{INPUT_STRUCTURE_ID_KEY}"
            )
        identifiers.append(value)
    return identifiers


def reference_energy(atoms: Atoms) -> float:
    """Read one energy label from ASE or the extxyz info convention."""

    values = []
    if atoms.calc is not None and "energy" in atoms.calc.results:
        values.append(float(atoms.calc.results["energy"]))
    if "energy" in atoms.info:
        values.append(float(atoms.info["energy"]))
    if not values:
        raise ScientificDataError("energy is missing")
    if any(value != values[0] for value in values[1:]):
        raise ScientificDataError("conflicting energy labels are present")
    return values[0]


def reference_forces(atoms: Atoms) -> np.ndarray:
    """Read force labels without rewriting GPUMD's ``force:R:3`` schema."""

    values = []
    if atoms.calc is not None and "forces" in atoms.calc.results:
        values.append(("calculator forces", np.asarray(atoms.calc.results["forces"])))
    for key in ("force", "forces"):
        if key in atoms.arrays:
            values.append((key, np.asarray(atoms.arrays[key])))
    if not values:
        raise ScientificDataError("forces are missing")
    reference = np.asarray(values[0][1], dtype=np.float64)
    for name, value in values[1:]:
        candidate = np.asarray(value, dtype=np.float64)
        if reference.shape != candidate.shape or not np.array_equal(
            reference,
            candidate,
        ):
            raise ScientificDataError(
                f"conflicting force labels are present ({values[0][0]} and {name})"
            )
    return reference


def validate_labeled_frame(atoms: Atoms, *, index: int | None = None) -> None:
    """Validate labels required by training and deterministic merging."""

    prefix = f"frame {index}: " if index is not None else ""
    try:
        energy = reference_energy(atoms)
        forces = reference_forces(atoms)
    except ScientificDataError as error:
        raise ScientificDataError(f"{prefix}{error}") from error
    except (TypeError, ValueError) as error:
        raise ScientificDataError(f"{prefix}energy or forces are missing") from error
    if not np.isfinite(energy):
        raise ScientificDataError(f"{prefix}energy is not finite")
    if forces.shape != (len(atoms), 3):
        raise ScientificDataError(
            f"{prefix}forces must have shape (natoms, 3), got {forces.shape}"
        )
    if not np.isfinite(forces).all():
        raise ScientificDataError(f"{prefix}forces contain non-finite values")

    if "virial" not in atoms.info:
        raise ScientificDataError(f"{prefix}virial is missing")
    virial = np.asarray(atoms.info["virial"], dtype=np.float64)
    if virial.shape != (3, 3):
        raise ScientificDataError(
            f"{prefix}virial must have shape (3, 3), got {virial.shape}"
        )
    if not np.isfinite(virial).all():
        raise ScientificDataError(f"{prefix}virial contains non-finite values")

    try:
        validate_spin_structure(atoms, require_mforce=True)
    except SpinDataError as error:
        raise ScientificDataError(f"{prefix}{error}") from error


def validate_labeled_frames(frames: Iterable[Atoms]) -> list[str]:
    """Validate frames and return their structure identities in order."""

    identifiers: list[str] = []
    for index, atoms in enumerate(frames):
        validate_labeled_frame(atoms, index=index)
        identifiers.append(structure_id(atoms))
    if not identifiers:
        raise ScientificDataError("labeled result contains no frames")
    return identifiers


__all__ = [
    "GEOMETRY_ID_VERSION",
    "INPUT_STRUCTURE_ID_KEY",
    "STRUCTURE_ID_VERSION",
    "ScientificDataError",
    "bind_labeled_frames_to_inputs",
    "geometry_id",
    "labeled_input_structure_ids",
    "reference_energy",
    "reference_forces",
    "structure_id",
    "validate_labeled_frame",
    "validate_labeled_frames",
]
