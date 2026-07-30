"""Structure-level feature construction from per-atom descriptors."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


GLOBAL_MEAN = "global_mean"
ELEMENTWISE_MEAN_STD = "elementwise_mean_std"
DESCRIPTOR_REDUCTIONS = frozenset({GLOBAL_MEAN, ELEMENTWISE_MEAN_STD})


def descriptor_elements(structures: Sequence[Any]) -> tuple[str, ...]:
    """Return the stable element order shared by a descriptor collection."""

    return tuple(
        sorted(
            {
                str(symbol)
                for structure in structures
                for symbol in structure.get_chemical_symbols()
            }
        )
    )


def reduce_atomic_descriptors(
    structures: Sequence[Any],
    atomic_descriptors: np.ndarray,
    *,
    reduction: str,
    elements: Sequence[str] | None = None,
) -> np.ndarray:
    """Reduce flat per-atom rows to one feature row per structure.

    ``global_mean`` preserves the historical all-atom arithmetic mean.
    ``elementwise_mean_std`` keeps each element in its own mean and standard
    deviation channels. Missing elements contribute zero channels, which keeps
    the feature width stable across exact-element-set FPS groups.
    """

    if reduction not in DESCRIPTOR_REDUCTIONS:
        raise ValueError(
            "descriptor reduction must be global_mean or elementwise_mean_std"
        )
    values = np.asarray(atomic_descriptors, dtype=np.float64)
    expected_atoms = sum(len(structure) for structure in structures)
    if values.ndim != 2 or len(values) != expected_atoms:
        raise ValueError(
            "atomic descriptors must be a two-dimensional array with one row "
            "per atom"
        )
    if values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("atomic descriptors must contain finite feature rows")
    if not structures:
        return np.empty((0, values.shape[1]), dtype=np.float64)

    element_order = tuple(elements or descriptor_elements(structures))
    if not element_order:
        raise ValueError("descriptor element order cannot be empty")
    unknown = sorted(
        {
            str(symbol)
            for structure in structures
            for symbol in structure.get_chemical_symbols()
        }
        - set(element_order)
    )
    if unknown:
        raise ValueError(
            "descriptor element order is missing elements: " + ", ".join(unknown)
        )

    rows: list[np.ndarray] = []
    offset = 0
    for structure in structures:
        stop = offset + len(structure)
        block = values[offset:stop]
        offset = stop
        if reduction == GLOBAL_MEAN:
            rows.append(block.mean(axis=0))
            continue

        symbols = np.asarray(structure.get_chemical_symbols())
        channels: list[np.ndarray] = []
        for element in element_order:
            selected = block[symbols == element]
            if len(selected):
                channels.extend((selected.mean(axis=0), selected.std(axis=0)))
            else:
                zeros = np.zeros(values.shape[1], dtype=np.float64)
                channels.extend((zeros, zeros.copy()))
        rows.append(np.concatenate(channels))
    return np.vstack(rows)


__all__ = [
    "DESCRIPTOR_REDUCTIONS",
    "ELEMENTWISE_MEAN_STD",
    "GLOBAL_MEAN",
    "descriptor_elements",
    "reduce_atomic_descriptors",
]
