"""Shared deterministic structures and features for workflow-development tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from ase import Atoms


def toy_base_frame(spin: bool) -> Atoms:
    atoms = Atoms(
        "Fe4",
        positions=[
            [1.2, 1.2, 1.2],
            [3.65, 1.2, 1.2],
            [1.2, 3.65, 1.2],
            [1.2, 1.2, 3.65],
        ],
        cell=[7.5, 7.5, 7.5],
        pbc=True,
    )
    if spin:
        atoms.set_array(
            "spin",
            np.asarray(
                [
                    [1.7, 0.0, 0.0],
                    [0.1, 1.65, 0.0],
                    [-1.55, 0.2, 0.1],
                    [0.0, -1.6, 0.3],
                ]
            ),
        )
    return atoms


def toy_candidate_frames(
    profile: str,
    seed: int,
    count: int = 24,
    *,
    generation: int = 1,
    temperatures: Sequence[float] = (300.0,),
    pressure: float = 0.0,
) -> list[Atoms]:
    if profile not in {"ordinary", "spin"}:
        raise ValueError("toy workflow profile must be ordinary or spin")
    if generation < 1 or count < 1 or not temperatures:
        raise ValueError("generation, count, and temperatures must be positive")
    rng = np.random.default_rng(seed)
    base = toy_base_frame(profile == "spin")
    frames: list[Atoms] = []
    generation_scale = 1.0 + 0.2 * (generation - 1)
    for index in range(count):
        atoms = base.copy()
        temperature = float(temperatures[index % len(temperatures)])
        thermal_scale = max(0.5, np.sqrt(temperature / 300.0))
        amplitude = (0.015 + 0.006 * index) * generation_scale * thermal_scale
        atoms.positions += rng.normal(0.0, amplitude, size=atoms.positions.shape)
        atoms.positions[1] += np.asarray([0.006 * index * generation_scale, 0.0, 0.0])
        if profile == "spin":
            spin = np.asarray(atoms.arrays["spin"], dtype=np.float64).copy()
            spin *= 1.0 + 0.012 * index * generation_scale
            spin += rng.normal(
                0.0,
                (0.012 + index * 0.001) * thermal_scale,
                size=spin.shape,
            )
            atoms.set_array("spin", spin)
        atoms.info.update(
            smoke_frame=index,
            generation=generation,
            temperature=temperature,
            pressure=float(pressure),
            source_id=f"g{generation}-T{temperature:g}-P{pressure:g}",
            frame_step=index,
        )
        frames.append(atoms)
    return frames


def toy_raw_features(frames: Sequence[Atoms], profile: str) -> np.ndarray:
    rows = []
    for atoms in frames:
        distances = sorted(
            atoms.get_distance(i, j, mic=True)
            for i in range(len(atoms))
            for j in range(i + 1, len(atoms))
        )
        row = list(distances)
        if profile == "spin":
            spin = np.asarray(atoms.arrays["spin"], dtype=np.float64)
            row.extend(sorted(np.linalg.norm(spin, axis=1)))
            row.extend(spin.mean(axis=0))
            row.extend(
                np.dot(spin[i], spin[j])
                for i in range(len(atoms))
                for j in range(i + 1, len(atoms))
            )
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def toy_features(frames: Sequence[Atoms], profile: str) -> np.ndarray:
    features = toy_raw_features(frames, profile)
    scale = features.std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    return (features - features.mean(axis=0)) / scale


def structure_id(atoms: Atoms) -> str:
    digest = hashlib.sha256()
    digest.update(" ".join(atoms.get_chemical_symbols()).encode())
    digest.update(np.asarray(atoms.cell, dtype=np.float64).tobytes())
    digest.update(np.asarray(atoms.positions, dtype=np.float64).tobytes())
    if "spin" in atoms.arrays:
        digest.update(np.asarray(atoms.arrays["spin"], dtype=np.float64).tobytes())
    return digest.hexdigest()


__all__ = [
    "structure_id",
    "toy_base_frame",
    "toy_candidate_frames",
    "toy_features",
    "toy_raw_features",
]
