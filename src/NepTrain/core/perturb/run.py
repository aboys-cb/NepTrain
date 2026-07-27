"""Deterministic structure perturbation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

from ..structures import read_structures


class PerturbError(ValueError):
    """Raised when a perturbation request is invalid."""


def _perturb_positions(
    atoms: Atoms,
    max_displacement: float,
    rng: np.random.Generator,
) -> Atoms:
    perturbed = atoms.copy()
    perturbed.positions += rng.uniform(
        -max_displacement,
        max_displacement,
        size=perturbed.positions.shape,
    )
    return perturbed


def _strained_structure(
    atoms: Atoms,
    cell_perturbation: float,
    max_displacement: float,
    rng: np.random.Generator,
) -> Atoms:
    strained = atoms.copy()
    strains = rng.uniform(-cell_perturbation, cell_perturbation, size=3)
    strained.set_cell(atoms.cell.array * (1.0 + strains), scale_atoms=True)
    return _perturb_positions(strained, max_displacement, rng)


def _deformed_structure(
    atoms: Atoms,
    cell_perturbation: float,
    max_displacement: float,
    rng: np.random.Generator,
) -> Atoms:
    deformed = atoms.copy()
    deformation = np.eye(3) + rng.uniform(
        -cell_perturbation,
        cell_perturbation,
        size=(3, 3),
    )
    deformed.set_cell(deformation @ atoms.cell.array, scale_atoms=True)
    return _perturb_positions(deformed, max_displacement, rng)


def perturb(
    source: str | Path,
    *,
    cell_perturbation: float = 0.04,
    max_displacement: float = 0.1,
    count: int = 50,
    seed: int = 42,
) -> list[Atoms]:
    """Perturb all input structures in stable file/frame order."""

    if count < 1:
        raise PerturbError("count must be at least 1")
    if cell_perturbation < 0:
        raise PerturbError("cell_perturbation must be non-negative")
    if max_displacement < 0:
        raise PerturbError("max_displacement must be non-negative")
    if seed < 0:
        raise PerturbError("seed must be non-negative")

    rng = np.random.default_rng(seed)
    result: list[Atoms] = []
    for atoms in read_structures(source):
        for index in range(count):
            mode = "deformed" if index % 2 == 0 else "strained"
            generated = (
                _deformed_structure(
                    atoms,
                    cell_perturbation,
                    max_displacement,
                    rng,
                )
                if mode == "deformed"
                else _strained_structure(
                    atoms,
                    cell_perturbation,
                    max_displacement,
                    rng,
                )
            )
            generated.info["Config_type"] = (
                f"perturb {index + 1} {mode} "
                f"cell_perturbation {cell_perturbation:g} "
                f"max_displacement {max_displacement:g} seed {seed}"
            )
            result.append(generated)
    return result


def run_perturb(args) -> None:
    """CLI adapter for deterministic structure perturbation."""

    frames = perturb(
        args.model_path,
        cell_perturbation=args.cell_pert_fraction,
        max_displacement=args.max_displacement,
        count=args.num,
        seed=args.seed,
    )
    output = Path(args.out_file_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    ase_write(
        output,
        frames,
        format="extxyz",
        append=args.append,
    )


__all__ = ["PerturbError", "perturb", "run_perturb"]
