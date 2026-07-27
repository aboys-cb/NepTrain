"""Create a small EMT-labeled Al seed set for workflow mechanics tutorials."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from ase.stress import voigt_6_to_full_3x3_stress


def _labeled_frame(scale: float, displacement: float):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.set_cell(atoms.cell * scale, scale_atoms=True)
    atoms.positions[0, 0] += displacement
    atoms.calc = EMT()
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    stress = voigt_6_to_full_3x3_stress(atoms.get_stress())
    virial = -stress * atoms.get_volume()
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    atoms.info["virial"] = virial
    atoms.info["Config_type"] = "tutorial-emt-seed"
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    structures = root / "structures"
    structures.mkdir(parents=True, exist_ok=True)

    train = [
        _labeled_frame(scale, displacement)
        for scale in np.linspace(0.97, 1.03, 12)
        for displacement in (-0.015, 0.015)
    ]
    validation = [
        _labeled_frame(scale, displacement)
        for scale, displacement in (
            (0.975, 0.0),
            (0.9925, 0.01),
            (1.0075, -0.01),
            (1.025, 0.0),
        )
    ]
    start = bulk("Al", "fcc", a=4.05, cubic=True)
    start.info["Config_type"] = "tutorial-al-start"

    write(root / "train.xyz", train, format="extxyz")
    write(root / "validation.xyz", validation, format="extxyz")
    write(structures / "al.xyz", start, format="extxyz")
    print(f"Wrote {len(train)} training frames to {root / 'train.xyz'}")
    print(
        f"Wrote {len(validation)} validation frames to "
        f"{root / 'validation.xyz'}"
    )
    print(f"Wrote the MD start structure to {structures / 'al.xyz'}")
    print("These EMT labels are for workflow mechanics only, not production.")


if __name__ == "__main__":
    main()
