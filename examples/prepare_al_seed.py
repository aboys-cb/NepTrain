"""Generate unlabeled Al seeds; label them with the workflow's DFT backend."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write


def _frame(scale: float, displacement: float):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.set_cell(atoms.cell * scale, scale_atoms=True)
    atoms.positions[0, 0] += displacement
    atoms.info["Config_type"] = "tutorial-al-seed"
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    targets = [
        root / "seed-train.xyz",
        root / "seed-validation.xyz",
        root / "structures/al.xyz",
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.force:
        parser.error(
            "files already exist (use --force to regenerate): " + ", ".join(existing)
        )
    targets[-1].parent.mkdir(parents=True, exist_ok=True)
    train = [
        _frame(scale, displacement)
        for scale in np.linspace(0.97, 1.03, 12)
        for displacement in (-0.015, 0.015)
    ]
    validation = [
        _frame(scale, displacement)
        for scale, displacement in (
            (0.975, 0.0),
            (0.9925, 0.01),
            (1.0075, -0.01),
            (1.025, 0.0),
        )
    ]
    for path, frames in zip(targets, (train, validation, [_frame(1.0, 0.0)])):
        write(path, frames, format="extxyz")
        print(f"Wrote {len(frames)} unlabeled frames to {path}")
    print(
        "Label seed-train.xyz with the same DFT inputs and resources used by project.yaml."
    )
    print("No train.xyz or validation.xyz was created; no EMT/DFT labels are mixed.")


if __name__ == "__main__":
    main()
