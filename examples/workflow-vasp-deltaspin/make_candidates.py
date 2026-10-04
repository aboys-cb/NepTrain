"""Generate Fe spin targets, without fabricating DeltaSpin labels."""

from pathlib import Path
import argparse
import numpy as np
from ase.build import bulk
from ase.io import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_dir)
    paths = [
        root / "seed-train.xyz",
        root / "structures/fm.xyz",
        root / "structures/canted.xyz",
    ]
    if not args.force and any(path.exists() for path in paths):
        parser.error("outputs already exist; use --force to regenerate")
    paths[-1].parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for scale in (0.98, 1.0, 1.02):
        for angle in (0.0, 0.1, 0.2, 0.3):
            atoms = bulk("Fe", "bcc", a=2.86 * scale, cubic=True).repeat((2, 2, 2))
            atoms.positions[0, 0] += 0.01
            spin = np.zeros((len(atoms), 3))
            spin[:, 2] = 2.2 * np.cos(angle)
            spin[:, 0] = (
                2.2 * np.sin(angle) * np.where(np.arange(len(atoms)) % 2, -1, 1)
            )
            atoms.set_array("spin", spin)
            atoms.info["Config_type"] = f"Fe-scale-{scale}-angle-{angle}"
            frames.append(atoms)
    write(paths[0], frames, format="extxyz")
    write(paths[1], frames[4], format="extxyz")
    write(paths[2], frames[6], format="extxyz")
    print(
        "Wrote 12 unlabeled 16-atom Fe frames and two route starts; moments are in mu_B."
    )
    print("Run DeltaSpin labeling to create train.xyz with actual spin and mforce.")


if __name__ == "__main__":
    main()
