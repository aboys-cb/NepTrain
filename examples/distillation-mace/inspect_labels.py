from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.io import read


def main() -> None:
    path = Path(__file__).with_name("labeled.xyz")
    if not path.is_file():
        raise SystemExit(
            "labeled.xyz does not exist; run the labeling command first"
        )
    frames = read(path, index=":")
    print("frame  energy(eV)  max|force|(eV/A)  virial  label_engine")
    for index, frame in enumerate(frames):
        forces = np.asarray(frame.get_forces())
        virial = np.asarray(frame.info["virial"])
        print(
            f"{index:5d}",
            f"{frame.get_potential_energy():11.6f}",
            f"{np.linalg.norm(forces, axis=1).max():18.6f}",
            f"{str(virial.shape):>7}",
            frame.info["neptrain_label_engine"],
        )
    hashes = {
        frame.info["neptrain_teacher_model_sha256"]
        for frame in frames
    }
    if len(hashes) != 1:
        raise SystemExit("frames do not share one Teacher model SHA256")
    print(f"OK: {len(frames)} frames, Teacher SHA256 {hashes.pop()}")


if __name__ == "__main__":
    main()
