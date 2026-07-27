from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import write


def main() -> None:
    frames = []
    for scale in (0.98, 1.00, 1.02):
        frame = Atoms(
            "OH2",
            positions=[
                [6.0, 6.0, 6.0],
                [6.0 + 0.9572 * scale, 6.0, 6.0],
                [6.0 - 0.2400 * scale, 6.0 + 0.9266 * scale, 6.0],
            ],
            cell=[12.0, 12.0, 12.0],
            pbc=True,
        )
        frame.info["Config_type"] = f"water-oh-scale-{scale:.2f}"
        frames.append(frame)
    output = Path(__file__).with_name("candidates.xyz")
    write(output, frames, format="extxyz")
    print(f"Wrote {len(frames)} structures to {output.name}")


if __name__ == "__main__":
    main()
