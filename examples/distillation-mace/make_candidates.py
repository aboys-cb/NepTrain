from __future__ import annotations

from pathlib import Path

from ase.build import bulk
from ase.io import write


def main() -> None:
    frames = []
    for scale in (0.98, 1.00, 1.02):
        for displacement in (-0.03, -0.01, 0.01, 0.03):
            frame = bulk("Al", "fcc", a=4.05, cubic=True)
            frame.set_cell(frame.cell * scale, scale_atoms=True)
            frame.positions[0, 0] += displacement
            frame.info["Config_type"] = f"Al-fcc-scale-{scale:.2f}-displacement-{displacement}"
            frames.append(frame)
    output = Path(__file__).with_name("candidates.xyz")
    write(output, frames, format="extxyz")
    print(f"Wrote {len(frames)} structures to {output.name}")


if __name__ == "__main__":
    main()
