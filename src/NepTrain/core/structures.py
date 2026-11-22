"""Strict, deterministic structure-file loading shared by CLI adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.io import read as ase_read


VIRIAL_ORDER_KEY = "neptrain_virial_order"


class StructureReadError(RuntimeError):
    """Raised when an input cannot be turned into one or more structures."""


def read_structures(
    source: str | Path,
    *,
    patterns: Sequence[str] = ("*.xyz", "*.extxyz", "*.vasp", "POSCAR*"),
    allow_empty: bool = False,
) -> list[Atoms]:
    """Read every matching structure and fail instead of returning partial data."""

    path = Path(source).expanduser()
    files: Iterable[Path]
    if path.is_dir():
        files = sorted(
            {
                candidate
                for pattern in patterns
                for candidate in path.glob(pattern)
                if candidate.is_file()
            }
        )
    else:
        files = (path,)
    frames: list[Atoms] = []
    for file_path in files:
        if not file_path.is_file():
            raise StructureReadError(
                f"structure input does not exist: {file_path}"
            )
        if allow_empty and file_path.stat().st_size == 0:
            continue
        try:
            loaded = ase_read(file_path, index=":")
            batch = loaded if isinstance(loaded, list) else [loaded]
            for frame in batch:
                order = frame.info.pop(VIRIAL_ORDER_KEY, None)
                if order is None:
                    continue
                if not isinstance(order, str) or order != "row_major":
                    raise ValueError(f"invalid {VIRIAL_ORDER_KEY}: {order!r}")
                keys = [key for key in frame.info if key.lower() == "virial"]
                if len(keys) != 1:
                    raise ValueError(f"{VIRIAL_ORDER_KEY} requires exactly one virial")
                key = keys[0]
                tensor = np.asarray(frame.info.pop(key), dtype=float)
                # ASE has already read lowercase virial as a column-major matrix.
                frame.info["virial"] = tensor.T.reshape(3, 3)
        except Exception as error:
            raise StructureReadError(
                f"failed to read structure input {file_path}: {error}"
            ) from error
        frames.extend(batch)
    if not frames and not allow_empty:
        raise StructureReadError(f"no structures found in {path}")
    return frames


__all__ = ["StructureReadError", "read_structures"]
