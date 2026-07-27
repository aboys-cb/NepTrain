"""Strict, deterministic structure-file loading shared by CLI adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from ase import Atoms
from ase.io import read as ase_read


class StructureReadError(RuntimeError):
    """Raised when an input cannot be turned into one or more structures."""


def read_structures(
    source: str | Path,
    *,
    patterns: Sequence[str] = ("*.vasp", "*.xyz"),
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
        try:
            loaded = ase_read(file_path, index=":")
        except Exception as error:
            raise StructureReadError(
                f"failed to read structure input {file_path}: {error}"
            ) from error
        frames.extend(loaded if isinstance(loaded, list) else [loaded])
    if not frames:
        raise StructureReadError(f"no structures found in {path}")
    return frames


__all__ = ["StructureReadError", "read_structures"]
