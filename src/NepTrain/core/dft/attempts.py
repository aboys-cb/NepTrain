"""Shared attempt-directory policy for native DFT backends."""

from __future__ import annotations

from pathlib import Path


def new_attempt_directory(
    work_dir: Path,
    index: int,
    formula: str,
    *,
    flat_single_case: bool = False,
) -> Path:
    """Return a fresh directory without counting publisher temp files."""

    if flat_single_case:
        work_dir.mkdir(parents=True, exist_ok=True)
        visible_entries = (
            entry for entry in work_dir.iterdir() if not entry.name.startswith(".")
        )
        if next(visible_entries, None) is None:
            return work_dir
        attempt = 2
        while (work_dir / f"retry-{attempt:04d}").exists():
            attempt += 1
        directory = work_dir / f"retry-{attempt:04d}"
        directory.mkdir()
        return directory

    case_root = work_dir / f"{index:06d}-{formula}"
    try:
        case_root.mkdir(parents=True)
        return case_root
    except FileExistsError:
        pass
    attempt = 2
    while (case_root / f"retry-{attempt:04d}").exists():
        attempt += 1
    directory = case_root / f"retry-{attempt:04d}"
    directory.mkdir()
    return directory


__all__ = ["new_attempt_directory"]
