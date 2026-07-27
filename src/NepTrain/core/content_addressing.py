"""Canonical content identifiers shared by workflow artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


_READ_BLOCK_SIZE = 1024 * 1024


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading the complete artifact into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value using NepTrain's canonical serialization."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = ["canonical_sha256", "file_sha256"]
