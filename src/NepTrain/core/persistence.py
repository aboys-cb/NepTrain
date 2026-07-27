"""Small durable-write helpers for workflow state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, value: Any) -> Path:
    """Write canonical, finite JSON through a sibling temporary file."""

    destination = Path(path)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)
    return destination


__all__ = ["atomic_write_json"]
