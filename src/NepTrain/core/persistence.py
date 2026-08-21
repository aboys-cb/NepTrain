"""Small durable-write helpers for workflow state."""

from __future__ import annotations

import errno
import json
from pathlib import Path
import time
from typing import Any


_TRANSIENT_WRITE_ERRNOS = {
    errno.EAGAIN,
    errno.EBUSY,
    errno.EFAULT,
    errno.EINTR,
    errno.EIO,
    errno.ESTALE,
    errno.ETIMEDOUT,
}
_WRITE_ATTEMPTS = 4
_INITIAL_RETRY_DELAY = 0.05


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
    for attempt in range(_WRITE_ATTEMPTS):
        try:
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(destination)
            break
        except OSError as error:
            if (
                error.errno not in _TRANSIENT_WRITE_ERRNOS
                or attempt == _WRITE_ATTEMPTS - 1
            ):
                raise
            time.sleep(_INITIAL_RETRY_DELAY * (2**attempt))
    return destination


__all__ = ["atomic_write_json"]
