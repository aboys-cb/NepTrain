"""Structure labeling through a stable Adapter seam."""

from __future__ import annotations

from .interface import LabelRequest, LabelResult, LabelingError, label


__all__ = [
    "LabelRequest",
    "LabelResult",
    "LabelingError",
    "label",
]
