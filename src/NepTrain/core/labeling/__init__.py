"""Structure labeling through one stable Interface and several Adapters."""

from __future__ import annotations

from .interface import LabelRequest, LabelResult, LabelingError, label


__all__ = [
    "LabelRequest",
    "LabelResult",
    "LabelingError",
    "label",
]
