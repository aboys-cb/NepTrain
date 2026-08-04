"""Trajectory dump spacing shared by the built-in MD backends."""

from __future__ import annotations


def adaptive_dump_interval(steps: int) -> int:
    """Keep about 100 frames while retaining at least one frame per 1000 steps."""

    return max(1, min(1000, steps // 100))


__all__ = ["adaptive_dump_interval"]
