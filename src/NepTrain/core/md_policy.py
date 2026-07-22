"""Dependency-light validation for MD trajectory health settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


class TrajectoryHealthError(ValueError):
    """Raised when trajectory health policy or input is invalid."""


@dataclass(frozen=True)
class TrajectoryHealthPolicy:
    """Conservative defaults intended to catch clearly broken MD frames."""

    min_distance_ratio: float | None = 0.5
    min_volume_ratio: float | None = 0.5
    max_volume_ratio: float | None = 2.0
    max_force: float | None = 100.0
    max_mforce: float | None = 100.0
    max_spin_magnitude: float | None = 20.0

    @classmethod
    def from_mapping(
        cls, settings: Mapping[str, Any] | None
    ) -> "TrajectoryHealthPolicy":
        if settings is None:
            settings = {}
        if not isinstance(settings, Mapping):
            raise TrajectoryHealthError("md.health must be a mapping")
        defaults = asdict(cls())
        unknown = sorted(set(settings) - set(defaults))
        if unknown:
            raise TrajectoryHealthError(
                f"unknown md.health settings: {', '.join(unknown)}"
            )
        values: dict[str, float | None] = {}
        for name, default in defaults.items():
            raw = settings.get(name, default)
            if raw is None:
                values[name] = None
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError) as error:
                raise TrajectoryHealthError(
                    f"md.health.{name} must be positive or null"
                ) from error
            if not math.isfinite(value) or value <= 0:
                raise TrajectoryHealthError(
                    f"md.health.{name} must be positive or null"
                )
            values[name] = value
        policy = cls(**values)
        if (
            policy.min_volume_ratio is not None
            and policy.max_volume_ratio is not None
            and policy.min_volume_ratio >= policy.max_volume_ratio
        ):
            raise TrajectoryHealthError(
                "md.health volume ratios must satisfy min_volume_ratio < max_volume_ratio"
            )
        return policy


__all__ = ["TrajectoryHealthError", "TrajectoryHealthPolicy"]
