"""Dependency-light validation for MD trajectory health settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    spin_magnitude_by_element: dict[str, dict[str, float | None]] = field(default_factory=dict)

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
        values: dict[str, Any] = {}
        for name, default in defaults.items():
            raw = settings.get(name, default)
            if name == "spin_magnitude_by_element":
                values[name] = _spin_ranges(raw)
                continue
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


def _spin_ranges(raw: Any) -> dict[str, dict[str, float | None]]:
    """Explicit element ranges replace the legacy global spin limit."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TrajectoryHealthError("md.health.spin_magnitude_by_element must be a mapping")
    from ase.data import atomic_numbers

    result = {}
    for element, bounds in raw.items():
        prefix = f"md.health.spin_magnitude_by_element.{element}"
        if not isinstance(element, str) or element not in atomic_numbers or element == "X":
            raise TrajectoryHealthError(f"{prefix}: expected a chemical element symbol")
        if not isinstance(bounds, Mapping) or set(bounds) - {"min", "max"}:
            raise TrajectoryHealthError(f"{prefix} must be a mapping with only min and max")
        limits = {}
        for key in ("min", "max"):
            value = bounds.get(key)
            if value is not None:
                try:
                    if isinstance(value, bool):
                        raise ValueError
                    value = float(value)
                    if not math.isfinite(value) or value < 0:
                        raise ValueError
                except (ValueError, TypeError) as error:
                    raise TrajectoryHealthError(
                        f"{prefix}.{key} must be finite and non-negative, or null"
                    ) from error
            limits[key] = value
        if limits["min"] is not None and limits["max"] is not None and limits["min"] > limits["max"]:
            raise TrajectoryHealthError(f"{prefix} must satisfy min <= max")
        result[element] = limits
    return result


__all__ = ["TrajectoryHealthError", "TrajectoryHealthPolicy"]
