"""Explicit, content-addressed sampling routes.

A route is deliberately a thin execution contract.  It binds inputs and the
trust-frontier policy, but does not interpret the physics encoded by the
LAMMPS template.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

from .content_addressing import canonical_sha256, file_sha256


class SamplingRouteError(ValueError):
    """Raised when an explicit sampling route is incomplete or ambiguous."""


MATURITY_STAGES = (
    "smoke_passed",
    "short_stable",
    "long_stable",
    "production_ready",
)
DEFAULT_PROGRESSION = {
    "steps": {
        "smoke_passed": 10000,
        "short_stable": 40000,
        "long_stable": 160000,
        "production_ready": 640000,
    },
    "replicas": {
        "smoke_passed": 1,
        "short_stable": 1,
        "long_stable": 2,
        "production_ready": 3,
    },
}


def normalized_progression(
    value: Mapping[str, Any] | None,
) -> dict[str, dict[str, int]]:
    source = value or {}
    steps = source.get("steps") or {}
    replicas = source.get("replicas") or {}
    return {
        "steps": {
            name: int(steps.get(name, DEFAULT_PROGRESSION["steps"][name]))
            for name in MATURITY_STAGES
        },
        "replicas": {
            name: int(
                replicas.get(name, DEFAULT_PROGRESSION["replicas"][name])
            )
            for name in MATURITY_STAGES
        },
    }


def _path_digest(path: Path) -> str:
    if path.is_file():
        return file_sha256(path)
    if path.is_dir():
        entries = [
            {
                "path": str(item.relative_to(path)),
                "sha256": file_sha256(item),
            }
            for item in sorted(path.rglob("*"))
            if item.is_file()
        ]
        return canonical_sha256(entries)
    raise SamplingRouteError(f"sampling route input does not exist: {path}")


@dataclass(frozen=True)
class SamplingRoute:
    route_id: str
    structure_paths: tuple[Path, ...]
    template_path: Path
    conditions: Mapping[str, Any]
    progression: Mapping[str, Any]
    template_sha256: str
    structure_source_sha256: tuple[str, ...]
    fingerprint: str

    @classmethod
    def from_config(
        cls,
        value: Mapping[str, Any],
        *,
        base_dir: Path,
    ) -> "SamplingRoute":
        route_id = str(value.get("id", "")).strip()
        if not route_id or re.fullmatch(r"[A-Za-z0-9_.-]+", route_id) is None:
            raise SamplingRouteError(
                "sampling route id must use only letters, digits, '.', '_', or '-'"
            )
        raw_structures = value.get("structures")
        if (
            not isinstance(raw_structures, list)
            or not raw_structures
            or any(not isinstance(item, str) or not item for item in raw_structures)
        ):
            raise SamplingRouteError(
                f"sampling route {route_id!r} requires a non-empty structures list"
            )

        def resolve(raw: str) -> Path:
            path = Path(raw).expanduser()
            return (
                (base_dir / path).resolve()
                if not path.is_absolute()
                else path.resolve()
            )

        structure_paths = tuple(resolve(item) for item in raw_structures)
        template_value = value.get("template_path")
        if not isinstance(template_value, str) or not template_value:
            raise SamplingRouteError(
                f"sampling route {route_id!r} requires template_path"
            )
        template_path = resolve(template_value)
        if not template_path.is_file():
            raise SamplingRouteError(
                f"sampling route template does not exist: {template_path}"
            )
        raw_conditions = value.get("conditions")
        raw_progression = value.get("progression")
        if not isinstance(raw_conditions, Mapping):
            raise SamplingRouteError(
                f"sampling route {route_id!r} requires conditions"
            )
        if raw_progression is not None and not isinstance(
            raw_progression, Mapping
        ):
            raise SamplingRouteError(
                f"sampling route {route_id!r} progression must be a mapping"
            )
        temperature_path = [
            float(item) for item in raw_conditions["temperature_path"]
        ]
        conditions: dict[str, Any] = {
            "temperature_path": temperature_path,
            "production_temperatures": [
                float(item)
                for item in raw_conditions.get(
                    "production_temperatures", temperature_path
                )
            ],
            "pressure": float(raw_conditions.get("pressure", 0.0)),
        }
        progression = normalized_progression(raw_progression)
        template_sha256 = file_sha256(template_path)
        source_hashes = tuple(_path_digest(path) for path in structure_paths)
        fingerprint_payload = {
            "route_id": route_id,
            "template_sha256": template_sha256,
            "structure_source_sha256": list(source_hashes),
            "conditions": conditions,
            "progression": progression,
        }
        return cls(
            route_id=route_id,
            structure_paths=structure_paths,
            template_path=template_path,
            conditions=conditions,
            progression=progression,
            template_sha256=template_sha256,
            structure_source_sha256=source_hashes,
            fingerprint=canonical_sha256(fingerprint_payload),
        )


def load_sampling_routes(
    sampling: Mapping[str, Any],
    *,
    base_dir: Path,
) -> tuple[SamplingRoute, ...]:
    raw_routes = sampling.get("routes")
    if not isinstance(raw_routes, list) or not raw_routes:
        raise SamplingRouteError(
            "sampling.routes must contain at least one explicit route"
        )
    routes = tuple(
        SamplingRoute.from_config(value, base_dir=base_dir)
        for value in raw_routes
    )
    ids = [route.route_id for route in routes]
    if len(set(ids)) != len(ids):
        raise SamplingRouteError("sampling route ids must be unique")
    return routes


__all__ = [
    "DEFAULT_PROGRESSION",
    "MATURITY_STAGES",
    "SamplingRoute",
    "SamplingRouteError",
    "load_sampling_routes",
    "normalized_progression",
]
