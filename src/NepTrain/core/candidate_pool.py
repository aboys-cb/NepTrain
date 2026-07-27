"""Model-bound candidate pools for one active-learning generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from collections import Counter
from typing import Mapping, Sequence

from ase import Atoms
from ase.io import write as ase_write

from .content_addressing import file_sha256
from .persistence import atomic_write_json


class CandidatePoolError(ValueError):
    """Raised when candidates are mixed across model generations."""


def regular_batch_minimum(max_selected: int) -> int:
    """Derive the normal labeling floor without adding another user knob."""

    if max_selected < 1:
        raise ValueError("max_selected must be positive")
    return max(1, (int(max_selected) + 1) // 2)


@dataclass(frozen=True)
class CandidatePoolManifest:
    generation: int
    model_sha256: str
    frame_count: int
    requested_md_runs: int
    available_md_runs: int
    scheduled_md_runs: int
    failed_md_runs: int
    route_fingerprints: Mapping[str, str]
    counts_by_route: Mapping[str, int]

    @property
    def frontier_exhausted(self) -> bool:
        return self.scheduled_md_runs >= self.available_md_runs


def write_candidate_pool(
    candidates_path: Path,
    manifest_path: Path,
    frames: Sequence[Atoms],
    *,
    generation: int,
    model_path: Path,
    requested_md_runs: int,
    available_md_runs: int,
    scheduled_md_runs: int,
    failed_md_runs: int,
) -> CandidatePoolManifest:
    """Persist candidates and bind every frame to the model that generated it."""

    if generation < 1 or not frames:
        raise CandidatePoolError(
            "candidate pool requires a positive generation and at least one frame"
        )
    model_id = file_sha256(model_path)
    output = []
    route_fingerprints: dict[str, str] = {}
    counts_by_route: Counter[str] = Counter()
    for frame in frames:
        copied = frame.copy()
        existing = copied.info.get("sampling_model_sha256")
        if existing is not None and str(existing) != model_id:
            raise CandidatePoolError(
                "candidate already belongs to a different sampling model"
            )
        copied.info["model_generation"] = int(generation)
        copied.info["sampling_model_sha256"] = model_id
        route_id = str(copied.info.get("route_id", ""))
        route_fingerprint = str(
            copied.info.get("route_fingerprint", "")
        )
        if not route_id or not route_fingerprint:
            raise CandidatePoolError(
                "candidate frame is missing sampling route provenance"
            )
        existing_route = route_fingerprints.setdefault(
            route_id, route_fingerprint
        )
        if existing_route != route_fingerprint:
            raise CandidatePoolError(
                "one route id has multiple fingerprints in the candidate pool"
            )
        counts_by_route[route_id] += 1
        output.append(copied)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(candidates_path, output, format="extxyz")
    manifest = CandidatePoolManifest(
        generation=int(generation),
        model_sha256=model_id,
        frame_count=len(output),
        requested_md_runs=int(requested_md_runs),
        available_md_runs=int(available_md_runs),
        scheduled_md_runs=int(scheduled_md_runs),
        failed_md_runs=int(failed_md_runs),
        route_fingerprints=dict(sorted(route_fingerprints.items())),
        counts_by_route=dict(sorted(counts_by_route.items())),
    )
    atomic_write_json(
        manifest_path,
        {
            "version": 2,
            **asdict(manifest),
            "frontier_exhausted": manifest.frontier_exhausted,
        },
    )
    return manifest


def validate_candidate_pool(
    frames: Sequence[Atoms],
    manifest_path: Path,
    *,
    generation: int,
    model_path: Path,
) -> CandidatePoolManifest:
    """Fail closed if a selection stage sees stale or mixed-model candidates."""

    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("version") != 2:
        raise CandidatePoolError("unsupported candidate pool manifest")
    manifest = CandidatePoolManifest(
        generation=int(value["generation"]),
        model_sha256=str(value["model_sha256"]),
        frame_count=int(value["frame_count"]),
        requested_md_runs=int(value["requested_md_runs"]),
        available_md_runs=int(value["available_md_runs"]),
        scheduled_md_runs=int(value["scheduled_md_runs"]),
        failed_md_runs=int(value["failed_md_runs"]),
        route_fingerprints={
            str(key): str(item)
            for key, item in value["route_fingerprints"].items()
        },
        counts_by_route={
            str(key): int(item)
            for key, item in value["counts_by_route"].items()
        },
    )
    expected_model = file_sha256(model_path)
    if manifest.generation != generation:
        raise CandidatePoolError("candidate pool belongs to another generation")
    if manifest.model_sha256 != expected_model:
        raise CandidatePoolError("candidate pool belongs to another sampling model")
    if manifest.frame_count != len(frames):
        raise CandidatePoolError("candidate pool frame count drifted")
    observed_routes: dict[str, str] = {}
    observed_counts: Counter[str] = Counter()
    for frame in frames:
        if int(frame.info.get("model_generation", -1)) != generation:
            raise CandidatePoolError(
                "candidate frame belongs to another model generation"
            )
        if str(frame.info.get("sampling_model_sha256", "")) != expected_model:
            raise CandidatePoolError(
                "candidate frame belongs to another sampling model"
            )
        route_id = str(frame.info.get("route_id", ""))
        route_fingerprint = str(frame.info.get("route_fingerprint", ""))
        if not route_id or not route_fingerprint:
            raise CandidatePoolError(
                "candidate frame is missing sampling route provenance"
            )
        previous = observed_routes.setdefault(route_id, route_fingerprint)
        if previous != route_fingerprint:
            raise CandidatePoolError(
                "candidate pool mixes fingerprints for one route id"
            )
        observed_counts[route_id] += 1
    if dict(sorted(observed_routes.items())) != dict(
        manifest.route_fingerprints
    ):
        raise CandidatePoolError("candidate pool route fingerprints drifted")
    if dict(sorted(observed_counts.items())) != dict(
        manifest.counts_by_route
    ):
        raise CandidatePoolError("candidate pool route counts drifted")
    return manifest


__all__ = [
    "CandidatePoolError",
    "CandidatePoolManifest",
    "file_sha256",
    "regular_batch_minimum",
    "validate_candidate_pool",
    "write_candidate_pool",
]
