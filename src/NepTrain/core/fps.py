"""Deterministic element-set hierarchical farthest-point sampling.

This module owns only selection policy.  Descriptor construction and workflow
state remain outside it so the same algorithm can serve automatic and manual
sampling paths.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import floor, sqrt
from typing import Any, Mapping, Sequence

import numpy as np


ElementSet = tuple[str, ...]


@dataclass(frozen=True)
class FPSGroupReport:
    """Selection diagnostics for one exact element-set group."""

    element_set: ElementSet
    candidate_count: int
    reference_count: int
    initial_quota: int
    selected_count: int
    selected_ids: tuple[str, ...]
    counts_by_stratum: Mapping[str, int]
    remaining_novelty: float


@dataclass(frozen=True)
class HierarchicalFPSResult:
    """Result of an element-set hierarchical FPS selection."""

    selected_indices: tuple[int, ...]
    selected_ids: tuple[str, ...]
    selected_novelty: tuple[float, ...]
    groups: Mapping[ElementSet, FPSGroupReport]
    counts_by_stratum: Mapping[str, int]
    remaining_novelty: float


@dataclass(frozen=True)
class FarthestPointResult:
    """Result from the shared single-group FPS interface."""

    selected_indices: tuple[int, ...]
    selected_ids: tuple[str, ...]
    selected_novelty: tuple[float, ...]
    counts_by_stratum: Mapping[str, int]
    remaining_novelty: float


@dataclass
class _GroupState:
    key: ElementSet
    original_indices: np.ndarray
    ids: tuple[str, ...]
    strata: tuple[str, ...]
    points: np.ndarray
    distances: np.ndarray
    reference_count: int
    quota: int
    available: np.ndarray
    selected_local: list[int]
    selected_novelty: list[float]
    counts_by_stratum: Counter[str]

    def eligible(self, min_novelty: float) -> np.ndarray:
        indices = np.flatnonzero(self.available)
        if not len(indices):
            return indices
        # A group without a warm-start reference needs one deterministic seed,
        # even when its only candidate lies exactly at the group centroid.
        if not self.reference_count and not self.selected_local:
            return indices[self.distances[indices] >= min_novelty]
        return indices[self.distances[indices] > min_novelty]

    def next_candidate(self, min_novelty: float) -> int | None:
        eligible = self.eligible(min_novelty)
        if not len(eligible):
            return None
        active_strata = {self.strata[index] for index in eligible}
        least_used = min(self.counts_by_stratum[name] for name in active_strata)
        balanced = [
            int(index)
            for index in eligible
            if self.counts_by_stratum[self.strata[index]] == least_used
        ]
        return min(
            balanced,
            key=lambda index: (-float(self.distances[index]), self.ids[index]),
        )

    def select(self, index: int) -> None:
        first_without_reference = (
            not self.reference_count and not self.selected_local
        )
        self.selected_local.append(index)
        self.selected_novelty.append(float(self.distances[index]))
        self.counts_by_stratum[self.strata[index]] += 1
        self.available[index] = False
        new_distances = np.linalg.norm(self.points - self.points[index], axis=1)
        self.distances = (
            new_distances
            if first_without_reference
            else np.minimum(self.distances, new_distances)
        )

    def remaining_novelty(self) -> float:
        return float(self.distances[self.available].max(initial=0.0))


def element_set(structure: Any) -> ElementSet:
    """Return the sorted unique chemical symbols of an ASE-like structure."""

    try:
        symbols = structure.get_chemical_symbols()
    except AttributeError as error:
        raise TypeError(
            "structures must provide get_chemical_symbols()"
        ) from error
    key = tuple(sorted(set(str(symbol) for symbol in symbols)))
    if not key:
        raise ValueError("structures must contain at least one atom")
    return key


def _validate_descriptors(
    values: np.ndarray,
    *,
    expected_rows: int,
    name: str,
    feature_count: int | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if len(array) != expected_rows:
        raise ValueError(f"{name} rows must match its structures")
    if feature_count is not None and array.shape[1] != feature_count:
        raise ValueError("candidate and reference feature dimensions must match")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _sqrt_quotas(
    group_sizes: Mapping[ElementSet, int],
    budget: int,
) -> dict[ElementSet, int]:
    """Apportion ``budget`` by sqrt(group size), with one seat per group."""

    keys = sorted(group_sizes)
    if budget < len(keys):
        raise ValueError(
            "selection budget is smaller than the number of non-empty "
            "element-set groups"
        )
    quotas = {key: 1 for key in keys}
    remaining = budget - len(keys)
    while remaining:
        active = [
            key for key in keys if quotas[key] < int(group_sizes[key])
        ]
        if not active:
            break
        total_weight = sum(sqrt(group_sizes[key]) for key in active)
        shares = {
            key: remaining * sqrt(group_sizes[key]) / total_weight
            for key in active
        }
        allocated = 0
        for key in active:
            extra = min(
                int(group_sizes[key]) - quotas[key],
                floor(shares[key]),
            )
            quotas[key] += extra
            allocated += extra
        remaining -= allocated
        if not remaining:
            break
        active = [
            key for key in keys if quotas[key] < int(group_sizes[key])
        ]
        if not active:
            break
        winner = min(
            active,
            key=lambda key: (
                -(shares.get(key, 0.0) - floor(shares.get(key, 0.0))),
                key,
            ),
        )
        quotas[winner] += 1
        remaining -= 1
    return quotas


def _normalized_group(
    points: np.ndarray,
    references: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.vstack((references, points)) if len(references) else points
    center = np.median(combined, axis=0)
    scale = np.std(combined, axis=0)
    scale[scale < 1.0e-12] = 1.0
    normalized_points = (points - center) / scale
    normalized_references = (references - center) / scale
    return normalized_points, normalized_references


def _nearest_distances(
    points: np.ndarray,
    references: np.ndarray,
    *,
    point_batch_size: int = 4096,
    reference_batch_size: int = 512,
) -> np.ndarray:
    """Return exact nearest-reference distances with bounded peak memory."""

    result = np.full(len(points), np.inf, dtype=np.float64)
    for point_start in range(0, len(points), point_batch_size):
        point_stop = min(point_start + point_batch_size, len(points))
        point_chunk = points[point_start:point_stop]
        point_norm = np.einsum("ij,ij->i", point_chunk, point_chunk)
        chunk_min = np.full(len(point_chunk), np.inf, dtype=np.float64)
        for reference_start in range(0, len(references), reference_batch_size):
            reference_stop = min(
                reference_start + reference_batch_size, len(references)
            )
            reference_chunk = references[reference_start:reference_stop]
            reference_norm = np.einsum(
                "ij,ij->i", reference_chunk, reference_chunk
            )
            squared = (
                point_norm[:, None]
                + reference_norm[None, :]
                - 2.0 * point_chunk @ reference_chunk.T
            )
            chunk_min = np.minimum(
                chunk_min,
                np.maximum(squared, 0.0).min(axis=1),
            )
        result[point_start:point_stop] = np.sqrt(chunk_min)
    return result


def farthest_point_sampling(
    candidate_descriptors: np.ndarray,
    *,
    budget: int,
    min_novelty: float = 0.0,
    reference_descriptors: np.ndarray | None = None,
    candidate_ids: Sequence[str] | None = None,
    strata: Sequence[str] | None = None,
) -> FarthestPointResult:
    """Select one deterministic, balanced FPS group.

    ``min_novelty`` is a strict gate after the deterministic no-reference seed.
    Exact duplicates of a reference or selected candidate are therefore never
    selected, including when the threshold is zero.
    """

    if budget < 0:
        raise ValueError("budget must be non-negative")
    if min_novelty < 0:
        raise ValueError("min_novelty must be non-negative")
    points = np.asarray(candidate_descriptors, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("candidate_descriptors must be a two-dimensional array")
    if not np.isfinite(points).all():
        raise ValueError("candidate_descriptors must contain only finite values")
    if reference_descriptors is None:
        references = np.empty((0, points.shape[1]), dtype=np.float64)
    else:
        references = np.asarray(reference_descriptors, dtype=np.float64)
        if references.ndim != 2 or references.shape[1] != points.shape[1]:
            raise ValueError(
                "candidate and reference descriptors must have matching features"
            )
        if not np.isfinite(references).all():
            raise ValueError(
                "reference_descriptors must contain only finite values"
            )
    ids = tuple(
        str(value)
        for value in (
            candidate_ids
            if candidate_ids is not None
            else (f"{index:012d}" for index in range(len(points)))
        )
    )
    if len(ids) != len(points):
        raise ValueError("candidate_ids must match candidate descriptors")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_ids must be unique")
    group_names = tuple(
        str(value)
        for value in (
            strata if strata is not None else ("all" for _ in range(len(points)))
        )
    )
    if len(group_names) != len(points):
        raise ValueError("strata must match candidate descriptors")
    if not len(points) or budget == 0:
        return FarthestPointResult((), (), (), {}, 0.0)

    original_indices = np.arange(len(points))
    order = np.asarray(sorted(range(len(points)), key=lambda index: ids[index]))
    points = points[order]
    original_indices = original_indices[order]
    ids = tuple(ids[index] for index in order)
    group_names = tuple(group_names[index] for index in order)
    normalized_points, normalized_references = _normalized_group(
        points, references
    )
    if len(normalized_references):
        distances = _nearest_distances(
            normalized_points, normalized_references
        )
    else:
        distances = np.linalg.norm(
            normalized_points - normalized_points.mean(axis=0),
            axis=1,
        )
    state = _GroupState(
        key=(),
        original_indices=original_indices,
        ids=ids,
        strata=group_names,
        points=normalized_points,
        distances=distances,
        reference_count=len(normalized_references),
        quota=min(int(budget), len(points)),
        available=np.ones(len(points), dtype=bool),
        selected_local=[],
        selected_novelty=[],
        counts_by_stratum=Counter(),
    )
    while len(state.selected_local) < state.quota:
        candidate = state.next_candidate(min_novelty)
        if candidate is None:
            break
        state.select(candidate)
    return FarthestPointResult(
        selected_indices=tuple(
            int(state.original_indices[index]) for index in state.selected_local
        ),
        selected_ids=tuple(state.ids[index] for index in state.selected_local),
        selected_novelty=tuple(state.selected_novelty),
        counts_by_stratum=dict(sorted(state.counts_by_stratum.items())),
        remaining_novelty=state.remaining_novelty(),
    )


def hierarchical_farthest_point_sampling(
    candidate_structures: Sequence[Any],
    candidate_descriptors: np.ndarray,
    candidate_ids: Sequence[str],
    strata: Sequence[str],
    *,
    budget: int,
    min_novelty: float = 0.0,
    reference_structures: Sequence[Any] = (),
    reference_descriptors: np.ndarray | None = None,
) -> HierarchicalFPSResult:
    """Select candidates using element groups, soft strata, and conditional FPS.

    Exact element sets define the outer groups.  Initial group quotas are
    proportional to the square root of candidate group size and guarantee one
    slot per non-empty group.  FPS inside a group is warm-started only by
    reference structures with that same element set.  Unused quota is
    redistributed to other groups that still have candidates above
    ``min_novelty``.
    """

    if budget < 0:
        raise ValueError("budget must be non-negative")
    if min_novelty < 0:
        raise ValueError("min_novelty must be non-negative")
    if len(candidate_structures) != len(candidate_ids) or len(
        candidate_structures
    ) != len(strata):
        raise ValueError("candidate metadata must match candidate structures")
    ids = tuple(str(value) for value in candidate_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_ids must be unique")

    points = _validate_descriptors(
        candidate_descriptors,
        expected_rows=len(candidate_structures),
        name="candidate_descriptors",
    )
    if reference_descriptors is None:
        references = np.empty((0, points.shape[1]), dtype=np.float64)
    else:
        references = _validate_descriptors(
            reference_descriptors,
            expected_rows=len(reference_structures),
            name="reference_descriptors",
            feature_count=points.shape[1],
        )
    if reference_descriptors is None and reference_structures:
        raise ValueError(
            "reference_descriptors are required when reference_structures "
            "are provided"
        )
    if reference_descriptors is not None and len(reference_structures) != len(
        references
    ):
        raise ValueError(
            "reference structures and descriptors must have the same length"
        )
    if not len(candidate_structures):
        return HierarchicalFPSResult((), (), (), {}, {}, 0.0)

    candidate_keys = tuple(element_set(value) for value in candidate_structures)
    reference_keys = tuple(element_set(value) for value in reference_structures)
    group_sizes = Counter(candidate_keys)
    effective_budget = min(int(budget), len(candidate_structures))
    quotas = _sqrt_quotas(group_sizes, effective_budget)

    states: dict[ElementSet, _GroupState] = {}
    for key in sorted(group_sizes):
        group_indices = sorted(
            (index for index, value in enumerate(candidate_keys) if value == key),
            key=lambda index: ids[index],
        )
        reference_indices = [
            index for index, value in enumerate(reference_keys) if value == key
        ]
        group_points = points[group_indices]
        group_references = references[reference_indices]
        normalized_points, normalized_references = _normalized_group(
            group_points, group_references
        )
        if len(normalized_references):
            distances = _nearest_distances(
                normalized_points, normalized_references
            )
        else:
            distances = np.linalg.norm(
                normalized_points - normalized_points.mean(axis=0),
                axis=1,
            )
        states[key] = _GroupState(
            key=key,
            original_indices=np.asarray(group_indices, dtype=int),
            ids=tuple(ids[index] for index in group_indices),
            strata=tuple(str(strata[index]) for index in group_indices),
            points=normalized_points,
            distances=distances,
            reference_count=len(reference_indices),
            quota=quotas[key],
            available=np.ones(len(group_indices), dtype=bool),
            selected_local=[],
            selected_novelty=[],
            counts_by_stratum=Counter(),
        )

    # First honor each group's sqrt-size quota.
    for key in sorted(states):
        state = states[key]
        while len(state.selected_local) < state.quota:
            candidate = state.next_candidate(min_novelty)
            if candidate is None:
                break
            state.select(candidate)

    # Then redistribute unused seats by the best still-eligible novelty.  This
    # never fills a budget with structures below the novelty threshold.
    selected_count = sum(len(state.selected_local) for state in states.values())
    while selected_count < effective_budget:
        choices = [
            (state.next_candidate(min_novelty), state)
            for state in states.values()
        ]
        choices = [
            (index, state)
            for index, state in choices
            if index is not None
        ]
        if not choices:
            break
        index, state = min(
            choices,
            key=lambda item: (
                -float(item[1].distances[item[0]]),
                item[1].key,
                item[1].ids[item[0]],
            ),
        )
        state.select(index)
        selected_count += 1

    selections: list[tuple[str, int, float, str]] = []
    reports: dict[ElementSet, FPSGroupReport] = {}
    total_counts: Counter[str] = Counter()
    for key in sorted(states):
        state = states[key]
        selected_ids = tuple(state.ids[index] for index in state.selected_local)
        for index, novelty in zip(
            state.selected_local, state.selected_novelty, strict=True
        ):
            selections.append(
                (
                    state.ids[index],
                    int(state.original_indices[index]),
                    novelty,
                    state.strata[index],
                )
            )
        total_counts.update(state.counts_by_stratum)
        reports[key] = FPSGroupReport(
            element_set=key,
            candidate_count=len(state.ids),
            reference_count=state.reference_count,
            initial_quota=state.quota,
            selected_count=len(state.selected_local),
            selected_ids=selected_ids,
            counts_by_stratum=dict(sorted(state.counts_by_stratum.items())),
            remaining_novelty=state.remaining_novelty(),
        )

    # The selection sequence is deterministic by policy.  It is intentionally
    # not sorted after the fact because novelty values describe that sequence.
    return HierarchicalFPSResult(
        selected_indices=tuple(value[1] for value in selections),
        selected_ids=tuple(value[0] for value in selections),
        selected_novelty=tuple(value[2] for value in selections),
        groups=reports,
        counts_by_stratum=dict(sorted(total_counts.items())),
        remaining_novelty=max(
            (report.remaining_novelty for report in reports.values()),
            default=0.0,
        ),
    )


__all__ = [
    "ElementSet",
    "FarthestPointResult",
    "FPSGroupReport",
    "HierarchicalFPSResult",
    "element_set",
    "farthest_point_sampling",
    "hierarchical_farthest_point_sampling",
]
