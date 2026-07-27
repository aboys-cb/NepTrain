"""Deterministic physical health checks for MD trajectories."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
from ..md_policy import TrajectoryHealthError, TrajectoryHealthPolicy


@dataclass(frozen=True)
class TrajectoryHealthReport:
    process_completed: bool
    trajectory_completed: bool
    frame_count: int
    first_bad_frame: int | None
    first_bad_step: int | None
    reason_codes: tuple[str, ...]
    first_bad_metrics: Mapping[str, float]
    windows: tuple[str, ...]
    counts_by_window: Mapping[str, int]
    thresholds: Mapping[str, float | None]
    available_signals: tuple[str, ...]
    unavailable_thresholds: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        ranges = []
        start = 0
        for index in range(1, len(self.windows) + 1):
            if index == len(self.windows) or self.windows[index] != self.windows[start]:
                ranges.append(
                    {
                        "window": self.windows[start],
                        "start_frame": start,
                        "end_frame": index - 1,
                    }
                )
                start = index
        return {
            "version": 1,
            "process_completed": self.process_completed,
            "trajectory_completed": self.trajectory_completed,
            "frame_count": self.frame_count,
            "first_bad_frame": self.first_bad_frame,
            "first_bad_step": self.first_bad_step,
            "reason_codes": list(self.reason_codes),
            "first_bad_metrics": dict(self.first_bad_metrics),
            "window_ranges": ranges,
            "counts_by_window": dict(self.counts_by_window),
            "thresholds": dict(self.thresholds),
            "available_signals": list(self.available_signals),
            "unavailable_thresholds": list(self.unavailable_thresholds),
        }


def is_structure_reasonable(
    atoms: Atoms,
    *,
    min_distance_ratio: float = 0.7,
) -> bool:
    """Return whether every periodic pair clears a covalent-radius cutoff."""

    if min_distance_ratio < 0:
        raise ValueError("min_distance_ratio must be non-negative")
    if not len(atoms) or min_distance_ratio == 0:
        return True
    radii = np.asarray(covalent_radii[atoms.numbers], dtype=np.float64)
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("all atoms must have finite positive covalent radii")
    pair_i, _ = neighbor_list("ij", atoms, radii * min_distance_ratio)
    return len(pair_i) == 0


def _maximum_norm(
    frame: Atoms, names: Sequence[str]
) -> tuple[float | None, bool]:
    for name in names:
        if name not in frame.arrays:
            continue
        values = np.asarray(frame.arrays[name], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            return None, False
        if values.size == 0:
            return 0.0, True
        return float(np.linalg.norm(values, axis=-1).max()), True
    return None, True


def _frame_health(
    frame: Atoms,
    *,
    reference_volume: float,
    policy: TrajectoryHealthPolicy,
) -> tuple[tuple[str, ...], dict[str, float]]:
    reasons: list[str] = []
    metrics: dict[str, float] = {}
    positions = np.asarray(frame.positions, dtype=np.float64)
    cell = np.asarray(frame.cell, dtype=np.float64)
    if not np.all(np.isfinite(positions)):
        reasons.append("non_finite_positions")
    if not np.all(np.isfinite(cell)):
        reasons.append("non_finite_cell")
    volume = float(abs(frame.get_volume())) if np.all(np.isfinite(cell)) else np.nan
    if np.isfinite(volume):
        volume_ratio = volume / reference_volume
        metrics["volume_ratio"] = volume_ratio
        if (
            policy.min_volume_ratio is not None
            and volume_ratio < policy.min_volume_ratio
        ):
            reasons.append("volume_ratio_below_min")
        if (
            policy.max_volume_ratio is not None
            and volume_ratio > policy.max_volume_ratio
        ):
            reasons.append("volume_ratio_above_max")
    else:
        reasons.append("non_finite_volume")

    if policy.min_distance_ratio is not None and not reasons:
        radii = np.asarray(
            [covalent_radii[number] for number in frame.numbers], dtype=np.float64
        )
        pair_i, pair_j, distances = neighbor_list(
            "ijd", frame, radii * policy.min_distance_ratio
        )
        if len(distances):
            ratios = distances / (radii[pair_i] + radii[pair_j])
            minimum_ratio = float(np.min(ratios))
            metrics["minimum_distance_ratio"] = minimum_ratio
            reasons.append("min_distance_ratio_below_min")

    maximum_force, force_finite = _maximum_norm(frame, ("nep_force", "forces"))
    if not force_finite:
        reasons.append("non_finite_force")
    elif maximum_force is not None:
        metrics["max_force"] = maximum_force
        if policy.max_force is not None and maximum_force > policy.max_force:
            reasons.append("max_force_above_limit")

    maximum_mforce, mforce_finite = _maximum_norm(frame, ("mforce",))
    if not mforce_finite:
        reasons.append("non_finite_mforce")
    elif maximum_mforce is not None:
        metrics["max_mforce"] = maximum_mforce
        if policy.max_mforce is not None and maximum_mforce > policy.max_mforce:
            reasons.append("max_mforce_above_limit")

    maximum_spin, spin_finite = _maximum_norm(frame, ("spin",))
    if not spin_finite:
        reasons.append("non_finite_spin")
    elif maximum_spin is not None:
        metrics["max_spin_magnitude"] = maximum_spin
        if (
            policy.max_spin_magnitude is not None
            and maximum_spin > policy.max_spin_magnitude
        ):
            reasons.append("max_spin_magnitude_above_limit")
    return tuple(dict.fromkeys(reasons)), metrics


def classify_trajectory(
    frames: Sequence[Atoms],
    reference: Atoms,
    *,
    process_completed: bool,
    policy: TrajectoryHealthPolicy | None = None,
    pre_failure_frames: int = 2,
    bad_tail_frames: int = 1,
) -> TrajectoryHealthReport:
    """Locate the first physically unhealthy frame and classify all windows."""

    if not frames:
        raise TrajectoryHealthError("trajectory health requires at least one frame")
    if pre_failure_frames < 0 or bad_tail_frames < 1:
        raise TrajectoryHealthError(
            "pre_failure_frames must be non-negative and bad_tail_frames at least 1"
        )
    policy = policy or TrajectoryHealthPolicy()
    reference_volume = float(abs(reference.get_volume()))
    if not np.isfinite(reference_volume) or reference_volume <= 0:
        raise TrajectoryHealthError(
            "trajectory health requires a finite positive reference volume"
        )

    first_bad: int | None = None
    reason_codes: tuple[str, ...] = ()
    bad_metrics: dict[str, float] = {}
    first_arrays = frames[0].arrays
    available_signals = {
        "min_distance_ratio",
        "volume_ratio",
    }
    if "nep_force" in first_arrays or "forces" in first_arrays:
        available_signals.add("force")
    if "mforce" in first_arrays:
        available_signals.add("mforce")
    if "spin" in first_arrays:
        available_signals.add("spin_magnitude")
    for index, frame in enumerate(frames):
        reasons, metrics = _frame_health(
            frame, reference_volume=reference_volume, policy=policy
        )
        if reasons:
            first_bad = index
            reason_codes = reasons
            bad_metrics = metrics
            break

    if first_bad is not None:
        bad_start = first_bad
    elif process_completed:
        bad_start = len(frames)
    else:
        bad_start = max(0, len(frames) - bad_tail_frames)
    pre_start = max(0, bad_start - pre_failure_frames)
    windows = tuple(
        "bad_tail"
        if index >= bad_start
        else "pre_failure"
        if index >= pre_start and bad_start < len(frames)
        else "stable_prefix"
        for index in range(len(frames))
    )
    counts = Counter(windows)
    unavailable_thresholds = []
    if policy.max_force is not None and "force" not in available_signals:
        unavailable_thresholds.append("max_force")
    if policy.max_mforce is not None and "mforce" not in available_signals:
        unavailable_thresholds.append("max_mforce")
    if (
        policy.max_spin_magnitude is not None
        and "spin_magnitude" not in available_signals
    ):
        unavailable_thresholds.append("max_spin_magnitude")
    first_bad_step = (
        None
        if first_bad is None
        else int(frames[first_bad].info.get("lammps_step", first_bad))
    )
    return TrajectoryHealthReport(
        process_completed=bool(process_completed),
        trajectory_completed=bool(process_completed and first_bad is None),
        frame_count=len(frames),
        first_bad_frame=first_bad,
        first_bad_step=first_bad_step,
        reason_codes=reason_codes,
        first_bad_metrics=bad_metrics,
        windows=windows,
        counts_by_window={name: counts[name] for name in sorted(counts)},
        thresholds=asdict(policy),
        available_signals=tuple(sorted(available_signals)),
        unavailable_thresholds=tuple(unavailable_thresholds),
    )


__all__ = [
    "TrajectoryHealthError",
    "TrajectoryHealthPolicy",
    "TrajectoryHealthReport",
    "classify_trajectory",
    "is_structure_reasonable",
]
