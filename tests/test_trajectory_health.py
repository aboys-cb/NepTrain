import numpy as np
import pytest
from ase import Atoms

from NepTrain.core.md.health import (
    TrajectoryHealthError,
    TrajectoryHealthPolicy,
    classify_trajectory,
)


def _frames(count: int = 5) -> list[Atoms]:
    result = []
    for index in range(count):
        frame = Atoms(
            "Fe2",
            positions=[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]],
            cell=[6.0, 6.0, 6.0],
            pbc=True,
        )
        frame.info["lammps_step"] = index * 10
        result.append(frame)
    return result


def test_physical_health_locates_first_bad_frame_and_keeps_boundary():
    frames = _frames()
    frames[3].positions[1] = [1.1, 1.0, 1.0]

    report = classify_trajectory(
        frames,
        frames[0],
        process_completed=True,
        pre_failure_frames=2,
        bad_tail_frames=1,
    )

    assert report.trajectory_completed is False
    assert report.first_bad_frame == 3
    assert report.first_bad_step == 30
    assert report.reason_codes == ("min_distance_ratio_below_min",)
    assert report.windows == (
        "stable_prefix",
        "pre_failure",
        "pre_failure",
        "bad_tail",
        "bad_tail",
    )
    assert report.first_bad_metrics["minimum_distance_ratio"] < 0.5
    assert report.unavailable_thresholds == (
        "max_force",
        "max_mforce",
        "max_spin_magnitude",
    )


def test_force_spike_marks_successful_process_as_unhealthy():
    frames = _frames(3)
    for frame in frames:
        frame.set_array("nep_force", np.zeros((2, 3)))
    frames[2].arrays["nep_force"][0, 0] = 101.0

    report = classify_trajectory(frames, frames[0], process_completed=True)

    assert report.process_completed is True
    assert report.trajectory_completed is False
    assert report.first_bad_step == 20
    assert report.reason_codes == ("max_force_above_limit",)
    assert "max_force" not in report.unavailable_thresholds


def test_nonzero_exit_without_physical_anomaly_uses_fallback_tail_window():
    frames = _frames()

    report = classify_trajectory(
        frames,
        frames[0],
        process_completed=False,
        pre_failure_frames=2,
        bad_tail_frames=1,
    )

    assert report.first_bad_frame is None
    assert report.windows[-3:] == ("pre_failure", "pre_failure", "bad_tail")


def test_health_policy_rejects_unknown_or_inverted_settings():
    with pytest.raises(TrajectoryHealthError, match="min_distnace"):
        TrajectoryHealthPolicy.from_mapping({"min_distnace": 0.5})
    with pytest.raises(TrajectoryHealthError, match="min_volume_ratio"):
        TrajectoryHealthPolicy.from_mapping(
            {"min_volume_ratio": 2.0, "max_volume_ratio": 1.0}
        )
