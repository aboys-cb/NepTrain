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


@pytest.mark.parametrize("key", ["nep_mforce", "mforce"])
def test_predicted_and_legacy_magnetic_force_health(key):
    frames = _frames(3)
    for frame in frames:
        frame.set_array(key, np.zeros((2, 3)))
    frames[2].arrays[key][0, 0] = 101.
    report = classify_trajectory(
        frames, frames[0], process_completed=True,
        policy=TrajectoryHealthPolicy(max_mforce=100.),
    )
    assert report.first_bad_step == 20
    assert report.reason_codes == ("max_mforce_above_limit",)
    assert "max_mforce" not in report.unavailable_thresholds


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


def _mixed_spin_frame():
    frame = Atoms('FeCoOY', positions=[[0, 0, 0], [3, 0, 0], [6, 0, 0], [9, 0, 0]],
                  cell=[15, 15, 15], pbc=True)
    frame.set_array('spin', np.array([[0., 0., 3.], [0., 0., 1.], [0., 0., 0.], [0., 0., 0.]]))
    return frame


def test_element_spin_ranges_exclude_collapsed_frame_and_tail_with_diagnostics():
    frames = [_mixed_spin_frame() for _ in range(4)]
    frames[2].arrays['spin'][0] = [0, 0, .42]
    frames[2].arrays['spin'][1] = [0, 0, 4]
    policy = TrajectoryHealthPolicy.from_mapping({'spin_magnitude_by_element': {
        'Fe': {'min': 1, 'max': 6}, 'Co': {'min': .5, 'max': 3.5},
    }})
    # DFT constraints must not exempt a site from a separately configured MD range.
    frames[2].set_array('spin_active', np.zeros(4, dtype=int))
    report = classify_trajectory(frames, frames[0], process_completed=True,
                                 policy=policy, pre_failure_frames=1)
    assert report.first_bad_frame == 2
    assert report.windows == ('stable_prefix', 'pre_failure', 'bad_tail', 'bad_tail')
    assert report.reason_codes == ('spin_magnitude_below_min:Fe', 'spin_magnitude_above_max:Co')
    data = report.to_dict()
    assert data['first_bad_spin_violation_counts'] == {'Fe': 1, 'Co': 1}
    assert data['first_bad_spin_violations'][0] == {
        'element': 'Fe', 'atom_number': 1, 'magnitude': .42, 'bound': 'min', 'limit': 1.,
    }
    assert 'Fe atom #1: spin magnitude 0.42 μB below minimum 1 μB' in report.failure_description()
    assert data['thresholds']['spin_magnitude_by_element']['Co']['max'] == 3.5


@pytest.mark.parametrize('ranges', [None, {}, {'Fe': {}}, {'Fe': {'min': None, 'max': None}}])
def test_omitted_or_null_element_limits_allow_zero_spin(ranges):
    frame = _mixed_spin_frame()
    frame.arrays['spin'][:] = 0
    policy = TrajectoryHealthPolicy.from_mapping({'spin_magnitude_by_element': ranges})
    report = classify_trajectory([frame], frame, process_completed=True, policy=policy)
    assert report.trajectory_completed


@pytest.mark.parametrize('bounds,spin,accepted', [
    ({'min': 1., 'max': 5.}, [0, 0, 1.], True),
    ({'min': 1., 'max': 5.}, [3, 4, 0], True),
    ({'min': 1., 'max': 5.}, [3, 4, 1], False),
    ({'min': 1.}, [0, 0, 30], True),  # Explicit entry replaces the global maximum.
    ({'max': 5.}, [0, 0, 0], True),
    ({'min': 0., 'max': 0.}, [0, 0, 0], True),
])
def test_element_bounds_use_vector_norm_and_inclusive_limits(bounds, spin, accepted):
    frame = _mixed_spin_frame()
    frame.arrays['spin'][0] = spin
    policy = TrajectoryHealthPolicy.from_mapping({'spin_magnitude_by_element': {'Fe': bounds}})
    report = classify_trajectory([frame], frame, process_completed=True, policy=policy)
    assert report.trajectory_completed is accepted


def test_global_spin_cap_still_applies_to_unlisted_elements_and_can_be_disabled():
    frame = _mixed_spin_frame()
    frame.arrays['spin'][1] = [0, 0, 21]
    settings = {'spin_magnitude_by_element': {'Fe': {'min': 1}}}
    report = classify_trajectory([frame], frame, process_completed=True,
                                 policy=TrajectoryHealthPolicy.from_mapping(settings))
    assert report.reason_codes == ('max_spin_magnitude_above_limit',)
    assert report.first_bad_spin_violations[0]['element'] == 'Co'
    settings['max_spin_magnitude'] = None
    assert classify_trajectory([frame], frame, process_completed=True,
                               policy=TrajectoryHealthPolicy.from_mapping(settings)).trajectory_completed


def test_element_ranges_report_missing_spin_and_ignore_absent_species():
    frame = _frames(1)[0]
    policy = TrajectoryHealthPolicy.from_mapping({'max_spin_magnitude': None,
        'spin_magnitude_by_element': {'Fe': {'min': 1}, 'Co': {'max': 3}}})
    report = classify_trajectory([frame], frame, process_completed=True, policy=policy)
    assert 'spin_magnitude_by_element.Fe' in report.unavailable_thresholds
    assert 'spin_magnitude_by_element.Co' not in report.unavailable_thresholds
    frame.set_array('spin', np.array([[0., 0., 2.], [np.nan, 0., 0.]]))
    report = classify_trajectory([frame], frame, process_completed=True, policy=policy)
    assert report.reason_codes == ('non_finite_spin',)


@pytest.mark.parametrize('ranges', [
    [], {'fe': {'min': 1}}, {'X': {'max': 1}}, {'Fe': 1},
    {'Fe': {'minimum': 1}}, {'Fe': {'min': -1}}, {'Fe': {'max': float('inf')}},
    {'Fe': {'min': float('nan')}}, {'Fe': {'max': True}}, {'Fe': {'min': 'oops'}},
    {'Fe': {'min': 4, 'max': 2}},
])
def test_element_spin_ranges_reject_invalid_configuration(ranges):
    with pytest.raises(TrajectoryHealthError, match='spin_magnitude_by_element'):
        TrajectoryHealthPolicy.from_mapping({'spin_magnitude_by_element': ranges})
