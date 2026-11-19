from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from NepTrain.core.scenario_diagnosis import crossing_step, observe, paired_metrics


LIMITS = {"energy_rmse": 1, "force_rmse": 1, "virial_rmse": 1}


def row(g, **overrides):
    return {"generation": g, "replica": 1, "attempt_id": f"a{g}", "model_sha256": f"m{g}",
            "model_updated": True, "steps": 400, "bad_step": 100, "completed": False,
            "signature": "volume_ratio_above_max", "timestep_ps": 0.001, "dump_interval": 20,
            "health_hash": "p1", "before": {}, "novel": False, **overrides}


def batch(record, g, pairs=None, **kwargs):
    observe(record, [row(g, **kwargs)], pairs or {}, LIMITS)
    return record["diagnosis"]


def errors(scale):
    return {**dict.fromkeys(LIMITS, scale), "pressure_bias_GPa": -2}


def pair(g, scale):
    return {f"a{g}": {"count": 2, "frame_ids": [f"{g}a", f"{g}b"],
                       "before": errors(3), "after": errors(scale),
                       "parent_model_sha256": f"m{g}", "child_model_sha256": f"m{g+1}"}}


def test_plateau_counts_distinct_updated_models_at_fixed_replica_index():
    record = {}
    observe(record, [row(1, replica=3, bad_step=350), row(1)], {}, LIMITS)
    state = batch(record, 2, model_updated=False, model_sha256="m1")
    assert state["baseline"]["bad_step"] == 100
    assert state["observations"][0]["replicas"] == 2
    assert "status" not in batch(record, 3)
    assert batch(record, 4)["reason"] == "repeated_failure_signature"


@pytest.mark.parametrize("change", [{"signature": "distance"}, {"timestep_ps": .002},
                                    {"health_hash": "p2"}, {"bad_step": 140}])
def test_comparison_resets_on_progress_or_changed_conditions(change):
    record = {}
    batch(record, 1)
    batch(record, 2)
    state = batch(record, 3, **change)
    assert "reason" not in state and state["baseline"]["generation"] == 3


def test_right_censor_only_clears_on_crossing_or_original_target():
    record = {}
    batch(record, 1)
    state = batch(record, 2, completed=True, steps=139, bad_step=None)
    assert state["observations"][-1]["right_censored"] and "baseline" in state
    assert "baseline" not in batch(record, 3, completed=True, steps=140, bad_step=None)
    batch(record, 4, steps=120)
    assert "baseline" not in batch(record, 5, completed=True, steps=120, bad_step=None)
    assert crossing_step(100, 1) == 126


@pytest.mark.parametrize("n,pressure,expected", [(1, True, False), (2, False, False), (2, True, True)])
def test_two_generation_shortcut_requires_distinct_frames_and_volume_evidence(n, pressure, expected):
    record = {}
    for g in (1, 2):
        values = errors(.8)
        if not pressure:
            del values["pressure_bias_GPa"]
        state = batch(record, g, before={"count": n, "frame_ids": [f"{g}-{i}" for i in range(n)],
                                        "metrics": values})
    assert (state.get("reason") == "low_pre_failure_errors") == expected


@pytest.mark.parametrize("scale,novel,reason", [(2.9, True, "poor_fit_on_same_frames"),
                                               (.7, False, "repeated_failure_signature"),
                                               (.7, True, None)])
def test_paired_fit_and_bounded_novelty_extension(scale, novel, reason):
    record = {}
    for g in range(1, 4):
        state = batch(record, g, pair(g - 1, scale), novel=novel)
    assert state.get("reason") == reason
    if reason is None:
        batch(record, 4, pair(3, scale), novel=True)
        assert batch(record, 5, pair(4, scale), novel=True)["status"] == "paused_for_diagnosis"


@pytest.mark.parametrize("orientation", [1, -1])
def test_pressure_bias_is_signed_frame_mean_with_unequal_volumes_and_atom_counts(orientation):
    frames = [Atoms("Fe", cell=np.diag([2*orientation, 2, 2])), Atoms("Fe2", cell=np.diag([2*orientation, 2, 4]))]
    reference = np.zeros((2, 3, 3))
    predicted = np.array([-np.eye(3), -3*np.eye(3)])
    def predict(*_):
        return SimpleNamespace(metrics={"virial_rmse": 1.}, comparisons={
            "virial": SimpleNamespace(predicted=predicted, reference=reference)})
    assert paired_metrics(predict, None, frames, None)["pressure_bias_GPa"] == pytest.approx(-.25 * 160.2176634)


def test_sibling_failure_sets_baseline_without_uncensoring_reference_or_clearing_it():
    record = {}
    for g in range(1, 4):
        observe(record, [row(g, completed=True, bad_step=None),
                         row(g, replica=2, attempt_id=f"s{g}", bad_step=100*g)], {}, LIMITS)
    state = record["diagnosis"]
    assert state["baseline"]["generation"] == 1
    assert state["baseline"]["bad_step"] == 100
    assert "reason" not in state
    latest = state["observations"][-1]
    assert latest["reference_right_censored"] and not latest["survival_comparable"]
    assert latest["reference_attempt_id"] == "a3" and latest["attempt_id"] == "s3"


def test_pool_all_failed_replica_labels_and_link_later_pairs_by_attempt_id():
    record = {}
    initial = [row(1), row(1, replica=2, attempt_id="r2", before={
        "count": 1, "frame_ids": ["x"], "metrics": errors(1)}),
        row(1, replica=3, attempt_id="r3", before={
        "count": 3, "frame_ids": ["y", "z", "w"], "metrics": {**errors(3), "pressure_bias_GPa": 2}})]
    observe(record, initial, {}, LIMITS)
    old = record["diagnosis"]["observations"][0]
    assert old["before"]["count"] == 4
    assert old["before"]["metrics"]["force_rmse"] == pytest.approx(np.sqrt(7))
    assert old["before"]["metrics"]["pressure_bias_GPa"] == 1
    pairs = {r["attempt_id"]: {"parent_model_sha256": "m1", "child_model_sha256": "m2",
             **{k: r["before"][k] for k in ("count", "frame_ids")},
             "before": r["before"]["metrics"], "after": errors(.1)} for r in initial[1:]}
    batch(record, 2, pairs)
    assert old["paired"]["before"] == old["before"]["metrics"]
    assert old["paired"]["frame_ids"] == ["x", "y", "z", "w"]
    assert [r["paired"]["count"] for r in old["batch"][1:]] == [1, 3]
