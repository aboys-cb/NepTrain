import pytest

from NepTrain.core.scenario import ScenarioLadder, ScenarioMaturityError


def _ladder(
    *,
    path=(300.0, 500.0, 700.0),
    production=(300.0, 700.0),
    replicas=None,
) -> ScenarioLadder:
    return ScenarioLadder(
        {
            "smoke_passed": 10,
            "short_stable": 40,
            "long_stable": 160,
            "production_ready": 640,
        },
        temperature_path=path,
        production_temperatures=production,
        replicas=replicas
        or {
            "smoke_passed": 1,
            "short_stable": 1,
            "long_stable": 1,
            "production_ready": 1,
        },
    )


def _record(
    ladder,
    attempts,
    *,
    history=None,
    accepted=True,
    completed=True,
    validation_accepted=False,
    model_improved=False,
    final_model_id="model-1",
):
    return ladder.record(
        attempts,
        completed={
            attempt.attempt_id: bool(completed) for attempt in attempts
        },
        diagnostic_accepted={
            attempt.attempt_id: bool(accepted) for attempt in attempts
        },
        history=history,
        validation_accepted=validation_accepted,
        model_improved=model_improved,
        novelty_converged=True,
        final_model_id=final_model_id,
    )


def test_temperature_path_unlocks_in_order_and_duration_runs_in_parallel():
    ladder = _ladder()
    first = ladder.schedule(
        ["structure-a", "structure-b"],
        pressure=0.0,
        generation=1,
        seed=1,
        limit=4,
        model_id="model-1",
    )

    assert len(first) == 2
    assert {attempt.temperature for attempt in first} == {300.0}
    assert {attempt.target_level for attempt in first} == {"smoke_passed"}

    history = _record(ladder, first)
    second = ladder.schedule(
        ["structure-a", "structure-b"],
        pressure=0.0,
        generation=2,
        seed=2,
        limit=4,
        model_id="model-1",
        history=history,
    )

    assert {(attempt.temperature, attempt.target_level) for attempt in second} == {
        (500.0, "smoke_passed"),
        (300.0, "short_stable"),
    }


def test_intermediate_temperature_stops_after_smoke():
    ladder = _ladder()
    first = ladder.schedule(
        ["structure-a"],
        pressure=0.0,
        generation=1,
        seed=1,
        limit=1,
        model_id="model-1",
    )
    history = _record(ladder, first)
    second = ladder.schedule(
        ["structure-a"],
        pressure=0.0,
        generation=2,
        seed=2,
        limit=2,
        model_id="model-1",
        history=history,
    )
    temperature_probe = tuple(
        attempt for attempt in second if attempt.temperature == 500.0
    )
    history = _record(ladder, temperature_probe, history=history)
    third = ladder.schedule(
        ["structure-a"],
        pressure=0.0,
        generation=3,
        seed=3,
        limit=3,
        model_id="model-1",
        history=history,
    )

    assert any(
        attempt.temperature == 700.0
        and attempt.target_level == "smoke_passed"
        for attempt in third
    )
    assert not any(
        attempt.temperature == 500.0
        and attempt.target_level != "smoke_passed"
        for attempt in third
    )


def test_long_and_production_levels_require_multiple_replicas():
    ladder = _ladder(
        path=(300.0,),
        production=(300.0,),
        replicas={
            "smoke_passed": 1,
            "short_stable": 1,
            "long_stable": 2,
            "production_ready": 3,
        },
    )
    history = None
    for generation, expected_target, expected_count in (
        (1, "smoke_passed", 1),
        (2, "short_stable", 1),
        (3, "long_stable", 2),
        (4, "production_ready", 3),
    ):
        attempts = ladder.schedule(
            ["structure-a"],
            pressure=0.0,
            generation=generation,
            seed=generation,
            limit=3,
            model_id="model-1",
            history=history,
        )
        assert len(attempts) == expected_count
        assert {attempt.target_level for attempt in attempts} == {
            expected_target
        }
        assert len({attempt.seed for attempt in attempts}) == expected_count
        history = _record(
            ladder,
            attempts,
            history=history,
            validation_accepted=generation == 4,
        )

    record = next(iter(history["scenarios"].values()))
    assert record["maturity"] == "production_ready"
    assert record["verified_model_id"] == "model-1"


def test_failed_attempt_does_not_promote_and_is_retried_at_same_frontier():
    ladder = _ladder(path=(300.0, 500.0), production=(500.0,))
    first = ladder.schedule(
        ["structure-a"],
        pressure=0.0,
        generation=1,
        seed=1,
        limit=1,
        model_id="model-1",
    )
    history = _record(
        ladder, first, accepted=False, completed=False, model_improved=True
    )
    retry = ladder.schedule(
        ["structure-a"],
        pressure=0.0,
        generation=2,
        seed=2,
        limit=1,
        model_id="model-2",
        history=history,
    )

    record = next(iter(history["scenarios"].values()))
    assert record["maturity"] == "untested"
    assert retry[0].temperature == 300.0
    assert retry[0].target_level == "smoke_passed"


def test_history_records_each_attempts_own_diagnostic_metrics():
    ladder = _ladder(path=(300.0,), production=(300.0,))
    attempts = ladder.schedule(
        ["structure-a", "structure-b"],
        pressure=0.0,
        generation=1,
        seed=1,
        limit=2,
        model_id="model-1",
    )
    force_rmse = {
        attempts[0].attempt_id: 0.2,
        attempts[1].attempt_id: 0.8,
    }

    history = ladder.record(
        attempts,
        completed={attempt.attempt_id: True for attempt in attempts},
        diagnostic_accepted={
            attempt.attempt_id: True for attempt in attempts
        },
        diagnostic={
            "current_model_force_rmse": 99.0,
            "attempt_metrics": {
                attempt_id: {"force_rmse": value}
                for attempt_id, value in force_rmse.items()
            },
        },
        validation_accepted=True,
        model_improved=False,
        novelty_converged=True,
        final_model_id="model-1",
    )

    for attempt in attempts:
        evidence = history["scenarios"][attempt.scenario_id]["evidence"][0]
        assert evidence["diagnostic"]["force_rmse"] == force_rmse[
            attempt.attempt_id
        ]


def test_production_readiness_is_rechecked_after_model_changes():
    ladder = _ladder(path=(300.0,), production=(300.0,))
    history = None
    for generation in range(1, 5):
        attempts = ladder.schedule(
            ["structure-a"],
            pressure=0.0,
            generation=generation,
            seed=generation,
            limit=1,
            model_id="model-1",
            history=history,
        )
        history = _record(ladder, attempts, history=history)

    assert ladder.production_ready(
        ["structure-a"],
        pressure=0.0,
        model_id="model-1",
        history=history,
    )
    assert not ladder.production_ready(
        ["structure-a"],
        pressure=0.0,
        model_id="model-2",
        history=history,
    )
    recheck = ladder.schedule(
        ["structure-a"],
        pressure=0.0,
        generation=5,
        seed=5,
        limit=1,
        model_id="model-2",
        history=history,
    )
    assert recheck[0].target_level == "production_ready"
    history = _record(
        ladder,
        recheck,
        history=history,
        accepted=False,
        final_model_id="model-2",
    )
    assert history["counts_by_maturity"] == {"long_stable": 1}
    assert next(iter(history["scenarios"].values()))["maturity"] == (
        "production_ready"
    )


def test_novel_structures_do_not_block_duration_progression():
    ladder = _ladder(path=(300.0,), production=(300.0,))
    first = ladder.schedule(
        ["structure-a"],
        pressure=0.0,
        generation=1,
        seed=1,
        limit=1,
        model_id="model-1",
    )
    history = ladder.record(
        first,
        completed={first[0].attempt_id: True},
        diagnostic_accepted={first[0].attempt_id: True},
        validation_accepted=False,
        model_improved=True,
        novelty_converged=False,
        final_model_id="model-2",
        novelty_converged_by_attempt={first[0].attempt_id: False},
    )

    progressed = ladder.schedule(
        ["structure-a"],
        pressure=0.0,
        generation=2,
        seed=2,
        limit=1,
        model_id="model-2",
        history=history,
    )

    assert history["counts_by_maturity"] == {"smoke_passed": 1}
    assert progressed[0].target_level == "short_stable"
    assert progressed[0].previous_level == "smoke_passed"


def test_condition_coverage_is_recorded_without_controlling_maturity():
    ladder = _ladder()
    first = ladder.schedule(
        ["structure-a"],
        pressure=0.0,
        generation=1,
        seed=1,
        limit=3,
        model_id="model-1",
    )
    history = _record(ladder, first)
    second = ladder.schedule(
        ["structure-a"],
        pressure=0.0,
        generation=2,
        seed=2,
        limit=3,
        model_id="model-2",
        history=history,
    )
    by_temperature = {attempt.temperature: attempt for attempt in second}
    history = ladder.record(
        second,
        completed={attempt.attempt_id: True for attempt in second},
        diagnostic_accepted={
            attempt.attempt_id: True for attempt in second
        },
        history=history,
        validation_accepted=False,
        model_improved=True,
        novelty_converged=False,
        final_model_id="model-3",
        novelty_converged_by_attempt={
            by_temperature[300.0].attempt_id: True,
            by_temperature[500.0].attempt_id: False,
        },
    )

    third = ladder.schedule(
        ["structure-a"],
        pressure=0.0,
        generation=3,
        seed=3,
        limit=3,
        model_id="model-3",
        history=history,
    )

    assert {(item.temperature, item.target_level) for item in third} == {
        (300.0, "long_stable"),
        (700.0, "smoke_passed"),
    }


def test_stuck_legacy_history_is_reconciled_before_scheduling():
    ladder = _ladder(path=(600.0, 800.0), production=(600.0, 800.0))
    smoke = ladder.schedule(
        ["structure-a"],
        pressure=0.0,
        generation=1,
        seed=1,
        limit=1,
        model_id="model-1",
    )
    history = ladder.record(
        smoke,
        completed={smoke[0].attempt_id: True},
        diagnostic_accepted={smoke[0].attempt_id: True},
        validation_accepted=False,
        model_improved=True,
        novelty_converged=False,
        final_model_id="model-2",
        novelty_converged_by_attempt={smoke[0].attempt_id: False},
    )
    record = next(iter(history["scenarios"].values()))
    record["maturity"] = "untested"
    history["counts_by_maturity"] = {"untested": 1}

    resumed = ladder.schedule(
        ["structure-a"],
        pressure=0.0,
        generation=2,
        seed=2,
        limit=2,
        model_id="model-2",
        history=history,
    )

    assert {(item.temperature, item.target_level) for item in resumed} == {
        (600.0, "short_stable"),
        (800.0, "smoke_passed"),
    }


def test_maturity_config_rejects_typos_instead_of_silently_using_defaults():
    with pytest.raises(ScenarioMaturityError, match="short_stabel"):
        ScenarioLadder.from_sampling(
            {
                "conditions": {"temperature_path": [300]},
                "progression": {
                    "steps": {
                        "smoke_passed": 10,
                        "short_stabel": 40,
                        "long_stable": 160,
                        "production_ready": 640,
                    }
                },
            }
        )


def test_temperature_path_must_be_monotonic_and_targets_must_be_on_path():
    with pytest.raises(ScenarioMaturityError, match="strictly"):
        _ladder(path=(300.0, 700.0, 500.0))
    with pytest.raises(ScenarioMaturityError, match="subset"):
        _ladder(path=(300.0, 500.0), production=(700.0,))
