import pytest

from NepTrain.core.scenario import ScenarioLadder, ScenarioMaturityError


def _ladder() -> ScenarioLadder:
    return ScenarioLadder(
        {
            "smoke_passed": 10,
            "short_stable": 40,
            "long_stable": 160,
            "production_ready": 640,
        }
    )


def test_scenario_ladder_promotes_one_level_at_a_time():
    ladder = _ladder()
    first = ladder.schedule(
        ["structure-a"],
        [300.0],
        pressure=0.0,
        generation=1,
        seed=1,
        limit=1,
    )

    assert first[0].previous_level == "untested"
    assert first[0].target_level == "smoke_passed"
    assert first[0].steps == 10

    history = ladder.record(first, accepted=True)
    second = ladder.schedule(
        ["structure-a"],
        [300.0],
        pressure=0.0,
        generation=2,
        seed=2,
        limit=1,
        history=history,
    )

    assert second[0].previous_level == "smoke_passed"
    assert second[0].target_level == "short_stable"
    assert second[0].steps == 40


def test_new_scenario_is_prioritized_without_skipping_smoke():
    ladder = _ladder()
    first = ladder.schedule(
        ["structure-a"],
        [300.0],
        pressure=0.0,
        generation=1,
        seed=1,
        limit=1,
    )
    history = ladder.record(first, accepted=True)
    attempts = ladder.schedule(
        ["structure-a"],
        [300.0, 500.0],
        pressure=0.0,
        generation=2,
        seed=2,
        limit=1,
        history=history,
    )

    assert attempts[0].temperature == 500.0
    assert attempts[0].target_level == "smoke_passed"
    assert attempts[0].steps == 10


def test_schedule_balances_temperature_and_structure_evidence():
    ladder = _ladder()
    attempts = ladder.schedule(
        ["structure-a", "structure-b"],
        [300.0, 500.0],
        pressure=0.0,
        generation=1,
        seed=7,
        limit=2,
    )

    assert {attempt.temperature for attempt in attempts} == {300.0, 500.0}
    assert {attempt.structure_id for attempt in attempts} == {
        "structure-a",
        "structure-b",
    }


def test_rejected_attempt_records_evidence_without_promotion():
    ladder = _ladder()
    attempt = ladder.schedule(
        ["structure-a"],
        [300.0],
        pressure=0.0,
        generation=1,
        seed=1,
        limit=1,
    )
    history = ladder.record(
        attempt,
        accepted=False,
        diagnostic={"current_model_force_rmse": 0.3},
        validation={"force_rmse": 0.4},
    )
    record = next(iter(history["scenarios"].values()))

    assert record["maturity"] == "untested"
    assert record["evidence"][0]["accepted"] is False
    assert record["evidence"][0]["diagnostic"]["current_model_force_rmse"] == 0.3


def test_failed_md_attempt_does_not_promote_after_validation_passes():
    ladder = _ladder()
    attempts = ladder.schedule(
        ["structure-a", "structure-b"],
        [300.0],
        pressure=0.0,
        generation=1,
        seed=4,
        limit=2,
    )
    completed = {
        attempts[0].scenario_id: True,
        attempts[1].scenario_id: False,
    }

    history = ladder.record(attempts, accepted=True, completed=completed)
    records = history["scenarios"]

    assert records[attempts[0].scenario_id]["maturity"] == "smoke_passed"
    assert records[attempts[1].scenario_id]["maturity"] == "untested"
    failed_evidence = records[attempts[1].scenario_id]["evidence"][0]
    assert failed_evidence["accepted"] is False
    assert failed_evidence["md_completed"] is False
    assert failed_evidence["validation_accepted"] is True


def test_maturity_config_rejects_typos_instead_of_silently_using_defaults():
    with pytest.raises(ScenarioMaturityError, match="short_stabel"):
        ScenarioLadder.from_campaign(
            {
                "initial_steps": 10,
                "maturity": {"levels": {"short_stabel": 40}},
            }
        )
