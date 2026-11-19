"""Cadence edits keep scientific history but never reuse execution identities."""
import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import read as ase_read, write as ase_write

from NepTrain.core.iteration import GenerationPlan, StageContext
from NepTrain.core.md import MdResult
from NepTrain.core.sampling_route import load_sampling_routes
from NepTrain.core.scenario import ScenarioLadder, ScenarioMaturityError
from NepTrain.core.scientific_data import structure_id
from NepTrain.core.workflow_iteration import WorkflowIterationAdapter, WorkflowRuntime, _previous_routes


TEMPLATE = """units metal
timestep 0.001
fix int all npt temp {{ temperature }} {{ temperature }} 0.1 tri 0 0 1.0
thermo 20
thermo_style custom step pe vol
dump trajectory all custom 20 {{ trajectory_file }} id type x y z fx fy fz
dump_modify trajectory sort id
run {{ steps }}
"""


@pytest.fixture
def route_config(tmp_path):
    structure = tmp_path / 'seed.xyz'
    template = tmp_path / 'route.in'
    ase_write(structure, Atoms('Fe', positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True))
    template.write_text(TEMPLATE)
    return {'routes': [{
        'id': 'npt', 'structures': [str(structure)],
        'template_path': str(template),
        'conditions': {'temperature_path': [50, 100], 'pressure': 0},
        'progression': {'steps': {'smoke_passed': 10, 'short_stable': 40,
                                 'long_stable': 160, 'production_ready': 640}},
    }], 'selection': {'max_selected': 4, 'novelty': 'auto'},
       'candidate_pool': {'health': {}, 'pre_failure_frames': 2, 'bad_tail_frames': 1}}


def route(config, tmp_path):
    return load_sampling_routes(config, base_dir=tmp_path)[0]


@pytest.mark.parametrize('before,after', [
    ('custom 20 ', 'custom 1000 '), ('thermo 20', 'thermo 100'),
    ('dump_modify trajectory every 20', 'dump_modify trajectory every 1000'),
])
def test_numeric_cadence_changes_only_execution_identity(route_config, tmp_path, before, after):
    path = Path(route_config['routes'][0]['template_path'])
    text = TEMPLATE + 'dump_modify trajectory every 20\n'
    path.write_text(text)
    old = route(route_config, tmp_path)
    path.write_text(text.replace(before, after))
    new = route(route_config, tmp_path)
    assert new.fingerprint != old.fingerprint
    assert new.template_sha256 != old.template_sha256
    assert new.maturity_fingerprint == old.maturity_fingerprint


@pytest.mark.parametrize('before,after', [
    ('0.001', '0.002'), ('all npt', 'all nvt'), ('0.1 tri', '0.2 tri'),
    ('tri 0 0 1.0', 'tri 0 0 5.0'), ('fx fy fz', 'fx fy'),
    ('thermo_style custom step pe vol', 'thermo_style custom step pe'),
    ('thermo 20', 'thermo 0'), ('custom 20', 'custom v_stride'),
    ('sort id', 'sort off'), ('run {{ steps }}', 'run 100'),
])
def test_other_template_edits_reset_maturity(route_config, tmp_path, before, after):
    old = route(route_config, tmp_path)
    Path(route_config['routes'][0]['template_path']).write_text(TEMPLATE.replace(before, after))
    assert route(route_config, tmp_path).maturity_fingerprint != old.maturity_fingerprint


@pytest.mark.parametrize('text', [
    'print """\nthermo 20\n"""\n',
    'fix a all print 1 &\nthermo 20\n',
    'variable stride equal 20\nthermo v_stride\n',
])
def test_unsupported_syntax_is_not_normalized(route_config, tmp_path, text):
    path = Path(route_config['routes'][0]['template_path'])
    path.write_text(text)
    old = route(route_config, tmp_path)
    path.write_text(text.replace('20', '1000'))
    assert route(route_config, tmp_path).maturity_fingerprint != old.maturity_fingerprint


@pytest.mark.parametrize('change', ['structure', 'pressure', 'temperature', 'steps', 'replicas', 'route_id'])
def test_route_policy_edits_reset_maturity(route_config, tmp_path, change):
    old = route(route_config, tmp_path)
    value = route_config['routes'][0]
    if change == 'structure':
        ase_write(value['structures'][0], Atoms('Fe2', positions=[[0, 0, 0], [2, 0, 0]]))
    elif change == 'pressure':
        value['conditions']['pressure'] = 1
    elif change == 'temperature':
        value['conditions']['temperature_path'] = [50, 200]
    elif change == 'steps':
        value['progression']['steps']['smoke_passed'] = 20
    elif change == 'replicas':
        value['progression']['replicas'] = {'smoke_passed': 2}
    else:
        value['id'] = 'other'
    assert route(route_config, tmp_path).maturity_fingerprint != old.maturity_fingerprint


def _history(old):
    ladder = ScenarioLadder.from_sampling({'conditions': old.conditions, 'progression': old.progression})
    attempts = ladder.schedule(['seed-sha'], route_id=old.route_id,
        route_fingerprint=old.fingerprint, template_sha256=old.template_sha256,
        pressure=0, generation=1, seed=1, limit=4, model_id='model')
    history = ladder.record(attempts, completed={a.attempt_id: True for a in attempts},
        diagnostic_accepted={a.attempt_id: True for a in attempts},
        diagnostic={'attempt_progression_confident': {a.attempt_id: True for a in attempts}},
        validation_accepted=None, model_improved=False, novelty_converged=False, final_model_id='model')
    return ladder, {'route_fingerprint': old.fingerprint, 'template_sha256': old.template_sha256,
                    'maturity_fingerprint': old.maturity_fingerprint, 'history': history}


def test_legacy_history_fails_closed_only_when_inputs_change(route_config, tmp_path):
    old = route(route_config, tmp_path)
    ladder, previous = _history(old)
    del previous['maturity_fingerprint']
    assert old.maturity_history(previous, ladder) == previous['history']
    Path(route_config['routes'][0]['template_path']).write_text(TEMPLATE.replace('custom 20', 'custom 1000'))
    assert route(route_config, tmp_path).maturity_history(previous, ladder) is None


def test_rebinding_is_repeatable_preserves_sources_and_rejects_corruption(route_config, tmp_path):
    old = route(route_config, tmp_path)
    ladder, previous = _history(old)
    original = copy.deepcopy(previous)
    path = Path(route_config['routes'][0]['template_path'])
    for cadence in (1000, 2000):
        path.write_text(TEMPLATE.replace('custom 20', f'custom {cadence}'))
        new = route(route_config, tmp_path)
        history = new.maturity_history(previous, ladder)
        assert history == new.maturity_history(previous, ladder)
        assert history['counts_by_maturity'] == {'smoke_passed': 1}
        record = next(iter(history['scenarios'].values()))
        assert record['route_fingerprint'] == new.fingerprint
        assert record['evidence'][0]['route_fingerprint'] == old.fingerprint
        assert record['evidence'][0]['template_sha256'] == old.template_sha256
        assert record['evidence'][0]['attempt_id'] == next(iter(original['history']['scenarios'].values()))['evidence'][0]['attempt_id']
        assert previous['history'] != history  # original namespace was not modified
        previous = {**previous, 'route_fingerprint': new.fingerprint,
                    'template_sha256': new.template_sha256, 'history': history}
    next(iter(previous['history']['scenarios'].values()))['structure_id'] = 'corrupt'
    path.write_text(TEMPLATE.replace('custom 20', 'custom 3000'))
    with pytest.raises(ScenarioMaturityError, match='identity changed'):
        route(route_config, tmp_path).maturity_history(previous, ladder)


def test_workflow_planning_explore_and_update_keep_temperature_frontier(route_config, tmp_path):
    """Exercise all three consumers, including deterministic task planning."""
    initial = Path(route_config['routes'][0]['structures'][0])
    observed = []

    def fake_md(request, backend):
        observed.append((request.temperature, request.steps))
        request.output_dir.mkdir(parents=True, exist_ok=True)
        frame = request.atoms.copy()
        frame.info['md_window'] = 'stable_prefix'
        ase_write(request.output_file, [frame], format='extxyz')
        return MdResult(backend, request.output_file, request.output_dir, 'cpu')

    config = {'training': {'backend': 'gpumd'}, 'md': {'backend': 'lammps'},
              'labeling': {'backend': 'toy'}, 'sampling': route_config}
    previous = {}
    old_sources = None
    for generation in (1, 2, 3):
        if generation > 1:
            unchanged_context = StageContext(
                generation=generation, generation_dir=tmp_path,
                plan=GenerationPlan(generation, generation, 4),
                artifacts={'model': initial}, previous_artifacts=previous,
                stage_dir=tmp_path / 'unchanged')
            unchanged = adapter.plan_explore_attempts(unchanged_context)
            Path(route_config['routes'][0]['template_path']).write_text(
                TEMPLATE.replace('custom 20', f'custom {generation * 1000}'))
        adapter = WorkflowIterationAdapter(config, initial_training=initial, runtime=WorkflowRuntime(md=fake_md))
        directory = tmp_path / f'g{generation}'
        directory.mkdir()
        context = StageContext(generation=generation, generation_dir=directory,
            plan=GenerationPlan(generation, generation, 4), artifacts={'model': initial},
            previous_artifacts=previous, stage_dir=directory / 'explore')
        planned = adapter.plan_explore_attempts(context)
        outcome = adapter._explore(context)
        plan = json.loads(outcome.artifacts['scenario_plan'].read_text())
        attempts = plan['routes'][0]['attempts']
        assert {a['attempt_id'] for a in attempts} == {a['attempt_id'] for a in planned}
        if generation > 1:
            assert {(a['temperature'], a['target_level'], a['steps']) for a in attempts} == {
                (a['temperature'], a['target_level'], a['steps']) for a in unchanged}
        if generation == 2:
            assert 100.0 in {a['temperature'] for a in attempts}
            assert (50.0, 'smoke_passed') not in {(a['temperature'], a['target_level']) for a in attempts}
        selection = directory / 'selection.json'
        selection.write_text('{}')
        update_context = StageContext(generation=generation, generation_dir=directory,
            plan=context.plan, artifacts={**outcome.artifacts, 'selection_result': selection},
            previous_artifacts=previous, stage_dir=directory / 'update')
        history, _ = adapter._record_route_histories(update_context,
            diagnostic={'attempt_accepted': {a['attempt_id']: True for a in attempts},
                        'attempt_progression_confident': {a['attempt_id']: True for a in attempts}},
            validation_metrics=None, evidence_validation=None, validation_accepted=None,
            model_improved=False, novelty_converged=False, final_model_id='model')
        record = history['routes']['npt']
        assert record['maturity_fingerprint'] == adapter.routes[0].maturity_fingerprint
        evidence = [e for s in record['history']['scenarios'].values() for e in s['evidence']]
        if old_sources:
            assert all(e in evidence for e in old_sources)
        old_sources = copy.deepcopy(evidence)
        history_path = directory / 'maturity.json'
        history_path.write_text(json.dumps(history))
        previous = {'scenario_maturity': history_path}


@pytest.mark.parametrize('portable', [False, True])
def test_selected_frame_without_label_is_not_treated_as_no_new_sample(route_config, tmp_path, portable):
    route_config['routes'][0]['conditions']['temperature_path'] = [50]
    initial = Path(route_config['routes'][0]['structures'][0])
    config = {'training': {'backend': 'gpumd'}, 'md': {'backend': 'lammps'},
              'labeling': {'backend': 'toy'}, 'sampling': route_config}
    adapter = WorkflowIterationAdapter(config, initial_training=initial)
    route = adapter.routes[0]
    ladder = adapter.scenario_ladders[route.route_id]
    attempt = ladder.schedule(
        [structure_id(ase_read(initial))], route_id=route.route_id,
        route_fingerprint=route.fingerprint,
        template_sha256=route.template_sha256, pressure=0,
        generation=1, seed=1, limit=1, model_id='model',
    )[0]
    scenario_plan = tmp_path / 'scenario-plan.json'
    scenario_plan.write_text(json.dumps({
        'version': 3, 'model_id': 'model', 'routes': [{
            'route_id': route.route_id,
            'route_fingerprint': route.fingerprint,
            'attempts': [asdict(attempt)],
            'completed': {attempt.attempt_id: True},
            'structure_ids': [attempt.structure_id], 'pressure': 0,
        }],
    }))
    selected = tmp_path / 'selected.xyz'
    frame = ase_read(initial)
    frame.info['scenario_attempt_id'] = attempt.attempt_id
    ase_write(selected, frame)
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({
        'remaining_novelty_by_condition': {'R=npt|T=50|P=0': 0},
        'resolved_completion_coverage_threshold': 1,
    }))
    artifacts = {'scenario_plan': scenario_plan, 'selection_result': selection}
    if portable:
        policy = json.loads(selection.read_text())
        policy['selected_attempt_ids'] = [attempt.attempt_id]
        selection.write_text(json.dumps(policy))
    else:
        artifacts['selected_input'] = selected
    context = StageContext(
        generation=1, generation_dir=tmp_path,
        plan=GenerationPlan(1, 1, 4),
        artifacts=artifacts,
        previous_artifacts={}, stage_dir=tmp_path / 'update',
    )
    result, _ = adapter._record_route_histories(
        context, diagnostic={'attempt_accepted': {}},
        validation_metrics=None, evidence_validation=None,
        validation_accepted=None, model_improved=False,
        novelty_converged=True, final_model_id='model',
    )
    record = next(iter(result['routes']['npt']['history']['scenarios'].values()))
    assert record['maturity'] == 'untested'
    assert record['evidence'][-1]['accepted'] is None
    assert record['evidence'][-1]['deferred_reason'] == 'missing_labels'

def test_legacy_failure_location_recovered_by_attempt_identity(tmp_path):
    previous = tmp_path / 'maturity.json'
    md = tmp_path / 'md-attempts.json'
    evidence = [{'attempt_id': 'a'}, {'attempt_id': 'b'}, {'attempt_id': 'c'}]
    previous.write_text(json.dumps({'version': 3, 'routes': {'r': {'history': {
        'scenarios': {'s': {'evidence': evidence}},
    }}}}))
    md.write_text(json.dumps({'attempts': [
        {'scenario_attempt_id': 'b', 'completed': False, 'last_step': 300,
         'health': {'first_hard_bad_step': 240}},
        {'scenario_attempt_id': 'a', 'completed': False, 'last_step': 100,
         'health': {'halt': {'step': 123}}},
        {'scenario_attempt_id': 'c', 'completed': False, 'last_step': 80},
    ]}))
    routes = _previous_routes({'scenario_maturity': previous, 'md_attempts': md})
    patched = routes['r']['history']['scenarios']['s']['evidence']
    assert [item.get('failure_step') for item in patched] == [123, 240, None]
    assert 'failure_step' not in previous.read_text()


def test_scoped_pause_survives_idle_route_and_explicit_release(route_config, tmp_path):
    from NepTrain.core.scenario_diagnosis import health_policy_hash
    config = {"training": {}, "labeling": {}, "md": {}, "sampling": route_config}
    initial = Path(route_config["routes"][0]["structures"][0])
    adapter = WorkflowIterationAdapter(config, initial_training=initial)
    route = adapter.routes[0]
    ladder = adapter.scenario_ladders[route.route_id]
    opts = dict(route_id=route.route_id, route_fingerprint=route.fingerprint,
                template_sha256=route.template_sha256, pressure=0, seed=1, limit=10)
    attempts = ladder.schedule(["a", "b"], model_id="m1", generation=1, **opts)
    hist = ladder.record(attempts, completed={a.attempt_id: True for a in attempts},
                         diagnostic_accepted={a.attempt_id: True for a in attempts},
                         validation_accepted=None, model_improved=True, novelty_converged=True,
                         final_model_id="m1")
    sid = attempts[0].scenario_id
    paused = hist["scenarios"][sid]
    paused["diagnosis"] = {"status": "paused_for_diagnosis", "paused_generation": 1,
                           "observations": [], "reason": "repeated_failure_signature",
                           "baseline": {"health_hash": health_policy_hash(config)}}
    prior = {"route_fingerprint": route.fingerprint, "history": hist}
    def available():
        return ladder.schedule(["a", "b"], model_id="m2", generation=2,
                               history=adapter._route_history(route, prior, ladder), **opts)
    assert {a.structure_id for a in available()} == {"a", "b"} - {paused["structure_id"]}
    assert not ladder.production_ready(["a", "b"], route_id=route.route_id,
                                       route_fingerprint=route.fingerprint, pressure=0,
                                       model_id="m2", history=hist)
    adapter.config["_diagnosis_releases"] = {sid: 1}
    assert {a.structure_id for a in available()} == {"a", "b"}
    assert paused["maturity"] == "smoke_passed"


@pytest.mark.parametrize("fail", [False, True])
def test_final_model_canaries_merge_each_worker_and_bind_the_model(route_config, tmp_path, fail):
    from dataclasses import replace
    from NepTrain.core.generation_policy import ACTIVE_LEARNING_FINALIZATION_STAGES
    from NepTrain.core.workflow_iteration import WorkflowIterationError
    initial = Path(route_config["routes"][0]["structures"][0])
    config = {"training": {}, "labeling": {}, "md": {}, "sampling": route_config}
    def md(request, backend):
        request.output_dir.mkdir(parents=True, exist_ok=True)
        ase_write(request.output_file, request.atoms)
        return MdResult(backend, request.output_file, request.output_dir, "cpu",
                        completed=not (fail and request.temperature == 100), last_step=request.steps)
    adapter = WorkflowIterationAdapter(config, initial_training=initial, runtime=WorkflowRuntime(md=md))
    context = StageContext(generation=5, generation_dir=tmp_path, plan=GenerationPlan(5, 31, 4),
                           generation_kind="finalization", stage_sequence=ACTIVE_LEARNING_FINALIZATION_STAGES,
                           artifacts={"model": initial}, previous_artifacts={})
    specs = adapter.plan_explore_attempts(context)
    assert {(s["temperature"], s["steps"]) for s in specs} == {(50, 40), (100, 40)}
    outcomes = [adapter.run_stage("explore", replace(context, stage_dir=tmp_path / s["attempt_id"],
                stage_input={"attempt_ids": [s["attempt_id"]], "allow_empty": True})) for s in specs]
    with pytest.raises(WorkflowIterationError, match="incomplete"):
        adapter.merge_explore_outcomes(context, outcomes[:1])
    route = adapter.routes[0]
    ladder = adapter.scenario_ladders[route.route_id]
    history = ladder.empty_history()
    for outcome in outcomes:
        attempt = json.loads(outcome.artifacts["md_attempts"].read_text())["attempts"][0]
        sid = attempt["scenario_id"]
        # A later failed probe cannot erase the earlier certified batch.
        evidence = [{"generation": g, "model_id": "old-model", "target_level": "production_ready",
                     "steps": 640, "accepted": g == 1}
                    for g in ((1, 1, 2) if attempt["temperature"] == 100 else (1,))]
        history["scenarios"][sid] = {"scenario_id": sid, "maturity": "long_stable",
            "temperature": attempt["temperature"], "evidence": evidence}
    maturity = tmp_path / "maturity.json"
    maturity.write_text(json.dumps({"version": 3, "routes": {route.route_id: {
        "route_fingerprint": route.fingerprint, "history": history,
        "production_status": {"minimum_successful_replicas": 2}}}}))
    merged = adapter.merge_explore_outcomes(replace(context, previous_artifacts={"scenario_maturity": maturity}), outcomes)
    assert merged.metrics["workflow_converged"] is (not fail)
    checks = {c["temperature"]: c for c in merged.metrics["final_model_checks"]}
    assert checks[100]["historical_model_sha256"] == "old-model"
    assert (checks[100]["historical_steps"], checks[100]["historical_replicas"]) == (640, 2)
    assert checks[50]["historical_model_sha256"] is None and checks[50]["historical_replicas"] == 1
    assert checks[100]["trajectory"].startswith("generations/0005/explore/calculations/")
    report = merged.artifacts["final_check_report"].read_text()
    assert checks[100]["trajectory"] in report and "640 steps × 2" in report
    assert set(merged.artifacts) == {"signals", "scenario_plan", "md_attempts", "final_check_report"}


def test_update_pauses_one_route_keeps_other_route_running_then_reports_diagnosis(route_config, tmp_path):
    from NepTrain.core.generation_policy import ACTIVE_LEARNING_ACQUISITION_STAGES
    from NepTrain.core.dft.toy import ToyTeacher
    from dataclasses import replace

    other = copy.deepcopy(route_config["routes"][0])
    other["id"] = "healthy"
    route_config["routes"].append(other)
    initial = Path(other["structures"][0])
    training = tmp_path / "train.xyz"
    ase_write(training, ToyTeacher("ordinary").label(ase_read(initial)))
    def md(request, backend):
        request.output_dir.mkdir(parents=True, exist_ok=True)
        failed = request.route_id == "npt"
        frames = [request.atoms.copy() for _ in range(3)]
        for i, f in enumerate(frames):
            f.info.update(md_step=i, md_window="stable_prefix")
        ase_write(request.output_file, frames)
        health = request.output_dir / "trajectory-health.json"
        health.write_text(json.dumps({"halt": {"step": 2, "reason_code": "volume_ratio_above_max"}} if failed else {}))
        return MdResult(backend, request.output_file, request.output_dir, "cpu",
                        completed=not failed, health_report=health, last_step=2 if failed else request.steps)
    adapter = WorkflowIterationAdapter({"md": {}, "sampling": route_config, "training": {}, "labeling": {}},
                                       initial_training=training, runtime=WorkflowRuntime(md=md))
    previous = {}
    for g in range(1, 12):
        work = tmp_path / f"g{g}"
        work.mkdir()
        model = work / "model"
        model.write_text(f"model {g}")
        ctx = StageContext(generation=g, generation_dir=work, plan=GenerationPlan(g, 11, 4),
                           artifacts={"model": model}, previous_artifacts=previous, generation_kind="acquisition",
                           stage_sequence=ACTIVE_LEARNING_ACQUISITION_STAGES)
        specs = adapter.plan_explore_attempts(ctx)
        if g > 3:
            assert all(s["route_id"] == "healthy" for s in specs)
        outcomes = [adapter.run_stage("explore", replace(ctx, stage_dir=work / s["attempt_id"],
                    stage_input={"attempt_ids": [s["attempt_id"]], "allow_empty": True})) for s in specs]
        explore = adapter.merge_explore_outcomes(ctx, outcomes)
        empty = work / "labels.xyz"
        empty.touch()
        diagnostic = work / "diagnostic.json"
        diagnostic.write_text(json.dumps({"attempt_accepted": {s["attempt_id"]: True for s in specs}}))
        simple = work / "empty.json"
        simple.write_text("{}")
        update = adapter.run_stage("update", replace(ctx, artifacts={**ctx.artifacts, **explore.artifacts,
            "training_input": training, "labeled": empty, "model_evaluation": simple,
            "acquisition_signals": diagnostic, "selection_result": simple}))
        previous = update.artifacts
        if g >= 3:
            assert len(update.metrics["paused_scenarios"]) == 1
            assert not update.metrics["production_ready"]
            paused = next(iter(update.metrics["paused_scenarios"].values()))
            assert paused["route"] == "npt" and paused["baseline"]["bad_step"] == 2
        if update.metrics["diagnosis_required"]:
            assert g > 3
            break
    else:
        pytest.fail("workflow still runnable after the healthy route completed")
