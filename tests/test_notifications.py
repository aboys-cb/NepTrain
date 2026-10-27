from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from NepTrain.core.notifications import (
    NotificationEvent,
    WorkflowNotificationWorker,
    _generation_event,
    _terminal_event,
    doctor_probe,
    send_feishu_text,
)
from NepTrain.core.workflow_workspace import WorkflowWorkspace


class FakePool:
    def __init__(self, payload=None):
        self.payload = payload or {"code": 0, "msg": "success"}
        self.requests = []

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        return SimpleNamespace(
            status=200,
            data=json.dumps(self.payload).encode("utf-8"),
        )


def _settings():
    return {
        "webhook": (
            "https://open.feishu.cn/open-apis/bot/v2/hook/test-token"
        ),
        "secret": "test-secret",
        "timeout_seconds": 1,
    }


def test_signed_feishu_text_uses_the_v2_payload_without_leaking_secret():
    pool = FakePool()

    result = send_feishu_text(_settings(), "hello", pool=pool)

    assert result.ok
    assert len(pool.requests) == 1
    method, url, options = pool.requests[0]
    assert method == "POST"
    assert url.endswith("/test-token")
    payload = json.loads(options["body"])
    assert payload["msg_type"] == "text"
    assert payload["content"]["text"] == "hello"
    assert payload["timestamp"].isdigit()
    assert payload["sign"]
    assert "test-secret" not in options["body"].decode("utf-8")


def test_doctor_probe_reports_feishu_application_error():
    pool = FakePool({"code": 19021, "msg": "sign match fail"})

    result = doctor_probe(_settings(), workflow_id="probe", pool=pool)

    assert not result.ok
    assert "19021" in result.detail


def test_doctor_probe_identifies_the_project(tmp_path: Path):
    pool = FakePool()
    project = tmp_path / "project.yaml"

    result = doctor_probe(
        _settings(),
        workflow_id="Fe-spin",
        project_path=project,
        pool=pool,
    )

    assert result.ok
    text = json.loads(pool.requests[0][2]["body"])["content"]["text"]
    assert "Workflow: Fe-spin" in text
    assert f"Path: {project.resolve()}" in text


def test_generation_report_contains_scientific_progress():
    plan = {
        "generation": 2,
        "max_selected": 80,
        "selection_novelty_threshold": 0,
        "completion_coverage_threshold": 0,
    }
    record = {
        "complete": True,
        "accepted": True,
        "stages": {
            "train": {"metrics": {"training_count": 100}},
            "explore": {
                "metrics": {
                    "candidate_count": 40,
                    "failed_source_count": 1,
                    "trajectory_health": {
                        "process_completed": 2, "process_failed": 0, "process_unknown": 0,
                        "stable_trajectories": 1, "unstable_trajectories": 1,
                        "usable_frames": 40, "spin_filtered_frames": 3, "bad_tail_frames": 0,
                    },
                    "routes": [
                        {"temperatures": [300, 500]},
                    ],
                    "scenario_temperatures": [300, 500],
                    "scenario_steps": [100, 400],
                }
            },
            "select": {
                "metrics": {
                    "candidate_count_after_deduplication": 30,
                    "selected_count": 10,
                }
            },
            "label": {
                "metrics": {
                    "labeled_count": 9,
                    "failed_batch_count": 1,
                }
            },
            "diagnose": {"metrics": {}},
            "merge": {"metrics": {"training_count": 109}},
            "retrain": {"metrics": {"training_count": 109}},
            "evaluate": {
                "metrics": {
                    "prediction_metric_basis": "per_atom_v1",
                    "accepted": True,
                    "added_training_count": 9,
                    "energy_rmse": 0.01,
                    "force_rmse": 0.2,
                    "mforce_rmse": 0.15,
                    "active_model_sha256": "a" * 64,
                }
            },
        },
    }

    event = _generation_event(
        "Fe",
        4,
        plan,
        record,
        workflow_path="/work/Fe-run",
    )

    assert event.event_id == "generation:2:accepted"
    assert "G2/4" in event.text
    assert "Workflow: Fe" in event.text
    assert "Path: /work/Fe-run" in event.text
    assert "MD runs: 2 completed, 0 interrupted" in event.text
    assert "Trajectory stability: 1 passed, 1 not established" in event.text
    assert "40 usable, 3 spin outliers excluded, 0 in quarantined tails" in event.text
    assert "Failed sampling sources" not in event.text
    assert "40 candidates" in event.text
    assert "failed batches: 1" in event.text
    assert "Generation completed with partial results" in event.text
    assert "E=10 meV/atom" in event.text
    assert "F=200 meV/Å" in event.text
    assert "M=150 meV/μB" in event.text
    assert "meV/spin unit" not in event.text
    assert "Temperature: 300/500 K" in event.text
    assert "Longest run: 400 steps" in event.text


def test_finalization_notification_does_not_claim_sampling_or_labeling():
    event = _generation_event(
        "Fe",
        4,
        {"generation": 3, "seed": 19, "max_selected": 100},
        {
            "kind": "finalization",
            "stage_sequence": ["train", "evaluate"],
            "complete": True,
            "accepted": True,
            "stages": {
                "train": {"metrics": {"training_count": 120}},
                "evaluate": {
                    "metrics": {
                        "prediction_metric_basis": "per_atom_v1",
                        "accepted": True,
                        "workflow_converged": True,
                        "energy_rmse": 0.01,
                        "force_rmse": 0.1,
                        "active_model_sha256": "b" * 64,
                    }
                },
            },
        },
    )

    assert "Final model ready" in event.text
    assert "Final generation: G3 (planned limit: G4)" in event.text
    assert "Final training set: 120 structures" in event.text
    assert "no MD, selection, or labeling" in event.text
    assert "Sampling: " not in event.text
    assert "Labeling: " not in event.text


def test_finalization_notification_omits_empty_quality_placeholders():
    event = _generation_event(
        "Fe",
        4,
        {"generation": 3, "max_selected": 100},
        {
            "kind": "finalization",
            "complete": True,
            "accepted": True,
            "stages": {
                "train": {"metrics": {"training_count": 120}},
                "evaluate": {
                    "metrics": {
                        "accepted": True,
                        "workflow_converged": True,
                        "active_model_sha256": "b" * 64,
                    }
                },
            },
        },
    )

    assert "RMSE" not in event.text
    assert "E=-" not in event.text


def test_complete_terminal_report_uses_actual_completed_generation():
    tick = SimpleNamespace(
        state="complete",
        generation=None,
        stage=None,
        detail="workflow converged after model generation 15",
    )
    state = {"state": "complete", "completed_generation": 15}

    event = _terminal_event("Fe", 17, state, tick)

    assert "Workflow completed" in event.text
    assert "Finished at: G15 (planned limit: G17)" in event.text
    assert "17/17" not in event.text
    assert "workflow converged" not in event.text


def test_terminal_report_is_stable_per_attempt():
    tick = SimpleNamespace(
        state="failed",
        generation=3,
        stage="label",
        detail="OUT_OF_MEMORY",
    )
    state = {"current": {"generation": 3, "stage": "label", "attempt": 2}}

    event = _terminal_event(
        "Fe",
        6,
        state,
        tick,
        workflow_path="/work/Fe-run",
    )

    assert event.event_id == "terminal:failed:3:label:2"
    assert "Workflow: Fe" in event.text
    assert "Path: /work/Fe-run" in event.text
    assert "OUT_OF_MEMORY" in event.text
    assert "resume" in event.text


@pytest.mark.parametrize(
    ("created", "display_time"),
    [
        ("2026-10-27T08:55:23.174284+00:00", "2026-10-27 16:55:23"),
        ("2026-12-31T20:00:00+00:00", "2027-01-01 04:00:00"),
        ("2026-10-27T01:55:23-07:00", "2026-10-27 16:55:23"),
    ],
)
def test_notification_displays_shanghai_event_time(
    tmp_path: Path, monkeypatch, created, display_time
):
    from NepTrain.core import notifications as module

    monkeypatch.setattr(module, "_now", lambda: created)
    workspace = WorkflowWorkspace.create(tmp_path / "workflow")
    pool = FakePool()
    worker = WorkflowNotificationWorker(_settings(), workspace, pool=pool)
    event = NotificationEvent("terminal:stalled:1:label:1", "Labeling blocked")
    assert worker.enqueue(event)
    _finish(worker)

    record = json.loads(workspace.notification_state.read_text())["events"][event.event_id]
    assert record["created_at"] == created
    expected = f"Labeling blocked\nEvent time: {display_time}"
    assert record["text"] == expected
    assert json.loads(pool.requests[0][2]["body"])["content"]["text"] == expected


def test_worker_does_not_block_controller_and_deduplicates(tmp_path: Path):
    entered = threading.Event()
    release = threading.Event()

    class SlowPool(FakePool):
        def request(self, method, url, **kwargs):
            entered.set()
            assert release.wait(2)
            return super().request(method, url, **kwargs)

    workspace = WorkflowWorkspace.create(tmp_path / "workflow")
    pool = SlowPool()
    worker = WorkflowNotificationWorker(_settings(), workspace, pool=pool)
    event = NotificationEvent("generation:1:accepted", "done")

    started = time.monotonic()
    worker.enqueue(event)
    worker.enqueue(event)
    elapsed = time.monotonic() - started

    assert elapsed < 0.1
    assert entered.wait(1)
    release.set()
    worker.close()
    assert worker._thread is not None
    worker._thread.join(2)
    assert not worker._thread.is_alive()
    assert len(pool.requests) == 1
    state = json.loads(
        workspace.notification_state.read_text(encoding="utf-8")
    )
    assert state["events"][event.event_id]["state"] == "delivered"


def test_notification_prioritizes_acquisition_and_explains_test_warning():
    event = _generation_event(
        "Fe",
        5,
        {"generation": 2, "max_selected": 10},
        {
            "kind": "acquisition",
            "stage_sequence": [
                "train",
                "validate",
                "explore",
                "select",
                "label",
                "evaluate",
                "update",
            ],
            "complete": True,
            "accepted": True,
            "stages": {
                "evaluate": {
                    "metrics": {
                        "prediction_metric_basis": "per_atom_v1",
                        "current_model_force_rmse": 0.08,
                    }
                },
                "validate": {
                    "metrics": {
                        "prediction_metric_basis": "per_atom_v1",
                        "force_rmse": 0.5,
                        "validation_warnings": [
                            "Optional test overlaps the training set; diagnostic only"
                        ],
                    }
                },
                "update": {
                    "metrics": {
                        "convergence_reasons": [
                            "Consecutive passing generations: 1/2. More sampling is needed."
                        ]
                    }
                },
            },
        },
    )
    assert "Prediction RMSE on new structures (before training on them)" in event.text
    assert "F=80 meV/Å" in event.text
    assert "F=500 meV/Å" not in event.text
    assert (
        "Note: Optional test overlaps the training set; diagnostic only" in event.text
    )
    assert "Convergence: Consecutive passing generations: 1/2" in event.text


def test_v3_notification_reports_merged_training_count():
    event = _generation_event(
        "Fe",
        8,
        {"generation": 1, "max_selected": 8},
        {
            "kind": "acquisition",
            "complete": True,
            "accepted": True,
            "stage_sequence": [
                "train",
                "validate",
                "explore",
                "select",
                "label",
                "evaluate",
                "update",
            ],
            "stages": {
                "train": {"metrics": {"training_count": 4}},
                "update": {"metrics": {"training_count": 12, "added_count": 8}},
            },
        },
    )
    assert "Training set: 4 → 12 (+8)" in event.text


@pytest.mark.parametrize("kind", ["acquisition", "finalization"])
@pytest.mark.parametrize("mforce", [None, 0.0, 0.15])
def test_notification_shows_magnetic_force_only_when_available(kind, mforce):
    final = kind == "finalization"
    prefix = "" if final else "current_model_"
    event = _generation_event(
        "Fe",
        3,
        {"generation": 1, "max_selected": 10},
        {
            "kind": kind,
            "stage_sequence": (
                ["train"]
                if final
                else ["train", "explore", "select", "label", "evaluate", "update"]
            ),
            "complete": True,
            "accepted": True,
            "stages": {
                "train" if final else "evaluate": {
                    "metrics": {
                        "prediction_metric_basis": "per_atom_v1",
                        prefix + "force_rmse": 0.1,
                        prefix + "mforce_rmse": mforce,
                    }
                }
            },
        },
    )
    assert "F=100 meV/Å" in event.text
    if mforce is None:
        assert "M=" not in event.text
        assert "μB" not in event.text
    else:
        assert f"M={mforce * 1000:g} meV/μB" in event.text


def _finish(worker):
    worker.close()
    if worker._thread is not None:
        worker._thread.join(5)
        assert not worker._thread.is_alive()


def _ledger(workspace, generations):
    workspace.ledger.write_text(json.dumps({'generations': generations}))


def _complete_generation(kind='acquisition'):
    return {'kind': kind, 'complete': True, 'accepted': True, 'stages': {
        'train': {'metrics': {'training_count': 12}},
        'evaluate': {'metrics': {'accepted': True, 'active_model_sha256': 'a' * 64}},
    }}


def test_terminal_delivery_retries_without_controller_polling(tmp_path, monkeypatch):
    from NepTrain.core import notifications as module
    monkeypatch.setattr(module, '_RETRY_DELAY_SECONDS', 0.01)
    class FlakyPool(FakePool):
        def request(self, *args, **kwargs):
            response = super().request(*args, **kwargs)
            if len(self.requests) == 1:
                raise TimeoutError('temporary outage')
            return response
    workspace = WorkflowWorkspace.create(tmp_path / 'run')
    pool = FlakyPool()
    worker = WorkflowNotificationWorker(_settings(), workspace, pool=pool)
    worker.enqueue(NotificationEvent('terminal:failed', 'Failed'))
    _finish(worker)
    record = json.loads(workspace.notification_state.read_text())['events']['terminal:failed']
    assert len(pool.requests) == 2
    assert record['state'] == 'delivered' and record['attempts'] == 2
    assert 'last_error' not in record


def test_failed_delivery_rearms_after_settings_change(tmp_path, monkeypatch):
    from NepTrain.core import notifications as module
    monkeypatch.setattr(module, '_RETRY_DELAY_SECONDS', 0.01)
    workspace = WorkflowWorkspace.create(tmp_path / 'run')
    bad = FakePool({'code': 19021, 'msg': 'bad signature'})
    worker = WorkflowNotificationWorker(_settings(), workspace, pool=bad)
    worker.enqueue(NotificationEvent('terminal:failed', 'Failed'))
    _finish(worker)
    assert len(bad.requests) == 3
    same = FakePool()
    worker = WorkflowNotificationWorker(_settings(), workspace, pool=same)
    worker.enqueue(NotificationEvent('terminal:failed', 'Failed'))
    _finish(worker)
    assert not same.requests
    good = FakePool()
    worker = WorkflowNotificationWorker({**_settings(), 'secret': 'corrected'}, workspace, pool=good)
    _finish(worker)
    assert len(good.requests) == 1
    record = json.loads(workspace.notification_state.read_text())['events']['terminal:failed']
    assert record['attempts'] == 4 and record['state'] == 'delivered'


def test_pending_delivery_survives_sender_restart(tmp_path, monkeypatch):
    workspace = WorkflowWorkspace.create(tmp_path / 'run')
    worker = WorkflowNotificationWorker(_settings(), workspace, pool=FakePool())
    monkeypatch.setattr(worker, '_ensure_thread', lambda: None)
    worker.enqueue(NotificationEvent('pending', 'Pending message'))
    assert json.loads(workspace.notification_state.read_text())['events']['pending']['text'].startswith('Pending message')
    _finish(worker)
    pool = FakePool()
    resumed = WorkflowNotificationWorker(_settings(), workspace, pool=pool)
    _finish(resumed)
    assert len(pool.requests) == 1


def test_overlapping_senders_preserve_records_and_send_once(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    class SlowPool(FakePool):
        def request(self, *args, **kwargs):
            entered.set()
            assert release.wait(3)
            return super().request(*args, **kwargs)
    workspace = WorkflowWorkspace.create(tmp_path / 'run')
    slow, fast = SlowPool(), FakePool()
    first = WorkflowNotificationWorker(_settings(), workspace, pool=slow)
    first.enqueue(NotificationEvent('first', 'First'))
    assert entered.wait(1)
    second = WorkflowNotificationWorker(_settings(), workspace, pool=fast)
    second.enqueue(NotificationEvent('first', 'First'))
    second.enqueue(NotificationEvent('second', 'Second'))
    first.close()
    release.set()
    _finish(first)
    _finish(second)
    events = json.loads(workspace.notification_state.read_text())['events']
    assert set(events) == {'first', 'second'}
    assert all(r['state'] == 'delivered' for r in events.values())
    assert len(slow.requests) + len(fast.requests) == 2


@pytest.mark.parametrize('corrupt', [False, True])
def test_first_enable_or_corruption_does_not_replay_history(tmp_path, corrupt):
    workspace = WorkflowWorkspace.create(tmp_path / 'run')
    _ledger(workspace, {'1': _complete_generation()})
    if corrupt:
        workspace.notification_state.write_text('{broken')
    pool = FakePool()
    worker = WorkflowNotificationWorker(_settings(), workspace, pool=pool)
    plans = [{'generation': 1, 'max_selected': 10}, {'generation': 2, 'max_selected': 10}]
    tick = SimpleNamespace(state='running')
    worker.observe(workflow_id='test', plans=plans, controller_state={}, tick=tick)
    assert not pool.requests
    _ledger(workspace, {'1': _complete_generation(), '2': _complete_generation()})
    worker.observe(workflow_id='test', plans=plans, controller_state={}, tick=tick)
    _finish(worker)
    assert len(pool.requests) == 1
    assert 'G2/2 complete' in json.loads(pool.requests[0][2]['body'])['content']['text']
    if corrupt:
        assert list(workspace.internal_dir.glob('notifications.corrupt-*.json'))


def test_bad_generation_report_does_not_suppress_current_failure(tmp_path):
    workspace = WorkflowWorkspace.create(tmp_path / 'run')
    pool = FakePool()
    worker = WorkflowNotificationWorker(_settings(), workspace, pool=pool)
    bad = _complete_generation()
    bad['stages']['explore'] = {'metrics': {'scenario_temperatures': ['invalid']}}
    _ledger(workspace, {'1': bad})
    worker.observe(workflow_id='test', plans=[{'generation': 1, 'max_selected': 10}],
                   controller_state={'current': {'generation': 2, 'stage': 'label'}},
                   tick=SimpleNamespace(state='failed', generation=2, stage='label', detail='DFT failed'))
    _finish(worker)
    assert len(pool.requests) == 1
    assert 'Labeling failed' in json.loads(pool.requests[0][2]['body'])['content']['text']


def test_final_completion_is_one_message_with_final_model(tmp_path):
    workspace = WorkflowWorkspace.create(tmp_path / 'run')
    pool = FakePool()
    worker = WorkflowNotificationWorker(_settings(), workspace, pool=pool)
    _ledger(workspace, {'1': _complete_generation('finalization')})
    worker.observe(workflow_id='test', plans=[{'generation': 1, 'max_selected': 10}],
                   controller_state={'completed_generation': 1}, tick=SimpleNamespace(state='complete'))
    _finish(worker)
    assert len(pool.requests) == 1
    text = json.loads(pool.requests[0][2]['body'])['content']['text']
    assert 'Workflow completed' in text
    assert 'Final training set: 12 structures' in text
    assert 'Model: aaaaaaaaaaaa' in text


def test_new_controller_failure_is_not_deduplicated_as_old_failure():
    tick = SimpleNamespace(state='failed', generation=3, stage='train', detail='controller error')
    base = {'current': {'generation': 3, 'stage': 'train', 'attempt': 1}, 'preserve_current_on_retry': True}
    first = _terminal_event('test', 8, {**base, 'notification_session': 'first'}, tick)
    second = _terminal_event('test', 8, {**base, 'notification_session': 'second'}, tick)
    assert first.event_id != second.event_id
    assert 'Controller failed' in first.text
    assert 'does not mean those jobs failed' in first.text


def test_terminal_generation_is_never_guessed_from_budget():
    tick = SimpleNamespace(state='stalled')
    unknown = _terminal_event('test', 8, {}, tick)
    known = _terminal_event('test', 8, {'last_generation': 3}, tick)
    assert 'Generation: unknown' in unknown.text and '8/8' not in unknown.text
    assert 'Generation: 3/8' in known.text


def test_monitoring_alert_is_delayed_and_recovery_is_sent_once(tmp_path, monkeypatch):
    from NepTrain.core import notifications as module
    workspace = WorkflowWorkspace.create(tmp_path / 'run')
    worker = WorkflowNotificationWorker(_settings(), workspace, pool=FakePool())
    events = []
    monkeypatch.setattr(worker, 'enqueue', events.append)
    clock = [1000.0]
    monkeypatch.setattr(module.time, 'time', lambda: clock[0])
    def observe(state):
        worker.observe(workflow_id='test', plans=[], controller_state={}, tick=SimpleNamespace(state=state, detail='SSH unavailable'))
    observe('degraded')
    clock[0] += 100
    observe('running')
    assert not events
    observe('degraded')
    clock[0] += 301
    observe('degraded')
    observe('degraded')
    observe('running')
    observe('running')
    assert len(events) == 2
    assert 'Job monitoring unavailable' in events[0].text
    assert 'Job monitoring restored' in events[1].text
    _finish(worker)


def test_notification_status_separates_history_from_current_health(tmp_path):
    from NepTrain.core.notifications import notification_summary
    workspace = WorkflowWorkspace.create(tmp_path / 'run')
    workspace.notification_state.write_text(json.dumps({'events': {
        'z-old': {'state': 'failed', 'last_error': 'old', 'last_attempt_at': '2026-01-01'},
        'a-new': {'state': 'failed', 'last_error': 'new', 'last_attempt_at': '2026-01-02'},
        'b-success': {'state': 'delivered', 'last_attempt_at': '2026-01-03', 'delivered_at': '2026-01-03'},
    }}))
    summary = notification_summary(_settings(), workspace)
    assert summary['state'] == 'ok' and summary['failed'] == 2
    assert summary['last_error'] == 'new'
    assert summary['last_success_at'] == '2026-01-03'
    assert notification_summary({}, workspace)['state'] == 'disabled'


def test_delivery_error_redacts_credentials():
    class BrokenPool:
        def request(self, *args, **kwargs):
            raise RuntimeError(_settings()['webhook'] + ' ' + _settings()['secret'])
    result = send_feishu_text(_settings(), 'hello', pool=BrokenPool())
    assert 'test-token' not in result.detail and 'test-secret' not in result.detail


def test_close_deadline_keeps_undelivered_outbox_for_next_sender(tmp_path, monkeypatch):
    from NepTrain.core import notifications as module
    monkeypatch.setattr(module, '_SHUTDOWN_GRACE_SECONDS', 0)
    entered, release = threading.Event(), threading.Event()
    class SlowFailingPool(FakePool):
        def request(self, *args, **kwargs):
            entered.set()
            assert release.wait(2)
            return super().request(*args, **kwargs)
    workspace = WorkflowWorkspace.create(tmp_path / 'run')
    pool = SlowFailingPool({'code': 1, 'msg': 'temporary'})
    worker = WorkflowNotificationWorker(_settings(), workspace, pool=pool)
    worker.enqueue(NotificationEvent('last', 'Final event'))
    assert entered.wait(1)
    worker.close()
    release.set()
    worker._thread.join(2)
    assert not worker._thread.is_alive()
    record = json.loads(workspace.notification_state.read_text())['events']['last']
    assert record['state'] == 'retrying' and record['text'].startswith('Final event')


def test_legacy_delivered_events_remain_deduplicated(tmp_path):
    workspace = WorkflowWorkspace.create(tmp_path / 'run')
    workspace.notification_state.write_text(json.dumps({'version': 1, 'events': {
        'generation:1:accepted': {'state': 'delivered', 'attempts': 1},
    }}))
    pool = FakePool()
    worker = WorkflowNotificationWorker(_settings(), workspace, pool=pool)
    worker.enqueue(NotificationEvent('generation:1:accepted', 'Already delivered'))
    _finish(worker)
    assert not pool.requests
    assert json.loads(workspace.notification_state.read_text())['version'] == 2


def test_transient_state_read_error_does_not_quarantine_valid_state(tmp_path, monkeypatch):
    workspace = WorkflowWorkspace.create(tmp_path / 'run')
    workspace.notification_state.write_text('{"version": 1, "events": {}}')
    original = Path.read_text
    def read(path, *args, **kwargs):
        if path == workspace.notification_state:
            raise OSError('temporary shared filesystem error')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    with pytest.raises(OSError):
        WorkflowNotificationWorker(_settings(), workspace, pool=FakePool())
    assert not list(workspace.internal_dir.glob('notifications.corrupt-*'))
    assert workspace.notification_state.read_bytes() == b'{"version": 1, "events": {}}'


def test_changed_credentials_do_not_inherit_inflight_failed_attempt(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class SlowFailingPool(FakePool):
        def request(self, *args, **kwargs):
            entered.set()
            assert release.wait(2)
            return super().request(*args, **kwargs)
    workspace = WorkflowWorkspace.create(tmp_path / 'run')
    first = WorkflowNotificationWorker(_settings(), workspace, pool=SlowFailingPool({'code': 1, 'msg': 'bad old key'}))
    first.enqueue(NotificationEvent('event', 'Event'))
    assert entered.wait(1)
    pool = FakePool()
    second = WorkflowNotificationWorker({**_settings(), 'secret': 'new'}, workspace, pool=pool)
    release.set()
    _finish(first)
    _finish(second)
    assert len(pool.requests) == 1
    assert json.loads(workspace.notification_state.read_text())['events']['event']['state'] == 'delivered'


def test_sampling_report_omits_absent_test_metrics():
    event = _generation_event('test', 3, {'generation': 1, 'max_selected': 10}, _complete_generation())
    assert 'RMSE' not in event.text


def test_server_error_redacts_credentials():
    settings = _settings()
    pool = FakePool({'code': 1, 'msg': settings['webhook'] + ' ' + settings['secret']})
    result = doctor_probe(settings, workflow_id='test', pool=pool)
    assert 'test-token' not in result.detail and 'test-secret' not in result.detail


def test_monitoring_recovery_is_retained_when_enqueue_fails(tmp_path, monkeypatch):
    workspace = WorkflowWorkspace.create(tmp_path / 'run')
    worker = WorkflowNotificationWorker(_settings(), workspace, pool=FakePool())
    with worker._transaction() as state:
        state['monitoring'] = {'since': 0, 'id': 'outage', 'alerted': True}
    monkeypatch.setattr(worker, 'enqueue', lambda event: False)
    worker._monitor('test', {}, SimpleNamespace(state='running'))
    assert json.loads(workspace.notification_state.read_text())['monitoring']['id'] == 'outage'
    events = []
    monkeypatch.setattr(worker, 'enqueue', events.append)
    worker._monitor('test', {}, SimpleNamespace(state='running'))
    worker._monitor('test', {}, SimpleNamespace(state='running'))
    assert len(events) == 1 and 'Job monitoring restored' in events[0].text
    _finish(worker)


def test_idle_sender_waits_without_rewriting_state_and_wakes_for_new_event(tmp_path):
    workspace = WorkflowWorkspace.create(tmp_path / 'run')
    pool = FakePool()
    worker = WorkflowNotificationWorker(_settings(), workspace, pool=pool)
    worker.enqueue(NotificationEvent('first', 'First'))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if json.loads(workspace.notification_state.read_text())['events']['first']['state'] == 'delivered':
            break
        time.sleep(0.01)
    time.sleep(0.1)  # Let the sender enter its idle wait after recording delivery.
    before = workspace.notification_state.stat().st_mtime_ns
    time.sleep(0.3)
    assert workspace.notification_state.stat().st_mtime_ns == before
    worker.enqueue(NotificationEvent('second', 'Second'))
    _finish(worker)
    assert len(pool.requests) == 2
