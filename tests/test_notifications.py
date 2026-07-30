from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

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
    assert "任务：Fe-spin" in text
    assert f"路径：{project.resolve()}" in text


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
                    "accepted": True,
                    "added_training_count": 9,
                    "energy_rmse": 0.01,
                    "force_rmse": 0.2,
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
    assert "任务：Fe" in event.text
    assert "路径：/work/Fe-run" in event.text
    assert "40 候选" in event.text
    assert "1 批失败" in event.text
    assert "部分成功并已接受" in event.text
    assert "F=200 meV/Å" in event.text
    assert "温度：300/500 K" in event.text
    assert "最长时长：400 steps" in event.text


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
    assert "任务：Fe" in event.text
    assert "路径：/work/Fe-run" in event.text
    assert "OUT_OF_MEMORY" in event.text
    assert "resume" in event.text


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
