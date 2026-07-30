"""Best-effort Feishu progress notifications for persistent workflows."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import urllib3

from .persistence import atomic_write_json
from .workflow import _generation_science
from .workflow_workspace import WorkflowWorkspace


_TERMINAL_STATES = {
    "complete",
    "rejected",
    "failed",
    "stalled",
    "budget_exhausted",
    "coverage_exhausted",
}
_MAX_DELIVERY_ATTEMPTS = 3
_STOP = object()


@dataclass(frozen=True)
class DeliveryResult:
    ok: bool
    detail: str


@dataclass(frozen=True)
class NotificationEvent:
    event_id: str
    text: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sign(timestamp: str, secret: str) -> str:
    signing_key = f"{timestamp}\n{secret}".encode("utf-8")
    digest = hmac.new(signing_key, b"", hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def _response_code(payload: Mapping[str, Any]) -> Any:
    if "code" in payload:
        return payload["code"]
    return payload.get("StatusCode")


def send_feishu_text(
    settings: Mapping[str, Any],
    text: str,
    *,
    pool: Any | None = None,
) -> DeliveryResult:
    """Send one signed text message and normalize the Feishu response."""

    timestamp = str(int(time.time()))
    payload = {
        "timestamp": timestamp,
        "sign": _sign(timestamp, str(settings["secret"])),
        "msg_type": "text",
        "content": {"text": text},
    }
    timeout = float(settings.get("timeout_seconds", 5))
    client = pool if pool is not None else urllib3.PoolManager()
    try:
        response = client.request(
            "POST",
            str(settings["webhook"]),
            body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=urllib3.Timeout(connect=timeout, read=timeout),
            retries=False,
        )
    except Exception as error:
        return DeliveryResult(False, f"{type(error).__name__}: {error}")
    try:
        response_payload = json.loads(
            response.data.decode("utf-8", errors="replace")
        )
    except (AttributeError, json.JSONDecodeError) as error:
        return DeliveryResult(
            False,
            f"HTTP {response.status}: invalid JSON response ({error})",
        )
    if not isinstance(response_payload, Mapping):
        return DeliveryResult(
            False,
            f"HTTP {response.status}: response is not an object",
        )
    code = _response_code(response_payload)
    if response.status == 200 and code == 0:
        return DeliveryResult(True, "success")
    message = response_payload.get("msg") or response_payload.get(
        "StatusMessage", "unknown error"
    )
    return DeliveryResult(
        False,
        f"HTTP {response.status}, code={code}: {message}",
    )


def doctor_probe(
    settings: Mapping[str, Any],
    *,
    workflow_id: str,
    project_path: Path | str | None = None,
    pool: Any | None = None,
) -> DeliveryResult:
    identity = _workflow_identity(workflow_id, project_path)
    return send_feishu_text(
        settings,
        "\n".join(
            (
                "🩺 [NepTrain] Doctor 通知链路验证",
                *identity,
                "结果：签名校验与消息投递成功",
                "说明：这不是工作流进度消息",
            )
        ),
        pool=pool,
    )


def _workflow_identity(
    workflow_id: str,
    workflow_path: Path | str | None,
) -> tuple[str, ...]:
    lines = [f"任务：{workflow_id}"]
    if workflow_path is not None:
        lines.append(f"路径：{Path(workflow_path).resolve()}")
    return tuple(lines)


def _number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _metric(value: Any, *, scale: float, unit: str) -> str:
    if value is None:
        return "-"
    return f"{float(value) * scale:.4g} {unit}"


def _sampling_range(record: Mapping[str, Any]) -> str | None:
    stages = record.get("stages", {})
    if not isinstance(stages, Mapping):
        return None
    explore = stages.get("explore", {})
    metrics = (
        explore.get("metrics", {})
        if isinstance(explore, Mapping)
        else {}
    )
    if not isinstance(metrics, Mapping):
        return None
    temperatures = sorted(
        {
            float(temperature)
            for temperature in metrics.get("scenario_temperatures", ())
        }
    )
    if not temperatures:
        temperatures = sorted(
            {
                float(temperature)
                for route in metrics.get("routes", ())
                if isinstance(route, Mapping)
                for temperature in route.get("temperatures", ())
            }
        )
    steps = sorted(
        {
            int(value)
            for value in metrics.get("scenario_steps", ())
        }
    )
    parts = []
    if temperatures:
        if len(temperatures) <= 5:
            values = "/".join(f"{value:g}" for value in temperatures)
        else:
            values = (
                f"{temperatures[0]:g}–{temperatures[-1]:g}"
                f"（{len(temperatures)} 个温度点）"
            )
        parts.append(f"温度：{values} K")
    if steps:
        parts.append(f"最长时长：{max(steps):,} steps")
    return "；".join(parts) or None


def _plan_mapping(plan: Any) -> Mapping[str, Any]:
    if isinstance(plan, Mapping):
        return plan
    if is_dataclass(plan) and not isinstance(plan, type):
        return asdict(plan)
    raise TypeError("workflow generation plan is not a mapping or dataclass")


def _generation_event(
    workflow_id: str,
    total_generations: int,
    plan: Any,
    record: Mapping[str, Any],
    *,
    workflow_path: Path | str | None = None,
) -> NotificationEvent:
    science = _generation_science(_plan_mapping(plan), record)
    generation = int(science["generation"])
    sampling = science["sampling"]
    training = science["training"]
    quality = science["quality"]["validation_rmse"]
    stages = record.get("stages", {})
    label_metrics = (
        stages.get("label", {}).get("metrics", {})
        if isinstance(stages, Mapping)
        else {}
    )
    failed_labels = int(label_metrics.get("failed_batch_count", 0) or 0)
    failed_sources = int(sampling.get("failed_source_count", 0) or 0)
    partial = failed_labels > 0 or failed_sources > 0
    icon = "⚠️" if partial else "✅"
    outcome = "部分成功并已接受" if partial else "本轮已接受"
    lines = [
        f"{icon} [NepTrain] G{generation}/{total_generations} 完成",
        *_workflow_identity(workflow_id, workflow_path),
        (
            "采样："
            f"{_number(sampling.get('candidate_count'))} 候选 → "
            f"{_number(sampling.get('candidate_count_after_deduplication'))} 去重 → "
            f"{_number(sampling.get('selected_count'))} 选中"
        ),
        (
            f"标注：{_number(sampling.get('labeled_count'))} 成功"
            + (f"，{failed_labels} 批失败" if failed_labels else "")
        ),
        "验证 RMSE："
        f"E={_metric(quality.get('energy_rmse'), scale=1, unit='eV')}，"
        f"F={_metric(quality.get('force_rmse'), scale=1000, unit='meV/Å')}，"
        f"V={_metric(quality.get('virial_rmse'), scale=1, unit='eV')}，"
        f"M={_metric(quality.get('mforce_rmse'), scale=1000, unit='meV/μB')}",
        f"结论：{outcome}",
    ]
    training_line = (
        "训练集："
        f"{_number(training.get('before_count'))} → "
        f"{_number(training.get('after_count'))}"
    )
    if training.get("added_count") is not None:
        training_line += f"（+{_number(training.get('added_count'))}）"
    lines.insert(3, training_line)
    if failed_sources:
        lines.insert(2, f"采样异常：{failed_sources} 个 source 失败")
    sampling_range = _sampling_range(record)
    if sampling_range:
        lines.insert(2, f"采样范围：{sampling_range}")
    model = training.get("active_model_sha256")
    if model:
        lines.insert(-1, f"模型：{str(model)[:12]}")
    return NotificationEvent(
        f"generation:{generation}:accepted",
        "\n".join(lines),
    )


def _terminal_event(
    workflow_id: str,
    total_generations: int,
    controller_state: Mapping[str, Any],
    tick: Any,
    *,
    workflow_path: Path | str | None = None,
) -> NotificationEvent:
    state = str(getattr(tick, "state", controller_state.get("state", "failed")))
    current = controller_state.get("current")
    current = current if isinstance(current, Mapping) else {}
    generation = getattr(tick, "generation", None) or current.get("generation")
    stage = getattr(tick, "stage", None) or current.get("stage")
    attempt = current.get("attempt", 1)
    detail = (
        getattr(tick, "detail", "")
        or controller_state.get("reason")
        or state
    )
    icons = {
        "complete": "🎉",
        "failed": "❌",
        "rejected": "❌",
        "stalled": "⚠️",
        "budget_exhausted": "⚠️",
        "coverage_exhausted": "⚠️",
    }
    labels = {
        "complete": "流程完成",
        "failed": "流程失败",
        "rejected": "评估未通过",
        "stalled": "流程停滞",
        "budget_exhausted": "轮次预算耗尽",
        "coverage_exhausted": "采样覆盖耗尽",
    }
    lines = [
        f"{icons.get(state, 'ℹ️')} [NepTrain] {labels.get(state, state)}",
        *_workflow_identity(workflow_id, workflow_path),
        f"进度：{generation or total_generations}/{total_generations}",
    ]
    if stage:
        lines.append(f"位置：G{generation or '-'} / {stage}")
    lines.append(f"原因：{detail}")
    if state != "complete":
        lines.append("已完成的科学结果和失败证据均保留，可检查后 resume")
    event_id = (
        "workflow:complete"
        if state == "complete"
        else f"terminal:{state}:{generation or 0}:{stage or 'none'}:{attempt}"
    )
    return NotificationEvent(event_id, "\n".join(lines))


class WorkflowNotificationWorker:
    """Render and deliver workflow events without network I/O on the controller."""

    def __init__(
        self,
        settings: Mapping[str, Any],
        workspace: WorkflowWorkspace,
        *,
        pool: Any | None = None,
    ):
        self.settings = dict(settings)
        self.workspace = workspace
        self.pool = pool
        self._queue: queue.Queue[Any] = queue.Queue()
        self._lock = threading.Lock()
        self._queued: set[str] = set()
        self._state = self._load_state()
        self._thread: threading.Thread | None = None
        self._closed = False

    def _load_state(self) -> dict[str, Any]:
        path = self.workspace.notification_state
        if not path.is_file():
            return {"version": 1, "provider": "feishu", "events": {}}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(value, Mapping)
                or not isinstance(value.get("events"), Mapping)
            ):
                raise ValueError("notification state must contain events")
            return {
                "version": 1,
                "provider": "feishu",
                "events": dict(value["events"]),
            }
        except Exception as error:
            self._warn(f"cannot read notification state: {error}")
            return {"version": 1, "provider": "feishu", "events": {}}

    @staticmethod
    def _warn(detail: str) -> None:
        print(
            f"NepTrain notification warning: {detail}",
            file=sys.stderr,
            flush=True,
        )

    def _ensure_thread(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="neptrain-feishu-notifications",
            daemon=False,
        )
        self._thread.start()

    def enqueue(self, event: NotificationEvent) -> None:
        """Queue immediately; all delivery and persistence happens off-thread."""

        try:
            with self._lock:
                if self._closed or event.event_id in self._queued:
                    return
                record = self._state["events"].get(event.event_id, {})
                if record.get("state") == "delivered":
                    return
                if int(record.get("attempts", 0)) >= _MAX_DELIVERY_ATTEMPTS:
                    return
                self._queued.add(event.event_id)
                self._ensure_thread()
                self._queue.put_nowait(event)
        except Exception as error:
            self._warn(f"cannot queue progress event: {error}")

    def observe(
        self,
        *,
        workflow_id: str,
        plans: Sequence[Any],
        controller_state: Mapping[str, Any],
        tick: Any,
    ) -> None:
        """Translate committed ledger/controller state into idempotent events."""

        try:
            ledger = {}
            if self.workspace.ledger.is_file():
                value = json.loads(
                    self.workspace.ledger.read_text(encoding="utf-8")
                )
                if isinstance(value, Mapping):
                    ledger = value
            generations = ledger.get("generations", {})
            if not isinstance(generations, Mapping):
                generations = {}
            terminal_state = str(getattr(tick, "state", ""))
            for plan in plans:
                plan_value = _plan_mapping(plan)
                generation = int(plan_value["generation"])
                record = generations.get(str(generation))
                if (
                    not isinstance(record, Mapping)
                    or record.get("complete") is not True
                    or record.get("accepted") is not True
                ):
                    continue
                self.enqueue(
                    _generation_event(
                        workflow_id,
                        len(plans),
                        plan,
                        record,
                        workflow_path=self.workspace.root,
                    )
                )
            if terminal_state in _TERMINAL_STATES:
                self.enqueue(
                    _terminal_event(
                        workflow_id,
                        len(plans),
                        controller_state,
                        tick,
                        workflow_path=self.workspace.root,
                    )
                )
        except Exception as error:
            self._warn(f"cannot build progress report: {error}")

    def _persist(self) -> None:
        try:
            atomic_write_json(self.workspace.notification_state, self._state)
        except Exception as error:
            self._warn(f"cannot persist notification state: {error}")

    def _deliver(self, event: NotificationEvent) -> None:
        result = send_feishu_text(
            self.settings,
            event.text,
            pool=self.pool,
        )
        with self._lock:
            previous = self._state["events"].get(event.event_id, {})
            record = {
                "state": "delivered" if result.ok else "failed",
                "attempts": int(previous.get("attempts", 0)) + 1,
                "created_at": previous.get("created_at", _now()),
                "last_attempt_at": _now(),
            }
            if result.ok:
                record["delivered_at"] = _now()
            else:
                record["last_error"] = result.detail
            self._state["events"][event.event_id] = record
            self._queued.discard(event.event_id)
            self._persist()
        if not result.ok:
            self._warn(f"{event.event_id} delivery failed: {result.detail}")

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                try:
                    self._deliver(item)
                except Exception as error:
                    self._warn(f"unexpected delivery failure: {error}")
                    with self._lock:
                        self._queued.discard(item.event_id)
            finally:
                self._queue.task_done()

    def close(self) -> None:
        """Stop after queued messages; the controller never waits here."""

        try:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                if self._thread is not None:
                    self._queue.put_nowait(_STOP)
        except Exception as error:
            self._warn(f"cannot stop notification thread: {error}")


def build_workflow_notifier(
    config: Mapping[str, Any],
    workspace: WorkflowWorkspace,
) -> WorkflowNotificationWorker | None:
    try:
        settings = config.get("notifications", {}).get("feishu", {})
        if not settings:
            return None
        return WorkflowNotificationWorker(settings, workspace)
    except Exception as error:
        WorkflowNotificationWorker._warn(
            f"cannot initialize Feishu notifications: {error}"
        )
        return None


__all__ = [
    "DeliveryResult",
    "NotificationEvent",
    "WorkflowNotificationWorker",
    "build_workflow_notifier",
    "doctor_probe",
    "send_feishu_text",
]
