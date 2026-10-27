"""Best-effort Feishu progress notifications for persistent workflows."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import hmac
import json
from pathlib import Path
import shlex
import uuid
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import urllib3

from .persistence import atomic_write_json
from .workflow import _generation_science, _sampling_health_lines
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
_RETRY_DELAY_SECONDS = 1.0
_SHUTDOWN_GRACE_SECONDS = 30.0
_MONITOR_WARNING_SECONDS = 300.0


@dataclass(frozen=True)
class DeliveryResult:
    ok: bool
    detail: str


@dataclass(frozen=True)
class NotificationEvent:
    event_id: str
    text: str
    priority: int = 10


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
        return DeliveryResult(False, _redact(f"{type(error).__name__}: {error}", settings))
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
        _redact(f"HTTP {response.status}, code={code}: {message}", settings),
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
                "🩺 [NepTrain] Notification check",
                *identity,
                "Result: Notification delivered successfully.",
                "This is a connectivity check from doctor, not a workflow update.",
            )
        ),
        pool=pool,
    )


def _workflow_identity(
    workflow_id: str,
    workflow_path: Path | str | None,
) -> tuple[str, ...]:
    lines = [f"Workflow: {workflow_id}"]
    if workflow_path is not None:
        lines.append(f"Path: {Path(workflow_path).resolve()}")
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
                f" ({len(temperatures)} temperatures)"
            )
        parts.append(f"Temperature: {values} K")
    if steps:
        parts.append(f"Longest run: {max(steps):,} steps")
    return "; ".join(parts) or None


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
    generation_kind = science.get("kind")
    sampling = science["sampling"]
    training = science["training"]
    validation_quality = science["quality"]["validation_rmse"]
    acquisition_quality = science["quality"]["acquisition_rmse"]
    has_acquisition = any(value is not None for value in acquisition_quality.values())
    quality = acquisition_quality if has_acquisition else validation_quality
    quality_label = (
        "Prediction RMSE on new structures (before training on them)"
        if has_acquisition
        else "Optional test RMSE (diagnostic only)"
    )
    quality_text = (
        f"{quality_label}: "
        f"E={_metric(quality.get('energy_rmse'), scale=1000, unit='meV/atom')}, "
        f"F={_metric(quality.get('force_rmse'), scale=1000, unit='meV/Å')}, "
        f"V={_metric(quality.get('virial_rmse'), scale=1000, unit='meV/atom')}"
    )
    if quality.get("mforce_rmse") is not None:
        quality_text += (
            f", M={_metric(quality['mforce_rmse'], scale=1000, unit='meV/μB')}"
        )
    decision_lines = [
        *(
            f"Note: {warning}"
            for warning in science["quality"].get("validation_warnings", [])
        ),
        *(
            f"Convergence: {reason}"
            for reason in science["quality"].get("convergence_reasons", [])
        ),
    ]
    if science["quality"].get("finalization_pending"):
        decision_lines.append(
            "Next: Accuracy and production coverage requirements met; proceeding to final training."
        )

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
    outcome = (
        "Generation completed with partial results; review sampling stability and labeling diagnostics."
        if partial
        else "Generation completed. Convergence is assessed separately."
    )
    if generation_kind == "finalization":
        final_quality = any(value is not None for value in quality.values())
        lines = [
            "🎯 [NepTrain] Final model ready",
            *_workflow_identity(workflow_id, workflow_path),
            f"Final generation: G{generation} (planned limit: G{total_generations})",
            (
                "Final training set: "
                f"{_number(training.get('after_count'))} structures"
            ),
            "Result: Sampling criteria met and final training complete.",
            "This generation ran final training and model checks only, with no MD, selection, or labeling. Optional tests do not determine convergence.",
        ]
        model = training.get("active_model_sha256")
        if model:
            lines.insert(-2, f"Model: {str(model)[:12]}")
        if final_quality:
            lines.insert(-2, quality_text)
        return NotificationEvent(
            f"generation:{generation}:accepted",
            "\n".join([*lines, *decision_lines]),
        )
    lines = [
        f"{icon} [NepTrain] G{generation}/{total_generations} complete",
        *_workflow_identity(workflow_id, workflow_path),
        (
            "Sampling: "
            f"{_number(sampling.get('candidate_count'))} candidates → "
            f"{_number(sampling.get('candidate_count_after_deduplication'))} after deduplication → "
            f"{_number(sampling.get('selected_count'))} selected"
        ),
        (
            f"Labeling: {_number(sampling.get('labeled_count'))} structures labeled"
            + (f"; failed batches: {failed_labels}" if failed_labels else "")
        ),
        *([quality_text] if any(value is not None for value in quality.values()) else []),
        *_sampling_health_lines(sampling),
        f"Outcome: {outcome}",
    ]
    training_line = (
        "Training set: "
        f"{_number(training.get('before_count'))} → "
        f"{_number(training.get('after_count'))}"
    )
    if training.get("added_count") is not None:
        training_line += f" (+{_number(training.get('added_count'))})"
    lines.insert(3, training_line)
    if generation_kind == "acquisition" and (training.get("added_count") or 0) > 0:
        decision_lines.append("Next: Train with the new labeled structures before further sampling.")
    if failed_sources and not sampling.get("trajectory_health"):
        lines.insert(2, f"Sampling sources without stable completion: {failed_sources}")
    sampling_range = _sampling_range(record)
    if sampling_range:
        lines.insert(2, f"Sampling range: {sampling_range}")
    model = training.get("active_model_sha256")
    if model:
        lines.insert(-1, f"Model: {str(model)[:12]}")
    return NotificationEvent(
        f"generation:{generation}:accepted",
        "\n".join([*lines, *decision_lines]),
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
    generation = (
        getattr(tick, "generation", None)
        or (
            controller_state.get("completed_generation")
            if state == "complete"
            else None
        )
        or current.get("generation")
        or controller_state.get("last_generation")
    )
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
        "complete": "Workflow completed",
        "failed": "Workflow failed",
        "rejected": "Acceptance criteria not met",
        "stalled": "Workflow stalled",
        "budget_exhausted": "Sampling budget exhausted",
        "coverage_exhausted": "Sampling coverage exhausted",
    }
    if state == "failed" and controller_state.get("preserve_current_on_retry"):
        labels[state] = "Controller failed"
    elif state == "failed" and stage in {"train", "label", "explore"}:
        labels[state] = {"train": "Training failed", "label": "Labeling failed", "explore": "Sampling failed"}[stage]
    elif state == "stalled":
        labels[state] = {"label": "Labeling blocked", "explore": "Sampling blocked"}.get(stage, "Sampling stalled")
    lines = [
        f"{icons.get(state, 'ℹ️')} [NepTrain] {labels.get(state, state)}",
        *_workflow_identity(workflow_id, workflow_path),
    ]
    if state == "complete":
        if generation is None:
            lines.append(f"Planned limit: G{total_generations}")
        else:
            lines.append(
                f"Finished at: G{generation} (planned limit: G{total_generations})"
            )
        lines.append("Status: Final results saved; workflow finished successfully.")
    else:
        lines.append(
            f"Generation: {generation}/{total_generations}" if generation is not None
            else f"Generation: unknown (planned limit: G{total_generations})"
        )
    if stage and state != "complete":
        lines.append(f"Stopped at: G{generation or '-'} / {stage}")
    if state != "complete":
        lines.append(f"Reason: {detail}")
    if state != "complete":
        advice = {
            "budget_exhausted": "Review the remaining accuracy and coverage requirements, then use neptrain workflow extend to increase the sampling budget.",
            "coverage_exhausted": "No sampling tasks remain, but convergence is not established. Review the sampling conditions and workflow.convergence.",
            "stalled": (
                f"Next: neptrain workflow restart {shlex.quote(str(workflow_path))} --generation {generation} --from {stage} --tasks failed --dry-run"
                if workflow_path and generation and stage else
                "Review the sampling accuracy and coverage reports before choosing a restart stage or preparing a revised workflow."
            ),
        }.get(
            state,
            "Completed results and failure diagnostics are retained. Resolve the error in the logs, then use neptrain workflow resume to continue.",
        )
        if controller_state.get("preserve_current_on_retry"):
            lines.append("Submitted job handles are retained. A controller failure does not mean those jobs failed.")
        lines.append(advice)
    event_id = (
        "workflow:complete"
        if state == "complete"
        else f"terminal:{state}:{generation or 0}:{stage or 'none'}:{attempt}"
    )
    if state != "complete" and controller_state.get("notification_session"):
        event_id += f":session:{controller_state['notification_session']}"
    return NotificationEvent(event_id, "\n".join(lines), 0)


class WorkflowNotificationWorker:
    """A durable outbox shared by controllers and short-lived CLI actions.

    State locks cover only local reads/writes. A separate delivery lock allows
    one HTTP sender at a time without blocking the controller on network I/O.
    """

    def __init__(self, settings, workspace, *, pool=None):
        self.settings = dict(settings)
        self.workspace = workspace
        self.pool = pool
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread = None
        self._closed = False
        self._close_deadline = None
        self._fingerprint = hashlib.sha256(json.dumps(
            self.settings, sort_keys=True,
        ).encode()).hexdigest()
        with self._transaction() as state:
            if state.get("settings_fingerprint") != self._fingerprint:
                for record in state["events"].values():
                    if record.get("state") == "failed" and record.get("text"):
                        record.update(state="pending", delivery_attempts=0, next_attempt_at=0)
                state["settings_fingerprint"] = self._fingerprint
        if any(r.get("state") in {"pending", "retrying", "sending"}
               for r in self._state["events"].values()):
            self._ensure_thread()

    def _baseline(self):
        try:
            ledger = json.loads(self.workspace.ledger.read_text())
            return [str(g) for g, r in ledger.get("generations", {}).items()
                    if isinstance(r, Mapping) and r.get("complete") and r.get("accepted")]
        except (OSError, ValueError, AttributeError):
            return []

    @contextmanager
    def _transaction(self):
        path = self.workspace.notification_state
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                value = json.loads(path.read_text()) if path.exists() else None
                if value is not None and (
                    not isinstance(value, dict) or not isinstance(value.get("events"), dict)
                    or any(not isinstance(r, dict) for r in value["events"].values())
                ):
                    raise ValueError("invalid notification event records")
            except ValueError as error:
                self._warn(f"cannot read notification state: {error}")
                # Preserve the damaged file, and baseline existing generations
                # rather than flooding the recipient with historical reports.
                if path.exists():
                    path.replace(path.with_name(f"notifications.corrupt-{uuid.uuid4().hex}.json"))
                value = None
            if value is None:
                value = {"version": 2, "provider": "feishu", "events": {},
                         "observed_generations": self._baseline()}
            if "observed_generations" not in value:
                value["observed_generations"] = self._baseline()
            value["version"] = 2
            try:
                yield value
            finally:
                atomic_write_json(path, value)
                self._state = value

    @staticmethod
    def _warn(detail):
        print(f"NepTrain notification warning: {detail}", file=sys.stderr, flush=True)

    def _ensure_thread(self):
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="neptrain-feishu-notifications", daemon=False,
                )
                self._thread.start()
        self._wake.set()

    def enqueue(self, event):
        """Persist before scheduling delivery; never send HTTP on this thread."""
        try:
            if self._closed:
                return False
            with self._transaction() as state:
                old = state["events"].get(event.event_id, {})
                if old.get("state") == "delivered":
                    return True
                if old.get("state") == "failed" and old.get("text"):
                    return True  # Exhausted retries; a settings change can rearm it.
                if not old.get("text"):
                    created = _now()
                    state["events"][event.event_id] = {
                        "state": "pending", "attempts": old.get("attempts", 0),
                        "delivery_attempts": 0, "created_at": created,
                        "text": _redact(event.text, self.settings) + f"\nEvent time (UTC): {created}",
                        "priority": event.priority, "next_attempt_at": 0,
                    }
            self._ensure_thread()
            return True
        except Exception as error:
            self._warn(f"cannot queue progress event: {error}")
            return False

    def observe(self, *, workflow_id, plans, controller_state, tick):
        terminal = str(getattr(tick, "state", ""))
        generations = {}
        try:
            if self.workspace.ledger.is_file():
                generations = json.loads(self.workspace.ledger.read_text()).get("generations", {})
            if not isinstance(generations, Mapping):
                generations = {}
        except Exception as error:
            self._warn(f"cannot read progress ledger: {error}")
        try:
            self._monitor(workflow_id, controller_state, tick)
        except Exception as error:
            self._warn(f"cannot report job monitoring: {error}")
        # A broken historical report must not suppress the current alarm.
        for plan in plans:
            try:
                mapping = _plan_mapping(plan)
                generation = str(mapping["generation"])
                record = generations.get(generation)
                if not isinstance(record, Mapping) or not record.get("complete") or not record.get("accepted"):
                    continue
                event_id = f"generation:{generation}:accepted"
                with self._transaction() as state:
                    old = state["events"].get(event_id, {})
                    observed = generation in state["observed_generations"]
                if observed and not (old.get("state") == "failed" and not old.get("text")):
                    continue
                # Final training is reported in the single workflow completion.
                if record.get("kind") != "finalization":
                    if self.enqueue(_generation_event(
                        workflow_id, len(plans), plan, record, workflow_path=self.workspace.root,
                    )) is False:
                        continue
                with self._transaction() as state:
                    if generation not in state["observed_generations"]:
                        state["observed_generations"].append(generation)
            except Exception as error:
                self._warn(f"cannot build generation report: {error}")
        if terminal in _TERMINAL_STATES:
            try:
                state = dict(controller_state)
                entered = [int(g) for g in generations if str(g).isdigit()]
                if entered:
                    state.setdefault("last_generation", max(entered))
                event = _terminal_event(
                    workflow_id, len(plans), state, tick, workflow_path=self.workspace.root,
                )
                if terminal == "complete" and entered:
                    try:
                        record = generations[str(state.get("completed_generation") or max(entered))]
                        final_generation = state.get("completed_generation") or max(entered)
                        final_plan = next((_plan_mapping(p) for p in plans
                                           if int(_plan_mapping(p)["generation"]) == int(final_generation)),
                                          {"generation": final_generation, "max_selected": 0})
                        science = _generation_science(final_plan, record)
                        training = science["training"]
                        details = [f"Final training set: {_number(training.get('after_count'))} structures"]
                        if training.get("active_model_sha256"):
                            details.append(f"Model: {training['active_model_sha256'][:12]}")
                        event = NotificationEvent(event.event_id, event.text + "\n" + "\n".join(details), event.priority)
                    except Exception as error:
                        self._warn(f"cannot add final model details: {error}")
                self.enqueue(event)
            except Exception as error:
                self._warn(f"cannot build terminal alert: {error}")

    def _monitor(self, workflow_id, controller_state, tick):
        event = None
        with self._transaction() as state:
            monitor = state.get("monitoring")
            if getattr(tick, "state", "") == "degraded":
                if not monitor:
                    monitor = state["monitoring"] = {"since": time.time(), "id": uuid.uuid4().hex, "alerted": False}
                if time.time() - monitor["since"] >= _MONITOR_WARNING_SECONDS and not monitor["alerted"]:
                    event = _lifecycle_event(workflow_id, self.workspace.root, "monitoring_unavailable",
                                             controller_state, monitor["id"], detail=getattr(tick, "detail", ""))
                    monitor["alerted"] = True
            elif monitor:
                if monitor["alerted"] and getattr(tick, "state", "") not in {"failed", "stopped"}:
                    event = _lifecycle_event(workflow_id, self.workspace.root, "monitoring_restored",
                                             controller_state, monitor["id"])
                else:
                    state.pop("monitoring", None)
        if event:
            queued = self.enqueue(event) is not False
            with self._transaction() as state:
                if not queued and event.event_id.startswith("controller:monitoring_unavailable:"):
                    state.get("monitoring", {})["alerted"] = False
                elif queued and event.event_id.startswith("controller:monitoring_restored:"):
                    if state.get("monitoring", {}).get("id") == monitor["id"]:
                        state.pop("monitoring", None)

    def _deliver_next(self):
        # A prior controller's sender may still be finishing a request.
        with self.workspace.notification_state.with_suffix(".delivery.lock").open("a+") as delivery:
            try:
                fcntl.flock(delivery, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            with self._transaction() as state:
                if state.get("settings_fingerprint") != self._fingerprint:
                    return False
                pending = [(key, r) for key, r in state["events"].items()
                           if r.get("text") and r.get("state") in {"pending", "retrying", "sending"}
                           and r.get("next_attempt_at", 0) <= time.time()]
                if not pending:
                    return False
                key, record = min(pending, key=lambda pair: (pair[1].get("priority", 10), pair[1].get("created_at", "")))
                record.update(state="sending", last_attempt_at=_now())
                text = record["text"]
            result = send_feishu_text(self.settings, text, pool=self.pool)
            with self._transaction() as state:
                record = state["events"][key]
                attempt = int(record.get("delivery_attempts", 0)) + 1
                record.update(attempts=int(record.get("attempts", 0)) + 1, delivery_attempts=attempt)
                if result.ok:
                    record.update(state="delivered", delivered_at=_now())
                    record.pop("last_error", None)
                elif state.get("settings_fingerprint") != self._fingerprint:
                    record.update(state="pending", delivery_attempts=0, next_attempt_at=0)
                else:
                    record.update(
                        state="retrying" if attempt < _MAX_DELIVERY_ATTEMPTS else "failed",
                        last_error=_redact(result.detail, self.settings),
                        next_attempt_at=time.time() + _RETRY_DELAY_SECONDS * 2 ** (attempt - 1),
                    )
            if not result.ok:
                self._warn(f"{key} delivery failed: {_redact(result.detail, self.settings)}")
            return True

    def _run(self):
        try:
            while True:
                self._wake.clear()
                if self._closed and time.monotonic() >= self._close_deadline:
                    return  # Pending messages remain durable for the next sender.
                try:
                    if self._deliver_next():
                        continue
                    with self._transaction() as state:
                        if state.get("settings_fingerprint") != self._fingerprint:
                            return
                        pending = any(r.get("state") in {"pending", "retrying", "sending"}
                                      for r in state["events"].values())
                    if self._closed and not pending:
                        return
                except Exception as error:
                    self._warn(f"notification sender stopped: {error}")
                    return  # Keep the durable outbox; do not affect computation.
                # Idle workflows need no repeated shared-filesystem writes.
                # enqueue/close wake us; only outstanding deliveries need polling.
                self._wake.wait(0.2 if pending else None)
        finally:
            self._wake.set()

    def close(self):
        self._close_deadline = time.monotonic() + _SHUTDOWN_GRACE_SECONDS
        self._closed = True
        self._wake.set()


def _redact(detail, settings):
    for key in ("webhook", "secret"):
        value = settings.get(key)
        if value:
            detail = detail.replace(str(value), "[redacted]")
    return detail


def _lifecycle_event(workflow_id, workflow_path, kind, state, occurrence, *, detail=""):
    labels = {
        "started": "▶️ [NepTrain] Workflow started",
        "resumed": "▶️ [NepTrain] Workflow resumed",
        "restarted": "▶️ [NepTrain] Stage restart started",
        "recovered": "ℹ️ [NepTrain] Workflow recovered",
        "paused": "⏸️ [NepTrain] Workflow paused",
        "stopped": "⏹️ [NepTrain] Workflow stopped",
        "interrupted": "⚠️ [NepTrain] Controller interrupted",
        "monitoring_unavailable": "⚠️ [NepTrain] Job monitoring unavailable",
        "monitoring_restored": "✅ [NepTrain] Job monitoring restored",
    }
    current = state.get("current") or {}
    generation = current.get("generation") or state.get("notification_start", {}).get("generation")
    stage = current.get("stage") or state.get("notification_start", {}).get("stage")
    lines = [labels[kind], *_workflow_identity(workflow_id, workflow_path)]
    if generation is not None:
        lines.append(f"Current stage: G{generation} / {stage or 'pending'}")
    tasks = current.get("tasks", [current])
    handles = [t["handle"] for t in tasks if t.get("handle") and not t.get("collected_bundle")]
    if kind in {"resumed", "recovered", "paused", "interrupted"}:
        lines.append(f"Retained job handles: {len(handles)}. Their scheduler state is checked separately.")
    if kind == "paused":
        lines.append("The controller is paused. Existing compute jobs were not cancelled.")
    elif kind == "interrupted":
        lines.append("The controller stopped unexpectedly. Submitted jobs may still be running.")
        lines.append(f"Next: neptrain workflow resume {shlex.quote(str(workflow_path))}")
    elif kind == "monitoring_unavailable":
        lines.append("Job status queries have failed for at least 5 minutes. The controller will keep retrying; jobs are not marked failed.")
    elif kind == "monitoring_restored":
        lines.append("Job status queries are working again.")
    if detail:
        lines.append(detail)
    return NotificationEvent(f"controller:{kind}:{occurrence}", "\n".join(lines), 5)


def notification_summary(settings, workspace):
    summary = dict(provider="feishu", state="configured" if settings else "disabled",
                   delivered=0, failed=0, pending=0, retrying=0, last_error=None, last_success_at=None)
    if not workspace.notification_state.exists():
        return summary
    try:
        data = json.loads(workspace.notification_state.read_text())
        events = data["events"]
        if not isinstance(events, Mapping) or any(not isinstance(r, Mapping) for r in events.values()):
            raise ValueError("invalid notification event records")
        records = list(events.values())
        failures = [r for r in records if r.get("state") == "failed"]
        summary.update(
            delivered=sum(r.get("state") == "delivered" for r in records), failed=len(failures),
            pending=sum(r.get("state") in {"pending", "sending"} for r in records),
            retrying=sum(r.get("state") == "retrying" for r in records),
            last_success_at=max((r.get("delivered_at", "") for r in records), default="") or None,
        )
        errors = [r for r in records if r.get("last_error")]
        if errors:
            summary["last_error"] = max(errors, key=lambda r: r.get("last_attempt_at", r.get("created_at", ""))).get("last_error")
        if settings:
            latest = max(records, key=lambda r: r.get("last_attempt_at", r.get("created_at", "")), default={})
            summary["state"] = ("retrying" if summary["retrying"] else "pending" if summary["pending"] else
                                "degraded" if latest.get("state") == "failed" else
                                "ok" if summary["delivered"] else "configured")
    except Exception as error:
        summary.update(state="degraded" if settings else "disabled", last_error=f"cannot read notification state: {error}")
    return summary


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
