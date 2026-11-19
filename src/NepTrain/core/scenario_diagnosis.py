"""Evidence for repeated trajectory failures of one route, structure, T and P."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from math import floor, isfinite

import numpy as np

from .content_addressing import canonical_sha256
from .md_policy import TrajectoryHealthPolicy


def health_policy_hash(config):
    settings = config.get("sampling", {}).get("candidate_pool", {}).get("health")
    return canonical_sha256(asdict(TrajectoryHealthPolicy.from_mapping(settings)))


def crossing_step(bad_step, dump_interval):
    return max(floor(1.25 * bad_step) + 1, bad_step + 2 * dump_interval)


class NonfinitePrediction(ValueError):
    pass


def paired_metrics(predict, model, frames, backend):
    evaluation = predict(model, frames, backend)
    metrics = dict(evaluation.metrics)
    virial = evaluation.comparisons.get("virial")
    if virial is not None:
        # The predictor's parity arrays contain virials in eV per atom.
        reference = np.asarray(virial.reference).reshape(len(frames), -1)
        predicted = np.asarray(virial.predicted).reshape(reference.shape)
        components = predicted - reference
        if components.shape[1] == 9:
            trace = components[:, [0, 4, 8]].sum(axis=1)
        elif components.shape[1] == 6:
            trace = components[:, :3].sum(axis=1)
        else:
            raise ValueError("per-frame virial must have six or nine components")
        volumes = np.array([abs(f.cell.volume) for f in frames])
        if np.all(np.isfinite(volumes) & (volumes > 0)):
            metrics["pressure_bias_GPa"] = float(np.mean(
                trace * [len(f) for f in frames] / (3 * volumes)
            ) * 160.2176634)
    if not all(isfinite(float(v)) for v in metrics.values()):
        raise NonfinitePrediction("model diagnostic contains nonfinite predictions")
    return metrics


def grouped_frames(frames, key):
    groups = defaultdict(list)
    for frame in frames:
        if frame.info.get(key):
            groups[str(frame.info[key])].append(frame)
    return groups


def _pooled(samples, fields):
    """Pool equal-composition replicas by frame count, keeping pressure signed."""
    if not samples:
        return {}
    count = sum(s["count"] for s in samples)
    result = {"count": count, "frame_ids": [f for s in samples for f in s["frame_ids"]]}
    for field in fields:
        common = set.intersection(*(set(s[field]) for s in samples))
        metrics = {}
        for k in common:
            if k.endswith("_rmse"):
                metrics[k] = (sum(s["count"] * s[field][k] ** 2 for s in samples) / count) ** .5
            elif k == "pressure_bias_GPa":
                metrics[k] = sum(s["count"] * s[field][k] for s in samples) / count
        result[field] = metrics
    return result


def _pause_reason(failures, limits):
    if len(failures) < 2 or not any(x.get("survival_comparable", True) for x in failures[-2:]):
        return None
    tail = failures[-2:]
    required = {"energy_rmse", "force_rmse"}
    if "spin" in tail[-1]["signature"]:
        required.add("mforce_rmse")
    volume_failure = "volume" in tail[-1]["signature"]
    if volume_failure:
        required.add("virial_rmse")

    def measurable(metrics):
        return (required <= limits.keys() and limits.keys() <= metrics.keys()
                and (not volume_failure or "pressure_bias_GPa" in metrics))

    def enough(samples):
        return all(samples) and len({f for s in samples for f in s["frame_ids"]}) >= 3

    pre = [x["before"] for x in tail]
    if enough(pre) and all(measurable(s["metrics"]) and
                           all(s["metrics"][k] <= 1.2 * v for k, v in limits.items()) for s in pre):
        return "low_pre_failure_errors"

    def relative_error(metrics):
        return max(metrics[k] / max(limit, 1e-30) for k, limit in limits.items())

    pairs = [x.get("paired") for x in failures[:-1][-2:]]
    improved = False
    if pairs and pairs[-1] and all(measurable(pairs[-1][side]) for side in ("before", "after")):
        before, after = (relative_error(pairs[-1][side]) for side in ("before", "after"))
        improved = before > 1.2 and after < 0.9 * before and tail[-1]["novel"]
    if len(pairs) == 2 and enough(pairs) and all(
        measurable(p["before"]) and measurable(p["after"]) and
        relative_error(p["after"]) > 1.2 and
        relative_error(p["after"]) >= 0.9 * relative_error(p["before"]) for p in pairs
    ):
        return "poor_fit_on_same_frames"
    comparable = sum(x.get("survival_comparable", True) for x in failures)
    if comparable >= 3:
        if improved:
            return "novel_pre_failure_budget_exhausted" if comparable >= 5 else "extend_for_novel_frames"
        return "repeated_failure_signature"
    return None


def record_paired_fit(record, pairs, active_model):
    if not pairs:
        return
    history = record.get("diagnosis", {}).get("observations", ())
    for old in reversed(history):
        # Older histories stored the representative alone at the top level.
        attempts = old.get("batch") or [old]
        matched = []
        for attempt in attempts:
            pair = pairs.get(attempt["attempt_id"])
            if (not attempt["completed"] and pair and pair["parent_model_sha256"] == old["model_sha256"]
                    and pair["child_model_sha256"] == active_model and active_model != old["model_sha256"]):
                attempt["paired"] = pair
                if attempt["signature"] == old["signature"]:
                    matched.append(pair)
        if matched:
            old["paired"] = {**_pooled(matched, ("before", "after")),
                             "parent_model_sha256": old["model_sha256"], "child_model_sha256": active_model}


def observe(record, rows, pairs, limits):
    """Keep the fixed survival sample and retain every failing replica's evidence."""
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["replica"])
    reference = rows[0]
    record_paired_fit(record, pairs, reference["model_sha256"])
    failed = [r for r in rows if not r["completed"]]
    if not failed and "diagnosis" not in record:
        return
    state = record.setdefault("diagnosis", {"observations": []})
    history = state["observations"]
    if history and history[-1]["generation"] >= reference["generation"]:
        return
    baseline = state.get("baseline")
    located = [r for r in failed if r["bad_step"] is not None and r["signature"]]
    fault = reference
    if failed and reference not in located:
        same_signature = [r for r in located if baseline and r["signature"] == baseline["signature"]]
        fault = (same_signature or located or failed)[0]
    matching = [r for r in failed if r["signature"] == fault["signature"]]
    row = {**fault, "replicas": len(rows), "right_censored": fault["completed"],
           "reference_attempt_id": reference["attempt_id"], "reference_right_censored": reference["completed"],
           "survival_comparable": fault is reference, "batch": [dict(r) for r in rows],
           "before": _pooled([r["before"] for r in matching if r["before"]], ("metrics",)),
           "selected_pre_failure": sum(r.get("selected_pre_failure", 0) for r in matching),
           "novel": any(r["novel"] for r in matching)}
    history.append(row)
    if state.get("status") == "paused_for_diagnosis":
        return
    if not row["timestep_ps"] or not row["dump_interval"]:
        row["survival_comparable"] = False
        if baseline and (row["health_hash"] != baseline["health_hash"] or
                         (row["timestep_ps"] and row["timestep_ps"] != baseline["timestep_ps"])):
            state.pop("baseline", None)
        return
    if baseline:
        policy_changed = any(row[k] != baseline[k] for k in ("health_hash", "timestep_ps"))
        reached = row["steps"] if row["completed"] else row["bad_step"]
        baseline["comparison_dump_interval"] = max(baseline["dump_interval"], row["dump_interval"])
        margin = crossing_step(baseline["bad_step"], baseline["comparison_dump_interval"])
        if (policy_changed or (row["survival_comparable"] and reached is not None and reached >= margin)
                or (row["completed"] and reached >= baseline["original_target"])):
            baseline = None
            state.pop("baseline", None)
    if row["completed"]:
        return
    if not row["signature"] or row["bad_step"] is None:
        return
    if baseline and not row["survival_comparable"] and row["signature"] != baseline["signature"]:
        return
    if not baseline or row["signature"] != baseline["signature"]:
        state["baseline"] = {
            **{k: row[k] for k in ("generation", "bad_step", "timestep_ps", "health_hash",
                                    "model_sha256", "dump_interval", "signature")},
            "original_target": row["steps"],
        }
    baseline = state["baseline"]
    distinct = {}
    for item in history:
        if (item["generation"] >= baseline["generation"] and item["model_updated"]
                and item["timestep_ps"] and item["dump_interval"]
                and not item["completed"] and item["signature"] == baseline["signature"]):
            distinct.setdefault(item["model_sha256"], item)
    failures = list(distinct.values())
    reason = _pause_reason(failures, limits)
    if reason == "extend_for_novel_frames":
        state["novelty_extension"] = {"generation": row["generation"], "models_observed": len(failures)}
        return
    if reason:
        state.update(status="paused_for_diagnosis", paused_generation=row["generation"], reason=reason)
