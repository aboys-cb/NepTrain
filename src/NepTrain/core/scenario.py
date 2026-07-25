"""Deterministic temperature-path and duration-frontier scheduling."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


MATURITY_LEVELS = (
    "untested",
    "smoke_passed",
    "short_stable",
    "long_stable",
    "production_ready",
)
_TARGET_LEVELS = MATURITY_LEVELS[1:]


class ScenarioMaturityError(ValueError):
    """Raised when scenario policy or history is inconsistent."""


@dataclass(frozen=True)
class ScenarioAttempt:
    attempt_id: str
    scenario_id: str
    route_id: str
    route_fingerprint: str
    template_sha256: str
    structure_id: str
    structure_hash: str
    temperature: float
    pressure: float
    previous_level: str
    target_level: str
    steps: int
    replica: int
    generation: int
    model_id: str
    seed: int


def _scenario_id(
    route_id: str,
    route_fingerprint: str,
    structure_id: str,
    temperature: float,
    pressure: float,
) -> str:
    payload = json.dumps(
        {
            "route_id": route_id,
            "route_fingerprint": route_fingerprint,
            "structure_id": structure_id,
            "temperature": float(temperature),
            "pressure": float(pressure),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _attempt_id(
    scenario_id: str,
    generation: int,
    target_level: str,
    replica: int,
    model_id: str,
) -> str:
    payload = json.dumps(
        {
            "scenario_id": scenario_id,
            "generation": int(generation),
            "target_level": target_level,
            "replica": int(replica),
            "model_id": model_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _replica_seed(attempt_id: str) -> int:
    return (
        int(hashlib.sha256(attempt_id.encode()).hexdigest()[:12], 16)
        % 900_000_000
    ) + 1


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in metrics.items()
        if isinstance(value, (bool, int, float, str)) or value is None
    }


class ScenarioLadder:
    """Own the sparse temperature/duration trust frontier.

    Temperatures are unlocked in the user-provided order. Every temperature is
    first tested at the smoke duration. Only production temperatures continue
    through the longer duration levels, avoiding a full Cartesian grid.
    """

    def __init__(
        self,
        steps: Mapping[str, int],
        *,
        temperature_path: Sequence[float],
        production_temperatures: Sequence[float] | None = None,
        replicas: Mapping[str, int] | None = None,
    ):
        observed = set(steps)
        expected = set(_TARGET_LEVELS)
        if observed != expected:
            raise ScenarioMaturityError(
                "maturity levels must define smoke_passed, short_stable, "
                "long_stable, and production_ready"
            )
        ordered_steps = tuple(int(steps[name]) for name in _TARGET_LEVELS)
        if any(value < 1 for value in ordered_steps):
            raise ScenarioMaturityError("maturity steps must be positive")
        if any(
            later <= earlier
            for earlier, later in zip(ordered_steps, ordered_steps[1:])
        ):
            raise ScenarioMaturityError("maturity steps must increase at every level")

        path = tuple(float(value) for value in temperature_path)
        if not path or len(set(path)) != len(path) or not all(np.isfinite(path)):
            raise ScenarioMaturityError(
                "temperature_path must contain unique finite temperatures"
            )
        if len(path) > 2:
            differences = np.diff(path)
            if not (np.all(differences > 0) or np.all(differences < 0)):
                raise ScenarioMaturityError(
                    "temperature_path must be strictly increasing or decreasing"
                )
        elif len(path) == 2 and path[0] == path[1]:
            raise ScenarioMaturityError("temperature_path temperatures must be unique")

        production = (
            path
            if production_temperatures is None
            else tuple(float(value) for value in production_temperatures)
        )
        if not production or len(set(production)) != len(production):
            raise ScenarioMaturityError(
                "production_temperatures must contain unique temperatures"
            )
        unknown_production = [value for value in production if value not in path]
        if unknown_production:
            raise ScenarioMaturityError(
                "production_temperatures must be a subset of temperature_path"
            )

        configured_replicas = {
            "smoke_passed": 1,
            "short_stable": 1,
            "long_stable": 2,
            "production_ready": 3,
            **dict(replicas or {}),
        }
        if set(configured_replicas) != expected or any(
            int(configured_replicas[name]) < 1 for name in _TARGET_LEVELS
        ):
            raise ScenarioMaturityError(
                "replicas must define positive counts for every maturity level"
            )

        self.steps = {
            name: value for name, value in zip(_TARGET_LEVELS, ordered_steps)
        }
        self.temperature_path = path
        self.production_temperatures = tuple(
            value for value in path if value in set(production)
        )
        self.replicas = {
            name: int(configured_replicas[name]) for name in _TARGET_LEVELS
        }

    @classmethod
    def from_sampling(cls, sampling: Mapping[str, Any]) -> ScenarioLadder:
        """Build the explicit trust frontier from sampling settings."""

        if not isinstance(sampling, Mapping):
            raise ScenarioMaturityError("sampling must be a mapping")
        conditions = sampling.get("conditions", {})
        progression = sampling.get("progression", {})
        if not isinstance(conditions, Mapping):
            raise ScenarioMaturityError("sampling.conditions must be a mapping")
        if not isinstance(progression, Mapping):
            raise ScenarioMaturityError("sampling.progression must be a mapping")
        steps = progression.get("steps", {})
        if not isinstance(steps, Mapping):
            raise ScenarioMaturityError(
                "sampling.progression.steps must be a mapping"
            )
        expected = set(_TARGET_LEVELS)
        unknown = sorted(set(steps) - expected)
        missing = sorted(expected - set(steps))
        if unknown or missing:
            details = []
            if unknown:
                details.append("unknown: " + ", ".join(unknown))
            if missing:
                details.append("missing: " + ", ".join(missing))
            raise ScenarioMaturityError(
                "invalid sampling.progression.steps (" + "; ".join(details) + ")"
            )
        return cls(
            {name: int(value) for name, value in steps.items()},
            temperature_path=conditions.get("temperature_path", ()),
            production_temperatures=conditions.get("production_temperatures"),
            replicas=progression.get("replicas"),
        )

    def empty_history(self) -> dict[str, Any]:
        return {
            "version": 2,
            "levels": dict(self.steps),
            "replicas": dict(self.replicas),
            "temperature_path": list(self.temperature_path),
            "production_temperatures": list(self.production_temperatures),
            "scenarios": {},
            "counts_by_maturity": {},
            "no_progress_rounds": 0,
        }

    def _validated_history(
        self, history: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        if history is None:
            return self.empty_history()
        copied = json.loads(json.dumps(history))
        expected_policy = self.empty_history()
        if copied.get("version") != 2:
            raise ScenarioMaturityError(
                "unsupported scenario maturity history version"
            )
        for key in (
            "levels",
            "replicas",
            "temperature_path",
            "production_temperatures",
        ):
            if copied.get(key) != expected_policy[key]:
                raise ScenarioMaturityError(
                    "scenario progression changed after workflow start"
                )
        scenarios = copied.get("scenarios")
        if not isinstance(scenarios, dict):
            raise ScenarioMaturityError("scenario maturity history is malformed")
        for scenario_id, record in scenarios.items():
            if record.get("scenario_id") != scenario_id:
                raise ScenarioMaturityError("scenario maturity id is inconsistent")
            if record.get("maturity") not in MATURITY_LEVELS:
                raise ScenarioMaturityError("scenario maturity level is invalid")
            if not isinstance(record.get("evidence", []), list):
                raise ScenarioMaturityError("scenario evidence must be a list")
        return copied

    def _record(
        self,
        scenarios: Mapping[str, Any],
        route_id: str,
        route_fingerprint: str,
        structure_id: str,
        temperature: float,
        pressure: float,
    ) -> Mapping[str, Any]:
        return scenarios.get(
            _scenario_id(
                route_id,
                route_fingerprint,
                structure_id,
                temperature,
                pressure,
            ),
            {"maturity": "untested", "evidence": []},
        )

    def _temperature_unlocked(
        self,
        scenarios: Mapping[str, Any],
        route_id: str,
        route_fingerprint: str,
        structure_id: str,
        temperature_index: int,
        pressure: float,
    ) -> bool:
        if temperature_index == 0:
            return True
        previous = self._record(
            scenarios,
            route_id,
            route_fingerprint,
            structure_id,
            self.temperature_path[temperature_index - 1],
            pressure,
        )
        return (
            MATURITY_LEVELS.index(str(previous["maturity"]))
            >= MATURITY_LEVELS.index("smoke_passed")
        )

    def _target(
        self, record: Mapping[str, Any], temperature: float, model_id: str
    ) -> str | None:
        current = str(record["maturity"])
        if current == "untested":
            return "smoke_passed"
        if record.get("canary_model_id") != model_id:
            return "smoke_passed"
        if temperature not in self.production_temperatures:
            return None
        if current == "production_ready":
            return (
                None
                if record.get("verified_model_id") == model_id
                else "production_ready"
            )
        return MATURITY_LEVELS[MATURITY_LEVELS.index(current) + 1]

    @staticmethod
    def _latest_failed(record: Mapping[str, Any], target: str) -> bool:
        relevant = [
            item
            for item in record.get("evidence", [])
            if item.get("target_level") == target
        ]
        return bool(relevant and relevant[-1].get("accepted") is False)

    def _accepted_replicas(
        self, record: Mapping[str, Any], target: str, model_id: str
    ) -> int:
        return sum(
            bool(item.get("accepted"))
            and item.get("target_level") == target
            and item.get("model_id") == model_id
            for item in record.get("evidence", [])
        )

    def schedule(
        self,
        structure_ids: Sequence[str],
        *,
        route_id: str = "default",
        route_fingerprint: str = "default",
        template_sha256: str = "default",
        pressure: float,
        generation: int,
        seed: int,
        limit: int,
        model_id: str,
        history: Mapping[str, Any] | None = None,
    ) -> tuple[ScenarioAttempt, ...]:
        """Return deterministic recovery-first attempts on the unlocked frontier."""

        if generation < 1 or limit < 1:
            raise ScenarioMaturityError(
                "generation and scenario limit must be positive"
            )
        if (
            not model_id
            or not route_id
            or not route_fingerprint
            or not template_sha256
        ):
            raise ScenarioMaturityError(
                "scenario scheduling requires route identity and model_id"
            )
        structures = sorted(set(str(value) for value in structure_ids))
        if not structures or any(not value for value in structures):
            raise ScenarioMaturityError(
                "scenario scheduling requires non-empty structure ids"
            )
        if not np.isfinite(float(pressure)):
            raise ScenarioMaturityError("scenario pressure must be finite")

        state = self._validated_history(history)
        scenarios = state["scenarios"]
        tie_order = np.random.default_rng(seed).permutation(
            len(structures) * len(self.temperature_path)
        )
        tie_rank = {
            int(candidate): rank for rank, candidate in enumerate(tie_order)
        }
        candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for structure_index, structure in enumerate(structures):
            for temperature_index, temperature in enumerate(self.temperature_path):
                if not self._temperature_unlocked(
                    scenarios,
                    route_id,
                    route_fingerprint,
                    structure,
                    temperature_index,
                    pressure,
                ):
                    continue
                identifier = _scenario_id(
                    route_id,
                    route_fingerprint,
                    structure,
                    temperature,
                    pressure,
                )
                record = self._record(
                    scenarios,
                    route_id,
                    route_fingerprint,
                    structure,
                    temperature,
                    pressure,
                )
                target = self._target(record, temperature, model_id)
                if target is None:
                    continue
                accepted = self._accepted_replicas(record, target, model_id)
                needed = self.replicas[target] - accepted
                if needed <= 0:
                    continue
                evidence_count = sum(
                    item.get("target_level") == target
                    for item in record.get("evidence", [])
                )
                retry_priority = 0 if self._latest_failed(record, target) else 1
                base_tie = tie_rank[
                    structure_index * len(self.temperature_path)
                    + temperature_index
                ]
                for offset in range(needed):
                    replica = evidence_count + offset + 1
                    candidates.append(
                        (
                            (
                                retry_priority,
                                MATURITY_LEVELS.index(target),
                                offset,
                                temperature_index,
                                base_tie,
                                identifier,
                            ),
                            {
                                "scenario_id": identifier,
                                "structure_id": structure,
                                "temperature": temperature,
                                "previous_level": str(record["maturity"]),
                                "target_level": target,
                                "replica": replica,
                            },
                        )
                    )

        # If the complete frontier still misses the global validation target,
        # run cheap additional production probes instead of pretending success.
        if not candidates and state.get("validation_accepted") is False:
            for structure in structures:
                for temperature in self.production_temperatures:
                    identifier = _scenario_id(
                        route_id,
                        route_fingerprint,
                        structure,
                        temperature,
                        pressure,
                    )
                    record = self._record(
                        scenarios,
                        route_id,
                        route_fingerprint,
                        structure,
                        temperature,
                        pressure,
                    )
                    if record["maturity"] != "production_ready":
                        continue
                    replica = 1 + sum(
                        item.get("target_level") == "production_ready"
                        for item in record.get("evidence", [])
                    )
                    candidates.append(
                        (
                            (
                                0,
                                MATURITY_LEVELS.index("production_ready"),
                                0,
                                self.temperature_path.index(temperature),
                                replica,
                                identifier,
                            ),
                            {
                                "scenario_id": identifier,
                                "structure_id": structure,
                                "temperature": temperature,
                                "previous_level": "production_ready",
                                "target_level": "production_ready",
                                "replica": replica,
                            },
                        )
                    )

        attempts = []
        for _, value in sorted(candidates, key=lambda item: item[0])[:limit]:
            attempt_id = _attempt_id(
                value["scenario_id"],
                generation,
                value["target_level"],
                value["replica"],
                model_id,
            )
            attempts.append(
                ScenarioAttempt(
                    attempt_id=attempt_id,
                    scenario_id=value["scenario_id"],
                    route_id=route_id,
                    route_fingerprint=route_fingerprint,
                    template_sha256=template_sha256,
                    structure_id=value["structure_id"],
                    structure_hash=value["structure_id"],
                    temperature=float(value["temperature"]),
                    pressure=float(pressure),
                    previous_level=value["previous_level"],
                    target_level=value["target_level"],
                    steps=self.steps[value["target_level"]],
                    replica=int(value["replica"]),
                    generation=generation,
                    model_id=model_id,
                    seed=_replica_seed(attempt_id),
                )
            )
        return tuple(attempts)

    def record(
        self,
        attempts: Sequence[ScenarioAttempt | Mapping[str, Any]],
        *,
        completed: Mapping[str, bool],
        diagnostic_accepted: Mapping[str, bool | None],
        history: Mapping[str, Any] | None = None,
        diagnostic: Mapping[str, Any] | None = None,
        validation: Mapping[str, Any] | None = None,
        evidence_validation: Mapping[str, Any] | None = None,
        validation_accepted: bool | None,
        model_improved: bool,
        novelty_converged: bool,
        final_model_id: str,
    ) -> dict[str, Any]:
        """Append attempt evidence and promote only trusted completed replicas."""

        state = self._validated_history(history)
        scenarios = state["scenarios"]
        normalized = [
            raw if isinstance(raw, ScenarioAttempt) else ScenarioAttempt(**raw)
            for raw in attempts
        ]
        attempt_ids = {attempt.attempt_id for attempt in normalized}
        for name, values in (
            ("completion", completed),
            ("diagnostic", diagnostic_accepted),
        ):
            valid_values = (
                (bool,)
                if name == "completion"
                else (bool, type(None))
            )
            if set(values) != attempt_ids or not all(
                isinstance(value, valid_values) for value in values.values()
            ):
                raise ScenarioMaturityError(
                    f"scenario {name} results must match every planned attempt"
                )

        starting_maturity = {
            attempt.scenario_id: scenarios.get(
                attempt.scenario_id, {}
            ).get("maturity", "untested")
            for attempt in normalized
        }
        attempt_diagnostics = (
            diagnostic.get("attempt_metrics", {})
            if isinstance(diagnostic, Mapping)
            else {}
        )
        for attempt in normalized:
            if starting_maturity[attempt.scenario_id] != attempt.previous_level:
                raise ScenarioMaturityError(
                    "scenario attempt was planned from stale history"
                )
            if attempt.steps != self.steps[attempt.target_level]:
                raise ScenarioMaturityError(
                    "scenario attempt steps do not match its level"
                )
            record = scenarios.setdefault(
                attempt.scenario_id,
                {
                    "scenario_id": attempt.scenario_id,
                    "route_id": attempt.route_id,
                    "route_fingerprint": attempt.route_fingerprint,
                    "template_sha256": attempt.template_sha256,
                    "structure_id": attempt.structure_id,
                    "structure_hash": attempt.structure_hash,
                    "temperature": attempt.temperature,
                    "pressure": attempt.pressure,
                    "maturity": "untested",
                    "evidence": [],
                },
            )
            if (
                record["route_id"] != attempt.route_id
                or record["route_fingerprint"] != attempt.route_fingerprint
                or record["template_sha256"] != attempt.template_sha256
                or record["structure_id"] != attempt.structure_id
                or record["structure_hash"] != attempt.structure_hash
                or float(record["temperature"]) != attempt.temperature
                or float(record["pressure"]) != attempt.pressure
            ):
                raise ScenarioMaturityError("scenario identity metadata changed")
            trusted = bool(
                completed[attempt.attempt_id]
                and diagnostic_accepted[attempt.attempt_id] is not False
            )
            record["evidence"].append(
                {
                    "attempt_id": attempt.attempt_id,
                    "generation": attempt.generation,
                    "model_id": attempt.model_id,
                    "replica": attempt.replica,
                    "previous_level": attempt.previous_level,
                    "target_level": attempt.target_level,
                    "steps": attempt.steps,
                    "accepted": trusted,
                    "md_completed": completed[attempt.attempt_id],
                    "diagnostic_accepted": diagnostic_accepted[
                        attempt.attempt_id
                    ],
                    "diagnostic": _compact_metrics(
                        attempt_diagnostics.get(attempt.attempt_id, {})
                    ),
                    "validation": _compact_metrics(
                        evidence_validation
                        if evidence_validation is not None
                        else (validation or {})
                    ),
                }
            )

        promoted_count = 0
        for scenario_id in {attempt.scenario_id for attempt in normalized}:
            record = scenarios[scenario_id]
            targets = {
                attempt.target_level
                for attempt in normalized
                if attempt.scenario_id == scenario_id
            }
            if len(targets) != 1:
                raise ScenarioMaturityError(
                    "one scenario cannot target multiple maturity levels at once"
                )
            target = targets.pop()
            model_id = next(
                attempt.model_id
                for attempt in normalized
                if attempt.scenario_id == scenario_id
            )
            accepted = self._accepted_replicas(record, target, model_id)
            if accepted < self.replicas[target]:
                continue
            if target != "production_ready":
                canary_changed = record.get("canary_model_id") != model_id
                record["canary_model_id"] = model_id
                if (
                    MATURITY_LEVELS.index(target)
                    > MATURITY_LEVELS.index(record["maturity"])
                ):
                    record["maturity"] = target
                    promoted_count += 1
                elif canary_changed and model_id == final_model_id:
                    promoted_count += 1
            else:
                changed = (
                    record["maturity"] != "production_ready"
                    or record.get("verified_model_id") != model_id
                )
                record["maturity"] = "production_ready"
                record["canary_model_id"] = model_id
                record["verified_model_id"] = model_id
                promoted_count += int(changed and model_id == final_model_id)

        counts = Counter(
            (
                "long_stable"
                if record["maturity"] == "production_ready"
                and record.get("verified_model_id") != final_model_id
                else record["maturity"]
            )
            for record in scenarios.values()
        )
        state["counts_by_maturity"] = {
            level: counts[level] for level in MATURITY_LEVELS if counts[level]
        }
        state["last_generation"] = max(
            (attempt.generation for attempt in normalized),
            default=state.get("last_generation", 0),
        )
        state["validation_accepted"] = validation_accepted
        state["last_model_id"] = final_model_id
        state["last_promoted_count"] = promoted_count
        progress_made = bool(promoted_count or model_improved)
        state["no_progress_rounds"] = (
            0
            if progress_made
            or validation_accepted is True
            or not novelty_converged
            else int(state.get("no_progress_rounds", 0)) + 1
        )
        return state

    def production_ready(
        self,
        structure_ids: Sequence[str],
        *,
        route_id: str = "default",
        route_fingerprint: str = "default",
        pressure: float,
        model_id: str,
        history: Mapping[str, Any],
    ) -> bool:
        """Return whether every requested production condition trusts this model."""

        state = self._validated_history(history)
        scenarios = state["scenarios"]
        return all(
            (
                record := self._record(
                    scenarios,
                    route_id,
                    route_fingerprint,
                    str(structure),
                    temperature,
                    pressure,
                )
            ).get("maturity")
            == "production_ready"
            and record.get("verified_model_id") == model_id
            for structure in sorted(set(str(value) for value in structure_ids))
            for temperature in self.production_temperatures
        )

    @staticmethod
    def serialize(attempts: Sequence[ScenarioAttempt]) -> list[dict[str, Any]]:
        return [asdict(attempt) for attempt in attempts]


__all__ = [
    "MATURITY_LEVELS",
    "ScenarioAttempt",
    "ScenarioLadder",
    "ScenarioMaturityError",
]
