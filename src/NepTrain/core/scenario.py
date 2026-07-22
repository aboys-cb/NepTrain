"""Deterministic per-scenario maturity scheduling for MD campaigns."""

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
    scenario_id: str
    structure_id: str
    temperature: float
    pressure: float
    previous_level: str
    target_level: str
    steps: int
    generation: int


def _scenario_id(structure_id: str, temperature: float, pressure: float) -> str:
    payload = json.dumps(
        {
            "structure_id": structure_id,
            "temperature": float(temperature),
            "pressure": float(pressure),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in metrics.items()
        if isinstance(value, (bool, int, float, str)) or value is None
    }


class ScenarioLadder:
    """Schedule the next evidence level and record promotion in one history."""

    def __init__(self, steps: Mapping[str, int]):
        observed = set(steps)
        expected = set(_TARGET_LEVELS)
        if observed != expected:
            raise ScenarioMaturityError(
                "maturity levels must define smoke_passed, short_stable, "
                "long_stable, and production_ready"
            )
        ordered = tuple(int(steps[name]) for name in _TARGET_LEVELS)
        if any(value < 1 for value in ordered):
            raise ScenarioMaturityError("maturity steps must be positive")
        if any(later <= earlier for earlier, later in zip(ordered, ordered[1:])):
            raise ScenarioMaturityError("maturity steps must increase at every level")
        self.steps = {name: value for name, value in zip(_TARGET_LEVELS, ordered)}

    @classmethod
    def from_campaign(
        cls, campaign: Mapping[str, Any]
    ) -> ScenarioLadder | None:
        """Build the optional ladder from user campaign settings.

        A campaign enables the ladder by default. ``enabled: false`` is the
        explicit legacy escape hatch.
        """

        if not isinstance(campaign, Mapping):
            raise ScenarioMaturityError("campaign must be a mapping")
        if not campaign:
            return None
        settings = campaign.get("maturity", {})
        if settings is None:
            settings = {}
        if not isinstance(settings, Mapping):
            raise ScenarioMaturityError("campaign.maturity must be a mapping")
        enabled = settings.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ScenarioMaturityError("campaign.maturity.enabled must be boolean")
        if enabled is False:
            return None
        initial = int(campaign.get("initial_steps", 100))
        configured = settings.get("levels", {})
        if not isinstance(configured, Mapping):
            raise ScenarioMaturityError("campaign.maturity.levels must be a mapping")
        defaults = {
            "smoke_passed": initial,
            "short_stable": initial * 4,
            "long_stable": initial * 16,
            "production_ready": initial * 64,
        }
        unknown = sorted(set(configured) - set(defaults))
        if unknown:
            raise ScenarioMaturityError(
                f"unknown campaign.maturity.levels: {', '.join(unknown)}"
            )
        return cls(
            {
                name: int(configured.get(name, value))
                for name, value in defaults.items()
            }
        )

    def empty_history(self) -> dict[str, Any]:
        return {
            "version": 1,
            "levels": dict(self.steps),
            "scenarios": {},
            "counts_by_maturity": {},
        }

    def _validated_history(
        self, history: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        if history is None:
            return self.empty_history()
        copied = json.loads(json.dumps(history))
        if copied.get("version") != 1:
            raise ScenarioMaturityError("unsupported scenario maturity history version")
        if copied.get("levels") != self.steps:
            raise ScenarioMaturityError("scenario maturity levels changed after campaign start")
        scenarios = copied.get("scenarios")
        if not isinstance(scenarios, dict):
            raise ScenarioMaturityError("scenario maturity history is malformed")
        for scenario_id, record in scenarios.items():
            if record.get("scenario_id") != scenario_id:
                raise ScenarioMaturityError("scenario maturity id is inconsistent")
            if record.get("maturity") not in MATURITY_LEVELS:
                raise ScenarioMaturityError("scenario maturity level is invalid")
        return copied

    def schedule(
        self,
        structure_ids: Sequence[str],
        temperatures: Sequence[float],
        *,
        pressure: float,
        generation: int,
        seed: int,
        limit: int,
        history: Mapping[str, Any] | None = None,
    ) -> tuple[ScenarioAttempt, ...]:
        """Return deterministic, least-mature-first MD attempts."""

        if generation < 1 or limit < 1:
            raise ScenarioMaturityError("generation and scenario limit must be positive")
        structures = sorted(set(str(value) for value in structure_ids))
        temperatures = tuple(dict.fromkeys(float(value) for value in temperatures))
        if not structures or not temperatures:
            raise ScenarioMaturityError("scenario scheduling requires structures and temperatures")
        if any(not value for value in structures):
            raise ScenarioMaturityError("scenario structure ids cannot be empty")
        if not np.isfinite(float(pressure)) or not all(
            np.isfinite(value) for value in temperatures
        ):
            raise ScenarioMaturityError("scenario temperature and pressure must be finite")
        state = self._validated_history(history)
        scenarios = state["scenarios"]
        pairs = [
            (structure, temperature)
            for structure in structures
            for temperature in temperatures
        ]
        order = np.random.default_rng(seed).permutation(len(pairs))
        candidates = []
        for tie_rank, pair_index in enumerate(order):
            structure, temperature = pairs[int(pair_index)]
            identifier = _scenario_id(structure, temperature, pressure)
            current = scenarios.get(identifier, {}).get("maturity", "untested")
            candidates.append(
                (
                    MATURITY_LEVELS.index(current),
                    tie_rank,
                    identifier,
                    structure,
                    temperature,
                    current,
                )
            )
        temperature_counts = Counter(
            (float(record["temperature"]), float(record["pressure"]))
            for record in scenarios.values()
        )
        structure_counts = Counter(
            str(record["structure_id"]) for record in scenarios.values()
        )
        attempts = []
        while candidates and len(attempts) < limit:
            selected_index = min(
                range(len(candidates)),
                key=lambda index: (
                    candidates[index][0],
                    temperature_counts[
                        (float(candidates[index][4]), float(pressure))
                    ],
                    structure_counts[str(candidates[index][3])],
                    candidates[index][1],
                    candidates[index][2],
                ),
            )
            _, _, identifier, structure, temperature, current = candidates.pop(
                selected_index
            )
            current_index = MATURITY_LEVELS.index(current)
            target = MATURITY_LEVELS[min(current_index + 1, len(MATURITY_LEVELS) - 1)]
            attempts.append(
                ScenarioAttempt(
                    scenario_id=identifier,
                    structure_id=structure,
                    temperature=temperature,
                    pressure=float(pressure),
                    previous_level=current,
                    target_level=target,
                    steps=self.steps[target],
                    generation=generation,
                )
            )
            temperature_counts[(float(temperature), float(pressure))] += 1
            structure_counts[str(structure)] += 1
        return tuple(attempts)

    def record(
        self,
        attempts: Sequence[ScenarioAttempt | Mapping[str, Any]],
        *,
        accepted: bool,
        completed: Mapping[str, bool] | None = None,
        history: Mapping[str, Any] | None = None,
        diagnostic: Mapping[str, Any] | None = None,
        validation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append evidence and promote only completed, accepted attempts."""

        state = self._validated_history(history)
        scenarios = state["scenarios"]
        attempt_ids = {
            raw.scenario_id
            if isinstance(raw, ScenarioAttempt)
            else str(raw["scenario_id"])
            for raw in attempts
        }
        if completed is not None:
            if set(completed) != attempt_ids or not all(
                isinstance(value, bool) for value in completed.values()
            ):
                raise ScenarioMaturityError(
                    "scenario completion results must match every planned attempt"
                )
        for raw in attempts:
            attempt = (
                raw if isinstance(raw, ScenarioAttempt) else ScenarioAttempt(**raw)
            )
            current = scenarios.get(attempt.scenario_id, {}).get("maturity", "untested")
            if current != attempt.previous_level:
                raise ScenarioMaturityError("scenario attempt was planned from stale history")
            current_index = MATURITY_LEVELS.index(current)
            expected_target = MATURITY_LEVELS[
                min(current_index + 1, len(MATURITY_LEVELS) - 1)
            ]
            if attempt.target_level != expected_target:
                raise ScenarioMaturityError("scenario attempt skips a maturity level")
            if attempt.steps != self.steps[attempt.target_level]:
                raise ScenarioMaturityError("scenario attempt steps do not match its level")
            record = scenarios.setdefault(
                attempt.scenario_id,
                {
                    "scenario_id": attempt.scenario_id,
                    "structure_id": attempt.structure_id,
                    "temperature": attempt.temperature,
                    "pressure": attempt.pressure,
                    "maturity": "untested",
                    "evidence": [],
                },
            )
            if (
                record["structure_id"] != attempt.structure_id
                or float(record["temperature"]) != attempt.temperature
                or float(record["pressure"]) != attempt.pressure
            ):
                raise ScenarioMaturityError("scenario identity metadata changed")
            md_completed = (
                True if completed is None else completed[attempt.scenario_id]
            )
            promoted = bool(accepted and md_completed)
            record["evidence"].append(
                {
                    "generation": attempt.generation,
                    "previous_level": attempt.previous_level,
                    "target_level": attempt.target_level,
                    "steps": attempt.steps,
                    "accepted": promoted,
                    "md_completed": md_completed,
                    "validation_accepted": bool(accepted),
                    "diagnostic": _compact_metrics(diagnostic or {}),
                    "validation": _compact_metrics(validation or {}),
                }
            )
            if promoted:
                record["maturity"] = attempt.target_level

        counts = Counter(record["maturity"] for record in scenarios.values())
        state["counts_by_maturity"] = {
            level: counts[level] for level in MATURITY_LEVELS if counts[level]
        }
        if attempts:
            state["last_generation"] = max(
                attempt.generation
                if isinstance(attempt, ScenarioAttempt)
                else int(attempt["generation"])
                for attempt in attempts
            )
        return state

    @staticmethod
    def serialize(attempts: Sequence[ScenarioAttempt]) -> list[dict[str, Any]]:
        return [asdict(attempt) for attempt in attempts]


__all__ = [
    "MATURITY_LEVELS",
    "ScenarioAttempt",
    "ScenarioLadder",
    "ScenarioMaturityError",
]
