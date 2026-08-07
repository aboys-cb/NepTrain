"""Resolve immutable generation kinds and their allowed stage sequences."""

from __future__ import annotations

from typing import Any, Mapping


LEGACY_GENERATION_PROTOCOL = "legacy_v1"
ADAPTIVE_GENERATION_PROTOCOL = "adaptive_v2"

LEGACY_STAGES = (
    "train",
    "explore",
    "select",
    "label",
    "diagnose",
    "merge",
    "retrain",
    "evaluate",
)
ACQUISITION_STAGES = (
    "train",
    "evaluate",
    "explore",
    "select",
    "label",
    "diagnose",
    "merge",
)
FINALIZATION_STAGES = ("train", "evaluate")

_KINDS = {
    "legacy": LEGACY_STAGES,
    "acquisition": ACQUISITION_STAGES,
    "finalization": FINALIZATION_STAGES,
}


def generation_stage_sequence(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return a committed sequence, falling back to the legacy contract."""

    raw = record.get("stage_sequence")
    if raw is None:
        return LEGACY_STAGES
    sequence = tuple(str(stage) for stage in raw)
    kind = str(record.get("kind", ""))
    if kind not in _KINDS or sequence != _KINDS[kind]:
        raise ValueError("generation kind and stage sequence are inconsistent")
    return sequence


def generation_disposition(record: Mapping[str, Any] | None) -> str | None:
    """Read the disposition from the last committed stage that reports one."""

    if not record:
        return None
    stages = record.get("stages", {})
    if not isinstance(stages, Mapping):
        return None
    for stage in reversed(generation_stage_sequence(record)):
        stage_record = stages.get(stage)
        if not isinstance(stage_record, Mapping):
            continue
        metrics = stage_record.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        value = metrics.get("generation_disposition")
        if value in {"continue", "finalize"}:
            return str(value)
    return None


def resolve_generation_kind(
    protocol: str,
    previous_record: Mapping[str, Any] | None,
) -> str:
    """Resolve a new generation once, before its first stage is committed."""

    if protocol == LEGACY_GENERATION_PROTOCOL:
        return "legacy"
    if protocol != ADAPTIVE_GENERATION_PROTOCOL:
        raise ValueError(f"unsupported generation protocol: {protocol}")
    return (
        "finalization"
        if generation_disposition(previous_record) == "finalize"
        else "acquisition"
    )


def stage_sequence_for_kind(kind: str) -> tuple[str, ...]:
    try:
        return _KINDS[kind]
    except KeyError as error:
        raise ValueError(f"unsupported generation kind: {kind}") from error


__all__ = [
    "ACQUISITION_STAGES",
    "ADAPTIVE_GENERATION_PROTOCOL",
    "FINALIZATION_STAGES",
    "LEGACY_GENERATION_PROTOCOL",
    "LEGACY_STAGES",
    "generation_disposition",
    "generation_stage_sequence",
    "resolve_generation_kind",
    "stage_sequence_for_kind",
]
