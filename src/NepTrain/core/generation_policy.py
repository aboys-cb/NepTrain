"""Resolve immutable generation kinds and their allowed stage sequences."""

from __future__ import annotations

from typing import Any, Mapping


LEGACY_GENERATION_PROTOCOL = "legacy_v1"
ADAPTIVE_GENERATION_PROTOCOL = "adaptive_v2"
ACTIVE_LEARNING_GENERATION_PROTOCOL = "active_learning_v3"

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
ACTIVE_LEARNING_ACQUISITION_STAGES = (
    "train",
    "validate",
    "explore",
    "select",
    "label",
    "evaluate",
    "update",
)
ACTIVE_LEARNING_FINALIZATION_STAGES = ("train", "validate")

_KIND_SEQUENCES = {
    "legacy": (LEGACY_STAGES,),
    "acquisition": (
        ACQUISITION_STAGES,
        ACTIVE_LEARNING_ACQUISITION_STAGES,
    ),
    "finalization": (
        FINALIZATION_STAGES,
        ACTIVE_LEARNING_FINALIZATION_STAGES,
    ),
}


def generation_stage_sequence(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return a committed sequence, falling back to the legacy contract."""

    raw = record.get("stage_sequence")
    if raw is None:
        return LEGACY_STAGES
    sequence = tuple(str(stage) for stage in raw)
    kind = str(record.get("kind", ""))
    if kind not in _KIND_SEQUENCES or sequence not in _KIND_SEQUENCES[kind]:
        raise ValueError("generation kind and stage sequence are inconsistent")
    return sequence


def stage_implementation(
    stage: str,
    stage_sequence: tuple[str, ...],
) -> str:
    """Return the existing scientific implementation for one public stage."""

    if stage_sequence in {
        ACTIVE_LEARNING_ACQUISITION_STAGES,
        ACTIVE_LEARNING_FINALIZATION_STAGES,
    }:
        return {
            "validate": "evaluate",
            "evaluate": "diagnose",
            "update": "merge",
        }.get(stage, stage)
    return stage


def stage_for_role(record: Mapping[str, Any], role: str) -> str | None:
    """Resolve one semantic role to the persisted stage name."""

    sequence = generation_stage_sequence(record)
    if sequence in {
        ACTIVE_LEARNING_ACQUISITION_STAGES,
        ACTIVE_LEARNING_FINALIZATION_STAGES,
    }:
        candidate = role
    else:
        candidate = {
            "validate": "evaluate",
            "evaluate": "diagnose",
            "update": "merge",
        }.get(role, role)
    return candidate if candidate in sequence else None


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
    if protocol not in {
        ADAPTIVE_GENERATION_PROTOCOL,
        ACTIVE_LEARNING_GENERATION_PROTOCOL,
    }:
        raise ValueError(f"unsupported generation protocol: {protocol}")
    return (
        "finalization"
        if generation_disposition(previous_record) == "finalize"
        else "acquisition"
    )


def stage_sequence_for_kind(
    kind: str,
    protocol: str = ADAPTIVE_GENERATION_PROTOCOL,
) -> tuple[str, ...]:
    if kind == "legacy":
        return LEGACY_STAGES
    if protocol == ADAPTIVE_GENERATION_PROTOCOL:
        sequences = {
            "acquisition": ACQUISITION_STAGES,
            "finalization": FINALIZATION_STAGES,
        }
    elif protocol == ACTIVE_LEARNING_GENERATION_PROTOCOL:
        sequences = {
            "acquisition": ACTIVE_LEARNING_ACQUISITION_STAGES,
            "finalization": ACTIVE_LEARNING_FINALIZATION_STAGES,
        }
    else:
        raise ValueError(f"unsupported generation protocol: {protocol}")
    try:
        return sequences[kind]
    except KeyError as error:
        raise ValueError(f"unsupported generation kind: {kind}") from error


__all__ = [
    "ACQUISITION_STAGES",
    "ACTIVE_LEARNING_ACQUISITION_STAGES",
    "ACTIVE_LEARNING_FINALIZATION_STAGES",
    "ACTIVE_LEARNING_GENERATION_PROTOCOL",
    "ADAPTIVE_GENERATION_PROTOCOL",
    "FINALIZATION_STAGES",
    "LEGACY_GENERATION_PROTOCOL",
    "LEGACY_STAGES",
    "generation_disposition",
    "generation_stage_sequence",
    "resolve_generation_kind",
    "stage_for_role",
    "stage_implementation",
    "stage_sequence_for_kind",
]
