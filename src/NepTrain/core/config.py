"""Strict project configuration for manual steps and workflows."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Mapping

from ruamel.yaml import YAML


CURRENT_SCHEMA_VERSION = 7


class ConfigError(ValueError):
    """Raised when a project configuration is incomplete or inconsistent."""


_FIELDS: dict[str, set[str]] = {
    "training": {
        "backend",
        "initial_path",
        "test_path",
        "config_path",
        "device",
        "torch_backend",
        "precision",
        "use_compile",
        "restart",
        "restart_steps",
        "finetune_lr_scale",
        "finetune_lr",
    },
    "md": {
        "backend",
        "inference_backend",
        "spin",
    },
    "sampling": {
        "routes",
        "candidate_pool",
        "selection",
    },
    "dft": {
        "backend",
        "gamma_centered",
        "input_path",
        "resource_path",
        "kpoint_mode",
        "kpoints",
        "kspacing",
    },
    "evaluation": {
        "validation_path",
        "inference_backend",
        "max_rmse",
    },
    "workflow": {
        "id",
        "max_model_generations",
        "seed",
    },
    "execution": {
        "poll_interval",
        "stage_targets",
        "sampling_route_targets",
        "targets",
    },
}
_SAMPLING_FIELDS = {
    "candidate_pool": {
        "pre_failure_frames",
        "bad_tail_frames",
        "health",
    },
    "selection": {
        "max_selected",
        "novelty",
    },
}
_ROUTE_FIELDS = {
    "id",
    "structures",
    "template_path",
    "conditions",
    "progression",
}
_ROUTE_CONDITION_FIELDS = {
    "temperature_path",
    "production_temperatures",
    "pressure",
}
_ROUTE_PROGRESSION_FIELDS = {"steps", "replicas"}
_MATURITY_STAGES = (
    "smoke_passed",
    "short_stable",
    "long_stable",
    "production_ready",
)
_TARGET_FIELDS = {
    "executor",
    "host",
    "work_root",
    "command",
    "setup_script",
    "partition",
    "qos",
    "time",
    "cpus_per_task",
    "gpus_per_node",
    "directives",
    "dft_resource_path",
    "environment",
}


def _mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _route_mapping(
    route: Mapping[str, Any], name: str, *, route_id: str
) -> Mapping[str, Any]:
    value = route.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"sampling.routes[{route_id}].{name} must be a mapping")
    return value


def _validate_numeric_path(value: Any, *, field: str) -> list[float]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ConfigError(f"{field} must be a non-empty finite numeric list")
    parsed = [float(item) for item in value]
    if len(set(parsed)) != len(parsed):
        raise ConfigError(f"{field} temperatures must be unique")
    direction = 1 if len(parsed) < 2 or parsed[1] > parsed[0] else -1
    if any(
        (later - earlier) * direction <= 0
        for earlier, later in zip(parsed, parsed[1:])
    ):
        raise ConfigError(f"{field} must be strictly increasing or decreasing")
    return parsed


def _validate_progression(
    progression: Mapping[str, Any], *, route_id: str
) -> None:
    unknown = sorted(set(progression) - _ROUTE_PROGRESSION_FIELDS)
    if unknown:
        raise ConfigError(
            f"unknown sampling.routes[{route_id}].progression fields: "
            + ", ".join(unknown)
        )
    steps = progression.get("steps", {})
    if not isinstance(steps, Mapping):
        raise ConfigError(
            f"sampling.routes[{route_id}].progression.steps must be a mapping"
        )
    if set(steps) != set(_MATURITY_STAGES):
        raise ConfigError(
            f"sampling.routes[{route_id}].progression.steps must define "
            + ", ".join(_MATURITY_STAGES)
        )
    if any(
        isinstance(steps[name], bool) or not isinstance(steps[name], int)
        for name in _MATURITY_STAGES
    ):
        raise ConfigError(
            f"sampling.routes[{route_id}].progression.steps must be integers"
        )
    ordered_steps = [int(steps[name]) for name in _MATURITY_STAGES]
    if any(value < 1 for value in ordered_steps) or any(
        later <= earlier
        for earlier, later in zip(ordered_steps, ordered_steps[1:])
    ):
        raise ConfigError(
            f"sampling.routes[{route_id}].progression.steps must be positive "
            "and strictly increasing"
        )
    replicas = progression.get("replicas")
    if not isinstance(replicas, Mapping) or set(replicas) != set(
        _MATURITY_STAGES
    ):
        raise ConfigError(
            f"sampling.routes[{route_id}].progression.replicas must define "
            + ", ".join(_MATURITY_STAGES)
        )
    if any(
        isinstance(replicas[name], bool) or not isinstance(replicas[name], int)
        for name in _MATURITY_STAGES
    ):
        raise ConfigError(
            f"sampling.routes[{route_id}].progression.replicas values "
            "must be positive integers"
        )
    invalid_replicas = any(
        int(replicas[name]) < 1 for name in _MATURITY_STAGES
    )
    if invalid_replicas:
        raise ConfigError(
            f"sampling.routes[{route_id}].progression.replicas values "
            "must be positive"
        )


def _validate_sampling_routes(sampling: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    routes = sampling.get("routes")
    if not isinstance(routes, list) or not routes:
        raise ConfigError("sampling.routes must be a non-empty list")
    route_ids: set[str] = set()
    parsed: list[Mapping[str, Any]] = []
    for index, raw_route in enumerate(routes):
        if not isinstance(raw_route, Mapping):
            raise ConfigError(f"sampling.routes[{index}] must be a mapping")
        unknown = sorted(set(raw_route) - _ROUTE_FIELDS)
        if unknown:
            raise ConfigError(
                f"unknown sampling.routes[{index}] fields: "
                + ", ".join(unknown)
            )
        route_id_value = raw_route.get("id")
        if not isinstance(route_id_value, str) or not route_id_value.strip():
            raise ConfigError(f"sampling.routes[{index}].id must be a non-empty string")
        route_id = route_id_value.strip()
        if route_id != route_id_value:
            raise ConfigError(
                f"sampling.routes[{index}].id must not contain surrounding whitespace"
            )
        if re.fullmatch(r"[A-Za-z0-9_.-]+", route_id) is None:
            raise ConfigError(
                f"sampling.routes[{index}].id must use only letters, digits, "
                "'.', '_', or '-'"
            )
        if route_id in route_ids:
            raise ConfigError(f"sampling.routes has duplicate id: {route_id}")
        route_ids.add(route_id)

        structures = raw_route.get("structures")
        if (
            not isinstance(structures, list)
            or not structures
            or any(not isinstance(item, str) or not item.strip() for item in structures)
        ):
            raise ConfigError(
                f"sampling.routes[{route_id}].structures must be a non-empty "
                "list of paths"
            )
        template_path = raw_route.get("template_path")
        if not isinstance(template_path, str) or not template_path.strip():
            raise ConfigError(
                f"sampling.routes[{route_id}].template_path is required"
            )

        conditions = _route_mapping(raw_route, "conditions", route_id=route_id)
        unknown_conditions = sorted(set(conditions) - _ROUTE_CONDITION_FIELDS)
        if unknown_conditions:
            raise ConfigError(
                f"unknown sampling.routes[{route_id}].conditions fields: "
                + ", ".join(unknown_conditions)
            )
        temperatures = _validate_numeric_path(
            conditions.get("temperature_path"),
            field=f"sampling.routes[{route_id}].conditions.temperature_path",
        )
        production = conditions.get("production_temperatures")
        if production is not None:
            production_temperatures = _validate_numeric_path(
                production,
                field=(
                    f"sampling.routes[{route_id}].conditions."
                    "production_temperatures"
                ),
            )
            if not set(production_temperatures).issubset(temperatures):
                raise ConfigError(
                    f"sampling.routes[{route_id}].conditions."
                    "production_temperatures must be a subset of temperature_path"
                )
        pressure = conditions.get("pressure")
        if (
            isinstance(pressure, bool)
            or not isinstance(pressure, (int, float))
            or not math.isfinite(float(pressure))
        ):
            raise ConfigError(
                f"sampling.routes[{route_id}].conditions.pressure must be numeric"
            )
        progression = _route_mapping(raw_route, "progression", route_id=route_id)
        _validate_progression(progression, route_id=route_id)
        parsed.append(raw_route)
    return parsed


def _reject_unknown(config: Mapping[str, Any]) -> None:
    allowed_top = {"schema_version", *_FIELDS}
    unknown_top = sorted(set(config) - allowed_top)
    if unknown_top:
        raise ConfigError(
            "unknown project fields: "
            + ", ".join(unknown_top)
            + f"; schema v{CURRENT_SCHEMA_VERSION} does not accept legacy fields"
        )
    for section, allowed in _FIELDS.items():
        value = _mapping(config, section)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ConfigError(
                f"unknown {section} fields: "
                + ", ".join(unknown)
                + "; remove obsolete or misspelled fields"
            )
    sampling = _mapping(config, "sampling")
    for section, allowed in _SAMPLING_FIELDS.items():
        value = _mapping(sampling, section)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ConfigError(
                f"unknown sampling.{section} fields: "
                + ", ".join(unknown)
                + "; remove obsolete or misspelled fields"
            )


def validate_config(config: Mapping[str, Any]) -> None:
    from .execution import ExecutionError, ExecutionTarget
    from .md_policy import TrajectoryHealthError, TrajectoryHealthPolicy

    try:
        schema = int(config.get("schema_version", 0))
    except (TypeError, ValueError) as error:
        raise ConfigError(
            f"schema_version must be {CURRENT_SCHEMA_VERSION}"
        ) from error
    if schema != CURRENT_SCHEMA_VERSION:
        raise ConfigError(
            f"unsupported schema_version {schema}; NepTrain requires "
            f"schema_version {CURRENT_SCHEMA_VERSION} and does not run legacy projects"
        )
    _reject_unknown(config)

    training = _mapping(config, "training")
    md = _mapping(config, "md")
    sampling = _mapping(config, "sampling")
    dft = _mapping(config, "dft")
    workflow = _mapping(config, "workflow")
    evaluation = _mapping(config, "evaluation")
    execution = _mapping(config, "execution")

    if training.get("backend") not in {"gpumd", "torchnep"}:
        raise ConfigError("training.backend must be gpumd or torchnep")
    if not training.get("initial_path"):
        raise ConfigError("training.initial_path is required")
    if not training.get("config_path"):
        raise ConfigError("training.config_path is required")
    if float(training.get("finetune_lr_scale", 0.1)) <= 0:
        raise ConfigError("training.finetune_lr_scale must be positive")
    if training.get("finetune_lr") is not None and float(
        training["finetune_lr"]
    ) <= 0:
        raise ConfigError("training.finetune_lr must be positive")

    if md.get("backend") not in {"gpumd", "lammps"}:
        raise ConfigError("md.backend must be gpumd or lammps")
    if md.get("inference_backend", "auto") not in {"auto", "cpu", "cuda"}:
        raise ConfigError("md.inference_backend must be auto, cpu, or cuda")
    sampling_routes = _validate_sampling_routes(sampling)
    candidate_pool = _mapping(sampling, "candidate_pool")
    selection = _mapping(sampling, "selection")
    if int(candidate_pool.get("pre_failure_frames", 2)) < 0:
        raise ConfigError(
            "sampling.candidate_pool.pre_failure_frames must be non-negative"
        )
    if int(candidate_pool.get("bad_tail_frames", 1)) < 1:
        raise ConfigError(
            "sampling.candidate_pool.bad_tail_frames must be at least 1"
        )
    try:
        TrajectoryHealthPolicy.from_mapping(candidate_pool.get("health", {}))
    except TrajectoryHealthError as error:
        raise ConfigError(str(error)) from error
    if int(selection.get("max_selected", 0)) < 1:
        raise ConfigError("sampling.selection.max_selected must be positive")
    novelty = selection.get("novelty", "auto")
    if novelty != "auto":
        if not isinstance(novelty, Mapping):
            raise ConfigError(
                "sampling.selection.novelty must be auto or a mapping"
            )
        unknown_novelty = sorted(
            set(novelty)
            - {"selection_threshold", "completion_threshold"}
        )
        if unknown_novelty:
            raise ConfigError(
                "sampling.selection.novelty has unknown fields: "
                + ", ".join(unknown_novelty)
            )
        if set(novelty) != {
            "selection_threshold",
            "completion_threshold",
        }:
            raise ConfigError(
                "explicit sampling.selection.novelty must define "
                "selection_threshold and completion_threshold"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in novelty.values()
        ):
            raise ConfigError(
                "sampling.selection.novelty thresholds must be finite and non-negative"
            )

    if dft.get("backend", "vasp") not in {"vasp", "abacus", "toy"}:
        raise ConfigError("dft.backend must be vasp, abacus, or toy")
    kpoint_mode = dft.get("kpoint_mode", "auto")
    if kpoint_mode not in {"auto", "kspacing", "kpoints"}:
        raise ConfigError(
            "dft.kpoint_mode must be auto, kspacing, or kpoints"
        )
    if not isinstance(dft.get("gamma_centered", False), bool):
        raise ConfigError("dft.gamma_centered must be boolean")
    if dft.get("backend", "vasp") != "toy" and kpoint_mode == "kspacing":
        kspacing = dft.get("kspacing")
        if (
            kspacing is None
            or isinstance(kspacing, bool)
            or not isinstance(kspacing, (int, float))
            or not math.isfinite(float(kspacing))
            or float(kspacing) <= 0
        ):
            raise ConfigError(
                "dft.kspacing must be positive when dft.kpoint_mode=kspacing"
            )
        if "kpoints" in dft:
            raise ConfigError(
                "dft.kpoints cannot be combined with dft.kpoint_mode=kspacing"
            )
    elif dft.get("backend", "vasp") != "toy" and kpoint_mode == "kpoints":
        raw_kpoints = dft.get("kpoints")
        values = (
            [raw_kpoints]
            if isinstance(raw_kpoints, int)
            else raw_kpoints
        )
        if (
            not isinstance(values, list)
            or len(values) not in {1, 3}
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in values
            )
        ):
            raise ConfigError(
                "dft.kpoints must contain one or three positive integers "
                "when dft.kpoint_mode=kpoints"
            )
        if "kspacing" in dft:
            raise ConfigError(
                "dft.kspacing cannot be combined with dft.kpoint_mode=kpoints"
            )
    elif dft.get("backend", "vasp") != "toy":
        if "kspacing" in dft or "kpoints" in dft:
            raise ConfigError(
                "dft.kspacing and dft.kpoints require an explicit "
                "dft.kpoint_mode; auto reads k-point settings from the DFT input"
            )

    if md.get("spin", False):
        if md.get("backend") != "lammps":
            raise ConfigError("spin MD currently requires md.backend=lammps")
        if dft.get("backend", "vasp") not in {"abacus", "toy"}:
            raise ConfigError(
                "spin workflows require dft.backend=abacus or toy; "
                "VASP labeling is non-magnetic only"
            )

    if workflow:
        if int(workflow.get("max_model_generations", 0)) < 1:
            raise ConfigError("workflow.max_model_generations must be positive")
        if int(workflow.get("seed", 20260721)) < 0:
            raise ConfigError("workflow.seed must be non-negative")
    if evaluation:
        if not evaluation.get("validation_path"):
            raise ConfigError(
                "evaluation.validation_path is required when evaluation is configured"
            )
        thresholds = dict(evaluation.get("max_rmse") or {})
        required = {"energy_rmse", "force_rmse"}
        if md.get("spin", False):
            required.add("mforce_rmse")
        missing = sorted(required - set(thresholds))
        if missing:
            raise ConfigError(
                "evaluation.max_rmse is missing " + ", ".join(missing)
            )
        if any(float(value) <= 0 for value in thresholds.values()):
            raise ConfigError("evaluation.max_rmse values must be positive")

    try:
        interval = float(execution.get("poll_interval", 30))
    except (TypeError, ValueError) as error:
        raise ConfigError("execution.poll_interval must be numeric") from error
    if interval < 0.2:
        raise ConfigError("execution.poll_interval must be at least 0.2 seconds")
    targets = _mapping(execution, "targets")
    routes = _mapping(execution, "stage_targets")
    required_routes = {"training", "sampling", "labeling", "analysis"}
    missing_routes = sorted(required_routes - set(routes))
    if missing_routes:
        raise ConfigError(
            "execution.stage_targets is missing " + ", ".join(missing_routes)
        )
    unknown_routes = sorted(set(routes) - required_routes)
    if unknown_routes:
        raise ConfigError(
            "execution.stage_targets has unknown roles: "
            + ", ".join(unknown_routes)
        )
    invalid_stage_targets = sorted(
        str(key)
        for key, value in routes.items()
        if not isinstance(value, str) or not value
    )
    if invalid_stage_targets:
        raise ConfigError(
            "execution.stage_targets values must be non-empty target names: "
            + ", ".join(invalid_stage_targets)
        )
    for name, raw_target in targets.items():
        if not isinstance(raw_target, Mapping):
            raise ConfigError(f"execution.targets.{name} must be a mapping")
        unknown = sorted(set(raw_target) - _TARGET_FIELDS)
        if unknown:
            raise ConfigError(
                f"execution.targets.{name} has unknown fields: "
                + ", ".join(unknown)
            )
    try:
        parsed_targets = {
            str(name): ExecutionTarget.from_mapping(str(name), value)
            for name, value in targets.items()
        }
    except (ExecutionError, TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error
    unknown_targets = sorted(set(routes.values()) - set(parsed_targets))
    if unknown_targets:
        raise ConfigError(
            "execution.stage_targets refers to unknown targets: "
            + ", ".join(str(value) for value in unknown_targets)
        )
    route_targets = _mapping(execution, "sampling_route_targets")
    invalid_route_targets = sorted(
        str(key)
        for key, value in route_targets.items()
        if not isinstance(value, str) or not value
    )
    if invalid_route_targets:
        raise ConfigError(
            "execution.sampling_route_targets values must be non-empty target names: "
            + ", ".join(invalid_route_targets)
        )
    unknown_route_ids = sorted(
        set(route_targets) - {str(route["id"]) for route in sampling_routes}
    )
    if unknown_route_ids:
        raise ConfigError(
            "execution.sampling_route_targets has unknown sampling route ids: "
            + ", ".join(unknown_route_ids)
        )
    unknown_sampling_targets = sorted(
        set(route_targets.values()) - set(parsed_targets)
    )
    if unknown_sampling_targets:
        raise ConfigError(
            "execution.sampling_route_targets refers to unknown targets: "
            + ", ".join(str(value) for value in unknown_sampling_targets)
        )


def load_config(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = YAML(typ="safe").load(handle) or {}
    if not isinstance(config, Mapping):
        raise ConfigError("project file must contain a YAML mapping")
    value = dict(config)
    validate_config(value)
    return value, []


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    validate_config(config)
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.dump(dict(config), handle)
