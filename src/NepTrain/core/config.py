"""Strict project configuration for manual steps and workflows."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Mapping

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from .composition import reduced_composition
from .sampling_route import MATURITY_STAGES


CURRENT_SCHEMA_VERSION = 8
DEFAULT_MAX_CONCURRENT = 20
DEFAULT_STRUCTURES_PER_LABEL_JOB = 1
DEFAULT_STRUCTURES_PER_MODEL_JOB = 64


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
        "excluded_compositions",
        "selection",
    },
    "labeling": {
        "backend",
        "gamma_centered",
        "input_path",
        "resource_path",
        "potcar_manifest_path",
        "resource_manifest_path",
        "kpoint_mode",
        "kpoints",
        "kspacing",
        "structures_per_job",
        "max_concurrent",
        "model_path",
        "model_name",
        "runner",
        "device",
        "precision",
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
        "convergence",
    },
    "execution": {
        "poll_interval",
        "stage_targets",
        "sampling_route_targets",
        "targets",
    },
    "notifications": {
        "feishu",
    },
}
_SAMPLING_FIELDS = {
    "candidate_pool": {
        "pre_failure_frames",
        "bad_tail_frames",
        "health",
    },
    "selection": {
        "descriptor_reduction",
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
    "memory_ladder",
    "labeling_resource_path",
    "environment",
}
_FEISHU_NOTIFICATION_FIELDS = {
    "webhook",
    "secret",
    "timeout_seconds",
}
_WORKFLOW_CONVERGENCE_FIELDS = {
    "acquisition_max_rmse",
    "acquisition_min_r2",
    "group_min_force_r2",
    "max_outlier_fraction",
    "min_selected",
    "consecutive_generations",
}
_RMSE_FIELDS = {
    "energy_rmse",
    "force_rmse",
    "virial_rmse",
    "mforce_rmse",
}
_R2_FIELDS = {
    "energy_r2",
    "force_r2",
    "virial_r2",
    "mforce_r2",
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
    if not progression:
        return
    steps = progression.get("steps", {})
    if not isinstance(steps, Mapping):
        raise ConfigError(
            f"sampling.routes[{route_id}].progression.steps must be a mapping"
        )
    if set(steps) != set(MATURITY_STAGES):
        raise ConfigError(
            f"sampling.routes[{route_id}].progression.steps must define "
            + ", ".join(MATURITY_STAGES)
        )
    if any(
        isinstance(steps[name], bool) or not isinstance(steps[name], int)
        for name in MATURITY_STAGES
    ):
        raise ConfigError(
            f"sampling.routes[{route_id}].progression.steps must be integers"
        )
    ordered_steps = [int(steps[name]) for name in MATURITY_STAGES]
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
        MATURITY_STAGES
    ):
        raise ConfigError(
            f"sampling.routes[{route_id}].progression.replicas must define "
            + ", ".join(MATURITY_STAGES)
        )
    if any(
        isinstance(replicas[name], bool) or not isinstance(replicas[name], int)
        for name in MATURITY_STAGES
    ):
        raise ConfigError(
            f"sampling.routes[{route_id}].progression.replicas values "
            "must be positive integers"
        )
    invalid_replicas = any(
        int(replicas[name]) < 1 for name in MATURITY_STAGES
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
        pressure = conditions.get("pressure", 0.0)
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
    notifications = _mapping(config, "notifications")
    feishu = _mapping(notifications, "feishu")
    unknown = sorted(set(feishu) - _FEISHU_NOTIFICATION_FIELDS)
    if unknown:
        raise ConfigError(
            "unknown notifications.feishu fields: "
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
    labeling = _mapping(config, "labeling")
    workflow = _mapping(config, "workflow")
    evaluation = _mapping(config, "evaluation")
    execution = _mapping(config, "execution")
    notifications = _mapping(config, "notifications")

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
    excluded_compositions = sampling.get("excluded_compositions", [])
    if not isinstance(excluded_compositions, list) or any(
        not isinstance(value, str) or not value.strip()
        for value in excluded_compositions
    ):
        raise ConfigError(
            "sampling.excluded_compositions must be a list of chemical formulas"
        )
    excluded_keys = []
    for formula in excluded_compositions:
        try:
            excluded_keys.append(reduced_composition(formula))
        except (KeyError, ValueError) as error:
            raise ConfigError(
                "sampling.excluded_compositions contains an invalid chemical "
                f"formula: {formula}"
            ) from error
    if len(set(excluded_keys)) != len(excluded_keys):
        raise ConfigError(
            "sampling.excluded_compositions contains duplicate reduced compositions"
        )
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
    if int(selection.get("max_selected", 100)) < 1:
        raise ConfigError("sampling.selection.max_selected must be positive")
    if selection.get("descriptor_reduction", "global_mean") not in {
        "global_mean",
        "elementwise_mean_std",
    }:
        raise ConfigError(
            "sampling.selection.descriptor_reduction must be global_mean "
            "or elementwise_mean_std"
        )
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

    backend = labeling.get("backend", "vasp")
    if backend not in {"vasp", "abacus", "model", "toy"}:
        raise ConfigError(
            "labeling.backend must be vasp, abacus, model, or toy"
        )
    if backend == "vasp":
        manifest_path = labeling.get("potcar_manifest_path")
        if not isinstance(manifest_path, str) or not manifest_path.strip():
            raise ConfigError(
                "labeling.potcar_manifest_path is required for VASP so POTCAR "
                "versions and hashes are part of task identity"
            )
    if backend == "abacus":
        manifest_path = labeling.get("resource_manifest_path")
        if not isinstance(manifest_path, str) or not manifest_path.strip():
            raise ConfigError(
                "labeling.resource_manifest_path is required for ABACUS so "
                "pseudopotential and orbital hashes are part of task identity"
            )
    if backend == "model":
        for field in ("model_path", "model_name", "runner"):
            value = labeling.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(
                    f"labeling.{field} is required when labeling.backend=model"
                )
        if labeling.get("device", "cuda") not in {"cpu", "cuda"}:
            raise ConfigError("labeling.device must be cpu or cuda")
        if labeling.get("precision", "float32") not in {
            "float32",
            "float64",
        }:
            raise ConfigError(
                "labeling.precision must be float32 or float64"
            )
    kpoint_mode = labeling.get("kpoint_mode", "auto")
    if backend in {"vasp", "abacus"} and kpoint_mode not in {
        "auto",
        "kspacing",
        "kpoints",
    }:
        raise ConfigError(
            "labeling.kpoint_mode must be auto, kspacing, or kpoints"
        )
    if not isinstance(labeling.get("gamma_centered", False), bool):
        raise ConfigError("labeling.gamma_centered must be boolean")
    for field, default in (
        ("structures_per_job", DEFAULT_STRUCTURES_PER_LABEL_JOB),
        ("max_concurrent", DEFAULT_MAX_CONCURRENT),
    ):
        value = labeling.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"labeling.{field} must be a positive integer")
    if backend in {"vasp", "abacus"} and kpoint_mode == "kspacing":
        kspacing = labeling.get("kspacing")
        if (
            kspacing is None
            or isinstance(kspacing, bool)
            or not isinstance(kspacing, (int, float))
            or not math.isfinite(float(kspacing))
            or float(kspacing) <= 0
        ):
            raise ConfigError(
                "labeling.kspacing must be positive when "
                "labeling.kpoint_mode=kspacing"
            )
        if "kpoints" in labeling:
            raise ConfigError(
                "labeling.kpoints cannot be combined with "
                "labeling.kpoint_mode=kspacing"
            )
    elif backend in {"vasp", "abacus"} and kpoint_mode == "kpoints":
        raw_kpoints = labeling.get("kpoints")
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
                "labeling.kpoints must contain one or three positive integers "
                "when labeling.kpoint_mode=kpoints"
            )
        if "kspacing" in labeling:
            raise ConfigError(
                "labeling.kspacing cannot be combined with "
                "labeling.kpoint_mode=kpoints"
            )
    elif backend in {"vasp", "abacus"}:
        if "kspacing" in labeling or "kpoints" in labeling:
            raise ConfigError(
                "labeling.kspacing and labeling.kpoints require an explicit "
                "labeling.kpoint_mode; auto reads k-point settings from the "
                "backend input"
            )

    if md.get("spin", False):
        if md.get("backend") != "lammps":
            raise ConfigError("spin MD currently requires md.backend=lammps")
        if backend not in {"abacus", "model", "toy"}:
            raise ConfigError(
                "spin workflows require labeling.backend=abacus, model, "
                "or toy; "
                "VASP collinear ISPIN=2 produces ordinary energy/force "
                "labels, not spin/mforce labels"
            )

    if workflow:
        workflow_id = workflow.get("id")
        if workflow_id is not None and (
            not isinstance(workflow_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", workflow_id)
            is None
        ):
            raise ConfigError(
                "workflow.id must be 1-64 safe characters: letters, numbers, "
                "dot, underscore, or hyphen, and must start with a letter or number"
            )
        if int(workflow.get("max_model_generations", 0)) < 1:
            raise ConfigError("workflow.max_model_generations must be positive")
        if int(workflow.get("seed", 20260721)) < 0:
            raise ConfigError("workflow.seed must be non-negative")
        convergence = _mapping(workflow, "convergence")
        unknown_convergence = sorted(
            set(convergence) - _WORKFLOW_CONVERGENCE_FIELDS
        )
        if unknown_convergence:
            raise ConfigError(
                "workflow.convergence has unknown fields: "
                + ", ".join(unknown_convergence)
            )
        if convergence:
            thresholds = _mapping(convergence, "acquisition_max_rmse")
            unknown_thresholds = sorted(set(thresholds) - _RMSE_FIELDS)
            if unknown_thresholds:
                raise ConfigError(
                    "workflow.convergence.acquisition_max_rmse has unknown "
                    "metrics: " + ", ".join(unknown_thresholds)
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
                for value in thresholds.values()
            ):
                raise ConfigError(
                    "workflow.convergence.acquisition_max_rmse values must "
                    "be finite and positive"
                )
            if thresholds:
                required_rmse = {"energy_rmse", "force_rmse"}
                if md.get("spin", False):
                    required_rmse.add("mforce_rmse")
                missing = sorted(required_rmse - set(thresholds))
                if missing:
                    raise ConfigError(
                        "workflow.convergence.acquisition_max_rmse is missing "
                        + ", ".join(missing)
                    )
            r2_thresholds = _mapping(convergence, "acquisition_min_r2")
            unknown_r2 = sorted(set(r2_thresholds) - _R2_FIELDS)
            if unknown_r2:
                raise ConfigError(
                    "workflow.convergence.acquisition_min_r2 has unknown "
                    "metrics: " + ", ".join(unknown_r2)
                )
            if not thresholds and not r2_thresholds:
                raise ConfigError(
                    "workflow.convergence requires acquisition_min_r2 or "
                    "acquisition_max_rmse"
                )
            required_r2 = {"energy_r2", "force_r2"}
            if md.get("spin", False):
                required_r2.add("mforce_r2")
            if r2_thresholds:
                missing = sorted(required_r2 - set(r2_thresholds))
                if missing:
                    raise ConfigError(
                        "workflow.convergence.acquisition_min_r2 is missing "
                        + ", ".join(missing)
                    )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in r2_thresholds.values()
            ):
                raise ConfigError(
                    "workflow.convergence.acquisition_min_r2 values must "
                    "be finite and between 0 and 1"
                )
            group_min = convergence.get("group_min_force_r2")
            if group_min is not None and (
                isinstance(group_min, bool)
                or not isinstance(group_min, (int, float))
                or not math.isfinite(float(group_min))
                or not 0.0 <= float(group_min) <= 1.0
            ):
                raise ConfigError(
                    "workflow.convergence.group_min_force_r2 must be finite "
                    "and between 0 and 1"
                )
            max_outliers = convergence.get("max_outlier_fraction")
            if max_outliers is not None and (
                isinstance(max_outliers, bool)
                or not isinstance(max_outliers, (int, float))
                or not math.isfinite(float(max_outliers))
                or not 0.0 <= float(max_outliers) <= 1.0
            ):
                raise ConfigError(
                    "workflow.convergence.max_outlier_fraction must be finite "
                    "and between 0 and 1"
                )
            min_selected = convergence.get("min_selected", 1)
            if (
                isinstance(min_selected, bool)
                or not isinstance(min_selected, int)
                or min_selected < 1
            ):
                raise ConfigError(
                    "workflow.convergence.min_selected must be a positive integer"
                )
            consecutive = convergence.get("consecutive_generations", 1)
            if (
                isinstance(consecutive, bool)
                or not isinstance(consecutive, int)
                or consecutive < 1
            ):
                raise ConfigError(
                    "workflow.convergence.consecutive_generations must be "
                    "a positive integer"
                )
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

    feishu = _mapping(notifications, "feishu")
    if feishu:
        for field in ("webhook", "secret"):
            value = feishu.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(
                    f"notifications.feishu.{field} must be a non-empty string"
                )
        if not str(feishu["webhook"]).startswith(
            "https://open.feishu.cn/open-apis/bot/v2/hook/"
        ):
            raise ConfigError(
                "notifications.feishu.webhook must be a Feishu custom-bot "
                "v2 webhook"
            )
        timeout = feishu.get("timeout_seconds", 5)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or not 0.2 <= float(timeout) <= 30
        ):
            raise ConfigError(
                "notifications.feishu.timeout_seconds must be between "
                "0.2 and 30 seconds"
            )

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
    if backend in {"vasp", "abacus"}:
        labeling_target = parsed_targets[str(routes["labeling"])]
        if not (
            labeling_target.labeling_resource_path
            or (
                isinstance(labeling.get("resource_path"), str)
                and str(labeling["resource_path"]).strip()
            )
        ):
            raise ConfigError(
                "VASP/ABACUS labeling requires labeling.resource_path or "
                "execution.targets.<labeling>.labeling_resource_path"
            )
        if (
            labeling_target.host
            and not labeling_target.labeling_resource_path
        ):
            raise ConfigError(
                "a remote labeling target requires its own absolute "
                "labeling_resource_path; a local project resource path is "
                "not portable over SSH"
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
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = YAML(typ="safe").load(handle) or {}
    except OSError as error:
        raise ConfigError(
            f"cannot read project configuration {config_path}: {error}"
        ) from error
    except YAMLError as error:
        raise ConfigError(
            f"invalid YAML in project configuration {config_path}: {error}"
        ) from error
    if not isinstance(config, Mapping):
        raise ConfigError("project file must contain a YAML mapping")
    value = dict(config)
    try:
        validate_config(value)
    except ConfigError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise ConfigError(f"invalid project configuration value: {error}") from error
    return value, []


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    validate_config(config)
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.dump(dict(config), handle)
