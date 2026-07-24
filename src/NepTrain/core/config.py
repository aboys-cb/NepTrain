"""Strict project configuration for manual steps and workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ruamel.yaml import YAML


CURRENT_SCHEMA_VERSION = 5


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
        "seed",
    },
    "md": {
        "backend",
        "inference_backend",
        "structures",
        "template_path",
        "spin",
        "lmp",
        "mpiexec",
        "mpi_ranks",
    },
    "sampling": {
        "mode",
        "conditions",
        "progression",
        "candidate_pool",
        "selection",
    },
    "dft": {
        "backend",
        "n_cpu",
        "kpoints_use_gamma",
        "input_path",
        "resource_path",
        "use_k_stype",
        "kpoints",
        "kspacing",
        "teacher_profile",
    },
    "evaluation": {
        "validation_path",
        "inference_backend",
        "max_rmse",
    },
    "workflow": {
        "id",
        "max_iterations",
        "seed",
    },
    "execution": {"poll_interval", "routes", "targets"},
}
_SAMPLING_FIELDS = {
    "conditions": {
        "temperature_path",
        "production_temperatures",
        "pressure",
        "spin_temperature",
    },
    "progression": {"md_runs_per_iteration", "steps", "replicas"},
    "candidate_pool": {
        "pre_failure_frames",
        "bad_tail_frames",
        "health",
    },
    "selection": {
        "method",
        "max_selected",
        "min_novelty",
    },
}
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
    "overrides",
    "environment",
}


def _mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


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
    from .scenario import ScenarioLadder, ScenarioMaturityError

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
    if int(training.get("seed", workflow.get("seed", 20260723))) < 0:
        raise ConfigError("training.seed must be non-negative")
    if training.get("finetune_lr") is not None and float(
        training["finetune_lr"]
    ) <= 0:
        raise ConfigError("training.finetune_lr must be positive")

    if md.get("backend") not in {"gpumd", "lammps"}:
        raise ConfigError("md.backend must be gpumd or lammps")
    if md.get("inference_backend", "auto") not in {"auto", "cpu", "cuda"}:
        raise ConfigError("md.inference_backend must be auto, cpu, or cuda")
    if not md.get("structures"):
        raise ConfigError("md.structures is required")
    if int(md.get("mpi_ranks", 1)) < 1:
        raise ConfigError("md.mpi_ranks must be at least 1")

    if sampling.get("mode", "auto") != "auto":
        raise ConfigError("sampling.mode currently must be auto")
    conditions = _mapping(sampling, "conditions")
    progression = _mapping(sampling, "progression")
    candidate_pool = _mapping(sampling, "candidate_pool")
    selection = _mapping(sampling, "selection")
    temperatures = conditions.get("temperature_path")
    if (
        not isinstance(temperatures, list)
        or not temperatures
        or any(not isinstance(value, (int, float)) for value in temperatures)
    ):
        raise ConfigError(
            "sampling.conditions.temperature_path must be a non-empty numeric list"
        )
    production_temperatures = conditions.get("production_temperatures")
    if production_temperatures is not None and (
        not isinstance(production_temperatures, list)
        or not production_temperatures
        or any(
            not isinstance(value, (int, float))
            for value in production_temperatures
        )
    ):
        raise ConfigError(
            "sampling.conditions.production_temperatures must be a "
            "non-empty numeric list when provided"
        )
    if not isinstance(conditions.get("pressure", 0.0), (int, float)):
        raise ConfigError("sampling.conditions.pressure must be numeric")
    steps = _mapping(progression, "steps")
    expected_steps = {
        "smoke_passed",
        "short_stable",
        "long_stable",
        "production_ready",
    }
    if set(steps) != expected_steps:
        raise ConfigError(
            "sampling.progression.steps must define smoke_passed, "
            "short_stable, long_stable, and production_ready"
        )
    ordered_steps = [
        int(steps[name])
        for name in (
            "smoke_passed",
            "short_stable",
            "long_stable",
            "production_ready",
        )
    ]
    if any(value < 1 for value in ordered_steps) or any(
        later <= earlier for earlier, later in zip(ordered_steps, ordered_steps[1:])
    ):
        raise ConfigError(
            "sampling.progression.steps must be positive and strictly increasing"
        )
    replicas = progression.get("replicas")
    if replicas is not None:
        if not isinstance(replicas, Mapping) or set(replicas) != expected_steps:
            raise ConfigError(
                "sampling.progression.replicas must define smoke_passed, "
                "short_stable, long_stable, and production_ready"
            )
        if any(int(value) < 1 for value in replicas.values()):
            raise ConfigError(
                "sampling.progression.replicas values must be positive"
            )
    if int(progression.get("md_runs_per_iteration", 1)) < 1:
        raise ConfigError(
            "sampling.progression.md_runs_per_iteration must be positive"
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
    if selection.get("method", "fps") != "fps":
        raise ConfigError("sampling.selection.method currently must be fps")
    if int(selection.get("max_selected", 0)) < 1:
        raise ConfigError("sampling.selection.max_selected must be positive")
    if float(selection.get("min_novelty", 0.0)) < 0:
        raise ConfigError("sampling.selection.min_novelty must be non-negative")

    if dft.get("backend", "vasp") not in {"vasp", "abacus", "toy"}:
        raise ConfigError("dft.backend must be vasp, abacus, or toy")
    if dft.get("teacher_profile", "ordinary") not in {"ordinary", "spin"}:
        raise ConfigError("dft.teacher_profile must be ordinary or spin")
    if int(dft.get("n_cpu", 1)) < 1:
        raise ConfigError("dft.n_cpu must be positive")
    if dft.get("use_k_stype", "kspacing") not in {"kspacing", "kpoints"}:
        raise ConfigError("dft.use_k_stype must be kspacing or kpoints")

    if md.get("spin", False):
        if md.get("backend") != "lammps":
            raise ConfigError("spin MD currently requires md.backend=lammps")
        if conditions.get("spin_temperature") is None:
            raise ConfigError(
                "spin MD requires sampling.conditions.spin_temperature"
            )
        if dft.get("backend", "vasp") not in {"abacus", "toy"}:
            raise ConfigError(
                "spin workflows require dft.backend=abacus or toy; "
                "VASP labeling is non-magnetic only"
            )

    if workflow:
        if int(workflow.get("max_iterations", 0)) < 1:
            raise ConfigError("workflow.max_iterations must be positive")
        if not evaluation.get("validation_path"):
            raise ConfigError("evaluation.validation_path is required for workflows")
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
        ScenarioLadder.from_sampling(sampling)
    except ScenarioMaturityError as error:
        raise ConfigError(str(error)) from error

    try:
        interval = float(execution.get("poll_interval", 30))
    except (TypeError, ValueError) as error:
        raise ConfigError("execution.poll_interval must be numeric") from error
    if interval < 0.2:
        raise ConfigError("execution.poll_interval must be at least 0.2 seconds")
    targets = _mapping(execution, "targets")
    routes = _mapping(execution, "routes")
    required_routes = {"training", "sampling", "labeling", "analysis"}
    missing_routes = sorted(required_routes - set(routes))
    if missing_routes:
        raise ConfigError(
            "execution.routes is missing " + ", ".join(missing_routes)
        )
    unknown_routes = sorted(set(routes) - required_routes)
    if unknown_routes:
        raise ConfigError(
            "execution.routes has unknown roles: " + ", ".join(unknown_routes)
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
        overrides = raw_target.get("overrides", {})
        if not isinstance(overrides, Mapping):
            raise ConfigError(
                f"execution.targets.{name}.overrides must be a mapping"
            )
        invalid_overrides = []
        for dotted in overrides:
            parts = str(dotted).split(".")
            if (
                len(parts) != 2
                or parts[0] not in _FIELDS
                or parts[1] not in _FIELDS[parts[0]]
            ):
                invalid_overrides.append(str(dotted))
        if invalid_overrides:
            raise ConfigError(
                f"execution.targets.{name} has unknown overrides: "
                + ", ".join(sorted(invalid_overrides))
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
            "execution.routes refers to unknown targets: "
            + ", ".join(str(value) for value in unknown_targets)
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
