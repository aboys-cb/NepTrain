"""Strict project configuration for manual steps and workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ruamel.yaml import YAML


CURRENT_SCHEMA_VERSION = 4


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
        "ensemble",
        "temperatures",
        "pressure",
        "initial_steps",
        "timestep",
        "tdamp",
        "pdamp",
        "dump_interval",
        "spin",
        "spin_temperature",
        "spin_alpha",
        "spin_seed",
        "midpoint_iter",
        "lmp",
        "mpiexec",
        "mpi_ranks",
        "plugin_path",
        "pre_failure_frames",
        "bad_tail_frames",
        "health",
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
        "generations",
        "seed",
        "initial_candidates",
        "dft_budget",
        "minimum_dft_budget",
        "frame_stride",
        "min_distance",
        "maturity",
    },
    "execution": {"poll_interval", "routes", "targets"},
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
            + "; schema v4 does not accept legacy fields"
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


def validate_config(config: Mapping[str, Any]) -> None:
    from .execution import ExecutionError, ExecutionTarget
    from .md_policy import TrajectoryHealthError, TrajectoryHealthPolicy
    from .scenario import ScenarioLadder, ScenarioMaturityError

    try:
        schema = int(config.get("schema_version", 0))
    except (TypeError, ValueError) as error:
        raise ConfigError("schema_version must be 4") from error
    if schema != CURRENT_SCHEMA_VERSION:
        raise ConfigError(
            f"unsupported schema_version {schema}; NepTrain requires "
            f"schema_version {CURRENT_SCHEMA_VERSION} and does not run legacy projects"
        )
    _reject_unknown(config)

    training = _mapping(config, "training")
    md = _mapping(config, "md")
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
    temperatures = md.get("temperatures")
    if (
        not isinstance(temperatures, list)
        or not temperatures
        or any(not isinstance(value, (int, float)) for value in temperatures)
    ):
        raise ConfigError("md.temperatures must be a non-empty numeric list")
    if int(md.get("initial_steps", 0)) < 1:
        raise ConfigError("md.initial_steps must be positive")
    if int(md.get("mpi_ranks", 1)) < 1:
        raise ConfigError("md.mpi_ranks must be at least 1")
    if int(md.get("dump_interval", 100)) < 1:
        raise ConfigError("md.dump_interval must be positive")
    if float(md.get("tdamp", 0.1)) <= 0 or float(md.get("pdamp", 1.0)) <= 0:
        raise ConfigError("md.tdamp and md.pdamp must be positive")
    if int(md.get("pre_failure_frames", 2)) < 0:
        raise ConfigError("md.pre_failure_frames must be non-negative")
    if int(md.get("bad_tail_frames", 1)) < 1:
        raise ConfigError("md.bad_tail_frames must be at least 1")
    try:
        TrajectoryHealthPolicy.from_mapping(md.get("health", {}))
    except TrajectoryHealthError as error:
        raise ConfigError(str(error)) from error

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
        if md.get("spin_temperature") is None:
            raise ConfigError("spin MD requires md.spin_temperature")
        if dft.get("backend", "vasp") not in {"abacus", "toy"}:
            raise ConfigError(
                "spin workflows require dft.backend=abacus or toy; "
                "VASP labeling is non-magnetic only"
            )

    if workflow:
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
    try:
        ScenarioLadder.from_workflow(
            {**dict(workflow), "initial_steps": int(md.get("initial_steps", 0))}
        )
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
