"""Versioned user configuration for the active-learning workflow."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ruamel.yaml import YAML


CURRENT_SCHEMA_VERSION = 2


class ConfigError(ValueError):
    """Raised when a workflow configuration is incomplete or inconsistent."""


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(value))


def migrate_config(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return schema-v2 configuration and a list of performed migrations."""

    migrated = _copy_mapping(config)
    changes: list[str] = []
    schema_version = int(migrated.get("schema_version", 1))
    if schema_version > CURRENT_SCHEMA_VERSION:
        raise ConfigError(
            f"config schema {schema_version} is newer than supported schema "
            f"{CURRENT_SCHEMA_VERSION}"
        )

    if "training" not in migrated and "nep" in migrated:
        migrated["training"] = migrated.pop("nep")
        changes.append("nep -> training")
    if "md" not in migrated and "gpumd" in migrated:
        migrated["md"] = migrated.pop("gpumd")
        changes.append("gpumd -> md")
    if "md_split_job" not in migrated and "gpumd_split_job" in migrated:
        migrated["md_split_job"] = migrated.pop("gpumd_split_job")
        changes.append("gpumd_split_job -> md_split_job")

    training = migrated.setdefault("training", {})
    md = migrated.setdefault("md", {})
    training.setdefault("backend", "gpumd")
    md.setdefault("backend", "gpumd")
    training.setdefault("device", "cuda")
    training.setdefault("torch_backend", "auto")
    training.setdefault("finetune_lr_scale", 0.1)
    md.setdefault("inference_backend", "auto")

    if "test_xyz_path" in training and "test_path" not in training:
        training["test_path"] = training.pop("test_xyz_path")
        changes.append("training.test_xyz_path -> training.test_path")
    if "test" in training and "test_path" not in training:
        training["test_path"] = training.pop("test")
        changes.append("training.test -> training.test_path")
    if "config_path" not in training and "nep_in_path" in training:
        training["config_path"] = training.pop("nep_in_path")
        changes.append("training.nep_in_path -> training.config_path")
    if "restart" not in training and "nep_restart" in training:
        training["restart"] = training.pop("nep_restart")
        changes.append("training.nep_restart -> training.restart")
    if "restart_steps" not in training and "nep_restart_step" in training:
        training["restart_steps"] = training.pop("nep_restart_step")
        changes.append("training.nep_restart_step -> training.restart_steps")
    if "run_in_path" in md and "template_path" not in md:
        md["template_path"] = md.pop("run_in_path")
        changes.append("md.run_in_path -> md.template_path")
    if "duration_ps_every_generation" not in md and "step_times" in md:
        md["duration_ps_every_generation"] = md.pop("step_times")
        changes.append("md.step_times -> md.duration_ps_every_generation")
    if "temperatures" not in md and "temperature_every_step" in md:
        md["temperatures"] = md.pop("temperature_every_step")
        changes.append("md.temperature_every_step -> md.temperatures")
    if "structures" not in md and "model_path" in md:
        md["structures"] = md.pop("model_path")
        changes.append("md.model_path -> md.structures")

    current_job = migrated.get("current_job", "training")
    current_job = {"nep": "training", "gpumd": "md", "vasp": "dft"}.get(
        current_job, current_job
    )
    migrated["current_job"] = current_job
    migrated["schema_version"] = CURRENT_SCHEMA_VERSION
    return migrated, changes


def validate_config(config: Mapping[str, Any]) -> None:
    from .md.health import TrajectoryHealthError, TrajectoryHealthPolicy
    from .scenario import ScenarioLadder, ScenarioMaturityError

    training = config.get("training", {})
    md = config.get("md", {})
    dft = config.get("dft", {})
    if training.get("backend") not in {"gpumd", "torchnep"}:
        raise ConfigError("training.backend must be gpumd or torchnep")
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
    if dft.get("software", "vasp") not in {"vasp", "abacus", "toy"}:
        raise ConfigError("dft.software must be vasp, abacus, or toy")
    if dft.get("teacher_profile", "ordinary") not in {"ordinary", "spin"}:
        raise ConfigError("dft.teacher_profile must be ordinary or spin")
    if int(dft.get("n_cpu", dft.get("cpu_core", 1))) < 1:
        raise ConfigError("dft.n_cpu/dft.cpu_core must be positive")
    if dft.get("use_k_stype", "kspacing") not in {"kspacing", "kpoints"}:
        raise ConfigError("dft.use_k_stype must be kspacing or kpoints")
    if config.get("current_job") not in {"training", "md", "select", "dft", "pred"}:
        raise ConfigError(
            "current_job must be training, md, select, dft, or pred"
        )
    ranks = int(md.get("mpi_ranks", 1))
    if ranks < 1:
        raise ConfigError("md.mpi_ranks must be at least 1")
    if int(md.get("dump_interval", 100)) < 1:
        raise ConfigError("md.dump_interval must be positive")
    if int(md.get("pre_failure_frames", 2)) < 0:
        raise ConfigError("md.pre_failure_frames must be non-negative")
    if int(md.get("bad_tail_frames", 1)) < 1:
        raise ConfigError("md.bad_tail_frames must be at least 1")
    try:
        TrajectoryHealthPolicy.from_mapping(md.get("health", {}))
    except TrajectoryHealthError as error:
        raise ConfigError(str(error)) from error
    if md.get("spin", False):
        if md.get("backend") != "lammps":
            raise ConfigError("spin MD currently requires md.backend=lammps")
        if md.get("spin_temperature") is None:
            raise ConfigError("spin MD requires md.spin_temperature")
        if dft.get("software", "vasp") != "toy":
            raise ConfigError(
                "spin campaigns currently require dft.software=toy; "
                "VASP/ABACUS production labeling is non-magnetic only"
            )
    try:
        ScenarioLadder.from_campaign(config.get("campaign", {}))
    except ScenarioMaturityError as error:
        raise ConfigError(str(error)) from error


def load_config(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = YAML(typ="safe").load(handle) or {}
    migrated, changes = migrate_config(raw)
    validate_config(migrated)
    return migrated, changes


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.dump(dict(config), handle)
