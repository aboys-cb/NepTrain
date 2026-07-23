from pathlib import Path

import pytest
from ruamel.yaml import YAML

from NepTrain.core.config import ConfigError, load_config


def _project(**overrides):
    value = {
        "schema_version": 4,
        "training": {
            "backend": "torchnep",
            "initial_path": "./train.xyz",
            "config_path": "./nep.in",
        },
        "md": {
            "backend": "lammps",
            "structures": "./structures",
            "temperatures": [300],
            "initial_steps": 100,
            "spin": False,
        },
        "dft": {"backend": "toy"},
        "evaluation": {
            "validation_path": "./validation.xyz",
            "max_rmse": {"energy_rmse": 1, "force_rmse": 1},
        },
        "workflow": {"generations": 1},
        "execution": {
            "routes": {
                "training": "local",
                "sampling": "local",
                "labeling": "local",
                "analysis": "local",
            },
            "targets": {"local": {"executor": "process"}},
        },
    }
    for section, replacement in overrides.items():
        value[section].update(replacement)
    return value


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "project.yaml"
    with path.open("w", encoding="utf-8") as handle:
        YAML().dump(value, handle)
    return path


def test_schema_v4_loads_without_migration(tmp_path):
    config, changes = load_config(_write(tmp_path, _project()))
    assert config["schema_version"] == 4
    assert config["md"]["temperatures"] == [300]
    assert changes == []


@pytest.mark.parametrize("version", [1, 2, 3, 5])
def test_legacy_and_future_schemas_are_rejected(tmp_path, version):
    value = _project()
    value["schema_version"] = version
    with pytest.raises(ConfigError, match="does not run legacy|requires schema_version"):
        load_config(_write(tmp_path, value))


def test_unknown_legacy_fields_are_rejected(tmp_path):
    value = _project()
    value["current_job"] = "training"
    with pytest.raises(ConfigError, match="legacy fields"):
        load_config(_write(tmp_path, value))


def test_unknown_section_fields_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match="md.*temperatrues"):
        load_config(
            _write(tmp_path, _project(md={"temperatrues": [500]}))
        )


def test_temperature_steps_and_pressure_have_one_md_source(tmp_path):
    config, _ = load_config(
        _write(
            tmp_path,
            _project(
                md={
                    "temperatures": [400, 800],
                    "initial_steps": 2000,
                    "pressure": 5.0,
                }
            ),
        )
    )
    assert config["md"]["temperatures"] == [400, 800]
    assert config["md"]["initial_steps"] == 2000
    assert config["md"]["pressure"] == 5.0
    assert not {"temperatures", "initial_steps", "pressure"} & set(
        config["workflow"]
    )


def test_torchnep_finetune_learning_rate_must_be_positive(tmp_path):
    with pytest.raises(ConfigError, match="finetune_lr_scale"):
        load_config(
            _write(tmp_path, _project(training={"finetune_lr_scale": 0}))
        )


def test_spin_md_requires_spin_temperature(tmp_path):
    with pytest.raises(ConfigError, match="spin_temperature"):
        load_config(_write(tmp_path, _project(md={"spin": True})))


def test_spin_workflow_accepts_abacus_deltaspin(tmp_path):
    config, _ = load_config(
        _write(
            tmp_path,
            _project(
                md={"spin": True, "spin_temperature": 300},
                dft={"backend": "abacus"},
                evaluation={
                    "max_rmse": {
                        "energy_rmse": 1,
                        "force_rmse": 1,
                        "mforce_rmse": 1,
                    }
                },
            ),
        )
    )
    assert config["dft"]["backend"] == "abacus"


def test_spin_workflow_rejects_vasp(tmp_path):
    with pytest.raises(ConfigError, match="VASP labeling"):
        load_config(
            _write(
                tmp_path,
                _project(
                    md={"spin": True, "spin_temperature": 300},
                    dft={"backend": "vasp"},
                ),
            )
        )


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"dump_interval": 0}, "dump_interval"),
        ({"pre_failure_frames": -1}, "pre_failure_frames"),
        ({"bad_tail_frames": 0}, "bad_tail_frames"),
        ({"health": {"min_distnace": 0.5}}, "min_distnace"),
    ],
)
def test_failure_window_config_is_validated(tmp_path, settings, message):
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, _project(md=settings)))
