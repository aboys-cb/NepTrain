from pathlib import Path

import pytest
from ruamel.yaml import YAML

from NepTrain.core.config import ConfigError, load_config


def _project(**overrides):
    value = {
        "schema_version": 5,
        "training": {
            "backend": "torchnep",
            "initial_path": "./train.xyz",
            "config_path": "./nep.in",
        },
        "md": {
            "backend": "lammps",
            "structures": "./structures",
            "spin": False,
        },
        "sampling": {
            "mode": "auto",
            "conditions": {
                "temperature_path": [300],
                "production_temperatures": [300],
                "pressure": 0.0,
                "spin_temperature": None,
            },
            "progression": {
                "md_runs_per_iteration": 1,
                "steps": {
                    "smoke_passed": 100,
                    "short_stable": 400,
                    "long_stable": 1600,
                    "production_ready": 6400,
                },
            },
            "candidate_pool": {
                "pre_failure_frames": 2,
                "bad_tail_frames": 1,
                "health": {},
            },
            "selection": {
                "method": "fps",
                "max_selected": 100,
                "min_novelty": 0.0,
            },
        },
        "dft": {"backend": "toy"},
        "evaluation": {
            "validation_path": "./validation.xyz",
            "max_rmse": {"energy_rmse": 1, "force_rmse": 1},
        },
        "workflow": {"max_iterations": 1},
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


def test_schema_v5_loads_without_migration(tmp_path):
    config, changes = load_config(_write(tmp_path, _project()))
    assert config["schema_version"] == 5
    assert config["sampling"]["conditions"]["temperature_path"] == [300]
    assert changes == []


@pytest.mark.parametrize("version", [1, 2, 3, 4, 6])
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
    with pytest.raises(ConfigError, match="sampling.conditions.*temperatrues"):
        value = _project()
        value["sampling"]["conditions"]["temperatrues"] = [500]
        load_config(
            _write(tmp_path, value)
        )


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("candidate_pool", "target"),
        ("candidate_pool", "frame_stride"),
        ("selection", "dft_budget"),
        ("selection", "dft_batch_size"),
    ],
)
def test_pre_fps_candidate_caps_are_rejected(tmp_path, section, field):
    value = _project()
    value["sampling"][section][field] = 20
    with pytest.raises(ConfigError, match=field):
        load_config(_write(tmp_path, value))


def test_temperature_steps_and_pressure_have_one_md_source(tmp_path):
    value = _project()
    value["sampling"]["conditions"].update(
        temperature_path=[400, 800],
        production_temperatures=[400, 800],
        pressure=5.0,
    )
    value["sampling"]["progression"]["steps"] = {
        "smoke_passed": 2000,
        "short_stable": 8000,
        "long_stable": 32000,
        "production_ready": 128000,
    }
    config, _ = load_config(
        _write(tmp_path, value)
    )
    assert config["sampling"]["conditions"]["temperature_path"] == [400, 800]
    assert config["sampling"]["conditions"]["pressure"] == 5.0
    assert config["sampling"]["progression"]["steps"]["smoke_passed"] == 2000
    assert not {"temperatures", "initial_steps", "pressure"} & set(config["md"])


def test_temperature_path_and_production_targets_are_validated(tmp_path):
    value = _project()
    value["sampling"]["conditions"].update(
        temperature_path=[300, 700, 500],
        production_temperatures=[300, 900],
    )
    with pytest.raises(ConfigError, match="strictly|subset"):
        load_config(_write(tmp_path, value))


def test_torchnep_finetune_learning_rate_must_be_positive(tmp_path):
    with pytest.raises(ConfigError, match="finetune_lr_scale"):
        load_config(
            _write(tmp_path, _project(training={"finetune_lr_scale": 0}))
        )


def test_spin_md_requires_spin_temperature(tmp_path):
    with pytest.raises(ConfigError, match="spin_temperature"):
        load_config(_write(tmp_path, _project(md={"spin": True})))


def test_spin_workflow_accepts_abacus_deltaspin(tmp_path):
    value = _project(
        md={"spin": True},
        dft={"backend": "abacus"},
        evaluation={
            "max_rmse": {
                "energy_rmse": 1,
                "force_rmse": 1,
                "mforce_rmse": 1,
            }
        },
    )
    value["sampling"]["conditions"]["spin_temperature"] = 300
    config, _ = load_config(
        _write(tmp_path, value)
    )
    assert config["dft"]["backend"] == "abacus"


def test_spin_workflow_rejects_vasp(tmp_path):
    value = _project(md={"spin": True}, dft={"backend": "vasp"})
    value["sampling"]["conditions"]["spin_temperature"] = 300
    with pytest.raises(ConfigError, match="VASP labeling"):
        load_config(_write(tmp_path, value))


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
    value = _project()
    if "dump_interval" in settings:
        value["md"].update(settings)
    else:
        value["sampling"]["candidate_pool"].update(settings)
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, value))
