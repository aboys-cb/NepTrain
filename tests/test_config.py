from pathlib import Path

import pytest
from ruamel.yaml import YAML

from NepTrain.core.config import ConfigError, load_config, migrate_config


def test_migrate_legacy_backend_names_and_paths():
    legacy = {
        "current_job": "nep",
        "nep": {"test_xyz_path": "test.xyz", "nep_in_path": "nep.in"},
        "gpumd": {
            "step_times": [10, 20],
            "temperature_every_step": [300],
            "model_path": "structure",
            "run_in_path": "run.in",
        },
    }
    config, changes = migrate_config(legacy)
    assert config["schema_version"] == 2
    assert config["current_job"] == "training"
    assert config["training"]["backend"] == "gpumd"
    assert config["training"]["test_path"] == "test.xyz"
    assert config["training"]["config_path"] == "nep.in"
    assert config["training"]["finetune_lr_scale"] == 0.1
    assert config["md"]["duration_ps_every_generation"] == [10, 20]
    assert config["md"]["temperatures"] == [300]
    assert config["md"]["structures"] == "structure"
    assert "nep -> training" in changes


def test_torchnep_finetune_learning_rate_must_be_positive(tmp_path: Path):
    path = tmp_path / "job.yaml"
    path.write_text(
        """
schema_version: 2
current_job: training
training: {backend: torchnep, finetune_lr_scale: 0}
md: {backend: lammps}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="finetune_lr_scale"):
        load_config(path)


def test_spin_md_requires_spin_temperature(tmp_path: Path):
    path = tmp_path / "job.yaml"
    path.write_text(
        """
schema_version: 2
current_job: md
training: {backend: torchnep}
md: {backend: lammps, spin: true, spin_temperature: null}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="spin_temperature"):
        load_config(path)


def test_spin_campaign_rejects_unvalidated_production_dft(tmp_path: Path):
    path = tmp_path / "job.yaml"
    path.write_text(
        """
schema_version: 2
current_job: md
training: {backend: torchnep}
md: {backend: lammps, spin: true, spin_temperature: auto}
dft: {software: abacus}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="non-magnetic only"):
        load_config(path)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"dump_interval": 0}, "dump_interval"),
        ({"pre_failure_frames": -1}, "pre_failure_frames"),
        ({"bad_tail_frames": 0}, "bad_tail_frames"),
        ({"health": {"min_distnace": 0.5}}, "min_distnace"),
    ],
)
def test_failure_window_config_is_validated(
    tmp_path: Path, settings: dict, message: str
):
    path = tmp_path / "job.yaml"
    yaml = YAML()
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(
            {
                "schema_version": 2,
                "current_job": "md",
                "training": {"backend": "gpumd"},
                "md": {"backend": "lammps", **settings},
            },
            handle,
        )
    with pytest.raises(ConfigError, match=message):
        load_config(path)
