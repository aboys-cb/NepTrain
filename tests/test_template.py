from pathlib import Path

import pytest

from NepTrain.core.config import load_config
from NepTrain.core.template import init_project


def test_init_project_writes_a_valid_narrow_schema_v5(tmp_path):
    project = init_project("local", tmp_path)
    config, changes = load_config(project)

    assert changes == []
    assert config["schema_version"] == 5
    assert config["training"]["backend"] == "torchnep"
    assert config["md"]["backend"] == "lammps"
    assert config["sampling"]["conditions"]["temperature_path"] == [300]
    assert config["sampling"]["conditions"]["production_temperatures"] == [300]
    assert config["sampling"]["progression"]["replicas"]["production_ready"] == 3
    assert (
        config["sampling"]["progression"]["steps"]["smoke_passed"]
        == 10000
    )
    assert "plugin_path" not in config["md"]
    template = (tmp_path / "lammps-nvt.in").read_text(encoding="utf-8")
    assert "{{ temperature }}" in template
    assert "{{ steps }}" in template
    assert "{{ tdamp }}" not in template
    assert "{{ plugin_command }}" not in template
    assert "current_job" not in config
    assert "duration_ps_every_generation" not in config["md"]
    assert (tmp_path / "lammps-nvt.in").is_file()
    assert (tmp_path / "INCAR").is_file()
    assert (tmp_path / "INPUT").is_file()


def test_init_never_rewrites_an_existing_project_without_force(tmp_path):
    project = init_project("local", tmp_path)
    original = project.read_bytes()

    with pytest.raises(FileExistsError, match="--force"):
        init_project("slurm", tmp_path)

    assert project.read_bytes() == original


def test_slurm_profile_creates_editable_environment_scripts(tmp_path):
    project = init_project("slurm", tmp_path)
    config, _ = load_config(project)

    for target in config["execution"]["targets"].values():
        setup = tmp_path / target["setup_script"]
        assert setup.is_file()
        assert setup.stat().st_mode & 0o100
