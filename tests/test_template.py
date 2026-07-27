from pathlib import Path

import pytest

from NepTrain.core.config import load_config
from NepTrain.core.template import init_project


def test_init_project_writes_a_valid_narrow_schema_v8(tmp_path):
    project = init_project("local", tmp_path)
    config, changes = load_config(project)

    assert changes == []
    assert config["schema_version"] == 8
    assert config["training"]["backend"] == "torchnep"
    assert config["md"]["backend"] == "lammps"
    route = config["sampling"]["routes"][0]
    assert route["id"] == "default"
    assert route["structures"] == ["./structures"]
    assert route["template_path"] == "./lammps.in"
    assert route["conditions"]["temperature_path"] == [300]
    assert "production_temperatures" not in route["conditions"]
    assert "pressure" not in route["conditions"]
    assert "progression" not in route
    assert "candidate_pool" not in config["sampling"]
    assert "selection" not in config["sampling"]
    assert "evaluation" not in config
    assert "structures" not in config["md"]
    assert "template_path" not in config["md"]
    assert "plugin_path" not in config["md"]
    assert config["labeling"]["kpoint_mode"] == "auto"
    assert config["labeling"]["resource_path"] == "./resources"
    assert config["labeling"]["potcar_manifest_path"] == "./vasp-resources.json"
    assert config["labeling"]["structures_per_job"] == 1
    assert config["labeling"]["max_concurrent"] == 20
    assert "kspacing" not in config["labeling"]
    assert "kpoints" not in config["labeling"]
    template = (tmp_path / "lammps.in").read_text(encoding="utf-8")
    assert "{{ temperature }}" in template
    assert "{{ steps }}" in template
    assert "{{ tdamp }}" not in template
    assert "{{ plugin_command }}" not in template
    assert "current_job" not in config
    assert "duration_ps_every_generation" not in config["md"]
    assert (tmp_path / "lammps.in").is_file()
    assert (tmp_path / "INCAR").is_file()
    assert (tmp_path / "vasp-resources.json").is_file()
    assert not (tmp_path / "INPUT").exists()
    assert "KSPACING = 0.2" in (tmp_path / "INCAR").read_text(encoding="utf-8")


def test_init_selects_only_the_requested_spin_abacus_templates(tmp_path):
    project = init_project(
        "local",
        tmp_path,
        ensemble="nvt",
        spin=True,
        dft_backend="abacus",
    )
    config, _ = load_config(project)

    assert config["md"]["spin"] is True
    assert config["labeling"]["backend"] == "abacus"
    assert config["labeling"]["input_path"] == "./INPUT"
    assert config["labeling"]["resource_manifest_path"] == "./abacus-resources.json"
    assert (tmp_path / "lammps.in").is_file()
    assert (tmp_path / "INPUT").is_file()
    assert (tmp_path / "abacus-resources.json").is_file()
    assert not (tmp_path / "INCAR").exists()
    assert "fix integrator all dynspin/glsd/nvt" in (
        tmp_path / "lammps.in"
    ).read_text(encoding="utf-8")


def test_init_rejects_spin_vasp_combination(tmp_path):
    with pytest.raises(ValueError, match="dft-backend abacus"):
        init_project("local", tmp_path, spin=True)


def test_force_switch_removes_obsolete_generated_dft_input(tmp_path):
    init_project("local", tmp_path)
    init_project(
        "local",
        tmp_path,
        spin=True,
        dft_backend="abacus",
        force=True,
    )

    assert (tmp_path / "INPUT").is_file()
    assert not (tmp_path / "INCAR").exists()


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
