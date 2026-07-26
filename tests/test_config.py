from pathlib import Path

import pytest
from ruamel.yaml import YAML

from NepTrain.core.config import ConfigError, load_config


def _project(**overrides):
    value = {
        "schema_version": 7,
        "training": {
            "backend": "torchnep",
            "initial_path": "./train.xyz",
            "config_path": "./nep.in",
        },
        "md": {
            "backend": "lammps",
            "spin": False,
        },
        "sampling": {
            "routes": [
                {
                    "id": "default",
                    "structures": ["./structures"],
                    "template_path": "./lammps.in",
                    "conditions": {
                        "temperature_path": [300],
                        "production_temperatures": [300],
                        "pressure": 0.0,
                    },
                    "progression": {
                        "steps": {
                            "smoke_passed": 100,
                            "short_stable": 400,
                            "long_stable": 1600,
                            "production_ready": 6400,
                        },
                        "replicas": {
                            "smoke_passed": 1,
                            "short_stable": 1,
                            "long_stable": 2,
                            "production_ready": 3,
                        },
                    },
                }
            ],
            "candidate_pool": {
                "pre_failure_frames": 2,
                "bad_tail_frames": 1,
                "health": {},
            },
            "selection": {
                "max_selected": 100,
                "novelty": "auto",
            },
        },
        "dft": {"backend": "toy"},
        "evaluation": {
            "validation_path": "./validation.xyz",
            "max_rmse": {"energy_rmse": 1, "force_rmse": 1},
        },
        "workflow": {"max_model_generations": 1},
        "execution": {
            "stage_targets": {
                "training": "local",
                "sampling": "local",
                "labeling": "local",
                "analysis": "local",
            },
            "sampling_route_targets": {},
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


def test_schema_v7_loads_without_migration(tmp_path):
    config, changes = load_config(_write(tmp_path, _project()))
    assert config["schema_version"] == 7
    assert config["sampling"]["routes"][0]["conditions"]["temperature_path"] == [300]
    assert changes == []


def test_sampling_defaults_do_not_need_to_be_repeated_in_project_yaml(tmp_path):
    value = _project()
    route = value["sampling"]["routes"][0]
    route["conditions"] = {"temperature_path": [300]}
    route.pop("progression")
    value["sampling"].pop("candidate_pool")
    value["sampling"].pop("selection")

    config, changes = load_config(_write(tmp_path, value))

    assert changes == []
    assert "progression" not in config["sampling"]["routes"][0]
    assert "candidate_pool" not in config["sampling"]
    assert "selection" not in config["sampling"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("structures_per_job", 0),
        ("structures_per_job", True),
        ("max_concurrent", 0),
        ("max_concurrent", 1.5),
    ],
)
def test_dft_parallelism_requires_positive_integers(tmp_path, field, value):
    with pytest.raises(ConfigError, match=rf"dft\.{field}"):
        load_config(
            _write(
                tmp_path,
                _project(dft={field: value}),
            )
        )


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 6, 8])
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
    with pytest.raises(ConfigError, match="conditions.*temperatrues"):
        value = _project()
        value["sampling"]["routes"][0]["conditions"]["temperatrues"] = [500]
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
    route = value["sampling"]["routes"][0]
    route["conditions"].update(
        temperature_path=[400, 800],
        production_temperatures=[400, 800],
        pressure=5.0,
    )
    route["progression"]["steps"] = {
        "smoke_passed": 2000,
        "short_stable": 8000,
        "long_stable": 32000,
        "production_ready": 128000,
    }
    config, _ = load_config(
        _write(tmp_path, value)
    )
    loaded_route = config["sampling"]["routes"][0]
    assert loaded_route["conditions"]["temperature_path"] == [400, 800]
    assert loaded_route["conditions"]["pressure"] == 5.0
    assert loaded_route["progression"]["steps"]["smoke_passed"] == 2000
    assert not {
        "structures",
        "template_path",
        "temperatures",
        "initial_steps",
        "pressure",
    } & set(config["md"])


def test_temperature_path_and_production_targets_are_validated(tmp_path):
    value = _project()
    value["sampling"]["routes"][0]["conditions"].update(
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


def test_spin_md_uses_lattice_temperature_by_default(tmp_path):
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
    config, _ = load_config(_write(tmp_path, value))
    assert "spin_temperature" not in config["sampling"]["routes"][0]["conditions"]


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
    config, _ = load_config(
        _write(tmp_path, value)
    )
    assert config["dft"]["backend"] == "abacus"


def test_spin_workflow_rejects_vasp(tmp_path):
    value = _project(md={"spin": True}, dft={"backend": "vasp"})
    with pytest.raises(ConfigError, match="VASP labeling"):
        load_config(_write(tmp_path, value))


def test_dft_kpoints_default_to_input_authoritative_auto_mode(tmp_path):
    value = _project(dft={"backend": "vasp"})

    config, _ = load_config(_write(tmp_path, value))

    assert config["dft"].get("kpoint_mode", "auto") == "auto"
    assert "kspacing" not in config["dft"]
    assert "kpoints" not in config["dft"]


@pytest.mark.parametrize(
    "dft",
    [
        {"backend": "vasp", "kpoint_mode": "auto", "kspacing": 0.2},
        {"backend": "vasp", "kpoint_mode": "auto", "kpoints": [4, 4, 4]},
        {
            "backend": "vasp",
            "kpoint_mode": "kspacing",
            "kspacing": 0.2,
            "kpoints": [4, 4, 4],
        },
        {
            "backend": "vasp",
            "kpoint_mode": "kpoints",
            "kpoints": [4, 4, 4],
            "kspacing": 0.2,
        },
    ],
)
def test_dft_kpoint_modes_reject_competing_authorities(tmp_path, dft):
    with pytest.raises(ConfigError, match="kspacing|kpoints"):
        load_config(_write(tmp_path, _project(dft=dft)))


def test_explicit_dft_kpoint_modes_are_valid(tmp_path):
    for dft in (
        {"backend": "vasp", "kpoint_mode": "kspacing", "kspacing": 0.2},
        {"backend": "abacus", "kpoint_mode": "kpoints", "kpoints": [4, 4, 4]},
    ):
        config, _ = load_config(_write(tmp_path, _project(dft=dft)))
        assert config["dft"]["kpoint_mode"] == dft["kpoint_mode"]


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


def test_sampling_routes_are_the_only_automatic_sampling_authority(tmp_path):
    value = _project()
    value["md"]["structures"] = "./legacy.xyz"
    with pytest.raises(ConfigError, match="unknown md fields.*structures"):
        load_config(_write(tmp_path, value))

    value = _project()
    value["sampling"]["conditions"] = {"temperature_path": [300]}
    with pytest.raises(ConfigError, match="unknown sampling fields.*conditions"):
        load_config(_write(tmp_path, value))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("md", "mpi_ranks"), 4),
        (("sampling", "mode"), "auto"),
        (
            ("sampling", "routes", 0, "progression", "md_runs_per_iteration"),
            4,
        ),
        (("sampling", "selection", "min_novelty"), 0.0),
        (("dft", "n_cpu"), 4),
        (("dft", "use_k_stype"), "kspacing"),
        (("workflow", "max_iterations"), 4),
        (("execution", "routes"), {}),
        (("execution", "targets", "local", "overrides"), {}),
    ],
)
def test_schema_v7_rejects_removed_duplicate_controls(tmp_path, path, value):
    project = _project()
    target = project
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConfigError, match=str(path[-1])):
        load_config(_write(tmp_path, project))


def test_routes_require_unique_ids_and_explicit_bindings(tmp_path):
    value = _project()
    value["sampling"]["routes"].append(
        {
            **value["sampling"]["routes"][0],
            "structures": ["./other.xyz"],
        }
    )
    with pytest.raises(ConfigError, match="duplicate id"):
        load_config(_write(tmp_path, value))

    value = _project()
    del value["sampling"]["routes"][0]["template_path"]
    with pytest.raises(ConfigError, match="template_path is required"):
        load_config(_write(tmp_path, value))

    value = _project()
    value["sampling"]["routes"][0]["structures"] = []
    with pytest.raises(ConfigError, match="structures must be a non-empty"):
        load_config(_write(tmp_path, value))


def test_same_structure_may_be_listed_by_multiple_explicit_routes(tmp_path):
    value = _project()
    second = {
        **value["sampling"]["routes"][0],
        "id": "second",
        "conditions": {
            **value["sampling"]["routes"][0]["conditions"],
            "temperature_path": [500, 1000],
            "production_temperatures": [1000],
        },
    }
    value["sampling"]["routes"].append(second)
    config, _ = load_config(_write(tmp_path, value))
    assert [route["structures"] for route in config["sampling"]["routes"]] == [
        ["./structures"],
        ["./structures"],
    ]


def test_workflow_allows_evaluation_to_be_omitted(tmp_path):
    value = _project()
    del value["evaluation"]
    config, _ = load_config(_write(tmp_path, value))
    assert "evaluation" not in config


def test_partial_evaluation_is_rejected(tmp_path):
    value = _project(evaluation={"validation_path": None})
    with pytest.raises(ConfigError, match="when evaluation is configured"):
        load_config(_write(tmp_path, value))
