from pathlib import Path
import subprocess
import sys

import pytest
from ase.io import read as ase_read

from NepTrain.core.config import load_config
from NepTrain.core.gpumd.io import RunInput


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("example", "runner"),
    [
        (
            "distillation-deepmd",
            "neptrain model-worker deepmd --head OMol25",
        ),
        ("distillation-mace", "neptrain model-worker mace"),
        (
            "distillation-tace",
            "neptrain model-worker tace --fidelity-index 0",
        ),
    ],
)
def test_distillation_workflow_examples_are_schema_valid(example, runner):
    root = ROOT / "examples" / example
    config, warnings = load_config(root / "project.yaml")

    assert warnings == []
    assert config["labeling"]["runner"] == runner
    assert config["md"]["backend"] == "gpumd"
    assert config["workflow"]["max_model_generations"] == 1


@pytest.mark.parametrize(
    "example",
    ["distillation-deepmd", "distillation-mace", "distillation-tace"],
)
def test_distillation_workflow_examples_use_adaptable_nve_template(
    example,
    tmp_path,
):
    root = ROOT / "examples" / example
    run_input = RunInput(tmp_path / "nep.txt")
    run_input.read_run(root / "gpumd-nve.in")
    run_input.configure(
        temperature=75.0,
        pressure=0.0,
        steps=4,
        timestep_fs=0.1,
        seed=42,
    )
    output = tmp_path / f"{example}.in"
    run_input.write_run(output)

    text = output.read_text(encoding="utf-8")
    assert "ensemble nve\n" in text
    assert "velocity 75.0 seed 42\n" in text
    assert "run 4\n" in text


@pytest.mark.parametrize(
    ("example", "backend", "target"),
    [
        ("workflow-vasp-slurm", "vasp", "vasp"),
        ("workflow-abacus-slurm", "abacus", "abacus"),
    ],
)
def test_dft_workflow_examples_are_schema_valid(example, backend, target):
    root = ROOT / "examples" / example
    config, warnings = load_config(root / "project.yaml")

    assert warnings == []
    assert config["labeling"]["backend"] == backend
    assert config["execution"]["stage_targets"]["labeling"] == target
    assert config["training"]["test_path"] == "./validation.xyz"
    assert config["workflow"]["max_model_generations"] == 1


def test_al_tutorial_seed_contains_complete_training_labels(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "prepare_al_seed.py"),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    train = ase_read(tmp_path / "train.xyz", index=":")
    validation = ase_read(tmp_path / "validation.xyz", index=":")
    start = ase_read(tmp_path / "structures" / "al.xyz")
    assert len(train) == 24
    assert len(validation) == 4
    assert len(start) == 4
    for frame in [*train, *validation]:
        assert frame.get_forces().shape == (4, 3)
        assert frame.info["virial"].shape == (3, 3)
