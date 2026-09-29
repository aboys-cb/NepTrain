from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from ase.io import read as ase_read
from ase.io import write as ase_write

from NepTrain.core.config import load_config
from NepTrain.core.gpumd.io import RunInput
from NepTrain.core.md import MdRequest, run_md
from NepTrain.core.sampling_route import SamplingRoute
from NepTrain.core.workflow import _md_timestep_ps


ROOT = Path(__file__).parents[1]
GPUMD_EXAMPLES = sorted(
    path.parent.name for path in (ROOT / "examples").glob("*/gpumd-nve.in")
)


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
    GPUMD_EXAMPLES,
)
def test_workflow_examples_use_placeholder_nve_template(
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
    assert "{{" not in text
    assert "potential nep.txt\n" in text
    assert "ensemble nve\n" in text
    assert "velocity 75.0 seed 42\n" in text
    assert "run 4\n" in text


@pytest.mark.parametrize("example", GPUMD_EXAMPLES)
def test_gpumd_examples_prepare_and_recover_every_sampling_stage(
    example, tmp_path, monkeypatch
):
    """Exercise shipped configs and file staging; GPUMD dynamics are mocked."""
    root = ROOT / "examples" / example
    config, warnings = load_config(root / "project.yaml")
    assert warnings == []
    route_config = config["sampling"]["routes"][0]
    template = root / route_config["template_path"]
    shutil.copy2(template, tmp_path / template.name)
    # Run the actual tutorial input generator outside the checkout.
    generator = root / "make_candidates.py"
    if generator.is_file():
        shutil.copy2(generator, tmp_path / generator.name)
        command = [sys.executable, str(tmp_path / generator.name)]
    else:
        command = [
            sys.executable, str(ROOT / "examples" / "prepare_al_seed.py"),
            "--output-dir", str(tmp_path),
        ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    route = SamplingRoute.from_config(route_config, base_dir=tmp_path)
    source = route.structure_paths[0]
    atoms = ase_read(source / "al.xyz" if source.is_dir() else source)
    timestep_fs = 0.1 if example.startswith("distillation-") else 0.5
    assert _md_timestep_ps(
        route_config, backend="gpumd", base_dir=tmp_path
    ) == pytest.approx(timestep_fs / 1000)
    model = tmp_path / "student-nep.txt"
    model.write_text("mock trained model\n")
    calls = []
    expected = {}

    def fake_gpumd(command, *, stdout, stderr, cwd, check):
        run_dir = Path(cwd)
        text = (run_dir / "run.in").read_text()
        assert "{{" not in text and "}}" not in text
        commands = [
            line.split("#", 1)[0].split()
            for line in text.splitlines()
            if line.split("#", 1)[0].strip()
        ]
        assert commands == [
            ["potential", "nep.txt"],
            ["velocity", str(expected["temperature"]), "seed", str(expected["seed"])],
            ["ensemble", "nve"],
            ["time_step", str(timestep_fs)],
            ["dump_exyz", "1", "0", "1"],
            ["run", str(expected["steps"])],
        ]
        assert (run_dir / "nep.txt").read_bytes() == model.read_bytes()
        prepared = ase_read(run_dir / "model.xyz")
        assert prepared.get_chemical_symbols() == atoms.get_chemical_symbols()
        np.testing.assert_allclose(prepared.positions, atoms.positions)
        np.testing.assert_allclose(prepared.cell, atoms.cell)
        assert prepared.pbc.all()
        # Simulate the dump contract to test recovery, not numerical acceptance.
        prepared.info["Time"] = expected["steps"] * timestep_fs
        prepared.set_array("forces", np.zeros((len(prepared), 3)))
        ase_write(run_dir / "dump.xyz", prepared, format="extxyz")
        calls.append(run_dir)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("NepTrain.core.gpumd.io.subprocess.run", fake_gpumd)
    for index, (stage, steps) in enumerate(route.progression["steps"].items()):
        expected.update(
            temperature=float(route.conditions["temperature_path"][0]),
            seed=int(config["workflow"]["seed"]) + index,
            steps=steps,
        )
        run_dir = tmp_path / stage
        result = run_md(
            MdRequest(
                atoms=atoms, model_file=model, output_dir=run_dir,
                output_file=run_dir / "trajectory.xyz",
                temperature=expected["temperature"], steps=steps,
                seed=expected["seed"], template_path=route.template_path,
                route_id=route.route_id, route_fingerprint=route.fingerprint,
            ),
            "gpumd",
        )
        assert result.completed is True
        assert result.last_step == steps
        recovered = ase_read(result.trajectory)
        assert recovered.info["md_step"] == steps
        assert recovered.arrays["nep_force"].shape == (len(atoms), 3)
        assert result.health_report.is_file()
    assert len(calls) == 4


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


@pytest.mark.parametrize(("backend", "text", "expected"), [
    ("lammps", "units metal\ntimestep {{ timestep_ps }}\n", 0.001),
    ("gpumd", "time_step {{ timestep_fs }}\n", 0.001),
    ("lammps", "units metal\ntimestep 0.005\n", 0.005),
    ("gpumd", "time_step 5\n", 0.005),
])
def test_workflow_progress_understands_explicit_timing_placeholders(tmp_path, backend, text, expected):
    path = tmp_path / "md.in"
    path.write_text(text)
    assert _md_timestep_ps({"template_path": path.name}, backend=backend, base_dir=tmp_path) == pytest.approx(expected)
