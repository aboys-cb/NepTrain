from pathlib import Path
import sys
import types

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from NepTrain.core.dft import LabelRequest, label
from NepTrain.core.select import farthest_point_sampling
from NepTrain.core.smoke import SmokeError, run_backend_workflow_smoke, run_smoke
from NepTrain.core.spin import validate_spin_dataset


def test_toy_label_adapter_preserves_spin_and_produces_all_labels(tmp_path: Path):
    atoms = Atoms(
        "Fe2",
        positions=[[1.0, 1.0, 1.0], [3.45, 1.0, 1.0]],
        cell=[7.0, 7.0, 7.0],
        pbc=True,
    )
    atoms.set_array("spin", np.asarray([[1.7, 0.0, 0.0], [0.0, 1.6, 0.2]]))
    source = tmp_path / "input.xyz"
    output = tmp_path / "labeled.xyz"
    write(source, atoms, format="extxyz")

    result = label(
        LabelRequest(source, output, tmp_path / "work", options={"profile": "spin"}),
        "toy",
    )

    assert len(result.frames) == 1
    restored = read(output, format="extxyz")
    validate_spin_dataset([restored], require_mforce=True)
    assert restored.calc.results["forces"].shape == (2, 3)
    assert np.asarray(restored.info["virial"]).shape == (3, 3)
    assert np.isfinite(restored.get_potential_energy())


def test_fps_is_deterministic_without_a_reference_and_never_repeats():
    points = np.asarray([[0.0], [1.0], [1.0], [2.0]])
    first = farthest_point_sampling(points, 10, min_dist=0.0)
    second = farthest_point_sampling(points, 10, min_dist=0.0)
    assert first == second
    assert len(first) == len(set(first))
    assert len(first) == 3


def test_all_toy_smoke_profiles_pass(tmp_path: Path):
    ordinary = run_smoke(tmp_path / "ordinary", profile="ordinary")
    spin = run_smoke(tmp_path / "spin", profile="spin")
    recovery = run_smoke(tmp_path / "recovery", profile="recovery")

    assert ordinary.passed
    assert spin.passed
    assert recovery.passed
    assert spin.derivative_mforce_max_error is not None
    assert recovery.recovery_match is True


def test_smoke_refuses_to_delete_the_working_directory():
    with pytest.raises(SmokeError, match="unsafe smoke output"):
        run_smoke(Path.cwd(), force=True)


def test_backend_workflow_smoke_reuses_real_workflow_config(tmp_path: Path, monkeypatch):
    teacher_input = Atoms(
        "Fe2",
        positions=[[1.0, 1.0, 1.0], [3.45, 1.0, 1.0]],
        cell=[7.0, 7.0, 7.0],
        pbc=True,
    )
    raw = tmp_path / "raw.xyz"
    train = tmp_path / "train.xyz"
    structure = tmp_path / "structure.xyz"
    write(raw, teacher_input, format="extxyz")
    write(structure, teacher_input, format="extxyz")
    label(LabelRequest(raw, train, tmp_path / "label"), "toy")
    (tmp_path / "nep.in").write_text("type 1 Fe\nepoch 100\n", encoding="utf-8")
    config = tmp_path / "job.yaml"
    config.write_text(
        """
schema_version: 2
current_job: training
generation: 1
init_train_xyz: ./train.xyz
training:
  backend: torchnep
  config_path: ./nep.in
md:
  backend: lammps
  inference_backend: cpu
  duration_ps_every_generation: [10]
  temperatures: [300]
  structures: ./structure.xyz
  timestep: 0.001
  spin: false
dft:
  software: vasp
  incar_path: auto
  cpu_core: 8
select:
  max_selected: 50
""",
        encoding="utf-8",
    )
    observed = {}

    class FakeWorker:
        def start(self, path):
            from NepTrain.core.config import load_config

            observed["config"], _ = load_config(path)

    monkeypatch.setitem(
        sys.modules,
        "NepTrain.core.train.run",
        types.SimpleNamespace(NepTrainWorker=FakeWorker),
    )
    report = run_backend_workflow_smoke(
        config,
        tmp_path / "smoke",
        profile="ordinary",
        training_steps=3,
        md_steps=2,
        dft_budget=7,
    )

    generated = observed["config"]
    assert report.completed
    assert generated["dft"]["software"] == "toy"
    assert generated["dft"]["incar_path"] is None
    assert generated["select"]["max_selected"] == 7
    assert generated["md"]["duration_ps_every_generation"] == [0.002]
    assert Path(generated["training"]["config_path"]).read_text().endswith("epoch 3\n")
