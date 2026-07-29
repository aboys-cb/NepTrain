from pathlib import Path
import subprocess

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read as ase_read
from ase.io import write as ase_write

from NepTrain.runners.tace import TaceRunnerError, label_frames


def _argument(command, name):
    return Path(command[command.index(name) + 1])


def test_tace_runner_normalizes_energy_forces_and_stress(tmp_path):
    source = tmp_path / "input.xyz"
    model = tmp_path / "teacher.pt"
    output = tmp_path / "labeled.xyz"
    frame = bulk("Al", "fcc", a=4.05, cubic=True)
    ase_write(source, frame, format="extxyz")
    model.write_bytes(b"tace fixture")
    stress = np.diag([0.1, 0.2, 0.3])
    calls = []

    def fake_tace_eval(command, **kwargs):
        calls.append((command, kwargs))
        inputs = ase_read(_argument(command, "--input"), index=":")
        assert inputs[0].info["fidelity_idx"] == 2
        prediction = inputs[0].copy()
        prediction.info["TACE_energy"] = -3.5
        prediction.info["TACE_stress"] = stress.reshape(-1)
        prediction.arrays["TACE_forces"] = np.full(
            (len(prediction), 3),
            0.25,
        )
        ase_write(
            _argument(command, "--output"),
            prediction,
            format="extxyz",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    labeled = label_frames(
        model,
        source,
        output,
        device="cpu",
        precision="float64",
        fidelity_index=2,
        command_runner=fake_tace_eval,
    )

    assert len(calls) == 1
    command = calls[0][0]
    assert command[0] == "tace-eval"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--dtype") + 1] == "float64"
    assert labeled[0].get_potential_energy() == pytest.approx(-3.5)
    np.testing.assert_allclose(labeled[0].get_forces(), 0.25)
    np.testing.assert_allclose(
        labeled[0].info["virial"],
        -stress * frame.get_volume(),
    )
    restored = ase_read(output)
    assert restored.get_potential_energy() == pytest.approx(-3.5)
    assert restored.info["virial"].shape == (3, 3)


def test_tace_runner_uses_direct_virial_and_maps_spin_mforce(tmp_path):
    source = tmp_path / "spin.xyz"
    model = tmp_path / "teacher.pt"
    output = tmp_path / "labeled.xyz"
    frame = bulk("Fe", "bcc", a=2.87, cubic=True)
    frame.arrays["spin"] = np.tile([0.0, 0.0, 2.0], (len(frame), 1))
    ase_write(source, frame, format="extxyz")
    model.write_bytes(b"spin tace fixture")
    virial = np.diag([1.0, 2.0, 3.0])
    mforce = np.tile([0.1, -0.2, 0.3], (len(frame), 1))

    def fake_tace_eval(command, **_kwargs):
        assert command[
            command.index("--initial_noncollinear_magmoms_key") + 1
        ] == "spin"
        prediction = ase_read(_argument(command, "--input"))
        prediction.info["TACE_energy"] = -8.0
        prediction.info["TACE_direct_virials"] = virial.reshape(-1)
        prediction.arrays["TACE_direct_forces"] = np.zeros(
            (len(prediction), 3)
        )
        prediction.arrays[
            "TACE_noncollinear_magnetic_forces"
        ] = mforce
        ase_write(
            _argument(command, "--output"),
            prediction,
            format="extxyz",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    labeled = label_frames(
        model,
        source,
        output,
        device="cuda",
        precision="float32",
        command_runner=fake_tace_eval,
    )

    np.testing.assert_allclose(labeled[0].info["virial"], virial)
    np.testing.assert_allclose(labeled[0].arrays["spin"], frame.arrays["spin"])
    np.testing.assert_allclose(labeled[0].arrays["mforce"], mforce)


def test_tace_runner_rejects_spin_model_without_magnetic_forces(tmp_path):
    source = tmp_path / "spin.xyz"
    model = tmp_path / "teacher.pt"
    output = tmp_path / "labeled.xyz"
    frame = bulk("Fe", "bcc", a=2.87, cubic=True)
    frame.arrays["spin"] = np.tile([0.0, 0.0, 2.0], (len(frame), 1))
    ase_write(source, frame, format="extxyz")
    model.write_bytes(b"ordinary tace fixture")

    def fake_tace_eval(command, **_kwargs):
        prediction = ase_read(_argument(command, "--input"))
        prediction.info["TACE_energy"] = -8.0
        prediction.info["TACE_virials"] = np.zeros(9)
        prediction.arrays["TACE_forces"] = np.zeros((len(prediction), 3))
        ase_write(
            _argument(command, "--output"),
            prediction,
            format="extxyz",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(
        TaceRunnerError,
        match="spin input requires TACE noncollinear magnetic forces",
    ):
        label_frames(
            model,
            source,
            output,
            device="cpu",
            precision="float32",
            command_runner=fake_tace_eval,
        )


def test_tace_runner_reports_missing_upstream_command(tmp_path):
    source = tmp_path / "input.xyz"
    model = tmp_path / "teacher.pt"
    frame = bulk("Al", "fcc", a=4.05, cubic=True)
    ase_write(source, frame, format="extxyz")
    model.write_bytes(b"tace fixture")

    def missing_command(*_args, **_kwargs):
        raise FileNotFoundError("tace-eval")

    with pytest.raises(TaceRunnerError, match="tace-eval is not installed"):
        label_frames(
            model,
            source,
            tmp_path / "output.xyz",
            device="cpu",
            precision="float32",
            command_runner=missing_command,
        )
