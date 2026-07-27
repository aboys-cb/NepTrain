from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write

from NepTrain.runners.mace import MaceRunnerError, label_frames


class _FakeMaceCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": -3.25,
            "forces": np.full((len(atoms), 3), 0.125),
            "stress": np.asarray([1.0, 2.0, 3.0, 0.4, 0.5, 0.6]),
        }


def _input(path: Path, *, spin: bool = False) -> Path:
    frames = [
        Atoms(
            "Al2",
            positions=[[0.0, 0.0, 0.0], [2.0 + offset, 2.0, 2.0]],
            cell=[4.0, 4.0, 4.0],
            pbc=True,
        )
        for offset in (0.0, 0.1)
    ]
    if spin:
        frames[0].set_array("spin", np.ones((2, 3)))
    write(path, frames, format="extxyz")
    return path


def test_mace_runner_writes_energy_forces_and_virial(tmp_path):
    model = tmp_path / "teacher.model"
    model.write_bytes(b"model")
    output = tmp_path / "labeled.xyz"
    seen = []

    def factory(model_path, device, precision):
        seen.append((model_path, device, precision))
        return _FakeMaceCalculator()

    labeled = label_frames(
        model,
        _input(tmp_path / "input.xyz"),
        output,
        device="cpu",
        precision="float64",
        calculator_factory=factory,
    )

    restored = read(output, index=":")
    assert len(labeled) == len(restored) == 2
    assert seen == [(model.resolve(), "cpu", "float64")]
    assert restored[0].get_potential_energy() == pytest.approx(-3.25)
    np.testing.assert_allclose(restored[0].get_forces(), 0.125)
    np.testing.assert_allclose(
        restored[0].info["virial"],
        -64.0
        * np.asarray(
            [[1.0, 0.6, 0.5], [0.6, 2.0, 0.4], [0.5, 0.4, 3.0]]
        ),
    )


def test_mace_runner_rejects_spin_inputs_without_magnetic_forces(tmp_path):
    model = tmp_path / "teacher.model"
    model.write_bytes(b"model")

    with pytest.raises(MaceRunnerError, match="magnetic forces"):
        label_frames(
            model,
            _input(tmp_path / "spin.xyz", spin=True),
            tmp_path / "labeled.xyz",
            device="cpu",
            precision="float32",
            calculator_factory=lambda *_: _FakeMaceCalculator(),
        )
