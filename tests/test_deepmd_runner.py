from __future__ import annotations

import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write

from NepTrain.runners.deepmd import (
    DeepmdRunnerError,
    _deepmd_calculator,
    label_frames,
)


class _FakeDeepmdCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": -5.5,
            "forces": np.full((len(atoms), 3), -0.25),
            "stress": np.asarray(
                [[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]]
            ),
        }


def _input(path: Path, *, spin: bool = False) -> Path:
    frames = [
        Atoms(
            "H2O",
            positions=[
                [5.0, 5.0, 5.0],
                [5.0 + scale * 0.96, 5.0, 5.0],
                [4.76, 5.93, 5.0],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        for scale in (1.0, 1.02)
    ]
    if spin:
        frames[0].set_array("spin", np.ones((3, 3)))
    write(path, frames, format="extxyz")
    return path


def test_deepmd_runner_writes_energy_forces_and_virial(tmp_path):
    model = tmp_path / "DPA-3.2-5M.pt"
    model.write_bytes(b"model")
    output = tmp_path / "labeled.xyz"
    seen = []

    def factory(model_path, device, precision, head):
        seen.append((model_path, device, precision, head))
        return _FakeDeepmdCalculator()

    labeled = label_frames(
        model,
        _input(tmp_path / "input.xyz"),
        output,
        device="cuda",
        precision="float32",
        head="OMol25",
        calculator_factory=factory,
    )

    restored = read(output, index=":")
    assert len(labeled) == len(restored) == 2
    assert seen == [(model.resolve(), "cuda", "float32", "OMol25")]
    assert restored[0].get_potential_energy() == pytest.approx(-5.5)
    np.testing.assert_allclose(restored[0].get_forces(), -0.25)
    np.testing.assert_allclose(
        restored[0].info["virial"],
        -1000.0
        * np.asarray(
            [[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]]
        ),
    )


def test_deepmd_runner_rejects_spin_inputs_without_magnetic_forces(tmp_path):
    model = tmp_path / "teacher.pt2"
    model.write_bytes(b"model")

    with pytest.raises(DeepmdRunnerError, match="magnetic forces"):
        label_frames(
            model,
            _input(tmp_path / "spin.xyz", spin=True),
            tmp_path / "labeled.xyz",
            device="cpu",
            precision="float64",
            calculator_factory=lambda *_: _FakeDeepmdCalculator(),
        )


def test_deepmd_runner_rejects_empty_head(tmp_path):
    model = tmp_path / "teacher.pt"
    model.write_bytes(b"model")

    with pytest.raises(DeepmdRunnerError, match="head must not be empty"):
        label_frames(
            model,
            _input(tmp_path / "input.xyz"),
            tmp_path / "labeled.xyz",
            device="cpu",
            precision="float64",
            head=" ",
            calculator_factory=lambda *_: _FakeDeepmdCalculator(),
        )


def test_deepmd_calculator_maps_cpu_precision_and_head(monkeypatch, tmp_path):
    seen = []
    deepmd_module = ModuleType("deepmd")
    calculator_module = ModuleType("deepmd.calculator")

    def fake_dp(*, model, head):
        seen.append((model, head))
        return _FakeDeepmdCalculator()

    calculator_module.DP = fake_dp
    deepmd_module.calculator = calculator_module
    monkeypatch.setitem(sys.modules, "deepmd", deepmd_module)
    monkeypatch.setitem(sys.modules, "deepmd.calculator", calculator_module)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("DP_INTERFACE_PREC", raising=False)
    model = tmp_path / "teacher.pt"

    _deepmd_calculator(model, "cpu", "float32", "OMol25")

    assert seen == [(str(model), "OMol25")]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["DP_INTERFACE_PREC"] == "low"


def test_deepmd_calculator_rejects_unavailable_cuda(monkeypatch, tmp_path):
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    with pytest.raises(DeepmdRunnerError, match="cannot access"):
        _deepmd_calculator(
            tmp_path / "teacher.pt",
            "cuda",
            "float32",
            None,
        )
