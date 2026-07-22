from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from NepTrain.core.dft import LabelRequest, label
from NepTrain.core.dft.vasp.native import NativeVaspError


class _FakeVaspInput:
    last_settings = {}
    converged = True
    result_payload = {
        "energy": -7.25,
        "forces": np.asarray([[0.1, 0.2, 0.3]]),
        "stress": np.asarray([1.0, 2.0, 3.0, 0.4, 0.5, 0.6]),
    }

    def __init__(self, *, pp_path):
        self.pp_path = Path(pp_path)
        self.int_params = {"ibrion": -1, "nsw": 0, "ispin": 1}
        self.bool_params = {"lnoncollinear": False, "lsorbit": False}
        self.float_params = {"kspacing": None}
        self.list_float_params = {"magmom": None}
        self.results = {}
        self.settings = {}

    def read_incar(self, path):
        text = Path(path).read_text(encoding="utf-8")
        if "IBRION = 2" in text:
            self.int_params["ibrion"] = 2
        if "NSW = 5" in text:
            self.int_params["nsw"] = 5
        if "ISPIN = 2" in text:
            self.int_params["ispin"] = 2
        if "KSPACING = 0.25" in text:
            self.float_params["kspacing"] = 0.25

    def set(self, **settings):
        self.settings.update(settings)
        type(self).last_settings = dict(self.settings)

    def calculate(self, atoms, properties):
        directory = Path(self.settings["directory"])
        directory.joinpath("INCAR").write_text("ISPIN = 1\n", encoding="utf-8")
        directory.joinpath("POSCAR").write_text("fake POSCAR\n", encoding="utf-8")
        directory.joinpath("KPOINTS").write_text("fake KPOINTS\n", encoding="utf-8")
        directory.joinpath("POTCAR").write_text("fake POTCAR\n", encoding="utf-8")
        self.results = {
            key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
            for key, value in self.result_payload.items()
        }


def _request(tmp_path: Path) -> LabelRequest:
    source = tmp_path / "selected.xyz"
    write(
        source,
        Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True),
        format="extxyz",
    )
    incar = tmp_path / "INCAR.template"
    incar.write_text("IBRION = -1\nNSW = 0\nISPIN = 1\n", encoding="utf-8")
    resources = tmp_path / "potpaw_PBE"
    resources.mkdir()
    return LabelRequest(
        source=source,
        output_file=tmp_path / "labeled.xyz",
        work_dir=tmp_path / "work",
        input_file=incar,
        resource_dir=resources,
        use_gamma=True,
        ka=(4, 4, 4),
    )


def test_vasp_nonmagnetic_single_point_is_normalized_and_provenanced(
    tmp_path: Path, monkeypatch
):
    native = __import__("NepTrain.core.dft.vasp.native", fromlist=["VaspInput"])
    monkeypatch.setattr(native, "VaspInput", _FakeVaspInput)

    result = label(_request(tmp_path), "vasp")
    restored = read(result.output_file)

    assert restored.get_potential_energy() == pytest.approx(-7.25)
    np.testing.assert_allclose(restored.get_forces(), [[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(
        restored.info["virial"],
        -64.0 * np.asarray([[1.0, 0.6, 0.5], [0.6, 2.0, 0.4], [0.5, 0.4, 3.0]]),
    )
    case = tmp_path / "work" / "000001-Al" / "attempt-0001"
    manifest = json.loads((case / "vasp-result.json").read_text())
    assert manifest["status"] == "completed"
    assert set(manifest["input_sha256"]) == {"INCAR", "POSCAR", "KPOINTS", "POTCAR"}


def test_vasp_retry_uses_a_fresh_attempt_directory(tmp_path: Path, monkeypatch):
    native = __import__("NepTrain.core.dft.vasp.native", fromlist=["VaspInput"])
    monkeypatch.setattr(native, "VaspInput", _FakeVaspInput)
    request = _request(tmp_path)

    label(request, "vasp")
    label(request, "vasp")

    case = tmp_path / "work" / "000001-Al"
    assert (case / "attempt-0001" / "vasp-result.json").is_file()
    assert (case / "attempt-0002" / "vasp-result.json").is_file()


def test_vasp_auto_mode_honors_template_kspacing(tmp_path: Path, monkeypatch):
    native = __import__("NepTrain.core.dft.vasp.native", fromlist=["VaspInput"])
    monkeypatch.setattr(native, "VaspInput", _FakeVaspInput)
    request = _request(tmp_path)
    request.input_file.write_text(
        "IBRION = -1\nNSW = 0\nISPIN = 1\nKSPACING = 0.25\n",
        encoding="utf-8",
    )

    label(request, "vasp")

    assert _FakeVaspInput.last_settings["kspacing"] == pytest.approx(0.25)
    assert "kpts" not in _FakeVaspInput.last_settings


def test_vasp_rejects_relaxation_template_before_launch(tmp_path: Path, monkeypatch):
    native = __import__("NepTrain.core.dft.vasp.native", fromlist=["VaspInput"])
    monkeypatch.setattr(native, "VaspInput", _FakeVaspInput)
    request = _request(tmp_path)
    request.input_file.write_text("IBRION = 2\nNSW = 5\nISPIN = 1\n", encoding="utf-8")

    with pytest.raises(NativeVaspError, match="fixed-geometry single point"):
        label(request, "vasp")

    manifest = json.loads(
        (
            tmp_path
            / "work"
            / "000001-Al"
            / "attempt-0001"
            / "vasp-result.json"
        ).read_text()
    )
    assert manifest["status"] == "failed"


def test_vasp_rejects_nonconverged_result(tmp_path: Path, monkeypatch):
    class NonConverged(_FakeVaspInput):
        converged = False

    native = __import__("NepTrain.core.dft.vasp.native", fromlist=["VaspInput"])
    monkeypatch.setattr(native, "VaspInput", NonConverged)

    with pytest.raises(NativeVaspError, match="did not converge"):
        label(_request(tmp_path), "vasp")


def test_vasp_rejects_magnetic_incar(tmp_path: Path, monkeypatch):
    native = __import__("NepTrain.core.dft.vasp.native", fromlist=["VaspInput"])
    monkeypatch.setattr(native, "VaspInput", _FakeVaspInput)
    request = _request(tmp_path)
    request.input_file.write_text("IBRION = -1\nNSW = 0\nISPIN = 2\n", encoding="utf-8")

    with pytest.raises(NativeVaspError, match="ISPIN=1"):
        label(request, "vasp")
