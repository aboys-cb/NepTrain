from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from NepTrain.core.labeling import LabelRequest, label
from NepTrain.core.dft.vasp.io import VaspInput
from NepTrain.core.dft.vasp.native import NativeVaspError
from NepTrain.core.dft.vasp.resources import (
    VaspResourceError,
    validate_vasp_resources,
)


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
    potcar = resources / "Al" / "POTCAR"
    potcar.parent.mkdir(parents=True)
    titel = "PAW_PBE Al 04Jan2001"
    potcar.write_text(
        f"TITEL = {titel}\nVRHFIN =Al: s2p1\n",
        encoding="utf-8",
    )
    resource_manifest = tmp_path / "vasp-resources.json"
    resource_manifest.write_text(
        json.dumps(
            {
                "protocol": "neptrain.vasp-resources.v1",
                "family": "PAW_PBE",
                "release": "test",
                "elements": {
                    "Al": {
                        "path": "Al/POTCAR",
                        "sha256": hashlib.sha256(potcar.read_bytes()).hexdigest(),
                        "titel": titel,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return LabelRequest(
        source=source,
        output_file=tmp_path / "labeled.xyz",
        work_dir=tmp_path / "work",
        settings={
            "input_file": incar,
            "resource_dir": resources,
            "resource_manifest": resource_manifest,
            "use_gamma": True,
            "ka": (4, 4, 4),
        },
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
    case = tmp_path / "work" / "000001-Al"
    manifest = json.loads((case / "vasp-result.json").read_text())
    assert manifest["status"] == "completed"
    assert set(manifest["input_sha256"]) == {"INCAR", "POSCAR", "KPOINTS", "POTCAR"}
    assert _FakeVaspInput.last_settings["setups"] == {
        "base": "minimal",
        "Al": "",
    }


def test_vasp_manifest_path_drives_the_exact_ase_potcar_selection(tmp_path):
    resources = tmp_path / "potpaw"
    pinned = resources / "Fe_pv" / "POTCAR"
    pinned.parent.mkdir(parents=True)
    pinned.write_text(
        "TITEL = PAW_PBE Fe_pv 02Aug2007\nVRHFIN =Fe: s2d6\n",
        encoding="utf-8",
    )
    decoy = resources / "Fe" / "POTCAR"
    decoy.parent.mkdir()
    decoy.write_text(
        "TITEL = PAW_PBE Fe 06Sep2000\nVRHFIN =Fe: s2d6\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "vasp-resources.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "neptrain.vasp-resources.v1",
                "family": "PAW_PBE",
                "release": "test",
                "elements": {
                    "Fe": {
                        "path": "Fe_pv/POTCAR",
                        "sha256": hashlib.sha256(
                            pinned.read_bytes()
                        ).hexdigest(),
                        "titel": "PAW_PBE Fe_pv 02Aug2007",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    atoms = Atoms(
        "Fe",
        positions=[[0.0, 0.0, 0.0]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )

    provenance = validate_vasp_resources(resources, manifest, atoms)
    calculator = VaspInput(pp_path=resources)
    calculator.set(
        pp="",
        setups={"base": "minimal", **provenance["ase_setups"]},
    )

    assert calculator._build_pp_list(atoms) == [str(pinned)]


def test_vasp_retry_uses_a_fresh_attempt_directory(tmp_path: Path, monkeypatch):
    native = __import__("NepTrain.core.dft.vasp.native", fromlist=["VaspInput"])
    monkeypatch.setattr(native, "VaspInput", _FakeVaspInput)
    request = _request(tmp_path)

    label(request, "vasp")
    label(request, "vasp")

    case = tmp_path / "work" / "000001-Al"
    assert (case / "vasp-result.json").is_file()
    assert (case / "retry-0002" / "vasp-result.json").is_file()


def test_vasp_resource_hash_drift_fails_before_calculator_launch(
    tmp_path: Path, monkeypatch
):
    request = _request(tmp_path)
    (request.settings["resource_dir"] / "Al" / "POTCAR").write_text(
        "TITEL = PAW_PBE Al changed\nVRHFIN =Al: s2p1\n",
        encoding="utf-8",
    )
    native = __import__("NepTrain.core.dft.vasp.native", fromlist=["VaspInput"])
    monkeypatch.setattr(
        native,
        "VaspInput",
        lambda **_kwargs: pytest.fail("calculator launched before POTCAR validation"),
    )

    with pytest.raises(VaspResourceError, match="hash mismatch for Al"):
        label(request, "vasp")


def test_vasp_resource_titel_pins_the_exact_variant_and_release(
    tmp_path: Path, monkeypatch
):
    request = _request(tmp_path)
    manifest = json.loads(request.settings["resource_manifest"].read_text())
    manifest["elements"]["Al"]["titel"] = "PAW_PBE Al_h 04Jan2001"
    request.settings["resource_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    native = __import__("NepTrain.core.dft.vasp.native", fromlist=["VaspInput"])
    monkeypatch.setattr(
        native,
        "VaspInput",
        lambda **_kwargs: pytest.fail("calculator launched before POTCAR validation"),
    )

    with pytest.raises(VaspResourceError, match="version mismatch for Al"):
        label(request, "vasp")


def test_vasp_single_structure_workflow_job_uses_flat_output(
    tmp_path: Path, monkeypatch
):
    native = __import__("NepTrain.core.dft.vasp.native", fromlist=["VaspInput"])
    monkeypatch.setattr(native, "VaspInput", _FakeVaspInput)
    base_request = _request(tmp_path)
    request = replace(
        base_request,
        output_file=tmp_path / "work" / "selected-labels.xyz",
        settings={**base_request.settings, "flat_single_case": True},
    )

    label(request, "vasp")
    label(request, "vasp")

    assert (request.work_dir / "INCAR").is_file()
    assert (request.work_dir / "vasp-result.json").is_file()
    assert (
        request.work_dir / "retry-0002" / "vasp-result.json"
    ).is_file()
    assert not (request.work_dir / "000001-Al").exists()


def test_vasp_auto_mode_honors_template_kspacing(tmp_path: Path, monkeypatch):
    native = __import__("NepTrain.core.dft.vasp.native", fromlist=["VaspInput"])
    monkeypatch.setattr(native, "VaspInput", _FakeVaspInput)
    request = _request(tmp_path)
    request.settings["input_file"].write_text(
        "IBRION = -1\nNSW = 0\nISPIN = 1\n"
        "KSPACING = 0.25\nKGAMMA = .TRUE.\n",
        encoding="utf-8",
    )

    label(request, "vasp")

    assert "kspacing" not in _FakeVaspInput.last_settings
    assert "kgamma" not in _FakeVaspInput.last_settings
    assert "kpts" not in _FakeVaspInput.last_settings


def test_vasp_rejects_relaxation_template_before_launch(tmp_path: Path, monkeypatch):
    native = __import__("NepTrain.core.dft.vasp.native", fromlist=["VaspInput"])
    monkeypatch.setattr(native, "VaspInput", _FakeVaspInput)
    request = _request(tmp_path)
    request.settings["input_file"].write_text("IBRION = 2\nNSW = 5\nISPIN = 1\n", encoding="utf-8")

    with pytest.raises(NativeVaspError, match="fixed-geometry single point"):
        label(request, "vasp")

    manifest = json.loads(
        (
            tmp_path
            / "work"
            / "000001-Al"
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


def test_vasp_accepts_collinear_spin_polarized_as_ordinary_labels(
    tmp_path: Path, monkeypatch
):
    native = __import__("NepTrain.core.dft.vasp.native", fromlist=["VaspInput"])
    monkeypatch.setattr(native, "VaspInput", _FakeVaspInput)
    request = _request(tmp_path)
    request.settings["input_file"].write_text("IBRION = -1\nNSW = 0\nISPIN = 2\n", encoding="utf-8")

    result = label(request, "vasp")

    frame = read(result.output_file)
    assert frame.info["dft_electronic_mode"] == "collinear_spin_polarized"
    assert "spin" not in frame.arrays
    assert "mforce" not in frame.arrays
    manifest = json.loads(
        next(request.work_dir.glob("*/vasp-result.json")).read_text(
            encoding="utf-8"
        )
    )
    assert manifest["electronic_mode"] == "collinear_spin_polarized"
    assert manifest["spin_force_labels"] is False


def test_vasp_requires_ispin2_for_nonzero_initial_magnetic_moments(
    tmp_path: Path, monkeypatch
):
    native = __import__("NepTrain.core.dft.vasp.native", fromlist=["VaspInput"])
    monkeypatch.setattr(native, "VaspInput", _FakeVaspInput)
    request = _request(tmp_path)
    frames = read(request.source, index=":")
    frames[0].set_initial_magnetic_moments([2.0])
    write(request.source, frames, format="extxyz")

    with pytest.raises(
        NativeVaspError,
        match="initial magnetic moments require collinear ISPIN=2",
    ):
        label(request, "vasp")

    request.settings["input_file"].write_text(
        "IBRION = -1\nNSW = 0\nISPIN = 2\n",
        encoding="utf-8",
    )
    result = label(request, "vasp")
    assert read(result.output_file).info["dft_electronic_mode"] == (
        "collinear_spin_polarized"
    )
