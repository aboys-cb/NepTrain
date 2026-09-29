from pathlib import Path
import hashlib
import json

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.io import read, write

from NepTrain.core.dft.vasp import native
from NepTrain.core.dft.vasp.deltaspin import (
    DeltaSpinError, numbers, prepare_input, read_result,
)
from NepTrain.core.dft.vasp.io import VaspInput, read_vasp_input
from NepTrain.core.labeling import LabelRequest, label


FIXTURE = Path(__file__).parent / "fixtures" / "vasp_deltaspin"
TARGETS = np.asarray([[0.35, 0, 2.677], [-0.35, 0, 2.677]])


def test_current_fe2_output_matches_printed_magnetic_labels():
    result = read_result(FIXTURE / "OUTCAR.final.txt", ["Fe", "Fe"], TARGETS)
    np.testing.assert_array_equal(result.spin, [
        [0.3500000809, -0.0000000002, 2.6769999639],
        [-0.3500000800, -0.0000000002, 2.6769999642],
    ])
    np.testing.assert_array_equal(result.mforce, [
        [0.0072313466, 0.0000003270, 0.0494295373],
        [-0.0072396120, 0.0000004429, 0.0494380481],
    ])
    assert result.max_error == 8.0856e-8
    assert result.tolerance == 1e-6
    assert result.moment_def == 0


@pytest.mark.parametrize("moment_def", [0, 1, 2])
def test_all_current_moment_definitions_use_the_same_table(tmp_path, moment_def):
    path = tmp_path / "OUTCAR"
    path.write_text((FIXTURE / "OUTCAR.final.txt").read_text().replace(
        "DELTASPIN_MOMENT_DEF = 0", f"DELTASPIN_MOMENT_DEF = {moment_def}"
    ))
    assert read_result(path, ["Fe", "Fe"], TARGETS).moment_def == moment_def


@pytest.mark.parametrize(
    ("old", "new", "error"),
    [
        ("SCF convergence        : reached", "SCF convergence        : not reached", "SCF convergence"),
        ("Moment constraint      : reached", "Moment constraint      : not reached", "Moment constraint"),
        ("8.0856E-08", "8.0856E-03", "exceeds"),
        ("0.0072313466", "NaN", "non-finite"),
        ("0.3500000809", "0.4500000809", "targets"),
        ("1   Fe  1  1  1", "1   Fe  1  0  1", "constraints"),
        ("2   Fe  1  1  1", "1   Fe  1  1  1", "atom order"),
        ("MFx", "Bx", "table"),
        ("spin_lambda", "minus_spin_lambda", "sign convention"),
    ],
)
def test_rejects_invalid_magnetic_labels(tmp_path, old, new, error):
    path = tmp_path / "OUTCAR"
    path.write_text((FIXTURE / "OUTCAR.final.txt").read_text().replace(old, new))
    with pytest.raises(DeltaSpinError, match=error):
        read_result(path, ["Fe", "Fe"], TARGETS)


def test_rejects_truncated_or_multiple_observables(tmp_path):
    text = (FIXTURE / "OUTCAR.final.txt").read_text()
    path = tmp_path / "OUTCAR"
    for content in [text.split("Magnetic force")[0], text + text]:
        path.write_text(content)
        with pytest.raises(DeltaSpinError):
            read_result(path, ["Fe", "Fe"], TARGETS)


def test_incar_keeps_custom_lists_comments_and_continuations(tmp_path):
    path = tmp_path / "INCAR"
    path.write_text(
        "LDELTASPIN = .TRUE.; LNONCOLLINEAR = .TRUE. ! comment\n"
        "M_DELTASPIN = 0.35 0 2.677 \\\n -0.35 0 2.677 # targets\n"
        "DELTASPIN_ATOMS = 2*1\nDELTASPIN_COMPONENTS = 3*1\n"
        "DELTASPIN_TOL = 1D-6\nIBRION=-1; NSW=0\n"
    )
    calc = Vasp()
    read_vasp_input(calc, path)
    assert native.validate_vasp_input_file(path) == "deltaspin"
    np.testing.assert_array_equal(numbers(calc.input_params["custom"]["m_deltaspin"]), TARGETS.ravel())
    assert calc.input_params["custom"]["deltaspin_atoms"] == "2*1"


@pytest.mark.parametrize("setting", [
    "LNONCOLLINEAR = .FALSE.", "DELTASPIN_CONSTRAINT_MODE = 1",
    "DELTASPIN_COMPONENTS = 1 0 1", "DELTASPIN_ATOMS = 1 0",
    "SAXIS = 1 0 0", "DELTASPIN_TOL = -1", "IBRION = 2",
])
def test_rejects_unsupported_spin_contract_before_launch(tmp_path, setting):
    path = tmp_path / "INCAR"
    key = setting.split("=")[0].strip()
    lines = [line for line in (FIXTURE / "INCAR").read_text().splitlines()
             if line.split("=")[0].strip() != key]
    path.write_text("\n".join(lines) + "\n" + setting + "\n")
    with pytest.raises(native.NativeVaspError):
        native.validate_vasp_input_file(path)


def _request(tmp_path, atoms, incar):
    source = tmp_path / "selected.xyz"
    write(source, atoms, format="extxyz")
    input_file = tmp_path / "INCAR.template"
    input_file.write_text(incar)
    resources = tmp_path / "resources"
    elements = {}
    for symbol in dict.fromkeys(atoms.get_chemical_symbols()):
        path = resources / symbol / "POTCAR"
        path.parent.mkdir(parents=True)
        titel = f"PAW_PBE {symbol} test"
        path.write_text(f"TITEL = {titel}\nVRHFIN ={symbol}: test\n")
        elements[symbol] = {"path": f"{symbol}/POTCAR", "titel": titel,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = tmp_path / "vasp-resources.json"
    manifest.write_text(json.dumps({
        "protocol": "neptrain.vasp-resources.v1", "family": "PAW_PBE",
        "release": "synthetic-test-resources", "elements": elements,
    }))
    return LabelRequest(source, tmp_path / "labeled.xyz", tmp_path / "work", settings={
        "input_file": input_file, "resource_dir": resources, "resource_manifest": manifest,
        "kpoint_mode": "auto", "ka": (1, 1, 1),
    })


def test_ordinary_vasp_still_rejects_spin_input(tmp_path):
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    atoms.set_array("spin", np.asarray([[0.0, 0.0, 2.0]]))
    request = _request(tmp_path, atoms, "IBRION=-1\nNSW=0\nISPIN=2\n")
    with pytest.raises(native.NativeVaspError, match="LDELTASPIN"):
        label(request, "vasp")
    assert not request.output_file.exists()


@pytest.mark.parametrize("missing_mforce", [False, True])
@pytest.mark.parametrize("template_magmom", [None, "0 0 4 0 0 0.5 0 0 5"])
def test_input_and_labels_restore_mixed_element_order(tmp_path, monkeypatch, missing_mforce, template_magmom):
    monkeypatch.delenv("NEPTRAIN_VASP_COMMAND", raising=False)
    atoms = Atoms("FeAlFe", positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]], cell=[6, 6, 6], pbc=True)
    spin = np.asarray([[0.1, 0, 2], [0.2, 0, 0.1], [-0.1, 0, 2]])
    atoms.set_array("spin", spin)
    atoms.set_array("mforce", np.full((3, 3), 999.0))
    incar = "IBRION=-1\nNSW=0\nLNONCOLLINEAR=.TRUE.\nLDELTASPIN=.TRUE.\n"
    if template_magmom is not None:
        incar += "MAGMOM=" + template_magmom + "\n"
    incar += "M_DELTASPIN = 9*99\n"  # Each sampled input owns its targets.
    request = _request(tmp_path, atoms, incar)

    class ReplayVasp(VaspInput):
        def _run(self, **kwargs):
            directory = Path(self.directory)
            assert kwargs["command"].endswith("vasp_ncl")
            parsed = Vasp()
            read_vasp_input(parsed, directory / "INCAR")
            np.testing.assert_array_equal(numbers(parsed.input_params["custom"]["m_deltaspin"]).reshape(-1, 3), spin[[0, 2, 1]])
            np.testing.assert_array_equal(np.asarray(parsed.list_float_params["magmom"]).reshape(-1, 3), spin[[0, 2, 1]])
            np.testing.assert_array_equal(self.atoms.get_initial_magnetic_moments(), spin[[0, 2, 1]])
            assert read(directory / "POSCAR").get_chemical_symbols() == ["Fe", "Fe", "Al"]
            assert "mforce" not in self.atoms.arrays
            header = (FIXTURE / "OUTCAR.final.txt").read_text().split("   ion elem")[0]
            header = header.replace("8.0856E-08", "0.0000E+00")
            moment_rows, force_rows = [], []
            for i, (symbol, target) in enumerate(zip(["Fe", "Fe", "Al"], spin[[0, 2, 1]]), 1):
                moment_rows.append(f"{i} {symbol} 1 1 1 " + " ".join(map(str, target)) + " 0")
                force_rows.append(f"{i} {symbol} 1 1 1 {i} {-i} {i * 2}")
            text = header + "ion elem cx cy cz Mx My Mz |M|\n---\n" + "\n".join(moment_rows)
            text += "\n\nMagnetic force (eV/uB)\nion elem cx cy cz MFx MFy MFz\n---\n" + "\n".join(force_rows)
            text += "\nmagnetic force = spin_lambda for L=E+lambda.(M-M_target)\n"
            if missing_mforce:
                text = text.split("Magnetic force")[0]
            (directory / "OUTCAR").write_text(text)
            return 0, ""

        def read_results(self):
            self.converged = True
            self.results = {"energy": -5., "forces": np.asarray([[1., 2, 3], [4, 5, 6], [7, 8, 9]]), "stress": np.arange(6.)}

    monkeypatch.setattr(native, "VaspInput", ReplayVasp)
    if missing_mforce:
        with pytest.raises(native.NativeVaspError, match="sign convention"):
            label(request, "vasp")
        assert not request.output_file.exists()
        manifest = json.loads(next(request.work_dir.glob("*/vasp-result.json")).read_text())
        assert manifest["status"] == "failed"
        return
    result = label(request, "vasp")
    frame = read(result.output_file)
    assert frame.get_chemical_symbols() == ["Fe", "Al", "Fe"]
    np.testing.assert_array_equal(frame.arrays["spin"], spin)
    np.testing.assert_array_equal(frame.arrays["mforce"], [[1, -1, 2], [3, -3, 6], [2, -2, 4]])
    np.testing.assert_array_equal(frame.get_forces(), [[1, 2, 3], [7, 8, 9], [4, 5, 6]])
    assert frame.info["dft_electronic_mode"] == "deltaspin"
    manifest = json.loads(next(request.work_dir.glob("*/vasp-result.json")).read_text())
    assert manifest["spin_force_labels"] is True
    assert manifest["status"] == "completed"
    assert manifest["deltaspin"]["mforce_unit"] == "eV/uB"
    assert "OUTCAR" in manifest["output_sha256"]
    assert "neptrain_input_structure_id" in frame.info
