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
    "DELTASPIN_COMPONENTS = 1 0 1", "DELTASPIN_ATOMS = 1 2", "DELTASPIN_ATOMS = 0 0",
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
@pytest.mark.parametrize("partial", [False, True, "auto"])
def test_input_and_labels_restore_mixed_element_order(tmp_path, monkeypatch, missing_mforce, template_magmom, partial):
    monkeypatch.delenv("NEPTRAIN_VASP_COMMAND", raising=False)
    atoms = Atoms("FeAlFe", positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]], cell=[6, 6, 6], pbc=True)
    spin = np.asarray([[0.1, 0, 2], [0.2, 0, 0.1], [-0.1, 0, 2]])
    if partial:
        spin[1] = [0.025, 0, 0]  # Unconstrained initial moments may be nonzero.
        if partial == "auto":
            spin[1] = 0
        else:
            atoms.set_array("spin_active", np.asarray([1, 0, 1]))
    atoms.set_array("spin", spin)
    atoms.set_initial_magnetic_moments(np.full((3, 3), 9.0))
    atoms.set_array("spin_projected", np.full((3, 3), 999.0))
    atoms.set_array("mforce", np.full((3, 3), 999.0))
    if partial:
        atoms.arrays["mforce"][1] = 0
    incar = "IBRION=-1\nNSW=0\nLNONCOLLINEAR=.TRUE.\nLDELTASPIN=.TRUE.\n"
    if template_magmom is not None:
        incar += "MAGMOM=" + template_magmom + "\n"
    incar += "M_DELTASPIN = 9*99\n"  # Each sampled input owns its targets.
    incar += "SCF_CUDA_NCL_GGA_MODE = 1\n"
    if partial is True:
        incar += "DELTASPIN_ATOMS = 2*0 2*1 6*0\n"  # Stale primitive mask must be replaced.
    request = _request(tmp_path, atoms, incar)

    class ReplayVasp(VaspInput):
        def _run(self, **kwargs):
            directory = Path(self.directory)
            assert kwargs["command"].endswith("vasp_ncl")
            parsed = Vasp()
            read_vasp_input(parsed, directory / "INCAR")
            assert parsed.input_params["custom"]["scf_cuda_ncl_gga_mode"] == "1"
            expected_mask = [1, 1, 0] if partial else [1, 1, 1]
            np.testing.assert_array_equal(numbers(parsed.input_params["custom"]["deltaspin_atoms"]), expected_mask)
            np.testing.assert_array_equal(numbers(parsed.input_params["custom"]["m_deltaspin"]).reshape(-1, 3), spin[[0, 2, 1]])
            np.testing.assert_array_equal(np.asarray(parsed.list_float_params["magmom"]).reshape(-1, 3), spin[[0, 2, 1]])
            np.testing.assert_array_equal(self.atoms.get_initial_magnetic_moments(), spin[[0, 2, 1]])
            assert read(directory / "POSCAR").get_chemical_symbols() == ["Fe", "Fe", "Al"]
            assert "mforce" not in self.atoms.arrays
            header = (FIXTURE / "OUTCAR.final.txt").read_text().split("   ion elem")[0]
            header = header.replace("8.0856E-08", "0.0000E+00")
            moment_rows, force_rows = [], []
            for i, (symbol, target) in enumerate(zip(["Fe", "Fe", "Al"], spin[[0, 2, 1]]), 1):
                if expected_mask[i - 1]:
                    moment_rows.append(f"{i} {symbol} 1 1 1 " + " ".join(map(str, target)) + " 0")
                    force_rows.append(f"{i} {symbol} 1 1 1 {i} {-i} {i * 2}")
                else:
                    force_rows.append(f"{i} {symbol} 0 0 0 0 0 0")
            text = header + "ion elem cx cy cz Mx My Mz |M|\n---\n" + "\n".join(moment_rows)
            text += "\n\nMagnetic force (eV/uB)\nion elem cx cy cz MFx MFy MFz\n---\n" + "\n".join(force_rows)
            text += "\nmagnetic force = spin_lambda for L=E+lambda.(M-M_target)\n"
            if partial:
                for axis in "xyz":
                    text += f"\nmagnetization ({axis})\n# of ion s p d tot\n---\n1 0 0 0 2.0\n2 0 0 0 2.0\n3 0 0 0 0.039\n---\n"
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
    expected_spin = spin.copy()
    if partial:
        expected_spin[1] = 0.039
    np.testing.assert_array_equal(frame.arrays["spin"], expected_spin)
    assert not {"spin_active", "spin_projected", "initial_magmoms"} & frame.arrays.keys()
    np.testing.assert_array_equal(frame.arrays["mforce"], [[1, -1, 2], [0, 0, 0] if partial else [3, -3, 6], [2, -2, 4]])
    np.testing.assert_array_equal(frame.get_forces(), [[1, 2, 3], [7, 8, 9], [4, 5, 6]])
    assert frame.info["dft_electronic_mode"] == "deltaspin"
    manifest = json.loads(next(request.work_dir.glob("*/vasp-result.json")).read_text())
    assert manifest["spin_force_labels"] is True
    assert manifest["status"] == "completed"
    assert manifest["deltaspin"]["mforce_unit"] == "eV/uB"
    assert "OUTCAR" in manifest["output_sha256"]
    assert "neptrain_input_structure_id" in frame.info


@pytest.mark.parametrize("mode", ["LNONCOLLINEAR=.TRUE.", "LSORBIT=.TRUE."])
@pytest.mark.parametrize("initial_source", ["incar", "structure", "default"])
def test_unconstrained_ncl_writes_vector_magmom_and_only_ordinary_labels(
    tmp_path, monkeypatch, mode, initial_source
):
    monkeypatch.delenv("NEPTRAIN_VASP_COMMAND", raising=False)
    atoms = Atoms("FeAlFe", positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]], cell=[6, 6, 6], pbc=True)
    moments = np.asarray([[0.1, 0, 2], [0.2, 0, 0.1], [-0.1, 0, 2]])
    incar = "IBRION=-1\nNSW=0\n" + mode + "\n"
    if initial_source == "incar":
        incar += "MAGMOM=" + " ".join(map(str, moments.ravel())) + "\n"
    elif initial_source == "structure":
        atoms.set_initial_magnetic_moments(moments)
    request = _request(tmp_path, atoms, incar)

    class ReplayVasp(VaspInput):
        def _run(self, **kwargs):
            assert kwargs["command"].endswith("vasp_ncl")
            parsed = Vasp()
            read_vasp_input(parsed, Path(self.directory) / "INCAR")
            assert parsed.int_params["ispin"] == 1
            if initial_source != "default":
                np.testing.assert_array_equal(
                    np.asarray(parsed.list_float_params["magmom"]).reshape(-1, 3),
                    moments[[0, 2, 1]],
                )
            assert read(Path(self.directory) / "POSCAR").get_chemical_symbols() == ["Fe", "Fe", "Al"]
            return 0, ""

        def read_results(self):
            self.converged = True
            # ASE normally restores its own atom ordering in read_results.
            forces = np.asarray([[1., 2, 3], [4, 5, 6], [7, 8, 9]])
            self.results = {"energy": -5., "forces": forces if initial_source != "default" else forces[[0, 2, 1]], "stress": np.arange(6.)}

    monkeypatch.setattr(native, "VaspInput", ReplayVasp)
    frame = read(label(request, "vasp").output_file)
    np.testing.assert_array_equal(frame.get_forces(), [[1, 2, 3], [7, 8, 9], [4, 5, 6]])
    assert frame.info["dft_electronic_mode"] == "noncollinear"
    assert "spin" not in frame.arrays and "mforce" not in frame.arrays
    manifest = json.loads(next(request.work_dir.glob("*/vasp-result.json")).read_text())
    assert manifest["spin_force_labels"] is False


def test_partial_bfo_preserves_induced_diagnostics_and_active_labels():
    mask = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    targets = np.zeros((10, 3)); targets[2, 0] = 4.154; targets[3, 0] = -4.154
    result = read_result(FIXTURE / 'OUTCAR.partial.txt', ['Bi']*2+['Fe']*2+['O']*6, targets, active=mask)
    np.testing.assert_array_equal(result.spin[2], [4.1540005830, -0.0000000336, 0.0000000663])
    assert np.all(result.spin[mask == 0] == 0)
    assert np.all(result.mforce[mask == 0] == 0)
    assert result.projected[4, 0] == -0.039
    assert result.projected[2, 0] == result.spin[2, 0]


@pytest.mark.parametrize('mutation', [
    'missing_moment', 'missing_force', 'wrong_mask', 'missing_induced',
    'unconverged', 'inactive_force', 'nonfinite_induced', 'unsupported_moment_def',
])
def test_partial_bfo_rejects_incomplete_or_inconsistent_output(tmp_path, mutation):
    mask = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    targets = np.zeros((10, 3)); targets[2, 0] = 4.154; targets[3, 0] = -4.154
    text = (FIXTURE / 'OUTCAR.partial.txt').read_text()
    if mutation == 'missing_moment':
        text = '\n'.join(line for line in text.splitlines() if '4.1540005830' not in line)
    elif mutation == 'missing_force':
        text = '\n'.join(line for line in text.splitlines() if 'Bi  0  0  0' not in line)
    elif mutation == 'wrong_mask':
        mask[4] = 1
    elif mutation == 'missing_induced':
        text = text.replace('magnetization (z)', 'missing (z)')
    elif mutation == 'inactive_force':
        text = text.replace(
            '1   Bi  0  0  0      0.0000000000',
            '1   Bi  0  0  0      0.1000000000',
        )
    elif mutation == 'nonfinite_induced':
        text = text.replace('-0.039', 'NaN')
    elif mutation == 'unsupported_moment_def':
        text = text.replace('DELTASPIN_MOMENT_DEF = 0', 'DELTASPIN_MOMENT_DEF = 1')
    else:
        text = text.replace('SCF convergence        : reached', 'SCF convergence        : failed')
    path = tmp_path/'OUTCAR'; path.write_text(text)
    with pytest.raises(DeltaSpinError):
        read_result(path, ['Bi']*2+['Fe']*2+['O']*6, targets, active=mask)


@pytest.mark.parametrize('mask', [[1, 2], [0, 0], [1], [1, np.nan]])
def test_bad_frame_masks_rejected_before_dft(mask):
    atoms = Atoms('FeO'); atoms.set_array('spin', np.zeros((2, 3)))
    # Wrong lengths cannot be attached through ASE's public setter.
    atoms.arrays['spin_active'] = np.asarray(mask)
    calc = Vasp(); calc.set(lnoncollinear=True, custom={'ldeltaspin': '.TRUE.'})
    with pytest.raises(DeltaSpinError, match='spin_active'):
        prepare_input(calc, atoms)


def test_partial_spin_mask_changes_input_identity_and_survives_extxyz(tmp_path):
    from NepTrain.core.scientific_data import structure_id
    from NepTrain.core.spin import validate_spin_structure, SpinDataError
    atoms = Atoms('FeO', cell=[5, 5, 5], pbc=True)
    atoms.set_array('spin', np.array([[0., 0., 4.], [0., 0., 0.]]))
    original = structure_id(atoms)
    atoms.set_array('spin_active', np.array([1, 1]))
    assert structure_id(atoms) != original
    atoms.set_array('spin_active', np.array([1, 0]))
    assert structure_id(atoms) == original
    path = tmp_path/'candidate.xyz'; write(path, atoms, format='extxyz')
    assert structure_id(read(path)) == structure_id(atoms)
    assert validate_spin_structure(read(path), require_mforce=False)
    atoms.arrays['spin'][1, 0] = 0.039
    assert validate_spin_structure(atoms, require_mforce=False)
    atoms.set_array('mforce', np.ones((2, 3)))
    with pytest.raises(SpinDataError, match='unconstrained'):
        validate_spin_structure(atoms, require_mforce=True)


def test_explicit_zero_constraint_identity_survives_xyz_rounding(tmp_path):
    from NepTrain.core.scientific_data import structure_id
    atoms = Atoms('FeO', cell=[5, 5, 5], pbc=True)
    atoms.set_array('spin', np.array([[0., 0., 4.], [1e-12, 0., 0.]]))
    atoms.set_array('spin_active', np.array([1, 1]))
    path = tmp_path / 'candidate.xyz'
    write(path, atoms, format='extxyz')
    assert structure_id(read(path)) == structure_id(atoms)
