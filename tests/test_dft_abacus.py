from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from NepTrain.core.labeling import LabelRequest, label
from NepTrain.core.dft.abacus.io import StructureVar, read_input_file
from NepTrain.core.dft.abacus.native import (
    NativeAbacusError,
    parse_running_scf,
    validate_abacus_spin_contract,
)
from NepTrain.core.dft.abacus.resources import AbacusResourceError
from NepTrain.core.labeling import LabelingError
from NepTrain.core.scientific_data import (
    INPUT_STRUCTURE_ID_KEY,
    structure_id,
)


def _arguments(tmp_path: Path, source: Path, resources: Path) -> SimpleNamespace:
    input_file = tmp_path / "INPUT"
    input_file.write_text(
        """INPUT_PARAMETERS
suffix ABACUS
calculation scf
basis_type lcao
ecutwfc 20
smearing_method gau
""",
        encoding="utf-8",
    )
    return SimpleNamespace(
        model_path=str(source),
        out_file_path=str(tmp_path / "labeled.xyz"),
        directory=str(tmp_path / "work"),
        append=False,
        incar=str(input_file),
        resource_dir=str(resources),
        n_cpu=1,
        use_gamma=True,
        kspacing=None,
        ka=[1, 1, 1],
    )


def _resource_files(path: Path, elements=("Al",)) -> None:
    path.mkdir()
    records = {}
    for element in elements:
        pseudo = path / f"{element}.UPF"
        orbital = path / f"{element}.ORB"
        pseudo.write_text(
            f'<PP_HEADER element="{element}"/>\n', encoding="utf-8"
        )
        orbital.write_text(
            f"Element {element}\n", encoding="utf-8"
        )
        records[element] = {
            "pseudopotential": {
                "path": pseudo.name,
                "sha256": hashlib.sha256(pseudo.read_bytes()).hexdigest(),
            },
            "orbital": {
                "path": orbital.name,
                "sha256": hashlib.sha256(orbital.read_bytes()).hexdigest(),
            },
        }
    (path / "abacus-resources.json").write_text(
        json.dumps(
            {
                "protocol": "neptrain.abacus-resources.v1",
                "release": "test",
                "elements": records,
            }
        ),
        encoding="utf-8",
    )


def _fake_command(tmp_path: Path, log_text: str) -> str:
    fixture = tmp_path / "running_scf.fixture"
    fixture.write_text(log_text, encoding="utf-8")
    script = tmp_path / "fake-abacus.sh"
    script.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "mkdir -p OUT.ABACUS\n"
        f"cp {shlex.quote(str(fixture))} OUT.ABACUS/running_scf.log\n",
        encoding="utf-8",
    )
    return shlex.join(["sh", str(script)])


def _ordinary_log() -> str:
    return """
 LCAO ALGORITHM --------------- ION=   1  ELEC=   8--------------------------------
 charge density convergence is achieved
------------------------------------------------------------------------------------------
 TOTAL-FORCE (eV/Angstrom)
------------------------------------------------------------------------------------------
                       Al1         0.1000000000         0.2000000000         0.3000000000
------------------------------------------------------------------------------------------
 TOTAL-STRESS (KBAR)
       -10.0000000000         0.0000000000         0.0000000000
         0.0000000000       -20.0000000000         0.0000000000
         0.0000000000         0.0000000000       -30.0000000000
 !FINAL_ETOT_IS -60.3206240097501123 eV
"""


def _spin_log() -> str:
    return """
 #ELEC ITER# 12
 SCF IS CONVERGED
 TOTAL-FORCE (eV/Angstrom)
 Fe1 1.0 1.1 1.2
 Fe2 2.0 2.1 2.2
 Al1 3.0 3.1 3.2
 TOTAL-STRESS (KBAR)
 1.0 0.0 0.0
 0.0 2.0 0.0
 0.0 0.0 3.0
 Total Magnetism (uB)
 Fe 1.0 0.0 0.0
    0.0 0.0 3.0
 Al 0.0 2.0 0.0
 Magnetic force (eV/uB)
 Fe 0.1 0.2 0.3
    0.4 0.5 0.6
 Al 0.7 0.8 0.9
 !FINAL_ETOT_IS -12.5 eV
"""


def test_abacus_input_header_can_include_a_description(tmp_path: Path):
    path = tmp_path / "INPUT"
    path.write_text(
        "INPUT_PARAMETERS RUNNING ABACUS-DFT\n"
        "suffix FeNi\n"
        "nspin 4 # non-collinear\n",
        encoding="utf-8",
    )

    assert read_input_file(path) == {"suffix": "FeNi", "nspin": "4"}


def test_abacus_deltaspin_switch_requires_canonical_spin_input():
    with pytest.raises(NativeAbacusError, match="magnetic forces would be discarded"):
        validate_abacus_spin_contract(
            {"nspin": "4", "sc_mag_switch": "1"},
            spin_frame=False,
        )

    parameters = {"nspin": "2"}
    assert (
        validate_abacus_spin_contract(parameters, spin_frame=False)
        == "collinear_spin_polarized"
    )


def test_native_abacus_labels_without_ase_plugin(tmp_path: Path, monkeypatch):
    module = importlib.import_module("NepTrain.core.dft.abacus.run")
    source = tmp_path / "selected.xyz"
    write(
        source,
        Atoms("Al", positions=[[0.0, 0.0, 0.0]], cell=[4.0, 4.0, 4.0], pbc=True),
    )
    resources = tmp_path / "resources"
    _resource_files(resources)
    monkeypatch.setenv("NEPTRAIN_ABACUS_COMMAND", _fake_command(tmp_path, _ordinary_log()))

    frames = module.run_abacus(_arguments(tmp_path, source, resources))

    assert len(frames) == 1
    np.testing.assert_allclose(frames[0].get_forces(), [[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(
        frames[0].info["virial"],
        np.diag([-10.0, -20.0, -30.0]) * 64.0 * 0.0006241509125883258,
    )
    case = tmp_path / "work" / "000001-Al"
    assert "Al.PBE" not in (case / "STRU").read_text(encoding="utf-8")
    assert "Al.UPF" in (case / "STRU").read_text(encoding="utf-8")
    assert "1 1 1 0 0 0" in (case / "KPT").read_text(encoding="utf-8")
    assert str(resources.resolve()) in (case / "INPUT").read_text(encoding="utf-8")
    assert "smearing_method gaussian" in (case / "INPUT").read_text(encoding="utf-8")
    manifest = (case / "abacus-result.json").read_text(encoding="utf-8")
    assert '"status": "completed"' in manifest
    assert '"parser_version": "neptrain-abacus-running-scf-v2"' in manifest
    assert '"electronic_mode": "non_spin_polarized"' in manifest
    assert '"spin_force_labels": false' in manifest
    input_manifest = json.loads((case / "abacus-input.json").read_text())
    resource_record = input_manifest["resources"]["Al"]
    assert resource_record["pseudopotential_sha256"] == hashlib.sha256(
        (resources / "Al.UPF").read_bytes()
    ).hexdigest()
    assert resource_record["orbital_sha256"] == hashlib.sha256(
        (resources / "Al.ORB").read_bytes()
    ).hexdigest()


def test_native_abacus_rejects_an_empty_command_before_creating_a_case(
    tmp_path: Path, monkeypatch
):
    module = importlib.import_module("NepTrain.core.dft.abacus.run")
    source = tmp_path / "selected.xyz"
    write(
        source,
        Atoms(
            "Al",
            positions=[[0.0, 0.0, 0.0]],
            cell=[4.0, 4.0, 4.0],
            pbc=True,
        ),
    )
    resources = tmp_path / "resources"
    _resource_files(resources)
    monkeypatch.setenv("NEPTRAIN_ABACUS_COMMAND", "")

    with pytest.raises(NativeAbacusError, match="must not be empty"):
        module.run_abacus(_arguments(tmp_path, source, resources))

    assert not (tmp_path / "work").exists()


def test_native_abacus_spin_roundtrip_writes_deltaspin_and_replaces_mforce(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "selected.xyz"
    atoms = Atoms(
        "FeAlFe",
        positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        cell=[6, 6, 6],
        pbc=True,
    )
    spin = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    atoms.set_array("spin", spin)
    atoms.set_array("mforce", np.full((3, 3), 99.0))
    write(source, atoms, format="extxyz")
    resources = tmp_path / "resources"
    _resource_files(resources, elements=("Fe", "Al"))
    arguments = _arguments(tmp_path, source, resources)
    monkeypatch.setenv("NEPTRAIN_ABACUS_COMMAND", _fake_command(tmp_path, _spin_log()))

    result = label(
        LabelRequest(
            source=source,
            output_file=tmp_path / "spin-labeled.xyz",
            work_dir=tmp_path / "work",
            settings={
                "input_file": Path(arguments.incar),
                "resource_dir": resources,
                "use_gamma": True,
                "kpoint_mode": "kpoints",
            },
        ),
        "abacus",
    )

    assert len(result.frames) == 1
    frame = result.frames[0]
    np.testing.assert_allclose(frame.arrays["spin"], spin)
    np.testing.assert_allclose(
        frame.arrays["mforce"],
        [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9], [0.4, 0.5, 0.6]],
    )
    assert frame.info["spin_constraint_rms_uB"] == pytest.approx(0.0)
    assert frame.info["dft_electronic_mode"] == "constrained_vector_spin_force"
    reread = read(result.output_file)
    np.testing.assert_allclose(reread.arrays["spin"], spin)
    np.testing.assert_allclose(reread.arrays["mforce"], frame.arrays["mforce"])

    case = tmp_path / "work" / "000001-AlFe2"
    rendered_input = (case / "INPUT").read_text(encoding="utf-8")
    for setting in (
        "nspin 4",
        "noncolin 1",
        "sc_mag_switch 1",
        "symmetry 0",
    ):
        assert setting in rendered_input
    assert "sc_direction_only" not in rendered_input
    structure = (case / "STRU").read_text(encoding="utf-8")
    assert "mag 1 0 0 sc 1 1 1" in structure
    assert "mag 0 0 3 sc 1 1 1" in structure
    assert "mag 0 2 0 sc 1 1 1" in structure


def test_native_abacus_spin_reports_vector_rms_per_atom(tmp_path: Path, monkeypatch):
    source = tmp_path / "selected.xyz"
    atoms = Atoms(
        "FeAlFe",
        positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        cell=[6, 6, 6],
        pbc=True,
    )
    atoms.set_array(
        "spin",
        np.asarray([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
    )
    write(source, atoms, format="extxyz")
    resources = tmp_path / "resources"
    _resource_files(resources, elements=("Fe", "Al"))
    arguments = _arguments(tmp_path, source, resources)
    monkeypatch.setenv("NEPTRAIN_ABACUS_COMMAND", _fake_command(tmp_path, _spin_log()))

    result = label(
        LabelRequest(
            source=source,
            output_file=tmp_path / "spin-labeled.xyz",
            work_dir=tmp_path / "work",
            settings={
                "input_file": Path(arguments.incar),
                "resource_dir": resources,
                "use_gamma": True,
                "kpoint_mode": "kpoints",
            },
        ),
        "abacus",
    )

    assert result.frames[0].info["spin_constraint_rms_uB"] == pytest.approx(
        1.0 / np.sqrt(3.0)
    )
    assert result.frames[0].info[INPUT_STRUCTURE_ID_KEY] == structure_id(atoms)
    assert structure_id(result.frames[0]) != structure_id(atoms)


def test_abacus_parser_keeps_mforce_available_for_future_spin_adapter(tmp_path: Path):
    log = tmp_path / "running_scf.log"
    log.write_text(_spin_log(), encoding="utf-8")

    parsed = parse_running_scf(log, expected_atoms=3)

    np.testing.assert_allclose(
        parsed.magnetization,
        [[1.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 2.0, 0.0]],
    )
    np.testing.assert_allclose(
        parsed.mforce,
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
    )


def test_native_abacus_spin_requires_magnetic_force(tmp_path: Path, monkeypatch):
    source = tmp_path / "selected.xyz"
    atoms = Atoms(
        "FeAlFe",
        positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        cell=[6, 6, 6],
        pbc=True,
    )
    atoms.set_array(
        "spin",
        np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
    )
    write(source, atoms, format="extxyz")
    resources = tmp_path / "resources"
    _resource_files(resources, elements=("Fe", "Al"))
    log_without_mforce = _spin_log().split(" Magnetic force (eV/uB)", 1)[0]
    log_without_mforce += "\n !FINAL_ETOT_IS -12.5 eV\n"
    monkeypatch.setenv(
        "NEPTRAIN_ABACUS_COMMAND", _fake_command(tmp_path, log_without_mforce)
    )

    module = importlib.import_module("NepTrain.core.dft.abacus.run")
    with pytest.raises(NativeAbacusError, match="mandatory magnetic force"):
        module.run_abacus(_arguments(tmp_path, source, resources))


def test_vasp_production_adapter_still_rejects_spin(tmp_path: Path):
    source = tmp_path / "selected.xyz"
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    atoms.set_array("spin", np.asarray([[1.0, 0.0, 0.0]]))
    write(source, atoms, format="extxyz")

    with pytest.raises(LabelingError, match="does not produce spin/mforce"):
        label(
            LabelRequest(
                source=source,
                output_file=tmp_path / "spin-labeled.xyz",
                work_dir=tmp_path / "work",
            ),
            "vasp",
        )


def test_abacus_rejects_direction_only_for_variable_magnitude_spin(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "selected.xyz"
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    atoms.set_array("spin", np.asarray([[1.0, 0.0, 0.0]]))
    write(source, atoms, format="extxyz")
    resources = tmp_path / "resources"
    _resource_files(resources, elements=("Fe",))
    arguments = _arguments(tmp_path, source, resources)
    Path(arguments.incar).write_text(
        "INPUT_PARAMETERS\nbasis_type lcao\nsc_direction_only 1\n",
        encoding="utf-8",
    )
    marker = tmp_path / "launched"
    monkeypatch.setenv("NEPTRAIN_ABACUS_COMMAND", f"touch {shlex.quote(str(marker))}")

    module = importlib.import_module("NepTrain.core.dft.abacus.run")
    with pytest.raises(NativeAbacusError, match="direction-only"):
        module.run_abacus(arguments)
    assert not marker.exists()


def test_native_abacus_rejects_unconverged_result(tmp_path: Path, monkeypatch):
    source = tmp_path / "selected.xyz"
    write(
        source,
        Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True),
        format="extxyz",
    )
    resources = tmp_path / "resources"
    _resource_files(resources)
    unconverged = _ordinary_log().replace(
        "charge density convergence is achieved", "SCF convergence has not been achieved"
    )
    monkeypatch.setenv("NEPTRAIN_ABACUS_COMMAND", _fake_command(tmp_path, unconverged))

    with pytest.raises(NativeAbacusError, match="did not converge"):
        label(
            LabelRequest(
                source=source,
                output_file=tmp_path / "labeled.xyz",
                work_dir=tmp_path / "work",
                settings={
                    "input_file": Path(
                        _arguments(tmp_path, source, resources).incar
                    ),
                    "resource_dir": resources,
                },
            ),
            "abacus",
        )


def test_abacus_auto_mode_honors_input_kspacing(tmp_path: Path, monkeypatch):
    module = importlib.import_module("NepTrain.core.dft.abacus.run")
    source = tmp_path / "selected.xyz"
    write(
        source,
        Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True),
        format="extxyz",
    )
    resources = tmp_path / "resources"
    _resource_files(resources)
    args = _arguments(tmp_path, source, resources)
    Path(args.incar).write_text(
        "INPUT_PARAMETERS\nsuffix ABACUS\nbasis_type lcao\nkspacing 0.25\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NEPTRAIN_ABACUS_COMMAND", _fake_command(tmp_path, _ordinary_log()))

    module.run_abacus(args)

    case = tmp_path / "work" / "000001-Al"
    assert "kspacing 0.25" in (case / "INPUT").read_text(encoding="utf-8")
    assert not (case / "KPT").exists()


def test_abacus_fails_before_launch_when_pseudopotential_is_missing(
    tmp_path: Path, monkeypatch
):
    module = importlib.import_module("NepTrain.core.dft.abacus.run")
    source = tmp_path / "selected.xyz"
    write(
        source,
        Atoms("Fe", positions=[[0.0, 0.0, 0.0]], cell=[3.0, 3.0, 3.0], pbc=True),
    )
    resources = tmp_path / "resources"
    _resource_files(resources, elements=("Al",))
    marker = tmp_path / "launched"
    monkeypatch.setenv("NEPTRAIN_ABACUS_COMMAND", f"touch {shlex.quote(str(marker))}")

    with pytest.raises(RuntimeError, match="missing elements: Fe"):
        module.run_abacus(_arguments(tmp_path, source, resources))
    assert not marker.exists()


def test_abacus_resource_hash_drift_fails_before_launch(tmp_path, monkeypatch):
    source = tmp_path / "selected.xyz"
    write(
        source,
        Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True),
        format="extxyz",
    )
    resources = tmp_path / "resources"
    _resource_files(resources)
    (resources / "Al.UPF").write_text(
        '<PP_HEADER element="Al"/> changed\n',
        encoding="utf-8",
    )
    marker = tmp_path / "launched"
    monkeypatch.setenv(
        "NEPTRAIN_ABACUS_COMMAND",
        f"touch {shlex.quote(str(marker))}",
    )

    with pytest.raises(AbacusResourceError, match="hash mismatch for Al"):
        importlib.import_module("NepTrain.core.dft.abacus.run").run_abacus(
            _arguments(tmp_path, source, resources)
        )
    assert not marker.exists()


def test_abacus_legacy_resource_collection_can_merge_two_directories(tmp_path: Path):
    structure_resources = tmp_path / "structure-resources"
    working_resources = tmp_path / "working-resources"
    structure_resources.mkdir()
    working_resources.mkdir()
    (structure_resources / "Fe.UPF").write_text('<PP_HEADER element="Fe"/>\n')
    (working_resources / "Al.UPF").write_text('<PP_HEADER element="Al"/>\n')

    StructureVar.init(structure_resources)
    StructureVar.init(working_resources, reset=False)

    assert StructureVar.pp_files == {"Fe": "Fe.UPF", "Al": "Al.UPF"}
