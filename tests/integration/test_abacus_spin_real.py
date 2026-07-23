"""Opt-in end-to-end regression for the pinned Fe2 DeltaSpin case."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase.io import read


EXPECTED_ABACUS_COMMIT = "de434f18e0c5f86e5f185db4958daab71b113666"
EXPECTED_ENERGY_EV = -6774.547461801574
EXPECTED_SPIN = np.asarray(
    [
        [1.1547004937, 1.1547004941, 1.1547004968],
        [-1.3333332841, 0.6666666392, 1.3333332840],
    ]
)
EXPECTED_MFORCE = np.asarray(
    [
        [0.1559439343, 0.2102762101, 0.2286433037],
        [-0.1906286548, 0.1425997438, 0.2533287115],
    ]
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "abacus_spin_fe2"

pytestmark = pytest.mark.real_abacus_spin


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is required for the real ABACUS spin test")
    return value


def test_real_abacus_spin_cli_roundtrip(tmp_path: Path):
    if os.environ.get("NEPTRAIN_RUN_REAL_ABACUS_SPIN") != "1":
        pytest.skip("set NEPTRAIN_RUN_REAL_ABACUS_SPIN=1 on a compute node")

    source_commit = _required_environment("NEPTRAIN_ABACUS_SOURCE_COMMIT")
    assert source_commit == EXPECTED_ABACUS_COMMIT
    command = _required_environment("NEPTRAIN_ABACUS_COMMAND")
    resources = Path(
        _required_environment("NEPTRAIN_ABACUS_SPIN_RESOURCES")
    ).resolve()
    assert resources.is_dir()
    assert any(path.suffix.lower() == ".upf" for path in resources.iterdir())
    assert any(path.suffix.lower() == ".orb" for path in resources.iterdir())

    output = tmp_path / "labeled-spin.xyz"
    work = tmp_path / "work"
    environment = os.environ.copy()
    environment["NEPTRAIN_ABACUS_COMMAND"] = command
    python_path = str(ROOT / "src")
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "NepTrain.cli.cli",
            "dft",
            str(FIXTURE / "selected-spin.xyz"),
            "--abacus",
            "--in",
            str(FIXTURE / "INPUT"),
            "--resource-dir",
            str(resources),
            "--directory",
            str(work),
            "--out",
            str(output),
            "--kspacing",
            "0.4",
            "-n",
            "4",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
    )

    frame = read(output)
    assert len(frame) == 2
    assert frame.get_potential_energy() == pytest.approx(
        EXPECTED_ENERGY_EV, abs=1e-4
    )
    np.testing.assert_allclose(frame.arrays["spin"], EXPECTED_SPIN, atol=1e-5)
    np.testing.assert_allclose(
        frame.arrays["mforce"], EXPECTED_MFORCE, atol=1e-5
    )
    assert np.isfinite(frame.get_forces()).all()
    assert np.isfinite(np.asarray(frame.info["virial"])).all()
    assert frame.info["spin_constraint_rms_uB"] < 1e-5
    assert not np.allclose(frame.arrays["mforce"], 99.0)

    manifest_path = (
        work / "000001-Fe2" / "attempt-0001" / "abacus-result.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["returncode"] == 0
    assert manifest["parser_version"] == "neptrain-abacus-running-scf-v2"
    assert manifest["result"] == {
        "atom_count": 2,
        "energy_eV": pytest.approx(EXPECTED_ENERGY_EV, abs=1e-4),
        "has_magnetization": True,
        "has_mforce": True,
        "scf_iterations": 20,
    }
