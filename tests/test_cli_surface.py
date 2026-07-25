from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _help(*arguments):
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    return subprocess.run(
        [sys.executable, "-m", "NepTrain.cli.cli", *arguments],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def test_primary_help_only_shows_the_new_product_surface():
    completed = _help("--help")
    assert completed.returncode == 0
    assert "{train,md,dft,select,perturb,workflow,task,doctor,smoke}" in completed.stdout
    for removed in ("vasp", "gpumd", "migrate"):
        assert f"    {removed} " not in completed.stdout
    assert "stage-worker" not in completed.stdout
    assert "manual-worker" not in completed.stdout


def test_workflow_commands_are_grouped():
    completed = _help("workflow", "--help")
    assert completed.returncode == 0
    assert "{init,run,status,resume,extend,stop}" in completed.stdout


def test_workflow_stop_exposes_explicit_job_cancellation():
    completed = _help("workflow", "stop", "--help")
    assert completed.returncode == 0
    assert "--cancel-jobs" in completed.stdout


def test_md_cli_keeps_template_owned_parameters_out_of_the_interface():
    completed = _help("md", "--help")
    assert completed.returncode == 0
    for removed in (
        "--plugin-path",
        "--timestep",
        "--tdamp",
        "--pdamp",
        "--dump-interval",
        "--spin-alpha",
        "--spin-seed",
        "--midpoint-iter",
    ):
        assert removed not in completed.stdout


def test_dft_cli_rejects_competing_kpoint_overrides():
    completed = _help(
        "dft",
        "input.xyz",
        "--kspacing",
        "0.2",
        "--ka",
        "4,4,4",
    )

    assert completed.returncode == 2
    assert "not allowed with argument" in completed.stderr
