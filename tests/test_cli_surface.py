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
