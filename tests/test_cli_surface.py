from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from NepTrain.cli.cli import _print_manual_status


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


def test_workflow_stop_exposes_explicit_job_preservation():
    completed = _help("workflow", "stop", "--help")
    assert completed.returncode == 0
    assert "--keep-jobs" in completed.stdout
    assert "--cancel-jobs" not in completed.stdout


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


def test_workflow_init_returns_success_for_console_entrypoint(tmp_path):
    completed = _help(
        "workflow",
        "init",
        "--profile",
        "local",
        "--directory",
        str(tmp_path),
    )

    assert completed.returncode == 0
    assert (tmp_path / "project.yaml").is_file()


def test_workflow_run_recognizes_directory_as_prepared_workflow(tmp_path):
    completed = _help("workflow", "run", str(tmp_path))

    assert completed.returncode == 2
    assert "prepared workflow manifest does not exist" in completed.stderr
    assert "requires a project YAML file" not in completed.stderr


def test_manual_commands_offer_human_output_and_explicit_json():
    for command in ("train", "md", "dft"):
        completed = _help(command, "--help")
        assert completed.returncode == 0
        assert "--json" in completed.stdout
    completed = _help("task", "status", "--help")
    assert completed.returncode == 0
    assert "--json" in completed.stdout


def test_manual_status_is_human_readable_by_default(capsys):
    value = {
        "operation_id": "dft-abc",
        "kind": "dft",
        "state": "submitted",
        "job_id": "123",
        "completed": 0,
        "total": 4,
        "run_directory": "/tmp/dft-run",
        "reason": "0/4 shards completed",
        "errors": [],
    }

    _print_manual_status(value)

    output = capsys.readouterr().out
    assert "Task: dft (dft-abc)" in output
    assert "State: submitted" in output
    assert "Progress: 0/4" in output
    assert "Next: neptrain task wait /tmp/dft-run" in output
    assert not output.lstrip().startswith("{")


def test_manual_status_json_is_opt_in(capsys):
    value = {"kind": "dft", "state": "complete"}

    _print_manual_status(value, json_output=True)

    assert json.loads(capsys.readouterr().out) == value
