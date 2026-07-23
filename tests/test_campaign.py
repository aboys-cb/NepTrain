from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from NepTrain.cli.cli import run_project_command, run_status_command
from NepTrain.core.campaign import (
    CampaignError,
    CampaignResume,
    campaign_status,
    extend_campaign,
    prepare_campaign,
    resume_campaign,
    submit_campaign,
)
from NepTrain.core.campaign_workspace import CampaignWorkspace


def _write(path: Path, text: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    initial = _write(tmp_path / "initial.xyz")
    _write(tmp_path / "nep.in")
    _write(tmp_path / "structure.xyz")
    _write(tmp_path / "validation.xyz")
    _write(tmp_path / "gpu-env.sh", "module load cuda\n")
    _write(tmp_path / "cpu-env.sh", "module load lammps\n")
    _write(tmp_path / "dft-env.sh", "module load vasp\n")
    config = _write(
        tmp_path / "job.yaml",
        """
schema_version: 2
training:
  backend: torchnep
  config_path: ./nep.in
  device: cuda
md:
  backend: lammps
  structures: ./structure.xyz
  inference_backend: cpu
  spin: false
dft:
  software: toy
evaluation:
  validation_path: ./validation.xyz
  max_rmse:
    energy_rmse: 1.0
    force_rmse: 1.0
campaign:
  id: controller-smoke
  generations: 3
  seed: 17
  initial_candidates: 12
  dft_budget: 6
  minimum_dft_budget: 2
  initial_steps: 10
  temperatures: [300, 500]
  frame_stride: 3
  command: NepTrain
  slurm:
    training:
      partition: 16V100
      qos: flood-1o2gpu
      gpus_per_node: 1
      setup_script: ./gpu-env.sh
    cpu:
      partition: DSPRHBM
      qos: rush-cpu
      cpus_per_task: 4
      setup_script: ./cpu-env.sh
    dft:
      partition: 16V100
      qos: flood-1o2gpu
      gpus_per_node: 1
      setup_script: ./dft-env.sh
""",
    )
    return config, initial


def _hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_campaign_prepares_controller_plans_and_readable_workspace(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    result = prepare_campaign(config, initial, tmp_path / "campaign")
    workspace = CampaignWorkspace.locate(result.output_dir)

    assert result.campaign_id == "controller-smoke"
    assert result.scripts == ()
    assert len(result.plans) == 3
    assert all(path.parent == workspace.plans_dir for path in result.plans)
    assert workspace.tasks_dir.is_dir()
    manifest = json.loads(result.manifest.read_text())
    assert manifest["version"] == 3
    assert manifest["orchestration"] == "controller-v1"
    assert manifest["scripts"] == []

    project_text = result.config_file.read_text()
    assert "schema_version: 3" in project_text
    assert "campaign.slurm" not in project_text
    assert "execution:" in project_text
    assert "inputs/training/nep.in" in project_text
    assert "inputs/md/structures.xyz" in project_text
    assert "inputs/validation/validation.xyz" in project_text
    plans = [json.loads(path.read_text()) for path in result.plans]
    assert [plan["steps"] for plan in plans] == [10, 40, 40]
    assert [plan["dft_budget"] for plan in plans] == [6, 5, 4]
    assert [plan["temperatures"] for plan in plans] == [[300.0], [300.0], [300.0, 500.0]]


def test_prepare_only_cli_does_not_start_controller(tmp_path: Path, capsys):
    config, _ = _inputs(tmp_path)
    output = tmp_path / "project"
    run_project_command(
        SimpleNamespace(
            project=str(config),
            initial_training=str(tmp_path / "initial.xyz"),
            output=str(output),
            campaign_id=None,
            prepare_only=True,
            foreground=False,
            poll_interval=None,
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["project"] == str(output)
    assert payload["started"] is False
    assert not (output / ".neptrain/controller.pid").exists()


def test_status_reports_prepared_controller_without_mutation(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    before = preparation.manifest.read_bytes()

    status = campaign_status(preparation.output_dir)

    assert status.state == "prepared"
    assert status.completed_generations == 0
    assert status.generation == 1
    assert status.stage == "train"
    assert status.jobs == ()
    assert status.next_action.startswith("NepTrain run ")
    assert [item["state"] for item in status.generations] == [
        "not_started",
        "not_started",
        "not_started",
    ]
    assert preparation.manifest.read_bytes() == before


def test_status_uses_controller_task_instead_of_scheduler_chain(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    workspace = CampaignWorkspace.locate(preparation.output_dir)
    workspace.controller_file.write_text(
        json.dumps(
            {
                "protocol": "neptrain.controller.v1",
                "campaign_id": preparation.campaign_id,
                "state": "launching",
                "current": {
                    "task_id": "abc",
                    "generation": 1,
                    "stage": "train",
                    "target": "training",
                    "attempt": 1,
                    "handle": {
                        "execution_id": "991",
                    },
                    "observed_state": "running",
                },
                "history": [],
            }
        ),
        encoding="utf-8",
    )

    status = campaign_status(preparation.output_dir)

    assert status.state == "paused"
    assert "not running" in status.reason
    assert status.jobs[0]["script"] == "training/train"
    assert status.jobs[0]["job_id"] == "991"
    assert status.jobs[0]["state"] == "RUNNING"


def test_status_cli_is_scientific_and_controller_focused(tmp_path: Path, capsys):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    run_status_command(
        SimpleNamespace(project=str(preparation.output_dir), json=False, jobs=False)
    )
    output = capsys.readouterr().out
    assert "State: prepared" in output
    assert "Ledger: generation 1, stage train" in output
    assert "G1 not started: plan 12 candidates, DFT budget 6" in output
    assert "Executor: 0/0 stages completed" in output


def test_completed_campaign_extends_plans_without_job_scripts(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    original = [path.read_bytes() for path in preparation.plans]
    generations = {}
    for path in preparation.plans:
        plan = json.loads(path.read_text())
        generations[str(plan["generation"])] = {
            "plan_sha256": _hash(plan),
            "stages": {stage: {} for stage in (
                "train", "explore", "select", "label", "diagnose", "merge", "retrain", "evaluate"
            )},
            "complete": True,
            "accepted": True,
        }
    workspace = CampaignWorkspace.locate(preparation.output_dir)
    workspace.ledger.write_text(
        json.dumps({"version": 1, "campaign_id": preparation.campaign_id, "generations": generations}),
        encoding="utf-8",
    )

    extended = extend_campaign(preparation.output_dir, 5)

    assert len(extended.plans) == 5
    assert extended.scripts == ()
    assert [path.read_bytes() for path in extended.plans[:3]] == original
    manifest = json.loads(extended.manifest.read_text())
    assert manifest["extensions"][-1]["to_generations"] == 5
    assert "script_start_index" not in manifest["extensions"][-1]


def test_extension_requires_completed_accepted_prefix(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    with pytest.raises(CampaignError, match="only be extended"):
        extend_campaign(preparation, 4)


def test_campaign_rejects_prepared_input_drift(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    preparation.config_file.write_text("changed: true\n", encoding="utf-8")
    with pytest.raises(CampaignError, match="artifact drifted"):
        campaign_status(preparation.output_dir)


def test_controller_campaign_rejects_legacy_dependency_submission(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    with pytest.raises(CampaignError, match="do not submit a Slurm dependency"):
        submit_campaign(preparation)


def test_run_on_existing_project_uses_resume_interface(tmp_path: Path, monkeypatch, capsys):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    expected = CampaignResume(
        preparation.campaign_id,
        "resume",
        ("123",),
        preparation.manifest,
    )
    monkeypatch.setattr("NepTrain.core.campaign.resume_campaign", lambda _path: expected)

    run_project_command(
        SimpleNamespace(
            project=str(preparation.output_dir),
            initial_training=None,
            output=None,
            campaign_id=None,
            prepare_only=False,
            foreground=False,
            poll_interval=None,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "resume"
    assert payload["job_ids"] == ["123"]


def test_run_existing_project_forwards_foreground_controller_options(
    tmp_path: Path, monkeypatch, capsys
):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    expected = CampaignResume(
        preparation.campaign_id,
        "resume",
        (),
        preparation.manifest,
        controller_exit_code=0,
    )
    calls = []

    def fake_resume(path, **options):
        calls.append((Path(path), options))
        return expected

    monkeypatch.setattr("NepTrain.core.campaign.resume_campaign", fake_resume)

    run_project_command(
        SimpleNamespace(
            project=str(preparation.output_dir),
            initial_training=None,
            output=None,
            campaign_id=None,
            prepare_only=False,
            foreground=True,
            poll_interval=0.5,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert calls == [
        (preparation.output_dir, {"foreground": True, "poll_interval": 0.5})
    ]
    assert payload["controller_exit_code"] == 0
    assert "job_ids" not in payload


def test_resume_completed_campaign_is_an_idempotent_noop(
    tmp_path: Path, monkeypatch
):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    monkeypatch.setattr(
        "NepTrain.core.campaign.campaign_status",
        lambda _path: SimpleNamespace(state="complete"),
    )

    result = resume_campaign(preparation.output_dir)

    assert result.action == "complete"
    assert result.job_ids == ()
    assert result.controller_pid is None
