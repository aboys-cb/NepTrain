from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write

from NepTrain.cli.cli import (
    run_project_command,
    run_resume_command,
    run_status_command,
)
from NepTrain.core.workflow import (
    WorkflowError,
    WorkflowResume,
    workflow_status,
    extend_workflow,
    prepare_workflow,
    resume_workflow,
    start_workflow,
)
from NepTrain.core.workflow_workspace import WorkflowWorkspace


def _write(path: Path, text: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_labeled(path: Path) -> Path:
    atoms = Atoms(
        "Fe",
        positions=[[0.0, 0.0, 0.0]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-1.0,
        forces=np.zeros((1, 3)),
    )
    atoms.info["virial"] = np.zeros((3, 3))
    ase_write(path, atoms, format="extxyz")
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    initial = _write_labeled(tmp_path / "initial.xyz")
    _write(tmp_path / "nep.in")
    ase_write(
        tmp_path / "structure.xyz",
        Atoms("Fe", positions=[[0.0, 0.0, 0.0]]),
        format="extxyz",
    )
    _write(tmp_path / "lammps.in", "run {{ steps }}\n")
    _write_labeled(tmp_path / "validation.xyz")
    _write(tmp_path / "gpu-env.sh", "module load cuda\n")
    _write(tmp_path / "cpu-env.sh", "module load lammps\n")
    _write(tmp_path / "dft-env.sh", "module load vasp\n")
    config = _write(
        tmp_path / "job.yaml",
        """
schema_version: 7
training:
  backend: torchnep
  initial_path: ./initial.xyz
  config_path: ./nep.in
  device: cuda
md:
  backend: lammps
  inference_backend: cpu
  spin: false
sampling:
  routes:
    - id: default
      structures: [./structure.xyz]
      template_path: ./lammps.in
      conditions:
        temperature_path: [300, 500]
        production_temperatures: [300, 500]
        pressure: 0
      progression:
        steps:
          smoke_passed: 10
          short_stable: 40
          long_stable: 160
          production_ready: 640
        replicas:
          smoke_passed: 1
          short_stable: 1
          long_stable: 2
          production_ready: 3
  candidate_pool:
    pre_failure_frames: 2
    bad_tail_frames: 1
    health: {}
  selection:
    max_selected: 100
    novelty: auto
dft:
  backend: toy
evaluation:
  validation_path: ./validation.xyz
  max_rmse:
    energy_rmse: 1.0
    force_rmse: 1.0
workflow:
  id: controller-smoke
  max_model_generations: 3
  seed: 17
execution:
  poll_interval: 0.2
  stage_targets:
    training: training
    sampling: cpu
    labeling: dft
    analysis: cpu
  sampling_route_targets: {}
  targets:
    training:
      executor: slurm
      partition: 16V100
      qos: flood-1o2gpu
      gpus_per_node: 1
      setup_script: ./gpu-env.sh
    cpu:
      executor: slurm
      partition: DSPRHBM
      qos: rush-cpu
      cpus_per_task: 4
      setup_script: ./cpu-env.sh
    dft:
      executor: slurm
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


def test_workflow_prepares_controller_plans_and_readable_workspace(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    result = prepare_workflow(config, initial, tmp_path / "workflow")
    workspace = WorkflowWorkspace.locate(result.output_dir)

    assert result.workflow_id == "controller-smoke"
    assert len(result.plans) == 3
    assert all(path.parent == workspace.plans_dir for path in result.plans)
    assert workspace.tasks_dir.is_dir()
    manifest = json.loads(result.manifest.read_text())
    assert manifest["version"] == 6
    assert len(manifest["instance_id"]) == 32
    assert manifest["orchestration"] == "controller"
    assert "scripts" not in manifest

    project_text = result.config_file.read_text()
    assert "schema_version: 7" in project_text
    assert "workflow.slurm" not in project_text
    assert "execution:" in project_text
    assert "inputs/training/nep.in" in project_text
    assert "inputs/sampling/routes/default/structures/0.xyz" in project_text
    assert "inputs/validation/validation.xyz" in project_text
    plans = [json.loads(path.read_text()) for path in result.plans]
    assert [plan["max_selected"] for plan in plans] == [100, 100, 100]
    assert all("temperatures" not in plan and "steps" not in plan for plan in plans)


def test_prepare_only_cli_does_not_start_controller(tmp_path: Path, capsys):
    config, _ = _inputs(tmp_path)
    output = tmp_path / "project"
    run_project_command(
        SimpleNamespace(
            project=str(config),
            initial_training=str(tmp_path / "initial.xyz"),
            output=str(output),
            workflow_id=None,
            prepare_only=True,
            foreground=False,
            poll_interval=None,
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["protocol"] == "neptrain.workflow-control.v1"
    assert payload["project"] == str(output)
    assert payload["action"] == "prepare"
    assert "started" not in payload
    assert payload["next_action"] == f"neptrain workflow run {output}"
    assert not (output / ".neptrain/controller.pid").exists()


def test_project_run_uses_the_same_control_schema_as_directory_run(
    tmp_path: Path, monkeypatch, capsys
):
    config, _ = _inputs(tmp_path)
    output = tmp_path / "project"

    def fake_start(path, *, foreground=False, poll_interval=None):
        assert Path(path) == output
        assert foreground is False
        assert poll_interval is None
        return 123

    monkeypatch.setattr("NepTrain.core.controller.start_controller", fake_start)
    run_project_command(
        SimpleNamespace(
            project=str(config),
            initial_training=str(tmp_path / "initial.xyz"),
            output=str(output),
            workflow_id=None,
            prepare_only=False,
            foreground=False,
            poll_interval=None,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "action": "start",
        "controller_pid": 123,
        "manifest": str(output / ".neptrain/manifest.json"),
        "project": str(output),
        "protocol": "neptrain.workflow-control.v1",
        "workflow_id": "controller-smoke",
    }


def test_status_reports_prepared_controller_without_mutation(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    before = preparation.manifest.read_bytes()

    status = workflow_status(preparation.output_dir)

    assert status.state == "prepared"
    assert status.completed_generations == 0
    assert status.generation == 1
    assert status.stage == "train"
    assert status.jobs == ()
    assert status.next_action.startswith("neptrain workflow run ")
    assert [item["state"] for item in status.generations] == [
        "not_started",
        "not_started",
        "not_started",
    ]
    assert preparation.manifest.read_bytes() == before


def test_status_uses_controller_task_instead_of_scheduler_chain(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    workspace = WorkflowWorkspace.locate(preparation.output_dir)
    workspace.controller_file.write_text(
        json.dumps(
            {
                "protocol": "neptrain.controller.v1",
                "workflow_id": preparation.workflow_id,
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

    status = workflow_status(preparation.output_dir)

    assert status.state == "paused"
    assert "not running" in status.reason
    assert status.jobs[0]["script"] == "training/train"
    assert status.jobs[0]["job_id"] == "991"
    assert status.jobs[0]["state"] == "RUNNING"


def test_status_cli_is_scientific_and_controller_focused(tmp_path: Path, capsys):
    config, initial = _inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    run_status_command(
        SimpleNamespace(project=str(preparation.output_dir), json=False, jobs=False)
    )
    output = capsys.readouterr().out
    assert "State: prepared" in output
    assert "Ledger: generation 1, stage train" in output
    assert "G1 not started: FPS selects up to 100" in output
    assert "Executor: 0/0 stages completed" in output


def test_completed_workflow_extends_plans(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    original = [path.read_bytes() for path in preparation.plans]
    generations = {}
    workspace = WorkflowWorkspace.locate(preparation.output_dir)
    for path in preparation.plans:
        plan = json.loads(path.read_text())
        generation = int(plan["generation"])
        artifacts = {}
        for name in ("activated_model", "training_set", "signals"):
            artifact = (
                workspace.generation_dir(generation) / "fixture" / f"{name}.dat"
            )
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(
                f"generation {generation} {name}\n",
                encoding="utf-8",
            )
            artifacts[name] = {
                "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        record = {
            "plan_sha256": _hash(plan),
            "stages": {
                "train": {},
                "explore": {},
                "select": {},
                "label": {},
                "diagnose": {},
                "merge": {"artifacts": {"training_set": artifacts["training_set"]}},
                "retrain": {},
                "evaluate": {
                    "artifacts": {
                        "signals": artifacts["signals"],
                        "activated_model": artifacts["activated_model"],
                    },
                    "metrics": {"accepted": True},
                },
            },
            "complete": True,
            "accepted": True,
        }
        record["publication"] = workspace.prepare_generation_publication(
            generation,
            record,
        )
        workspace.activate_generation(generation, record)
        generations[str(generation)] = record
    workspace.ledger.write_text(
        json.dumps({"version": 1, "workflow_id": preparation.workflow_id, "generations": generations}),
        encoding="utf-8",
    )

    extended = extend_workflow(preparation.output_dir, 5)

    assert len(extended.plans) == 5
    assert [path.read_bytes() for path in extended.plans[:3]] == original
    manifest = json.loads(extended.manifest.read_text())
    assert manifest["extensions"][-1]["to_generations"] == 5
    assert "script_start_index" not in manifest["extensions"][-1]


def test_extension_requires_completed_accepted_prefix(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    with pytest.raises(WorkflowError, match="only be extended"):
        extend_workflow(preparation, 4)


def test_workflow_rejects_prepared_input_drift(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    preparation.config_file.write_text("changed: true\n", encoding="utf-8")
    with pytest.raises(WorkflowError, match="artifact drifted"):
        workflow_status(preparation.output_dir)


def test_prepare_rejects_invalid_training_labels_before_any_task_exists(tmp_path):
    config, initial = _inputs(tmp_path)
    initial.write_text("not extxyz\n", encoding="utf-8")

    with pytest.raises(WorkflowError, match="initial training dataset is invalid"):
        prepare_workflow(config, initial, tmp_path / "workflow")
    assert not (tmp_path / "workflow").exists()


def test_prepare_rejects_sampling_spin_mode_mismatch(tmp_path):
    config, initial = _inputs(tmp_path)
    spin_frame = Atoms(
        "Fe",
        positions=[[0.0, 0.0, 0.0]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    spin_frame.set_array("spin", np.asarray([[1.0, 0.0, 0.0]]))
    ase_write(tmp_path / "structure.xyz", spin_frame, format="extxyz")

    with pytest.raises(WorkflowError, match="must all be ordinary.*md.spin"):
        prepare_workflow(config, initial, tmp_path / "workflow")


def test_status_reports_corrupt_controller_metadata_without_traceback(
    tmp_path: Path,
):
    config, initial = _inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    workspace = WorkflowWorkspace.locate(preparation.output_dir)
    workspace.controller_file.write_text(
        '{"state": "idle", "history": "not-a-list"}',
        encoding="utf-8",
    )

    with pytest.raises(WorkflowError, match="controller state is malformed"):
        workflow_status(preparation.output_dir)

    workspace.controller_file.write_text("{", encoding="utf-8")
    with pytest.raises(WorkflowError, match="cannot read controller state JSON"):
        workflow_status(preparation.output_dir)


def test_run_starts_existing_prepared_workflow_directory(
    tmp_path: Path, monkeypatch, capsys
):
    config, initial = _inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    expected = WorkflowResume(
        preparation.workflow_id,
        "start",
        preparation.manifest,
        controller_pid=123,
    )

    def fake_start(path, *, foreground=False, poll_interval=None):
        assert Path(path) == preparation.output_dir
        assert foreground is False
        assert poll_interval is None
        return expected

    monkeypatch.setattr("NepTrain.core.workflow.start_workflow", fake_start)
    run_project_command(
        SimpleNamespace(
            project=str(preparation.output_dir),
            initial_training=None,
            output=None,
            workflow_id=None,
            prepare_only=False,
            foreground=False,
            poll_interval=None,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["protocol"] == "neptrain.workflow-control.v1"
    assert payload["action"] == "start"
    assert payload["project"] == str(preparation.output_dir)
    assert payload["controller_pid"] == 123
    assert payload["manifest"] == str(preparation.manifest)


def test_run_directory_rejects_project_creation_options(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")

    with pytest.raises(WorkflowError, match="--prepare-only"):
        run_project_command(
            SimpleNamespace(
                project=str(preparation.output_dir),
                initial_training=None,
                output=None,
                workflow_id=None,
                prepare_only=True,
                foreground=False,
                poll_interval=None,
            )
        )


def test_resume_command_uses_existing_workflow_interface(
    tmp_path: Path, monkeypatch, capsys
):
    config, initial = _inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    expected = WorkflowResume(
        preparation.workflow_id,
        "resume",
        preparation.manifest,
        controller_pid=123,
    )

    def fake_resume(path):
        assert Path(path) == preparation.output_dir
        return expected

    monkeypatch.setattr("NepTrain.core.workflow.resume_workflow", fake_resume)
    run_resume_command(SimpleNamespace(project=str(preparation.output_dir)))

    payload = json.loads(capsys.readouterr().out)
    assert payload["protocol"] == "neptrain.workflow-control.v1"
    assert payload["action"] == "resume"
    assert payload["project"] == str(preparation.output_dir)
    assert payload["controller_pid"] == 123
    assert "job_ids" not in payload


def test_resume_completed_workflow_is_an_idempotent_noop(
    tmp_path: Path, monkeypatch
):
    config, initial = _inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    monkeypatch.setattr(
        "NepTrain.core.workflow.workflow_status",
        lambda _path: SimpleNamespace(state="complete"),
    )

    result = resume_workflow(preparation.output_dir)

    assert result.action == "complete"
    assert result.controller_pid is None


def test_resume_prepared_workflow_requires_run(
    tmp_path: Path, monkeypatch
):
    config, initial = _inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    with pytest.raises(WorkflowError, match="has not been started.*workflow run"):
        resume_workflow(preparation.output_dir)


def test_run_rejects_a_recovery_state_and_points_to_resume(
    tmp_path: Path, monkeypatch
):
    config, initial = _inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    monkeypatch.setattr(
        "NepTrain.core.workflow.workflow_status",
        lambda _path: SimpleNamespace(state="failed"),
    )

    with pytest.raises(
        WorkflowError,
        match=r"run only starts a prepared workflow.*workflow resume",
    ):
        start_workflow(preparation.output_dir)
