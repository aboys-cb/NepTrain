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
    _print_job_batches,
    _print_precision,
    run_project_command,
    run_resume_command,
    run_status_command,
)
from NepTrain.core.workflow import (
    WorkflowError,
    WorkflowResume,
    _generation_science,
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
    _write(
        tmp_path / "lammps.in",
        "units metal\ntimestep 0.001\nrun {{ steps }}\n",
    )
    _write_labeled(tmp_path / "validation.xyz")
    _write(tmp_path / "gpu-env.sh", "module load cuda\n")
    _write(tmp_path / "cpu-env.sh", "module load lammps\n")
    _write(tmp_path / "label-env.sh", "module load vasp\n")
    config = _write(
        tmp_path / "job.yaml",
        """
schema_version: 8
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
labeling:
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
    labeling: label
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
    label:
      executor: slurm
      partition: 16V100
      qos: flood-1o2gpu
      gpus_per_node: 1
      setup_script: ./label-env.sh
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
    assert manifest["version"] == 7
    assert manifest["structure_id_version"] == "neptrain.structure-id.v3"
    assert len(manifest["instance_id"]) == 32
    assert manifest["orchestration"] == "controller"
    assert "scripts" not in manifest

    project_text = result.config_file.read_text()
    assert "schema_version: 8" in project_text
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


def test_status_reports_notification_health_without_affecting_workflow(
    tmp_path: Path,
):
    config, initial = _inputs(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8")
        + """
notifications:
  feishu:
    webhook: https://open.feishu.cn/open-apis/bot/v2/hook/test-token
    secret: test-secret
""",
        encoding="utf-8",
    )
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    workspace = WorkflowWorkspace.locate(preparation.output_dir)
    workspace.notification_state.write_text(
        json.dumps(
            {
                "version": 1,
                "provider": "feishu",
                "events": {
                    "generation:1:accepted": {
                        "state": "failed",
                        "attempts": 1,
                        "last_error": "connection timeout",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    status = workflow_status(preparation.output_dir)

    assert status.state == "prepared"
    assert status.notifications is not None
    assert status.notifications["state"] == "degraded"
    assert status.notifications["failed"] == 1
    assert status.notifications["last_error"] == "connection timeout"


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
    assert "NepTrain · controller-smoke" in output
    assert f"路径：{preparation.output_dir}" in output
    assert "状态：待启动 | 第 1/3 代 | 训练" in output
    assert "300 K ○ → 500 K ○" in output
    assert "验证集精度：" in output
    assert "G1" in output
    assert "未开始" in output
    assert "执行批次：" not in output


def test_status_reports_live_md_temperature_and_real_ps(tmp_path: Path, capsys):
    config, initial = _inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    workspace = WorkflowWorkspace.locate(preparation.output_dir)
    bundle = workspace.tasks_dir / "live-md"
    output = bundle / ".output-building-123"
    _write(
        output / "log.lammps",
        """
Step Temp PotEng
0 300 -1
3200 500 -1
""",
    )
    workspace.controller_file.write_text(
        json.dumps(
            {
                "protocol": "neptrain.controller.v1",
                "workflow_id": preparation.workflow_id,
                "state": "running",
                "heartbeat_at": "2026-07-30T06:00:00+00:00",
                "history": [],
                "current": {
                    "kind": "task_group",
                    "generation": 1,
                    "stage": "explore",
                    "attempt": 1,
                    "tasks": [
                        {
                            "route_id": "default",
                            "temperature": 500,
                            "steps": 10_000,
                            "bundle": str(bundle),
                            "target": "cpu",
                            "handle": {"execution_id": "42"},
                            "observed_state": "running",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    status = workflow_status(preparation.output_dir)

    cells = status.sampling_routes[0]["temperatures"]
    assert cells[0]["state"] == "complete"
    assert cells[1]["state"] == "active"
    assert cells[1]["current_ps"] == pytest.approx(3.2)
    assert cells[1]["target_ps"] == pytest.approx(10.0)
    run_status_command(
        SimpleNamespace(project=str(preparation.output_dir), json=False, jobs=False)
    )
    output_text = capsys.readouterr().out
    assert "300 K ✓ → 500 K ● 3.2/10 ps" in output_text
    run_status_command(
        SimpleNamespace(project=str(preparation.output_dir), json=True, jobs=False)
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["protocol"] == "neptrain.workflow-status.v1"
    assert payload["sampling_routes"][0]["temperatures"][1][
        "current_ps"
    ] == pytest.approx(3.2)


def test_status_precision_table_shows_generation_deltas(capsys):
    generations = []
    for generation, force in ((1, 0.2), (2, 0.16)):
        generations.append(
            {
                "generation": generation,
                "state": "accepted",
                "quality": {
                    "accepted": True,
                    "validation_rmse": {
                        "energy_rmse": 0.02,
                        "force_rmse": force,
                        "virial_rmse": 0.04,
                        "mforce_rmse": 0.15,
                    },
                },
            }
        )
    status = SimpleNamespace(
        precision_basis="validation",
        generations=tuple(generations),
        generation=2,
        stage=None,
    )

    _print_precision(status)

    output = capsys.readouterr().out
    assert "验证集精度：" in output
    assert "F/meV·Å⁻¹" in output
    assert "M/meV/μB" in output
    assert "160 ↓20%" in output


def test_status_does_not_present_training_error_as_validation(capsys):
    status = SimpleNamespace(
        precision_basis=None,
        generations=(),
        generation=1,
        stage="train",
    )

    _print_precision(status)

    assert (
        "精度变化：暂无可比较数据（未配置独立验证集）"
        in capsys.readouterr().out
    )


def test_jobs_are_compacted_by_generation_stage_and_attempt(capsys):
    jobs = [
        {
            "generation": 3,
            "stage": "explore",
            "attempt": "attempt-1",
            "job_id": str(1000 + index),
            "state": "COMPLETED" if index < 12 else "RUNNING",
            "script": "cpu/explore",
        }
        for index in range(20)
    ]
    jobs.extend(
        {
            "generation": 3,
            "stage": "label",
            "attempt": "attempt-1",
            "job_id": str(2000 + index),
            "state": "COMPLETED" if index < 80 else "PENDING",
            "script": "dft/label",
        }
        for index in range(100)
    )

    _print_job_batches(jobs)

    output = capsys.readouterr().out
    assert "G3 采样 attempt-1：20 个任务 | 完成 12 | 运行 8" in output
    assert "Job 1000–1019" in output
    assert "G3 标注 attempt-1：100 个任务 | 完成 80 | 等待 20" in output
    assert len(output.splitlines()) == 4


def test_generation_status_reports_activation_result_not_retrain_candidate():
    summary = _generation_science(
        {
            "generation": 1,
            "max_selected": 1,
            "selection_novelty_threshold": 0.0,
            "completion_coverage_threshold": 0.0,
        },
        {
            "complete": True,
            "accepted": True,
            "stages": {
                "retrain": {
                    "metrics": {
                        "training_count": 97,
                        "model_updated": True,
                    }
                },
                "evaluate": {
                    "metrics": {
                        "accepted": True,
                        "active_model_sha256": "parent-model",
                        "model_updated": False,
                    }
                },
            },
        },
    )

    assert summary["training"]["after_count"] == 97
    assert summary["training"]["active_model_sha256"] == "parent-model"
    assert summary["training"]["model_updated"] is False


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


def test_extension_accepts_an_incomplete_valid_prefix(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    original = [path.read_bytes() for path in preparation.plans]

    extended = extend_workflow(preparation, 4)

    assert len(extended.plans) == 4
    assert [path.read_bytes() for path in extended.plans[:3]] == original


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
