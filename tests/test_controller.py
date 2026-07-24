from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from NepTrain.core.workflow import extend_workflow, workflow_status, prepare_workflow
from NepTrain.core.controller import (
    ControllerError,
    PersistentController,
    controller_running,
    start_controller,
    stop_controller,
    stop_workflow,
)
from NepTrain.core.execution import (
    ExecutionError,
    ExecutionHandle,
    ExecutionStatus,
    ExecutionTarget,
    ProcessExecutor,
    SlurmExecutor,
    StageTask,
    build_stage_task,
    load_stage_result,
    run_stage_worker,
)
from NepTrain.core.iteration import GenerationPlan, StageContext, StageOutcome


def _write(path: Path, text: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> GenerationPlan:
    return GenerationPlan(
        generation=1,
        seed=7,
        candidate_count=4,
        dft_budget=2,
        min_novelty=0.0,
        temperatures=(300.0,),
        pressure=0.0,
        steps=2,
        frame_stride=1,
    )


def test_portable_stage_worker_verifies_and_collects_results(tmp_path, monkeypatch):
    initial = _write(tmp_path / "initial.xyz")
    training_config = _write(tmp_path / "nep.in")
    validation = _write(tmp_path / "validation.xyz")
    model = _write(tmp_path / "model.txt")
    plan = _plan()
    target = ExecutionTarget("local", "process")
    task = build_stage_task(
        tmp_path / "tasks",
        workflow_root=tmp_path,
        workflow_id="demo",
        generation=1,
        stage="explore",
        attempt=1,
        target=target,
        plan=plan,
        config={
            "training": {"config_path": str(training_config), "initial_path": str(initial)},
            "evaluation": {"validation_path": str(validation), "max_rmse": {"energy_rmse": 1, "force_rmse": 1}},
            "md": {"backend": "lammps", "spin": False},
            "dft": {"software": "toy"},
            "workflow": {},
            "execution": {},
        },
        initial_training=initial,
        context=StageContext(
            generation=1,
            generation_dir=tmp_path / "generation",
            plan=plan,
            artifacts={"model": model},
            previous_artifacts={},
        ),
    )

    class FakeWorkflow:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_stage(self, stage, context):
            assert stage == "explore"
            assert context.artifacts["model"].read_text() == "fixture\n"
            output = _write(context.work_dir / "candidates.xyz", "candidate\n")
            return StageOutcome({"candidates": output}, {"candidate_count": 1})

    monkeypatch.setattr(
        "NepTrain.core.workflow_iteration.WorkflowIterationAdapter", FakeWorkflow
    )

    assert run_stage_worker(task.bundle) == 0
    value, outcome = load_stage_result(task.bundle)
    assert value["task_id"] == task.task_id
    assert outcome.metrics == {"candidate_count": 1}
    assert outcome.artifacts["candidates"].read_text() == "candidate\n"


def test_stage_bundle_only_copies_inputs_consumed_by_that_stage(tmp_path):
    initial = _write(tmp_path / "initial.xyz")
    validation = _write(tmp_path / "validation.xyz")
    dft_resource = tmp_path / "large-dft-resource"
    _write(dft_resource / "POTCAR", "large resource placeholder\n")
    artifacts = {
        name: _write(tmp_path / "artifacts" / f"{name}.dat")
        for name in (
            "model",
            "training_input",
            "candidates",
            "labeled",
            "training_set",
        )
    }
    task = build_stage_task(
        tmp_path / "tasks",
        workflow_root=tmp_path,
        workflow_id="filtered",
        generation=1,
        stage="explore",
        attempt=1,
        target=ExecutionTarget("local", "process"),
        plan=_plan(),
        config={
            "training": {"initial_path": str(initial)},
            "md": {"structures": str(initial), "spin": False},
            "dft": {"resource_path": str(dft_resource)},
            "evaluation": {
                "validation_path": str(validation),
                "max_rmse": {"energy_rmse": 1, "force_rmse": 1},
            },
        },
        initial_training=initial,
        context=StageContext(
            generation=1,
            generation_dir=tmp_path / "generation",
            plan=_plan(),
            artifacts=artifacts,
            previous_artifacts={},
        ),
    )

    descriptor = json.loads(task.descriptor.read_text())
    assert set(descriptor["artifacts"]) == {"model"}
    assert descriptor["config"]["dft"]["resource_path"] == str(dft_resource)
    assert not any(path.name == "POTCAR" for path in task.bundle.rglob("*"))


def test_stage_result_rejects_paths_outside_the_bundle(tmp_path):
    outside = _write(tmp_path / "outside.dat")
    bundle = tmp_path / "task"
    bundle.mkdir()
    (bundle / "result.json").write_text(
        json.dumps(
            {
                "protocol": "neptrain.stage-result.v1",
                "artifacts": {
                    "model": {
                        "path": "../outside.dat",
                        "sha256": _sha256(outside),
                        "size": outside.stat().st_size,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ExecutionError, match="inside the task bundle"):
        load_stage_result(bundle)


_STAGE_ARTIFACTS = {
    "train": ("training_input", "model"),
    "explore": ("candidates",),
    "select": ("selected_input", "selection_result"),
    "label": ("labeled",),
    "diagnose": ("acquisition_signals",),
    "merge": ("training_set",),
    "retrain": ("retrained_model",),
    "evaluate": ("signals",),
}


class ImmediateExecutor:
    def __init__(self, target, launches):
        self.target = target
        self.launches = launches

    def launch(self, task):
        self.launches.append((task.stage, task.target))
        result_root = task.bundle / "result" / "artifacts"
        artifacts = {}
        for name in _STAGE_ARTIFACTS[task.stage]:
            path = _write(result_root / name / f"{name}.dat", name + "\n")
            artifacts[name] = {
                "path": str(path.relative_to(task.bundle)),
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
        descriptor = json.loads(task.descriptor.read_text(encoding="utf-8"))
        metrics = (
            {"accepted": True, "workflow_converged": True}
            if task.stage == "evaluate"
            else {}
        )
        (task.bundle / "result.json").write_text(
            json.dumps(
                {
                    "protocol": "neptrain.stage-result.v1",
                    "task_id": task.task_id,
                    "workflow_id": task.workflow_id,
                    "generation": task.generation,
                    "stage": task.stage,
                    "plan_sha256": descriptor["identity"]["plan_sha256"],
                    "artifacts": artifacts,
                    "metrics": metrics,
                }
            ),
            encoding="utf-8",
        )
        return ExecutionHandle(
            task.task_id,
            task.target,
            "process",
            f"fake-{len(self.launches)}",
            str(task.bundle),
        )

    def inspect(self, _handle):
        return ExecutionStatus("completed")

    def collect(self, handle):
        return Path(handle.local_bundle)


def _controller_inputs(tmp_path: Path):
    initial = _write(tmp_path / "initial.xyz")
    _write(tmp_path / "nep.in")
    _write(tmp_path / "structures.xyz")
    _write(tmp_path / "validation.xyz")
    config = _write(
        tmp_path / "project.yaml",
        """
schema_version: 4
training:
  backend: gpumd
  initial_path: ./initial.xyz
  config_path: ./nep.in
md:
  backend: lammps
  structures: ./structures.xyz
  spin: false
sampling:
  mode: auto
  conditions:
    temperature_path: [300]
    production_temperatures: [300]
    pressure: 0
    spin_temperature:
  progression:
    md_runs_per_iteration: 1
    steps:
      smoke_passed: 2
      short_stable: 8
      long_stable: 32
      production_ready: 128
  candidate_pool:
    target: 4
    growth: 1
    frame_stride: 1
    pre_failure_frames: 2
    bad_tail_frames: 1
    health: {}
  selection:
    method: fps
    dft_budget: 2
    minimum_dft_budget: 1
    budget_decay: 0.75
    min_novelty: 0
dft:
  backend: toy
evaluation:
  validation_path: ./validation.xyz
  max_rmse:
    energy_rmse: 1
    force_rmse: 1
workflow:
  id: controller-test
  max_iterations: 1
execution:
  poll_interval: 0.2
  routes:
    training: gpu
    sampling: md
    labeling: dft
    analysis: cpu
  targets:
    gpu: {executor: process}
    md: {executor: process}
    dft: {executor: process}
    cpu: {executor: process}
""",
    )
    return config, initial


def test_controller_routes_every_stage_without_scheduler_dependencies(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []

    def factory(target):
        return ImmediateExecutor(target, launches)

    controller = PersistentController(preparation.output_dir, executor_factory=factory)
    for _ in range(20):
        tick = controller.tick()
        if tick.state == "complete":
            break
    else:
        raise AssertionError("controller did not complete")

    assert launches == [
        ("train", "gpu"),
        ("explore", "md"),
        ("select", "cpu"),
        ("label", "dft"),
        ("diagnose", "cpu"),
        ("merge", "cpu"),
        ("retrain", "gpu"),
        ("evaluate", "cpu"),
    ]
    ledger = json.loads((preparation.output_dir / ".neptrain/ledger.json").read_text())
    assert ledger["generations"]["1"]["accepted"] is True
    manifest = json.loads(preparation.manifest.read_text())
    assert manifest["orchestration"] == "controller-v1"
    assert manifest["scripts"] == []


class EvaluationStateExecutor(ImmediateExecutor):
    def __init__(self, target, launches, evaluation_metrics):
        super().__init__(target, launches)
        self.evaluation_metrics = evaluation_metrics

    def launch(self, task):
        handle = super().launch(task)
        if task.stage == "evaluate":
            result_path = task.bundle / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["metrics"] = dict(self.evaluation_metrics)
            result_path.write_text(json.dumps(result), encoding="utf-8")
        return handle


@pytest.mark.parametrize(
    ("metrics", "expected_state"),
    [
        (
            {
                "accepted": True,
                "workflow_converged": False,
                "workflow_stalled": False,
            },
            "budget_exhausted",
        ),
        (
            {
                "accepted": True,
                "workflow_converged": False,
                "workflow_stalled": True,
            },
            "stalled",
        ),
    ],
)
def test_controller_does_not_report_unconverged_work_as_complete(
    tmp_path, metrics, expected_state
):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []

    def factory(target):
        return EvaluationStateExecutor(target, launches, metrics)

    controller = PersistentController(
        preparation.output_dir, executor_factory=factory
    )
    for _ in range(20):
        tick = controller.tick()
        if tick.state == expected_state:
            break
    else:
        raise AssertionError(f"controller did not reach {expected_state}")

    status = workflow_status(preparation.output_dir)
    assert status.state == expected_state
    assert status.state != "complete"
    if expected_state == "budget_exhausted":
        extend_workflow(preparation.output_dir, 2)
        resumed = workflow_status(preparation.output_dir)
        assert resumed.state == "prepared"


class SameSchedulerIdExecutor(ImmediateExecutor):
    """Model independent Slurm namespaces that reuse one numeric job id."""

    def launch(self, task):
        handle = super().launch(task)
        return ExecutionHandle(
            handle.task_id,
            handle.target,
            "slurm",
            "12345",
            handle.local_bundle,
        )


def test_controller_namespaces_equal_slurm_job_ids_by_target(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []
    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: SameSchedulerIdExecutor(target, launches),
    )

    for _ in range(20):
        if controller.tick().state == "complete":
            break
    else:
        raise AssertionError("controller did not complete")

    state = json.loads(
        (preparation.output_dir / ".neptrain/controller.json").read_text()
    )
    assert len(state["history"]) == 8
    assert {item["handle"]["execution_id"] for item in state["history"]} == {
        "12345"
    }
    assert [item["target"] for item in state["history"]] == [
        "gpu",
        "md",
        "cpu",
        "dft",
        "cpu",
        "cpu",
        "gpu",
        "cpu",
    ]

    status = workflow_status(preparation.output_dir)
    assert len(status.jobs) == 8
    assert {job["job_id"] for job in status.jobs} == {"12345"}
    assert [job["script"] for job in status.jobs] == [
        "gpu/train",
        "md/explore",
        "cpu/select",
        "dft/label",
        "cpu/diagnose",
        "cpu/merge",
        "gpu/retrain",
        "cpu/evaluate",
    ]


def test_controller_refuses_drifted_workflow_inputs(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    (preparation.output_dir / "inputs/md/structures.xyz").write_text(
        "changed\n", encoding="utf-8"
    )

    with pytest.raises(ControllerError, match="artifact drifted"):
        PersistentController(preparation.output_dir)


def test_process_executor_detaches_and_reports_completion(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    script = _write(
        tmp_path / "dummy_worker.py",
        """
import json
import pathlib
import sys
bundle = pathlib.Path(sys.argv[-1])
(bundle / 'result').mkdir(exist_ok=True)
(bundle / 'result.json').write_text('{}')
(bundle / 'execution.json').write_text(json.dumps({'state': 'COMPLETED'}))
""",
    )
    task = StageTask("abc", "demo", 1, "train", "local", bundle)
    executor = ProcessExecutor(
        ExecutionTarget(
            "local",
            "process",
            command=f"{sys.executable} {script}",
        )
    )

    handle = executor.launch(task)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status = executor.inspect(handle)
        if status.terminal:
            break
        time.sleep(0.01)

    assert status.state == "completed"
    assert executor.collect(handle) == bundle


def test_process_executor_cancels_its_own_process_group(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    script = _write(
        tmp_path / "slow_worker.py",
        """
import time
time.sleep(30)
""",
    )
    task = StageTask("abc", "demo", 1, "train", "local", bundle)
    executor = ProcessExecutor(
        ExecutionTarget(
            "local",
            "process",
            command=f"{sys.executable} {script}",
        )
    )
    handle = executor.launch(task)

    status = executor.cancel(handle)

    assert status.state == "cancelled"
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        completed = subprocess.run(
            ["ps", "-p", handle.execution_id, "-o", "stat="],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            break
        if completed.stdout.strip().startswith("Z"):
            break
        time.sleep(0.02)
    else:
        raise AssertionError("cancelled process group is still running")


class SlurmRunner:
    def __init__(self):
        self.submissions = 0
        self.known = False
        self.state = "RUNNING"
        self.cancelled = []

    def __call__(self, args, **_kwargs):
        args = list(args)
        if args[0] == "sbatch":
            self.submissions += 1
            self.known = True
            return subprocess.CompletedProcess(args, 0, "123\n", "")
        if args[0] == "squeue" and "--name" in args:
            output = "123|nt-abc\n" if self.known else ""
            return subprocess.CompletedProcess(args, 0, output, "")
        if args[0] == "sacct" and "--name" in args:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[0] == "squeue" and "--jobs" in args:
            output = f"{self.state}\n" if self.state == "RUNNING" else ""
            return subprocess.CompletedProcess(args, 0, output, "")
        if args[0] == "sacct" and "--jobs" in args:
            return subprocess.CompletedProcess(
                args, 0, f"{self.state}|0:0|\n", ""
            )
        if args[0] == "scancel":
            self.cancelled.append(args[1])
            self.state = "CANCELLED"
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)


def test_slurm_executor_is_idempotent_without_afterok(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    task = StageTask("abc", "demo", 1, "train", "slurm", bundle)
    runner = SlurmRunner()
    executor = SlurmExecutor(
        ExecutionTarget("slurm", "slurm", partition="cpu", cpus_per_task=4),
        runner=runner,
    )

    first = executor.launch(task)
    second = executor.launch(task)

    assert first.execution_id == second.execution_id == "123"
    assert runner.submissions == 1
    script = (bundle / "job.sbatch").read_text()
    assert "#SBATCH --partition=cpu" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "--dependency" not in script
    assert executor.inspect(first).state == "running"
    runner.state = "COMPLETED"
    assert executor.inspect(first).state == "completed"


def test_slurm_executor_recognizes_site_annotated_cancelled_state(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    task = StageTask("abc", "demo", 1, "label", "slurm", bundle)
    runner = SlurmRunner()
    runner.known = True
    runner.state = "CANCELLED by 1478400058"
    executor = SlurmExecutor(
        ExecutionTarget("slurm", "slurm", partition="gpu"), runner=runner
    )
    handle = ExecutionHandle("abc", "slurm", "slurm", "123", str(bundle))

    status = executor.inspect(handle)

    assert status.state == "failed"
    assert status.detail == "Slurm CANCELLED exit=0:0"


def test_slurm_terminal_state_overrides_stale_running_worker_file(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    (bundle / "execution.json").write_text(
        json.dumps({"state": "RUNNING"}), encoding="utf-8"
    )
    runner = SlurmRunner()
    runner.state = "CANCELLED by 1478400058"
    executor = SlurmExecutor(
        ExecutionTarget("slurm", "slurm", partition="gpu"), runner=runner
    )
    handle = ExecutionHandle("abc", "slurm", "slurm", "123", str(bundle))

    assert executor.inspect(handle).state == "failed"


def test_slurm_executor_recovers_completed_worker_before_resubmitting(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    (bundle / "execution.json").write_text(
        json.dumps({"state": "COMPLETED"}), encoding="utf-8"
    )
    task = StageTask("abc", "demo", 1, "train", "slurm", bundle)
    runner = SlurmRunner()
    executor = SlurmExecutor(
        ExecutionTarget("slurm", "slurm", partition="cpu"), runner=runner
    )

    handle = executor.launch(task)

    assert handle.execution_id == "recovered-completed"
    assert runner.submissions == 0
    assert executor.inspect(handle).state == "completed"


def test_slurm_executor_cancels_the_exact_current_job(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    runner = SlurmRunner()
    executor = SlurmExecutor(
        ExecutionTarget("slurm", "slurm", partition="cpu"), runner=runner
    )
    handle = ExecutionHandle("abc", "slurm", "slurm", "123", str(bundle))

    status = executor.cancel(handle)

    assert status.state == "cancelled"
    assert runner.cancelled == ["123"]


def test_remote_slurm_cancel_runs_on_the_target_host(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    calls = []
    state = {"cancelled": False}

    def runner(args, **_kwargs):
        args = list(args)
        calls.append(args)
        if args[:6] == ["ssh", "remote", "bash", "-s", "--", "/remote/task"]:
            command = args[6:]
            if not command:
                return subprocess.CompletedProcess(args, 3, "", "")
            if command[0] == "squeue":
                output = "" if state["cancelled"] else "RUNNING\n"
                return subprocess.CompletedProcess(args, 0, output, "")
            if command[0] == "sacct":
                output = (
                    "CANCELLED|0:0|\n"
                    if state["cancelled"]
                    else "RUNNING|0:0|\n"
                )
                return subprocess.CompletedProcess(args, 0, output, "")
            if command[0] == "scancel":
                state["cancelled"] = True
                return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)

    executor = SlurmExecutor(
        ExecutionTarget(
            "remote",
            "slurm",
            host="remote",
            work_root="/remote/root",
            partition="cpu",
        ),
        runner=runner,
    )
    handle = ExecutionHandle(
        "abc",
        "remote",
        "slurm",
        "321",
        str(bundle),
        remote_bundle="/remote/task",
    )

    status = executor.cancel(handle)

    assert status.state == "cancelled"
    assert any(call[-2:] == ["scancel", "321"] for call in calls)


class CancellableExecutor:
    def __init__(self, target, cancellations):
        self.target = target
        self.cancellations = cancellations

    def cancel(self, handle):
        self.cancellations.append((handle.target, handle.execution_id))
        return ExecutionStatus("cancelled", "test cancellation accepted")


def test_stop_workflow_cancels_and_archives_current_execution(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    controller = PersistentController(preparation.output_dir)
    controller.state["state"] = "stopped"
    controller.state["current"] = {
        "task_id": "task-1",
        "generation": 1,
        "stage": "train",
        "resource": "training",
        "target": "gpu",
        "attempt": 1,
        "bundle": str(preparation.output_dir / ".neptrain/tasks/task-1"),
        "handle": {
            "task_id": "task-1",
            "target": "gpu",
            "executor": "slurm",
            "execution_id": "123",
            "local_bundle": str(
                preparation.output_dir / ".neptrain/tasks/task-1"
            ),
        },
    }
    controller._save()
    cancellations = []

    result = stop_workflow(
        preparation.output_dir,
        cancel_jobs=True,
        executor_factory=lambda target: CancellableExecutor(
            target, cancellations
        ),
    )

    assert result["controller"] == "already_stopped"
    assert result["current_execution"]["action"] == "cancelled"
    assert cancellations == [("gpu", "123")]
    state = json.loads(
        (preparation.output_dir / ".neptrain/controller.json").read_text()
    )
    assert state["current"] is None
    assert state["history"][-1]["cancellation"]["execution_id"] == "123"
    assert state["state"] == "stopped"
    status = workflow_status(preparation.output_dir)
    assert status.state == "paused"
    assert status.jobs[-1]["state"] == "CANCELLED"


def test_stop_workflow_preserves_current_until_cancellation_is_terminal(
    tmp_path,
):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    controller = PersistentController(preparation.output_dir)
    controller.state["state"] = "stopped"
    controller.state["current"] = {
        "task_id": "task-1",
        "generation": 1,
        "stage": "train",
        "resource": "training",
        "target": "gpu",
        "attempt": 1,
        "bundle": str(preparation.output_dir / ".neptrain/tasks/task-1"),
        "handle": {
            "task_id": "task-1",
            "target": "gpu",
            "executor": "slurm",
            "execution_id": "123",
            "local_bundle": str(
                preparation.output_dir / ".neptrain/tasks/task-1"
            ),
        },
    }
    controller._save()

    class PendingCancellation:
        def cancel(self, _handle):
            return ExecutionStatus(
                "cancelling", "scheduler accepted cancellation"
            )

    result = stop_workflow(
        preparation.output_dir,
        cancel_jobs=True,
        executor_factory=lambda _target: PendingCancellation(),
    )

    assert result["current_execution"]["action"] == "cancelling"
    state = json.loads(
        (preparation.output_dir / ".neptrain/controller.json").read_text()
    )
    assert state["current"]["handle"]["execution_id"] == "123"
    assert state["current"]["cancellation_requested_at"]
    assert state["history"] == []


class FailingExecutor:
    def __init__(self, target, task_ids):
        self.target = target
        self.task_ids = task_ids

    def launch(self, task):
        self.task_ids.append(task.task_id)
        return ExecutionHandle(
            task.task_id,
            task.target,
            "process",
            f"failed-{len(self.task_ids)}",
            str(task.bundle),
        )

    def inspect(self, _handle):
        return ExecutionStatus("failed", "intentional failure")

    def collect(self, _handle):
        raise AssertionError("failed executions cannot be collected")


def test_controller_retry_creates_a_new_traceable_attempt(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    task_ids = []
    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: FailingExecutor(target, task_ids),
    )

    assert controller.tick().state == "running"
    assert controller.tick().state == "failed"
    controller.retry()
    assert controller.tick().state == "running"

    assert len(task_ids) == 2
    assert task_ids[0] != task_ids[1]
    state = json.loads(
        (preparation.output_dir / ".neptrain/controller.json").read_text()
    )
    assert state["current"]["attempt"] == 2
    assert state["history"][0]["failure"] == "intentional failure"


def test_detached_controller_completes_a_real_multi_process_workflow(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    worker = _write(
        tmp_path / "portable_dummy_worker.py",
        """
import hashlib
import json
import os
import pathlib
import sys

bundle = pathlib.Path(sys.argv[-1])
task = json.loads((bundle / 'task.json').read_text())
stage = task['identity']['stage']
names = {
    'train': ('training_input', 'model'),
    'explore': ('candidates',),
    'select': ('selected_input', 'selection_result'),
    'label': ('labeled',),
    'diagnose': ('acquisition_signals',),
    'merge': ('training_set',),
    'retrain': ('retrained_model',),
    'evaluate': ('signals',),
}[stage]
artifacts = {}
for name in names:
    path = bundle / 'result' / 'artifacts' / name / f'{name}.dat'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name + '\\n')
    artifacts[name] = {
        'path': str(path.relative_to(bundle)),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'size': path.stat().st_size,
    }
result = {
    'protocol': 'neptrain.stage-result.v1',
    'task_id': task['task_id'],
    'workflow_id': task['identity']['workflow_id'],
    'generation': task['identity']['generation'],
    'stage': stage,
    'plan_sha256': task['identity']['plan_sha256'],
    'artifacts': artifacts,
    'metrics': {
        'accepted': True, 'workflow_converged': True
    } if stage == 'evaluate' else {},
}
(bundle / 'result.json').write_text(json.dumps(result))
(bundle / 'execution.json').write_text(json.dumps({
    'task_id': task['task_id'], 'state': 'COMPLETED', 'pid': os.getpid()
}))
""",
    )
    command = f"{sys.executable} {worker}"
    config.write_text(
        config.read_text().replace(
            "{executor: process}",
            "{executor: process, command: '" + command + "'}",
        ),
        encoding="utf-8",
    )
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")

    pid = start_controller(preparation.output_dir, poll_interval=0.2)
    assert pid > 0
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        status = workflow_status(preparation.output_dir)
        if status.state == "complete":
            break
        if status.state in {"failed", "rejected"}:
            raise AssertionError(status.reason)
        time.sleep(0.1)
    else:
        raise AssertionError("detached controller did not complete")

    assert not controller_running(preparation.output_dir)
    assert (preparation.output_dir / "results/nep.txt").is_file()
    state = json.loads(
        (preparation.output_dir / ".neptrain/controller.json").read_text()
    )
    assert len(state["history"]) == 8


def test_stopped_controller_resumes_an_inflight_process(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    worker = _write(
        tmp_path / "slow_dummy_worker.py",
        """
import hashlib
import json
import os
import pathlib
import sys
import time

bundle = pathlib.Path(sys.argv[-1])
task = json.loads((bundle / 'task.json').read_text())
stage = task['identity']['stage']
time.sleep(0.35)
names = {
    'train': ('training_input', 'model'),
    'explore': ('candidates',),
    'select': ('selected_input', 'selection_result'),
    'label': ('labeled',),
    'diagnose': ('acquisition_signals',),
    'merge': ('training_set',),
    'retrain': ('retrained_model',),
    'evaluate': ('signals',),
}[stage]
artifacts = {}
for name in names:
    path = bundle / 'result' / 'artifacts' / name / f'{name}.dat'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name + '\\n')
    artifacts[name] = {
        'path': str(path.relative_to(bundle)),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'size': path.stat().st_size,
    }
(bundle / 'result.json').write_text(json.dumps({
    'protocol': 'neptrain.stage-result.v1',
    'task_id': task['task_id'],
    'workflow_id': task['identity']['workflow_id'],
    'generation': task['identity']['generation'],
    'stage': stage,
    'plan_sha256': task['identity']['plan_sha256'],
    'artifacts': artifacts,
    'metrics': {
        'accepted': True, 'workflow_converged': True
    } if stage == 'evaluate' else {},
}))
(bundle / 'execution.json').write_text(json.dumps({
    'task_id': task['task_id'], 'state': 'COMPLETED', 'pid': os.getpid()
}))
""",
    )
    command = f"{sys.executable} {worker}"
    config.write_text(
        config.read_text().replace(
            "{executor: process}",
            "{executor: process, command: '" + command + "'}",
        ),
        encoding="utf-8",
    )
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")

    # A long scheduler poll must not make an explicit stop wait for that poll.
    start_controller(preparation.output_dir, poll_interval=30)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state_path = preparation.output_dir / ".neptrain/controller.json"
        if not state_path.is_file():
            time.sleep(0.02)
            continue
        state = json.loads(state_path.read_text())
        if (state.get("current") or {}).get("handle"):
            break
        time.sleep(0.02)
    else:
        raise AssertionError("controller did not launch a stage")
    stop_controller(preparation.output_dir)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and controller_running(preparation.output_dir):
        time.sleep(0.02)
    assert not controller_running(preparation.output_dir)

    start_controller(preparation.output_dir, poll_interval=0.2)
    restarted = json.loads(
        (preparation.output_dir / ".neptrain/controller.json").read_text()
    )
    assert "reason" not in restarted
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = workflow_status(preparation.output_dir)
        if status.state == "complete":
            break
        if status.state in {"failed", "rejected"}:
            raise AssertionError(status.reason)
        time.sleep(0.1)
    else:
        raise AssertionError("resumed controller did not complete")

    state = json.loads(
        (preparation.output_dir / ".neptrain/controller.json").read_text()
    )
    train_history = [item for item in state["history"] if item["stage"] == "train"]
    assert len(train_history) == 1
    assert train_history[0]["attempt"] == 1
