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
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

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
        max_selected=2,
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
    descriptor = json.loads(task.descriptor.read_text())
    assert descriptor["protocol"] == "neptrain.stage-task.v2"
    assert task.bundle.name.startswith("g0001-md-a1-")
    assert (task.bundle / "input").is_dir()
    assert not (task.bundle / "inputs").exists()
    assert not (task.bundle / "manifest.json").exists()

    class FakeWorkflow:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_stage(self, stage, context):
            assert stage == "explore"
            assert context.flat_output is True
            assert context.artifacts["model"].read_text() == "fixture\n"
            output = _write(context.work_dir / "candidates.xyz", "candidate\n")
            return StageOutcome({"candidates": output}, {"candidate_count": 1})

    monkeypatch.setattr(
        "NepTrain.core.workflow_iteration.WorkflowIterationAdapter", FakeWorkflow
    )

    assert run_stage_worker(task.bundle) == 0
    value, outcome = load_stage_result(task.bundle)
    assert value["task_id"] == task.task_id
    assert value["protocol"] == "neptrain.stage-result.v2"
    assert outcome.metrics == {"candidate_count": 1}
    assert outcome.artifacts["candidates"].read_text() == "candidate\n"
    assert outcome.artifacts["candidates"] == task.bundle / "output" / "candidates.xyz"
    assert not (task.bundle / "work").exists()
    assert not (task.bundle / "result").exists()


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
    assert "resource_path" not in descriptor["config"]["dft"]
    assert descriptor["initial_training"] is None
    assert "routes" not in descriptor["config"].get("sampling", {})
    assert "validation_path" not in descriptor["config"]["evaluation"]
    assert "initial_path" not in descriptor["config"]["training"]
    assert not (task.bundle / "input" / "initial").exists()
    assert not (task.bundle / "input" / "sampling").exists()
    assert not (task.bundle / "input" / "config" / "evaluation").exists()
    assert not any(path.name == "POTCAR" for path in task.bundle.rglob("*"))


def test_train_stage_bundle_skips_sampling_route_inputs(tmp_path):
    initial = _write(tmp_path / "initial.xyz")
    training_config = _write(tmp_path / "nep.in")
    template = _write(tmp_path / "lammps.in", "run {{ steps }}\n")
    structure = _write(tmp_path / "structure.xyz")
    task = build_stage_task(
        tmp_path / "tasks",
        workflow_root=tmp_path,
        workflow_id="portable-routes",
        generation=1,
        stage="train",
        attempt=1,
        target=ExecutionTarget("local", "process"),
        plan=_plan(),
        config={
            "training": {
                "initial_path": str(initial),
                "config_path": str(training_config),
            },
            "sampling": {
                "routes": [
                    {
                        "id": "default",
                        "structures": [str(structure)],
                        "template_path": str(template),
                        "conditions": {
                            "temperature_path": [50],
                            "pressure": 0,
                        },
                        "progression": {
                            "steps": {"smoke_passed": 20},
                            "replicas": {"smoke_passed": 1},
                        },
                    }
                ]
            },
            "md": {"spin": False},
        },
        initial_training=initial,
        context=StageContext(
            generation=1,
            generation_dir=tmp_path / "generation",
            plan=_plan(),
            artifacts={},
            previous_artifacts={},
        ),
    )

    descriptor = json.loads(task.descriptor.read_text())
    assert "routes" not in descriptor["config"]["sampling"]
    assert (task.bundle / descriptor["initial_training"]).read_text() == "fixture\n"
    assert not (task.bundle / "input" / "sampling").exists()


def test_remote_stage_bundle_carries_only_activated_model_across_generations(
    tmp_path,
):
    initial = _write(tmp_path / "initial.xyz")
    validation = _write(tmp_path / "validation.xyz")
    training_config = _write(tmp_path / "nep.in")
    current = {
        name: _write(tmp_path / "current" / f"{name}.dat")
        for name in (
            "training_input",
            "training_set",
            "model",
            "checkpoint",
            "labeled",
            "retrained_model",
            "retrained_checkpoint",
            "retraining_decision",
            "model_lineage",
            "scenario_plan",
            "acquisition_signals",
            "selection_result",
            "md_attempts",
        )
    }
    config = {
        "training": {
            "initial_path": str(initial),
            "config_path": str(training_config),
        },
        "md": {"spin": False},
        "evaluation": {
            "validation_path": str(validation),
            "max_rmse": {"energy_rmse": 1, "force_rmse": 1},
        },
    }
    evaluate_task = build_stage_task(
        tmp_path / "evaluate-tasks",
        workflow_root=tmp_path,
        workflow_id="activation",
        generation=1,
        stage="evaluate",
        attempt=1,
        target=ExecutionTarget("local", "process"),
        plan=_plan(),
        config=config,
        initial_training=initial,
        context=StageContext(
            generation=1,
            generation_dir=tmp_path / "generation-1",
            plan=_plan(),
            artifacts=current,
            previous_artifacts={},
        ),
    )
    evaluate_descriptor = json.loads(
        evaluate_task.descriptor.read_text()
    )
    assert set(evaluate_descriptor["artifacts"]) == set(current)

    previous = {
        "training_set": current["training_set"],
        "activated_model": _write(
            tmp_path / "previous" / "activated-model.dat"
        ),
        "activated_checkpoint": _write(
            tmp_path / "previous" / "activated-checkpoint.dat"
        ),
        "active_model_lineage": _write(
            tmp_path / "previous" / "active-model-lineage.json",
            "{}\n",
        ),
        "retrained_model": current["retrained_model"],
        "model_lineage": current["model_lineage"],
    }
    train_task = build_stage_task(
        tmp_path / "train-tasks",
        workflow_root=tmp_path,
        workflow_id="activation",
        generation=2,
        stage="train",
        attempt=1,
        target=ExecutionTarget("local", "process"),
        plan=GenerationPlan(
            generation=2,
            seed=8,
            max_selected=2,
        ),
        config=config,
        initial_training=initial,
        context=StageContext(
            generation=2,
            generation_dir=tmp_path / "generation-2",
            plan=GenerationPlan(
                generation=2,
                seed=8,
                max_selected=2,
            ),
            artifacts={},
            previous_artifacts=previous,
        ),
    )
    train_descriptor = json.loads(train_task.descriptor.read_text())
    assert train_descriptor["initial_training"] is None
    assert "config_path" not in train_descriptor["config"]["training"]
    assert not (train_task.bundle / "input" / "initial").exists()
    assert set(train_descriptor["previous_artifacts"]) == {
        "training_set",
        "activated_model",
        "activated_checkpoint",
        "active_model_lineage",
    }


def test_stage_result_rejects_paths_outside_the_bundle(tmp_path):
    outside = _write(tmp_path / "outside.dat")
    bundle = tmp_path / "task"
    bundle.mkdir()
    (bundle / "result.json").write_text(
        json.dumps(
            {
                "protocol": "neptrain.stage-result.v2",
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
    "evaluate": ("signals", "activated_model"),
}


class ImmediateExecutor:
    def __init__(self, target, launches):
        self.target = target
        self.launches = launches

    def launch(self, task):
        self.launches.append((task.stage, task.target))
        result_root = task.bundle / "result" / "artifacts"
        artifacts = {}
        descriptor = json.loads(task.descriptor.read_text(encoding="utf-8"))
        names = _STAGE_ARTIFACTS[task.stage]
        if task.stage == "explore":
            names = ("candidates", "md_attempts", "scenario_plan")
        for name in names:
            filename = (
                f"{name}.xyz"
                if name in {"candidates", "selected_input", "labeled"}
                else f"{name}.json"
            )
            path = (
                task.bundle / "output" / filename
                if task.stage == "label"
                else result_root / name / filename
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "candidates":
                route = descriptor["identity"]["sampling_routes"][0]
                frame = Atoms("Fe", positions=[[0.01, 0.0, 0.0]])
                frame.info.update(
                    route_id=route["route_id"],
                    route_fingerprint=route["route_fingerprint"],
                )
                ase_write(path, frame, format="extxyz")
            elif name == "selected_input":
                ase_write(
                    path,
                    [
                        Atoms("Fe", positions=[[0.01 * index, 0.0, 0.0]])
                        for index in range(1, 4)
                    ],
                    format="extxyz",
                )
            elif name == "labeled":
                source = task.bundle / descriptor["artifacts"]["selected_input"]
                ase_write(
                    path,
                    ase_read(source, index=":"),
                    format="extxyz",
                )
            elif name == "md_attempts":
                model_path = task.bundle / descriptor["artifacts"]["model"]
                path.write_text(
                    json.dumps(
                        {
                            "version": 2,
                            "model_sha256": _sha256(model_path),
                            "routes": [],
                            "attempts": [{"completed": True}],
                        }
                    ),
                    encoding="utf-8",
                )
            elif name == "scenario_plan":
                model_path = task.bundle / descriptor["artifacts"]["model"]
                path.write_text(
                    json.dumps(
                        {
                            "version": 3,
                            "model_id": _sha256(model_path),
                            "routes": [],
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                _write(path, name + "\n")
            artifacts[name] = {
                "path": str(path.relative_to(task.bundle)),
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
        if task.stage == "evaluate":
            metrics = {"accepted": True, "workflow_converged": True}
        elif task.stage == "label":
            metrics = {
                "backend": descriptor["config"]["dft"]["backend"],
                "labeled_count": len(
                    ase_read(
                        task.bundle / descriptor["artifacts"]["selected_input"],
                        index=":",
                    )
                ),
            }
        else:
            metrics = {}
        (task.bundle / "result.json").write_text(
            json.dumps(
                {
                    "protocol": "neptrain.stage-result.v2",
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
    ase_write(
        tmp_path / "structures.xyz",
        Atoms("Fe", positions=[[0.0, 0.0, 0.0]]),
        format="extxyz",
    )
    _write(tmp_path / "lammps.in", "run {{ steps }}\n")
    _write(tmp_path / "validation.xyz")
    config = _write(
        tmp_path / "project.yaml",
        """
schema_version: 7
training:
  backend: gpumd
  initial_path: ./initial.xyz
  config_path: ./nep.in
md:
  backend: lammps
  spin: false
sampling:
  routes:
    - id: default
      structures: [./structures.xyz]
      template_path: ./lammps.in
      conditions:
        temperature_path: [300]
        production_temperatures: [300]
        pressure: 0
      progression:
        steps:
          smoke_passed: 2
          short_stable: 8
          long_stable: 32
          production_ready: 128
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
    max_selected: 2
    novelty: auto
dft:
  backend: toy
evaluation:
  validation_path: ./validation.xyz
  max_rmse:
    energy_rmse: 1
    force_rmse: 1
workflow:
  id: controller-test
  max_model_generations: 1
execution:
  poll_interval: 0.2
  stage_targets:
    training: gpu
    sampling: md
    labeling: dft
    analysis: cpu
  sampling_route_targets: {}
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
    assert manifest["orchestration"] == "controller"
    assert "scripts" not in manifest


def test_controller_publishes_flat_outputs_and_real_calculation_link(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    controller = PersistentController(preparation.output_dir)
    plan = controller.plans[0]
    bundle = preparation.output_dir / ".neptrain" / "jobs" / "training-task"
    calculation = bundle / "output"
    model = _write(calculation / "nep.txt", "model\n")
    loss = _write(calculation / "loss.out", "loss\n")

    link = controller._publish_calculation_link(
        generation=1,
        stage="train",
        bundle=bundle,
    )
    summary = controller._install_outcome(
        plan=plan,
        stage="train",
        attempt=1,
        outcome=StageOutcome(
            artifacts={"model": model, "training_output_loss_out": loss}
        ),
    )

    training = preparation.output_dir / "generations" / "0001" / "train"
    assert link == training / "calculation"
    assert link.is_symlink()
    assert (link / "loss.out").read_text(encoding="utf-8") == "loss\n"
    assert (training / "nep.txt").read_text() == "model\n"
    assert (training / "loss.out").read_text() == "loss\n"
    assert summary.artifacts["model"] == training / "nep.txt"


def test_controller_publishes_flat_md_calculation_link(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    controller = PersistentController(preparation.output_dir)
    bundle = (
        preparation.output_dir
        / ".neptrain"
        / "jobs"
        / "g0001-md-default-a1-demo"
    )
    output = bundle / "output"
    _write(output / "log.lammps", "done\n")
    _write(
        output / "md-attempts.json",
        json.dumps(
            {
                "attempts": [
                    {
                        "source_id": "g1-default-sdemo-T50-P0-r1-attempt",
                    }
                ]
            }
        ),
    )

    link = controller._publish_calculation_link(
        generation=1,
        stage="explore",
        bundle=bundle,
        grouped=True,
    )

    expected = (
        preparation.output_dir
        / "generations"
        / "0001"
        / "md"
        / "calculations"
        / "g1-default-sdemo-T50-P0-r1-attempt"
    )
    assert link == expected
    assert link.is_symlink()
    assert (link / "log.lammps").read_text(encoding="utf-8") == "done\n"
    assert not (output / "md").exists()


def test_controller_submits_every_unlocked_route_attempt_as_one_md_wave(
    tmp_path,
):
    config, initial = _controller_inputs(tmp_path)
    text = config.read_text(encoding="utf-8")
    second_route = """
    - id: second
      structures: [./structures.xyz]
      template_path: ./lammps.in
      conditions:
        temperature_path: [500]
        production_temperatures: [500]
        pressure: 0
      progression:
        steps:
          smoke_passed: 2
          short_stable: 8
          long_stable: 32
          production_ready: 128
        replicas:
          smoke_passed: 1
          short_stable: 1
          long_stable: 2
          production_ready: 3
"""
    text = text.replace(
        "  candidate_pool:\n", second_route + "  candidate_pool:\n"
    )
    text = text.replace(
        "  sampling_route_targets: {}\n",
        "  sampling_route_targets:\n    second: md2\n",
    )
    text = text.replace(
        "    dft: {executor: process}\n",
        "    dft: {executor: process}\n    md2: {executor: process}\n",
    )
    config.write_text(text, encoding="utf-8")
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []
    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: ImmediateExecutor(target, launches),
    )

    controller.tick()
    controller.tick()

    current = controller.state["current"]
    assert current["kind"] == "task_group"
    assert len(current["tasks"]) == 2
    assert {item["target"] for item in current["tasks"]} == {"md", "md2"}
    assert {item["route_id"] for item in current["tasks"]} == {
        "default",
        "second",
    }
    assert all(item["handle"] is not None for item in current["tasks"])
    assert [stage for stage, _ in launches].count("explore") == 2


def test_controller_splits_dft_labels_and_limits_concurrency(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    _write(tmp_path / "INCAR", "IBRION = -1\nNSW = 0\nISPIN = 2\n")
    (tmp_path / "potpaw").mkdir()
    text = config.read_text(encoding="utf-8").replace(
        "dft:\n  backend: toy\n",
        (
            "dft:\n"
            "  backend: vasp\n"
            "  input_path: ./INCAR\n"
            "  resource_path: ./potpaw\n"
            "  structures_per_job: 1\n"
            "  max_concurrent: 2\n"
        ),
    )
    config.write_text(text, encoding="utf-8")
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []
    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: ImmediateExecutor(target, launches),
    )

    for _ in range(20):
        controller.tick()
        current = controller.state.get("current") or {}
        if current.get("stage") == "label":
            break
    else:
        raise AssertionError("controller did not reach grouped DFT labeling")

    assert current["kind"] == "task_group"
    assert current["max_concurrent"] == 2
    assert len(current["tasks"]) == 3
    assert sum(item["handle"] is not None for item in current["tasks"]) == 2
    assert [stage for stage, _ in launches].count("label") == 2
    for item in current["tasks"]:
        descriptor = json.loads(
            (Path(item["bundle"]) / "task.json").read_text(encoding="utf-8")
        )
        selected = (
            Path(item["bundle"])
            / descriptor["artifacts"]["selected_input"]
        )
        assert len(ase_read(selected, index=":")) == 1

    controller.tick()
    assert [stage for stage, _ in launches].count("label") == 2
    controller.tick()
    assert [stage for stage, _ in launches].count("label") == 3
    controller.tick()
    controller.tick()

    ledger = json.loads(
        (preparation.output_dir / ".neptrain/ledger.json").read_text()
    )
    label = ledger["generations"]["1"]["stages"]["label"]
    assert label["metrics"]["labeled_count"] == 3
    assert label["metrics"]["batch_count"] == 3
    dft_root = preparation.output_dir / "generations" / "0001" / "dft"
    assert len(list(dft_root.glob("00000*-Fe"))) == 3
    assert not (dft_root / "calculations").exists()


def test_md_wave_retry_preserves_completed_attempts(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    text = config.read_text(encoding="utf-8")
    second_route = """
    - id: second
      structures: [./structures.xyz]
      template_path: ./lammps.in
      conditions:
        temperature_path: [500]
        production_temperatures: [500]
        pressure: 0
      progression:
        steps:
          smoke_passed: 2
          short_stable: 8
          long_stable: 32
          production_ready: 128
        replicas:
          smoke_passed: 1
          short_stable: 1
          long_stable: 2
          production_ready: 3
"""
    text = text.replace(
        "  candidate_pool:\n", second_route + "  candidate_pool:\n"
    )
    text = text.replace(
        "    dft: {executor: process}\n",
        "    dft: {executor: process}\n    md2: {executor: process}\n",
    )
    text = text.replace(
        "  sampling_route_targets: {}\n",
        "  sampling_route_targets:\n    second: md2\n",
    )
    config.write_text(text, encoding="utf-8")
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []
    failed_once = {"md2": False}

    class FailOnceExecutor(ImmediateExecutor):
        def inspect(self, handle):
            if self.target.name == "md2" and not failed_once["md2"]:
                failed_once["md2"] = True
                return ExecutionStatus("failed", "node failure")
            return ExecutionStatus("completed")

    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: FailOnceExecutor(target, launches),
    )
    controller.tick()
    controller.tick()
    assert controller.tick().state == "failed"
    current = controller.state["current"]
    failed_task = next(
        item for item in current["tasks"] if item["target"] == "md2"
    )
    completed_task = next(
        item for item in current["tasks"] if item["target"] == "md"
    )
    assert failed_task["terminal_failure"] is True
    assert completed_task["collected_bundle"]

    controller.retry()
    controller.tick()
    controller.tick()

    assert [
        target
        for stage, target in launches
        if stage == "explore"
    ] == ["md", "md2", "md2"]
    ledger = json.loads(
        (preparation.output_dir / ".neptrain/ledger.json").read_text()
    )
    assert "explore" in ledger["generations"]["1"]["stages"]


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
    execution_ids = {
        task["handle"]["execution_id"]
        for item in state["history"]
        for task in (
            item["tasks"]
            if item.get("kind") == "task_group"
            else [item]
        )
    }
    assert execution_ids == {"12345"}
    assert [
        (
            item["tasks"][0]["target"]
            if item.get("kind") == "task_group"
            else item["target"]
        )
        for item in state["history"]
    ] == [
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
    (
        preparation.output_dir
        / "inputs/sampling/routes/default/structures/0.xyz"
    ).write_text(
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
    script = (bundle / "submit.sh").read_text()
    assert "#SBATCH --partition=cpu" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert f"#SBATCH --output={bundle}/stdout-%j.log" in script
    assert "--dependency" not in script
    assert executor.inspect(first).state == "running"
    runner.state = "COMPLETED"
    assert executor.inspect(first).state == "completed"


def test_slurm_executor_does_not_claim_historical_same_name_job(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    task = StageTask("abc", "demo", 1, "train", "slurm", bundle)

    class HistoricalRunner(SlurmRunner):
        def __call__(self, args, **kwargs):
            args = list(args)
            if args[0] == "squeue" and "--name" in args:
                return subprocess.CompletedProcess(args, 0, "", "")
            if args[0] == "sacct" and "--name" in args:
                return subprocess.CompletedProcess(
                    args, 0, "111|nt-abc\n", ""
                )
            return super().__call__(args, **kwargs)

    runner = HistoricalRunner()
    executor = SlurmExecutor(
        ExecutionTarget("slurm", "slurm", partition="cpu"),
        runner=runner,
    )

    handle = executor.launch(task)

    assert handle.execution_id == "123"
    assert runner.submissions == 1


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


def test_stop_workflow_cancels_and_archives_current_execution_by_default(tmp_path):
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
        "bundle": str(preparation.output_dir / ".neptrain/jobs/task-1"),
        "handle": {
            "task_id": "task-1",
            "target": "gpu",
            "executor": "slurm",
            "execution_id": "123",
            "local_bundle": str(
                preparation.output_dir / ".neptrain/jobs/task-1"
            ),
        },
    }
    controller._save()
    cancellations = []

    result = stop_workflow(
        preparation.output_dir,
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
        "bundle": str(preparation.output_dir / ".neptrain/jobs/task-1"),
        "handle": {
            "task_id": "task-1",
            "target": "gpu",
            "executor": "slurm",
            "execution_id": "123",
            "local_bundle": str(
                preparation.output_dir / ".neptrain/jobs/task-1"
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
    'explore': ('candidates', 'md_attempts', 'scenario_plan'),
    'select': ('selected_input', 'selection_result'),
    'label': ('labeled',),
    'diagnose': ('acquisition_signals',),
    'merge': ('training_set',),
    'retrain': ('retrained_model',),
    'evaluate': ('signals', 'activated_model'),
}[stage]
artifacts = {}
for name in names:
    suffix = '.xyz' if name == 'candidates' else '.dat'
    path = bundle / 'result' / 'artifacts' / name / f'{name}{suffix}'
    path.parent.mkdir(parents=True, exist_ok=True)
    if stage == 'explore' and name == 'candidates':
        route = task['identity']['sampling_routes'][0]
        path.write_text(
            '1\\n'
            f"route_id={route['route_id']} "
            f"route_fingerprint={route['route_fingerprint']}\\n"
            'Fe 0.0 0.0 0.0\\n'
        )
    elif stage == 'explore' and name in {'md_attempts', 'scenario_plan'}:
        model_path = bundle / task['artifacts']['model']
        model_id = hashlib.sha256(model_path.read_bytes()).hexdigest()
        value = (
            {'version': 2, 'model_sha256': model_id, 'routes': [],
             'attempts': [{'completed': True}]}
            if name == 'md_attempts'
            else {'version': 3, 'model_id': model_id, 'routes': []}
        )
        path.write_text(json.dumps(value))
    else:
        path.write_text(name + '\\n')
    artifacts[name] = {
        'path': str(path.relative_to(bundle)),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'size': path.stat().st_size,
    }
result = {
    'protocol': 'neptrain.stage-result.v2',
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
    'explore': ('candidates', 'md_attempts', 'scenario_plan'),
    'select': ('selected_input', 'selection_result'),
    'label': ('labeled',),
    'diagnose': ('acquisition_signals',),
    'merge': ('training_set',),
    'retrain': ('retrained_model',),
    'evaluate': ('signals', 'activated_model'),
}[stage]
artifacts = {}
for name in names:
    suffix = '.xyz' if name == 'candidates' else '.dat'
    path = bundle / 'result' / 'artifacts' / name / f'{name}{suffix}'
    path.parent.mkdir(parents=True, exist_ok=True)
    if stage == 'explore' and name == 'candidates':
        route = task['identity']['sampling_routes'][0]
        path.write_text(
            '1\\n'
            f"route_id={route['route_id']} "
            f"route_fingerprint={route['route_fingerprint']}\\n"
            'Fe 0.0 0.0 0.0\\n'
        )
    elif stage == 'explore' and name in {'md_attempts', 'scenario_plan'}:
        model_path = bundle / task['artifacts']['model']
        model_id = hashlib.sha256(model_path.read_bytes()).hexdigest()
        value = (
            {'version': 2, 'model_sha256': model_id, 'routes': [],
             'attempts': [{'completed': True}]}
            if name == 'md_attempts'
            else {'version': 3, 'model_id': model_id, 'routes': []}
        )
        path.write_text(json.dumps(value))
    else:
        path.write_text(name + '\\n')
    artifacts[name] = {
        'path': str(path.relative_to(bundle)),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'size': path.stat().st_size,
    }
(bundle / 'result.json').write_text(json.dumps({
    'protocol': 'neptrain.stage-result.v2',
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
