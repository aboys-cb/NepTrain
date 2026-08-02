from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read as ase_read
from ase.io import write as ase_write

from NepTrain.core.workflow import (
    WorkflowError,
    extend_workflow,
    prepare_workflow,
    resume_workflow,
    workflow_status,
)
from NepTrain.core.controller import (
    ControllerTick,
    ControllerError,
    PersistentController,
    controller_running,
    run_controller,
    start_controller,
    stop_controller,
    stop_workflow,
)
from NepTrain.core.execution import (
    ExecutionError,
    ExecutionHandle,
    ExecutionStatus,
    ExecutionTarget,
    ExecutionTransport,
    PermanentExecutionError,
    ProcessExecutor,
    SlurmExecutor,
    StageTask,
    SubmissionDeferred,
    build_stage_task,
    load_stage_result,
    run_stage_worker,
)
from NepTrain.core.iteration import GenerationPlan, StageContext, StageOutcome
from NepTrain.core.scientific_data import (
    INPUT_STRUCTURE_ID_KEY,
    structure_id,
)
from NepTrain.core.workflow_workspace import WorkflowWorkspace
import NepTrain.core.controller as controller_module
import NepTrain.core.execution as execution_module
import NepTrain.core.workflow_iteration as workflow_iteration_module


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
    atoms.info["virial"] = np.zeros((3, 3))
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-1.0,
        forces=np.zeros((1, 3)),
    )
    ase_write(path, atoms, format="extxyz")
    return path


def _write_vasp_resources(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "potpaw"
    potcar = _write(
        root / "Fe" / "POTCAR",
        "TITEL = PAW_PBE Fe 06Sep2000\nVRHFIN =Fe: s2d6\n",
    )
    manifest = tmp_path / "vasp-resources.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "neptrain.vasp-resources.v1",
                "family": "PAW_PBE",
                "release": "test",
                "elements": {
                    "Fe": {
                        "path": "Fe/POTCAR",
                        "sha256": hashlib.sha256(
                            potcar.read_bytes()
                        ).hexdigest(),
                        "titel": "PAW_PBE Fe 06Sep2000",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return root, manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> GenerationPlan:
    return GenerationPlan(
        generation=1,
        seed=7,
        max_selected=2,
    )


def test_empty_label_task_omits_dft_resources(tmp_path):
    initial = _write(tmp_path / "initial.xyz")
    selected = tmp_path / "selected-input.xyz"
    selected.touch()
    missing_incar = tmp_path / "missing-INCAR"
    missing_resources = tmp_path / "missing-potcar"
    plan = _plan()
    task = build_stage_task(
        tmp_path / "tasks",
        workflow_root=tmp_path,
        workflow_id="empty-label",
        generation=1,
        stage="label",
        attempt=1,
        target=ExecutionTarget("analysis", "process"),
        plan=plan,
        config={
            "training": {},
            "sampling": {},
            "md": {},
            "labeling": {
                "backend": "vasp",
                "input_path": str(missing_incar),
                "resource_path": str(missing_resources),
            },
            "workflow": {},
            "execution": {},
            "notifications": {
                "feishu": {
                    "webhook": "https://open.feishu.cn/open-apis/bot/v2/hook/test",
                    "secret": "must-not-leave-controller",
                }
            },
        },
        initial_training=initial,
        context=StageContext(
            generation=1,
            generation_dir=tmp_path / "generation",
            plan=plan,
            artifacts={"selected_input": selected},
            previous_artifacts={},
        ),
        stage_input={"empty_selection": True},
    )

    descriptor = json.loads(task.descriptor.read_text())
    assert descriptor["identity"]["target"] == "analysis"
    assert descriptor["stage_input"] == {"empty_selection": True}
    assert "input_path" not in descriptor["config"]["labeling"]
    assert "resource_path" not in descriptor["config"]["labeling"]
    assert "notifications" not in descriptor["config"]
    assert "must-not-leave-controller" not in task.descriptor.read_text()


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
            "labeling": {"software": "toy"},
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
    assert descriptor["protocol"] == "neptrain.stage-task.v3"
    assert descriptor["task_id"] == descriptor["spec_sha256"][:24]
    assert task.bundle.name.startswith("g0001-md-a1-")
    assert task.bundle.stat().st_mode & 0o070 == 0o070
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
    assert value["protocol"] == "neptrain.stage-result.v3"
    assert outcome.metrics == {"candidate_count": 1}
    assert outcome.artifacts["candidates"].read_text() == "candidate\n"
    assert outcome.artifacts["candidates"] == task.bundle / "output" / "candidates.xyz"
    assert not (task.bundle / "work").exists()
    assert not (task.bundle / "result").exists()


def test_stage_task_identity_covers_bundle_content_and_workflow_instance(tmp_path):
    initial = _write(tmp_path / "initial.xyz")
    model = _write(tmp_path / "model.txt", "model-one\n")
    config = {
        "training": {},
        "evaluation": {},
        "md": {"spin": False},
        "labeling": {},
        "workflow": {},
        "execution": {},
    }
    context = StageContext(
        generation=1,
        generation_dir=tmp_path / "generation",
        plan=_plan(),
        artifacts={"model": model},
        previous_artifacts={},
    )

    first = build_stage_task(
        tmp_path / "tasks",
        workflow_root=tmp_path,
        workflow_id="content-addressed",
        workflow_instance_id="instance-one",
        generation=1,
        stage="explore",
        attempt=1,
        target=ExecutionTarget("local", "process"),
        plan=_plan(),
        config=config,
        initial_training=initial,
        context=context,
    )
    model.write_text("model-two\n", encoding="utf-8")
    second = build_stage_task(
        tmp_path / "tasks",
        workflow_root=tmp_path,
        workflow_id="content-addressed",
        workflow_instance_id="instance-one",
        generation=1,
        stage="explore",
        attempt=1,
        target=ExecutionTarget("local", "process"),
        plan=_plan(),
        config=config,
        initial_training=initial,
        context=context,
    )
    third = build_stage_task(
        tmp_path / "tasks",
        workflow_root=tmp_path,
        workflow_id="content-addressed",
        workflow_instance_id="instance-two",
        generation=1,
        stage="explore",
        attempt=1,
        target=ExecutionTarget("local", "process"),
        plan=_plan(),
        config=config,
        initial_training=initial,
        context=context,
    )

    assert len({first.task_id, second.task_id, third.task_id}) == 3
    first_descriptor = json.loads(first.descriptor.read_text())
    assert first_descriptor["artifacts"]["model"] == "input/nep.txt"
    assert (
        first.bundle / first_descriptor["artifacts"]["model"]
    ).read_text(encoding="utf-8") == "model-one\n"


def test_label_task_identity_includes_the_resource_manifest_content(tmp_path):
    initial = _write(tmp_path / "initial.xyz")
    selected = _write(tmp_path / "selected.xyz")
    manifest = _write(tmp_path / "vasp-resources.json", '{"release":"one"}\n')
    context = StageContext(
        generation=1,
        generation_dir=tmp_path / "generation",
        plan=_plan(),
        artifacts={"selected_input": selected},
        previous_artifacts={},
    )
    config = {
        "training": {},
        "evaluation": {},
        "md": {"spin": False},
        "labeling": {
            "backend": "vasp",
            "resource_path": "/remote/potpaw",
            "potcar_manifest_path": str(manifest),
        },
        "workflow": {},
        "execution": {},
    }
    target = ExecutionTarget(
        "label",
        "slurm",
        host="fixture",
        work_root="/remote/work",
        partition="cpu",
        labeling_resource_path="/remote/potpaw",
    )
    first = build_stage_task(
        tmp_path / "tasks",
        workflow_root=tmp_path,
        workflow_id="resource-identity",
        workflow_instance_id="instance",
        generation=1,
        stage="label",
        attempt=1,
        target=target,
        plan=_plan(),
        config=config,
        initial_training=initial,
        context=context,
    )
    manifest.write_text('{"release":"two"}\n', encoding="utf-8")
    second = build_stage_task(
        tmp_path / "tasks",
        workflow_root=tmp_path,
        workflow_id="resource-identity",
        workflow_instance_id="instance",
        generation=1,
        stage="label",
        attempt=1,
        target=target,
        plan=_plan(),
        config=config,
        initial_training=initial,
        context=context,
    )

    assert first.task_id != second.task_id
    first_descriptor = json.loads(first.descriptor.read_text())
    bundled_manifest = (
        first.bundle
        / first_descriptor["config"]["labeling"]["potcar_manifest_path"]
    )
    assert bundled_manifest.read_text(encoding="utf-8") == '{"release":"one"}\n'


def test_remote_deploy_atomically_replaces_only_an_incomplete_exact_task(
    tmp_path,
):
    initial = _write(tmp_path / "initial.xyz")
    model = _write(tmp_path / "model.txt")
    remote_root = tmp_path / "remote"
    target = ExecutionTarget(
        "remote",
        "slurm",
        host="fixture",
        work_root="~/remote",
        partition="cpu",
        command=(
            f"env PYTHONPATH={Path(__file__).resolve().parents[1] / 'src'} "
            f"{sys.executable} -m NepTrain.cli.cli"
        ),
    )
    task = build_stage_task(
        tmp_path / "tasks",
        workflow_root=tmp_path,
        workflow_id="atomic-deploy",
        workflow_instance_id="instance",
        generation=1,
        stage="explore",
        attempt=1,
        target=target,
        plan=_plan(),
        config={
            "training": {},
            "evaluation": {},
            "md": {"spin": False},
            "labeling": {},
            "workflow": {},
            "execution": {},
        },
        initial_training=initial,
        context=StageContext(
            generation=1,
            generation_dir=tmp_path / "generation",
            plan=_plan(),
            artifacts={"model": model},
            previous_artifacts={},
        ),
    )

    class LocalRemoteTransport(ExecutionTransport):
        def run_script(self, script, *arguments, check=False, timeout=60):
            completed = subprocess.run(
                ["bash", "-s", "--", *(str(value) for value in arguments)],
                input=script,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                env={**os.environ, "HOME": str(tmp_path)},
            )
            if check and completed.returncode:
                raise ExecutionError(completed.stderr)
            return completed

        def copy(self, source, destination, **_kwargs):
            remote_path = Path(str(destination).split(":", 1)[1])
            remote_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, remote_path)
            return subprocess.CompletedProcess([], 0, "", "")

    destination = (
        remote_root / task.workflow_id / "jobs" / task.bundle.name
    )
    destination.mkdir(parents=True)
    (destination / "partial-upload").write_text("incomplete\n")
    transport = LocalRemoteTransport(target)

    assert transport.deploy(task) == str(destination)
    assert (destination / "task.json").read_bytes() == task.descriptor.read_bytes()
    quarantined = list(destination.parent.glob(destination.name + ".incomplete.*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "partial-upload").is_file()

    # A complete identical bundle is reused without another upload.
    assert transport.deploy(task) == str(destination)

    descriptor = json.loads(task.descriptor.read_text(encoding="utf-8"))
    missing = descriptor["files"][0]
    (destination / missing["path"]).unlink()
    assert transport.deploy(task) == str(destination)
    assert _sha256(destination / missing["path"]) == missing["sha256"]
    quarantined = list(destination.parent.glob(destination.name + ".incomplete.*"))
    assert len(quarantined) == 2

    (destination / "task.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ExecutionError, match="conflicts with local"):
        transport.deploy(task)


def test_remote_batch_deploy_uses_one_archive_transfer(tmp_path):
    initial = _write(tmp_path / "initial.xyz")
    model = _write(tmp_path / "model.txt")
    remote_root = tmp_path / "remote"
    target = ExecutionTarget(
        "remote",
        "slurm",
        host="fixture",
        work_root="~/remote",
        partition="cpu",
        command=(
            f"env PYTHONPATH={Path(__file__).resolve().parents[1] / 'src'} "
            f"{sys.executable} -m NepTrain.cli.cli"
        ),
    )
    context = StageContext(
        generation=1,
        generation_dir=tmp_path / "generation",
        plan=_plan(),
        artifacts={"model": model},
        previous_artifacts={},
    )
    config = {
        "training": {},
        "evaluation": {},
        "md": {"spin": False},
        "labeling": {},
        "workflow": {},
        "execution": {},
    }
    tasks = [
        build_stage_task(
            tmp_path / "tasks",
            workflow_root=tmp_path,
            workflow_id="batch-deploy",
            workflow_instance_id="instance",
            generation=1,
            stage="explore",
            attempt=attempt,
            target=target,
            plan=_plan(),
            config=config,
            initial_training=initial,
            context=context,
        )
        for attempt in (1, 2)
    ]
    copies = []

    class LocalRemoteTransport(ExecutionTransport):
        def run_script(self, script, *arguments, check=False, timeout=60):
            completed = subprocess.run(
                ["bash", "-s", "--", *(str(value) for value in arguments)],
                input=script,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                env={**os.environ, "HOME": str(tmp_path)},
            )
            if check and completed.returncode:
                raise ExecutionError(completed.stderr)
            return completed

        def copy(self, source, destination, **_kwargs):
            copies.append((source, destination))
            remote_path = Path(str(destination).split(":", 1)[1])
            remote_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, remote_path)
            return subprocess.CompletedProcess([], 0, "", "")

    transport = LocalRemoteTransport(target)
    deployed = transport.deploy_many(tasks)

    assert len(copies) == 1
    assert deployed == tuple(
        str(remote_root / "batch-deploy" / "jobs" / task.bundle.name)
        for task in tasks
    )
    for task, destination in zip(tasks, deployed, strict=True):
        assert (
            Path(destination) / "task.json"
        ).read_bytes() == task.descriptor.read_bytes()


def test_process_identity_checks_request_untruncated_commands(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return subprocess.CompletedProcess(args, 0, "controller /long/task/path\n", "")

    monkeypatch.setattr(execution_module.subprocess, "run", fake_run)
    assert execution_module._pid_matches_bundle(123, "/long/task/path")
    assert calls[-1][0] == ["ps", "-ww", "-p", "123", "-o", "command="]
    assert calls[-1][1]["env"]["COLUMNS"] == "10000"

    monkeypatch.setattr(controller_module.subprocess, "run", fake_run)
    assert controller_module._process_matches(456, Path("/long/task/path"))
    assert calls[-1][0] == ["ps", "-ww", "-p", "456", "-o", "command="]
    assert calls[-1][1]["env"]["COLUMNS"] == "10000"


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
            "labeling": {"resource_path": str(dft_resource)},
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
    assert "resource_path" not in descriptor["config"]["labeling"]
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
    assert evaluate_descriptor["artifacts"]["model"] == "input/nep.dat"
    assert (
        evaluate_descriptor["artifacts"]["retrained_model"]
        == "input/candidate-nep.dat"
    )
    assert (
        evaluate_descriptor["artifacts"]["training_input"]
        == "input/base-train.dat"
    )
    assert (
        evaluate_descriptor["artifacts"]["training_set"]
        == "input/train.dat"
    )

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
    model = _write(tmp_path / "model.dat")
    task = build_stage_task(
        tmp_path / "tasks",
        workflow_root=tmp_path,
        workflow_id="path-safety",
        generation=1,
        stage="explore",
        attempt=1,
        target=ExecutionTarget("local", "process"),
        plan=_plan(),
        config={
            "training": {},
            "evaluation": {},
            "md": {"spin": False},
            "labeling": {},
            "workflow": {},
            "execution": {},
        },
        initial_training=_write(tmp_path / "initial.xyz"),
        context=StageContext(
            generation=1,
            generation_dir=tmp_path / "generation",
            plan=_plan(),
            artifacts={"model": model},
            previous_artifacts={},
        ),
    )
    descriptor = json.loads(task.descriptor.read_text())
    (task.bundle / "result.json").write_text(
        json.dumps(
            {
                "protocol": "neptrain.stage-result.v3",
                "task_id": task.task_id,
                "task_spec_sha256": descriptor["spec_sha256"],
                "workflow_id": task.workflow_id,
                "workflow_instance_id": descriptor["identity"][
                    "workflow_instance_id"
                ],
                "generation": task.generation,
                "stage": task.stage,
                "plan_sha256": descriptor["identity"]["plan_sha256"],
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
        load_stage_result(task.bundle)


_STAGE_ARTIFACTS = {
    "train": ("training_input", "model"),
    "explore": ("candidates",),
    "select": ("selected_input", "selection_result"),
    "label": ("labeled", "label_provenance"),
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
        result_root = task.bundle / "output" / "artifacts"
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
            path = result_root / name / filename
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
                frames = ase_read(source, index=":")
                for frame in frames:
                    frame.info[INPUT_STRUCTURE_ID_KEY] = structure_id(frame)
                    frame.info["virial"] = np.zeros((3, 3))
                    frame.calc = SinglePointCalculator(
                        frame,
                        energy=-1.0,
                        forces=[[0.0, 0.0, 0.0]],
                    )
                ase_write(
                    path,
                    frames,
                    format="extxyz",
                )
            elif name == "label_provenance":
                path.write_text(
                    json.dumps(
                        {
                            "version": 1,
                            "backend": descriptor["config"]["labeling"][
                                "backend"
                            ],
                            "origin": "development",
                        }
                    ),
                    encoding="utf-8",
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
                "backend": descriptor["config"]["labeling"]["backend"],
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
                    "protocol": "neptrain.stage-result.v3",
                    "task_id": task.task_id,
                    "task_spec_sha256": descriptor["spec_sha256"],
                    "workflow_id": task.workflow_id,
                    "workflow_instance_id": descriptor["identity"][
                        "workflow_instance_id"
                    ],
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
    initial = _write_labeled(tmp_path / "initial.xyz")
    _write(tmp_path / "nep.in")
    ase_write(
        tmp_path / "structures.xyz",
        Atoms("Fe", positions=[[0.0, 0.0, 0.0]]),
        format="extxyz",
    )
    _write(tmp_path / "lammps.in", "run {{ steps }}\n")
    _write_labeled(tmp_path / "validation.xyz")
    config = _write(
        tmp_path / "project.yaml",
        """
schema_version: 8
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
labeling:
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
    labeling: label
    analysis: cpu
  sampling_route_targets: {}
  targets:
    gpu: {executor: process}
    md: {executor: process}
    label: {executor: process}
    cpu: {executor: process}
""",
    )
    return config, initial


def test_run_controller_scopes_stop_event_and_restores_signal_handlers(
    tmp_path: Path, monkeypatch
):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    previous_handlers = {
        signal.SIGTERM: object(),
        signal.SIGINT: object(),
    }
    installed_handlers = {}
    signal_calls = []

    monkeypatch.setattr(
        controller_module.signal,
        "getsignal",
        lambda signum: previous_handlers[signum],
    )

    def record_signal(signum, handler):
        signal_calls.append((signum, handler))
        if callable(handler):
            installed_handlers[signum] = handler
        return previous_handlers[signum]

    monkeypatch.setattr(controller_module.signal, "signal", record_signal)

    def request_stop_on_first_tick(_controller, *, should_stop=None):
        assert should_stop is not None
        installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
        return ControllerTick("running")

    monkeypatch.setattr(
        PersistentController,
        "tick",
        request_stop_on_first_tick,
    )

    assert run_controller(preparation.output_dir, poll_interval=0.2) == 0

    state = json.loads(
        WorkflowWorkspace.locate(
            preparation.output_dir
        ).controller_file.read_text(encoding="utf-8")
    )
    assert state["state"] == "stopped"
    assert state["reason"] == "controller stopped by user"
    assert signal_calls[-2:] == [
        (signal.SIGTERM, previous_handlers[signal.SIGTERM]),
        (signal.SIGINT, previous_handlers[signal.SIGINT]),
    ]
    assert not WorkflowWorkspace.locate(
        preparation.output_dir
    ).controller_pid.exists()


def test_notification_failure_never_changes_controller_result(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")

    class BrokenNotifier:
        def observe(self, **_kwargs):
            raise RuntimeError("intentional notification failure")

        def close(self):
            raise RuntimeError("intentional close failure")

    monkeypatch.setattr(
        "NepTrain.core.notifications.build_workflow_notifier",
        lambda *_args, **_kwargs: BrokenNotifier(),
    )
    monkeypatch.setattr(
        PersistentController,
        "tick",
        lambda *_args, **_kwargs: ControllerTick("complete"),
    )

    assert run_controller(preparation.output_dir, poll_interval=0.2) == 0
    assert "intentional notification failure" in capsys.readouterr().err


def test_group_submission_checks_stop_between_parallel_launch_batches(
    tmp_path: Path,
):
    config, initial = _controller_inputs(tmp_path)
    _write(tmp_path / "INCAR", "IBRION = -1\nNSW = 0\nISPIN = 1\n")
    _write_vasp_resources(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "labeling:\n  backend: toy\n",
            (
                "labeling:\n"
                "  backend: vasp\n"
                "  input_path: ./INCAR\n"
                "  resource_path: ./potpaw\n"
                "  potcar_manifest_path: ./vasp-resources.json\n"
                "  structures_per_job: 1\n"
                "  max_concurrent: 20\n"
            ),
        ),
        encoding="utf-8",
    )
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []

    class ManySelectedExecutor(ImmediateExecutor):
        def launch(self, task):
            handle = super().launch(task)
            if task.stage != "select":
                return handle
            result_path = task.bundle / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            record = result["artifacts"]["selected_input"]
            selected = task.bundle / record["path"]
            ase_write(
                selected,
                [
                    Atoms("Fe", positions=[[0.01 * index, 0.0, 0.0]])
                    for index in range(1, 11)
                ],
                format="extxyz",
            )
            record["sha256"] = _sha256(selected)
            record["size"] = selected.stat().st_size
            result_path.write_text(json.dumps(result), encoding="utf-8")
            return handle

    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: ManySelectedExecutor(target, launches),
    )

    for _ in range(20):
        controller.tick(
            should_stop=lambda: sum(
                stage == "label" for stage, _ in launches
            )
            >= 4
        )
        current = controller.state.get("current") or {}
        if current.get("stage") == "label":
            break
    else:
        raise AssertionError("controller did not reach label submission")

    label_launches = [item for item in launches if item[0] == "label"]
    assert len(current["tasks"]) == 10
    assert len(label_launches) == 4
    assert sum(bool(item.get("handle")) for item in current["tasks"]) == 4
    controller.tick(should_stop=lambda: True)
    assert len([item for item in launches if item[0] == "label"]) == 4


def test_grouped_stages_use_bulk_launch_and_collection_when_supported(
    tmp_path: Path,
):
    config, initial = _controller_inputs(tmp_path)
    _write(tmp_path / "INCAR", "IBRION = -1\nNSW = 0\nISPIN = 1\n")
    _write_vasp_resources(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "labeling:\n  backend: toy\n",
            (
                "labeling:\n"
                "  backend: vasp\n"
                "  input_path: ./INCAR\n"
                "  resource_path: ./potpaw\n"
                "  potcar_manifest_path: ./vasp-resources.json\n"
                "  structures_per_job: 1\n"
                "  max_concurrent: 20\n"
            ),
        ),
        encoding="utf-8",
    )
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []
    bulk_launches = []
    bulk_collections = []

    class BulkImmediateExecutor(ImmediateExecutor):
        def launch_many(self, tasks):
            bulk_launches.append(
                [(task.stage, task.task_id) for task in tasks]
            )
            return tuple(self.launch(task) for task in tasks)

        def collect_many(self, handles):
            bulk_collections.append(
                [Path(handle.local_bundle).name for handle in handles]
            )
            return tuple(self.collect(handle) for handle in handles)

    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: BulkImmediateExecutor(
            target,
            launches,
        ),
    )
    for _ in range(40):
        tick = controller.tick()
        if tick.state in {"complete", "converged"}:
            break
    else:
        raise AssertionError("controller did not complete the bulk fixture")

    label_launch = next(
        batch
        for batch in bulk_launches
        if batch and batch[0][0] == "label"
    )
    assert len(label_launch) == 3
    assert any(
        len(batch) == 3
        and all(name.startswith("g0001-label-") for name in batch)
        for batch in bulk_collections
    ), bulk_collections


def test_remote_batch_collection_uses_one_compact_archive(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    _write(tmp_path / "INCAR", "IBRION = -1\nNSW = 0\nISPIN = 1\n")
    _write_vasp_resources(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "labeling:\n  backend: toy\n",
            (
                "labeling:\n"
                "  backend: vasp\n"
                "  input_path: ./INCAR\n"
                "  resource_path: ./potpaw\n"
                "  potcar_manifest_path: ./vasp-resources.json\n"
                "  structures_per_job: 1\n"
                "  max_concurrent: 20\n"
            ),
        ),
        encoding="utf-8",
    )
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []
    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: ImmediateExecutor(target, launches),
    )
    for _ in range(30):
        controller.tick()
        label = next(
            (
                item
                for item in controller.state.get("history", [])
                if item.get("kind") == "task_group"
                and item.get("stage") == "label"
            ),
            None,
        )
        if label is not None:
            break
    else:
        raise AssertionError("controller did not complete the label fixture")

    remote_root = tmp_path / "remote" / "workflow"
    handles = []
    for item in label["tasks"][:2]:
        local = Path(item["bundle"])
        remote = remote_root / "jobs" / local.name
        shutil.copytree(local, remote)
        descriptor = json.loads(
            (local / "task.json").read_text(encoding="utf-8")
        )
        (remote / "execution.json").write_text(
            json.dumps(
                {
                    "state": "COMPLETED",
                    "task_id": descriptor["task_id"],
                    "task_spec_sha256": descriptor["spec_sha256"],
                }
            ),
            encoding="utf-8",
        )
        shutil.rmtree(local / "output")
        (local / "result.json").unlink()
        (local / "execution.json").unlink(missing_ok=True)
        handles.append(
            ExecutionHandle(
                descriptor["task_id"],
                "remote",
                "slurm",
                f"job-{len(handles)}",
                str(local),
                remote_bundle=str(remote),
            )
        )

    target = ExecutionTarget(
        "remote",
        "slurm",
        host="fixture",
        work_root=str(tmp_path / "remote"),
        partition="cpu",
    )
    copies = []

    class LocalRemoteTransport(ExecutionTransport):
        def run_script(self, script, *arguments, check=False, timeout=60):
            completed = subprocess.run(
                ["bash", "-s", "--", *(str(value) for value in arguments)],
                input=script,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
            if check and completed.returncode:
                raise ExecutionError(completed.stderr)
            return completed

        def copy(self, source, destination, **_kwargs):
            copies.append((source, destination))
            source_path = Path(str(source).split(":", 1)[1])
            shutil.copy2(source_path, destination)
            return subprocess.CompletedProcess([], 0, "", "")

    transport = LocalRemoteTransport(target)
    collected = transport.collect_many(handles)

    assert len(copies) == 1
    assert all(isinstance(path, Path) for path in collected)
    for handle in handles:
        _, outcome = load_stage_result(handle.local_bundle)
        assert outcome.metrics["labeled_count"] == 1


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
        ("label", "label"),
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


def test_status_audits_and_resume_repairs_committed_result_projection(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: ImmediateExecutor(target, []),
    )
    for _ in range(20):
        if controller.tick().state == "complete":
            break
    else:
        raise AssertionError("controller did not complete")

    workspace = WorkflowWorkspace.locate(preparation.output_dir)
    visible_model = workspace.results_dir / "nep.txt"
    visible_model.unlink()
    status = workflow_status(preparation.output_dir)
    assert status.state == "damaged"
    assert "results/nep.txt" in status.reason

    assert controller.generation_controller.next_stage(controller.plans[0]) is None
    assert visible_model.is_symlink()
    assert workflow_status(preparation.output_dir).state == "complete"

    publication_model = visible_model.resolve()
    publication_model.unlink()
    status = workflow_status(preparation.output_dir)
    assert status.state == "damaged"
    assert "accepted result drifted" in status.reason

    # Repair uses the still hash-checked stage artifact, never an unverified
    # visible result.
    assert controller.generation_controller.next_stage(controller.plans[0]) is None
    assert publication_model.is_file()
    assert workflow_status(preparation.output_dir).state == "complete"

    ledger = json.loads(workspace.ledger.read_text(encoding="utf-8"))
    committed_model = Path(
        ledger["generations"]["1"]["stages"]["evaluate"]["artifacts"][
            "activated_model"
        ]["path"]
    )
    committed_model.unlink()
    status = workflow_status(preparation.output_dir)
    assert status.state == "damaged"
    assert "committed artifact drifted or is missing" in status.reason

    repaired = resume_workflow(preparation.output_dir)
    assert repaired.action == "repair"
    assert committed_model.is_file()
    assert workflow_status(preparation.output_dir).state == "complete"


def test_resume_refuses_irrecoverable_committed_artifact_damage(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: ImmediateExecutor(target, []),
    )
    for _ in range(20):
        if controller.tick().state == "complete":
            break
    else:
        raise AssertionError("controller did not complete")

    workspace = WorkflowWorkspace.locate(preparation.output_dir)
    ledger = json.loads(workspace.ledger.read_text(encoding="utf-8"))
    label_record = ledger["generations"]["1"]["stages"]["label"]["artifacts"][
        "labeled"
    ]
    expected = label_record["sha256"]
    committed = Path(label_record["path"])
    committed.unlink()
    for result_path in workspace.tasks_dir.glob("*/result.json"):
        value = json.loads(result_path.read_text(encoding="utf-8"))
        for artifact in value.get("artifacts", {}).values():
            candidate = result_path.parent / artifact["path"]
            if artifact.get("sha256") == expected and candidate.is_file():
                candidate.unlink()

    with pytest.raises(
        WorkflowError,
        match="authoritative damage.*resume was refused",
    ):
        resume_workflow(preparation.output_dir)
    assert workflow_status(preparation.output_dir).state == "damaged"


def test_missing_scientific_ledger_is_not_treated_as_a_fresh_workflow(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: ImmediateExecutor(target, []),
    )
    for _ in range(20):
        if controller.tick().state == "complete":
            break
    else:
        raise AssertionError("controller did not complete")

    workspace = WorkflowWorkspace.locate(preparation.output_dir)
    workspace.ledger.unlink()

    status = workflow_status(preparation.output_dir)
    assert status.state == "damaged"
    assert "scientific ledger is missing" in status.reason
    with pytest.raises(WorkflowError, match="authoritative damage"):
        resume_workflow(preparation.output_dir)


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
        "    label: {executor: process}\n",
        "    label: {executor: process}\n    md2: {executor: process}\n",
    )
    config.write_text(text, encoding="utf-8")
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []
    tracker = {"active": 0, "maximum": 0}
    tracker_lock = threading.Lock()

    class TrackingImmediateExecutor(ImmediateExecutor):
        def inspect(self, handle):
            with tracker_lock:
                tracker["active"] += 1
                tracker["maximum"] = max(
                    tracker["maximum"],
                    tracker["active"],
                )
            try:
                time.sleep(0.03)
                return super().inspect(handle)
            finally:
                with tracker_lock:
                    tracker["active"] -= 1

    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: TrackingImmediateExecutor(
            target,
            launches,
        ),
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
    assert {
        item["route_id"]: (item["temperature"], item["steps"])
        for item in current["tasks"]
    } == {"default": (300.0, 2), "second": (500.0, 2)}
    assert all(item["target_level"] == "smoke_passed" for item in current["tasks"])
    assert all(item["handle"] is not None for item in current["tasks"])
    assert [stage for stage, _ in launches].count("explore") == 2
    controller.tick()
    assert tracker["maximum"] == 2


def test_controller_splits_dft_labels_and_limits_concurrency(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    _write(tmp_path / "INCAR", "IBRION = -1\nNSW = 0\nISPIN = 1\n")
    _write_vasp_resources(tmp_path)
    text = config.read_text(encoding="utf-8").replace(
        "labeling:\n  backend: toy\n",
        (
            "labeling:\n"
            "  backend: vasp\n"
            "  input_path: ./INCAR\n"
            "  resource_path: ./potpaw\n"
            "  potcar_manifest_path: ./vasp-resources.json\n"
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
    dft_root = preparation.output_dir / "generations" / "0001" / "label"
    assert len(list(dft_root.glob("00000*-Fe"))) == 3
    assert not (dft_root / "calculations").exists()


def test_controller_reports_exhausted_sampling_coverage(
    tmp_path, monkeypatch
):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []
    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: ImmediateExecutor(target, launches),
    )
    monkeypatch.setattr(
        workflow_iteration_module.WorkflowIterationAdapter,
        "plan_explore_attempts",
        lambda _self, _context: (),
    )

    for _ in range(6):
        tick = controller.tick()
        if tick.state == "coverage_exhausted":
            break
    else:
        raise AssertionError("controller did not stop at exhausted coverage")

    assert controller.state["current"] is None
    assert "independent validation" in controller.state["reason"]
    assert [stage for stage, _ in launches] == ["train"]


def test_label_oom_is_not_retried_and_does_not_stop_sibling_tasks(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    _write(tmp_path / "INCAR", "IBRION = -1\nNSW = 0\nISPIN = 1\n")
    _write_vasp_resources(tmp_path)
    text = config.read_text(encoding="utf-8").replace(
        "labeling:\n  backend: toy\n",
        (
            "labeling:\n"
            "  backend: vasp\n"
            "  input_path: ./INCAR\n"
            "  resource_path: ./potpaw\n"
            "  potcar_manifest_path: ./vasp-resources.json\n"
            "  structures_per_job: 1\n"
            "  max_concurrent: 2\n"
        ),
    )
    config.write_text(text, encoding="utf-8")
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []

    class OneOomExecutor(ImmediateExecutor):
        def launch(self, task):
            descriptor = json.loads(task.descriptor.read_text(encoding="utf-8"))
            if (
                task.stage == "label"
                and descriptor["stage_input"]["batch_index"] == 1
            ):
                self.launches.append((task.stage, task.target))
                return ExecutionHandle(
                    task.task_id,
                    task.target,
                    "slurm",
                    "oom-label",
                    str(task.bundle),
                )
            return super().launch(task)

        def inspect(self, handle):
            if handle.execution_id == "oom-label":
                return ExecutionStatus(
                    "failed",
                    "Slurm OUT_OF_MEMORY exit=0:125",
                    "out_of_memory",
                )
            return super().inspect(handle)

    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: OneOomExecutor(target, launches),
    )

    for _ in range(40):
        tick = controller.tick()
        if tick.state == "complete":
            break
    else:
        raise AssertionError("controller did not finish after partial DFT OOM")

    state = json.loads(
        (preparation.output_dir / ".neptrain/controller.json").read_text()
    )
    label = next(item for item in state["history"] if item["stage"] == "label")
    assert [item.get("retryable", True) for item in label["tasks"]] == [
        False,
        True,
        True,
    ]
    assert [item["failure_kind"] for item in label["tasks"][:1]] == [
        "out_of_memory"
    ]
    assert [stage for stage, _ in launches].count("label") == 3
    assert label["metrics"]["requested_count"] == 3
    assert label["metrics"]["labeled_count"] == 2
    assert label["metrics"]["failed_frame_indices"] == [0]
    failure_file = (
        preparation.output_dir
        / "generations/0001/label/label-failures.json"
    )
    failures = json.loads(failure_file.read_text(encoding="utf-8"))
    assert failures["failures"][0]["failure_kind"] == "out_of_memory"
    status = workflow_status(preparation.output_dir)
    assert any(job["state"] == "SKIPPED" for job in status.jobs)


def test_failed_labels_are_skipped_without_blocking_successful_siblings(
    tmp_path,
):
    config, initial = _controller_inputs(tmp_path)
    _write(tmp_path / "INCAR", "IBRION = -1\nNSW = 0\nISPIN = 1\n")
    _write_vasp_resources(tmp_path)
    text = config.read_text(encoding="utf-8").replace(
        "labeling:\n  backend: toy\n",
        (
            "labeling:\n"
            "  backend: vasp\n"
            "  input_path: ./INCAR\n"
            "  resource_path: ./potpaw\n"
            "  potcar_manifest_path: ./vasp-resources.json\n"
            "  structures_per_job: 1\n"
            "  max_concurrent: 2\n"
        ),
    )
    config.write_text(text, encoding="utf-8")
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []

    class PartiallyFailingExecutor(ImmediateExecutor):
        def launch(self, task):
            descriptor = json.loads(task.descriptor.read_text(encoding="utf-8"))
            batch_index = descriptor.get("stage_input", {}).get("batch_index")
            if task.stage == "label" and batch_index in {1, 2}:
                self.launches.append((task.stage, task.target))
                return ExecutionHandle(
                    task.task_id,
                    task.target,
                    "slurm",
                    f"failed-label-{batch_index}",
                    str(task.bundle),
                )
            return super().launch(task)

        def inspect(self, handle):
            if handle.execution_id == "failed-label-1":
                return ExecutionStatus(
                    "failed",
                    "VASP electronic SCF did not converge",
                    "non_convergence",
                )
            if handle.execution_id == "failed-label-2":
                return ExecutionStatus(
                    "failed",
                    "MPI process exited after segmentation fault",
                )
            return super().inspect(handle)

    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: PartiallyFailingExecutor(
            target, launches
        ),
    )

    for _ in range(40):
        tick = controller.tick()
        if tick.state == "complete":
            break
    else:
        raise AssertionError(
            "controller did not finish after partial label failures"
        )

    state = json.loads(
        (preparation.output_dir / ".neptrain/controller.json").read_text()
    )
    label = next(item for item in state["history"] if item["stage"] == "label")
    assert [item.get("retryable", True) for item in label["tasks"]] == [
        False,
        False,
        True,
    ]
    assert [item["failure_kind"] for item in label["tasks"][:2]] == [
        "non_convergence",
        "execution_failure",
    ]
    assert [stage for stage, _ in launches].count("label") == 3
    assert label["metrics"]["requested_count"] == 3
    assert label["metrics"]["labeled_count"] == 1
    assert label["metrics"]["failed_batch_count"] == 2
    assert label["metrics"]["failed_frame_indices"] == [0, 1]
    failure_file = (
        preparation.output_dir
        / "generations/0001/label/label-failures.json"
    )
    failures = json.loads(failure_file.read_text(encoding="utf-8"))
    assert [
        item["failure_kind"] for item in failures["failures"]
    ] == ["non_convergence", "execution_failure"]
    status = workflow_status(preparation.output_dir)
    assert [job["state"] for job in status.jobs].count("SKIPPED") == 2


def test_controller_stalls_when_every_label_fails(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    _write(tmp_path / "INCAR", "IBRION = -1\nNSW = 0\nISPIN = 1\n")
    _write_vasp_resources(tmp_path)
    text = config.read_text(encoding="utf-8").replace(
        "labeling:\n  backend: toy\n",
        (
            "labeling:\n"
            "  backend: vasp\n"
            "  input_path: ./INCAR\n"
            "  resource_path: ./potpaw\n"
            "  potcar_manifest_path: ./vasp-resources.json\n"
            "  structures_per_job: 1\n"
            "  max_concurrent: 2\n"
        ),
    )
    config.write_text(text, encoding="utf-8")
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []

    class AllLabelsFailExecutor(ImmediateExecutor):
        def launch(self, task):
            if task.stage == "label":
                descriptor = json.loads(
                    task.descriptor.read_text(encoding="utf-8")
                )
                batch_index = descriptor["stage_input"]["batch_index"]
                self.launches.append((task.stage, task.target))
                return ExecutionHandle(
                    task.task_id,
                    task.target,
                    "slurm",
                    f"failed-label-{batch_index}",
                    str(task.bundle),
                )
            return super().launch(task)

        def inspect(self, handle):
            if handle.execution_id.startswith("failed-label-"):
                return ExecutionStatus("cancelled", "scheduler cancelled job")
            return super().inspect(handle)

    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: AllLabelsFailExecutor(
            target, launches
        ),
    )

    for _ in range(40):
        tick = controller.tick()
        if tick.state == "stalled":
            break
    else:
        raise AssertionError("controller did not stall after all labels failed")

    assert tick.detail == (
        "all 3 label tasks failed, were cancelled, or returned invalid "
        "labels; failure evidence was preserved"
    )
    assert [stage for stage, _ in launches].count("label") == 3
    current = controller.state["current"]
    assert current["stage"] == "label"
    assert all(item["terminal_failure"] for item in current["tasks"])
    assert all(not item["retryable"] for item in current["tasks"])
    assert all(
        item["failure_kind"] == "cancelled" for item in current["tasks"]
    )
    assert not (
        preparation.output_dir
        / "generations/0001/label/selected-labels.xyz"
    ).exists()
    with pytest.raises(
        ControllerError, match="has no failed or unfinished tasks to retry"
    ):
        controller.retry()
    status = workflow_status(preparation.output_dir)
    assert status.state == "stalled"
    assert [job["state"] for job in status.jobs].count("SKIPPED") == 3


def test_invalid_collected_label_is_skipped_without_blocking_siblings(
    tmp_path,
):
    config, initial = _controller_inputs(tmp_path)
    _write(tmp_path / "INCAR", "IBRION = -1\nNSW = 0\nISPIN = 1\n")
    _write_vasp_resources(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "labeling:\n  backend: toy\n",
            (
                "labeling:\n"
                "  backend: vasp\n"
                "  input_path: ./INCAR\n"
                "  resource_path: ./potpaw\n"
                "  potcar_manifest_path: ./vasp-resources.json\n"
                "  structures_per_job: 1\n"
                "  max_concurrent: 2\n"
            ),
        ),
        encoding="utf-8",
    )
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []

    class OneInvalidCollection(ImmediateExecutor):
        def launch(self, task):
            handle = super().launch(task)
            descriptor = json.loads(
                task.descriptor.read_text(encoding="utf-8")
            )
            if (
                task.stage == "label"
                and descriptor["stage_input"]["batch_index"] == 1
            ):
                return ExecutionHandle(
                    handle.task_id,
                    handle.target,
                    handle.executor,
                    "invalid-label-result",
                    handle.local_bundle,
                )
            return handle

        def collect(self, handle):
            if handle.execution_id == "invalid-label-result":
                raise PermanentExecutionError(
                    "remote label result failed hash validation"
                )
            return super().collect(handle)

    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: OneInvalidCollection(
            target, launches
        ),
    )

    for _ in range(40):
        tick = controller.tick()
        if tick.state == "complete":
            break
    else:
        raise AssertionError(
            "controller did not finish after one invalid label result"
        )

    label = next(
        item
        for item in controller.state["history"]
        if item["stage"] == "label"
    )
    assert label["metrics"]["labeled_count"] == 2
    assert label["metrics"]["failed_frame_indices"] == [0]
    assert label["tasks"][0]["failure_kind"] == "result_validation_failure"
    assert label["tasks"][0]["retryable"] is False


def test_label_submission_rejection_fails_before_submitting_siblings(
    tmp_path,
):
    config, initial = _controller_inputs(tmp_path)
    _write(tmp_path / "INCAR", "IBRION = -1\nNSW = 0\nISPIN = 1\n")
    _write_vasp_resources(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "labeling:\n  backend: toy\n",
            (
                "labeling:\n"
                "  backend: vasp\n"
                "  input_path: ./INCAR\n"
                "  resource_path: ./potpaw\n"
                "  potcar_manifest_path: ./vasp-resources.json\n"
                "  structures_per_job: 1\n"
                "  max_concurrent: 2\n"
            ),
        ),
        encoding="utf-8",
    )
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []

    class RejectedLabelSubmission(ImmediateExecutor):
        def launch(self, task):
            if task.stage == "label":
                raise PermanentExecutionError(
                    "Slurm rejected the labeling target"
                )
            return super().launch(task)

    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: RejectedLabelSubmission(
            target, launches
        ),
    )

    for _ in range(20):
        tick = controller.tick()
        if tick.state == "failed":
            break
    else:
        raise AssertionError("label submission rejection was not terminal")

    tasks = controller.state["current"]["tasks"]
    assert tasks[0]["failure_kind"] == "submission_failure"
    assert tasks[0]["retryable"] is True
    assert all(item["handle"] is None for item in tasks[1:])


def test_permanent_single_stage_collection_error_is_terminal(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []

    class InvalidTrainingCollection(ImmediateExecutor):
        def collect(self, _handle):
            raise PermanentExecutionError(
                "remote training result failed validation"
            )

    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: InvalidTrainingCollection(
            target, launches
        ),
    )

    assert controller.tick().state == "running"
    failed = controller.tick()

    assert failed.state == "failed"
    assert failed.stage == "train"
    assert controller.state["current"]["terminal_failure"] is True
    assert (
        controller.state["current"]["failure_kind"]
        == "result_validation_failure"
    )


def test_controller_stalls_when_md_wave_has_no_safe_candidate_frames(
    tmp_path,
):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []

    class EmptyExploreExecutor(ImmediateExecutor):
        def launch(self, task):
            handle = super().launch(task)
            if task.stage == "explore":
                result_path = task.bundle / "result.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result["artifacts"].pop("candidates")
                result["metrics"]["candidate_count"] = 0
                result_path.write_text(
                    json.dumps(result), encoding="utf-8"
                )
            return handle

    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: EmptyExploreExecutor(
            target, launches
        ),
    )

    for _ in range(10):
        tick = controller.tick()
        if tick.state == "stalled":
            break
    else:
        raise AssertionError("empty MD wave did not enter stalled")

    assert tick.detail.startswith(
        "MD exploration produced no safe candidate frames"
    )
    assert controller.state["current"]["stage"] == "explore"
    assert all(
        item.get("collected_bundle")
        for item in controller.state["current"]["tasks"]
    )


def test_group_merge_failure_can_retry_collected_results(
    tmp_path,
    monkeypatch,
):
    from NepTrain.core.workflow_iteration import (
        WorkflowIterationAdapter,
        WorkflowIterationError,
    )

    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []
    original_merge = WorkflowIterationAdapter.merge_explore_outcomes
    failures = {"remaining": 1}

    def fail_once(self, context, outcomes):
        if failures["remaining"]:
            failures["remaining"] -= 1
            raise WorkflowIterationError("temporary merge failure")
        return original_merge(self, context, outcomes)

    monkeypatch.setattr(
        WorkflowIterationAdapter,
        "merge_explore_outcomes",
        fail_once,
    )
    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: ImmediateExecutor(
            target, launches
        ),
    )

    for _ in range(10):
        tick = controller.tick()
        if tick.state == "failed":
            break
    else:
        raise AssertionError("injected group merge failure was not observed")

    assert all(
        item.get("collected_bundle")
        for item in controller.state["current"]["tasks"]
    )
    controller.retry()
    assert controller.state["state"] == "launching"
    assert controller.state["current"]["merge_retry_count"] == 1

    controller.tick()
    assert any(
        item["stage"] == "explore"
        for item in controller.state["history"]
    )


def test_controller_waits_and_reuses_task_after_submission_throttle(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []
    deferred = {"remaining": 1}

    class DeferredOnceExecutor(ImmediateExecutor):
        def launch(self, task):
            if deferred["remaining"]:
                deferred["remaining"] -= 1
                raise SubmissionDeferred(
                    "Job violates accounting/QOS policy "
                    "(job submit limit, user's size and/or time limits)"
                )
            return super().launch(task)

    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: DeferredOnceExecutor(target, launches),
    )

    first = controller.tick()
    first_task_id = controller.state["current"]["task_id"]
    second = controller.tick()

    assert first.state == "waiting"
    assert second.state == "running"
    assert controller.state["current"]["task_id"] == first_task_id
    assert controller.state["current"]["attempt"] == 1
    assert "terminal_failure" not in controller.state["current"]
    assert launches == [("train", "gpu")]


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
        "    label: {executor: process}\n",
        "    label: {executor: process}\n    md2: {executor: process}\n",
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
        "label",
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
        "label/label",
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
    (bundle / "execution.json").write_text("{", encoding="utf-8")
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


def test_process_executor_reports_corrupt_execution_metadata_as_failed(
    tmp_path, monkeypatch
):
    bundle = tmp_path / "task"
    bundle.mkdir()
    (bundle / "execution.json").write_text("{", encoding="utf-8")
    local = ProcessExecutor(ExecutionTarget("local", "process"))
    handle = ExecutionHandle("abc", "local", "process", "999999", str(bundle))

    local_status = local.inspect(handle)

    assert local_status.state == "failed"
    assert "execution descriptor is unreadable" in local_status.detail

    remote = ProcessExecutor(
        ExecutionTarget(
            "remote",
            "process",
            host="fixture",
            work_root="/remote/work",
        )
    )
    monkeypatch.setattr(
        remote.transport,
        "run_script",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, "{", ""
        ),
    )
    remote_handle = ExecutionHandle(
        "abc",
        "remote",
        "process",
        "123",
        str(bundle),
        remote_bundle="/remote/work/task",
    )

    remote_status = remote.inspect(remote_handle)

    assert remote_status.state == "failed"
    assert "remote execution descriptor is unreadable" in remote_status.detail


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


def test_remote_slurm_executor_submits_a_group_through_one_remote_script(
    tmp_path,
):
    target = ExecutionTarget(
        "slurm",
        "slurm",
        host="remote",
        work_root="/remote/work",
        partition="cpu",
    )
    tasks = []
    for index in range(2):
        bundle = tmp_path / f"task-{index}"
        bundle.mkdir()
        tasks.append(
            StageTask(
                f"task-{index}",
                "demo",
                1,
                "label",
                "slurm",
                bundle,
            )
        )
    executor = SlurmExecutor(target)
    calls = []

    executor.transport.resolve_remote_path = lambda _path: "/remote/work"

    def fake_deploy_many(received, *, resolved_root=None):
        assert list(received) == tasks
        assert resolved_root == "/remote/work"
        assert all((task.bundle / "submit.sh").is_file() for task in tasks)
        return tuple(
            f"/remote/work/demo/jobs/{task.bundle.name}"
            for task in tasks
        )

    def fake_run_script(script, *arguments, **_kwargs):
        calls.append((script, arguments))
        assert len(arguments) == 4
        return subprocess.CompletedProcess(
            ["ssh"],
            0,
            (
                "JOB\tnt-task-0\t101\n"
                "JOB\tnt-task-1\t102\n"
            ),
            "",
        )

    executor.transport.deploy_many = fake_deploy_many
    executor.transport.run_script = fake_run_script

    handles = executor.launch_many(tasks)

    assert len(calls) == 1
    assert [handle.execution_id for handle in handles] == ["101", "102"]
    assert all(
        isinstance(handle, ExecutionHandle)
        for handle in handles
    )


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


def test_slurm_executor_classifies_out_of_memory_as_non_retryable_kind(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    runner = SlurmRunner()
    runner.state = "OUT_OF_MEMORY"
    executor = SlurmExecutor(
        ExecutionTarget("slurm", "slurm", partition="gpu"), runner=runner
    )
    handle = ExecutionHandle("abc", "slurm", "slurm", "123", str(bundle))

    status = executor.inspect(handle)

    assert status.state == "failed"
    assert status.failure_kind == "out_of_memory"
    assert status.detail == "Slurm OUT_OF_MEMORY exit=0:0"


def test_execution_failure_kind_classifies_scf_nonconvergence():
    assert (
        execution_module._failure_kind(
            "VASP electronic SCF did not converge in /tmp/calculation"
        )
        == "non_convergence"
    )


def test_remote_slurm_failure_collection_is_compact(tmp_path):
    executor = SlurmExecutor(
        ExecutionTarget(
            "dft",
            "slurm",
            host="remote",
            work_root="/remote/work",
            partition="cpu",
        )
    )
    fetched = []

    def fake_run_script(script, *arguments, **kwargs):
        assert arguments == ("/remote/work/task",)
        assert 'find "$path" -type f -print' in script
        return subprocess.CompletedProcess(
            ["ssh"],
            0,
            stdout=(
                ".output-building-123/retry-0002/OUTCAR\n"
                ".output-building-123/retry-0002/vasp.out\n"
                ".output-building-123/retry-0002/PROCAR\n"
                "../outside/OUTCAR\n"
            ),
            stderr="",
        )

    def fake_fetch(remote_root, members, destination_root):
        fetched.append(
            (str(remote_root), tuple(members), Path(destination_root))
        )
        return tuple(str(member) for member in members)

    executor.transport.run_script = fake_run_script
    executor.transport.fetch_paths = fake_fetch
    handle = ExecutionHandle(
        task_id="abc",
        target="dft",
        executor="slurm",
        execution_id="701234",
        local_bundle=str(tmp_path / "task"),
        remote_bundle="/remote/work/task",
    )

    evidence = executor.collect_failure(handle)

    assert evidence == (
        tmp_path / "task" / "failure-evidence" / "slurm-701234"
    )
    assert fetched == [
        (
            "/remote/work/task",
            (
                "execution.json",
                "submit.sh",
                "stdout-701234.log",
                ".output-building-123/retry-0002/OUTCAR",
                ".output-building-123/retry-0002/vasp.out",
            ),
            evidence,
        )
    ]


def test_slurm_executor_preserves_worker_failure_detail(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    (bundle / "execution.json").write_text(
        json.dumps(
            {
                "state": "FAILED",
                "error": (
                    "VASP electronic SCF did not converge in calculation"
                ),
            }
        ),
        encoding="utf-8",
    )
    runner = SlurmRunner()
    runner.state = "FAILED"
    executor = SlurmExecutor(
        ExecutionTarget("slurm", "slurm", partition="gpu"),
        runner=runner,
    )
    handle = ExecutionHandle("abc", "slurm", "slurm", "123", str(bundle))

    status = executor.inspect(handle)

    assert status.state == "failed"
    assert status.failure_kind == "non_convergence"
    assert status.detail == (
        "VASP electronic SCF did not converge in calculation; "
        "Slurm FAILED exit=0:0"
    )


def test_slurm_submission_limit_is_deferred_instead_of_failed(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    task = StageTask("abc", "demo", 1, "train", "slurm", bundle)

    class ThrottledRunner(SlurmRunner):
        def __call__(self, args, **kwargs):
            args = list(args)
            if args[0] == "sbatch":
                self.submissions += 1
                return subprocess.CompletedProcess(
                    args,
                    1,
                    "",
                    "sbatch: error: QOSMaxSubmitJobPerUserLimit",
                )
            return super().__call__(args, **kwargs)

    executor = SlurmExecutor(
        ExecutionTarget("slurm", "slurm", partition="cpu"),
        runner=ThrottledRunner(),
    )

    with pytest.raises(SubmissionDeferred, match="QOSMaxSubmitJobPerUserLimit"):
        executor.launch(task)


def test_slurm_permanent_submission_rejection_is_terminal(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    task = StageTask("abc", "demo", 1, "train", "slurm", bundle)

    class RejectedRunner(SlurmRunner):
        def __call__(self, args, **kwargs):
            args = list(args)
            if args[0] == "sbatch":
                return subprocess.CompletedProcess(
                    args,
                    1,
                    "",
                    "sbatch: error: invalid partition specified",
                )
            return super().__call__(args, **kwargs)

    executor = SlurmExecutor(
        ExecutionTarget("slurm", "slurm", partition="missing"),
        runner=RejectedRunner(),
    )

    with pytest.raises(
        PermanentExecutionError,
        match="invalid partition specified",
    ):
        executor.launch(task)


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
        if args[9:14] == [
            "remote",
            "bash",
            "-s",
            "--",
            "/remote/task",
        ]:
            command = args[14:]
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


def test_stop_prepared_workflow_is_an_idempotent_no_op(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    workspace = WorkflowWorkspace.locate(preparation.output_dir)

    result = stop_workflow(preparation.output_dir)

    assert result["controller"] == "already_stopped"
    assert result["current_execution"]["action"] == "none"
    assert not workspace.controller_file.exists()
    assert not workspace.ledger.exists()
    status = workflow_status(preparation.output_dir)
    assert status.state == "prepared"
    assert "workflow run" in status.next_action


def test_stop_complete_workflow_does_not_rewrite_it_as_paused(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    controller = PersistentController(
        preparation.output_dir,
        executor_factory=lambda target: ImmediateExecutor(target, []),
    )
    for _ in range(20):
        if controller.tick().state == "complete":
            break
    else:
        raise AssertionError("controller did not complete")
    workspace = WorkflowWorkspace.locate(preparation.output_dir)
    ledger_before = workspace.ledger.read_bytes()

    result = stop_workflow(preparation.output_dir)

    assert result["current_execution"]["action"] == "none"
    assert workspace.ledger.read_bytes() == ledger_before
    assert workflow_status(preparation.output_dir).state == "complete"


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
    ledger = json.loads(
        (preparation.output_dir / ".neptrain/ledger.json").read_text()
    )
    stop_event = ledger["execution_events"][-1]
    assert stop_event["event"] == "workflow_stop"
    assert stop_event["cancel_jobs"] is True
    assert stop_event["current_execution"]["action"] == "cancelling"


def test_group_stop_preserves_completed_labels_and_skips_cancelled_shard(
    tmp_path,
):
    config, initial = _controller_inputs(tmp_path)
    _write(tmp_path / "INCAR", "IBRION = -1\nNSW = 0\nISPIN = 1\n")
    _write_vasp_resources(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "labeling:\n  backend: toy\n",
            (
                "labeling:\n"
                "  backend: vasp\n"
                "  input_path: ./INCAR\n"
                "  resource_path: ./potpaw\n"
                "  potcar_manifest_path: ./vasp-resources.json\n"
                "  structures_per_job: 1\n"
                "  max_concurrent: 2\n"
            ),
        ),
        encoding="utf-8",
    )
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []
    cancelled_ids = set()
    inspected_ids = []

    class StopAwareExecutor(ImmediateExecutor):
        def inspect(self, handle):
            inspected_ids.append(handle.execution_id)
            if handle.execution_id in cancelled_ids:
                return ExecutionStatus("cancelled", "fixture cancellation")
            return ExecutionStatus("completed")

        def cancel(self, handle):
            cancelled_ids.add(handle.execution_id)
            return ExecutionStatus("cancelled", "fixture cancellation")

    factory = lambda target: StopAwareExecutor(target, launches)
    controller = PersistentController(
        preparation.output_dir,
        executor_factory=factory,
    )
    for _ in range(30):
        controller.tick()
        current = controller.state.get("current") or {}
        if (
            current.get("stage") == "label"
            and sum(
                bool(item.get("collected_bundle"))
                for item in current.get("tasks", [])
            )
            == 2
            and sum(
                bool(item.get("handle"))
                and not item.get("collected_bundle")
                for item in current.get("tasks", [])
            )
            == 1
        ):
            break
    else:
        raise AssertionError("controller did not reach a partially collected label group")

    original_ids = [item["task_id"] for item in current["tasks"]]
    result = stop_workflow(
        preparation.output_dir,
        executor_factory=factory,
    )

    assert result["current_execution"]["action"] == "group_cancellation_requested"
    stopped = json.loads(
        (preparation.output_dir / ".neptrain/controller.json").read_text()
    )
    assert stopped["current"] is not None
    assert sum(
        bool(item.get("collected_bundle"))
        for item in stopped["current"]["tasks"]
    ) == 2
    status = workflow_status(preparation.output_dir)
    assert status.state == "paused"
    assert [job["state"] for job in status.jobs[-3:]] == [
        "COMPLETED",
        "COMPLETED",
        "CANCELLED",
    ]
    inspected_before_resume = list(inspected_ids)

    resumed = PersistentController(
        preparation.output_dir,
        executor_factory=factory,
    )
    resumed.resume_stopped()
    current = resumed.state["current"]
    assert inspected_ids == inspected_before_resume
    assert current["attempt"] == 1
    assert [item["task_id"] for item in current["tasks"]] == original_ids
    assert all(
        item.get("collected_bundle") for item in current["tasks"][:2]
    )
    assert current["tasks"][2]["terminal_failure"] is True
    assert current["tasks"][2]["retryable"] is False
    assert current["tasks"][2]["failure_kind"] == "cancelled"

    resumed.tick()
    label = next(
        item
        for item in resumed.state["history"]
        if item["stage"] == "label"
    )
    assert label["metrics"]["labeled_count"] == 2
    assert label["metrics"]["failed_frame_indices"] == [2]


def test_group_stop_retries_all_labels_when_none_were_collected(tmp_path):
    config, initial = _controller_inputs(tmp_path)
    _write(tmp_path / "INCAR", "IBRION = -1\nNSW = 0\nISPIN = 1\n")
    _write_vasp_resources(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "labeling:\n  backend: toy\n",
            (
                "labeling:\n"
                "  backend: vasp\n"
                "  input_path: ./INCAR\n"
                "  resource_path: ./potpaw\n"
                "  potcar_manifest_path: ./vasp-resources.json\n"
                "  structures_per_job: 1\n"
                "  max_concurrent: 2\n"
            ),
        ),
        encoding="utf-8",
    )
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    launches = []
    cancelled_ids = set()
    inspected_ids = []

    class StopAwareExecutor(ImmediateExecutor):
        def inspect(self, handle):
            inspected_ids.append(handle.execution_id)
            if handle.execution_id in cancelled_ids:
                return ExecutionStatus("cancelled", "fixture cancellation")
            return ExecutionStatus("completed")

        def cancel(self, handle):
            cancelled_ids.add(handle.execution_id)
            return ExecutionStatus("cancelled", "fixture cancellation")

    factory = lambda target: StopAwareExecutor(target, launches)
    controller = PersistentController(
        preparation.output_dir,
        executor_factory=factory,
    )
    for _ in range(30):
        controller.tick()
        current = controller.state.get("current") or {}
        if (
            current.get("stage") == "label"
            and sum(
                bool(item.get("handle"))
                for item in current.get("tasks", [])
            )
            == 2
            and not any(
                item.get("collected_bundle")
                for item in current.get("tasks", [])
            )
        ):
            break
    else:
        raise AssertionError("controller did not reach an active label group")

    original_ids = [item["task_id"] for item in current["tasks"]]
    stop_workflow(
        preparation.output_dir,
        executor_factory=factory,
    )
    inspected_before_resume = list(inspected_ids)

    resumed = PersistentController(
        preparation.output_dir,
        executor_factory=factory,
    )
    resumed.resume_stopped()
    current = resumed.state["current"]

    assert inspected_ids == inspected_before_resume
    assert resumed.state["state"] == "launching"
    assert current["attempt"] == 2
    assert [item["task_id"] for item in current["tasks"]] != original_ids
    assert all(item.get("handle") is None for item in current["tasks"])
    assert all(not item.get("terminal_failure") for item in current["tasks"])
    assert all(len(item["retry_history"]) == 1 for item in current["tasks"])


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


def test_controller_turns_persistently_unknown_execution_into_retryable_failure(
    tmp_path,
    monkeypatch,
):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")

    class UnknownExecutor:
        def __init__(self, target):
            self.target = target

        def launch(self, task):
            return ExecutionHandle(
                task.task_id,
                task.target,
                "slurm",
                "701234",
                str(task.bundle),
            )

        def inspect(self, _handle):
            return ExecutionStatus(
                "unknown",
                "Slurm has no accounting record yet",
            )

    monkeypatch.setattr(
        controller_module,
        "_EXECUTION_UNKNOWN_GRACE_SECONDS",
        0.0,
    )
    controller = PersistentController(
        preparation.output_dir,
        executor_factory=UnknownExecutor,
    )

    assert controller.tick().state == "running"
    failed = controller.tick()

    assert failed.state == "failed"
    assert "recovery grace period" in failed.detail
    controller.retry()
    assert controller.state["current"] is None


def test_controller_recovers_after_repeated_transport_errors(tmp_path, monkeypatch):
    config, initial = _controller_inputs(tmp_path)
    preparation = prepare_workflow(config, initial, tmp_path / "workflow")
    task_id = "preserved-task"

    transport_calls = []

    def intermittent_transport(controller, *, should_stop=None):
        assert should_stop is not None
        if controller.state.get("current") is None:
            controller.state["current"] = {
                "task_id": task_id,
                "generation": 1,
                "stage": "train",
                "resource": "training",
                "target": "gpu",
                "attempt": 1,
                "bundle": str(
                    preparation.output_dir / ".neptrain/jobs/preserved-task"
                ),
                "handle": {
                    "task_id": task_id,
                    "target": "gpu",
                    "executor": "slurm",
                    "execution_id": "701234",
                    "local_bundle": str(
                        preparation.output_dir
                        / ".neptrain/jobs/preserved-task"
                    ),
                },
            }
            controller._save()
            return ControllerTick("running")
        transport_calls.append(controller.state.get("state"))
        if len(transport_calls) <= 3:
            raise ExecutionError("SSH transport is unavailable")
        controller.state["state"] = "complete"
        controller._save()
        return ControllerTick("complete")

    monkeypatch.setattr(PersistentController, "tick", intermittent_transport)

    assert (
        run_controller(preparation.output_dir, poll_interval=0.2)
        == 0
    )
    state = json.loads(
        (preparation.output_dir / ".neptrain/controller.json").read_text()
    )
    assert state["state"] == "complete"
    assert state["transport_failures"] == 0
    assert "last_transport_error" not in state
    assert state["current"]["task_id"] == task_id
    assert transport_calls == ["running", "degraded", "degraded", "degraded"]
    assert state["current"]["handle"]["execution_id"] == "701234"


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
    path = bundle / 'output' / 'artifacts' / name / f'{name}{suffix}'
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
    'protocol': 'neptrain.stage-result.v3',
    'task_id': task['task_id'],
    'task_spec_sha256': task['spec_sha256'],
    'workflow_id': task['identity']['workflow_id'],
    'workflow_instance_id': task['identity']['workflow_instance_id'],
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
    'task_id': task['task_id'], 'task_spec_sha256': task['spec_sha256'],
    'state': 'COMPLETED', 'pid': os.getpid()
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
    path = bundle / 'output' / 'artifacts' / name / f'{name}{suffix}'
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
    'protocol': 'neptrain.stage-result.v3',
    'task_id': task['task_id'],
    'task_spec_sha256': task['spec_sha256'],
    'workflow_id': task['identity']['workflow_id'],
    'workflow_instance_id': task['identity']['workflow_instance_id'],
    'generation': task['identity']['generation'],
    'stage': stage,
    'plan_sha256': task['identity']['plan_sha256'],
    'artifacts': artifacts,
    'metrics': {
        'accepted': True, 'workflow_converged': True
    } if stage == 'evaluate' else {},
}))
(bundle / 'execution.json').write_text(json.dumps({
    'task_id': task['task_id'], 'task_spec_sha256': task['spec_sha256'],
    'state': 'COMPLETED', 'pid': os.getpid()
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
