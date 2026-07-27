from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import numpy as np
import pytest

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from NepTrain.core.execution import (
    ExecutionError,
    ExecutionHandle,
    ExecutionTarget,
    ExecutionTransport,
    build_stage_task,
)
from NepTrain.core.iteration import GenerationPlan, StageContext
from NepTrain.core.scientific_data import (
    labeled_input_structure_ids,
    structure_id,
)
from NepTrain.core.manual import (
    ManualTaskError,
    cancel_operation,
    load_operation,
    prepare_labeling,
    prepare_md,
    operation_logs,
    refresh_operation,
    retry_failed,
    run_manual_worker,
    submit_operation,
    wait_operation,
)
import NepTrain.core.manual as manual_module


def _structures(path: Path, count: int = 3) -> Path:
    frames = [
        Atoms(
            "Fe2",
            positions=[[0, 0, 0], [2.2 + index * 0.05, 0, 0]],
            cell=[6, 6, 6],
            pbc=True,
        )
        for index in range(count)
    ]
    write(path, frames, format="extxyz")
    return path


def test_local_manual_dft_splits_and_publishes_in_input_order(tmp_path):
    source = _structures(tmp_path / "input.xyz")
    output = tmp_path / "labeled.xyz"
    operation = prepare_labeling(
        source,
        backend="toy",
        output=output,
        workdir=tmp_path / "run",
        target=ExecutionTarget("local", "process"),
        structures_per_job=2,
        max_concurrent=2,
    )

    status = submit_operation(operation)

    assert status["state"] == "complete"
    assert status["total"] == 2
    frames = read(output, index=":")
    assert len(frames) == 3
    assert [round(frame.positions[1, 0], 2) for frame in frames] == [
        2.2,
        2.25,
        2.3,
    ]
    assert all(frame.calc and "energy" in frame.calc.results for frame in frames)
    assert (operation.root / "labeled.xyz").is_symlink()
    assert (operation.root / "labeled.xyz").resolve() == output
    assert (operation.root / "jobs" / "000000" / "labeled.xyz").is_file()
    assert not (operation.root / "jobs" / "000000" / "result").exists()
    assert not (operation.root / "shards").exists()


def test_manual_spin_dft_uses_input_ownership_when_final_spin_relaxes(
    tmp_path, monkeypatch
):
    source_frame = Atoms(
        "Fe",
        positions=[[0.0, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    source_frame.set_array("spin", np.asarray([[1.0, 0.0, 0.0]]))
    source = tmp_path / "spin-input.xyz"
    output = tmp_path / "spin-labeled.xyz"
    write(source, source_frame, format="extxyz")

    def relaxed_spin_teacher(request):
        frame = read(request.source)
        frame.set_array("spin", np.asarray([[0.0, 0.8, 0.0]]))
        frame.set_array("mforce", np.asarray([[0.1, 0.2, 0.3]]))
        frame.info["virial"] = np.zeros((3, 3))
        frame.calc = SinglePointCalculator(
            frame,
            energy=-1.0,
            forces=np.zeros((1, 3)),
        )
        write(request.output_file, frame, format="extxyz")
        return [frame]

    monkeypatch.setattr(
        "NepTrain.core.dft.toy.run_toy_teacher",
        relaxed_spin_teacher,
    )
    operation = prepare_labeling(
        source,
        backend="toy",
        output=output,
        workdir=tmp_path / "run",
        target=ExecutionTarget("local", "process"),
        teacher_profile="spin",
    )

    assert run_manual_worker(operation.root, 0) == 0
    status = refresh_operation(operation)

    assert status["state"] == "complete"
    restored = read(output)
    assert labeled_input_structure_ids([restored]) == [structure_id(source_frame)]
    assert structure_id(restored) != structure_id(source_frame)


def test_manual_dft_defaults_to_input_authoritative_kpoints(tmp_path):
    source = _structures(tmp_path / "input.xyz", count=1)
    dft_input = tmp_path / "INPUT"
    dft_input.write_text("INPUT_PARAMETERS\nkspacing 0.25\n", encoding="utf-8")
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=ExecutionTarget("local", "process"),
        input_file=dft_input,
    )

    request = json.loads(
        (
            operation.root / "jobs" / "000000" / "request.json"
        ).read_text(encoding="utf-8")
    )
    assert request["kpoint_mode"] == "auto"
    assert request["kspacing"] is None


def test_manual_dft_rejects_mixed_ordinary_and_spin_inputs(tmp_path):
    ordinary = Atoms(
        "Fe",
        positions=[[0, 0, 0]],
        cell=[4, 4, 4],
        pbc=True,
    )
    spin = ordinary.copy()
    spin.set_array("spin", np.asarray([[1.0, 0.0, 0.0]]))
    source = tmp_path / "mixed.xyz"
    write(source, [ordinary, spin], format="extxyz")

    with pytest.raises(ManualTaskError, match="ordinary and spin frames cannot be mixed"):
        prepare_labeling(
            source,
            backend="toy",
            output=tmp_path / "labeled.xyz",
            workdir=tmp_path / "run",
            target=ExecutionTarget("local", "process"),
        )


def test_manual_step_refuses_to_overwrite_an_existing_result(tmp_path):
    source = _structures(tmp_path / "input.xyz")
    output = tmp_path / "labeled.xyz"
    output.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(ManualTaskError, match="--force"):
        prepare_labeling(
            source,
            backend="toy",
            output=output,
            workdir=tmp_path / "run",
            target=ExecutionTarget("local", "process"),
        )

    assert output.read_text(encoding="utf-8") == "keep me\n"
    assert not (tmp_path / "run").exists()


def test_slurm_manual_dft_uses_one_throttled_job_array(tmp_path, monkeypatch):
    source = _structures(tmp_path / "input.xyz", count=5)
    target = ExecutionTarget(
        "dft",
        "slurm",
        partition="cpu",
        cpus_per_task=4,
        command="neptrain",
    )
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=target,
        structures_per_job=2,
        max_concurrent=2,
    )
    calls = []

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="701234\n", stderr="")

    monkeypatch.setattr("NepTrain.core.manual.subprocess.run", fake_run)
    status = submit_operation(operation)

    assert status["state"] == "submitted"
    assert status["job_id"] == "701234"
    assert calls[0][0] == ["sbatch", "--parsable", "job.sbatch"]
    script = (operation.root / "job.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --array=0,1,2%2" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert f"#SBATCH --output={operation.root}/logs/scheduler-%A_%a.out" in script
    assert "manual-worker" in script
    assert "${SLURM_ARRAY_TASK_ID}" in script


def test_slurm_manual_dft_materializes_one_hundred_ordered_shards_with_limit(
    tmp_path, monkeypatch
):
    source = _structures(tmp_path / "input.xyz", count=100)
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=ExecutionTarget(
            "dft",
            "slurm",
            partition="cpu",
            command="neptrain",
        ),
        structures_per_job=1,
        max_concurrent=20,
    )

    def fake_run(args, **kwargs):
        if args[0] == "sbatch":
            return subprocess.CompletedProcess(
                args, 0, stdout="701234\n", stderr=""
            )
        if args[0] == "squeue":
            return subprocess.CompletedProcess(
                args, 0, stdout="PENDING\n", stderr=""
            )
        raise AssertionError(args)

    monkeypatch.setattr("NepTrain.core.manual.subprocess.run", fake_run)
    status = submit_operation(operation)

    descriptor = json.loads(operation.descriptor.read_text(encoding="utf-8"))
    assert operation.shard_count == status["total"] == 100
    assert len(descriptor["jobs"]) == 100
    assert len({job["frame_ids"][0] for job in descriptor["jobs"]}) == 100
    assert (operation.jobs_root / "000000" / "input.xyz").is_file()
    assert (operation.jobs_root / "000099" / "input.xyz").is_file()
    script = (operation.root / "job.sbatch").read_text(encoding="utf-8")
    expected_indices = ",".join(str(index) for index in range(100))
    assert f"#SBATCH --array={expected_indices}%20" in script


def test_execution_transport_sends_remote_scripts_via_stdin():
    calls = []

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="ok\n", stderr="")

    transport = ExecutionTransport(
        ExecutionTarget(
            "remote",
            "slurm",
            host="remote",
            work_root="/remote/work",
            partition="cpu",
        ),
        runner=fake_run,
    )

    completed = transport.run_script(
        'cd "$1"\nprintf "%s\\n" "$2"',
        "/remote/work/run with spaces",
        "value; still one argument",
        check=True,
    )

    assert completed.stdout == "ok\n"
    assert calls[0][0] == [
        "ssh",
        "remote",
        "bash",
        "-s",
        "--",
        "/remote/work/run with spaces",
        "value; still one argument",
    ]
    assert "-lc" not in calls[0][0]
    assert calls[0][1]["input"].endswith("\n")


def test_execution_transport_times_out_remote_scheduler_commands():
    def timed_out(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    transport = ExecutionTransport(
        ExecutionTarget(
            "remote",
            "slurm",
            host="remote",
            work_root="/remote/work",
            partition="cpu",
        ),
        runner=timed_out,
    )

    with pytest.raises(ExecutionError, match="timed out after 7s"):
        transport.run(
            ["/remote/work/run", "squeue", "-h"],
            timeout=7,
        )


def test_execution_transport_fetches_many_paths_with_one_copy(
    tmp_path, monkeypatch
):
    remote = tmp_path / "remote"
    (remote / "000000" / "result").mkdir(parents=True)
    (remote / "000000" / "execution.json").write_text(
        '{"state":"COMPLETED"}\n', encoding="utf-8"
    )
    (remote / "000000" / "result" / "labeled.xyz").write_text(
        "fixture\n", encoding="utf-8"
    )
    target = ExecutionTarget(
        "remote",
        "slurm",
        host="remote",
        work_root="/remote/work",
        partition="cpu",
    )
    transport = ExecutionTransport(target)
    remote_archives = []
    copies = []

    def fake_run_script(script, *arguments, **kwargs):
        if "existing=()" in script:
            root = Path(arguments[0])
            archive = root / str(arguments[1])
            remote_archives.append(archive)
            with tarfile.open(archive, "w:gz") as handle:
                for value in arguments[2:]:
                    path = root / str(value)
                    if path.exists():
                        handle.add(path, arcname=str(value))
        else:
            Path(arguments[0]).unlink(missing_ok=True)
        return subprocess.CompletedProcess(["ssh"], 0, stdout="", stderr="")

    def fake_copy(source, destination, **kwargs):
        copies.append((str(source), str(destination)))
        remote_path = Path(str(source).split(":", 1)[1])
        shutil.copy2(remote_path, destination)
        return subprocess.CompletedProcess(["scp"], 0, stdout="", stderr="")

    monkeypatch.setattr(transport, "run_script", fake_run_script)
    monkeypatch.setattr(transport, "copy", fake_copy)

    fetched = transport.fetch_paths(
        remote,
        (
            "000000/execution.json",
            "000000/result",
            "000001/execution.json",
        ),
        tmp_path / "local",
    )

    assert len(copies) == 1
    assert "000000/execution.json" in fetched
    assert (
        tmp_path / "local" / "000000" / "result" / "labeled.xyz"
    ).read_text(encoding="utf-8") == "fixture\n"
    assert all(not path.exists() for path in remote_archives)


def test_execution_transport_collects_stage_artifacts_in_one_fetch(
    tmp_path, monkeypatch
):
    remote = tmp_path / "remote"
    target = ExecutionTarget(
        "remote",
        "slurm",
        host="remote",
        work_root="/remote/work",
        partition="cpu",
    )
    initial = _structures(tmp_path / "initial.xyz", count=1)
    model_input = tmp_path / "model.input"
    model_input.write_text("input model\n", encoding="utf-8")
    plan = GenerationPlan(generation=1, seed=7, max_selected=2)
    task = build_stage_task(
        tmp_path / "tasks",
        workflow_root=tmp_path,
        workflow_id="collect",
        workflow_instance_id="instance",
        generation=1,
        stage="explore",
        attempt=1,
        target=target,
        plan=plan,
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
            plan=plan,
            artifacts={"model": model_input},
            previous_artifacts={},
        ),
    )
    descriptor = json.loads(task.descriptor.read_text(encoding="utf-8"))
    local = task.bundle
    artifacts = remote / "output" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "model.nep").write_text("model\n", encoding="utf-8")
    (artifacts / "loss.out").write_text("loss\n", encoding="utf-8")
    records = {
        name: {
            "path": f"output/artifacts/{path.name}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for name, path in {
            "model": artifacts / "model.nep",
            "loss": artifacts / "loss.out",
        }.items()
    }
    (remote / "result.json").write_text(
        json.dumps(
            {
                "protocol": "neptrain.stage-result.v3",
                "task_id": task.task_id,
                "task_spec_sha256": descriptor["spec_sha256"],
                "workflow_id": task.workflow_id,
                "workflow_instance_id": "instance",
                "generation": 1,
                "stage": "explore",
                "plan_sha256": plan.sha256,
                "artifacts": records,
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    (remote / "execution.json").write_text(
        json.dumps(
            {
                "state": "COMPLETED",
                "task_id": task.task_id,
                "task_spec_sha256": descriptor["spec_sha256"],
            }
        ),
        encoding="utf-8",
    )
    transport = ExecutionTransport(target)
    fetches = []

    def fake_fetch(remote_root, members, destination_root):
        fetches.append((str(remote_root), tuple(members)))
        shutil.copytree(remote, destination_root, dirs_exist_ok=True)
        return ("execution.json", "result.json", "output")

    monkeypatch.setattr(transport, "fetch_paths", fake_fetch)
    handle = ExecutionHandle(
        task_id=task.task_id,
        target="remote",
        executor="slurm",
        execution_id="701234",
        local_bundle=str(local),
        remote_bundle="/remote/work/task",
    )

    collected = transport.collect(handle)

    assert collected == local
    assert fetches == [
        (
            "/remote/work/task",
            ("result.json", "execution.json", "output"),
        )
    ]
    assert (local / "output" / "artifacts" / "model.nep").is_file()
    assert (local / "output" / "artifacts" / "loss.out").is_file()


def test_remote_manual_dft_uses_shared_execution_transport(
    tmp_path, monkeypatch, capsys
):
    source = _structures(tmp_path / "input.xyz", count=1)
    target = ExecutionTarget(
        "dft",
        "slurm",
        host="remote",
        work_root="/remote/work",
        partition="cpu",
        command="neptrain",
    )
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=target,
        structures_per_job=1,
        max_concurrent=1,
    )
    scripts = []
    copies = []

    class FakeTransport:
        def __init__(self, selected_target):
            assert selected_target == target

        def run_script(self, script, *arguments, **kwargs):
            scripts.append((script, arguments, kwargs))
            if "sbatch --parsable" in script:
                stdout = "701234\n"
            elif arguments and arguments[0] == "squeue":
                stdout = "PENDING\n"
            else:
                stdout = ""
            return subprocess.CompletedProcess(
                ["ssh"], 0, stdout=stdout, stderr=""
            )

        def copy(self, source_path, destination_path, **kwargs):
            copies.append((str(source_path), str(destination_path), kwargs))
            return subprocess.CompletedProcess(
                ["scp"], 0, stdout="", stderr=""
            )

        def fetch_paths(self, remote_root, members, destination_root):
            return ()

    monkeypatch.setattr("NepTrain.core.manual.ExecutionTransport", FakeTransport)

    status = submit_operation(operation)

    assert status["state"] == "submitted"
    assert status["job_id"] == "701234"
    assert any("tar -xzf" in script for script, _, _ in scripts)
    assert any("sbatch --parsable" in script for script, _, _ in scripts)
    assert all("-lc" not in script for script, _, _ in scripts)
    assert any(
        ":/remote/work/manual/.incoming/" in destination
        for _, destination, _ in copies
    )
    assert any(
        destination.endswith("/job.sbatch")
        for _, destination, _ in copies
    )
    progress = capsys.readouterr().err
    assert "uploading" in progress
    assert "submitted Slurm job 701234" in progress


def test_remote_status_bulk_sync_resumes_partial_job_results(
    tmp_path, monkeypatch
):
    source = _structures(tmp_path / "input.xyz", count=2)
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=ExecutionTarget(
            "dft",
            "slurm",
            host="remote",
            work_root="/remote/work",
            partition="cpu",
            command="neptrain",
        ),
        structures_per_job=1,
    )
    remote = tmp_path / "remote"
    for index in range(2):
        assert run_manual_worker(operation.root, index) == 0
    shutil.copytree(operation.root / "jobs", remote / "jobs")
    # Keep one valid local result and remove the other to exercise resumable
    # bulk collection without re-fetching completed work.
    for name in ("execution.json", "result.json", "labeled.xyz", "calculation"):
        path = operation.root / "jobs" / "000001" / name
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    descriptor = json.loads(operation.descriptor.read_text(encoding="utf-8"))
    descriptor.update(
        {
            "state": "submitted",
            "job_id": "701234",
            "remote_root": str(remote),
        }
    )
    operation.descriptor.write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    fetches = []

    class FakeTransport:
        def __init__(self, selected_target):
            assert selected_target == operation.target

        def run_script(self, script, *arguments, **kwargs):
            return subprocess.CompletedProcess(
                ["ssh"], 0, stdout="COMPLETED\n", stderr=""
            )

        def fetch_paths(self, remote_root, members, destination_root):
            fetches.append(tuple(str(value) for value in members))
            source_root = Path(remote_root)
            destination = Path(destination_root)
            fetched = []
            for value in members:
                source_path = source_root / str(value)
                if not source_path.exists():
                    continue
                destination_path = destination / str(value)
                if source_path.is_dir():
                    shutil.copytree(
                        source_path, destination_path, dirs_exist_ok=True
                    )
                else:
                    destination_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, destination_path)
                fetched.append(str(value))
            return tuple(fetched)

    monkeypatch.setattr("NepTrain.core.manual.ExecutionTransport", FakeTransport)

    status = refresh_operation(operation)

    assert len(fetches) == 1
    assert "000001/labeled.xyz" in fetches[0]
    assert "000000/labeled.xyz" not in fetches[0]
    assert status["state"] == "complete"
    assert status["completed"] == 2
    assert len(read(tmp_path / "labeled.xyz", index=":")) == 2


def test_remote_status_uses_one_result_transfer_for_fifty_jobs(
    tmp_path, monkeypatch
):
    source = _structures(tmp_path / "input.xyz", count=50)
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=ExecutionTarget(
            "dft",
            "slurm",
            host="remote",
            work_root="/remote/work",
            partition="cpu",
            command="neptrain",
        ),
        structures_per_job=1,
    )
    descriptor = json.loads(operation.descriptor.read_text(encoding="utf-8"))
    descriptor.update(
        {
            "state": "submitted",
            "job_id": "701234",
            "remote_root": "/remote/manual",
        }
    )
    operation.descriptor.write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    fetches = []

    class FakeTransport:
        def __init__(self, selected_target):
            assert selected_target == operation.target

        def run_script(self, script, *arguments, **kwargs):
            return subprocess.CompletedProcess(
                ["ssh"], 0, stdout="PENDING\n", stderr=""
            )

        def fetch_paths(self, remote_root, members, destination_root):
            fetches.append(tuple(str(value) for value in members))
            return ()

    monkeypatch.setattr("NepTrain.core.manual.ExecutionTransport", FakeTransport)

    status = refresh_operation(operation)

    assert status["state"] == "submitted"
    assert len(fetches) == 1
    assert len(fetches[0]) == 150
    assert fetches[0][:3] == (
        "000000/execution.json",
        "000000/result.json",
        "000000/labeled.xyz",
    )
    assert fetches[0][-3:] == (
        "000049/execution.json",
        "000049/result.json",
        "000049/labeled.xyz",
    )


def test_remote_logs_are_fetched_in_one_archive(tmp_path, monkeypatch):
    source = _structures(tmp_path / "input.xyz", count=2)
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=ExecutionTarget(
            "dft",
            "slurm",
            host="remote",
            work_root="/remote/work",
            partition="cpu",
            command="neptrain",
        ),
    )
    descriptor = json.loads(operation.descriptor.read_text(encoding="utf-8"))
    descriptor.update(
        {
            "state": "complete",
            "remote_root": "/remote/manual",
            "attempts": [
                {
                    "job_id": "1",
                    "indices": [0, 1],
                }
            ],
        }
    )
    operation.descriptor.write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    fetches = []
    listings = []

    class FakeTransport:
        def __init__(self, selected_target):
            assert selected_target == operation.target

        def run_script(self, script, *arguments, **kwargs):
            listings.append((script, arguments))
            return subprocess.CompletedProcess(
                ["ssh"],
                0,
                stdout=(
                    "/remote/manual/scheduler-1_0.out\n"
                    "/remote/manual/scheduler-1_1.out\n"
                ),
                stderr="",
            )

        def fetch_paths(self, remote_root, members, destination_root):
            fetches.append((str(remote_root), tuple(members)))
            for name in members:
                (Path(destination_root) / name).write_text(
                    name + "\n", encoding="utf-8"
                )
            return tuple(str(value) for value in members)

    monkeypatch.setattr("NepTrain.core.manual.ExecutionTransport", FakeTransport)

    logs = operation_logs(operation)
    cached_logs = operation_logs(operation)

    assert len(fetches) == 1
    assert len(listings) == 1
    assert fetches[0][1] == (
        "scheduler-1_0.out",
        "scheduler-1_1.out",
    )
    assert len(logs) == 2
    assert cached_logs == logs


def test_slurm_manual_md_expands_structure_temperature_matrix(
    tmp_path, monkeypatch
):
    source = _structures(tmp_path / "input.xyz", count=2)
    model = tmp_path / "nep.txt"
    model.write_text("fixture\n", encoding="utf-8")
    operation = prepare_md(
        source,
        backend="lammps",
        model_file=model,
        temperatures=[300, 600, 900],
        output=tmp_path / "trajectory.xyz",
        workdir=tmp_path / "run",
        target=ExecutionTarget(
            "md",
            "slurm",
            partition="cpu",
            cpus_per_task=4,
            command="neptrain",
        ),
        steps=100,
        max_concurrent=3,
    )

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="701234\n", stderr="")

    monkeypatch.setattr("NepTrain.core.manual.subprocess.run", fake_run)

    status = submit_operation(operation)

    assert status["total"] == 6
    script = (operation.root / "job.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --array=0,1,2,3,4,5%3" in script


def test_manual_md_runtime_options_are_explicit_request_inputs(tmp_path):
    source = _structures(tmp_path / "input.xyz", count=1)
    model = tmp_path / "nep.txt"
    model.write_text("fixture\n", encoding="utf-8")
    operation = prepare_md(
        source,
        backend="lammps",
        model_file=model,
        temperatures=[300],
        output=tmp_path / "trajectory.xyz",
        workdir=tmp_path / "run",
        target=ExecutionTarget(
            "md",
            "slurm",
            partition="cpu",
            cpus_per_task=4,
        ),
        steps=10,
        lmp="/opt/lammps/lmp",
        mpi_ranks=4,
        inference_backend="cpu",
    )

    request = json.loads(
        (
            operation.root / "jobs" / "000000" / "request.json"
        ).read_text(encoding="utf-8")
    )

    assert request["lmp"] == "/opt/lammps/lmp"
    assert request["mpi_ranks"] == 4
    assert request["inference_backend"] == "cpu"


def test_array_results_can_be_collected_after_workers_finish(tmp_path):
    source = _structures(tmp_path / "input.xyz", count=2)
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=ExecutionTarget("local", "process"),
        structures_per_job=1,
    )
    for index in range(operation.shard_count):
        assert run_manual_worker(operation.root, index) == 0

    status = refresh_operation(operation)

    assert status["state"] == "complete"
    assert len(read(tmp_path / "labeled.xyz", index=":")) == 2
    descriptor = json.loads(operation.descriptor.read_text(encoding="utf-8"))
    assert descriptor["state"] == "complete"


def test_corrupt_job_execution_is_reported_as_invalid_and_retryable(tmp_path):
    source = _structures(tmp_path / "input.xyz", count=1)
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=ExecutionTarget("local", "process"),
    )
    assert run_manual_worker(operation.root, 0) == 0
    execution = operation.jobs_root / "000000" / "execution.json"
    execution.write_text("{", encoding="utf-8")

    status = refresh_operation(operation)

    assert status["state"] == "failed"
    assert status["jobs"] == [{"index": 0, "state": "INVALID"}]
    assert status["errors"][0]["index"] == 0
    assert str(execution) in status["errors"][0]["error"]
    assert status["next_action"].startswith("neptrain task retry ")


def test_wait_reports_state_changes_to_stderr(tmp_path, capsys):
    source = _structures(tmp_path / "input.xyz", count=1)
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=ExecutionTarget("local", "process"),
        structures_per_job=1,
    )
    assert run_manual_worker(operation.root, 0) == 0
    capsys.readouterr()

    status = wait_operation(operation, poll_interval=0.01)

    assert status["state"] == "complete"
    progress = capsys.readouterr().err
    assert "[NepTrain] label: state=complete" in progress
    assert "completed=1/1" in progress


def test_scheduler_failure_marks_every_missing_shard_retryable(
    tmp_path, monkeypatch
):
    source = _structures(tmp_path / "input.xyz", count=2)
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=ExecutionTarget(
            "dft", "slurm", partition="cpu", command="neptrain"
        ),
        structures_per_job=1,
    )
    descriptor = json.loads(operation.descriptor.read_text(encoding="utf-8"))
    descriptor.update({"state": "submitted", "job_id": "701234"})
    operation.descriptor.write_text(
        json.dumps(descriptor), encoding="utf-8"
    )

    def fake_run(args, **kwargs):
        if args[0] == "squeue":
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="FAILED|\n", stderr="")

    monkeypatch.setattr("NepTrain.core.manual.subprocess.run", fake_run)

    status = refresh_operation(operation)

    assert status["state"] == "failed"
    assert status["scheduler_state"] == "FAILED"
    assert [item["index"] for item in status["errors"]] == [0, 1]


def test_retry_archives_failed_metadata_and_submits_only_failed_indices(
    tmp_path, monkeypatch
):
    source = _structures(tmp_path / "input.xyz", count=2)
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=ExecutionTarget(
            "dft", "slurm", partition="cpu", command="neptrain"
        ),
        structures_per_job=1,
    )
    assert run_manual_worker(operation.root, 0) == 0
    failed_execution = operation.root / "jobs" / "000001" / "execution.json"
    failed_execution.write_text(
        json.dumps({"state": "FAILED", "error": "fixture failure"}),
        encoding="utf-8",
    )
    descriptor = json.loads(operation.descriptor.read_text(encoding="utf-8"))
    descriptor.update({"state": "submitted", "job_id": "701234"})
    operation.descriptor.write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    calls = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[0] == "squeue":
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[0] == "sacct":
            return subprocess.CompletedProcess(args, 0, stdout="FAILED|\n", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="701235\n", stderr="")

    monkeypatch.setattr("NepTrain.core.manual.subprocess.run", fake_run)

    status = retry_failed(operation)

    assert status["job_id"] == "701235"
    assert "#SBATCH --array=1" in (
        operation.root / "retry.sbatch"
    ).read_text(encoding="utf-8")
    assert list(
        (operation.root / "jobs" / "000001" / "attempts").glob(
            "*/execution.json"
        )
    )
    assert ["sbatch", "--parsable", "retry.sbatch"] in calls


def test_manual_dft_rejects_a_hash_consistent_but_duplicated_result(
    tmp_path,
):
    source = _structures(tmp_path / "input.xyz", count=2)
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=ExecutionTarget(
            "remote",
            "slurm",
            host="remote",
            work_root="/remote/work",
            partition="cpu",
        ),
        structures_per_job=2,
    )
    assert run_manual_worker(operation.root, 0) == 0
    job = operation.root / "jobs" / "000000"
    labeled = job / "labeled.xyz"
    frames = read(labeled, index=":")
    write(labeled, [frames[0], frames[0], frames[1]], format="extxyz")
    result = json.loads((job / "result.json").read_text(encoding="utf-8"))
    result["artifact"]["sha256"] = hashlib.sha256(labeled.read_bytes()).hexdigest()
    result["artifact"]["size"] = labeled.stat().st_size
    (job / "result.json").write_text(json.dumps(result), encoding="utf-8")

    status = refresh_operation(operation)

    assert status["state"] == "failed"
    assert status["errors"][0]["index"] == 0
    assert "missing, duplicate, or reordered" in status["errors"][0]["error"]
    assert not (tmp_path / "labeled.xyz").exists()


def test_manual_complete_result_is_audited_and_rebuilt_from_valid_jobs(
    tmp_path,
):
    source = _structures(tmp_path / "input.xyz", count=2)
    output = tmp_path / "labeled.xyz"
    operation = prepare_labeling(
        source,
        backend="toy",
        output=output,
        workdir=tmp_path / "run",
        target=ExecutionTarget("local", "process"),
        structures_per_job=1,
    )
    assert submit_operation(operation)["state"] == "complete"
    expected_hash = hashlib.sha256(output.read_bytes()).hexdigest()

    output.unlink()
    status = refresh_operation(operation)

    assert status["state"] == "complete"
    assert hashlib.sha256(output.read_bytes()).hexdigest() == expected_hash
    assert len(read(output, index=":")) == 2

    output.unlink()
    (operation.root / "jobs" / "000001" / "labeled.xyz").unlink()
    status = refresh_operation(operation)
    assert status["state"] == "damaged"
    assert status["errors"]


def test_cancel_preserves_completed_jobs_and_retry_selects_only_unfinished(
    tmp_path,
    monkeypatch,
):
    source = _structures(tmp_path / "input.xyz", count=2)
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=ExecutionTarget(
            "dft",
            "slurm",
            partition="cpu",
            command="neptrain",
        ),
        structures_per_job=1,
    )
    assert run_manual_worker(operation.root, 0) == 0
    descriptor = json.loads(operation.descriptor.read_text(encoding="utf-8"))
    descriptor.update({"state": "submitted", "job_id": "701234"})
    operation.descriptor.write_text(json.dumps(descriptor), encoding="utf-8")
    cancelled = False
    calls = []

    def fake_run(args, **kwargs):
        nonlocal cancelled
        calls.append(list(args))
        if args[0] == "squeue":
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="" if cancelled else "RUNNING\n",
                stderr="",
            )
        if args[0] == "sacct":
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="CANCELLED|\n" if cancelled else "RUNNING|\n",
                stderr="",
            )
        if args[0] == "scancel":
            cancelled = True
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[0] == "sbatch":
            return subprocess.CompletedProcess(
                args, 0, stdout="701235\n", stderr=""
            )
        raise AssertionError(args)

    monkeypatch.setattr("NepTrain.core.manual.subprocess.run", fake_run)

    cancelled_status = cancel_operation(operation)
    assert cancelled_status["state"] == "cancelled"
    assert cancelled_status["completed"] == 1
    assert [item["index"] for item in cancelled_status["errors"]] == [1]
    completed_hash = hashlib.sha256(
        (operation.root / "jobs" / "000000" / "labeled.xyz").read_bytes()
    ).hexdigest()

    cancelled = False
    retried = retry_failed(operation)
    assert retried["job_id"] == "701235"
    assert "#SBATCH --array=1" in (
        operation.root / "retry.sbatch"
    ).read_text(encoding="utf-8")
    assert not (operation.root / "jobs" / "000000" / "attempts").exists()
    assert hashlib.sha256(
        (operation.root / "jobs" / "000000" / "labeled.xyz").read_bytes()
    ).hexdigest() == completed_hash
    descriptor = json.loads(operation.descriptor.read_text(encoding="utf-8"))
    assert descriptor["cancellations"][0]["job_id"] == "701234"


def test_missing_slurm_job_becomes_retryable_after_bounded_grace(
    tmp_path,
    monkeypatch,
):
    source = _structures(tmp_path / "input.xyz", count=2)
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=ExecutionTarget("dft", "slurm", partition="cpu"),
        structures_per_job=1,
    )
    descriptor = json.loads(operation.descriptor.read_text(encoding="utf-8"))
    descriptor.update({"state": "submitted", "job_id": "701234"})
    operation.descriptor.write_text(json.dumps(descriptor), encoding="utf-8")

    monkeypatch.setattr(
        manual_module,
        "_SCHEDULER_MISSING_GRACE_SECONDS",
        0.0,
    )
    monkeypatch.setattr(
        "NepTrain.core.manual.subprocess.run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="", stderr=""
        ),
    )

    status = refresh_operation(operation)

    assert status["state"] == "failed"
    assert status["scheduler_state"] == "LOST"
    assert [item["index"] for item in status["errors"]] == [0, 1]


def test_legacy_manual_operation_is_refused_without_unsafe_collection(tmp_path):
    source = _structures(tmp_path / "input.xyz", count=1)
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=ExecutionTarget("local", "process"),
    )
    descriptor = json.loads(operation.descriptor.read_text(encoding="utf-8"))
    descriptor["protocol"] = "neptrain.manual-operation.v2"
    operation.descriptor.write_text(json.dumps(descriptor), encoding="utf-8")

    with pytest.raises(ManualTaskError, match="unsafe legacy protocol"):
        load_operation(operation.root)


def test_remote_manual_bundle_is_published_atomically(tmp_path, monkeypatch):
    source = _structures(tmp_path / "input.xyz", count=1)
    remote_root = tmp_path / "remote"
    target = ExecutionTarget(
        "dft",
        "slurm",
        host="fixture",
        work_root=str(remote_root),
        partition="cpu",
    )
    operation = prepare_labeling(
        source,
        backend="toy",
        output=tmp_path / "labeled.xyz",
        workdir=tmp_path / "run",
        target=target,
    )

    class LocalRemoteTransport:
        def __init__(self, selected_target):
            assert selected_target == target

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
                raise ManualTaskError(completed.stderr)
            return completed

        def copy(self, source_path, destination_path, **_kwargs):
            destination = Path(str(destination_path).split(":", 1)[1])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(
        manual_module,
        "ExecutionTransport",
        LocalRemoteTransport,
    )
    destination = (
        remote_root / "manual" / operation.operation_id
    )
    destination.mkdir(parents=True)
    (destination / "partial").write_text("partial\n", encoding="utf-8")

    remote = manual_module._deploy_remote(operation)

    assert remote == str(destination)
    assert (destination / "operation.json").is_file()
    assert subprocess.run(
        ["sha256sum", "-c", "input-manifest.sha256"],
        cwd=destination,
        capture_output=True,
        check=False,
    ).returncode == 0
    quarantined = list(destination.parent.glob(destination.name + ".incomplete.*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "partial").is_file()
