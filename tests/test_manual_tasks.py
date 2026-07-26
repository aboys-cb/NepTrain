from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import pytest

from ase import Atoms
from ase.io import read, write

from NepTrain.core.execution import (
    ExecutionError,
    ExecutionHandle,
    ExecutionTarget,
    ExecutionTransport,
)
from NepTrain.core.manual import (
    ManualTaskError,
    prepare_dft,
    prepare_md,
    operation_logs,
    refresh_operation,
    retry_failed,
    run_manual_worker,
    submit_operation,
    wait_operation,
)


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
    operation = prepare_dft(
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


def test_manual_dft_defaults_to_input_authoritative_kpoints(tmp_path):
    source = _structures(tmp_path / "input.xyz", count=1)
    dft_input = tmp_path / "INPUT"
    dft_input.write_text("INPUT_PARAMETERS\nkspacing 0.25\n", encoding="utf-8")
    operation = prepare_dft(
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


def test_manual_step_refuses_to_overwrite_an_existing_result(tmp_path):
    source = _structures(tmp_path / "input.xyz")
    output = tmp_path / "labeled.xyz"
    output.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(ManualTaskError, match="--force"):
        prepare_dft(
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
    operation = prepare_dft(
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
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    artifacts = remote / "result" / "artifacts"
    artifacts.mkdir(parents=True)
    (remote / "execution.json").write_text(
        '{"state":"COMPLETED"}\n', encoding="utf-8"
    )
    (artifacts / "model.nep").write_text("model\n", encoding="utf-8")
    (artifacts / "loss.out").write_text("loss\n", encoding="utf-8")
    (remote / "result.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "model": {"path": "result/artifacts/model.nep"},
                    "loss": {"path": "result/artifacts/loss.out"},
                }
            }
        ),
        encoding="utf-8",
    )
    transport = ExecutionTransport(
        ExecutionTarget(
            "remote",
            "slurm",
            host="remote",
            work_root="/remote/work",
            partition="cpu",
        )
    )
    fetches = []

    def fake_fetch(remote_root, members, destination_root):
        fetches.append((str(remote_root), tuple(members)))
        shutil.copytree(remote, destination_root, dirs_exist_ok=True)
        return ("execution.json", "result.json", "result")

    monkeypatch.setattr(transport, "fetch_paths", fake_fetch)
    handle = ExecutionHandle(
        task_id="task",
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
            ("result.json", "execution.json", "result"),
        )
    ]
    assert (local / "result" / "artifacts" / "model.nep").is_file()
    assert (local / "result" / "artifacts" / "loss.out").is_file()


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
    operation = prepare_dft(
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
        destination.endswith(":/remote/work/manual/")
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
    operation = prepare_dft(
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
        job = remote / "jobs" / f"{index:06d}"
        job.mkdir(parents=True)
        (job / "execution.json").write_text(
            '{"state":"COMPLETED"}\n', encoding="utf-8"
        )
        (job / "result.json").write_text("{}\n", encoding="utf-8")
        write(
            job / "labeled.xyz",
            read(source, index=index),
            format="extxyz",
        )
    local_execution = (
        operation.root / "jobs" / "000000" / "execution.json"
    )
    local_execution.write_text(
        '{"state":"COMPLETED"}\n', encoding="utf-8"
    )
    (local_execution.parent / "result.json").write_text(
        "{}\n", encoding="utf-8"
    )
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
    assert "000000/labeled.xyz" in fetches[0]
    assert status["state"] == "complete"
    assert status["completed"] == 2
    assert len(read(tmp_path / "labeled.xyz", index=":")) == 2


def test_remote_status_uses_one_result_transfer_for_fifty_jobs(
    tmp_path, monkeypatch
):
    source = _structures(tmp_path / "input.xyz", count=50)
    operation = prepare_dft(
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
    operation = prepare_dft(
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
    operation = prepare_dft(
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


def test_wait_reports_state_changes_to_stderr(tmp_path, capsys):
    source = _structures(tmp_path / "input.xyz", count=1)
    operation = prepare_dft(
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
    assert "[NepTrain] dft: state=complete" in progress
    assert "completed=1/1" in progress


def test_scheduler_failure_marks_every_missing_shard_retryable(
    tmp_path, monkeypatch
):
    source = _structures(tmp_path / "input.xyz", count=2)
    operation = prepare_dft(
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
    operation = prepare_dft(
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
