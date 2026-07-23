from __future__ import annotations

import json
from pathlib import Path
import subprocess
import pytest

from ase import Atoms
from ase.io import read, write

from NepTrain.core.execution import ExecutionTarget
from NepTrain.core.manual import (
    ManualTaskError,
    prepare_dft,
    prepare_md,
    refresh_operation,
    retry_failed,
    run_manual_worker,
    submit_operation,
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
    assert "manual-worker" in script
    assert "${SLURM_ARRAY_TASK_ID}" in script


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


def test_target_overrides_are_recorded_in_manual_requests(tmp_path):
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
            overrides={
                "md.lmp": "/opt/lammps/lmp",
                "md.mpi_ranks": 4,
                "md.inference_backend": "cpu",
            },
        ),
        steps=10,
    )

    request = json.loads(
        (
            operation.root / "shards" / "000000" / "request.json"
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
    failed_execution = operation.root / "shards" / "000001" / "execution.json"
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
        (operation.root / "shards" / "000001" / "attempts").glob(
            "*/execution.json"
        )
    )
    assert ["sbatch", "--parsable", "retry.sbatch"] in calls
