from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
from ase.io import write as ase_write

import NepTrain.core.campaign as campaign_module
from NepTrain.core.campaign import (
    CampaignError,
    campaign_status,
    prepare_campaign,
    retry_failed_campaign,
    submit_campaign,
)
from NepTrain.core.dft.toy import ToyTeacher
from NepTrain.core.toy_workflow import toy_base_frame, toy_candidate_frames
from NepTrain.cli.cli import _iteration_execution, run_campaign_command


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    initial = tmp_path / "initial.xyz"
    structure = tmp_path / "structure.xyz"
    validation = tmp_path / "validation.xyz"
    frame = ToyTeacher("spin").label(toy_base_frame(True))
    ase_write(initial, [frame], format="extxyz")
    ase_write(structure, frame, format="extxyz")
    ase_write(
        validation,
        [ToyTeacher("spin").label(toy_candidate_frames("spin", 911, 1)[0])],
        format="extxyz",
    )
    (tmp_path / "nep.in").write_text(
        "type 1 Fe\nspin_descriptor spin_nep_lite\n", encoding="utf-8"
    )
    (tmp_path / "gpu-env.sh").write_text("export GPU_ENV=1\n", encoding="utf-8")
    (tmp_path / "cpu-env.sh").write_text(
        "module load lammps/nep-release\n", encoding="utf-8"
    )
    config = tmp_path / "job.yaml"
    config.write_text(
        """
schema_version: 2
current_job: training
training:
  backend: torchnep
  config_path: ./nep.in
  device: cuda
md:
  backend: lammps
  structures: ./structure.xyz
  inference_backend: cpu
  spin: true
  spin_temperature: auto
  mpi_ranks: 1
dft:
  software: toy
  teacher_profile: spin
evaluation:
  validation_path: ./validation.xyz
  max_rmse:
    energy_rmse: 1.0
    force_rmse: 1.0
    mforce_rmse: 1.0
campaign:
  id: spin-smoke
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
""",
        encoding="utf-8",
    )
    return config, initial


def test_campaign_prepares_progressive_plans_and_resource_scripts(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    result = prepare_campaign(config, initial, tmp_path / "campaign")

    assert result.campaign_id == "spin-smoke"
    assert len(result.plans) == 3
    assert len(result.scripts) == 10
    plans = [json.loads(path.read_text(encoding="utf-8")) for path in result.plans]
    assert [plan["steps"] for plan in plans] == [10, 40, 40]
    assert [plan["dft_budget"] for plan in plans] == [6, 5, 4]
    assert [plan["frame_stride"] for plan in plans] == [3, 3, 3]
    assert [plan["temperatures"] for plan in plans] == [[300.0], [300.0], [300.0, 500.0]]

    bootstrap = result.scripts[0].read_text(encoding="utf-8")
    sample = result.scripts[1].read_text(encoding="utf-8")
    retrain = result.scripts[2].read_text(encoding="utf-8")
    evaluate = result.scripts[3].read_text(encoding="utf-8")
    assert "#SBATCH --partition=16V100" in bootstrap
    assert "#SBATCH --gpus-per-node=1" in bootstrap
    assert "--resource training" in bootstrap
    assert "#SBATCH --partition=DSPRHBM" in sample
    assert "#SBATCH --cpus-per-task=4" in sample
    assert "--resource cpu" in sample
    assert "#SBATCH --partition=16V100" in retrain
    assert "--resource training" in retrain
    assert evaluate.count("iteration-resource") == 2
    assert "generation-2.json" in evaluate
    assert "module load" not in sample
    assert f"source {tmp_path / 'cpu-env.sh'}" in sample


def test_generated_resource_command_can_load_its_plan(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    result = prepare_campaign(config, initial, tmp_path / "campaign")

    plan, _, controller = _iteration_execution(
        SimpleNamespace(
            config_path=str(result.config_file),
            plan=str(result.plans[0]),
            initial_training=str(initial),
            campaign_dir=str(tmp_path / "state"),
            campaign_id=result.campaign_id,
        )
    )

    assert plan.generation == 1
    assert controller.next_stage(plan) == "train"


def test_campaign_submission_is_dependency_chained_and_idempotent(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    calls = []

    def runner(args, cwd):
        calls.append((list(args), cwd))
        return str(9000 + len(calls))

    first = submit_campaign(preparation, runner=runner)
    second = submit_campaign(preparation.output_dir, runner=runner)

    assert first.job_ids == second.job_ids == tuple(
        str(9000 + i) for i in range(1, 11)
    )
    assert calls[0][0][:2] == ["sbatch", "--parsable"]
    assert calls[0][0][2].startswith("--job-name=nt-")
    assert calls[0][0][-1] == str(preparation.scripts[0])
    assert "--dependency=afterok:9001" in calls[1][0]
    assert "--dependency=afterok:9009" in calls[-1][0]
    assert len(calls) == 10


def test_campaign_status_reports_prepared_campaign_without_mutation(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    before = preparation.manifest.read_bytes()

    status = campaign_status(preparation.output_dir)

    assert status.state == "prepared"
    assert status.completed_generations == 0
    assert status.generation == 1
    assert status.stage == "train"
    assert status.next_action.endswith("--submit")
    assert [generation["state"] for generation in status.generations] == [
        "not_started",
        "not_started",
        "not_started",
    ]
    assert status.generations[0]["plan"]["dft_budget"] == 6
    assert {job["state"] for job in status.jobs} == {"NOT_SUBMITTED"}
    assert preparation.manifest.read_bytes() == before


def test_campaign_status_summarizes_scientific_progress_without_mutation(
    tmp_path: Path,
):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    plan = json.loads(preparation.plans[0].read_text(encoding="utf-8"))
    plan_hash = hashlib.sha256(
        json.dumps(
            plan, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    ledger = {
        "version": 1,
        "campaign_id": preparation.campaign_id,
        "generations": {
            "1": {
                "plan_sha256": plan_hash,
                "stages": {
                    "train": {"metrics": {"training_count": 5}},
                    "explore": {
                        "metrics": {
                            "candidate_count": 8,
                            "candidate_counts_by_window": {
                                "stable_prefix": 6,
                                "pre_failure": 2,
                            },
                            "scheduled_source_count": 2,
                            "completed_source_count": 1,
                            "failed_source_count": 1,
                            "scenario_steps": [10, 40],
                            "scenario_targets": ["smoke_passed", "short_stable"],
                        }
                    },
                    "select": {
                        "metrics": {
                            "candidate_count_before_thinning": 8,
                            "candidate_count_after_thinning": 5,
                            "duplicate_candidate_count": 1,
                            "selected_count": 2,
                            "counts_by_stratum": {"stable": 1, "failure": 1},
                        }
                    },
                    "label": {"metrics": {"labeled_count": 2}},
                    "diagnose": {
                        "metrics": {
                            "current_model_energy_rmse": 0.2,
                            "current_model_force_rmse": 0.3,
                            "current_model_mforce_rmse": 0.4,
                        }
                    },
                    "merge": {
                        "metrics": {"training_count": 7, "added_count": 2}
                    },
                    "retrain": {"metrics": {"training_count": 7}},
                    "evaluate": {
                        "metrics": {
                            "accepted": True,
                            "added_training_count": 2,
                            "energy_rmse": 0.1,
                            "force_rmse": 0.15,
                            "virial_rmse": 0.2,
                            "mforce_rmse": 0.25,
                            "evaluated_count": 6,
                            "spin_frame_count": 6,
                            "scenario_counts_by_maturity": {
                                "short_stable": 1,
                                "untested": 1,
                            },
                        }
                    },
                },
                "complete": True,
                "accepted": True,
            }
        },
    }
    state_dir = preparation.output_dir / "state"
    state_dir.mkdir()
    ledger_path = state_dir / "campaign-ledger.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    before = ledger_path.read_bytes()

    status = campaign_status(preparation.output_dir)

    first = status.generations[0]
    assert first["state"] == "accepted"
    assert first["sampling"]["candidate_counts_by_window"] == {
        "stable_prefix": 6,
        "pre_failure": 2,
    }
    assert first["sampling"]["selected_count"] == 2
    assert first["training"] == {
        "before_count": 5,
        "merged_count": 7,
        "after_count": 7,
        "added_count": 2,
    }
    assert first["quality"]["acquisition_rmse"]["mforce_rmse"] == 0.4
    assert first["quality"]["validation_rmse"]["mforce_rmse"] == 0.25
    assert first["scenarios"]["counts_by_maturity"] == {
        "short_stable": 1,
        "untested": 1,
    }
    assert status.generations[1]["state"] == "not_started"
    assert ledger_path.read_bytes() == before


def test_campaign_status_cli_is_human_readable(tmp_path: Path, capsys):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")

    run_campaign_command(
        SimpleNamespace(
            status=True,
            json=False,
            retry_failed=False,
            submit=False,
            output=str(preparation.output_dir),
            config_path=None,
            initial_training=None,
            campaign_id=None,
        )
    )

    output = capsys.readouterr().out
    assert "State: prepared" in output
    assert "Ledger: generation 1, stage train" in output
    assert "Science:" in output
    assert "G1 not started: plan 12 candidates, DFT budget 6" in output
    assert "generation-1-bootstrap.sbatch" in output
    assert "Next: NepTrain campaign" in output


def test_campaign_cli_submits_prepared_output_without_config(
    tmp_path: Path, monkeypatch, capsys
):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    received = []

    def submit(output):
        received.append(output)
        return SimpleNamespace(
            campaign_id=preparation.campaign_id,
            job_ids=("9001",),
            manifest=preparation.manifest,
        )

    monkeypatch.setattr(campaign_module, "submit_campaign", submit)
    run_campaign_command(
        SimpleNamespace(
            status=False,
            json=False,
            retry_failed=False,
            submit=True,
            output=str(preparation.output_dir),
            config_path=None,
            initial_training=None,
            campaign_id=None,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert received == [str(preparation.output_dir)]
    assert payload["job_ids"] == ["9001"]
    assert payload["submitted"] is True


def test_campaign_submission_recovers_job_id_after_post_sbatch_crash(
    tmp_path: Path, monkeypatch
):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    scheduled = {}
    calls = 0
    crash = True

    def resolver(record, cwd):
        return scheduled.get(record["submission_token"])

    monkeypatch.setattr(campaign_module, "_resolve_submission", resolver)

    def runner(args, cwd):
        nonlocal calls, crash
        calls += 1
        token = next(
            value.split("=", 1)[1]
            for value in args
            if value.startswith("--job-name=")
        )
        job_id = str(9000 + calls)
        scheduled[token] = job_id
        if crash:
            crash = False
            raise KeyboardInterrupt
        return job_id

    with pytest.raises(KeyboardInterrupt):
        submit_campaign(
            preparation,
            runner=runner,
        )
    interrupted = json.loads(preparation.manifest.read_text(encoding="utf-8"))
    assert interrupted["jobs"][0]["submission_state"] == "intent"
    assert "job_id" not in interrupted["jobs"][0]

    result = submit_campaign(
        preparation,
        runner=runner,
    )

    assert result.job_ids == tuple(str(job_id) for job_id in range(9001, 9011))
    assert calls == len(preparation.scripts)
    manifest = json.loads(preparation.manifest.read_text(encoding="utf-8"))
    assert manifest["jobs"][0]["job_id"] == "9001"
    assert manifest["jobs"][0]["submission_state"] == "submitted"
    assert "reconciled_at" in manifest["jobs"][0]


def test_campaign_submission_lock_prevents_concurrent_duplicate_jobs(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    calls = []
    calls_lock = threading.Lock()
    start = threading.Barrier(2)

    def runner(args, cwd):
        with calls_lock:
            calls.append((list(args), cwd))
            job_id = str(9000 + len(calls))
        time.sleep(0.002)
        return job_id

    def submit():
        start.wait()
        return submit_campaign(preparation, runner=runner)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: submit(), range(2)))

    assert results[0].job_ids == results[1].job_ids
    assert len(calls) == len(preparation.scripts)


def test_campaign_rejects_prepared_input_drift(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    output = tmp_path / "campaign"
    prepare_campaign(config, initial, output)
    initial.write_text("drifted\n", encoding="utf-8")

    with pytest.raises(CampaignError, match="preparation changed"):
        prepare_campaign(config, initial, output)


def test_campaign_rejects_runtime_setup_drift(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    output = tmp_path / "campaign"
    prepare_campaign(config, initial, output)
    (tmp_path / "cpu-env.sh").write_text("module load changed\n", encoding="utf-8")

    with pytest.raises(CampaignError, match="preparation changed"):
        prepare_campaign(config, initial, output)


def test_campaign_rejects_drift_at_submission_boundary(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    (tmp_path / "cpu-env.sh").write_text("module load changed\n", encoding="utf-8")

    with pytest.raises(CampaignError, match="artifact drifted"):
        submit_campaign(preparation, runner=lambda args, cwd: "9001")


def test_campaign_rejects_nested_slurm_submission(tmp_path: Path, monkeypatch):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    monkeypatch.setenv("SLURM_JOB_ID", "123")

    with pytest.raises(CampaignError, match="login node"):
        submit_campaign(preparation)
    manifest = json.loads(preparation.manifest.read_text(encoding="utf-8"))
    assert manifest["jobs"] == []


def test_campaign_retry_resumes_at_ledger_breakpoint_and_preserves_history(
    tmp_path: Path, monkeypatch,
):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    submit_calls = []

    def initial_runner(args, cwd):
        submit_calls.append((list(args), cwd))
        return str(9000 + len(submit_calls))

    submit_campaign(preparation, runner=initial_runner)
    plan = json.loads(preparation.plans[0].read_text(encoding="utf-8"))
    ledger = {
        "version": 1,
        "campaign_id": preparation.campaign_id,
        "generations": {
            "1": {
                "plan_sha256": hashlib.sha256(
                    json.dumps(
                        plan, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode()
                ).hexdigest(),
                "stages": {
                    "train": {},
                    "explore": {},
                    "select": {},
                    "label": {},
                    "diagnose": {},
                    "merge": {},
                },
            }
        },
    }
    state_dir = preparation.output_dir / "state"
    state_dir.mkdir()
    (state_dir / "campaign-ledger.json").write_text(
        json.dumps(ledger), encoding="utf-8"
    )

    retry_calls = []
    canceled = []

    def retry_runner(args, cwd):
        retry_calls.append((list(args), cwd))
        return str(10000 + len(retry_calls))

    def state_runner(job_id, cwd):
        return "FAILED" if job_id == "9002" else "PENDING"

    monkeypatch.setattr(campaign_module, "_job_state", state_runner)
    manifest_before_status = preparation.manifest.read_bytes()
    ledger_path = state_dir / "campaign-ledger.json"
    ledger_before_status = ledger_path.read_bytes()
    status = campaign_status(preparation.output_dir)

    assert status.state == "blocked"
    assert status.generation == 1
    assert status.stage == "retrain"
    assert "failed job 9002" in status.reason
    assert status.next_action.endswith("--retry-failed")
    assert preparation.manifest.read_bytes() == manifest_before_status
    assert ledger_path.read_bytes() == ledger_before_status

    result = retry_failed_campaign(
        preparation.output_dir,
        runner=retry_runner,
        state_runner=state_runner,
        cancel_runner=lambda job_id, cwd: canceled.append(job_id),
    )

    assert result.from_generation == 1
    assert result.from_stage == "retrain"
    assert retry_calls[0][0][-1].endswith("generation-1-retrain.sbatch")
    assert len(retry_calls) == 8
    assert canceled == [str(job_id) for job_id in range(9003, 9011)]
    manifest = json.loads(preparation.manifest.read_text(encoding="utf-8"))
    assert [record["job_id"] for record in manifest["jobs"]] == [
        str(job_id) for job_id in range(9001, 9011)
    ]
    assert manifest["retries"][0]["from_stage"] == "retrain"
    assert [record["job_id"] for record in manifest["retries"][0]["jobs"]] == [
        str(job_id) for job_id in range(10001, 10009)
    ]


def test_campaign_retry_is_idempotent_while_recovery_chain_is_active(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    count = 0

    def initial_runner(args, cwd):
        nonlocal count
        count += 1
        return str(9000 + count)

    submit_campaign(preparation, runner=initial_runner)
    retry_calls = []

    def retry_runner(args, cwd):
        retry_calls.append(list(args))
        return str(10000 + len(retry_calls))

    def first_state(job_id, cwd):
        return "FAILED" if job_id == "9001" else "PENDING"

    first = retry_failed_campaign(
        preparation.output_dir,
        runner=retry_runner,
        state_runner=first_state,
        cancel_runner=lambda job_id, cwd: None,
    )
    second = retry_failed_campaign(
        preparation.output_dir,
        runner=retry_runner,
        state_runner=lambda job_id, cwd: "PENDING",
        cancel_runner=lambda job_id, cwd: None,
    )

    assert second.job_ids == first.job_ids
    assert len(retry_calls) == len(preparation.scripts)


def test_campaign_retry_recovers_job_id_after_post_sbatch_crash(
    tmp_path: Path, monkeypatch
):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    initial_count = 0

    def initial_runner(args, cwd):
        nonlocal initial_count
        initial_count += 1
        return str(9000 + initial_count)

    submit_campaign(preparation, runner=initial_runner)
    scheduled = {}
    retry_calls = 0
    crash = True

    def resolver(record, cwd):
        return scheduled.get(record["submission_token"])

    monkeypatch.setattr(campaign_module, "_resolve_submission", resolver)

    def retry_runner(args, cwd):
        nonlocal retry_calls, crash
        retry_calls += 1
        token = next(
            value.split("=", 1)[1]
            for value in args
            if value.startswith("--job-name=")
        )
        job_id = str(10000 + retry_calls)
        scheduled[token] = job_id
        if crash:
            crash = False
            raise KeyboardInterrupt
        return job_id

    def state_runner(job_id, cwd):
        return "FAILED" if job_id == "9001" else "PENDING"

    arguments = {
        "runner": retry_runner,
        "state_runner": state_runner,
        "cancel_runner": lambda job_id, cwd: None,
    }
    with pytest.raises(KeyboardInterrupt):
        retry_failed_campaign(preparation.output_dir, **arguments)

    result = retry_failed_campaign(preparation.output_dir, **arguments)

    assert result.job_ids == tuple(str(job_id) for job_id in range(10001, 10011))
    assert retry_calls == len(preparation.scripts)
    manifest = json.loads(preparation.manifest.read_text(encoding="utf-8"))
    assert len(manifest["retries"]) == 1
    assert manifest["retries"][0]["jobs"][0]["job_id"] == "10001"
    assert "reconciled_at" in manifest["retries"][0]["jobs"][0]


def test_campaign_retry_lock_prevents_concurrent_recovery_chains(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    submit_count = 0

    def initial_runner(args, cwd):
        nonlocal submit_count
        submit_count += 1
        return str(9000 + submit_count)

    submit_campaign(preparation, runner=initial_runner)
    retry_calls = []
    calls_lock = threading.Lock()
    start = threading.Barrier(2)

    def retry_runner(args, cwd):
        with calls_lock:
            retry_calls.append(list(args))
            job_id = str(10000 + len(retry_calls))
        time.sleep(0.002)
        return job_id

    def state_runner(job_id, cwd):
        return "FAILED" if job_id == "9001" else "PENDING"

    def retry():
        start.wait()
        return retry_failed_campaign(
            preparation.output_dir,
            runner=retry_runner,
            state_runner=state_runner,
            cancel_runner=lambda job_id, cwd: None,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: retry(), range(2)))

    assert results[0].job_ids == results[1].job_ids
    assert len(retry_calls) == len(preparation.scripts)


def test_campaign_retry_requires_submitted_job_history(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")

    with pytest.raises(CampaignError, match="use campaign --submit"):
        retry_failed_campaign(preparation.output_dir)


def test_campaign_retry_rejects_completed_campaign(tmp_path: Path):
    config, initial = _inputs(tmp_path)
    preparation = prepare_campaign(config, initial, tmp_path / "campaign")
    plans = [json.loads(path.read_text(encoding="utf-8")) for path in preparation.plans]
    generations = {}
    for plan in plans:
        plan_hash = hashlib.sha256(
            json.dumps(
                plan, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        generations[str(plan["generation"])] = {
            "plan_sha256": plan_hash,
            "stages": {
                stage: {}
                for stage in (
                    "train",
                    "explore",
                    "select",
                    "label",
                    "diagnose",
                    "merge",
                    "retrain",
                    "evaluate",
                )
            },
            "complete": True,
            "accepted": True,
        }
    state_dir = preparation.output_dir / "state"
    state_dir.mkdir()
    (state_dir / "campaign-ledger.json").write_text(
        json.dumps(
            {
                "version": 1,
                "campaign_id": preparation.campaign_id,
                "generations": generations,
            }
        ),
        encoding="utf-8",
    )

    status = campaign_status(preparation.output_dir)
    assert status.state == "complete"
    assert status.completed_generations == len(plans)
    assert status.generation is None
    assert status.stage is None
    assert status.next_action is None

    with pytest.raises(CampaignError, match="already complete"):
        retry_failed_campaign(preparation.output_dir)
