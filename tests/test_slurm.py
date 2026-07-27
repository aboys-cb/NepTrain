from __future__ import annotations

import subprocess

from NepTrain.core.slurm import (
    SlurmScript,
    SlurmSubmissionThrottled,
    aggregate_states,
    parse_submission_job_id,
    query_job,
    render_script,
    setup_line,
    submit_job,
    submission_is_throttled,
)


def test_shared_script_renderer_owns_resources_environment_and_array():
    script = render_script(
        SlurmScript(
            job_name="nt-example",
            output_path="/work/logs/%A_%a.out",
            workdir="/work/run with spaces",
            command='neptrain manual-worker /work/run "$SLURM_ARRAY_TASK_ID"',
            partition="cpu",
            time_limit="02:00:00",
            qos="normal",
            cpus_per_task=8,
            gpus_per_node=1,
            directives=("--exclusive", "#SBATCH --mem=16G"),
            environment={"NEP_MODE": "value with spaces"},
            setup_line="source /work/setup.sh",
            array="0,2,4%2",
        )
    )

    assert "#SBATCH --array=0,2,4%2" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --gpus-per-node=1" in script
    assert "#SBATCH --exclusive" in script
    assert "#SBATCH --mem=16G" in script
    assert "cd '/work/run with spaces'" in script
    assert 'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"' in script
    assert "export NEP_MODE='value with spaces'" in script
    assert script.endswith("\n")


def test_shared_query_aggregates_array_failure_from_accounting():
    calls = []

    def run(command):
        calls.append(list(command))
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(
            command,
            0,
            "COMPLETED|0:0|\nOUT_OF_MEMORY|0:125|\n",
            "",
        )

    result = query_job(run, "123")

    assert result.state == "OUT_OF_MEMORY"
    assert result.exit_code == "0:125"
    assert result.source == "sacct"
    assert calls[1] == [
        "sacct",
        "--noheader",
        "-X",
        "--parsable2",
        "--jobs",
        "123",
        "--format",
        "State,ExitCode",
    ]


def test_shared_state_and_submission_policy_cover_site_annotations():
    assert aggregate_states(["COMPLETED+", "COMPLETED"]) == "COMPLETED"
    assert aggregate_states(["CANCELLED by 1000", "CANCELLED"]) == "CANCELLED"
    assert parse_submission_job_id("701234;cluster\n") == "701234"
    assert parse_submission_job_id("Submitted batch job 701234") is None
    assert submission_is_throttled(
        "sbatch: error: QOSMaxSubmitJobPerUserLimit"
    )


def test_shared_setup_line_uses_packaged_remote_copy(tmp_path):
    setup = tmp_path / "setup.sh"
    setup.write_text("module load nep\n", encoding="utf-8")

    assert setup_line(str(setup), local=True) == f"source {setup.resolve()}"
    assert setup_line(
        str(setup),
        local=False,
        packaged_remote_path="/remote/task/target-setup.sh",
    ) == "source /remote/task/target-setup.sh"


def test_shared_submission_parses_cluster_suffix_and_classifies_throttling():
    def accepted(command):
        return subprocess.CompletedProcess(
            command,
            0,
            "701234;cluster\n",
            "",
        )

    assert submit_job(accepted, "job.sbatch") == "701234"

    def throttled(command):
        return subprocess.CompletedProcess(
            command,
            1,
            "",
            "sbatch: error: QOSMaxSubmitJobPerUserLimit",
        )

    try:
        submit_job(throttled, "job.sbatch")
    except SlurmSubmissionThrottled as error:
        assert "QOSMaxSubmitJobPerUserLimit" in str(error)
    else:  # pragma: no cover - assertion explains the expected exception
        raise AssertionError("submission throttling was not classified")
