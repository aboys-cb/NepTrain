"""Shared Slurm policy for workflow and manual execution paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shlex
import subprocess
from typing import Callable, Mapping, Sequence


ACTIVE_STATES = frozenset(
    {
        "CONFIGURING",
        "COMPLETING",
        "PENDING",
        "REQUEUED",
        "RESIZING",
        "RUNNING",
        "STAGE_OUT",
        "SUSPENDED",
    }
)
FAILURE_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)
SUCCESS_STATES = frozenset({"COMPLETED"})


@dataclass(frozen=True)
class SlurmScript:
    """Declarative input for one batch script."""

    job_name: str
    output_path: str
    workdir: str
    command: str
    partition: str
    time_limit: str
    qos: str | None = None
    cpus_per_task: int | None = None
    gpus_per_node: int | None = None
    directives: Sequence[str] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    setup_line: str | None = None
    array: str | None = None


@dataclass(frozen=True)
class SlurmQuery:
    """Normalized scheduler observation for one job or job array."""

    state: str | None
    exit_code: str = ""
    source: str | None = None
    error: str | None = None


SchedulerRunner = Callable[
    [Sequence[str]],
    subprocess.CompletedProcess,
]


class SlurmSubmissionError(RuntimeError):
    """Raised when ``sbatch`` rejects or cannot identify a submission."""

    def __init__(self, detail: str, *, accepted: bool = False):
        super().__init__(detail)
        self.accepted = accepted


class SlurmSubmissionThrottled(SlurmSubmissionError):
    """Raised when scheduler policy temporarily limits submissions."""


def normalize_state(value: str) -> str:
    """Normalize site annotations such as ``COMPLETED+`` and ``CANCELLED by``."""

    state = value.strip().split("|", 1)[0]
    return re.split(r"[+\s]", state, maxsplit=1)[0].upper()


def aggregate_states(states: Sequence[str]) -> str | None:
    """Aggregate Slurm array rows without hiding terminal failures."""

    values = [normalize_state(value) for value in states if value.strip()]
    if not values:
        return None
    if any(value in {"RUNNING", "COMPLETING"} for value in values):
        return "RUNNING"
    if any(value in ACTIVE_STATES for value in values):
        return "PENDING"
    failures = [value for value in values if value in FAILURE_STATES]
    if failures:
        if len(failures) == len(values) and all(
            value == "CANCELLED" for value in failures
        ):
            return "CANCELLED"
        return failures[0]
    if all(value in SUCCESS_STATES for value in values):
        return "COMPLETED"
    return values[0]


def render_script(spec: SlurmScript) -> str:
    """Render common resource, environment, setup, and execution semantics."""

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={spec.job_name}",
        f"#SBATCH --output={spec.output_path}",
        f"#SBATCH --time={spec.time_limit}",
        f"#SBATCH --partition={spec.partition}",
    ]
    if spec.array is not None:
        lines.append(f"#SBATCH --array={spec.array}")
    if spec.qos:
        lines.append(f"#SBATCH --qos={spec.qos}")
    if spec.cpus_per_task is not None:
        lines.append(f"#SBATCH --cpus-per-task={spec.cpus_per_task}")
    if spec.gpus_per_node is not None:
        lines.append(f"#SBATCH --gpus-per-node={spec.gpus_per_node}")
    lines.extend(
        directive
        if directive.startswith("#SBATCH ")
        else f"#SBATCH {directive}"
        for directive in spec.directives
    )
    lines.extend(["", "set -eo pipefail", f"cd {shlex.quote(spec.workdir)}"])
    if spec.setup_line:
        lines.append(spec.setup_line)
    if spec.cpus_per_task is not None:
        lines.append('export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"')
    for key, value in spec.environment.items():
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.extend([spec.command, ""])
    return "\n".join(lines)


def setup_line(
    setup_script: str | None,
    *,
    local: bool,
    packaged_remote_path: str | None = None,
) -> str | None:
    """Resolve a target setup script consistently for local and remote jobs."""

    if not setup_script:
        return None
    candidate = Path(setup_script).expanduser()
    if candidate.is_file():
        source = (
            str(candidate.resolve())
            if local
            else packaged_remote_path or setup_script
        )
    else:
        source = setup_script
    if source.startswith("~/"):
        return f'source "$HOME"/{shlex.quote(source[2:])}'
    return f"source {shlex.quote(source)}"


def submission_is_throttled(detail: str) -> bool:
    """Recognize common scheduler job-count throttling diagnostics."""

    normalized = re.sub(r"[^a-z0-9]+", " ", detail.lower())
    compact = normalized.replace(" ", "")
    markers = (
        "qosmaxsubmitjobperuserlimit",
        "assocmaxsubmitjoblimit",
        "qosmaxjobsperuserlimit",
        "assocmaxjobsperuserlimit",
        "maximum number of jobs",
        "max number of jobs",
        "job submit limit",
        "job submission limit",
        "too many jobs",
    )
    return any(
        marker in normalized or marker.replace(" ", "") in compact
        for marker in markers
    )


def parse_submission_job_id(output: str) -> str | None:
    """Return the numeric job id from ``sbatch --parsable`` output."""

    value = output.strip().split(";", 1)[0]
    return value if value.isdigit() else None


def submit_job(run: SchedulerRunner, script_name: str) -> str:
    """Submit one script and return its numeric id with shared diagnostics."""

    completed = run(["sbatch", "--parsable", script_name])
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        if submission_is_throttled(detail):
            raise SlurmSubmissionThrottled(detail)
        raise SlurmSubmissionError(detail)
    job_id = parse_submission_job_id(completed.stdout)
    if job_id is None:
        raise SlurmSubmissionError(
            f"Slurm accepted an unparseable submission: "
            f"{completed.stdout.strip()}",
            accepted=True,
        )
    return job_id


def query_job(run: SchedulerRunner, job_id: str | None) -> SlurmQuery:
    """Query ``squeue`` then ``sacct`` through an injected command adapter."""

    if not job_id or not str(job_id).isdigit():
        return SlurmQuery(None)
    selected_id = str(job_id)
    queue = run(
        [
            "squeue",
            "--noheader",
            "--jobs",
            selected_id,
            "--format",
            "%T",
        ]
    )
    if queue.returncode == 0:
        state = aggregate_states(queue.stdout.splitlines())
        if state is not None:
            return SlurmQuery(state, source="squeue")

    account = run(
        [
            "sacct",
            "--noheader",
            "-X",
            "--parsable2",
            "--jobs",
            selected_id,
            "--format",
            "State,ExitCode",
        ]
    )
    if account.returncode == 0:
        rows = [
            line.split("|")
            for line in account.stdout.splitlines()
            if line.strip()
        ]
        state = aggregate_states([row[0] for row in rows])
        if state is not None:
            exits = [row[1] for row in rows if len(row) > 1 and row[1]]
            exit_code = next(
                (value for value in exits if not value.startswith("0:0")),
                exits[0] if exits else "",
            )
            return SlurmQuery(state, exit_code, "sacct")
    detail = (
        account.stderr.strip()
        or queue.stderr.strip()
        or "job is absent from squeue and sacct"
    )
    return SlurmQuery(None, source="sacct", error=detail)


__all__ = [
    "ACTIVE_STATES",
    "FAILURE_STATES",
    "SUCCESS_STATES",
    "SlurmQuery",
    "SlurmScript",
    "SlurmSubmissionError",
    "SlurmSubmissionThrottled",
    "aggregate_states",
    "normalize_state",
    "parse_submission_job_id",
    "query_job",
    "render_script",
    "setup_line",
    "submit_job",
    "submission_is_throttled",
]
