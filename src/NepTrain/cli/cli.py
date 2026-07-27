#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 14:33
# @Author  : 兵
# @email    : 1747193328@qq.com
import argparse
import json
import os
from pathlib import Path
import shlex
from NepTrain import __version__
from NepTrain.core.config import (
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_STRUCTURES_PER_LABEL_JOB,
    DEFAULT_STRUCTURES_PER_MODEL_JOB,
)


def _print_json(value):
    try:
        payload = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SystemExit(
            f"NepTrain: error: cannot serialize stable JSON output: {error}"
        ) from error
    print(payload)


def init_template(args):
    from NepTrain.core.template import init_template as implementation
    return implementation(args)


def run_perturb(args):
    from NepTrain.core.perturb import run_perturb as implementation
    return implementation(args)


def run_select(args):
    from NepTrain.core.select import run_select as implementation
    return implementation(args)


def run_smoke_command(args):
    from dataclasses import asdict

    from NepTrain.core.smoke import run_smoke

    report = run_smoke(
        args.output,
        profile=args.profile,
        seed=args.seed,
        max_selected=args.max_selected,
        force=args.force,
    )
    result = {"protocol": "neptrain.smoke.v1", "smoke": asdict(report)}
    if args.iterations:
        from NepTrain.core.toy_iteration import run_toy_iteration_smoke

        iteration = run_toy_iteration_smoke(
            Path(args.output) / "iteration",
            profile="spin" if args.profile == "recovery" else args.profile,
            generations=args.iterations,
            seed=args.seed,
            max_selected=args.max_selected,
            force=args.force,
        )
        result["iteration"] = asdict(iteration)
    _print_json(result)


def _science_value(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _print_scientific_progress(generations):
    print("Science:")
    for generation in generations:
        number = generation["generation"]
        state = generation["state"].replace("_", " ")
        plan = generation["plan"]
        if generation["state"] == "not_started":
            print(
                f"  G{number} {state}: FPS selects up to "
                f"{plan['max_selected']}; MD conditions come from "
                "sampling routes"
            )
            continue

        sampling = generation["sampling"]
        training = generation["training"]
        flow = []
        if sampling["candidate_count"] is not None:
            flow.append(f"MD {sampling['candidate_count']} candidates")
        if sampling["candidate_count_after_deduplication"] is not None:
            flow.append(
                f"{sampling['candidate_count_after_deduplication']} unique eligible"
            )
        if sampling["selected_count"] is not None:
            flow.append(f"FPS {sampling['selected_count']}")
        if sampling["labeled_count"] is not None:
            flow.append(f"labels {sampling['labeled_count']}")
        before = training["before_count"]
        after = training["after_count"]
        if before is not None or after is not None:
            training_text = f"training {_science_value(before)}"
            if after is not None:
                training_text += f" -> {after}"
            if training["added_count"] is not None:
                training_text += f" (+{training['added_count']})"
            flow.append(training_text)
        if not flow:
            completed = ", ".join(generation["completed_stages"]) or "none"
            flow.append(f"completed stages: {completed}")
        print(f"  G{number} {state}: " + " -> ".join(flow))
        windows = sampling["candidate_counts_by_window"]
        duplicates = sampling["duplicate_candidate_count"]
        if windows or duplicates is not None:
            details = [
                f"{name}={count}" for name, count in sorted(windows.items())
            ]
            if duplicates is not None:
                details.append(f"duplicates={duplicates}")
            print("    sampling: " + ", ".join(details))
        batch_kind = sampling.get("batch_kind")
        batch_floor = sampling.get("regular_batch_minimum")
        if batch_kind:
            print(
                f"    batch: {batch_kind}, regular floor={batch_floor}, "
                f"active model={str(sampling.get('sampling_model_sha256') or '-')[:12]}"
            )
        active_model = training.get("active_model_sha256")
        if active_model:
            action = (
                "updated"
                if training.get("model_updated")
                else "reused for certification"
            )
            print(f"    model: {str(active_model)[:12]} ({action})")
        validation = generation["quality"]["validation_rmse"]
        if any(value is not None for value in validation.values()):
            print(
                "    validation RMSE: "
                f"E={_science_value(validation['energy_rmse'])}, "
                f"F={_science_value(validation['force_rmse'])}, "
                f"V={_science_value(validation['virial_rmse'])}, "
                f"M={_science_value(validation['mforce_rmse'])}"
            )
        maturity = generation["scenarios"]["counts_by_maturity"]
        if maturity:
            print(
                "    scenario maturity: "
                + ", ".join(
                    f"{name}={count}" for name, count in sorted(maturity.items())
                )
            )


def _print_workflow_status(status, *, show_jobs: bool = True):
    print(f"Workflow: {status.workflow_id}")
    print(f"State: {status.state}")
    print(
        f"Progress: {status.completed_generations}/{status.total_generations} "
        "model generations"
    )
    if status.generation is not None:
        print(f"Ledger: generation {status.generation}, stage {status.stage}")
    else:
        print("Ledger: complete")
    print(f"Reason: {status.reason}")
    _print_scientific_progress(status.generations)
    if show_jobs:
        print("Executions:")
        for job in status.jobs:
            marker = "*" if job["current"] else "-"
            attempt = job["attempt"] or "not-submitted"
            job_id = job["job_id"] or "-"
            print(
                f"  {marker} {attempt:13} {job_id:>8} "
                f"{job['state']:>20}  {Path(job['script']).name}"
            )
    else:
        active = [
            job
            for job in status.jobs
            if job["current"] and job["state"] not in {"NOT_SUBMITTED", "COMPLETED"}
        ]
        if active:
            job = active[-1]
            print(
                f"Executor: {job['job_id'] or '-'} {job['state']} "
                f"({Path(job['script']).name})"
            )
        else:
            completed = sum(job["state"] == "COMPLETED" for job in status.jobs)
            print(f"Executor: {completed}/{len(status.jobs)} stages completed")
    if status.next_action:
        print(f"Next: {status.next_action}")


def run_project_command(args):
    """Start from either a project YAML file or a prepared workflow."""

    from NepTrain.core.workflow import (
        WorkflowError,
        prepare_workflow,
        start_workflow,
    )
    from NepTrain.core.controller import start_controller

    project = Path(args.project).expanduser()
    try:
        if project.is_dir():
            invalid = [
                option
                for option, value in (
                    ("--initial-training", args.initial_training),
                    ("--output", args.output),
                    ("--workflow-id", args.workflow_id),
                    ("--prepare-only", args.prepare_only),
                )
                if value
            ]
            if invalid:
                raise WorkflowError(
                    f"{', '.join(invalid)} can only be used with a project "
                    "YAML file"
                )
            result = start_workflow(
                project,
                foreground=getattr(args, "foreground", False),
                poll_interval=getattr(args, "poll_interval", None),
            )
            payload = _workflow_resume_payload(result)
        else:
            initial_training = args.initial_training
            output = args.output
            if not initial_training or not output:
                from NepTrain.core.config import ConfigError, load_config

                try:
                    config, _ = load_config(project)
                except ConfigError as error:
                    raise WorkflowError(
                        f"invalid project configuration: {error}"
                    ) from error
                value = config.get("training", {}).get("initial_path")
                if value:
                    path = Path(value).expanduser()
                    initial_training = str(
                        (project.parent / path).resolve()
                        if not path.is_absolute()
                        else path.resolve()
                    )
                if not output:
                    workflow_id = str(
                        config.get("workflow", {}).get("id", "")
                    ).strip()
                    if workflow_id:
                        output = str((project.parent / workflow_id).resolve())
            if not initial_training or not output:
                raise WorkflowError(
                    "starting from a project file requires "
                    "training.initial_path (or --initial-training) and "
                    "workflow.id (or --output)"
                )
            preparation = prepare_workflow(
                project,
                initial_training,
                output,
                workflow_id=args.workflow_id,
            )
            payload = {
                "protocol": "neptrain.workflow-control.v1",
                "workflow_id": preparation.workflow_id,
                "action": "prepare" if args.prepare_only else "start",
                "project": str(preparation.output_dir),
                "manifest": str(preparation.manifest),
            }
            if args.prepare_only:
                payload["next_action"] = (
                    f"neptrain workflow run "
                    f"{shlex.quote(str(preparation.output_dir))}"
                )
            else:
                try:
                    controller_result = start_controller(
                        preparation.output_dir,
                        foreground=getattr(args, "foreground", False),
                        poll_interval=getattr(args, "poll_interval", None),
                    )
                except Exception as error:
                    raise WorkflowError(str(error)) from error
                if getattr(args, "foreground", False):
                    payload["controller_exit_code"] = controller_result
                else:
                    payload["controller_pid"] = controller_result
    except WorkflowError:
        raise
    _print_json(payload)


def run_status_command(args):
    from dataclasses import asdict
    from NepTrain.core.workflow import WorkflowError, workflow_status

    try:
        status = workflow_status(args.project)
    except WorkflowError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    if args.json:
        _print_json(
            {
                "protocol": "neptrain.workflow-status.v1",
                **asdict(status),
            }
        )
    else:
        _print_workflow_status(status, show_jobs=args.jobs)


def run_resume_command(args):
    from NepTrain.core.workflow import WorkflowError, resume_workflow

    try:
        result = resume_workflow(args.project)
    except WorkflowError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    payload = _workflow_resume_payload(result)
    _print_json(payload)


def _workflow_resume_payload(result):
    from dataclasses import asdict
    from NepTrain.core.workflow_workspace import WorkflowWorkspace

    payload = asdict(result)
    payload["protocol"] = "neptrain.workflow-control.v1"
    payload["manifest"] = str(result.manifest)
    payload["project"] = str(WorkflowWorkspace.locate(result.manifest).root)
    if result.controller_pid is not None:
        payload.pop("controller_exit_code", None)
    elif result.controller_exit_code is not None:
        payload.pop("controller_pid", None)
    else:
        payload.pop("controller_pid", None)
        payload.pop("controller_exit_code", None)
    return payload


def run_extend_command(args):
    from NepTrain.core.workflow import WorkflowError, extend_workflow

    try:
        preparation = extend_workflow(args.project, args.generations)
    except WorkflowError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    _print_json(
        {
            "protocol": "neptrain.workflow-extend.v1",
            "workflow_id": preparation.workflow_id,
            "total_model_generations": len(preparation.plans),
            "project": str(preparation.output_dir),
        }
    )


def run_stop_command(args):
    from NepTrain.core.controller import ControllerError, stop_workflow

    try:
        result = stop_workflow(
            args.project, cancel_jobs=bool(args.cancel_jobs)
        )
    except ControllerError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    _print_json(
        {
            "protocol": "neptrain.workflow-stop.v1",
            **result,
        }
    )


def run_controller_command(args):
    from NepTrain.core.controller import run_controller

    return run_controller(args.project, poll_interval=args.poll_interval)


def run_stage_worker_command(args):
    from NepTrain.core.execution import run_stage_worker

    return run_stage_worker(args.bundle)


def run_stage_verify_command(args):
    from NepTrain.core.execution import verify_stage_task

    verify_stage_task(args.bundle)
    return 0


def _doctor_resource_contract(config, project):
    """Resolve the authoritative resource contract for a labeling Adapter."""

    backend = str(config.get("labeling", {}).get("backend", "vasp"))
    if backend in {"model", "toy"}:
        return None
    labeling = config["labeling"]
    execution = config["execution"]
    target_name = str(execution["stage_targets"]["labeling"])
    target = execution["targets"][target_name]
    resource_root = (
        target.get("labeling_resource_path")
        or labeling.get("resource_path")
    )
    if not resource_root:
        raise ValueError("labeling Adapter has no configured resource root")
    if not target.get("labeling_resource_path"):
        candidate = Path(str(resource_root)).expanduser()
        if not candidate.is_absolute():
            resource_root = str((project.parent / candidate).resolve())
    if backend == "vasp":
        from NepTrain.core.dft.vasp.resources import vasp_resource_files

        manifest = Path(
            str(labeling["potcar_manifest_path"])
        ).expanduser()
        label = "VASP POTCAR"
        records = vasp_resource_files(
            manifest if manifest.is_absolute() else project.parent / manifest
        )
    elif backend == "abacus":
        from NepTrain.core.dft.abacus.resources import abacus_resource_files

        manifest = Path(
            str(labeling["resource_manifest_path"])
        ).expanduser()
        label = "ABACUS pseudopotential/orbital"
        records = abacus_resource_files(
            manifest if manifest.is_absolute() else project.parent / manifest
        )
    else:  # validated schema owns this invariant
        raise ValueError(f"unsupported labeling backend: {backend}")
    return target_name, str(resource_root), label, records


def _doctor_resource_probe(resource_root, records):
    lines = [
        "set -eo pipefail",
        f"resource_root={shlex.quote(str(resource_root))}",
        'case "$resource_root" in "~/"*) '
        'resource_root="$HOME/${resource_root#~/}";; esac',
        'test -d "$resource_root" || { '
        'echo "resource root is missing: $resource_root" >&2; exit 3; }',
        "if command -v sha256sum >/dev/null; then",
        "  file_sha256() { sha256sum \"$1\" | awk '{print $1}'; }",
        "elif command -v shasum >/dev/null; then",
        "  file_sha256() { shasum -a 256 \"$1\" | awk '{print $1}'; }",
        "else",
        '  echo "sha256sum or shasum is required" >&2',
        "  exit 4",
        "fi",
    ]
    for record in records:
        relative = str(record["path"])
        expected = str(record["sha256"])
        lines.extend(
            [
                f"resource_path=\"$resource_root\"/{shlex.quote(relative)}",
                'test -f "$resource_path" || { '
                'echo "resource file is missing: $resource_path" >&2; exit 5; }',
                'actual=$(file_sha256 "$resource_path")',
                f'test "$actual" = {shlex.quote(expected)} || {{ '
                'echo "resource hash mismatch: $resource_path" >&2; exit 6; }',
            ]
        )
    return "\n".join(lines) + "\n"


def run_doctor(args):
    import importlib.util
    import shutil
    import shlex
    import subprocess
    import tempfile

    failures = []
    package_status = {}
    for package in ("nep_adapters", "ase"):
        available = importlib.util.find_spec(package) is not None
        package_status[package] = available
        print(f"{'OK' if available else 'FAIL'} package {package}")
        if not available:
            failures.append(package)
    if args.training_backend == "torchnep":
        available = importlib.util.find_spec("torchnep") is not None
        print(f"{'OK' if available else 'FAIL'} package torchnep")
        if not available:
            failures.append("torchnep")
    selected_inference = args.inference_backend
    model_info = None
    if args.project:
        from NepTrain.core.config import ConfigError, load_config
        from NepTrain.core.execution import ExecutionTarget

        project = Path(args.project).expanduser().resolve()
        try:
            config, _ = load_config(project)
        except ConfigError as error:
            raise SystemExit(f"NepTrain: error: invalid project configuration: {error}") from error
        try:
            resource_contract = _doctor_resource_contract(config, project)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            raise SystemExit(
                f"NepTrain: error: invalid labeling resource contract: {error}"
            ) from error
        labeling = config.get("labeling", {})
        labeling_backend = str(labeling.get("backend", "vasp"))
        labeling_target_name = str(
            config["execution"]["stage_targets"]["labeling"]
        )
        if labeling_backend == "model":
            teacher = Path(str(labeling["model_path"])).expanduser()
            if not teacher.is_absolute():
                teacher = (project.parent / teacher).resolve()
            available = teacher.is_file()
            print(
                f"{'OK' if available else 'FAIL'} teacher model {teacher}"
            )
            if not available:
                failures.append(f"teacher model {teacher}")
        resolved_targets = {}
        for name, raw_target in config.get("execution", {}).get("targets", {}).items():
            value = dict(raw_target)
            setup = value.get("setup_script")
            setup_is_local = False
            if setup:
                candidate = Path(setup).expanduser()
                local = (project.parent / candidate).resolve() if not candidate.is_absolute() else candidate
                if local.is_file():
                    value["setup_script"] = str(local)
                    setup_is_local = True
            target = ExecutionTarget.from_mapping(str(name), value)
            resolved_targets[str(name)] = target
            required = []
            if target.executor == "slurm":
                required.extend(["sbatch", "squeue", "sacct"])
            required.append(shlex.split(target.command)[0])
            if (
                labeling_backend == "model"
                and str(name) == labeling_target_name
            ):
                required.append(shlex.split(str(labeling["runner"]))[0])
            setup_line = ""
            path_check = ""
            if target.setup_script and target.executor == "process":
                setup_path = target.setup_script
                if Path(setup_path).is_file():
                    setup_line = Path(setup_path).read_text(encoding="utf-8") + "\n"
                else:
                    setup_line = f"source {shlex.quote(setup_path)}\n"
            elif target.setup_script and not setup_is_local:
                path_check = f"test -r {shlex.quote(target.setup_script)}\n"
            probe = (
                "set -eo pipefail\n"
                + setup_line
                + path_check
                + "for tool in "
                + " ".join(shlex.quote(tool) for tool in required)
                + "; do command -v \"$tool\" >/dev/null; done\n"
            )
            command = (
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    target.host,
                    "bash",
                    "-s",
                ]
                if target.host
                else ["bash", "-s"]
            )
            try:
                completed = subprocess.run(
                    command,
                    input=probe,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                completed = subprocess.CompletedProcess(
                    command,
                    124,
                    stdout="",
                    stderr="probe timed out after 30s",
                )
            available = completed.returncode == 0
            location = target.host or "local"
            print(
                f"{'OK' if available else 'FAIL'} execution target {name} "
                f"({target.executor} on {location})"
            )
            if not available:
                failures.append(f"execution target {name}")
                detail = (completed.stderr or completed.stdout).strip()
                if detail:
                    print(f"  {detail}")
        if resource_contract is not None:
            target_name, resource_root, label, records = resource_contract
            target = resolved_targets[target_name]
            command = (
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    target.host,
                    "bash",
                    "-s",
                ]
                if target.host
                else ["bash", "-s"]
            )
            try:
                completed = subprocess.run(
                    command,
                    input=_doctor_resource_probe(resource_root, records),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                completed = subprocess.CompletedProcess(
                    command,
                    124,
                    stdout="",
                    stderr="resource probe timed out after 30s",
                )
            available = completed.returncode == 0
            location = target.host or "local"
            print(
                f"{'OK' if available else 'FAIL'} {label} resources "
                f"on {target_name} ({location})"
            )
            if not available:
                failures.append(f"{label} resources on {target_name}")
                detail = (completed.stderr or completed.stdout).strip()
                if detail:
                    print(f"  {detail}")
    if args.model and package_status["nep_adapters"]:
        from nep_adapters import inspect_model
        from NepTrain.core.nep.calculator import resolve_backend

        model_info = inspect_model(args.model)
        selected_inference = resolve_backend(args.model, args.inference_backend)
        print(
            f"OK model type={model_info.model_type} elements={','.join(model_info.elements)} "
            f"backend={selected_inference}"
        )
    if args.md_backend == "lammps":
        executable = shutil.which(args.lmp.split()[0])
        print(f"{'OK' if executable else 'FAIL'} LAMMPS executable {args.lmp}")
        if not executable:
            failures.append("lammps")
        else:
            probe_command = shlex.split(args.lmp)
            probe_command.append("-h")
            try:
                completed = subprocess.run(
                    probe_command,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                completed = subprocess.CompletedProcess(
                    probe_command,
                    124,
                    stdout="",
                    stderr="LAMMPS help probe timed out after 30s",
                )
            help_text = completed.stdout + completed.stderr
            pair = "nep/gpu/kk" if selected_inference == "cuda" else "nep/cpu"
            pair_available = pair in help_text
            print(f"{'OK' if pair_available else 'FAIL'} LAMMPS pair style {pair}")
            if not pair_available:
                failures.append(pair)
    if args.structure and args.model and args.md_backend == "lammps" and not failures:
        from ase.io import read as ase_read
        from NepTrain.core.md import MdRequest, run_md

        atoms = ase_read(args.structure, index=0)
        spin = bool(model_info and model_info.supports("spin"))
        with tempfile.TemporaryDirectory(prefix="neptrain-doctor-") as directory:
            root = Path(directory)
            run_md(
                MdRequest(
                    atoms=atoms,
                    model_file=Path(args.model),
                    output_dir=root / "run",
                    output_file=root / "trajectory.xyz",
                    temperature=300,
                    spin_temperature=300 if spin else None,
                    steps=1,
                    timestep=0.0001,
                    spin=spin,
                    inference_backend=selected_inference,
                    lmp_command=args.lmp,
                    mpiexec=args.mpiexec,
                    mpi_ranks=args.mpi_ranks,
                ),
                "lammps",
            )
        print(f"OK real LAMMPS smoke at mpi_ranks={args.mpi_ranks}")
    if failures:
        raise SystemExit("Doctor failed: " + ", ".join(failures))
    print("Doctor completed successfully.")

def build_perturb(subparsers):
    parser = subparsers.add_parser(
        "perturb",
        help="Generate perturbed structures.",
    )
    parser.set_defaults(func=run_perturb)
    parser.add_argument(
        "model_path",
        help="Structure file or directory containing XYZ/VASP structures.",
    )
    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default=20,
        help="Number of perturbations generated per input structure.",
    )
    parser.add_argument(
        "--cell-perturbation",
        "--cell",
        "-c",
        dest="cell_pert_fraction",
        type=float,
        default=0.03,
        help="Maximum cell deformation fraction, default 0.03.",
    )
    parser.add_argument(
        "--max-displacement",
        "--distance",
        "-d",
        type=float,
        dest="max_displacement",
        default=0.1,
        help="Maximum Cartesian displacement amplitude in Å, default 0.1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible perturbations, default 42.",
    )
    parser.add_argument(
        "--out",
        "-o",
        dest="out_file_path",
        default="./perturb.xyz",
        help="Output extxyz path, default ./perturb.xyz.",
    )
    parser.add_argument(
        "--append",
        "-a",
        action="store_true",
        help="Append to an existing output file.",
    )

def build_doctor(subparsers):
    parser = subparsers.add_parser("doctor", help="Check selected runtime capabilities.")
    parser.set_defaults(func=run_doctor)
    parser.add_argument("--training-backend", choices=["gpumd", "torchnep"], default="gpumd")
    parser.add_argument("--md-backend", choices=["gpumd", "lammps"], default="gpumd")
    parser.add_argument("--inference-backend", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--model", default=None)
    parser.add_argument("--structure", default=None)
    parser.add_argument("--lmp", default="lmp")
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument(
        "--project",
        default=None,
        help="Check every execution target in a project YAML.",
    )

def build_smoke(subparsers):
    parser = subparsers.add_parser(
        "smoke",
        help="Run the deterministic Toy Teacher workflow-development smoke.",
    )
    parser.set_defaults(func=run_smoke_command)
    parser.add_argument(
        "--profile", choices=["ordinary", "spin", "recovery"], default="ordinary"
    )
    parser.add_argument("--output", default="./outputs/smoke")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max-selected", type=int, default=8)
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Also run a resumable progressive Toy workflow for this many generations.",
    )
    parser.add_argument("--force", action="store_true")


def build_select(subparsers):
    parser_select = subparsers.add_parser(
        "select",
        help="Select structures manually with the production FPS policy.",
    )
    parser_select.set_defaults(func=run_select)

    parser_select.add_argument(
        "trajectory_paths",
        nargs="+",
        help="Candidate extxyz trajectory files.",
    )
    parser_select.add_argument(
        "--base",
        "-base",
        help="Optional reference extxyz dataset used to warm-start FPS.",
    )
    parser_select.add_argument(
        "--nep",
        "-nep",
        help="NEP model used for descriptors; without it SOAP is used.",
    )
    parser_select.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="NEPAdapters descriptor backend, default auto.",
    )
    parser_select.add_argument(
        "--max-selected",
        "-max",
        type=int,
        default=20,
        help="Maximum number of selected structures, default 20.",
    )
    parser_select.add_argument(
        "--min-novelty",
        "--min-distance",
        "--min_distance",
        "-d",
        type=float,
        dest="min_novelty",
        default=0.01,
        help="Strict normalized descriptor novelty threshold, default 0.01.",
    )
    parser_select.add_argument(
        "--filter",
        "-f",
        type=float,
        const=0.6,
        nargs="?",
        default=False,
        help="Reject short bonds using this covalent-radius coefficient.",
    )
    parser_select.add_argument(
        "--rejected-out",
        help="Optional extxyz output for structures rejected by --filter.",
    )
    parser_select.add_argument(
        "--out",
        "-o",
        dest="out_file_path",
        default="./selected.xyz",
        help="Selected extxyz output, default ./selected.xyz.",
    )
    parser_select.add_argument(
        "--report",
        help="Selection JSON report; defaults beside --out.",
    )

    group = parser_select.add_argument_group("SOAP parameters")
    group.add_argument("--r-cut", "--r_cut", "-r", type=float, default=6.0)
    group.add_argument("--n-max", "--n_max", "-n", type=int, default=8)
    group.add_argument("--l-max", "--l_max", "-l", type=int, default=6)

def _print_manual_status(value, *, json_output=False):
    if json_output:
        _print_json(value)
        return
    kind = value.get("kind")
    operation_id = value.get("operation_id")
    if kind or operation_id:
        task = str(kind or "task")
        if operation_id:
            task += f" ({operation_id})"
        print(f"Task: {task}")
    labels = (
        ("state", "State"),
        ("scheduler_state", "Scheduler"),
        ("job_id", "Job"),
        ("run_directory", "Run directory"),
        ("remote_directory", "Remote directory"),
        ("result", "Result"),
        ("reason", "Reason"),
    )
    for key, label in labels:
        item = value.get(key)
        if item not in (None, "", [], {}):
            print(f"{label}: {item}")
    completed = value.get("completed")
    total = value.get("total")
    if completed is not None and total is not None:
        print(f"Progress: {completed}/{total}")
    errors = value.get("errors")
    if errors:
        print("Errors:")
        for error in errors:
            if isinstance(error, dict):
                index = error.get("index")
                prefix = "collection" if index is None else f"job {index}"
                print(f"- {prefix}: {error.get('error', '')}")
            else:
                print(f"- {error}")
    logs = value.get("logs")
    if logs:
        print("Logs:")
        for log in logs:
            print(f"- {log}")
    run_directory = value.get("run_directory")
    state = str(value.get("state", "")).lower()
    next_action = value.get("next_action")
    if next_action:
        print(f"Next: {next_action}")
    elif run_directory and state in {"prepared", "submitted", "running", "unknown"}:
        print(f"Next: neptrain task wait {shlex.quote(str(run_directory))}")


def _manual_project(project):
    if not project:
        return {}, Path.cwd()
    from NepTrain.core.config import load_config

    path = Path(project).expanduser().resolve()
    config, _ = load_config(path)
    return config, path.parent


def _project_path(base, value):
    if value is None:
        return None
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def _manual_sampling_route(project, base, route_id):
    from NepTrain.core.manual import ManualTaskError

    if not project:
        if route_id:
            raise ManualTaskError("--route requires --project")
        return None
    routes = list(project.get("sampling", {}).get("routes") or [])
    if route_id:
        matches = [route for route in routes if route.get("id") == route_id]
        if not matches:
            available = ", ".join(str(route.get("id")) for route in routes)
            raise ManualTaskError(
                f"unknown sampling route {route_id!r}; available routes: {available}"
            )
        selected = dict(matches[0])
    elif len(routes) == 1:
        selected = dict(routes[0])
    else:
        raise ManualTaskError(
            "project defines multiple sampling routes; select one with --route"
        )
    selected["template_path"] = _project_path(base, selected["template_path"])
    return selected


def run_manual_train_command(args):
    from NepTrain.core.manual import (
        prepare_training,
        submit_operation,
        target_from_project,
    )

    project, base = _manual_project(args.project)
    settings = project.get("training", {})
    target = target_from_project(args.project, args.target, route="training")
    operation = prepare_training(
        args.input,
        backend=args.backend or settings.get("backend", "torchnep"),
        config_file=args.config
        or _project_path(base, settings.get("config_path"))
        or "./nep.in",
        output=args.output,
        workdir=args.workdir,
        target=target,
        test_file=args.test or _project_path(base, settings.get("test_path")),
        restart_file=args.restart,
        device=args.device or settings.get("device", "cuda"),
        torch_backend=args.torch_backend
        or settings.get("torch_backend", "auto"),
        precision=args.precision or settings.get("precision", "float32"),
        use_compile=(
            args.use_compile
            if args.use_compile is not None
            else bool(settings.get("use_compile", False))
        ),
        seed=args.seed if args.seed is not None else int(settings.get("seed", 20260723)),
        force=args.force,
    )
    _print_manual_status(
        submit_operation(
            operation, wait=args.wait, poll_interval=args.poll_interval
        ),
        json_output=args.json,
    )


def run_manual_md_command(args):
    from NepTrain.core.manual import (
        prepare_md,
        submit_operation,
        target_from_project,
    )

    project, base = _manual_project(args.project)
    settings = project.get("md", {})
    sampling = project.get("sampling", {})
    candidate_pool = sampling.get("candidate_pool", {})
    route = _manual_sampling_route(project, base, args.route)
    route_id = None if route is None else str(route["id"])
    target = target_from_project(
        args.project,
        args.target,
        route="sampling",
        sampling_route_id=route_id,
    )
    conditions = dict((route or {}).get("conditions") or {})
    from NepTrain.core.sampling_route import normalized_progression

    progression = normalized_progression((route or {}).get("progression"))
    operation = prepare_md(
        args.input,
        backend=args.backend or settings.get("backend", "lammps"),
        model_file=args.model,
        temperatures=(
            args.temperature
            if args.temperature is not None
            else conditions.get("temperature_path", [300.0])
        ),
        output=args.output,
        workdir=args.workdir,
        target=target,
        steps=(
            args.steps
            if args.steps is not None
            else progression["steps"][args.maturity]
        ),
        seed=(
            getattr(args, "seed", None)
            if getattr(args, "seed", None) is not None
            else int(project.get("workflow", {}).get("seed", 12345))
        ),
        pressure=(
            args.pressure
            if args.pressure is not None
            else float(conditions.get("pressure", 0.0))
        ),
        ensemble=args.ensemble or "nvt",
        template_path=args.template or (route or {}).get("template_path"),
        spin=args.spin
        if args.spin is not None
        else bool(settings.get("spin", False)),
        spin_temperature=args.spin_temperature
        if args.spin_temperature is not None
        else None,
        inference_backend=args.inference_backend
        or settings.get("inference_backend", "auto"),
        lmp=args.lmp or "lmp",
        mpiexec=args.mpiexec or "mpirun",
        mpi_ranks=args.mpi_ranks
        if args.mpi_ranks is not None
        else 1,
        pre_failure_frames=(
            args.pre_failure_frames
            if args.pre_failure_frames is not None
            else int(candidate_pool.get("pre_failure_frames", 2))
        ),
        bad_tail_frames=(
            args.bad_tail_frames
            if args.bad_tail_frames is not None
            else int(candidate_pool.get("bad_tail_frames", 1))
        ),
        health=dict(candidate_pool.get("health") or {}),
        max_concurrent=(
            args.max_concurrent
            if args.max_concurrent is not None
            else DEFAULT_MAX_CONCURRENT
        ),
        force=args.force,
    )
    _print_manual_status(
        submit_operation(
            operation, wait=args.wait, poll_interval=args.poll_interval
        ),
        json_output=args.json,
    )


def run_manual_label_command(args):
    from NepTrain.core.manual import (
        prepare_labeling,
        submit_operation,
        target_from_project,
    )

    project, base = _manual_project(args.project)
    settings = project.get("labeling", {})
    target = target_from_project(args.project, args.target, route="labeling")
    if args.resources is not None:
        resource_dir = args.resources
    elif target.labeling_resource_path:
        resource_dir = None
    else:
        resource_dir = _project_path(base, settings.get("resource_path"))
    if args.kspacing is not None:
        kpoint_mode = "kspacing"
        kspacing = args.kspacing
    elif args.ka is not None:
        kpoint_mode = "kpoints"
        kspacing = None
    else:
        kpoint_mode = settings.get("kpoint_mode", "auto")
        kspacing = (
            settings.get("kspacing")
            if kpoint_mode == "kspacing"
            else None
        )
    raw_ka = args.ka if args.ka is not None else settings.get("kpoints", [1, 1, 1])
    if isinstance(raw_ka, int):
        ka = [raw_ka, raw_ka, raw_ka]
    else:
        ka = list(raw_ka)
        if len(ka) == 1:
            ka *= 3
    backend = args.backend or settings.get("backend", "vasp")
    operation = prepare_labeling(
        args.input,
        backend=backend,
        output=args.output,
        workdir=args.workdir,
        target=target,
        input_file=args.dft_input
        or _project_path(base, settings.get("input_path")),
        resource_dir=resource_dir,
        resource_manifest=(
            (
                args.potcar_manifest
                or _project_path(base, settings.get("potcar_manifest_path"))
            )
            if backend == "vasp"
            else (
                args.resource_manifest
                or _project_path(base, settings.get("resource_manifest_path"))
            )
        ),
        n_cpu=args.cpus,
        use_gamma=(
            args.gamma
            if args.gamma is not None
            else bool(settings.get("gamma_centered", False))
        ),
        kpoint_mode=kpoint_mode,
        kspacing=kspacing,
        ka=ka,
        structures_per_job=(
            args.structures_per_job
            if args.structures_per_job is not None
            else int(
                settings.get(
                    "structures_per_job",
                    DEFAULT_STRUCTURES_PER_MODEL_JOB
                    if backend == "model"
                    else DEFAULT_STRUCTURES_PER_LABEL_JOB,
                )
            )
        ),
        max_concurrent=(
            args.max_concurrent
            if args.max_concurrent is not None
            else int(
                settings.get("max_concurrent", DEFAULT_MAX_CONCURRENT)
            )
        ),
        teacher_profile=args.teacher_profile or "ordinary",
        model_file=args.model
        or _project_path(base, settings.get("model_path")),
        model_name=args.model_name or settings.get("model_name"),
        runner=args.runner or settings.get("runner"),
        device=args.device or settings.get("device", "cuda"),
        precision=args.precision
        or settings.get("precision", "float32"),
        force=args.force,
    )
    _print_manual_status(
        submit_operation(
            operation, wait=args.wait, poll_interval=args.poll_interval
        ),
        json_output=args.json,
    )


def run_task_command(args):
    from NepTrain.core.manual import (
        cancel_operation,
        load_operation,
        operation_logs,
        refresh_operation,
        retry_failed,
        wait_operation,
    )

    operation = load_operation(args.run)
    if args.task_action == "status":
        value = refresh_operation(operation)
    elif args.task_action == "wait":
        value = wait_operation(operation, poll_interval=args.poll_interval)
    elif args.task_action == "retry":
        value = retry_failed(operation)
    elif args.task_action == "cancel":
        value = cancel_operation(operation)
    elif args.task_action == "logs":
        value = {
            "protocol": "neptrain.manual-logs.v1",
            "operation_id": operation.operation_id,
            "kind": operation.kind,
            "run_directory": str(operation.root),
            "logs": operation_logs(operation),
        }
    else:  # pragma: no cover - argparse owns this invariant
        raise ValueError(args.task_action)
    _print_manual_status(value, json_output=args.json)


def run_manual_worker_command(args):
    from NepTrain.core.manual import run_manual_worker

    return run_manual_worker(args.run, args.index)


def run_spin_migration_command(args):
    from NepTrain.core.spin import migrate_spin_dataset

    result = migrate_spin_dataset(
        args.input,
        args.output,
        force=args.force,
    )
    if args.json:
        _print_json(result)
    else:
        print(
            f"Migrated {result['spin_frames']}/{result['frames']} spin frames "
            f"to {result['output']}"
        )
        print(f"Removed legacy fields: {result['legacy_fields_removed']}")


def _add_execution_options(parser):
    parser.add_argument(
        "--project",
        help="Schema-v8 project providing reusable execution targets.",
    )
    parser.add_argument("--target", help="Execution target name from project.yaml.")
    parser.add_argument("--workdir", help="Durable run directory.")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for submitted Slurm work and publish the final result.",
    )
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the machine-readable task record.",
    )


def _parse_ka(value):
    values = [item.strip() for item in str(value).split(",")]
    if len(values) == 1:
        values *= 3
    if len(values) != 3 or any(not item.isdigit() for item in values):
        raise argparse.ArgumentTypeError("--ka must be one integer or x,y,z")
    return [int(item) for item in values]


def build_manual_train(subparsers):
    parser = subparsers.add_parser(
        "train", help="Train one NEP model locally or on a configured target."
    )
    parser.set_defaults(func=run_manual_train_command)
    parser.add_argument("input", help="Labeled training extxyz.")
    parser.add_argument("--backend", choices=["gpumd", "torchnep"])
    parser.add_argument("--config")
    parser.add_argument("--test")
    parser.add_argument("--restart")
    parser.add_argument("--device")
    parser.add_argument("--torch-backend", choices=["auto", "loop", "bmm"])
    parser.add_argument("--precision", choices=["float32", "float64"])
    parser.add_argument(
        "--compile",
        dest="use_compile",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", "-o", default="./nep.txt")
    parser.add_argument("--force", action="store_true")
    _add_execution_options(parser)


def build_manual_md(subparsers):
    parser = subparsers.add_parser(
        "md", help="Run GPUMD or LAMMPS over structures and temperatures."
    )
    parser.set_defaults(func=run_manual_md_command)
    parser.add_argument("input", help="Structure file, extxyz, or directory.")
    parser.add_argument("--backend", choices=["gpumd", "lammps"])
    parser.add_argument("--model", default="./nep.txt")
    parser.add_argument("--temperature", type=float, nargs="+")
    parser.add_argument("--pressure", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ensemble", choices=["nvt", "npt"])
    parser.add_argument("--template")
    parser.add_argument(
        "--route",
        help=(
            "Sampling route whose template, conditions, progression, and "
            "route-specific target provide defaults."
        ),
    )
    parser.add_argument(
        "--maturity",
        choices=[
            "smoke_passed",
            "short_stable",
            "long_stable",
            "production_ready",
        ],
        default="smoke_passed",
        help="Route progression level used when --steps is omitted.",
    )
    parser.add_argument(
        "--spin", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--spin-temperature", type=float)
    parser.add_argument("--inference-backend", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lmp")
    parser.add_argument("--mpiexec")
    parser.add_argument("--mpi-ranks", type=int)
    parser.add_argument("--pre-failure-frames", type=int)
    parser.add_argument("--bad-tail-frames", type=int)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--output", "-o", default="./trajectory.xyz")
    parser.add_argument("--force", action="store_true")
    _add_execution_options(parser)


def build_manual_label(subparsers):
    parser = subparsers.add_parser(
        "label",
        aliases=["dft"],
        help=(
            "Label structures with VASP, ABACUS, a teacher model, or the "
            "development Adapter."
        ),
    )
    parser.set_defaults(func=run_manual_label_command)
    parser.add_argument("input", help="Structure file, extxyz, or directory.")
    parser.add_argument(
        "--backend",
        choices=["vasp", "abacus", "model", "toy"],
    )
    parser.add_argument("--teacher-profile", choices=["ordinary", "spin"])
    parser.add_argument("--input-file", dest="dft_input")
    parser.add_argument("--resources")
    parser.add_argument(
        "--potcar-manifest",
        help=(
            "Local JSON manifest pinning each VASP POTCAR path, SHA256, "
            "TITEL, family, and release."
        ),
    )
    parser.add_argument(
        "--resource-manifest",
        help=(
            "Local JSON manifest pinning ABACUS pseudopotential and orbital "
            "paths and SHA256 values."
        ),
    )
    parser.add_argument("--cpus", type=int)
    parser.add_argument(
        "--gamma", action=argparse.BooleanOptionalAction, default=None
    )
    kpoints = parser.add_mutually_exclusive_group()
    kpoints.add_argument("--kspacing", type=float)
    kpoints.add_argument("--ka", type=_parse_ka)
    parser.add_argument("--structures-per-job", type=int)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--model", help="Fine-tuned teacher model file.")
    parser.add_argument(
        "--model-name",
        help="Stable teacher family/name recorded in label provenance.",
    )
    parser.add_argument(
        "--runner",
        help=(
            "Installed runner command implementing the NepTrain model-label "
            "protocol."
        ),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--precision", choices=["float32", "float64"])
    parser.add_argument("--output", "-o", default="./labeled.xyz")
    parser.add_argument("--force", action="store_true")
    _add_execution_options(parser)


def build_task_commands(subparsers):
    parser = subparsers.add_parser(
        "task", help="Inspect and control a detached manual step."
    )
    actions = parser.add_subparsers(dest="task_action", required=True)
    action_help = {
        "status": "Refresh scheduler state and collect completed results.",
        "logs": "List scheduler log files for the run.",
        "retry": "Resubmit only failed Slurm array elements.",
        "cancel": "Cancel the active Slurm job for the run.",
    }
    for name, help_text in action_help.items():
        command = actions.add_parser(name, help=help_text)
        command.set_defaults(func=run_task_command)
        command.add_argument("run", help="Manual run directory.")
        command.add_argument("--json", action="store_true")
    wait = actions.add_parser(
        "wait", help="Wait until the run completes, fails, or is cancelled."
    )
    wait.set_defaults(func=run_task_command)
    wait.add_argument("run", help="Manual run directory.")
    wait.add_argument("--poll-interval", type=float, default=10.0)
    wait.add_argument("--json", action="store_true")


def build_data_commands(subparsers):
    parser = subparsers.add_parser(
        "data",
        help="Validate or explicitly migrate scientific dataset contracts.",
    )
    actions = parser.add_subparsers(dest="data_action", required=True)
    migrate = actions.add_parser(
        "migrate-spin",
        help="Rewrite legacy spins/mforces aliases to spin/mforce atomically.",
    )
    migrate.set_defaults(func=run_spin_migration_command)
    migrate.add_argument("input")
    migrate.add_argument("output")
    migrate.add_argument("--force", action="store_true")
    migrate.add_argument("--json", action="store_true")


def build_workflow_commands(subparsers):
    parser = subparsers.add_parser(
        "workflow", help="Prepare and control an automated active-learning workflow."
    )
    actions = parser.add_subparsers(dest="workflow_action", required=True)
    init = actions.add_parser("init", help="Create a strict schema-v8 project.")
    init.set_defaults(func=init_template)
    init.add_argument("--profile", choices=["local", "slurm"], default="slurm")
    init.add_argument("--ensemble", choices=["npt", "nvt"], default="npt")
    init.add_argument("--spin", action="store_true")
    init.add_argument(
        "--dft-backend",
        choices=["vasp", "abacus"],
        default="vasp",
    )
    init.add_argument("--directory", default=".")
    init.add_argument("--force", action="store_true")

    run = actions.add_parser(
        "run",
        help=(
            "Create a workflow from project YAML, or start an existing "
            "prepared workflow directory."
        ),
    )
    run.set_defaults(func=run_project_command)
    run.add_argument("project", help="Project YAML or prepared workflow directory.")
    run.add_argument("--initial-training")
    run.add_argument("--output")
    run.add_argument("--workflow-id")
    run.add_argument("--prepare-only", action="store_true")
    run.add_argument("--foreground", action="store_true")
    run.add_argument("--poll-interval", type=float)

    status = actions.add_parser(
        "status", help="Show scientific progress and current execution state."
    )
    status.set_defaults(func=run_status_command)
    status.add_argument("project")
    status.add_argument("--json", action="store_true")
    status.add_argument("--jobs", action="store_true")

    resume = actions.add_parser(
        "resume", help="Start or restart an existing workflow controller."
    )
    resume.set_defaults(func=run_resume_command)
    resume.add_argument("project")

    extend = actions.add_parser(
        "extend", help="Increase the maximum model-generation budget."
    )
    extend.set_defaults(func=run_extend_command)
    extend.add_argument("project")
    extend.add_argument("generations", type=int)

    stop = actions.add_parser(
        "stop", help="Stop the controller and cancel its current compute jobs."
    )
    stop.set_defaults(func=run_stop_command)
    stop.add_argument("project")
    cancellation = stop.add_mutually_exclusive_group()
    cancellation.set_defaults(cancel_jobs=True)
    cancellation.add_argument(
        "--keep-jobs",
        dest="cancel_jobs",
        action="store_false",
        help="Stop only the controller and keep current compute jobs running.",
    )


def build_internal_commands(subparsers):
    controller = subparsers.add_parser("controller", help=argparse.SUPPRESS)
    subparsers._choices_actions.pop()
    controller.set_defaults(func=run_controller_command)
    controller.add_argument("project")
    controller.add_argument("--poll-interval", type=float)

    worker = subparsers.add_parser("stage-worker", help=argparse.SUPPRESS)
    subparsers._choices_actions.pop()
    worker.set_defaults(func=run_stage_worker_command)
    worker.add_argument("bundle")

    verifier = subparsers.add_parser("stage-verify", help=argparse.SUPPRESS)
    subparsers._choices_actions.pop()
    verifier.set_defaults(func=run_stage_verify_command)
    verifier.add_argument("bundle")

    manual = subparsers.add_parser("manual-worker", help=argparse.SUPPRESS)
    subparsers._choices_actions.pop()
    manual.set_defaults(func=run_manual_worker_command)
    manual.add_argument("run")
    manual.add_argument("index", type=int)


def main():
    parser = argparse.ArgumentParser(
        prog="neptrain",
        description=(
            "Run individual NEP training, MD, labeling and sampling steps, "
            "or compose the same steps into an automated workflow."
        ),
    )
    parser.add_argument("-v", "--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar=(
            "{train,md,label,select,perturb,workflow,task,data,doctor,smoke}"
        ),
    )
    build_manual_train(subparsers)
    build_manual_md(subparsers)
    build_manual_label(subparsers)
    build_select(subparsers)
    build_perturb(subparsers)
    build_workflow_commands(subparsers)
    build_task_commands(subparsers)
    build_data_commands(subparsers)
    build_doctor(subparsers)
    build_smoke(subparsers)
    build_internal_commands(subparsers)
    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:
        pass
    args = parser.parse_args()
    try:
        result = args.func(args)
        return result if type(result) is int else None
    except Exception as error:
        from NepTrain.core.workflow import WorkflowError
        from NepTrain.core.config import ConfigError
        from NepTrain.core.controller import ControllerError
        from NepTrain.core.execution import ExecutionError
        from NepTrain.core.iteration import IterationError
        from NepTrain.core.manual import ManualTaskError

        lightweight_errors = (
            WorkflowError,
            ConfigError,
            ControllerError,
            ExecutionError,
            IterationError,
            ManualTaskError,
        )
        scientific_error_names = {
            "LabelingError",
            "MdError",
            "SmokeError",
            "SpinDataError",
            "SelectionError",
            "TrainingError",
            "WorkflowIterationError",
        }
        is_scientific_error = (
            type(error).__module__.startswith("NepTrain.core.")
            and type(error).__name__ in scientific_error_names
        )
        if isinstance(error, lightweight_errors) or is_scientific_error:
            parser.exit(2, f"NepTrain: error: {error}\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
