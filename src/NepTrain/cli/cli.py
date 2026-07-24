#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 14:33
# @Author  : 兵
# @email    : 1747193328@qq.com
import argparse
import json
import os
from pathlib import Path
from NepTrain import __version__


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
    import json

    from NepTrain.core.smoke import run_smoke

    report = run_smoke(
        args.output,
        profile=args.profile,
        seed=args.seed,
        max_selected=args.max_selected,
        force=args.force,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
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
        print(json.dumps(asdict(iteration), indent=2, sort_keys=True))


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
            temperatures = ", ".join(f"{value:g}" for value in plan["temperatures"])
            print(
                f"  G{number} {state}: FPS selects up to "
                f"{plan['max_selected']}, {plan['steps']} MD steps, "
                f"T={temperatures} K"
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
            flow.append(f"DFT {sampling['labeled_count']}")
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
        "iterations"
    )
    if status.generation is not None:
        print(f"Ledger: iteration {status.generation}, stage {status.stage}")
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
    """Start or continue a workflow through the small user interface."""

    from NepTrain.core.workflow import (
        WorkflowError,
        prepare_workflow,
        resume_workflow,
    )
    from NepTrain.core.controller import start_controller

    project = Path(args.project).expanduser()
    try:
        if project.is_dir():
            resume_options = {}
            if getattr(args, "foreground", False):
                resume_options["foreground"] = True
            if getattr(args, "poll_interval", None) is not None:
                resume_options["poll_interval"] = args.poll_interval
            result = resume_workflow(project, **resume_options)
            payload = {
                "workflow_id": result.workflow_id,
                "action": result.action,
                "manifest": str(result.manifest),
            }
            if result.controller_pid is not None:
                payload["controller_pid"] = result.controller_pid
            elif result.controller_exit_code is not None:
                payload["controller_exit_code"] = result.controller_exit_code
            else:
                payload["job_ids"] = list(result.job_ids)
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
                    workflow_id = str(config.get("workflow", {}).get("id", "")).strip()
                    if workflow_id:
                        output = str((project.parent / workflow_id).resolve())
            if not initial_training or not output:
                raise WorkflowError(
                    "starting from a project file requires training.initial_path "
                    "(or --initial-training) and workflow.id (or --output)"
                )
            preparation = prepare_workflow(
                project,
                initial_training,
                output,
                workflow_id=args.workflow_id,
            )
            payload = {
                "workflow_id": preparation.workflow_id,
                "project": str(preparation.output_dir),
                "manifest": str(preparation.manifest),
                "started": not args.prepare_only,
            }
            if not args.prepare_only:
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
    except WorkflowError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_status_command(args):
    from dataclasses import asdict
    from NepTrain.core.workflow import WorkflowError, workflow_status

    try:
        status = workflow_status(args.project)
    except WorkflowError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    if args.json:
        print(json.dumps(asdict(status), indent=2, sort_keys=True))
    else:
        _print_workflow_status(status, show_jobs=args.jobs)


def run_resume_command(args):
    from dataclasses import asdict
    from NepTrain.core.workflow import WorkflowError, resume_workflow

    try:
        result = resume_workflow(args.project)
    except WorkflowError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    payload = asdict(result)
    payload["job_ids"] = list(result.job_ids)
    payload["manifest"] = str(result.manifest)
    if result.controller_pid is not None:
        payload.pop("job_ids", None)
        payload.pop("controller_exit_code", None)
    elif result.controller_exit_code is not None:
        payload.pop("job_ids", None)
        payload.pop("controller_pid", None)
    else:
        payload.pop("controller_pid", None)
        payload.pop("controller_exit_code", None)
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_extend_command(args):
    from NepTrain.core.workflow import WorkflowError, extend_workflow

    try:
        preparation = extend_workflow(args.project, args.iterations)
    except WorkflowError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    print(
        json.dumps(
            {
                "workflow_id": preparation.workflow_id,
                "total_iterations": len(preparation.plans),
                "project": str(preparation.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


def run_stop_command(args):
    from NepTrain.core.controller import ControllerError, stop_workflow

    try:
        result = stop_workflow(
            args.project, cancel_jobs=bool(args.cancel_jobs)
        )
    except ControllerError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))


def run_controller_command(args):
    from NepTrain.core.controller import run_controller

    return run_controller(args.project, poll_interval=args.poll_interval)


def run_stage_worker_command(args):
    from NepTrain.core.execution import run_stage_worker

    return run_stage_worker(args.bundle)


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
            required = []
            if target.executor == "slurm":
                required.extend(["sbatch", "squeue", "sacct"])
            required.append(shlex.split(target.command)[0])
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
            if target.host:
                completed = subprocess.run(
                    [
                        "ssh",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "ConnectTimeout=10",
                        target.host,
                        "bash",
                        "-s",
                    ],
                    input=probe,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            else:
                completed = subprocess.run(
                    ["bash", "-s"],
                    input=probe,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            available = completed.returncode == 0
            location = target.host or "local"
            print(
                f"{'OK' if available else 'FAIL'} execution target {name} "
                f"({target.executor} on {location})"
            )
            if not available:
                failures.append(f"execution target {name}")
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
            completed = subprocess.run(
                probe_command,
                capture_output=True,
                text=True,
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
    parser_perturb = subparsers.add_parser(
        "perturb",
        help="Generate perturbed structures.",
    )

    parser_perturb.set_defaults(func=run_perturb)

    parser_perturb.add_argument("model_path",
                             type=str,

                             help="The structure path or structure file required for calculation only supports files in xyz and vasp formats.")
    parser_perturb.add_argument("--num","-n",
                             type=int,
                                default=20,
                             help="The number of perturbations for each structure, if a folder is input, the final number generated should be the number of structures multiplied by num.default 20.")

    parser_perturb.add_argument("--cell", "-c",
                                dest="cell_pert_fraction",
                                type=float,
                                default=0.03,
                                help="The deformation ratio,default 0.03.")

    parser_perturb.add_argument("--distance", "-d",
                                type=float,
                                dest="min_distance",
                                default=0.1,
                                help="Min atom distance, unit Å, default 0.1.")

    parser_perturb.add_argument("--out", "-o",
                             dest="out_file_path",
                             type=str,
                             help="Output file for perturbed structures, default ./perturb.xyz.",
                             default="./perturb.xyz"
                             )
    parser_perturb.add_argument("--append", "-a",
                             dest="append", action='store_true', default=False,
                             help="Write to out_file_path in append mode, default False.",

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
        help="Select samples.",
    )
    parser_select.set_defaults(func=run_select,decomposition='pca')

    parser_select.add_argument("trajectory_paths",
                              nargs="+",
                             help="The trajectory files needed for sampling is in xyz format.")


    parser_select.add_argument("--base", "-base",
                               type=str,
                               default="train.xyz",
                               help="Provide a path to base.xyz, and sample the trajectory based on base.xyz, default is train.xyz."
                               )
    parser_select.add_argument("--nep", "-nep",
                               type=str,
                               default="./nep.txt",
                               help="Provide a path to a nep.txt file to extract descriptors for the structure, default is ./nep.txt. If the file does not exist, use SOAP descriptors."
                               )
    parser_select.add_argument("--max-selected", "-max", type=int,
                               help="Maximum number of structures to select, default is 20.",
                               default=20)
    parser_select.add_argument("--min_distance","-d", type=float,
                               help="Minimum bond length for farthest-point sampling, default is 0.01.",
                               default=0.01)
    parser_select.add_argument("--filter", "-f", type=float,
                               const=0.6,nargs='?',
                               help="Whether to filter based on covalent radius, the default is False. If True, the default coefficient is 0.6, and a coefficient can be passed in",
                               default=False)

    dc_group = parser_select.add_mutually_exclusive_group(required=False)
    dc_group.add_argument('-pca',"--pca", action='store_const', const='pca', dest='decomposition',
                       help='Use PCA for decomposition')
    dc_group.add_argument('-umap',"--umap", action='store_const', const='umap', dest='decomposition',
                       help='Use UMAP for decomposition')

    parser_select.add_argument("--out", "-o",
                               dest="out_file_path",

                               type=str,
                               default="./selected.xyz",
                               help="Output path for selected structures.default ./selected.xyz"
                               )

    group= parser_select.add_argument_group("SOAP","SOAP Parameters")

    group.add_argument("--r_cut", "-r", type=float, help="A cutoff for local region in angstroms,default 6", default=6)
    group.add_argument("--n_max", "-n", type=int, help="The number of radial basis functions,default 8", default=8)
    group.add_argument("--l_max", "-l", type=int, help="The maximum degree of spherical harmonics,default 6", default=6)

def _print_manual_status(value):
    print(json.dumps(value, indent=2, sort_keys=True))


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
        )
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
    conditions = sampling.get("conditions", {})
    progression = sampling.get("progression", {})
    candidate_pool = sampling.get("candidate_pool", {})
    progression_steps = progression.get("steps", {})
    target = target_from_project(args.project, args.target, route="sampling")
    operation = prepare_md(
        args.input,
        backend=args.backend or settings.get("backend", "lammps"),
        model_file=args.model,
        temperatures=args.temperature
        or conditions.get("temperature_path", [300.0]),
        output=args.output,
        workdir=args.workdir,
        target=target,
        steps=args.steps
        if args.steps is not None
        else int(progression_steps.get("smoke_passed", 10000)),
        pressure=args.pressure
        if args.pressure is not None
        else float(conditions.get("pressure", 0.0)),
        ensemble=args.ensemble or "nvt",
        template_path=args.template
        or _project_path(base, settings.get("template_path")),
        spin=args.spin
        if args.spin is not None
        else bool(settings.get("spin", False)),
        spin_temperature=args.spin_temperature
        if args.spin_temperature is not None
        else conditions.get("spin_temperature"),
        inference_backend=args.inference_backend
        or settings.get("inference_backend", "auto"),
        lmp=args.lmp or settings.get("lmp", "lmp"),
        mpiexec=args.mpiexec or settings.get("mpiexec", "mpirun"),
        mpi_ranks=args.mpi_ranks
        if args.mpi_ranks is not None
        else int(settings.get("mpi_ranks", 1)),
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
        max_concurrent=args.max_concurrent,
        force=args.force,
    )
    _print_manual_status(
        submit_operation(
            operation, wait=args.wait, poll_interval=args.poll_interval
        )
    )


def run_manual_dft_command(args):
    from NepTrain.core.manual import (
        prepare_dft,
        submit_operation,
        target_from_project,
    )

    project, base = _manual_project(args.project)
    settings = project.get("dft", {})
    target = target_from_project(args.project, args.target, route="labeling")
    use_k_stype = settings.get("use_k_stype", "kspacing")
    kspacing = (
        args.kspacing
        if args.kspacing is not None
        else settings.get("kspacing")
    )
    operation = prepare_dft(
        args.input,
        backend=args.backend or settings.get("backend", "vasp"),
        output=args.output,
        workdir=args.workdir,
        target=target,
        input_file=args.dft_input
        or _project_path(base, settings.get("input_path")),
        resource_dir=args.resources
        or _project_path(base, settings.get("resource_path")),
        n_cpu=args.cpus,
        use_gamma=(
            args.gamma
            if args.gamma is not None
            else bool(settings.get("kpoints_use_gamma", False))
        ),
        kpoint_mode=(
            "kspacing" if args.kspacing is not None else use_k_stype
        ),
        kspacing=kspacing,
        ka=args.ka or settings.get("kpoints", [1, 1, 1]),
        structures_per_job=args.structures_per_job,
        max_concurrent=args.max_concurrent,
        teacher_profile=args.teacher_profile
        or settings.get("teacher_profile", "ordinary"),
        force=args.force,
    )
    _print_manual_status(
        submit_operation(
            operation, wait=args.wait, poll_interval=args.poll_interval
        )
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
        value = {"run_directory": str(operation.root), "logs": operation_logs(operation)}
    else:  # pragma: no cover - argparse owns this invariant
        raise ValueError(args.task_action)
    _print_manual_status(value)


def run_manual_worker_command(args):
    from NepTrain.core.manual import run_manual_worker

    return run_manual_worker(args.run, args.index)


def _add_execution_options(parser):
    parser.add_argument(
        "--project",
        help="Schema-v4 project providing reusable execution targets.",
    )
    parser.add_argument("--target", help="Execution target name from project.yaml.")
    parser.add_argument("--workdir", help="Durable run directory.")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for submitted Slurm work and publish the final result.",
    )
    parser.add_argument("--poll-interval", type=float, default=10.0)


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
    parser.add_argument("--ensemble", choices=["nvt", "npt"])
    parser.add_argument("--template")
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
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument("--output", "-o", default="./trajectory.xyz")
    parser.add_argument("--force", action="store_true")
    _add_execution_options(parser)


def build_manual_dft(subparsers):
    parser = subparsers.add_parser(
        "dft", help="Label structures with VASP, ABACUS, or the development teacher."
    )
    parser.set_defaults(func=run_manual_dft_command)
    parser.add_argument("input", help="Structure file, extxyz, or directory.")
    parser.add_argument("--backend", choices=["vasp", "abacus", "toy"])
    parser.add_argument("--teacher-profile", choices=["ordinary", "spin"])
    parser.add_argument("--input-file", dest="dft_input")
    parser.add_argument("--resources")
    parser.add_argument("--cpus", type=int)
    parser.add_argument(
        "--gamma", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--kspacing", type=float)
    parser.add_argument("--ka", type=_parse_ka)
    parser.add_argument("--structures-per-job", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=20)
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
    wait = actions.add_parser(
        "wait", help="Wait until the run completes, fails, or is cancelled."
    )
    wait.set_defaults(func=run_task_command)
    wait.add_argument("run", help="Manual run directory.")
    wait.add_argument("--poll-interval", type=float, default=10.0)


def build_workflow_commands(subparsers):
    parser = subparsers.add_parser(
        "workflow", help="Prepare and control an automated active-learning workflow."
    )
    actions = parser.add_subparsers(dest="workflow_action", required=True)
    init = actions.add_parser("init", help="Create a strict schema-v5 project.")
    init.set_defaults(func=init_template)
    init.add_argument("--profile", choices=["local", "slurm"], default="slurm")
    init.add_argument("--directory", default=".")
    init.add_argument("--force", action="store_true")

    run = actions.add_parser(
        "run", help="Prepare a workflow and start its persistent controller."
    )
    run.set_defaults(func=run_project_command)
    run.add_argument("project", help="Project YAML or workflow directory.")
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
        "resume", help="Restart a stopped or failed workflow controller."
    )
    resume.set_defaults(func=run_resume_command)
    resume.add_argument("project")

    extend = actions.add_parser(
        "extend", help="Increase the maximum model-iteration budget."
    )
    extend.set_defaults(func=run_extend_command)
    extend.add_argument("project")
    extend.add_argument("iterations", type=int)

    stop = actions.add_parser(
        "stop", help="Stop the controller without cancelling compute jobs."
    )
    stop.set_defaults(func=run_stop_command)
    stop.add_argument("project")
    stop.add_argument(
        "--cancel-jobs",
        action="store_true",
        help="Also cancel the workflow's current process or Slurm job.",
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

    manual = subparsers.add_parser("manual-worker", help=argparse.SUPPRESS)
    subparsers._choices_actions.pop()
    manual.set_defaults(func=run_manual_worker_command)
    manual.add_argument("run")
    manual.add_argument("index", type=int)


def main():
    parser = argparse.ArgumentParser(
        prog="neptrain",
        description=(
            "Run individual NEP training, MD, DFT and sampling steps, or compose "
            "the same steps into an automated workflow."
        ),
    )
    parser.add_argument("-v", "--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{train,md,dft,select,perturb,workflow,task,doctor,smoke}",
    )
    build_manual_train(subparsers)
    build_manual_md(subparsers)
    build_manual_dft(subparsers)
    build_select(subparsers)
    build_perturb(subparsers)
    build_workflow_commands(subparsers)
    build_task_commands(subparsers)
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
        return args.func(args)
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
