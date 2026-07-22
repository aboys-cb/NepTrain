#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 14:33
# @Author  : 兵
# @email    : 1747193328@qq.com
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.append('../../')
from NepTrain import __version__
import warnings
try:
    from dpdispatcher.dlog import dlog_stdout, dlog
    dlog.removeHandler(dlog_stdout)
except ImportError:
    pass
# 禁用所有 UserWarning
warnings.simplefilter('ignore', UserWarning)


def init_template(args):
    from NepTrain.core.template import init_template as implementation
    return implementation(args)


def run_perturb(args):
    from NepTrain.core.perturb import run_perturb as implementation
    return implementation(args)


def run_select(args):
    from NepTrain.core.select import run_select as implementation
    return implementation(args)


def run_dft(args):
    from NepTrain.core.dft import run_dft as implementation
    implementation(args)


def run_vasp(args):
    from NepTrain.core.dft.vasp import run_vasp as implementation
    implementation(args)


def run_nep(args):
    from NepTrain.core.nep import run_nep as implementation
    return implementation(args)


def run_gpumd(args):
    from NepTrain.core.gpumd import run_gpumd as implementation
    return implementation(args)


def train_nep(args):
    from NepTrain.core.train import train_nep as implementation
    return implementation(args)


def run_md_command(args):
    from ase.io import read as ase_read
    from NepTrain.core.md import MdRequest, run_md

    source = Path(args.model_path)
    paths = sorted(
        path for pattern in ("*.xyz", "*.vasp", "POSCAR*") for path in source.glob(pattern)
    ) if source.is_dir() else [source]
    frames = []
    for path in paths:
        loaded = ase_read(path, index=":", format=None)
        frames.extend(loaded if isinstance(loaded, list) else [loaded])
    if not frames:
        raise FileNotFoundError(f"no readable structures found in {source}")
    output = Path(args.out_file_path)
    if output.exists() and not args.append:
        output.unlink()
    for frame_index, atoms in enumerate(frames):
        digest = hashlib.sha256(atoms.positions.tobytes()).hexdigest()[:12]
        for temperature in args.temperature:
            directory = Path(args.directory) / f"{frame_index}-{digest}-{temperature:g}K"
            request = MdRequest(
                atoms=atoms,
                model_file=Path(args.nep_txt_path),
                output_dir=directory,
                output_file=output,
                temperature=temperature,
                spin_temperature=args.spin_temperature,
                steps=args.steps,
                timestep=args.timestep,
                ensemble=args.ensemble,
                pressure=args.pressure,
                tdamp=args.tdamp,
                pdamp=args.pdamp,
                dump_interval=args.dump_interval,
                spin=args.spin,
                spin_alpha=args.spin_alpha,
                spin_seed=args.spin_seed,
                midpoint_iter=args.midpoint_iter,
                template_path=Path(args.template) if args.template else None,
                inference_backend=args.inference_backend,
                lmp_command=args.lmp,
                mpiexec=args.mpiexec,
                mpi_ranks=args.mpi_ranks,
                plugin_path=args.plugin_path,
            )
            run_md(request, args.backend)


def run_migrate(args):
    from NepTrain.core.config import load_config, save_config

    config, changes = load_config(args.config_path)
    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} already exists; pass --force to overwrite")
    save_config(config, output)
    if changes:
        print("Migrated: " + ", ".join(changes))
    print(f"Wrote schema-v2 config: {output}")


def run_smoke_command(args):
    from dataclasses import asdict
    import json

    from NepTrain.core.smoke import run_backend_workflow_smoke, run_smoke

    report = run_smoke(
        args.output,
        profile=args.profile,
        seed=args.seed,
        dft_budget=args.dft_budget,
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
            dft_budget=args.dft_budget,
            force=args.force,
        )
        print(json.dumps(asdict(iteration), indent=2, sort_keys=True))
    if args.workflow_config:
        workflow = run_backend_workflow_smoke(
            args.workflow_config,
            args.output,
            profile="spin" if args.profile == "recovery" else args.profile,
            training_steps=args.training_steps,
            md_steps=args.md_steps,
            dft_budget=args.dft_budget,
        )
        print(json.dumps(asdict(workflow), indent=2, sort_keys=True))


def _iteration_execution(args):
    from NepTrain.core.config import load_config
    from NepTrain.core.iteration import GenerationController, GenerationPlan
    from NepTrain.core.workflow_iteration import WorkflowIterationAdapter

    config_path = Path(args.config_path).expanduser().resolve()
    config, _ = load_config(config_path)
    plan_data = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    plan_data["temperatures"] = tuple(plan_data["temperatures"])
    plan = GenerationPlan(**plan_data)
    adapter = WorkflowIterationAdapter(
        config,
        initial_training=args.initial_training,
        base_dir=config_path.parent,
    )
    controller = GenerationController(args.campaign_dir, args.campaign_id)
    return plan, adapter, controller


def run_iteration_stage_command(args):
    from dataclasses import asdict
    import json

    plan, adapter, controller = _iteration_execution(args)
    result = controller.run_stage(plan, adapter, args.stage)
    payload = asdict(result)
    payload["artifacts"] = {
        name: str(path) for name, path in result.artifacts.items()
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if result.generation_complete and result.accepted is False:
        raise SystemExit(
            f"Generation {result.generation} was rejected; dependent jobs will not run."
        )


def run_iteration_resource_command(args):
    from dataclasses import asdict
    import json

    from NepTrain.core.iteration import IterationError

    plan, adapter, controller = _iteration_execution(args)
    allowed = {"train", "retrain"} if args.resource == "training" else {
        "explore",
        "select",
        "label",
        "diagnose",
        "merge",
        "evaluate",
    }
    results = []
    while True:
        stage = controller.next_stage(plan)
        if stage is None:
            break
        if stage not in allowed:
            if results or args.resource == "training":
                break
            raise IterationError(
                f"resource {args.resource} cannot execute pending stage {stage}"
            )
        result = controller.run_stage(plan, adapter, stage)
        payload = asdict(result)
        payload["artifacts"] = {
            name: str(path) for name, path in result.artifacts.items()
        }
        results.append(payload)
        if result.generation_complete and result.accepted is False:
            print(json.dumps(results, indent=2, sort_keys=True))
            raise SystemExit(
                f"Generation {result.generation} was rejected; dependent jobs will not run."
            )
    print(json.dumps(results, indent=2, sort_keys=True))


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
                f"  G{number} {state}: plan {plan['candidate_target']} candidates, "
                f"DFT budget {plan['dft_budget']}, {plan['steps']} MD steps, "
                f"T={temperatures} K"
            )
            continue

        sampling = generation["sampling"]
        training = generation["training"]
        flow = []
        if sampling["candidate_count"] is not None:
            flow.append(f"MD {sampling['candidate_count']} candidates")
        if sampling["candidate_count_after_thinning"] is not None:
            flow.append(f"{sampling['candidate_count_after_thinning']} eligible")
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


def run_campaign_command(args):
    from dataclasses import asdict
    import json

    if args.json and not args.status:
        raise SystemExit("campaign --json requires --status")

    if args.status:
        from NepTrain.core.campaign import CampaignError, campaign_status

        try:
            status = campaign_status(args.output)
        except CampaignError as error:
            raise SystemExit(f"NepTrain: error: {error}") from error
        if args.json:
            print(json.dumps(asdict(status), indent=2, sort_keys=True))
            return
        print(f"Campaign: {status.campaign_id}")
        print(f"State: {status.state}")
        print(
            f"Progress: {status.completed_generations}/{status.total_generations} "
            "generations"
        )
        if status.generation is not None:
            print(f"Ledger: generation {status.generation}, stage {status.stage}")
        else:
            print("Ledger: complete")
        print(f"Reason: {status.reason}")
        _print_scientific_progress(status.generations)
        print("Jobs:")
        for job in status.jobs:
            marker = "*" if job["current"] else "-"
            attempt = job["attempt"] or "not-submitted"
            job_id = job["job_id"] or "-"
            print(
                f"  {marker} {attempt:13} {job_id:>8} "
                f"{job['state']:>20}  {Path(job['script']).name}"
            )
        if status.next_action:
            print(f"Next: {status.next_action}")
        return

    if args.retry_failed:
        from NepTrain.core.campaign import CampaignError, retry_failed_campaign

        try:
            retry = retry_failed_campaign(args.output)
        except CampaignError as error:
            raise SystemExit(f"NepTrain: error: {error}") from error
        payload = asdict(retry)
        payload["job_ids"] = list(retry.job_ids)
        payload["manifest"] = str(retry.manifest)
        payload["retried"] = True
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.submit and not args.config_path and not args.initial_training:
        from NepTrain.core.campaign import CampaignError, submit_campaign

        try:
            submission = submit_campaign(args.output)
        except CampaignError as error:
            raise SystemExit(f"NepTrain: error: {error}") from error
        print(
            json.dumps(
                {
                    "campaign_id": submission.campaign_id,
                    "job_ids": list(submission.job_ids),
                    "manifest": str(submission.manifest),
                    "submitted": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not args.config_path or not args.initial_training:
        raise SystemExit(
            "campaign preparation requires config_path and --initial-training"
        )

    from NepTrain.core.campaign import prepare_campaign, submit_campaign

    preparation = prepare_campaign(
        args.config_path,
        args.initial_training,
        args.output,
        campaign_id=args.campaign_id,
    )
    payload = asdict(preparation)
    payload = {
        key: [str(item) for item in value]
        if isinstance(value, tuple)
        else str(value)
        if isinstance(value, Path)
        else value
        for key, value in payload.items()
    }
    if args.submit:
        submission = submit_campaign(preparation)
        payload["job_ids"] = list(submission.job_ids)
        payload["submitted"] = True
    else:
        payload["submitted"] = False
    print(json.dumps(payload, indent=2, sort_keys=True))


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
            environment = os.environ.copy()
            if args.plugin_path:
                environment["LAMMPS_PLUGIN_PATH"] = str(Path(args.plugin_path).expanduser())
            probe_command = shlex.split(args.lmp)
            probe_input = None
            if args.plugin_path:
                plugin = Path(args.plugin_path).expanduser().resolve()
                if plugin.is_dir():
                    name = "nepadaptersgpuplugin.so" if selected_inference == "cuda" else "nepadapterscpuplugin.so"
                    plugin = plugin / name
                probe_input = f"plugin load {plugin}\ninfo styles pair\n"
                probe_command.extend(["-log", "none"])
            else:
                probe_command.append("-h")
            completed = subprocess.run(
                probe_command,
                input=probe_input,
                capture_output=True,
                text=True,
                env=environment,
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
                    dump_interval=1,
                    spin=spin,
                    inference_backend=selected_inference,
                    lmp_command=args.lmp,
                    mpiexec=args.mpiexec,
                    mpi_ranks=args.mpi_ranks,
                    plugin_path=args.plugin_path,
                ),
                "lammps",
            )
        print(f"OK real LAMMPS smoke at mpi_ranks={args.mpi_ranks}")
    if failures:
        raise SystemExit("Doctor failed: " + ", ".join(failures))
    print("Doctor completed successfully.")
def check_kpoints_number(value):
    """检查值是否为单个数字或三个数字的字符串"""

    if isinstance(value, str):
        values = value.split(',')

        if len(values) == 3 and all(v.isdigit() for v in values):
            return list(map(int, values))
        elif len(values) == 1 and value.isdigit():
            return [int(value),int(value),int(value)]
        else:
            raise argparse.ArgumentTypeError("The ka parameter must be a single number or three numbers separated by `,`.")
    elif isinstance(value, int):
        return value
    else:
        raise argparse.ArgumentTypeError("The ka parameter must be a single number or three numbers separated by `,`.")

def build_init(subparsers):
    parser_init = subparsers.add_parser(
        "init",
        help="Initialize some file templates",
    )
    parser_init.add_argument("type",
                             type=str,
                            choices=["bohrium","slurm","pbs","shell"],default="slurm",
                             help="How to call a task")

    parser_init.add_argument("-f", "--force", action='store_true',
                             default=False,
                             help="Force overwriting of generated templates"
                             )

    parser_init.set_defaults(func=init_template)



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

def build_vasp(subparsers):
    parser_vasp = subparsers.add_parser(
        "vasp",
        help="Calculate single-point energy using VASP.",
    )
    parser_vasp.set_defaults(func=run_vasp)

    parser_vasp.add_argument("model_path",
                             type=str,

                             help="The required structure path or structure file only supports files in xyz and vasp formats.")
    parser_vasp.add_argument("--directory", "-dir",

                             type=str,
                             help="Set the VASP calculation path. default ./cache/vasp.",
                             default="./cache/vasp"
                             )

    parser_vasp.add_argument("--out", "-o",
                             dest="out_file_path",
                             type=str,
                             help="Structure output file after calculation. default ./vasp_scf.xyz",
                             default="./vasp_scf.xyz"
                             )

    parser_vasp.add_argument("--append", "-a",
                             dest="append", action='store_true', default=False,
                             help="Write to out_file_path in append mode, default False.",

                             )
    parser_vasp.add_argument("--gamma", "-g",
                             dest="use_gamma", action='store_true', default=False,
                             help="Default to using Monkhorst-Pack k-points, add -g to use Gamma-centered k-point scheme.",

                             )
    parser_vasp.add_argument("-n", "-np",
                             dest="n_cpu",
                             default=1,
                             type=int,
                             help="Set the number of CPU cores, default 1.")

    parser_vasp.add_argument("--incar",

                             help="Input path for INCAR file, default is ./INCAR.",default="./INCAR")



    k_group = parser_vasp.add_mutually_exclusive_group(required=False)
    k_group.add_argument("--kspacing", "-kspacing",

                         type=float,
                         help="Set kspacing, which can also be defined in the INCAR template.")
    k_group.add_argument("--ka", "-ka",
                         default=[1, 1, 1],
                         type=check_kpoints_number,
                         help="ka takes 1 or 3 numbers (comma-separated), sets k-points to (k[0]/a, k[1]/b, k[2]/c). default 1.")
def build_dft(subparsers):
    parser_dft = subparsers.add_parser(
        "dft",
        help="Calculate single-point energy using DFT software.",
    )
    parser_dft.set_defaults(func=run_dft)

    parser_dft.add_argument("model_path",
                             type=str,

                             help="The required structure path or structure file only supports files in xyz and vasp formats.")
    parser_dft.add_argument("--directory", "-dir",

                             type=str,
                             help="Set the VASP calculation path. default ./cache/software.",
                             default=None
                             )

    parser_dft.add_argument("--out", "-o",
                             dest="out_file_path",
                             type=str,
                             help="Structure output file after calculation. default ./software_scf.xyz",
                             default=None
                             )

    parser_dft.add_argument("--append", "-a",
                             dest="append", action='store_true', default=False,
                             help="Write to out_file_path in append mode, default False.",

                             )
    parser_dft.add_argument("--gamma", "-g",
                             dest="use_gamma", action='store_true', default=False,
                             help="Default to using Monkhorst-Pack k-points, add -g to use Gamma-centered k-point scheme.",

                             )
    parser_dft.add_argument("-n", "-np",
                             dest="n_cpu",
                             default=1,
                             type=int,
                             help="Set the number of CPU cores, default 1.")

    parser_dft.add_argument("--in",
                                dest="incar",
                             help="Input path for INCAR file, default is ./INCAR or ./INPUT.",default=None)



    k_group = parser_dft.add_mutually_exclusive_group(required=False)
    k_group.add_argument("--kspacing", "-kspacing",

                         type=float,
                         help="Set kspacing, which can also be defined in the INCAR template.")
    k_group.add_argument("--ka", "-ka",
                         default=[1, 1, 1],
                         type=check_kpoints_number,
                         help="ka takes 1 or 3 numbers (comma-separated), sets k-points to (k[0]/a, k[1]/b, k[2]/c). default 1.")

    software_group = parser_dft.add_mutually_exclusive_group(required=False)
    software_group.add_argument("--vasp" ,
                                dest="software",

                                action='store_const', const='vasp',
                         help="use vasp.(default)")
    software_group.add_argument("--abacus",
                                dest="software",
                                action='store_const', const='abacus',
                                help="use abacus")
    software_group.add_argument("--toy",
                                dest="software",
                                action="store_const", const="toy",
                                help="use the deterministic development teacher")
    parser_dft.add_argument(
        "--teacher-profile",
        choices=["ordinary", "spin"],
        default="ordinary",
        help="Toy teacher contract; ignored by production DFT Adapters.",
    )



def build_nep(subparsers):
    parser_nep = subparsers.add_parser(
        "nep",
        help="Train potential functions using NEP.",
    )
    parser_nep.set_defaults(func=run_nep)


    parser_nep.add_argument("--directory", "-dir",
                             type=str,
                             help="Set the path for NEP calculations. default ./cache/nep",
                             default="./cache/nep"
                             )

    parser_nep.add_argument("--in", "-in",
                            dest="nep_in_path",
                             type=str,
                             help="Set the path for the nep.in file; if not present, generate it based on train.xyz. default ./nep.in",
                             default="./nep.in"
                             )

    parser_nep.add_argument("--train", "-train",
                             dest="train_path",

                             type=str,
                             help="Set the path for the train.xyz file, default  ./train.xyz.",
                             default="./train.xyz"
                             )

    parser_nep.add_argument("--test", "-test",
                             dest="test_path",
                             type=str,
                             help="Set the path for the test.xyz file, default is ./test.xyz.",
                             default="./test.xyz"
                             )

    parser_nep.add_argument("--nep", "-nep",
                            dest="nep_txt_path",
                             type=str,
                             help="restart and prediction require the use of a potential function, default is ./nep.txt.",
                             default="./nep.txt"
                             )

    parser_nep.add_argument("--prediction", "-pred","--pred",

                             action="store_true",
                             help="Set the forecast mode，default False",
                             default=False
                             )

    parser_nep.add_argument("--restart_file", "-restart","--restart",

                            type=str,

                            help="To restart running, simply provide a valid path; default is None.",
                             default=None
                             )

    parser_nep.add_argument("--continue_step", "-cs",
                            type=int,
                            help="If a restart_file is provided, this parameter will take effect, continuing for continue_step steps, with a default value of 10000.",
                             default=10000
                             )
    parser_nep.add_argument("--backend", choices=["gpumd", "torchnep"], default="gpumd")
    parser_nep.add_argument("--device", default="cuda")
    parser_nep.add_argument("--torch-backend", choices=["auto", "loop", "bmm"], default="auto")
    parser_nep.add_argument("--precision", choices=["float32", "float64"], default="float32")
    parser_nep.add_argument("--compile", dest="use_compile", action="store_true")
    parser_nep.add_argument(
        "--inference-backend", choices=["auto", "cpu", "cuda"], default="auto"
    )


def build_md(subparsers):
    parser_md = subparsers.add_parser("md", help="Run MD with GPUMD or LAMMPS.")
    parser_md.set_defaults(func=run_md_command)
    parser_md.add_argument("model_path", help="Input structure or extxyz path.")
    parser_md.add_argument("--backend", choices=["gpumd", "lammps"], default="gpumd")
    parser_md.add_argument("--nep", dest="nep_txt_path", default="./nep.txt")
    parser_md.add_argument("--directory", default="./cache/md")
    parser_md.add_argument("--out", dest="out_file_path", default="./trajectory.xyz")
    parser_md.add_argument("--append", action="store_true")
    parser_md.add_argument("--template", default=None)
    parser_md.add_argument("--temperature", type=float, nargs="+", default=[300.0])
    parser_md.add_argument("--spin-temperature", type=float, default=None)
    parser_md.add_argument("--steps", type=int, default=10000)
    parser_md.add_argument("--timestep", type=float, default=0.001)
    parser_md.add_argument("--ensemble", choices=["nvt", "npt"], default="nvt")
    parser_md.add_argument("--pressure", type=float, default=0.0)
    parser_md.add_argument("--tdamp", type=float, default=0.1)
    parser_md.add_argument("--pdamp", type=float, default=1.0)
    parser_md.add_argument("--dump-interval", type=int, default=100)
    parser_md.add_argument("--spin", action="store_true")
    parser_md.add_argument("--spin-alpha", type=float, default=0.01)
    parser_md.add_argument("--spin-seed", type=int, default=12345)
    parser_md.add_argument("--midpoint-iter", type=int, default=3)
    parser_md.add_argument(
        "--inference-backend", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser_md.add_argument("--lmp", default="lmp")
    parser_md.add_argument("--mpiexec", default="mpirun")
    parser_md.add_argument("--mpi-ranks", type=int, default=1)
    parser_md.add_argument("--plugin-path", default=None)


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
    parser.add_argument("--plugin-path", default=None)


def build_migrate(subparsers):
    parser = subparsers.add_parser("migrate", help="Migrate a legacy job config to schema v2.")
    parser.set_defaults(func=run_migrate)
    parser.add_argument("config_path")
    parser.add_argument("--output", "-o", default="job.v2.yaml")
    parser.add_argument("--force", action="store_true")


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
    parser.add_argument("--dft-budget", type=int, default=8)
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Also run a resumable progressive Toy campaign for this many generations.",
    )
    parser.add_argument(
        "--workflow-config",
        default=None,
        help="Run the real configured training/MD workflow with Toy Teacher labeling.",
    )
    parser.add_argument("--training-steps", type=int, default=2)
    parser.add_argument("--md-steps", type=int, default=2)
    parser.add_argument("--force", action="store_true")


def build_iteration_stage(subparsers):
    parser = subparsers.add_parser(
        "iteration-stage",
        help="Run the next hash-checked iteration stage (for split Slurm jobs).",
    )
    parser.set_defaults(func=run_iteration_stage_command)
    _add_iteration_arguments(parser)
    parser.add_argument(
        "--stage",
        choices=[
            "train",
            "explore",
            "select",
            "label",
            "diagnose",
            "merge",
            "retrain",
            "evaluate",
        ],
        help="Expected stage; omit to execute the ledger's next stage.",
    )


def _add_iteration_arguments(parser):
    parser.add_argument("config_path", help="Schema-v2 workflow configuration.")
    parser.add_argument("--plan", required=True, help="GenerationPlan JSON file.")
    parser.add_argument(
        "--initial-training", required=True, help="Initial labeled extxyz dataset."
    )
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--campaign-id", required=True)


def build_iteration_resource(subparsers):
    parser = subparsers.add_parser(
        "iteration-resource",
        help="Resume every pending stage assigned to one Slurm resource class.",
    )
    parser.set_defaults(func=run_iteration_resource_command)
    _add_iteration_arguments(parser)
    parser.add_argument("--resource", choices=["training", "cpu"], required=True)


def build_campaign(subparsers):
    parser = subparsers.add_parser(
        "campaign",
        help="Prepare an iteration campaign and optionally submit its Slurm chain.",
    )
    parser.set_defaults(func=run_campaign_command)
    parser.add_argument(
        "config_path",
        nargs="?",
        help=(
            "Schema-v2 workflow configuration (not needed for prepared "
            "--submit, --retry-failed, or --status)."
        ),
    )
    parser.add_argument(
        "--initial-training", help="Initial labeled extxyz dataset."
    )
    parser.add_argument("--output", required=True, help="Durable campaign directory.")
    parser.add_argument("--campaign-id", default=None)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--submit",
        action="store_true",
        help="Submit all generated jobs as one Slurm afterok chain.",
    )
    action.add_argument(
        "--retry-failed",
        action="store_true",
        help="Cancel the stale dependency tail and resume from the ledger breakpoint.",
    )
    action.add_argument(
        "--status",
        action="store_true",
        help="Show ledger progress, live Slurm states, and the next safe action.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON with --status.",
    )
    parser.set_defaults(retry_failed=False, status=False)



def build_gpumd(subparsers):
    parser_gpumd = subparsers.add_parser(
        "gpumd",
        help="run molecular dynamics using GPUMD.",
    )
    parser_gpumd.set_defaults(func=run_gpumd)

    parser_gpumd.add_argument("model_path",
                             type=str,

                             help="The required structure path or structure file only supports files in xyz and vasp formats.")
    parser_gpumd.add_argument("--directory", "-dir",

                             type=str,
                             help="Set the GPUMD calculation path, default is ./cache/gpumd.",
                             default="./cache/gpumd"
                             )
    parser_gpumd.add_argument("--in","-in",dest="run_in_path", type=str,
                              help="The filename for the command _template file, default is ./run.in.", default="./run.in")

    parser_gpumd.add_argument("--nep", "-nep",
                            dest="nep_txt_path",
                             type=str,
                             help="Potential function path, default is ./nep.txt.",
                             default="./nep.txt"
                             )
    parser_gpumd.add_argument("--time", "-t", type=int, help="Molecular dynamics time, unit ps, default 10 ps.", default=10)
    parser_gpumd.add_argument("--temperature", "-T", type=int, help="Molecular dynamics temperature in Kelvin,multiple integers can be input. default is 300 K", nargs="*", default=[300])

    parser_gpumd.add_argument("--out", "-o",
                               dest="out_file_path",

                               type=str,
                               default="./trajectory.xyz",
                               help="Output path for structures."
                               )

def build_train(subparsers):
    parser_train = subparsers.add_parser(
        "train",
        help="Automatic training.",
    )
    parser_train.set_defaults(func=train_nep)

    parser_train.add_argument("config_path",
                             type=str,

                             help="The required structure path or structure file only supports files in XYZ and VASP formats.")


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
    parser_select.add_argument("--max_selected", "-max", type=int,
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

def main():
    parser = argparse.ArgumentParser(
        description="""
        NepTrain is a tool for automatically training NEP potential functions""",

    )
    parser.add_argument(
        "-v", "--version", action="version", version=__version__
    )



    subparsers = parser.add_subparsers()


    build_init(subparsers)

    build_perturb(subparsers)

    build_select(subparsers)
    build_dft(subparsers)

    build_vasp(subparsers)

    build_nep(subparsers)
    build_gpumd(subparsers)
    build_md(subparsers)
    build_train(subparsers)
    build_doctor(subparsers)
    build_migrate(subparsers)
    build_smoke(subparsers)
    build_iteration_stage(subparsers)
    build_iteration_resource(subparsers)
    build_campaign(subparsers)



    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:

        pass


    args = parser.parse_args()

    try:
        _ = args.func
    except AttributeError as exc:
        parser.print_help()
        raise SystemExit("Please specify a command.") from exc
    try:
        return args.func(args)
    except Exception as error:
        from NepTrain.core.config import ConfigError
        from NepTrain.core.md import MdError
        from NepTrain.core.spin import SpinDataError
        from NepTrain.core.training import TrainingError
        from NepTrain.core.dft import LabelingError
        from NepTrain.core.smoke import SmokeError
        from NepTrain.core.iteration import IterationError
        from NepTrain.core.workflow_iteration import WorkflowIterationError
        from NepTrain.core.campaign import CampaignError

        if isinstance(error, (ConfigError, MdError, SpinDataError, TrainingError, LabelingError, SmokeError, IterationError, WorkflowIterationError, CampaignError)):
            parser.exit(2, f"NepTrain: error: {error}\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
