"""Create one strict schema-v4 project without touching existing files."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
import shutil

from ruamel.yaml import YAML

from NepTrain import utils


def _project(profile: str) -> dict:
    if profile == "local":
        targets = {"local": {"executor": "process"}}
        routes = {
            "training": "local",
            "sampling": "local",
            "labeling": "local",
            "analysis": "local",
        }
    else:
        targets = {
            "v100": {
                "executor": "slurm",
                "partition": "gpu",
                "time": "24:00:00",
                "gpus_per_node": 1,
                "setup_script": "./env-training.sh",
            },
            "cpu": {
                "executor": "slurm",
                "partition": "cpu",
                "time": "04:00:00",
                "cpus_per_task": 4,
                "setup_script": "./env-cpu.sh",
            },
            "dft": {
                "executor": "slurm",
                "partition": "cpu",
                "time": "24:00:00",
                "cpus_per_task": 4,
                "setup_script": "./env-dft.sh",
            },
        }
        routes = {
            "training": "v100",
            "sampling": "cpu",
            "labeling": "dft",
            "analysis": "cpu",
        }
    return {
        "schema_version": 4,
        "training": {
            "backend": "torchnep",
            "initial_path": "./train.xyz",
            "config_path": "./nep.in",
            "device": "cuda",
            "torch_backend": "auto",
            "precision": "float32",
            "use_compile": False,
            "finetune_lr_scale": 0.1,
            "seed": 20260723,
        },
        "md": {
            "backend": "lammps",
            "inference_backend": "auto",
            "structures": "./structures",
            "template_path": "./lammps-nvt.in",
            "spin": False,
            "lmp": "lmp",
            "mpiexec": "mpirun",
            "mpi_ranks": 1,
        },
        "sampling": {
            "mode": "auto",
            "conditions": {
                "temperature_path": [300],
                "production_temperatures": [300],
                "pressure": 0.0,
                "spin_temperature": None,
            },
            "progression": {
                "md_runs_per_iteration": 4,
                "steps": {
                    "smoke_passed": 10000,
                    "short_stable": 40000,
                    "long_stable": 160000,
                    "production_ready": 640000,
                },
                "replicas": {
                    "smoke_passed": 1,
                    "short_stable": 1,
                    "long_stable": 2,
                    "production_ready": 3,
                },
            },
            "candidate_pool": {
                "target": 200,
                "growth": 1.0,
                "frame_stride": 2,
                "pre_failure_frames": 2,
                "bad_tail_frames": 1,
                "health": {
                    "min_distance_ratio": 0.5,
                    "min_volume_ratio": 0.5,
                    "max_volume_ratio": 2.0,
                    "max_force": 100.0,
                    "max_mforce": 100.0,
                    "max_spin_magnitude": 20.0,
                },
            },
            "selection": {
                "method": "fps",
                "dft_budget": 20,
                "minimum_dft_budget": 8,
                "budget_decay": 0.75,
                "min_novelty": 0.0,
            },
        },
        "dft": {
            "backend": "vasp",
            "n_cpu": 1,
            "kpoints_use_gamma": True,
            "input_path": "./INCAR",
            "resource_path": None,
            "use_k_stype": "kspacing",
            "kspacing": 0.2,
        },
        "evaluation": {
            "validation_path": "./validation.xyz",
            "inference_backend": "auto",
            "max_rmse": {
                "energy_rmse": 0.05,
                "force_rmse": 0.2,
            },
        },
        "workflow": {
            "id": "workflow",
            "max_iterations": 12,
            "seed": 20260721,
        },
        "execution": {
            "poll_interval": 30,
            "routes": routes,
            "targets": targets,
        },
    }


def init_project(profile: str, destination: str | Path, *, force: bool = False) -> Path:
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    project = root / "project.yaml"
    if project.exists() and not force:
        raise FileExistsError(
            f"{project} already exists; use --force to replace generated templates"
        )
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    with project.open("w", encoding="utf-8") as handle:
        yaml.dump(_project(profile), handle)
    (root / "structures").mkdir(exist_ok=True)
    for name in ("nvt.in", "npt.in", "spin-nvt.in", "spin-npt.in"):
        source = files("NepTrain.core.md").joinpath(f"templates/{name}")
        shutil.copyfile(source, root / f"lammps-{name}")
    shutil.copyfile(files("NepTrain.core.dft.vasp").joinpath("INCAR"), root / "INCAR")
    shutil.copyfile(files("NepTrain.core.dft.abacus").joinpath("INPUT"), root / "INPUT")
    if profile == "slurm":
        scripts = {
            "env-training.sh": (
                "#!/bin/bash\n"
                "# Load PyTorch/TorchNEP and activate the NepTrain environment here.\n"
                "# module load cuda\n"
            ),
            "env-cpu.sh": (
                "#!/bin/bash\n"
                "# Load LAMMPS/NEPAdapters and activate the NepTrain environment here.\n"
                "# module load lammps/nep-release\n"
                "# export LAMMPS_PLUGIN_PATH=/path/to/nepadapters/lib\n"
            ),
            "env-dft.sh": (
                "#!/bin/bash\n"
                "# Load VASP or ABACUS and activate the NepTrain environment here.\n"
            ),
        }
        for name, content in scripts.items():
            path = root / name
            if not path.exists() or force:
                path.write_text(content, encoding="utf-8")
                path.chmod(0o755)
    utils.print_success(
        f"Created {project}. Add train.xyz, validation.xyz, nep.in and structures; "
        "then run `neptrain doctor --project project.yaml`."
    )
    return project


def init_template(args):
    """Argparse adapter kept internal to the new ``workflow init`` command."""

    return init_project(args.profile, args.directory, force=args.force)
