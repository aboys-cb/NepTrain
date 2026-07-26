"""Create one strict schema-v7 project without touching existing files."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
import shutil

from ruamel.yaml import YAML

from NepTrain import utils
from .config import DEFAULT_MAX_CONCURRENT, DEFAULT_STRUCTURES_PER_DFT_JOB


def _project(
    profile: str,
    *,
    ensemble: str,
    spin: bool,
    dft_backend: str,
) -> dict:
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
        "schema_version": 7,
        "training": {
            "backend": "torchnep",
            "initial_path": "./train.xyz",
            "config_path": "./nep.in",
            "device": "cuda",
        },
        "md": {
            "backend": "lammps",
            "inference_backend": "auto",
            "spin": spin,
        },
        "sampling": {
            "routes": [
                {
                    "id": "default",
                    "structures": ["./structures"],
                    "template_path": "./lammps.in",
                    "conditions": {
                        "temperature_path": [300],
                    },
                },
            ],
        },
        "dft": {
            "backend": dft_backend,
            "input_path": "./INCAR" if dft_backend == "vasp" else "./INPUT",
            "resource_path": "./resources",
            **(
                {"potcar_manifest_path": "./vasp-resources.json"}
                if dft_backend == "vasp"
                else {"resource_manifest_path": "./abacus-resources.json"}
            ),
            "kpoint_mode": "auto",
            "structures_per_job": DEFAULT_STRUCTURES_PER_DFT_JOB,
            "max_concurrent": DEFAULT_MAX_CONCURRENT,
        },
        "workflow": {
            "id": "workflow",
            "max_model_generations": 12,
            "seed": 20260721,
        },
        "execution": {
            "stage_targets": routes,
            "targets": targets,
        },
    }


def init_project(
    profile: str,
    destination: str | Path,
    *,
    ensemble: str = "npt",
    spin: bool = False,
    dft_backend: str = "vasp",
    force: bool = False,
) -> Path:
    if ensemble not in {"npt", "nvt"}:
        raise ValueError("ensemble must be npt or nvt")
    if dft_backend not in {"vasp", "abacus"}:
        raise ValueError("dft_backend must be vasp or abacus")
    if spin and dft_backend != "abacus":
        raise ValueError("spin workflows require --dft-backend abacus")
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
        yaml.dump(
            _project(
                profile,
                ensemble=ensemble,
                spin=spin,
                dft_backend=dft_backend,
            ),
            handle,
        )
    (root / "structures").mkdir(exist_ok=True)
    if force:
        for obsolete in (
            "lammps-nvt.in",
            "lammps-npt.in",
            "lammps-spin-nvt.in",
            "lammps-spin-npt.in",
            "INCAR" if dft_backend == "abacus" else "INPUT",
            "vasp-resources.json",
            "abacus-resources.json",
        ):
            (root / obsolete).unlink(missing_ok=True)
    template_name = f"{'spin-' if spin else ''}{ensemble}.in"
    source = files("NepTrain.core.md").joinpath(f"templates/{template_name}")
    shutil.copyfile(source, root / "lammps.in")
    if dft_backend == "vasp":
        shutil.copyfile(
            files("NepTrain.core.dft.vasp").joinpath("INCAR"),
            root / "INCAR",
        )
        manifest = root / "vasp-resources.json"
        if not manifest.exists() or force:
            manifest.write_text(
                json.dumps(
                    {
                        "protocol": "neptrain.vasp-resources.v1",
                        "family": "PAW_PBE",
                        "release": "REPLACE_WITH_DISTRIBUTION_RELEASE",
                        "elements": {},
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    else:
        shutil.copyfile(
            files("NepTrain.core.dft.abacus").joinpath("INPUT"),
            root / "INPUT",
        )
        manifest = root / "abacus-resources.json"
        if not manifest.exists() or force:
            manifest.write_text(
                json.dumps(
                    {
                        "protocol": "neptrain.abacus-resources.v1",
                        "release": "REPLACE_WITH_RESOURCE_RELEASE",
                        "elements": {},
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
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
        f"Created {project}. Add train.xyz, nep.in and route structures; "
        "optionally configure evaluation, then run "
        "`neptrain doctor --project project.yaml`."
    )
    return project


def init_template(args):
    """Argparse adapter kept internal to the new ``workflow init`` command."""

    return init_project(
        args.profile,
        args.directory,
        ensemble=args.ensemble,
        spin=args.spin,
        dft_backend=args.dft_backend,
        force=args.force,
    )
