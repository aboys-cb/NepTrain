"""Create one strict schema-v8 project without touching existing files."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
import shutil
import shlex

from ruamel.yaml import YAML

from .config import DEFAULT_MAX_CONCURRENT, DEFAULT_STRUCTURES_PER_LABEL_JOB


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
            "label": {
                "executor": "slurm",
                "partition": "cpu",
                "time": "24:00:00",
                "cpus_per_task": 4,
                "setup_script": "./env-label.sh",
            },
        }
        routes = {
            "training": "v100",
            "sampling": "cpu",
            "labeling": "label",
            "analysis": "cpu",
        }
    return {
        "schema_version": 8,
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
        "labeling": {
            "backend": dft_backend,
            "input_path": "./INCAR" if dft_backend == "vasp" else "./INPUT",
            "resource_path": "./resources",
            **(
                {"potcar_manifest_path": "./vasp-resources.json"}
                if dft_backend == "vasp"
                else {"resource_manifest_path": "./abacus-resources.json"}
            ),
            "kpoint_mode": "auto",
            "structures_per_job": DEFAULT_STRUCTURES_PER_LABEL_JOB,
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
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    project = root / "project.yaml"
    generated = ["project.yaml", "lammps.in", "INCAR" if dft_backend == "vasp" else "INPUT"]
    existing = [name for name in generated if (root / name).exists()]
    if existing and not force:
        raise FileExistsError(
            f"项目文件已存在：{', '.join(existing)}（{root}）。"
            f"继续使用请运行 neptrain doctor --project {shlex.quote(str(project))}。"
            "需要重新生成时先备份；--force 会覆盖配置、模板和环境脚本，并重建资源清单。"
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
    with project.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n# 新项目使用 active_learning_v4；未配置 convergence 时不自动判定收敛。\n"
            "# 在上面的 workflow 下添加 convergence。以下仅示范语法，阈值须按目标体系确定：\n"
            "#   convergence:\n"
            "#     acquisition_max_rmse:\n"
            "#       energy_rmse: 0.01  # eV/atom\n"
            "#       force_rmse: 0.1    # eV/Angstrom\n"
            + ("#       mforce_rmse: 0.05  # eV/mu_B\n" if spin else "")
            + "#     min_selected: 10    # 不得超过 sampling.selection.max_selected\n"
            "#     consecutive_generations: 2\n"
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
            files("NepTrain.core.dft.vasp").joinpath("INCAR.deltaspin" if spin else "INCAR"),
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
            "env-label.sh": (
                "#!/bin/bash\n"
                "# Load the selected labeling Adapter and activate NepTrain here.\n"
            ),
        }
        for name, content in scripts.items():
            path = root / name
            if not path.exists() or force:
                path.write_text(content, encoding="utf-8")
                path.chmod(0o755)
    print(f"已创建项目：{project}")
    print("待补齐：带标签的 train.xyz、nep.in、structures/ 中的结构、资源清单及运行环境。")
    print("收敛：未启用自动收敛；project.yaml 末尾有带单位的配置示例。")
    print("可选：training.test_path 和 evaluation.validation_path 可省略。")
    print(f"下一步：neptrain doctor --project {shlex.quote(str(project))}")
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
