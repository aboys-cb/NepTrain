#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/25 18:12
# @Author  : 兵
# @email    : 1747193328@qq.com
import os.path

from ase.io import read as ase_read
from ruamel.yaml import YAML

from NepTrain import module_path, utils, __version__
from .utils import check_env

def get_job_config(job_type):
    with open(os.path.join(module_path, "core/train/_template/job.yaml"), "r", encoding="utf8") as f:
        base_config = YAML().load(f)

    with open(os.path.join(module_path, f"core/train/_template/{job_type}.yaml"), "r", encoding="utf8") as f:
        type_config = YAML().load(f)
    job = utils.merge_yaml(base_config, type_config)
    if "execution" in type_config:
        job["execution"] = type_config["execution"]
    return job
def init_template(argparse):
    if argparse.type=="bohrium":
        utils.print_tip(
            "The legacy Bohrium template requires "
            "'pip install NepTrain[bohrium]'."
        )


    if not argparse.force:
        utils.print_tip("For existing files, we choose to skip; if you need to forcibly generate and overwrite, please use -f or --force.")

    if not os.path.exists("./structure"):
        os.mkdir("./structure")
        utils.print_tip("Create the directory ./structure, please place the expanded structures that need to run MD into this folder!" )
    if not os.path.exists("./job.yaml") or argparse.force:
        utils.print_tip("Check the training, md, and dft sections in job.yaml before running.")


        config = get_job_config(argparse.type)
        config["version"]=__version__
        if os.path.exists("train.xyz"):
            #检查下第一个结构有没有计算
            atoms=ase_read("./train.xyz",0,format="extxyz")

            if not (atoms.calc and "energy"   in atoms.calc.results):
                config["current_job"]="dft"
                utils.print_warning("The first frame has no energy label; set current_job to dft.")
        else:
            utils.print_warning("Detected that there is no train.xyz in the current directory; please check the directory structure!")
            utils.print_tip("If there is a training set but the filename is not train.xyz, please unify the job.yaml.")


        with open("./job.yaml","w",encoding="utf8") as f:
            YAML().dump(config,f  )
    else:

        #已经存在 如果执行init  更新下
        base_config = get_job_config(argparse.type)

        with open("./job.yaml","r",encoding="utf8") as f:
            user_config = YAML().load(f)
        job=utils.merge_yaml(base_config,user_config)
        job["version"]=__version__

        with open("./job.yaml","w",encoding="utf8") as f:
            YAML().dump(job,f  )


    if not os.path.exists("./run.in")  or argparse.force:
        utils.print_tip("Create the default GPUMD run.in template.")

        utils.copy(os.path.join(module_path,"core/gpumd/run.in"),"./run.in")

    for template_name in ("nvt.in", "npt.in", "spin-nvt.in", "spin-npt.in"):
        destination = f"./lammps-{template_name}"
        if not os.path.exists(destination) or argparse.force:
            utils.copy(
                os.path.join(module_path, "core/md/templates", template_name),
                destination,
            )

    if argparse.type in {"pbs", "bohrium"}:
        utils.print_warning(
            "PBS and Bohrium templates are legacy compatibility inputs. "
            "The persistent campaign controller currently supports process and Slurm targets."
        )
    utils.print_success(
        "Initialization is complete. Check job.yaml, run "
        "`NepTrain doctor --project job.yaml`, then start with "
        "`NepTrain run job.yaml --output <campaign>`."
    )
