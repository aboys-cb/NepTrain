#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/25 13:37
# @Author  : 兵
# @email    : 1747193328@qq.com
"""
自动训练的逻辑
"""
import os.path
import shutil
import shlex
import subprocess
import asyncio
from pathlib import Path
from typing import List, Tuple
from ase.io import read as ase_read
from ase.io import write as ase_write
from ruamel.yaml import YAML

from NepTrain import utils
from ..config import load_config, save_config
from ..spin import validate_spin_dataset

from ..utils import check_env


def submit_job(*args, **kwargs):
    from .worker import submit_job as implementation

    return implementation(*args, **kwargs)


async def async_submit_job(*args, **kwargs):
    from .worker import async_submit_job as implementation

    return await implementation(*args, **kwargs)


async def _await_tasks(tasks):
    """Helper coroutine to run multiple async tasks."""
    await asyncio.gather(*tasks)

def filter_file_path(forward_files,base_dir=""):
    new_files = []
    for forward_file in forward_files:
        if os.path.exists(os.path.join(base_dir,forward_file)):
            new_files.append(forward_file)
    return new_files
def relpath_from_files(files, start):
    if isinstance(files, (list, tuple)):
        return [os.path.relpath(file,start) for file in files]
    return os.path.relpath(files, start)
PARAMS = Tuple[str,list ]



class Manager:
    def __init__(self, options):
        self.options = options
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.options):
            self.index = 0
        value = self.options[self.index]
        self.index += 1
        return value

    def set_next(self, option):
        index=self.options.index(option)
        # 设置当前索引，注意索引从0开始
        if 0 <= index < len(self.options):
            self.index = index
        else:
            raise IndexError("Index out of range.")

    def peek(self):
        return self.options[self.index % len(self.options)]


class PathManager:



    def __init__(self, root):
        self.root = root

    def __getattr__(self, item):
        return os.path.join(self.root, item)

def params2str(params):
    tokens = []
    for item in params:
        if isinstance(item, (tuple, list)):
            tokens.extend(str(value) for value in item)
        else:
            tokens.append(str(item))
    return shlex.join(tokens)

class NepTrainWorker:
    pass
    def __init__(self):
        self.config={}
        self.job_list=["training","md","select","dft","pred", ]
        self.manager=Manager(self.job_list)



    def __getattr__(self, item):

        if item.startswith("last_"):
            item=item.replace("last_","")
            generation_path=os.path.join((self.config.get("work_path")), f"Generation-{self.generation-1}")
        else:
            generation_path=os.path.join((self.config.get("work_path")), f"Generation-{self.generation}")

        if item=="generation_path":

            return generation_path

        items= item.split("_")
        path_alias = {"nep": "training", "gpumd": "md"}
        path_job = path_alias.get(items[0], items[0])
        if path_job in self.job_list:
            items.pop(0)
            job_path=os.path.join(generation_path, path_job)
        else:
            job_path=generation_path
        fin_path=os.path.join(job_path, "_".join(items[:-1]) )
        if items[-1]=="path":
            pass
            utils.verify_path(fin_path)
        else:
            last_underscore_index = fin_path.rfind('_')
            if last_underscore_index != -1:
                # 替换最后一个下划线为点
                fin_path = fin_path[:last_underscore_index] + '.' + fin_path[last_underscore_index + 1:]
            else:
                fin_path = fin_path

            utils.verify_path(os.path.dirname(fin_path))


        return fin_path



    @property
    def generation(self):
        return self.config.get("generation")
    @generation.setter
    def generation(self,value):
        self.config["generation"] = value



    def split_dft_job_xyz(self,xyz_file):
        addxyz = ase_read(xyz_file, ":", format="extxyz")

        split_addxyz_list = utils.split_list(addxyz, self.config["dft_job"])


        for i, xyz in enumerate(split_addxyz_list):
            if xyz:
                ase_write(self.__getattr__(f"dft_learn_add_{i + 1}_xyz_file"), xyz, format="extxyz")

    def check_env(self):


        if self.config.get("restart") :
            utils.print("No need for initialization check.")
            utils.print_msg("--" * 4,
                            f"Restarting to train the potential function for the {self.generation}th generation.",
                            "--" * 4)

            return

        if self.config["current_job"]=="dft":

            self.generation=0
            utils.copy(self.config["init_train_xyz"], self.select_selected_xyz_file)

            # if self.config["dft_job"] != 1:
            #
            #
            #     self.split_dft_job_xyz(self.config["init_train_xyz"])
        elif self.config["current_job"]=="training":
           

            utils.copy(self.config["init_train_xyz"], self.last_all_learn_calculated_xyz_file)
            # utils.copy(self.config["init_train_xyz"], self.last_all_learn_calculated_xyz_file )
            #如果势函数有效  直接先复制过来
        elif self.config["current_job"]=="md":

            utils.copy(self.config["init_train_xyz"],self.nep_train_xyz_file )

            if os.path.exists(self.config["init_nep_txt"]):
                utils.copy(self.config["init_nep_txt"],
                            self.nep_nep_txt_file )
            else:
                raise FileNotFoundError("Starting task as gpumd requires specifying a valid potential function path!")
        else:
            raise ValueError("current_job can only be training, md, or dft.")

    def read_config(self,config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"The file at {config_path} does not exist.")
        self.config, changes = load_config(config_path)
        if changes:
            utils.print_tip("Config migrated in memory: " + ", ".join(changes))
        if self.config["dft"]["software"] == "toy":
            self.config["dft"]["incar_path"] = None
        elif self.config["dft"]["incar_path"]=="auto":
            if self.config["dft"]["software"]=="abacus":
                self.config["dft"]["incar_path"]="./INPUT"
            else:
                self.config["dft"]["incar_path"]="./INCAR"
            
    def build_pred_params(self):
        nep=self.config["training"]

        utils.copy(nep.get("config_path"), self.pred_nep_in_file)
        utils.copy(self.nep_nep_txt_file, self.pred_nep_txt_file)
        utils.copy(self.all_learn_calculated_xyz_file, self.pred_train_xyz_file)

        params=[]
        params.append("NepTrain")
        params.append("nep")

        params.append("--directory")
        params.append("./")

        params.append("--in")
        params.append("nep.in")

        params.append("--train")
        params.append("train.xyz")

        params.append("--nep")
        params.append("nep.txt")

        params.append("--prediction")

        return params2str(params)


    def build_nep_params(self) :
        nep=self.config["training"]

        utils.copy(self.last_improved_train_xyz_file, self.nep_train_xyz_file)
        utils.copy(nep.get("config_path"), self.nep_nep_in_file)
        utils.copy(nep.get("test_path"), self.nep_test_xyz_file)
        if nep.get("backend") == "torchnep":
            utils.copy(self.last_nep_checkpoint_pt_file, self.nep_checkpoint_pt_file)
        else:
            utils.copy(self.last_nep_nep_restart_file, self.nep_nep_restart_file)

        params=[]
        params.append("NepTrain")
        params.append("nep")
        params.extend(["--backend", nep.get("backend", "gpumd")])

        params.append("--directory")
        params.append("./")

        params.append("--in")
        params.append("nep.in")


        params.append("--train")
        params.append("train.xyz")

        # params.append(relpath_from_files(self.last_improved_train_xyz_file,self.nep_path))

        if nep.get("test_path"):
            params.append("--test")
            params.append("test.xyz")
        #
        # params.append(relpath_from_files(nep.get("test_xyz_path"),self.nep_path))

        if nep.get("restart", True) and self.generation not in [1,len(self.config["md"]["duration_ps_every_generation"])+1]:
            #开启续跑
            #如果上一级的势函数路径有效  就传入一下续跑的参数

            restart_file = (
                self.last_nep_checkpoint_pt_file
                if nep.get("backend") == "torchnep"
                else self.last_nep_nep_restart_file
            )
            if os.path.exists(restart_file):
                utils.print_tip("Start the restart mode!")
                params.append("--restart_file")
                params.append("checkpoint.pt" if nep.get("backend") == "torchnep" else "nep.restart")
                # params.append(relpath_from_files(self.last_nep_nep_restart_file,self.nep_path))
                params.append("--continue_step")
                params.append(nep.get("restart_steps", 20000))
        if nep.get("backend") == "torchnep":
            params.extend(["--device", nep.get("device", "cuda")])
            params.extend(["--torch-backend", nep.get("torch_backend", "auto")])
            params.extend(["--precision", nep.get("precision", "float32")])
            if nep.get("use_compile", False):
                params.append("--compile")

        return params2str(params)
    def build_gpumd_params(self,model_path,temperature,n_job=1,):
        gpumd=self.config["md"]
        base_name = os.path.basename(model_path)
        utils.copy(model_path, os.path.join(self.gpumd_path, base_name))

        utils.copy(gpumd.get("template_path"), self.gpumd_run_in_file)

        utils.copy(self.nep_nep_txt_file, self.gpumd_nep_txt_file)

        params=[]
        params.append("NepTrain")
        params.append("md")

        params.append(base_name)

        params.extend(["--backend", gpumd.get("backend", "gpumd")])

        params.append("--directory")

        params.append("./")

        if gpumd.get("template_path"):
            params.append("--template")
            params.append("run.in")
        params.append("--nep")
        params.append( "nep.txt")
        duration_ps = gpumd.get("duration_ps_every_generation")[self.generation-1]
        timestep = float(gpumd.get("timestep", 0.001))
        params.append("--steps")
        params.append(int(float(duration_ps) / timestep))
        params.extend(["--timestep", timestep])

        params.append("--temperature")

        params.append(temperature)

        params.extend(["--ensemble", gpumd.get("ensemble", "nvt")])
        params.extend(["--pressure", gpumd.get("pressure", 0.0)])
        params.extend(["--dump-interval", gpumd.get("dump_interval", 100)])
        params.extend(["--inference-backend", gpumd.get("inference_backend", "auto")])
        if gpumd.get("spin", False):
            params.append("--spin")
            params.extend(["--spin-temperature", gpumd["spin_temperature"]])
            params.extend(["--spin-alpha", gpumd.get("spin_alpha", 0.01)])
            params.extend(["--spin-seed", gpumd.get("spin_seed", 12345)])
            params.extend(["--midpoint-iter", gpumd.get("midpoint_iter", 3)])
        if gpumd.get("backend") == "lammps":
            params.extend(["--lmp", gpumd.get("lmp", "lmp")])
            params.extend(["--mpiexec", gpumd.get("mpiexec", "mpirun")])
            params.extend(["--mpi-ranks", gpumd.get("mpi_ranks", 1)])
            if gpumd.get("plugin_path"):
                params.extend(["--plugin-path", gpumd["plugin_path"]])




        params.append("--out")
        params.append( f"./trajectory_{n_job}.xyz")




        return params2str(params)
    def build_select_params(self):
        select=self.config["select"]
        utils.copy(self.nep_nep_txt_file, self.select_nep_txt_file)

        utils.copy(self.nep_train_xyz_file, self.select_train_xyz_file)

        params=[]
        params.append("NepTrain")
        params.append("select")
        #总的
        params.append("trajectorys.xyz")
        #分开
        # params.append(relpath_from_files(self.__getattr__(f"select_md_*_xyz_file"),self.select_path ))
        params.append("--nep")
        params.append( relpath_from_files(self.select_nep_txt_file,self.select_path ))

        params.append("--base")
        params.append( relpath_from_files(self.select_train_xyz_file ,self.select_path ))
        params.append("--max_selected")
        params.append(select["max_selected"])
        params.append("--min_distance")
        params.append(select["min_distance"])
        params.append("--out")
        params.append(relpath_from_files(self.select_selected_xyz_file,self.select_path ))

        if select.get("filter",False):

            params.append("--filter")
            params.append(select.get("filter" ) if isinstance(select.get("filter" ),float) else 0.6)


        return params2str(params)


    def build_dft_params(self,n_job=1):
        dft=self.config["dft"]

        if dft.get("incar_path"):
            utils.copy(dft["incar_path"], self.dft_path)

        params=[]
        params.append("NepTrain")
        params.append("dft")

        if self.config["dft_job"] == 1:

            if not os.path.exists(self.dft_learn_add_xyz_file):
                return None
            params.append(relpath_from_files(self.dft_learn_add_xyz_file,self.dft_path ))
        else:
            path=self.__getattr__(f"dft_learn_add_{n_job}_xyz_file")
            if not os.path.exists(path):
                return None
            params.append(relpath_from_files(path,self.dft_path ))

        params.append("--directory")

        params.append(relpath_from_files(self.__getattr__(f"dft_cache{n_job}_path"),self.dft_path ))


        params.append("-np")
        params.append(dft["cpu_core"])
        if dft["kpoints_use_gamma"]:
            params.append("--gamma")

        if dft["incar_path"]:

            params.append("--in")

            params.append(os.path.basename(dft["incar_path"]))
        if dft["use_k_stype"] == "kpoints":
            if dft.get("kpoints"):
                params.append("-ka")
                if isinstance(dft["kpoints"],list):
                    params.append(",".join([str(i) for i in dft["kpoints"]]))
                else:
                    params.append(dft["kpoints"])
        else:

            if dft.get("kspacing") :
                params.append("--kspacing")
                params.append(dft["kspacing"])
        # params.append("--software")
        params.append("--" + dft["software"])
        if dft["software"] == "toy":
            params.extend(["--teacher-profile", dft.get("teacher_profile", "ordinary")])
        params.append("--out")
        params.append( relpath_from_files(self.__getattr__(f"dft_learn_calculated_{n_job}_xyz_file"),self.dft_path ))


        return params2str(params)
    def sub_select(self):
        # utils.cat(self.__getattr__(f"select_md_*_xyz_file"),
        #           self.select_all_md_dummp_xyz_file
        #           )
        utils.print_msg(f"Start sampling from the trajectory.")
        utils.cat(self.__getattr__(f"gpumd_trajectory_*_xyz_file"),
                  self.select_trajectorys_xyz_file
                  )

        if utils.is_file_empty(self.select_trajectorys_xyz_file):
            utils.print_warning(f"No trajectory file, skip sampling")

            return

        cmd = self.build_select_params()


        submit_job(
            machine_dict=self.config["select"]["machine"],
            resources_dict=self.config["select"]["resources"],
            task_dict_list=[
                dict(
                    command=cmd,
                    task_work_path="./",
                    forward_files=filter_file_path(["nep.txt", "train.xyz","trajectorys.xyz"],self.select_path),
                    backward_files=relpath_from_files([
                        self.select_selected_xyz_file,
                        self.select_selected_png_file,
                                    self.__getattr__(f"select_selected_md_*_*_file")
                                    ],self.select_path),
                )
            ],
            submission_dict=dict(
                work_base=self.select_path,
                forward_common_files=[],
                backward_common_files=[],

            )

        )








    def sub_dft(self):
        utils.print_msg(
            f"Beginning {self.config['dft']['software']} single-point labeling."
        )
        # break
        utils.cat(self.select_selected_xyz_file,
                  self.dft_learn_add_xyz_file
                  )

        if not utils.is_file_empty(self.dft_learn_add_xyz_file):
            if self.config["dft"]["software"] == "abacus":
                from NepTrain.core.dft.abacus import StructureVar
                StructureVar.init(self.config["md"]["structures"])
                StructureVar.init("./", reset=False)
                for pp in StructureVar.pp_files.values():
                    curr_p=f"./{pp}"
                    stru_p=f'{self.config["md"]["structures"]}/{pp}'
                    if os.path.exists(curr_p):
                        shutil.copy(curr_p,self.dft_path)
                    elif os.path.exists(stru_p):
                        shutil.copy(stru_p,self.dft_path)
                for orb in StructureVar.orbs.values():
                    curr_orb=f"./{orb}"
                    stru_orb=f'{self.config["md"]["structures"]}/{orb}'
                    if os.path.exists(curr_orb):
                        shutil.copy(curr_orb,self.dft_path)
                    elif os.path.exists(stru_orb):
                        shutil.copy(stru_orb,self.dft_path)
            if self.config["dft_job"] != 1:
                # Split xyz for parallel submission
                self.split_dft_job_xyz(self.dft_learn_add_xyz_file)

            tasks = []
            for i in range(self.config["dft_job"]):
                cmd = self.build_dft_params(i + 1)
                if cmd is None:
                    continue
                input_name = (
                    [os.path.basename(self.config["dft"]["incar_path"])]
                    if self.config["dft"].get("incar_path")
                    else []
                )
                if self.config["dft_job"] == 1:
                    forward_files=["learn_add.xyz", *input_name]
                else:
                    forward_files=[f"learn_add_{i + 1}.xyz", *input_name]

                tasks.append(
                    async_submit_job(
                        machine_dict=self.config["dft"]["machine"],
                        resources_dict=self.config["dft"]["resources"],
                        task_dict_list=[
                            dict(
                                command=cmd,
                                task_work_path="./",
                                forward_files=filter_file_path(forward_files,self.dft_path),
                                backward_files=[f"learn_calculated_{i +1}.xyz"],
                            )
                        ],
                        submission_dict=dict(
                            work_base=self.dft_path,
                            forward_common_files=[],
                            backward_common_files=[],
                        ),
                    )
                )

            if tasks:
                asyncio.run(_await_tasks(tasks))

            utils.cat(self.__getattr__(f"dft_learn_calculated_*_xyz_file"),
                      self.all_learn_calculated_xyz_file
                      )
            if not utils.is_file_empty(self.all_learn_calculated_xyz_file):
                validate_spin_dataset(
                    ase_read(self.all_learn_calculated_xyz_file, ":", format="extxyz"),
                    require_mforce=True,
                )
            if self.config.get("limit",{}).get("force") and not utils.is_file_empty(self.all_learn_calculated_xyz_file):
                bad_structure = []
                good_structure = []
                structures=ase_read(self.all_learn_calculated_xyz_file,":")
                for structure in structures:

                    if abs(structure.calc.results["forces"]).max() <= self.config.get("limit",{}).get("force"):
                        good_structure.append(structure)
                    else:
                        bad_structure.append(structure)

                ase_write(self.all_learn_calculated_xyz_file,good_structure,append=False,format="extxyz")
                if bad_structure:
                    ase_write(self.remove_by_force_xyz_file, bad_structure, append=False, format="extxyz")

        else:
            utils.print_warning("Detected that the calculation input file is empty, proceeding directly to the next step!")

            utils.cat(self.dft_learn_add_xyz_file,
                      self.all_learn_calculated_xyz_file
                      )

    def sub_nep(self):
        utils.print_msg("--" * 4, f"Starting to train the potential function for the {self.generation}th generation.", "--" * 4)

        if not utils.is_file_empty(self.last_all_learn_calculated_xyz_file):


            if os.path.exists(self.last_nep_train_xyz_file):
                utils.cat([self.last_nep_train_xyz_file,
                           self.last_all_learn_calculated_xyz_file
                           ],
                          self.last_improved_train_xyz_file

                          )
            else:
                utils.copy(self.last_all_learn_calculated_xyz_file,
                            self.last_improved_train_xyz_file)

            utils.print_msg(f"Starting to train the potential function.")
            cmd = self.build_nep_params()


            submit_job(
                machine_dict = self.config["training"]["machine"],
                resources_dict = self.config["training"]["resources"],
                task_dict_list = [
                    dict(
                        command=cmd,
                        task_work_path="./",
                        forward_files= filter_file_path(["nep.in","nep.restart","checkpoint.pt","train.xyz","test.xyz"],self.nep_path),
                        backward_files = ["nep*.txt", "nep.restart", "*.pt", "loss.out", "output.log"],
                    )
                ],
                submission_dict = dict(
                    work_base=self.nep_path,
                    forward_common_files=[],
                    backward_common_files=[],

                )

            )

        else:
            utils.print_warning("The dataset has not changed, directly copying the potential function from the last time!")

            utils.copy_files(self.last_nep_path, self.nep_path)

    def sub_nep_pred(self):

        if utils.is_file_empty(self.nep_nep_txt_file):
            utils.print_msg(f"No potential function available, skipping prediction.")
            return
        if not utils.is_file_empty(self.all_learn_calculated_xyz_file):
            utils.print_msg(f"Starting to predict new dataset.")
            cmd = self.build_pred_params()
            submit_job(
                machine_dict=self.config["training"]["machine"],
                resources_dict=self.config["training"]["resources"],
                task_dict_list=[
                    dict(
                        command=cmd,
                        task_work_path="./",
                        forward_files=filter_file_path(["nep.in","nep.txt","train.xyz"],self.pred_path),
                        backward_files=["energy_train.out", "force_train.out", "virial_train.out", "mforce_train.out"],
                    )
                ],
                submission_dict=dict(
                    work_base=self.pred_path,
                    forward_common_files=[],
                    backward_common_files=[],
                ),
            )
        else:
            utils.print_msg(f"The dataset has not changed, skipping prediction.")


    def sub_gpumd(self):


        utils.print_msg(f"Starting active learning.")
        tasks = []
        if self.config.get("md_split_job", "temperature") == "temperature":
            for i, temp in enumerate(self.config["md"]["temperatures"]):
                cmd = self.build_gpumd_params(
                    self.config["md"].get("structures"),
                    temp,
                    i,
                )
                base_name=os.path.basename(self.config["md"].get("structures"))
                tasks.append(
                    async_submit_job(
                        machine_dict=self.config["md"]["machine"],
                        resources_dict=self.config["md"]["resources"],
                        task_dict_list=[
                            dict(
                                command=cmd,
                                task_work_path="./",
                                forward_files=filter_file_path(["run.in","nep.txt",base_name],self.gpumd_path),
                                backward_files=[f"./trajectory_{i}.xyz"],
                            )
                        ],
                        submission_dict=dict(
                            work_base=self.gpumd_path,
                            forward_common_files=[],
                            backward_common_files=[],
                        ),
                    )
                )
        else:
            if os.path.isdir(self.config["md"]["structures"]):
                for i, file in enumerate(os.listdir(self.config["md"]["structures"])):
                    cmd = self.build_gpumd_params(
                        os.path.join(self.config["md"]["structures"], file),
                        self.config["md"]["temperatures"],
                        i,
                    )
                    tasks.append(
                        async_submit_job(
                            machine_dict=self.config["md"]["machine"],
                            resources_dict=self.config["md"]["resources"],
                            task_dict_list=[
                                dict(
                                    command=cmd,
                                    task_work_path="./",
                                    forward_files=["run.in","nep.txt",file],
                                    backward_files=[f"./trajectory_{i}.xyz"],
                                )
                            ],
                            submission_dict=dict(
                                work_base= self.gpumd_path,
                                forward_common_files=[],
                                backward_common_files=[],
                            ),
                        )
                    )
        if tasks:
            asyncio.run(_await_tasks(tasks))

        # utils.cat(self.__getattr__(f"gpumd_trajectory_*_xyz_file"),
        #           self.select_trajectorys_xyz_file
        #           )


    def start(self,config_path):
        utils.print_msg("Welcome to NepTrain automatic training!")

        self.read_config(config_path)
        self.check_env()



        self.manager.set_next(self.config.get("current_job"))

        while True:

            #开始循环
            job = next(self.manager)
            # utils.print_msg(f"[Generation {self.generation}] Starting job: {job}")
            self.config["current_job"]=job
            if job=="dft":

                self.sub_dft()

            elif job=="pred":

                self.sub_nep_pred()
                self.generation += 1

            elif job=="training":

                self.sub_nep()
                if self.generation>len(self.config["md"]["duration_ps_every_generation"]):
                   utils.print_success("Training completed!")
                   break
            elif job=="select":

                self.sub_select()

            else:
                if utils.is_file_empty(self.nep_nep_txt_file):
                    utils.print_warning(f"No potential function available, break!!!")
                    break
                self.sub_gpumd()

            self.config["current_job"] = self.manager.peek()
            self.save_restart()


    def save_restart(self):
        self.config["restart"]=True
        save_config(self.config, "./restart.yaml")

def train_nep(argparse):
    """
    首先检查下当前的进度 看从哪开始
    :return:
    """


    worker = NepTrainWorker()

    worker.start(argparse.config_path)
if __name__ == '__main__':
    train =NepTrainWorker()
    train.generation=1
    train.config["work_path"]="./cache"
    print(train.nep_path)

    print(train.__getattr__(f"dft_learn_calculated_*_xyz_file"))
