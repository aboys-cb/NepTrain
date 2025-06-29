#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/25 13:37
# @Author  : 兵
# @email    : 1747193328@qq.com
"""
自动训练的逻辑
"""
import os.path
import subprocess
import asyncio
from pathlib import Path
from typing import List, Tuple
from ase.io import read as ase_read
from ase.io import write as ase_write
from .worker import submit_job, async_submit_job
from ruamel.yaml import YAML

from NepTrain import utils

from ..utils import check_env


async def _await_tasks(tasks):
    """Helper coroutine to run multiple async tasks."""
    await asyncio.gather(*tasks)

def filter_file_path(forward_files):
    new_files = []
    for forward_file in forward_files:
        if os.path.exists(forward_file):
            new_files.append(forward_file)
    return new_files


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


class PathManager:



    def __init__(self, root):
        self.root = root

    def __getattr__(self, item):
        return os.path.join(self.root, item)

def params2str(params):
    text=""
    for i in  params:
        if isinstance(i, str):
            text += i
        elif isinstance(i, (tuple,list)):
            for j in i:
                text += str(j)
                text += " "
        else:
            text += str(i)
        text += " "
    return text

class NepTrainWorker:
    pass
    def __init__(self):
        self.config={}
        self.job_list=["nep","gpumd","select","vasp","pred", ]
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
        if items[0] in self.job_list:
            job_path=os.path.join(generation_path, items.pop(0))
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



    def split_vasp_job_xyz(self,xyz_file):
        addxyz = ase_read(xyz_file, ":", format="extxyz")

        split_addxyz_list = utils.split_list(addxyz, self.config["vasp_job"])

        for i, xyz in enumerate(split_addxyz_list):
            if xyz:
                ase_write(self.__getattr__(f"vasp_learn_add_{i + 1}_xyz_file"), xyz, format="extxyz")

    def check_env(self):


        if self.config.get("restart") :
            utils.print("No need for initialization check.")
            utils.print_msg("--" * 4,
                            f"Restarting to train the potential function for the {self.generation}th generation.",
                            "--" * 4)

            return

        if self.config["current_job"]=="vasp":

            self.generation=0
            utils.copy(self.config["init_train_xyz"], self.vasp_learn_add_xyz_file)

            if self.config["vasp_job"] != 1:


                self.split_vasp_job_xyz(self.config["init_train_xyz"])
        elif self.config["current_job"]=="nep":
           

            utils.copy(self.config["init_train_xyz"], self.last_all_learn_calculated_xyz_file)
            # utils.copy(self.config["init_train_xyz"], self.last_all_learn_calculated_xyz_file )
            #如果势函数有效  直接先复制过来
        elif self.config["current_job"]=="gpumd":

            utils.copy(self.config["init_train_xyz"],self.nep_train_xyz_file )

            if os.path.exists(self.config["init_nep_txt"]):
                utils.copy(self.config["init_nep_txt"],
                            self.nep_nep_txt_file )
            else:
                raise FileNotFoundError("Starting task as gpumd requires specifying a valid potential function path!")
        else:
            raise ValueError("current_job can only be one of nep, gpumd, or vasp.")

    def read_config(self,config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"The file at {config_path} does not exist.")
        with open(config_path,"r",encoding="utf8") as f:


            self.config=YAML().load(f )

    def build_pred_params(self):
        nep=self.config["nep"]
        params=[]
        params.append("NepTrain")
        params.append("nep")

        params.append("--directory")
        params.append(self.pred_path)

        params.append("--in")
        params.append((nep.get("nep_in_path")))

        params.append("--train")
        params.append(self.all_learn_calculated_xyz_file)

        params.append("--nep")
        params.append(self.nep_nep_txt_file)

        params.append("--prediction")

        return params2str(params)


    def build_nep_params(self) :
        nep=self.config["nep"]
        params=[]
        params.append("NepTrain")
        params.append("nep")

        params.append("--directory")
        params.append(self.nep_path)

        params.append("--in")
        # params.append("nep.in")

        params.append((nep.get("nep_in_path")))

        params.append("--train")
        # params.append("train.xyz")

        params.append(self.last_improved_train_xyz_file)

        params.append("--test")
        # params.append("test.xyz")

        params.append((nep.get("test_xyz_path")))

        if self.config["nep"]["nep_restart"] and self.generation not in [1,len(self.config["gpumd"]["step_times"])+1]:
            #开启续跑
            #如果上一级的势函数路径有效  就传入一下续跑的参数

            if os.path.exists(self.last_nep_nep_restart_file):
                utils.print_tip("Start the restart mode!")


                params.append("--restart_file")
                params.append(self.last_nep_nep_restart_file)
                params.append("--continue_step")
                params.append(self.config["nep"]["nep_restart_step"])

        return params2str(params)
    def build_gpumd_params(self,model_path,temperature,n_job=1,):
        gpumd=self.config["gpumd"]
        params=[]
        params.append("NepTrain")
        params.append("gpumd")

        params.append((model_path))

        params.append("--directory")

        params.append(self.gpumd_path)

        params.append("--in")
        params.append((gpumd.get("run_in_path")))
        params.append("--nep")
        params.append( self.nep_nep_txt_file)
        params.append("--time")
        params.append(gpumd.get("step_times")[self.generation-1])

        params.append("--temperature")

        params.append(temperature)




        params.append("--out")
        params.append(self.__getattr__(f"select_md_{n_job}_xyz_file"))




        return params2str(params)
    def build_select_params(self):
        select=self.config["select"]
        params=[]
        params.append("NepTrain")
        params.append("select")
        #总的
        # params.append(self.select_all_md_dummp_xyz_file)
        #分开
        params.append(self.__getattr__(f"select_md_*_xyz_file"))
        params.append("--nep")
        params.append( self.nep_nep_txt_file)

        params.append("--base")
        params.append( self.nep_train_xyz_file )
        params.append("--max_selected")
        params.append(select["max_selected"])
        params.append("--min_distance")
        params.append(select["min_distance"])
        params.append("--out")
        params.append(self.select_selected_xyz_file)

        if select.get("filter",False):

            params.append("--filter")
            params.append(select.get("filter" ) if isinstance(select.get("filter" ),float) else 0.6)


        return params2str(params)
    def build_vasp_params(self,n_job=1):
        vasp=self.config["vasp"]
        params=[]
        params.append("NepTrain")
        params.append("vasp")

        if self.config["vasp_job"] == 1:

            if not os.path.exists(self.vasp_learn_add_xyz_file):
                return None
            params.append(self.vasp_learn_add_xyz_file)
        else:
            path=self.__getattr__(f"vasp_learn_add_{n_job}_xyz_file")
            if not os.path.exists(path):
                return None
            params.append(path)

        params.append("--directory")

        params.append(self.__getattr__(f"vasp_cache{n_job}_path"))


        params.append("-np")
        params.append(vasp["cpu_core"])
        if vasp["kpoints_use_gamma"]:
            params.append("--gamma")

        if vasp["incar_path"]:

            params.append("--incar")
            params.append((vasp["incar_path"]))
        if vasp["use_k_stype"]=="kpoints":
            if vasp.get("kpoints"):
                params.append("-ka")
                if isinstance(vasp["kpoints"],list):
                    params.append(",".join([str(i) for i in vasp["kpoints"]]))
                else:
                    params.append(vasp["kpoints"])
        else:

            if vasp.get("kspacing") :
                params.append("--kspacing")
                params.append(vasp["kspacing"])
        params.append("--out")
        params.append( self.__getattr__(f"vasp_learn_calculated_{n_job}_xyz_file"))


        return params2str(params)

    def sub_select(self):
        # utils.cat(self.__getattr__(f"select_md_*_xyz_file"),
        #           self.select_all_md_dummp_xyz_file
        #           )
        utils.print_msg(f"Start sampling from the trajectory.")

        cmd = self.build_select_params()

        forward_files = filter_file_path([self.select_path,
                         self.nep_nep_txt_file,
                         self.nep_train_xyz_file
                         ])
        submit_job(
            machine_dict=self.config["select"]["machine"],
            resources_dict=self.config["select"]["resources"],
            task_dict_list=[
                dict(
                    command=cmd,
                    task_work_path="./",
                    forward_files=forward_files,
                    backward_files=[
                        self.select_selected_xyz_file,
                        self.select_selected_png_file,
                                    self.__getattr__(f"select_selected_md_*_*_file")
                                    ],
                )
            ],
            submission_dict=dict(
                work_base="./",
                forward_common_files=[],
                backward_common_files=[],

            )

        )


        utils.cat(self.select_selected_xyz_file,
                  self.vasp_learn_add_xyz_file
                  )





    def sub_vasp(self):
        utils.print_msg("Beginning the execution of VASP for single-point energy calculations.")
        # break

        if not utils.is_file_empty(self.vasp_learn_add_xyz_file):

            if self.config["vasp_job"] != 1:
                # Split xyz for parallel submission
                self.split_vasp_job_xyz(self.vasp_learn_add_xyz_file)

            tasks = []
            for i in range(self.config["vasp_job"]):
                cmd = self.build_vasp_params(i + 1)
                if cmd is None:
                    continue

                tasks.append(
                    async_submit_job(
                        machine_dict=self.config["vasp"]["machine"],
                        resources_dict=self.config["vasp"]["resources"],
                        task_dict_list=[
                            dict(
                                command=cmd,
                                task_work_path="./",
                                forward_files=filter_file_path([
                                    self.config["vasp"]["incar_path"],
                                    self.vasp_learn_add_xyz_file,
                                    self.__getattr__(f"vasp_learn_add_{i +1}_xyz_file")

                                ]),
                                backward_files=[self.__getattr__(f"vasp_learn_calculated_{i +1}_xyz_file")],
                            )
                        ],
                        submission_dict=dict(
                            work_base="./",
                            forward_common_files=[],
                            backward_common_files=[],
                        ),
                    )
                )

            if tasks:
                asyncio.run(_await_tasks(tasks))

            utils.cat(self.__getattr__(f"vasp_learn_calculated_*_xyz_file"),
                      self.all_learn_calculated_xyz_file

                      )
            if self.config.get("limit",{}).get("force") and not utils.is_file_empty(self.all_learn_calculated_xyz_file):
                bad_structure = []
                good_structure = []
                structures=ase_read(self.all_learn_calculated_xyz_file,":")
                for structure in structures:

                    if structure.calc.results["forces"].max() <= self.config.get("limit",{}).get("force"):
                        good_structure.append(structure)
                    else:
                        bad_structure.append(structure)

                ase_write(self.all_learn_calculated_xyz_file,good_structure,append=False,format="extxyz")
                if bad_structure:
                    ase_write(self.remove_by_force_xyz_file, bad_structure, append=False, format="extxyz")

        else:
            utils.print_warning("Detected that the calculation input file is empty, proceeding directly to the next step!")

            utils.cat(self.vasp_learn_add_xyz_file,
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
            forward_files = filter_file_path([self.config["nep"].get("nep_in_path"),
                             self.last_improved_train_xyz_file,
                             self.config["nep"].get("test_xyz_path"),
                             self.last_nep_nep_restart_file
             ] )

            submit_job(
                machine_dict = self.config["nep"]["machine"],
                resources_dict = self.config["nep"]["resources"],
                task_dict_list = [
                    dict(
                        command=cmd,
                        task_work_path="./",
                        forward_files= forward_files,
                        backward_files = [self.nep_path],
                    )
                ],
                submission_dict = dict(
                    work_base="./",
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
                machine_dict=self.config["nep"]["machine"],
                resources_dict=self.config["nep"]["resources"],
                task_dict_list=[
                    dict(
                        command=cmd,
                        task_work_path="./",
                        forward_files=filter_file_path([
                            self.nep_nep_txt_file,
                            self.all_learn_calculated_xyz_file,
                            self.config["nep"].get("nep_in_path")
                        ]),
                        backward_files=[self.pred_path],
                    )
                ],
                submission_dict=dict(
                    work_base="./",
                    forward_common_files=[],
                    backward_common_files=[],
                ),
            )
        else:
            utils.print_msg(f"The dataset has not changed, skipping prediction.")


    def sub_gpumd(self):
        utils.print_msg(f"Starting active learning.")
        tasks = []
        if self.config.get("gpumd_split_job", "temperature") == "temperature":
            for i, temp in enumerate(self.config["gpumd"]["temperature_every_step"]):
                cmd = self.build_gpumd_params(
                    self.config["gpumd"].get("model_path"),
                    temp,
                    i,
                )
                tasks.append(
                    async_submit_job(
                        machine_dict=self.config["gpumd"]["machine"],
                        resources_dict=self.config["gpumd"]["resources"],
                        task_dict_list=[
                            dict(
                                command=cmd,
                                task_work_path="./",
                                forward_files=filter_file_path([
                                    self.config["gpumd"]["model_path"],
                                    self.config["gpumd"].get("run_in_path"),
                                    self.nep_nep_txt_file

                                ]),
                                backward_files=[self.__getattr__(f"select_md_{i}_xyz_file")],
                            )
                        ],
                        submission_dict=dict(
                            work_base="./",
                            forward_common_files=[],
                            backward_common_files=[],
                        ),
                    )
                )
        else:
            if os.path.isdir(self.config["gpumd"]["model_path"]):
                for i, file in enumerate(os.listdir(self.config["gpumd"]["model_path"])):
                    cmd = self.build_gpumd_params(
                        os.path.join(self.config["gpumd"]["model_path"], file),
                        self.config["gpumd"]["temperature_every_step"],
                        i,
                    )
                    tasks.append(
                        async_submit_job(
                            machine_dict=self.config["nep"]["machine"],
                            resources_dict=self.config["nep"]["resources"],
                            task_dict_list=[
                                dict(
                                    command=cmd,
                                    task_work_path="./",
                                    forward_files=filter_file_path([
                                    os.path.join(self.config["gpumd"]["model_path"], file),
                                    self.config["gpumd"].get("run_in_path"),
                                    self.nep_nep_txt_file

                                    ]),
                                    backward_files=[self.gpumd_path],
                                )
                            ],
                            submission_dict=dict(
                                work_base="./",
                                forward_common_files=[],
                                backward_common_files=[],
                            ),
                        )
                    )
        if tasks:
            asyncio.run(_await_tasks(tasks))





    def start(self,config_path):
        utils.print_msg("Welcome to NepTrain automatic training!")

        self.read_config(config_path)
        self.check_env()



        self.manager.set_next(self.config.get("current_job"))

        while True:

            #开始循环
            job=next(self.manager)
            self.config["current_job"]=job
            self.save_restart()
            if job=="vasp":

                self.sub_vasp()

            elif job=="pred":

                self.sub_nep_pred()
                self.generation += 1

            elif job=="nep":

                self.sub_nep()
                if self.generation>len(self.config["gpumd"]["step_times"]):
                   utils.print_success("Training completed!")
                   break
            elif job=="select":
                self.sub_select()

            else:
                self.sub_gpumd()


    def save_restart(self):
        with open("./restart.yaml","w",encoding="utf-8") as f:
            self.config["restart"]=True

            YAML().dump(self.config,f)

def train_nep(argparse):
    """
    首先检查下当前的进度 看从哪开始
    :return:
    """
    check_env()

    worker = NepTrainWorker()

    worker.start(argparse.config_path)
if __name__ == '__main__':
    train =NepTrainWorker()
    train.generation=1
    train.config["work_path"]="./cache"
    print(train.nep_path)

    print(train.__getattr__(f"vasp_learn_calculated_*_xyz_file"))
