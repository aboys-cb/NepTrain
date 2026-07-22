#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 15:42
# @Author  : 兵
# @email    : 1747193328@qq.com
import os
import re
import shutil
import subprocess

from ase.io import write as ase_write

from NepTrain import utils, Config
from .plot import *





class RunInput:


    def __init__(self,nep_txt_path):
        self.nep_txt_path=nep_txt_path
        self.command=Config.get('environ','gpumd_path')
        self.time_step=1
        self.dump_thermo=False
        self.dump_exyz=False
        self.total_time=0
        self.run_in=[]

    def set_time_temp(self,times=None,temperature=None):
        """
        None表示不修改
        :param times: 单位ps
        :param temperature:
        :return:
        """
        if times   is not None:
            self.total_time=0
        for run_index in range(len(self.run_in)):
            run=self.run_in[run_index]
            if run[0]=="ensemble":
                if temperature is not None:

                    run[1][1]=str(temperature)
                    run[1][2]=str(temperature)
            elif run[0]=="run":
                if times is not None:

                    run[1][0]=str(int(int(times)*1000*1/self.time_step))
                    self.total_time+=int(int(times)*1000*1/self.time_step)

    def read_run(self,file_name):
        self.run_in.clear()
        with open(file_name,'r',encoding="utf8") as f:
            groups=re.findall(r"^([A-Za-z_]+)\s+(.*)",f.read() ,re.MULTILINE)

            for group in groups:

                key,value=group[0].strip(),group[1].strip()
                if not key:
                    continue
                if key=="time_step":
                    self.time_step=float(value)
                elif key=="dump_thermo":
                    self.dump_thermo=True
                elif key=="dump_exyz":
                    self.dump_exyz=True
                elif key=="run":
                    self.total_time+=int(value)
                self.run_in.append([key,[i for i in value.split(" ") if i.strip()]])


    def write_run(self,file_name):
        with open(file_name,'w',encoding="utf8") as f:
            for i in self.run_in:

                f.write(f"{i[0]}    {' '.join(i[1])}\n")

    def calculate(self,atoms,directory,show_progress=True):
        """
        计算所需要的文件
            nep.txt
            run.in
            model.xyz
        :param directory:
        :param show_progress:
        :return:
        """
        if not os.path.exists(directory):
            os.makedirs(directory )

        self.write_run(os.path.join(directory,"run.in"))
        ase_write(os.path.join(directory,"model.xyz"),atoms,format="extxyz")
        if self.nep_txt_path is not None and os.path.exists(self.nep_txt_path):
            if utils.is_diff_path(self.nep_txt_path, os.path.join(directory, "nep.txt")):

                shutil.copy(self.nep_txt_path, os.path.join(directory, "nep.txt"))
        else:
            raise ValueError(f"{self.nep_txt_path} is an invalid path！")

        with   open(os.path.join(directory,"gpumd.out"), "w") as f_std, open(os.path.join(directory,"gpumd.err"), "w", buffering=1) as f_err:

            errorcode = subprocess.call(self.command,
                                        shell=True,
                                        stdout=f_std,
                                        stderr=f_err,
                                        cwd=directory)

        if errorcode != 0:
            raise RuntimeError(
                f"GPUMD MD failed with exit code {errorcode}; "
                f"see {os.path.join(directory, 'gpumd.err')}"
            )

        plot_md_thermo(os.path.join(directory,"thermo.out") )

if __name__ == '__main__':
    # read_thermo("1.out",80)
    run=RunInput()
    run.read_run("./run.in")
    run.set_time_temp(1,300)
