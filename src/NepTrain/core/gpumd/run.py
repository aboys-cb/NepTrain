#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/29 15:06
# @Author  : 兵
# @email    : 1747193328@qq.com

import os.path
import shutil

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from NepTrain import utils, module_path

from ..select import select_structures, filter_by_bonds


from .io import RunInput


from ..utils import check_env

import zlib
def array_to_id(arr):
    arr = np.ascontiguousarray(arr)  # 确保内存连续
    return zlib.crc32(arr.tobytes())


@utils.iter_path_to_atoms(["*.vasp","*.xyz"],show_progress=False)
def calculate_gpumd(atoms:Atoms,argparse):

    atoms_index = array_to_id(atoms.positions)

    new_atoms=[]

    if os.path.exists(argparse.out_file_path):
        os.remove(argparse.out_file_path)
    for temperature in argparse.temperature:

        run = RunInput(argparse.nep_txt_path)
        if utils.is_file_empty(argparse.run_in_path):
            run_in_path=os.path.join(module_path,"core/gpumd/run.in")
        else:
            run_in_path=argparse.run_in_path
        run.read_run(run_in_path)
        run.set_time_temp(argparse.time,temperature)
        directory=os.path.join(argparse.directory,f"{atoms_index}-{atoms.symbols}@{temperature}k-{argparse.time}ps")
        utils.print_msg(f"GPUMD is running, temperature: {temperature}k. Time: {argparse.time}ps" )

        run.calculate(atoms,directory)

        dump = ase_read(os.path.join(directory,"dump.xyz"), ":", format="extxyz", do_not_split_by_at_sign=True)
        for i, atom in enumerate(dump):
            atom.info["Config_type"] = f"{atom.symbols}-epoch-{argparse.time}ps-{temperature}k-{i + 1}"



        ase_write(argparse.out_file_path,dump,append=True)







    return new_atoms
def run_gpumd(argparse):
    check_env()
    utils.verify_path(os.path.dirname(os.path.abspath(argparse.out_file_path)))
    result = calculate_gpumd(argparse.model_path,argparse)









