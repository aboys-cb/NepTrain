#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/8/2 17:41
# @Author  : 兵
# @email    : 1747193328@qq.com
import math
import os.path

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from ase.calculators.abacus import Abacus, AbacusProfile

from NepTrain import utils, Config, module_path

from .io import read_input_file,StructureVar

atoms_index=1

@utils.iter_path_to_atoms(["*.vasp","*.xyz"],show_progress=True,
                 description="ABACUS calculation progress" )
def calculate_abacus(atoms:Atoms,argparse):
    global atoms_index
    StructureVar.init("./")

    if argparse.incar is not None and os.path.exists(argparse.incar):
        input_dict = read_input_file(argparse.incar)
    else:
        input_dict = read_input_file(os.path.join(module_path,"core/dft/abacus/INPUT"))
    directory=os.path.join(argparse.directory,f"{atoms_index}-{atoms.symbols}")
    atoms_index+=1
    command = f"{Config.get('environ','mpirun_path')} -n {argparse.n_cpu} {Config.get('environ','abacus_path',fallback='abacus')}"
    if "NEPTRAIN_ABACUS_COMMAND" in os.environ:
        command = os.environ["NEPTRAIN_ABACUS_COMMAND"]
    profile = AbacusProfile(command=command)


    a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()

    if argparse.kspacing is not None:
        input_dict["kspacing"]=argparse.kspacing

    pp_files,orb_files  =StructureVar.completion_abacus(atoms=atoms)


    calc = Abacus(profile=profile,
                  directory=directory,
                  pp=pp_files, basis=orb_files,
                  kpts=(math.ceil(argparse.ka[0]/a)  ,
                  math.ceil(argparse.ka[1]/b)  ,
                  math.ceil(argparse.ka[2]/c) ),
                  **input_dict
                  )


    calc.calculate(atoms, ('energy',"forces","stress"),None)
    atoms.calc = calc

    xx, yy, zz, yz, xz, xy = -calc.results['stress'] * atoms.get_volume()  # *160.21766
    atoms.info['virial'] = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
    # 这里没想好怎么设计config的格式化  就先使用原来的
    if "Config_type" not in atoms.info:
        atoms.info['Config_type'] = "NepTrain scf "
    atoms.info['Weight'] = 1.0
    del atoms.calc.results['stress']
    del atoms.calc.results['free_energy']

    return atoms



def run_abacus(argparse):
    result = calculate_abacus(argparse.model_path, argparse)
    path = os.path.dirname(argparse.out_file_path)
    if path and not os.path.exists(path):
        os.makedirs(path)
    if len(result) and isinstance(result[0], list):
        result = [atoms for _list in result for atoms in _list]
    ase_write(argparse.out_file_path, result, format="extxyz", append=argparse.append)

    utils.print_success("ABACUS calculation task completed!")

