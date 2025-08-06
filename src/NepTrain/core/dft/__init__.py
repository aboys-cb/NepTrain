#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/8/2 11:07
# @Author  : 兵
# @email    : 1747193328@qq.com
from .vasp import run_vasp
from .abacus import run_abacus
def run_dft(argparse):
    # 默认是 vasp

    if argparse.software is None:

        argparse.software = 'vasp'


    if argparse.directory is None:
        argparse.directory = f"./cache/{argparse.software}"
    if argparse.out_file_path is None:
        argparse.out_file_path = f"./{argparse.software}_scf.xyz"

    if argparse.software =="vasp":

        if argparse.incar is None:
            argparse.incar=f"./INCAR"

        return run_vasp(argparse)
    elif argparse.software =="abacus":

        if argparse.incar is None:
            argparse.incar=f"./INPUT"
        return run_abacus(argparse)