#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/25 19:03
# @Author  : 兵
# @email    : 1747193328@qq.com

import math
import os.path

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from ase.io.vasp import read_vasp

from NepTrain import utils, Config, module_path
from ..utils import check_env
from .io import VaspInput,write_to_xyz

atoms_index=1

@utils.iter_path_to_atoms(["*.vasp","*.xyz"],show_progress=True,
                 description="VASP calculation progress" )
def calculate_vasp(atoms:Atoms,argparse):
    global atoms_index

    vasp = VaspInput()
    if argparse.incar is not None and os.path.exists(argparse.incar):
        vasp.read_incar(argparse.incar)
    else:
        vasp.read_incar(os.path.join(module_path,"core/vasp/INCAR"))
    directory=os.path.join(argparse.directory,f"{atoms_index}-{atoms.symbols}")
    atoms_index+=1
    command=f"{Config.get('environ','mpirun_path')} -n {argparse.n_cpu} {Config.get('environ','vasp_path')}"

    a,b,c,alpha, beta, gamma=atoms.get_cell_lengths_and_angles()
    if argparse.kspacing is not None:
        vasp.set(kspacing=argparse.kspacing)
    vasp.set(
             directory=directory,
             command=command,
            kpts=(math.ceil(argparse.ka[0]/a)  ,
                  math.ceil(argparse.ka[1]/b)  ,
                  math.ceil(argparse.ka[2]/c) ),
             gamma=argparse.use_gamma,
             )
    magmom_line = set_magmom(directory)
    print(f"Directory: {directory}, magmom_line: {magmom_line}")
    if magmom_line:
        vasp.set(
                ispin=2,
                magmom=magmom_line,
                )

    if vasp.int_params["ibrion"] ==0:
        #分子动力学
        vasp.calculate(atoms, ('energy'))

        atoms_list = write_to_xyz(os.path.join(directory,"vasprun.xml"),os.path.join(directory,f"aimd_{vasp.float_params['tebeg']}k_{vasp.float_params['teend']}k.xyz"),"aimd",False)
        return atoms_list
    else:
        vasp.calculate(atoms, ('energy'))
        atoms.calc = vasp._xml_calc
        xx, yy, zz, yz, xz, xy = -vasp.results['stress'] * atoms.get_volume()  # *160.21766
        atoms.info['virial'] = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
        # 这里没想好怎么设计config的格式化  就先使用原来的
        if "Config_type" not in atoms.info:
            atoms.info['Config_type'] = "NepTrain scf "
        atoms.info['Weight'] = 1.0
        del atoms.calc.results['stress']
        del atoms.calc.results['free_energy']
        if vasp.converged:
            return atoms
        else:
            raise ValueError(f"{directory}: VASP not converged")
def run_vasp(argparse):
    check_env()

    result = calculate_vasp(argparse.model_path,argparse)
    path=os.path.dirname(argparse.out_file_path)
    if path and  not os.path.exists(path):
        os.makedirs(path)
    if len(result) and isinstance(result[0],list):
        result=[atoms for _list in result for atoms in _list]
    ase_write(argparse.out_file_path,result,format="extxyz",append=argparse.append)

    utils.print_success("VASP calculation task completed!" )

def set_magmom(directory):
  if 'magmom' in Config:
      items = config.items('magmom')
      if items:
          element_magmoms = {}
          for symbol, moment_str in Config['magmom'].items():
              try:
                  element_magmoms[symbol] = float(moment_str.strip())
              except ValueError:
                  element_magmoms[symbol] = 0.0
          nonzero = False
          for value in element_magmoms.values():
              if value != 0.0:
                  nonzero = True
          if nonzero == True:
              poscar_path = os.path.join(directory, "POSCAR")
              atoms = read_vasp(poscar_path)
              symbols = atoms.get_chemical_symbols()
              unique_symbols_ordered = []
              seen_symbols = set()
              for symbol in symbols:
                  if symbol not in seen_symbols:
                      unique_symbols_ordered.append(symbol)
                      seen_symbols.add(symbol)
          
              symbol_counts = {symbol: symbols.count(symbol) for symbol in symbols}
          
              magmom_lines = []
              for symbol in unique_symbols_ordered:
                  count = symbol_counts[symbol]
                  try:
                      magmom_lines.append(f"{element_magmoms[symbol]}*{count}")
                  except:
                      magmom_lines.append(f"0.0*{count}")
          
              magmom_string = " ".join(magmom_lines)
              magmom_line = f"{magmom_string}\n"
              return magmom_line
          else:
              return None
      else:
          return None
  else:
      return None

if __name__ == '__main__':
    calculate_vasp("./")
