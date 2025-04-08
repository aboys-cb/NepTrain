#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 15:53
# @Author  : 兵
# @email    : 1747193328@qq.com
import os
import subprocess
from pathlib import Path

import numpy as np
from ase.calculators import calculator
from ase.calculators.calculator import Calculator
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.vasp import check_atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from NepTrain import Config
def write_to_xyz(vaspxml_path, save_path, Config_type, append=True):

    atoms_list = []
    atoms = ase_read(vaspxml_path, index=":")
    index = 1
    for atom in atoms:
        xx, yy, zz, yz, xz, xy = -atom.calc.results['stress'] * atom.get_volume()  # *160.21766
        atom.info['virial'] = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])

        atom.calc.results['energy'] = atom.calc.results['free_energy']

        atom.info['Config_type'] = Config_type + str(index)
        atom.info['Weight'] = 1.0
        del atom.calc.results['stress']
        del atom.calc.results['free_energy']
        atoms_list.append(atom)
        index += 1

    ase_write(save_path, atoms_list, format='extxyz', append=append)
    return atoms_list
class VaspInput(Vasp):


    def __init__(self,*args,**kwargs):

        super(VaspInput,self).__init__(*args,**kwargs)
        self.input_params["setups"] = {"base": "recommended"}
        self.input_params["pp"] = ''
        
        os.environ[self.VASP_PP_PATH] = os.path.expanduser(Config.get("environ", "potcar_path"))

    def calculate(self,
                  atoms=None,
                  properties=('energy', ),
                  system_changes=tuple(calculator.all_changes)):
        """Do a VASP calculation in the specified directory.

        This will generate the necessary VASP input files, and then
        execute VASP. After execution, the energy, forces. etc. are read
        from the VASP output files.
        """
        if 'magmom' in Config:
            items = config.items('magmom')
            if items:
                for symbol, moment_str in Config['magmom'].items():
                    try:
                        element_magmoms[symbol] = float(moment_str.strip())
                    except ValueError:
                        element_magmoms[symbol] = 0.0
                magnetic_moments = []
                for atom in self.atoms:
                    if atom.symbol in element_magmom.keys():
                        magnetic_moments.append(element_magmom[atom.symbol])
                    else:
                        magnetic_moments.append(0.0)
                self.atoms.set_initial_magnetic_moments(magnetic_moments)
                      
        Calculator.calculate(self, atoms, properties, system_changes)
        # Check for zero-length lattice vectors and PBC
        # and that we actually have an Atoms object.
        check_atoms(self.atoms)

        self.clear_results()

        command = self.make_command(self.command)
        self.write_input(self.atoms, properties, system_changes)

        with self._txt_outstream() as out:
            errorcode, stderr = self._run(command=command,
                                          out=out,
                                          directory=self.directory)

        if errorcode:
            raise calculator.CalculationFailed(
                '{} in {} returned an error: {:d} stderr {}'.format(
                    self.name, Path(self.directory).resolve(), errorcode,
                    stderr))

        # Read results from calculation
        self.update_atoms(atoms)
        self.read_results()
    def _run(self, command=None, out=None, directory=None):
        """Method to explicitly execute VASP"""
        if command is None:
            command = self.command
        if directory is None:
            directory = self.directory

        result = subprocess.run(command,
                                shell=True,
                                cwd=directory,
                                capture_output=True,
                                text=True)
        if out is not None:
            out.write(result.stdout)
            out.write(result.stderr)

        return result.returncode, result.stderr


if __name__ == '__main__':
    vasp=VaspInput()




    atoms=ase_read("./POSCAR",format='vasp')
    vasp.read_incar("./INCAR")
    vasp.calculate(atoms,('energy'))
    print(vasp.results)
    print(vasp.atoms.info)
    print(atoms.calc.results)
