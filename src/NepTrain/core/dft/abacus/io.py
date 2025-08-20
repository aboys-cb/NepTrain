#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/8/2 18:23
# @Author  : 兵
# @email    : 1747193328@qq.com
import os
import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers


def read_input_file(file_name: str) -> dict:
    input_key = {}
    if  not os.path.exists(file_name):
        return input_key
    pattern = r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s+([^#\n]*?))?\s*(?:#.*)?$"
    with open(file_name, 'r', encoding="utf8") as f:
        # print(repr(f.read()))
        # groups = re.findall(r"^([A-Za-z_]+)(?:\s+([^#\n]*?))?\s*(?:#.*)?$", f.read(), re.MULTILINE)
        # for group in groups:
        #     input_key[group[0].strip()] = group[1].strip()
        for line in f:
            line = line.strip()
            if not line or line == "INPUT_PARAMETERS":
                continue
            m = re.match(pattern, line)
            if m:
                key, value = m.groups()
                input_key[key]= value.strip()

    return input_key



class StructureVar:
    pp_files={}
    orbs={}
    # masses={ symbol:atomic_masses[z] for symbol ,z in atomic_numbers.items()}
    @classmethod
    def init(cls,path):
        path = Path(path)
        upfs= path.glob("*.upf")
        for upf in upfs:
            try:
                with open( upf,"r") as f:
                    ufp_content = f.read()
                elem = re.search('element="(\w+)"',ufp_content).group(1)
                cls.pp_files[elem] = upf.name
            except:
                pass
        orbs= path.glob("*.orb")
        for orb in orbs:
            try:
                with open( orb,"r") as f:
                    orb_content = f.read()
                elem = re.search('Element\s+(\w+)',orb_content).group(1)
                cls.orbs[elem] = orb.name
            except:
                pass
        pass
    @classmethod
    def update(cls,structure):
        atom_names = structure["atom_names"]
        if "pp_files" in structure.data:
            for atom ,pp in zip(atom_names,structure["pp_files"]):
                StructureVar.pp_files[atom]=pp

        if "orb_files" in structure.data:
            for atom ,pp in zip(atom_names,structure["orb_files"]):
                StructureVar.orbs[atom]=pp
        # if "masses" in structure.data:
        #     for atom ,pp in zip(atom_names,structure["masses"]):
        #         StructureVar.masses[atom]=pp
    @classmethod
    def completion_abacus(cls,atoms:Atoms):

        atom_names =atoms.get_chemical_symbols()
        pp_files= {}
        orb_files= {}
        # masses=[]
        for atom in atom_names:
            if atom in cls.pp_files:
                pp_files[atom] = cls.pp_files[atom]
            if atom in cls.orbs:
                orb_files[atom] = cls.orbs[atom]

            # masses.append(cls.masses[atom])

        return pp_files,orb_files

