#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/8/2 18:23
# @Author  : 兵
# @email    : 1747193328@qq.com
import os
import re
from pathlib import Path

from ase import Atoms


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
                input_key[key] = (value or "").strip()

    return input_key



class StructureVar:
    pp_files={}
    orbs={}
    # masses={ symbol:atomic_masses[z] for symbol ,z in atomic_numbers.items()}
    @classmethod
    def init(cls, path, *, reset: bool = True):
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"ABACUS resource directory does not exist: {path}")
        if reset:
            cls.pp_files = {}
            cls.orbs = {}
        upfs = (item for item in path.iterdir() if item.suffix.lower() == ".upf")
        for upf in upfs:
            try:
                with open(upf, "r", encoding="utf-8", errors="ignore") as f:
                    ufp_content = f.read()
                elem = re.search(r'element\s*=\s*["\'](\w+)["\']', ufp_content).group(1)
                cls.pp_files[elem] = upf.name
            except (AttributeError, OSError):
                pass
        orbs = (item for item in path.iterdir() if item.suffix.lower() == ".orb")
        for orb in orbs:
            try:
                with open(orb, "r", encoding="utf-8", errors="ignore") as f:
                    orb_content = f.read()
                elem = re.search(r'Element\s+(\w+)', orb_content).group(1)
                cls.orbs[elem] = orb.name
            except (AttributeError, OSError):
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
    def completion_abacus(cls, atoms: Atoms, *, require_orbitals: bool):

        atom_names = atoms.get_chemical_symbols()
        pp_files= {}
        orb_files= {}
        # masses=[]
        for atom in atom_names:
            if atom in cls.pp_files:
                pp_files[atom] = cls.pp_files[atom]
            if atom in cls.orbs:
                orb_files[atom] = cls.orbs[atom]

            # masses.append(cls.masses[atom])

        elements = set(atom_names)
        missing_pp = sorted(elements - pp_files.keys())
        if missing_pp:
            raise FileNotFoundError(
                "missing ABACUS pseudopotential for: " + ", ".join(missing_pp)
            )
        missing_orb = sorted(elements - orb_files.keys())
        if require_orbitals and missing_orb:
            raise FileNotFoundError(
                "missing ABACUS orbital for: " + ", ".join(missing_orb)
            )
        return pp_files, orb_files
