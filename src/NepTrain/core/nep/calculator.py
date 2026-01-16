#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/11/21 14:22
# @Author  : 兵
# @email    : 1747193328@qq.com
import contextlib
import os

import numpy as np
import numpy.typing as npt
from ase import Atoms
from functools import partial
from typing import Iterable
from NepTrain.nep_cpu import CpuNep



def split_by_natoms(array, natoms_list:list[int]) -> list[npt.NDArray]:
    """Split a flat array into sub-arrays according to the number of atoms in each structure."""
    if array.size == 0:
        return array
    counts = np.asarray(list(natoms_list), dtype=int)
    split_indices = np.cumsum(counts)[:-1]
    split_arrays = np.split(array, split_indices)
    return split_arrays
def aggregate_per_atom_to_structure(
    array: npt.NDArray[np.float32],
    atoms_num_list: Iterable[int],
    map_func=np.linalg.norm,
    axis: int = 0,
) -> npt.NDArray[np.float32]:
    """Aggregate per-atom data into per-structure values based on atom counts."""
    split_arrays = split_by_natoms(array, atoms_num_list)
    func = partial(map_func, axis=axis)
    return np.array(list(map(func, split_arrays)))

class Nep3Calculator:

    def __init__(self, model_file="nep.txt"):
        if not isinstance(model_file, str):
            model_file=str(model_file,encoding="utf-8")
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                self.nep3 = CpuNep(model_file)
        self.element_list=self.nep3.get_element_list()
        self.type_dict = {e: i for i, e in enumerate(self.element_list)}
    @staticmethod
    def _ensure_structure_list(
        structures: Iterable[Atoms] | Atoms,
    ) -> list[Atoms]:
        if isinstance(structures, ( Atoms)):
            return [structures]
        if isinstance(structures, list):
            return structures
        return list(structures)

    def compose_structures(
        self,
        structures: Iterable[Atoms] ,
    ) -> tuple[list[list[int]], list[list[float]], list[list[float]], list[int]]:
        structure_list = self._ensure_structure_list(structures)
        group_sizes: list[int] = []
        atom_types: list[list[int]] = []
        boxes: list[list[float]] = []
        positions: list[list[float]] = []
        for structure in structure_list:
            symbols = structure.get_chemical_symbols()
            mapped_types = [self.type_dict[symbol] for symbol in symbols]
            box = structure.cell.transpose(1, 0).reshape(-1).tolist()
            coords = structure.positions.transpose(1, 0).reshape(-1).tolist()
            atom_types.append(mapped_types)
            boxes.append(box)
            positions.append(coords)
            group_sizes.append(len(mapped_types))
        return atom_types, boxes, positions, group_sizes

    def get_descriptors(self,structure):
        symbols = structure.get_chemical_symbols()
        _type = [self.type_dict[k] for k in symbols]
        _box = structure.cell.transpose(1, 0).reshape(-1).tolist()
        _position = structure.get_positions().transpose(1, 0).reshape(-1).tolist()
        descriptor = self.nep3.get_descriptor(_type, _box, _position)
        descriptors_per_atom = np.array(descriptor).reshape(-1, len(structure)).T

        return descriptors_per_atom
    def get_structure_descriptors(self, structure):
        descriptors_per_atom=self.get_descriptors(structure)
        return descriptors_per_atom.mean(axis=0)

    def get_structures_descriptors(self,structures:[Atoms]):
        _types=[]
        _boxs=[]
        _positions=[]

        types, boxes, positions, group_sizes = self.compose_structures(structures)


        descriptor = self.nep3.get_structures_descriptor(types, boxes, positions)
        descriptor = np.asarray(descriptor, dtype=np.float32)

        structure_descriptor = aggregate_per_atom_to_structure(descriptor, group_sizes, map_func=np.mean, axis=0)


        return structure_descriptor


    def calculate(self,structures:list[Atoms],mean_virial=True):

        types, boxes, positions, group_sizes = self.compose_structures(structures)


        potentials, forces, virials = self.nep3.calculate(types, boxes, positions)


        potentials_arr = np.asarray(potentials, dtype=np.float32)
        forces_arr = np.asarray(forces, dtype=np.float32)
        virials_arr = np.asarray(virials, dtype=np.float32)
        if potentials_arr.size == 0:
            return [], [], []
        if forces_arr.ndim == 1:
            forces_arr = forces_arr.reshape(-1, 3)
        if virials_arr.ndim == 1:
            virials_arr = virials_arr.reshape(-1, 9)
        potentials_array = aggregate_per_atom_to_structure(potentials_arr, group_sizes, map_func=np.sum,
                                                           axis=None).tolist()
        forces_blocks = split_by_natoms(forces_arr, group_sizes)
        if mean_virial:
            virials_blocks = aggregate_per_atom_to_structure(virials_arr, group_sizes, map_func=np.mean,
                                                             axis=0).tolist()
        else:
            virials_blocks = split_by_natoms(virials_arr, group_sizes)

        return potentials_array, forces_blocks, virials_blocks


class DescriptorCalculator:
    def __init__(self, calculator_type="nep",**calculator_kwargs):
        self.calculator_type=calculator_type
        if calculator_type == "nep":
            self.calculator=Nep3Calculator(**calculator_kwargs)
        elif calculator_type == "soap":
            from dscribe.descriptors import SOAP

            self.calculator = SOAP(
                **calculator_kwargs,dtype="float32"
            )
        else:
            raise ValueError("calculator_type must be nep or soap")


    def get_structures_descriptors(self,structures:[Atoms]):

        if len(structures)==0:
            return np.array([])

        if self.calculator_type == "nep":
            return self.calculator.get_structures_descriptors(structures)
        else:

            return  np.array([self.calculator.create_single(structure).mean(0) for structure in structures])


if __name__ == '__main__':
    from NepTrain.core.nep import Nep3Calculator
    nep3 = Nep3Calculator(model_file="/mnt/d/Desktop/vispy/KNbO3/nep.txt")
    from ase.io import read
    import time
    structures = read("/mnt/d/Desktop/vispy/KNbO3/train.xyz",index=":")
    start=time.time()

    descriptors = nep3.get_structures_descriptors(structures)
    print(f"计算描述符：{len(structures)}个结构，耗时：{time.time()-start:.3f}s")
    print("descriptors",descriptors.shape)
    start=time.time()

    potentials ,forces ,virials   = nep3.calculate(structures)

    print(f"计算性质：{len(structures)}个结构，耗时：{time.time()-start:.3f}s")
    print("potentials",potentials.shape)
    print("forces",forces.shape)
    print("virials",virials.shape)

