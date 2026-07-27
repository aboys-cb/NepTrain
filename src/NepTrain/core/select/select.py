#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/29 19:52
# @Author  : 兵
# @email    : 1747193328@qq.com

import time
import numpy as np
from ..fps import farthest_point_sampling as _shared_fps






def farthest_point_sampling(points, n_samples, min_dist=0.1, selected_data=None):
    """
    最远点采样：支持已有样本扩展，并加入最小距离限制。

    参数:
        points (ndarray): 点集，形状为 (N, D)。
        n_samples (int): 最大采样点的数量。
        min_dist (float): 最小距离阈值。
        initial_indices (list or None): 已选择的样本索引列表，默认无。

    返回:
        sampled_indices (list): 采样点的索引。
    """
    points = np.asarray(points, dtype=np.float64)
    references = None
    if selected_data is not None and np.asarray(selected_data).size:
        references = np.asarray(selected_data, dtype=np.float64)
    result = _shared_fps(
        points,
        budget=int(n_samples),
        min_novelty=float(min_dist),
        reference_descriptors=references,
    )
    return list(result.selected_indices)


def select_structures(train, new_atoms ,descriptor, max_selected=20, min_distance=0.01 ):
    # 首先去掉跑崩溃的结构
    if descriptor is None:
        train_des=np.mean(train,axis=1)
        new_des=np.mean(new_atoms,axis=1)
    else:
        train_des = np.array([np.mean(descriptor.get_descriptors(i ), axis=0) for i in train])

        new_des = np.array([np.mean(descriptor.get_descriptors(i), axis=0) for i in new_atoms])
    # print(train_des.shape)
    selected_i =farthest_point_sampling(new_des,max_selected,min_distance,selected_data=train_des)

    return [new_atoms[i] for i in selected_i]

# 加速计算每对元素的最小键长
def compute_min_bond_lengths(atoms ):
    # 获取原子符号
    dist_matrix = atoms.get_all_distances(mic=True)
    symbols = atoms.get_chemical_symbols()
    # 提取上三角矩阵（排除对角线）
    i, j = np.triu_indices(len(atoms), k=1)
    # 用字典来存储每种元素对的最小键长
    bond_lengths = {}
    # 遍历所有原子对，计算每一对元素的最小键长
    for idx in range(len(i)):
        atom_i, atom_j = symbols[i[idx]], symbols[j[idx]]
        # if atom_i==atom_j:
        #     continue
        # 获取当前键长
        bond_length = dist_matrix[i[idx], j[idx]]
        # if bond_length>5:
        #     continue
        # 确保元素对按字母顺序排列，避免 Cs-Ag 和 Ag-Cs 视为不同
        element_pair = tuple(sorted([atom_i, atom_j]))
        # 如果该元素对尚未存在于字典中，初始化其最小键长
        if element_pair not in bond_lengths:
            bond_lengths[element_pair] = bond_length
        else:
            # 更新最小键长
            bond_lengths[element_pair] = min(bond_lengths[element_pair], bond_length)

    return bond_lengths



def process_trajectory(trajectory):
    # 读取轨迹文件（假设为 xyz 格式）


    # 存储所有帧的结果
    all_bond_lengths = []

    # 遍历轨迹中的每一帧
    for atoms in trajectory:
        # 获取当前帧的距离矩阵

        # 计算当前帧的最小键长
        bond_lengths = compute_min_bond_lengths(atoms )
        all_bond_lengths.append(bond_lengths)

    return all_bond_lengths

def filter_by_bonds(trajectory,model):
    good_structure=[]
    bad_structure=[]
    base_bond=compute_min_bond_lengths(model )
    bonds=process_trajectory(trajectory)
    for index,bond in enumerate(bonds):
        # print(bond)
        # condition = [utils.radius_table[a]+utils.radius_table[b] > a_b for (a,b),a_b in bond.items()]

        # A candidate containing an element pair absent from the reference
        # cannot be validated against that reference and must not be silently
        # accepted.
        condition = [
            key not in base_bond or base_bond[key] * 0.6 > a_b
            for key, a_b in bond.items()
        ]
        # print(condition)
        if any(condition):
            bad_structure.append(trajectory[index])
        else:
            good_structure.append(trajectory[index])
    return good_structure, bad_structure
