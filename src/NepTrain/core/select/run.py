#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/11/13 19:36
# @Author  : 兵
# @email    : 1747193328@qq.com
import os.path
from pathlib import Path

import numpy as np


# from joblib import Parallel, delayed
from tqdm import tqdm
from NepTrain import utils
from ase.io import read as ase_read
from ase.io import write as ase_write
from .select import select_structures, filter_by_bonds, farthest_point_sampling
from .filter import adjust_reasonable, parallel_filter_trajectory
from ..gpumd.plot import plot_md_selected

from ..nep.calculator import DescriptorCalculator


def run_select(argparse):
    import matplotlib.pyplot as plt
    map_path_index=[]
    all_trajectory=[]
    plot_config=[]
    trajectory_structures=[]
    filter_structures = []
    for index,_path in enumerate(argparse.trajectory_paths):

        if utils.is_file_empty(_path):
            utils.print_warning(f"An invalid file path was provided: {argparse.trajectory_paths}.")
            continue

        utils.print_msg(f"Reading trajectory {_path}")

        trajectory=ase_read(_path,":",format="extxyz")

        if argparse.filter:
            utils.print_msg(f"Start filtering...")
            file_name = os.path.basename(_path)

            # 使用示例
            trajectory, filter_structures = parallel_filter_trajectory(
                trajectory, argparse.filter, n_jobs=os.cpu_count()-2  # -1 表示使用所有CPU核心
            )


            if len(filter_structures) > 0:
                utils.print_msg(f"Filtering {len(filter_structures)} structures.")
                ase_write(os.path.join(os.path.dirname(_path),f"filter_{file_name}.xyz"),filter_structures,append=False)



        map_path_index.append(np.full(len(trajectory),index))
        trajectory_structures.extend(trajectory)
    map_path_index=np.concatenate(map_path_index)
    if len(trajectory_structures)==0:
        utils.print_warning("no structure.")
        ase_write(argparse.out_file_path, trajectory_structures)

        return

    if utils.is_file_empty(argparse.base):
        base_train=[]
    else:
        base_train=ase_read(argparse.base,":",format="extxyz")

    if utils.is_file_empty(argparse.nep):
        utils.print_msg("An invalid path for nep.txt was provided, using SOAP descriptors instead.")
        species=set()
        for atoms in trajectory_structures+base_train:
            for i in atoms.get_chemical_symbols():
                species.add(i)
        kwargs_dict={
            "species":list(species),
            "r_cut":argparse.r_cut,
            "n_max": argparse.n_max,
            "l_max": argparse.l_max

        }

        descriptor =DescriptorCalculator("soap",**kwargs_dict)

    else:
        descriptor =DescriptorCalculator("nep", model_file=argparse.nep)

    utils.print_msg("Start generating structure descriptor, please wait")
    train_structure_des =descriptor.get_structures_descriptors(base_train)
    trajectory_structure_des=[]
    uniqe_index = np.unique(map_path_index)

    # 使用 'viridis' colormap 从色图中获取颜色
    cmap = plt.cm.viridis  # 你也可以选择其他色图，例如 'plasma', 'inferno', 'cividis' 等
    num_colors = len(uniqe_index)

    # 创建颜色数组
    colors = [cmap(i / num_colors) for i in range(num_colors)]
    for index  in uniqe_index:
        indices = np.where(map_path_index == index)[0]
        trajectory_structure=[trajectory_structures[i] for i in indices]
        new =descriptor.get_structures_descriptors(trajectory_structure)
        trajectory_structure_des.append(new)
        label = Path(argparse.trajectory_paths[index]).name

        plot_config.append((new,label, colors[index]))

    utils.print_msg("Starting to select points, please wait...")
    new_structure_des=np.vstack(trajectory_structure_des)



    selected_i =farthest_point_sampling(new_structure_des,argparse.max_selected,argparse.min_distance,selected_data=train_structure_des)

    selected_structures=[]

    base_dir=os.path.dirname(argparse.out_file_path)

    utils.remove_file_by_re(os.path.join(base_dir,"selected*.xyz"))

    for i in selected_i:
        file_index=map_path_index[i]

        fila_name = Path(argparse.trajectory_paths[file_index]).name
        save_path=os.path.join(base_dir,f"selected_{fila_name}.xyz")


        structure = trajectory_structures[i]
        ase_write(save_path,structure,append=True)
        selected_structures.append(structure)
    selected_des=new_structure_des[selected_i,:]


    utils.print_msg(f"Obtained {len(selected_i)} structures." )
    ase_write(argparse.out_file_path, selected_structures)
    png_path=os.path.join(os.path.dirname(argparse.out_file_path),"selected.png")
    plot_md_selected(train_structure_des,
                     plot_config,
                     selected_des,
                       png_path ,
                     argparse.decomposition
                     )
    utils.print_msg(f"The point selection distribution chart is saved to {png_path}." )
    utils.print_msg(f"The selected structures are saved to {argparse.out_file_path}." )

