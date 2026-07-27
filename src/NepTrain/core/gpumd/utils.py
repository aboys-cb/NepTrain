#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/11/26 19:39
# @Author  : 兵
# @email    : 1747193328@qq.com
import os

import numpy as np


def calculate_angle(x, y):
    dot_product = np.einsum('ij,ij->i', x, y)
    norm_x = np.linalg.norm(x, axis=1)
    norm_y = np.linalg.norm(y, axis=1)
    angle_radians = np.arccos(dot_product / (norm_x * norm_y))
    return np.degrees(angle_radians)

def calculate_volume(a, b, c):
    volume = np.einsum('ij,ij->i', a, np.cross(b, c))
    return np.abs(volume)

# Determine dump_interval from run.in file
def get_dump_interval(run_in_file="run.in"):
    dump_interval = 10  # Default value
    if os.path.exists(run_in_file):
        with open(run_in_file, 'r') as file:
            for line in file:
                if "dump_thermo" in line:
                    try:
                        dump_interval = int(line.split()[1])
                        break
                    except (IndexError, ValueError) as error:
                        raise ValueError(
                            f"invalid dump_thermo interval in {run_in_file}: "
                            f"{line.strip()}"
                        ) from error
    return dump_interval
