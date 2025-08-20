#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/29 21:52
# @Author  : 兵
# @email    : 1747193328@qq.com
import os

from NepTrain import Config
from NepTrain import utils


def check_env():

    if not os.path.exists(os.path.expanduser(Config.get("environ", "potcar_path"))):
        raise FileNotFoundError("Please edit the pseudopotential file path in ~/.NepTrain to set a valid path!")

    for option in ["vasp_path","abacus_path", "mpirun_path", "nep_path", "gpumd_path"]:
        try:
            if utils.get_command_result(["which", Config.get("environ", option)]) is None:
                utils.print_warning(f"The environment variable {option.replace('_path', '')} is not set. If you have set the environment in the submission script, please ignore this warning.")
        except Exception as e:
            pass
