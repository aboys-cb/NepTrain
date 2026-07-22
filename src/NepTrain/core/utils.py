#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/29 21:52
# @Author  : 兵
# @email    : 1747193328@qq.com
import os

from NepTrain import Config
from NepTrain import utils


def check_env(*, potcar_path=None, require_potcar=False, commands=()):
    if require_potcar:
        configured_potcar = potcar_path or Config.get("environ", "potcar_path")
        if not os.path.exists(os.path.expanduser(str(configured_potcar))):
            raise FileNotFoundError(
                f"VASP pseudopotential root does not exist: {configured_potcar}"
            )

    for option in commands:
        try:
            if utils.get_command_result(["which", Config.get("environ", option)]) is None:
                utils.print_warning(f"The environment variable {option.replace('_path', '')} is not set. If you have set the environment in the submission script, please ignore this warning.")
        except Exception:
            pass
