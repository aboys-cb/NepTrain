#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/28 15:01
# @Author  : 兵
# @email    : 1747193328@qq.com


from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from NepTrain import utils
from .calculator import Nep3Calculator
from .plot import plot_nep_result
from ..training import TrainingRequest, train


def _predict(argparse):
    frames = ase_read(argparse.train_path, index=":", format="extxyz")
    with Nep3Calculator(argparse.nep_txt_path, backend=argparse.inference_backend) as calculator:
        if calculator.model_info.supports("spin"):
            energies, forces, virials, mforces = calculator.calculate_spin(frames)
            np.savetxt(Path(argparse.directory) / "mforce_train.out", np.vstack(mforces))
        else:
            energies, forces, virials = calculator.calculate(frames)
        np.savetxt(Path(argparse.directory) / "energy_train.out", np.asarray(energies))
        np.savetxt(Path(argparse.directory) / "force_train.out", np.vstack(forces))
        np.savetxt(Path(argparse.directory) / "virial_train.out", np.asarray(virials))

def run_nep(argparse):
    Path(argparse.directory).mkdir(parents=True, exist_ok=True)
    if argparse.prediction:
        _predict(argparse)
        plot_nep_result(argparse.directory)
        utils.print_success("NEP prediction completed through NEPAdapters!")
        return
    request = TrainingRequest(
        config_file=Path(argparse.nep_in_path),
        train_file=Path(argparse.train_path),
        test_file=Path(argparse.test_path) if argparse.test_path and Path(argparse.test_path).is_file() else None,
        output_dir=Path(argparse.directory),
        restart_file=Path(argparse.restart_file) if argparse.restart_file else None,
        continue_steps=argparse.continue_step,
        device=argparse.device,
        torch_backend=argparse.torch_backend,
        precision=argparse.precision,
        use_compile=argparse.use_compile,
    )
    result = train(request, argparse.backend)
    plot_nep_result(argparse.directory)
    utils.print_success(
        f"NEP training completed with {result.backend}; best model: {result.best_model}"
    )
