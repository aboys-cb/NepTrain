"""VASP labeling entry point."""

from __future__ import annotations

import os
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write

from NepTrain import Config, module_path, utils
from NepTrain.core.utils import check_env

from .native import NativeVaspRequest, run_native_vasp


atoms_index = 1


@utils.iter_path_to_atoms(
    ["*.vasp", "*.xyz"],
    show_progress=True,
    fail_fast=True,
    description="VASP calculation progress",
)
def calculate_vasp(atoms: Atoms, args):
    global atoms_index
    resource_dir = Path(
        getattr(args, "resource_dir", None)
        or Config.get("environ", "potcar_path")
    ).expanduser().resolve()
    input_file = (
        Path(args.incar)
        if args.incar is not None and os.path.exists(args.incar)
        else Path(module_path) / "core/dft/vasp/INCAR"
    )
    command = (
        f"{Config.get('environ', 'mpirun_path')} -n {args.n_cpu} "
        f"{Config.get('environ', 'vasp_path')}"
    )
    command = os.environ.get("NEPTRAIN_VASP_COMMAND", command)
    result = run_native_vasp(
        atoms,
        NativeVaspRequest(
            work_dir=Path(args.directory).resolve(),
            resource_dir=resource_dir,
            command=command,
            input_file=input_file.resolve(),
            use_gamma=bool(args.use_gamma),
            kpoint_mode=str(getattr(args, "kpoint_mode", "auto")),
            kspacing=args.kspacing,
            ka=tuple(int(value) for value in args.ka),
        ),
        case_index=atoms_index,
    )
    atoms_index += 1
    return result


def run_vasp(args):
    global atoms_index
    atoms_index = 1
    check_env(
        potcar_path=getattr(args, "resource_dir", None),
        require_potcar=True,
        commands=("vasp_path", "mpirun_path"),
    )
    result = calculate_vasp(args.model_path, args)
    output = Path(args.out_file_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(result) and isinstance(result[0], list):
        result = [atoms for group in result for atoms in group]
    if not result:
        raise RuntimeError("VASP produced no labeled structures")
    ase_write(output, result, format="extxyz", append=args.append)
    utils.print_success("VASP calculation task completed!")
    return result


__all__ = ["calculate_vasp", "run_vasp"]
