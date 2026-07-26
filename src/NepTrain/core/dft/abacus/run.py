"""ABACUS labeling entry point."""

import os
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write

from NepTrain import utils

from .io import StructureVar, read_input_file
from .native import NativeAbacusRequest, run_native_abacus

atoms_index = 1


@utils.iter_path_to_atoms(
    ["*.vasp", "*.xyz"],
    show_progress=True,
    fail_fast=True,
    description="ABACUS calculation progress",
)
def calculate_abacus(atoms: Atoms, args):
    global atoms_index
    StructureVar.init(getattr(args, "resource_dir", None) or "./")

    if args.incar is not None and os.path.exists(args.incar):
        input_dict = read_input_file(args.incar)
    else:
        input_dict = read_input_file(Path(__file__).with_name("INPUT"))
    command = os.environ.get(
        "NEPTRAIN_ABACUS_COMMAND",
        f"mpirun -n {args.n_cpu} abacus",
    )
    result = run_native_abacus(
        atoms,
        NativeAbacusRequest(
            work_dir=Path(args.directory).resolve(),
            resource_dir=Path(getattr(args, "resource_dir", None) or "./").resolve(),
            command=command,
            input_parameters=input_dict,
            use_gamma=bool(args.use_gamma),
            kpoint_mode=str(getattr(args, "kpoint_mode", "auto")),
            kspacing=args.kspacing,
            ka=tuple(int(value) for value in args.ka),
            flat_single_case=bool(
                getattr(args, "flat_single_case", False)
            ),
        ),
        case_index=atoms_index,
    )
    atoms_index += 1
    return result


def run_abacus(args):
    global atoms_index
    atoms_index = 1
    result = calculate_abacus(args.model_path, args)
    path = os.path.dirname(args.out_file_path)
    if path and not os.path.exists(path):
        os.makedirs(path)
    if len(result) and isinstance(result[0], list):
        result = [atoms for _list in result for atoms in _list]
    if not result:
        raise RuntimeError("ABACUS produced no labeled structures")
    ase_write(args.out_file_path, result, format="extxyz", append=args.append)

    utils.print_success("ABACUS calculation task completed!")
    return result
