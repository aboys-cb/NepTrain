"""VASP labeling entry point."""

from __future__ import annotations

import os
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write

from NepTrain import utils

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
    if not getattr(args, "resource_dir", None):
        raise FileNotFoundError(
            "VASP labeling requires dft.resource_path or --resources"
        )
    resource_dir = Path(args.resource_dir).expanduser().resolve()
    input_file = (
        Path(args.incar)
        if args.incar is not None and os.path.exists(args.incar)
        else Path(__file__).with_name("INCAR")
    )
    command = os.environ.get(
        "NEPTRAIN_VASP_COMMAND",
        f"mpirun -n {args.n_cpu} vasp_std",
    )
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
            flat_single_case=bool(
                getattr(args, "flat_single_case", False)
            ),
        ),
        case_index=atoms_index,
    )
    atoms_index += 1
    return result


def run_vasp(args):
    global atoms_index
    atoms_index = 1
    raw_resource = getattr(args, "resource_dir", None)
    if not raw_resource:
        raise FileNotFoundError(
            "VASP labeling requires dft.resource_path or --resources"
        )
    resource = Path(raw_resource).expanduser()
    if not resource.is_dir():
        raise FileNotFoundError(
            f"VASP pseudopotential root does not exist: {resource}"
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
