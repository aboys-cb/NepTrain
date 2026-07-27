"""ABACUS labeling entry point."""

import os
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write
from rich.progress import track

from ...structures import read_structures
from .io import read_input_file
from .native import NativeAbacusRequest, run_native_abacus


def calculate_abacus(atoms: Atoms, args, *, case_index: int = 1):
    resource_dir = Path(getattr(args, "resource_dir", None) or "./").resolve()
    raw_manifest = getattr(args, "resource_manifest", None)
    resource_manifest = (
        Path(raw_manifest).expanduser().resolve()
        if raw_manifest
        else resource_dir / "abacus-resources.json"
    )
    if not resource_manifest.is_file():
        raise FileNotFoundError(
            f"ABACUS resource manifest does not exist: {resource_manifest}"
        )

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
            resource_dir=resource_dir,
            resource_manifest=resource_manifest,
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
        case_index=case_index,
    )
    return result


def run_abacus(args):
    result = [
        calculate_abacus(atoms, args, case_index=index)
        for index, atoms in enumerate(
            track(
                read_structures(args.model_path),
                description="ABACUS calculation progress",
            ),
            start=1,
        )
    ]
    path = os.path.dirname(args.out_file_path)
    if path:
        os.makedirs(path, exist_ok=True)
    if len(result) and isinstance(result[0], list):
        result = [atoms for _list in result for atoms in _list]
    if not result:
        raise RuntimeError("ABACUS produced no labeled structures")
    ase_write(args.out_file_path, result, format="extxyz", append=args.append)

    print("ABACUS calculation task completed!")
    return result
