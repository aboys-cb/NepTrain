"""VASP labeling entry point."""

from __future__ import annotations

import os
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write
from rich.progress import track

from ...structures import read_structures
from .native import NativeVaspRequest, run_native_vasp


def calculate_vasp(atoms: Atoms, args, *, case_index: int = 1):
    if not getattr(args, "resource_dir", None):
        raise FileNotFoundError(
            "VASP labeling requires labeling.resource_path or --resources"
        )
    resource_dir = Path(args.resource_dir).expanduser().resolve()
    raw_manifest = getattr(args, "resource_manifest", None)
    if not raw_manifest:
        raise FileNotFoundError(
            "VASP labeling requires a content-addressed POTCAR manifest"
        )
    resource_manifest = Path(raw_manifest).expanduser().resolve()
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
            resource_manifest=resource_manifest,
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
        case_index=case_index,
    )
    return result


def run_vasp(args):
    raw_resource = getattr(args, "resource_dir", None)
    if not raw_resource:
        raise FileNotFoundError(
            "VASP labeling requires labeling.resource_path or --resources"
        )
    resource = Path(raw_resource).expanduser()
    if not resource.is_dir():
        raise FileNotFoundError(
            f"VASP pseudopotential root does not exist: {resource}"
        )
    manifest = getattr(args, "resource_manifest", None)
    if not manifest or not Path(manifest).expanduser().is_file():
        raise FileNotFoundError(
            "VASP POTCAR manifest does not exist: "
            f"{manifest or '<not configured>'}"
        )
    result = [
        calculate_vasp(atoms, args, case_index=index)
        for index, atoms in enumerate(
            track(
                read_structures(args.model_path),
                description="VASP calculation progress",
            ),
            start=1,
        )
    ]
    output = Path(args.out_file_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(result) and isinstance(result[0], list):
        result = [atoms for group in result for atoms in group]
    if not result:
        raise RuntimeError("VASP produced no labeled structures")
    ase_write(output, result, format="extxyz", append=args.append)
    print("VASP calculation task completed!")
    return result


__all__ = ["calculate_vasp", "run_vasp"]
