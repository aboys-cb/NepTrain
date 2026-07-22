"""Structure labeling through a stable Adapter seam."""

from __future__ import annotations

from pathlib import Path

from .interface import LabelRequest, LabelResult, LabelingError, label


def run_dft(argparse):
    backend = argparse.software or "vasp"
    output = Path(argparse.out_file_path or f"./{backend}_scf.xyz")
    work_dir = Path(argparse.directory or f"./cache/{backend}")
    options = {"profile": getattr(argparse, "teacher_profile", "ordinary")}
    return label(
        LabelRequest(
            source=Path(argparse.model_path),
            output_file=output,
            work_dir=work_dir,
            append=argparse.append,
            input_file=Path(argparse.incar) if argparse.incar else None,
            resource_dir=Path(getattr(argparse, "resource_dir", ""))
            if getattr(argparse, "resource_dir", None)
            else None,
            n_cpu=argparse.n_cpu,
            use_gamma=argparse.use_gamma,
            kpoint_mode="kspacing" if argparse.kspacing is not None else "auto",
            kspacing=argparse.kspacing,
            ka=tuple(argparse.ka),
            options=options,
        ),
        backend,
    )


def run_vasp(argparse):
    from .vasp import run_vasp as implementation

    return implementation(argparse)


__all__ = [
    "LabelRequest",
    "LabelResult",
    "LabelingError",
    "label",
    "run_dft",
    "run_vasp",
]
