"""Stable labeling Interface with production and development Adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from ase import Atoms
from ase.io import read as ase_read


class LabelingError(RuntimeError):
    """Raised when a labeling Adapter cannot satisfy the data contract."""


@dataclass(frozen=True)
class LabelRequest:
    source: Path
    output_file: Path
    work_dir: Path
    append: bool = False
    input_file: Path | None = None
    resource_dir: Path | None = None
    n_cpu: int = 1
    use_gamma: bool = False
    kpoint_mode: str = "auto"
    kspacing: float | None = None
    ka: tuple[int, int, int] = (1, 1, 1)
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelResult:
    backend: str
    output_file: Path
    frames: tuple[Atoms, ...]


LabelAdapter = Callable[[LabelRequest], LabelResult]


@dataclass(frozen=True)
class _AdapterSpec:
    label: LabelAdapter
    supports_spin: bool


def _namespace(request: LabelRequest) -> SimpleNamespace:
    """Translate the narrow Interface to the legacy calculator arguments."""

    return SimpleNamespace(
        model_path=str(request.source),
        out_file_path=str(request.output_file),
        directory=str(request.work_dir),
        append=request.append,
        incar=str(request.input_file) if request.input_file else None,
        resource_dir=str(request.resource_dir) if request.resource_dir else None,
        n_cpu=request.n_cpu,
        use_gamma=request.use_gamma,
        kpoint_mode=request.kpoint_mode,
        kspacing=request.kspacing,
        ka=list(request.ka),
        flat_single_case=bool(request.options.get("flat_single_case", False)),
    )


def _label_vasp(request: LabelRequest) -> LabelResult:
    from .vasp import run_vasp

    frames = run_vasp(_namespace(request))
    return LabelResult("vasp", request.output_file, tuple(frames))


def _label_abacus(request: LabelRequest) -> LabelResult:
    from .abacus import run_abacus

    frames = run_abacus(_namespace(request))
    return LabelResult("abacus", request.output_file, tuple(frames))


def _label_toy(request: LabelRequest) -> LabelResult:
    from .toy import run_toy_teacher

    frames = run_toy_teacher(request)
    return LabelResult("toy", request.output_file, tuple(frames))


_ADAPTERS: dict[str, _AdapterSpec] = {
    "vasp": _AdapterSpec(_label_vasp, supports_spin=False),
    "abacus": _AdapterSpec(_label_abacus, supports_spin=True),
    "toy": _AdapterSpec(_label_toy, supports_spin=True),
}


def _source_contains_spin(source: Path) -> bool:
    paths = [source]
    if source.is_dir():
        paths = sorted(
            {
                path
                for pattern in ("*.xyz", "*.extxyz", "*.vasp", "POSCAR*")
                for path in source.glob(pattern)
                if path.is_file()
            }
        )
    for path in paths:
        frames = ase_read(path, index=":")
        if not isinstance(frames, list):
            frames = [frames]
        if any("spin" in frame.arrays for frame in frames):
            return True
    return False


def label(request: LabelRequest, backend: str) -> LabelResult:
    """Label structures through the selected Adapter."""

    try:
        spec = _ADAPTERS[backend]
    except KeyError as error:
        raise LabelingError("label backend must be vasp, abacus, or toy") from error
    if not request.source.exists():
        raise LabelingError(f"label source does not exist: {request.source}")
    if request.n_cpu < 1:
        raise LabelingError("n_cpu must be at least 1")
    if request.kpoint_mode not in {"auto", "kspacing", "kpoints"}:
        raise LabelingError("kpoint_mode must be auto, kspacing, or kpoints")
    if not spec.supports_spin and _source_contains_spin(request.source):
        raise LabelingError(
            f"{backend} production labeling currently supports non-magnetic "
            "structures only; use the ABACUS Adapter for spin labeling"
        )
    result = spec.label(request)
    if not result.frames:
        raise LabelingError(f"{backend} produced no labeled structures")
    if not result.output_file.is_file():
        raise LabelingError(f"{backend} did not produce {result.output_file}")
    return result


__all__ = ["LabelRequest", "LabelResult", "LabelingError", "label"]
