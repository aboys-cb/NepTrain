"""Stable labeling Interface with production and development Adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from ase import Atoms


class LabelingError(RuntimeError):
    """Raised when a labeling Adapter cannot satisfy the data contract."""


@dataclass(frozen=True)
class LabelRequest:
    source: Path
    output_file: Path
    work_dir: Path
    append: bool = False
    input_file: Path | None = None
    n_cpu: int = 1
    use_gamma: bool = False
    kspacing: float | None = None
    ka: tuple[int, int, int] = (1, 1, 1)
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelResult:
    backend: str
    output_file: Path
    frames: tuple[Atoms, ...]


LabelAdapter = Callable[[LabelRequest], LabelResult]


def _namespace(request: LabelRequest) -> SimpleNamespace:
    """Translate the narrow Interface to the legacy calculator arguments."""

    return SimpleNamespace(
        model_path=str(request.source),
        out_file_path=str(request.output_file),
        directory=str(request.work_dir),
        append=request.append,
        incar=str(request.input_file) if request.input_file else None,
        n_cpu=request.n_cpu,
        use_gamma=request.use_gamma,
        kspacing=request.kspacing,
        ka=list(request.ka),
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


_ADAPTERS: dict[str, LabelAdapter] = {
    "vasp": _label_vasp,
    "abacus": _label_abacus,
    "toy": _label_toy,
}


def label(request: LabelRequest, backend: str) -> LabelResult:
    """Label structures through the selected Adapter."""

    try:
        adapter = _ADAPTERS[backend]
    except KeyError as error:
        raise LabelingError("label backend must be vasp, abacus, or toy") from error
    if not request.source.exists():
        raise LabelingError(f"label source does not exist: {request.source}")
    if request.n_cpu < 1:
        raise LabelingError("n_cpu must be at least 1")
    result = adapter(request)
    if not result.frames:
        raise LabelingError(f"{backend} produced no labeled structures")
    if not result.output_file.is_file():
        raise LabelingError(f"{backend} did not produce {result.output_file}")
    return result


__all__ = ["LabelRequest", "LabelResult", "LabelingError", "label"]
