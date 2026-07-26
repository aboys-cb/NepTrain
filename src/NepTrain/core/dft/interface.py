"""Stable labeling Interface with production and development Adapters."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from ..scientific_data import (
    ScientificDataError,
    bind_labeled_frames_to_inputs,
    labeled_input_structure_ids,
    structure_id,
    validate_labeled_frames,
)


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
    resource_manifest: Path | None = None
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
        resource_manifest=(
            str(request.resource_manifest) if request.resource_manifest else None
        ),
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


def _source_frames(source: Path) -> list[Atoms]:
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
    loaded_frames = []
    for path in paths:
        loaded = ase_read(path, index=":")
        if not isinstance(loaded, list):
            loaded = [loaded]
        loaded_frames.extend(loaded)
    return loaded_frames


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
    input_frames = _source_frames(request.source)
    if not input_frames:
        raise LabelingError(f"label source contains no readable structures: {request.source}")
    if not spec.supports_spin and any(
        "spin" in frame.arrays for frame in input_frames
    ):
        raise LabelingError(
            f"{backend} does not produce spin/mforce labels; use the ABACUS "
            "Adapter for canonical spin:R:3 input"
        )
    previous_frames = []
    if request.append and request.output_file.is_file():
        previous = ase_read(request.output_file, index=":")
        previous_frames = previous if isinstance(previous, list) else [previous]
        try:
            validate_labeled_frames(previous_frames)
            previous_ids = labeled_input_structure_ids(previous_frames)
        except ScientificDataError as error:
            raise LabelingError(
                f"existing append target violates the scientific data "
                f"contract: {error}"
            ) from error
    else:
        previous_ids = []
    input_ids = [structure_id(frame) for frame in input_frames]
    duplicates = sorted(set(previous_ids).intersection(input_ids))
    if duplicates:
        raise LabelingError(
            "append would label an input structure already present in the "
            f"output ({duplicates[0]})"
        )
    if len(input_ids) != len(set(input_ids)):
        raise LabelingError("label source contains duplicate physical structures")

    request.output_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{request.output_file.name}.",
        suffix=".extxyz.tmp",
        dir=request.output_file.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        adapter_request = replace(
            request,
            output_file=temporary,
            append=False,
        )
        result = spec.label(adapter_request)
        if not result.frames:
            raise LabelingError(f"{backend} produced no labeled structures")
        if Path(result.output_file).resolve() != temporary.resolve():
            raise LabelingError(
                f"{backend} wrote outside its unpublished result path"
            )
        if not temporary.is_file():
            raise LabelingError(f"{backend} did not produce {temporary}")
        persisted = ase_read(temporary, index=":", format="extxyz")
        output_frames = persisted if isinstance(persisted, list) else [persisted]
        if len(output_frames) != len(result.frames):
            raise LabelingError(
                f"{backend} returned {len(result.frames)} frames but persisted "
                f"{len(output_frames)}"
            )
        try:
            bound_ids = bind_labeled_frames_to_inputs(input_frames, output_frames)
            validate_labeled_frames(output_frames)
        except ScientificDataError as error:
            raise LabelingError(
                f"{backend} result violates the scientific data contract: {error}"
            ) from error
        expected_ids = [*previous_ids, *bound_ids]
        ase_write(
            temporary,
            [*previous_frames, *output_frames],
            format="extxyz",
        )
        restored = ase_read(temporary, index=":", format="extxyz")
        restored_frames = restored if isinstance(restored, list) else [restored]
        try:
            validate_labeled_frames(restored_frames)
            restored_ids = labeled_input_structure_ids(restored_frames)
        except ScientificDataError as error:
            raise LabelingError(
                f"{backend} result failed its publication roundtrip: {error}"
            ) from error
        if restored_ids != expected_ids:
            raise LabelingError(
                f"{backend} result publication changed input ownership or order"
            )
        temporary.replace(request.output_file)
    finally:
        temporary.unlink(missing_ok=True)
    return LabelResult(result.backend, request.output_file, tuple(output_frames))


__all__ = ["LabelRequest", "LabelResult", "LabelingError", "label"]
