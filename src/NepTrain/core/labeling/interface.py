"""Stable labeling Interface for DFT and teacher-model Adapters."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
import shlex
import subprocess
import tempfile
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
    settings: Mapping[str, Any] = field(default_factory=dict)
    append: bool = False


@dataclass(frozen=True)
class LabelResult:
    backend: str
    output_file: Path
    frames: tuple[Atoms, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)


LabelAdapter = Callable[[LabelRequest], LabelResult]


@dataclass(frozen=True)
class _AdapterSpec:
    label: LabelAdapter
    origin: str
    supports_spin_input: bool


def _path_setting(request: LabelRequest, name: str) -> Path | None:
    value = request.settings.get(name)
    if value in {None, ""}:
        return None
    return Path(value).expanduser().resolve()


def _dft_namespace(request: LabelRequest):
    from types import SimpleNamespace

    settings = request.settings
    return SimpleNamespace(
        model_path=str(request.source),
        out_file_path=str(request.output_file),
        directory=str(request.work_dir),
        append=False,
        incar=(
            str(_path_setting(request, "input_file"))
            if _path_setting(request, "input_file")
            else None
        ),
        resource_dir=(
            str(_path_setting(request, "resource_dir"))
            if _path_setting(request, "resource_dir")
            else None
        ),
        resource_manifest=(
            str(_path_setting(request, "resource_manifest"))
            if _path_setting(request, "resource_manifest")
            else None
        ),
        n_cpu=int(settings.get("n_cpu", 1)),
        use_gamma=bool(settings.get("use_gamma", False)),
        kpoint_mode=str(settings.get("kpoint_mode", "auto")),
        kspacing=settings.get("kspacing"),
        ka=[int(value) for value in settings.get("ka", (1, 1, 1))],
        flat_single_case=bool(settings.get("flat_single_case", False)),
    )


def _label_vasp(request: LabelRequest) -> LabelResult:
    from ..dft.vasp import run_vasp

    frames = run_vasp(_dft_namespace(request))
    return LabelResult(
        "vasp",
        request.output_file,
        tuple(frames),
        {"origin": "dft", "engine": "vasp"},
    )


def _label_abacus(request: LabelRequest) -> LabelResult:
    from ..dft.abacus import run_abacus

    frames = run_abacus(_dft_namespace(request))
    return LabelResult(
        "abacus",
        request.output_file,
        tuple(frames),
        {"origin": "dft", "engine": "abacus"},
    )


def _label_toy(request: LabelRequest) -> LabelResult:
    from ..dft.toy import run_toy_teacher

    frames = run_toy_teacher(request)
    return LabelResult(
        "toy",
        request.output_file,
        tuple(frames),
        {"origin": "development", "engine": "toy"},
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_model(request: LabelRequest) -> LabelResult:
    settings = request.settings
    model = _path_setting(request, "model_file")
    if model is None or not model.is_file():
        raise LabelingError("model labeling requires an existing model_file")
    runner = str(settings.get("runner", "")).strip()
    command = shlex.split(runner)
    if not command:
        raise LabelingError("model labeling requires a runner command")
    model_name = str(settings.get("model_name", "")).strip()
    if not model_name:
        raise LabelingError("model labeling requires model_name")
    device = str(settings.get("device", "cuda"))
    precision = str(settings.get("precision", "float32"))
    request.work_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            *command,
            "--model",
            str(model),
            "--input",
            str(request.source),
            "--output",
            str(request.output_file),
            "--device",
            device,
            "--precision",
            precision,
        ],
        cwd=request.work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    (request.work_dir / "runner.stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (request.work_dir / "runner.stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise LabelingError(
            f"{model_name} runner exited with code {completed.returncode}"
            + (f": {detail}" if detail else "")
        )
    if not request.output_file.is_file():
        raise LabelingError(
            f"{model_name} runner completed without producing its output"
        )
    loaded = ase_read(request.output_file, index=":", format="extxyz")
    frames = loaded if isinstance(loaded, list) else [loaded]
    return LabelResult(
        "model",
        request.output_file,
        tuple(frames),
        {
            "origin": "teacher_model",
            "engine": model_name,
            "model_sha256": _sha256(model),
            "runner": runner,
            "device": device,
            "precision": precision,
        },
    )


_ADAPTERS: dict[str, _AdapterSpec] = {
    "vasp": _AdapterSpec(_label_vasp, "dft", supports_spin_input=False),
    "abacus": _AdapterSpec(_label_abacus, "dft", supports_spin_input=True),
    "model": _AdapterSpec(
        _label_model,
        "teacher_model",
        supports_spin_input=True,
    ),
    "toy": _AdapterSpec(_label_toy, "development", supports_spin_input=True),
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


def _annotate_provenance(
    frames: list[Atoms],
    *,
    backend: str,
    provenance: Mapping[str, Any],
) -> None:
    for frame in frames:
        frame.info["neptrain_label_backend"] = backend
        frame.info["neptrain_label_origin"] = str(provenance["origin"])
        if provenance.get("engine"):
            frame.info["neptrain_label_engine"] = str(provenance["engine"])
        if provenance.get("model_sha256"):
            frame.info["neptrain_teacher_model_sha256"] = str(
                provenance["model_sha256"]
            )


def label(request: LabelRequest, backend: str) -> LabelResult:
    """Label structures through the selected Adapter and publish atomically."""

    try:
        spec = _ADAPTERS[backend]
    except KeyError as error:
        raise LabelingError(
            "labeling backend must be vasp, abacus, model, or toy"
        ) from error
    if not request.source.exists():
        raise LabelingError(f"label source does not exist: {request.source}")
    input_frames = _source_frames(request.source)
    if not input_frames:
        raise LabelingError(
            f"label source contains no readable structures: {request.source}"
        )
    if not spec.supports_spin_input and any(
        "spin" in frame.arrays for frame in input_frames
    ):
        raise LabelingError(
            f"{backend} does not produce spin/mforce labels; use an Adapter "
            "that emits canonical spin:R:3 and mforce:R:3 data"
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
                "existing append target violates the scientific data "
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
        provenance = {
            "version": 1,
            "backend": backend,
            "origin": spec.origin,
            **dict(result.provenance),
        }
        _annotate_provenance(
            output_frames,
            backend=backend,
            provenance=provenance,
        )
        try:
            bound_ids = bind_labeled_frames_to_inputs(
                input_frames,
                output_frames,
            )
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
    return LabelResult(
        result.backend,
        request.output_file,
        tuple(output_frames),
        provenance,
    )


__all__ = ["LabelRequest", "LabelResult", "LabelingError", "label"]
