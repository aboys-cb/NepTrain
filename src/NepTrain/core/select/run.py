"""Manual structure selection through the production FPS policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from ..content_addressing import file_sha256
from ..fps import hierarchical_farthest_point_sampling
from ..nep.calculator import DescriptorCalculator
from ..scientific_data import STRUCTURE_ID_VERSION, structure_id
from ..md.health import is_structure_reasonable
from ..persistence import atomic_write_json


class SelectionError(RuntimeError):
    """Raised when a manual selection cannot produce a valid result."""


@dataclass(frozen=True)
class _Candidate:
    frame: Atoms
    source: str
    frame_index: int

    @property
    def preference(self) -> tuple[str, int]:
        return self.source, self.frame_index


def _frames(path: str | Path, *, role: str) -> list[Atoms]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise SelectionError(f"{role} does not exist: {source}")
    try:
        value = ase_read(source, index=":", format="extxyz")
    except Exception as error:
        raise SelectionError(f"cannot read {role} {source}: {error}") from error
    frames = value if isinstance(value, list) else [value]
    if not frames:
        raise SelectionError(f"{role} contains no structures: {source}")
    return frames


def _stratum(frame: Atoms, source: str) -> str:
    values = []
    for label, key in (
        ("W", "md_window"),
        ("R", "route_id"),
        ("T", "temperature"),
        ("P", "pressure"),
    ):
        if key in frame.info:
            values.append(f"{label}={frame.info[key]}")
    return "|".join(values) if values else f"source={Path(source).name}"


def _descriptor_calculator(args, frames: Sequence[Atoms]) -> tuple[Any, dict[str, Any]]:
    if args.nep:
        model = Path(args.nep).expanduser().resolve()
        if not model.is_file():
            raise SelectionError(f"NEP model does not exist: {model}")
        calculator = DescriptorCalculator(
            "nep",
            model_file=model,
            backend=args.backend,
        )
        return calculator, {
            "kind": "nep",
            "model": str(model),
            "model_sha256": file_sha256(model),
            "backend": args.backend,
        }
    species = sorted(
        {
            symbol
            for frame in frames
            for symbol in frame.get_chemical_symbols()
        }
    )
    try:
        calculator = DescriptorCalculator(
            "soap",
            species=species,
            r_cut=args.r_cut,
            n_max=args.n_max,
            l_max=args.l_max,
        )
    except ImportError as error:
        raise SelectionError(
            "SOAP selection requires dscribe; install NepTrain[soap] "
            "or pass --nep"
        ) from error
    return calculator, {
        "kind": "soap",
        "species": species,
        "r_cut": args.r_cut,
        "n_max": args.n_max,
        "l_max": args.l_max,
    }


def run_select(args) -> dict[str, Any]:
    if args.max_selected <= 0:
        raise SelectionError("--max-selected must be a positive integer")
    if args.min_novelty < 0:
        raise SelectionError("--min-novelty must be non-negative")
    if args.filter is not False and args.filter is not None and args.filter < 0:
        raise SelectionError("--filter coefficient must be non-negative")

    records = []
    for raw_source in args.trajectory_paths:
        source = str(Path(raw_source).expanduser().resolve())
        records.extend(
            _Candidate(frame, source, index)
            for index, frame in enumerate(
                _frames(source, role="candidate trajectory")
            )
        )
    read_count = len(records)

    rejected = []
    if args.filter is not False and args.filter is not None:
        coefficient = float(args.filter)
        accepted = []
        for record in records:
            if is_structure_reasonable(
                record.frame,
                min_distance_ratio=coefficient,
            ):
                accepted.append(record)
            else:
                rejected.append(record.frame)
        records = accepted
    if args.rejected_out and rejected:
        rejected_path = Path(args.rejected_out).expanduser().resolve()
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        ase_write(rejected_path, rejected, format="extxyz")

    unique: dict[str, _Candidate] = {}
    for record in records:
        identifier = structure_id(record.frame)
        current = unique.get(identifier)
        if current is None or record.preference < current.preference:
            unique[identifier] = record
    identifiers = sorted(unique)
    candidates = [unique[identifier].frame for identifier in identifiers]
    if not candidates:
        raise SelectionError("no valid candidate structures remain after filtering")

    references = (
        _frames(args.base, role="reference dataset") if args.base else []
    )
    calculator, descriptor_record = _descriptor_calculator(
        args, [*candidates, *references]
    )
    candidate_descriptors = np.asarray(
        calculator.get_structures_descriptors(candidates),
        dtype=np.float64,
    )
    reference_descriptors = (
        np.asarray(
            calculator.get_structures_descriptors(references),
            dtype=np.float64,
        )
        if references
        else None
    )
    result = hierarchical_farthest_point_sampling(
        candidates,
        candidate_descriptors,
        identifiers,
        [
            _stratum(unique[identifier].frame, unique[identifier].source)
            for identifier in identifiers
        ],
        budget=args.max_selected,
        min_novelty=args.min_novelty,
        reference_structures=references,
        reference_descriptors=reference_descriptors,
    )
    if not result.selected_indices:
        raise SelectionError(
            "FPS selected no structures; lower --min-novelty or check that "
            "the candidates are not already present in --base"
        )

    selected = [candidates[index] for index in result.selected_indices]
    output = Path(args.out_file_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ase_write(output, selected, format="extxyz")
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else output.with_suffix(".selection.json")
    )
    report = {
        "protocol": "neptrain.manual-selection.v1",
        "structure_id_version": STRUCTURE_ID_VERSION,
        "candidate_count_read": read_count,
        "candidate_count_after_filter": len(records),
        "candidate_count_after_deduplication": len(candidates),
        "duplicate_count": len(records) - len(candidates),
        "rejected_count": len(rejected),
        "reference_count": len(references),
        "selected_count": len(selected),
        "selected_ids": list(result.selected_ids),
        "selected_novelty": list(result.selected_novelty),
        "counts_by_stratum": dict(result.counts_by_stratum),
        "remaining_novelty": result.remaining_novelty,
        "groups": {
            "+".join(key): asdict(group)
            for key, group in result.groups.items()
        },
        "descriptor": descriptor_record,
        "output": str(output),
    }
    atomic_write_json(report_path, report)
    print(
        f"Selected {len(selected)} of {read_count} structures -> {output}\n"
        f"Selection report -> {report_path}"
    )
    return report


__all__ = ["SelectionError", "run_select"]
