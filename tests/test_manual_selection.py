from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from NepTrain.core.scientific_data import STRUCTURE_ID_VERSION, structure_id
from NepTrain.core.select import run as select_run


class _PositionDescriptor:
    def __init__(self, *_args, **_kwargs):
        pass

    def get_structures_descriptors(self, structures):
        return np.asarray(
            [[float(frame.positions[0, 0])] for frame in structures],
            dtype=np.float64,
        )

    def get_structures_atomic_descriptors(self, structures):
        return self.get_structures_descriptors(structures)


def _frame(x: float) -> Atoms:
    return Atoms(
        "H",
        positions=[[x, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )


def test_manual_selection_uses_shared_policy_and_writes_provenance(
    tmp_path: Path,
    monkeypatch,
):
    candidates_path = tmp_path / "candidates.xyz"
    reference_path = tmp_path / "reference.xyz"
    output_path = tmp_path / "selected.xyz"
    report_path = tmp_path / "selection.json"
    ase_write(
        candidates_path,
        [_frame(0.0), _frame(1.0), _frame(2.0), _frame(2.0)],
        format="extxyz",
    )
    ase_write(reference_path, [_frame(0.0)], format="extxyz")
    monkeypatch.setattr(
        select_run,
        "DescriptorCalculator",
        _PositionDescriptor,
    )
    args = SimpleNamespace(
        trajectory_paths=[str(candidates_path)],
        base=str(reference_path),
        nep=None,
        backend="auto",
        descriptor_reduction="global_mean",
        max_selected=3,
        min_novelty=0.0,
        filter=False,
        rejected_out=None,
        out_file_path=str(output_path),
        report=str(report_path),
        r_cut=6.0,
        n_max=8,
        l_max=6,
    )

    report = select_run.run_select(args)

    selected = ase_read(output_path, index=":", format="extxyz")
    assert report["candidate_count_read"] == 4
    assert report["candidate_count_after_deduplication"] == 3
    assert report["duplicate_count"] == 1
    assert report["selected_count"] == 2
    assert len(selected) == 2
    assert structure_id(_frame(0.0)) not in report["selected_ids"]
    assert set(report["selected_ids"]) == {
        structure_id(_frame(1.0)),
        structure_id(_frame(2.0)),
    }

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["protocol"] == "neptrain.manual-selection.v1"
    assert persisted["structure_id_version"] == STRUCTURE_ID_VERSION
    assert persisted["descriptor"]["kind"] == "soap"
    assert persisted["descriptor"]["reduction"] == "global_mean"
    assert persisted["output"] == str(output_path.resolve())
