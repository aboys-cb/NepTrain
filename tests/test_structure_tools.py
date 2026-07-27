from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from NepTrain.core.md.health import is_structure_reasonable
from NepTrain.core.perturb.run import perturb
from NepTrain.core.structures import StructureReadError, read_structures


def test_reasonable_filter_detects_overlap_across_periodic_boundary():
    atoms = Atoms(
        "H2",
        positions=[[0.05, 0.0, 0.0], [3.95, 0.0, 0.0]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )

    assert not is_structure_reasonable(atoms, min_distance_ratio=0.7)


def test_perturbations_are_reproducible_with_the_same_seed(tmp_path):
    source = tmp_path / "structure.xyz"
    write(
        source,
        Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]],
            cell=[5.0, 5.0, 5.0],
            pbc=True,
        ),
    )

    first = perturb(source, num=2, rng=np.random.default_rng(17))[0]
    second = perturb(source, num=2, rng=np.random.default_rng(17))[0]

    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(left.cell.array, right.cell.array)
        np.testing.assert_allclose(left.positions, right.positions)


def test_structure_reader_does_not_swallow_keyboard_interrupt(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "structure.xyz"
    source.write_text("unused\n", encoding="utf-8")

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("NepTrain.core.structures.ase_read", interrupted)
    with pytest.raises(KeyboardInterrupt):
        read_structures(source)


def test_structure_reader_rejects_unreadable_input(tmp_path):
    source = tmp_path / "broken.xyz"
    source.write_text("not an xyz file\n", encoding="utf-8")

    with pytest.raises(StructureReadError, match="failed to read"):
        read_structures(source)
