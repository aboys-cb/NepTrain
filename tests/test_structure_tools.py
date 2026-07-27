from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from NepTrain.core.md.health import is_structure_reasonable
from NepTrain.core.perturb.run import PerturbError, perturb, run_perturb
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

    first = perturb(source, count=2, seed=17)
    second = perturb(source, count=2, seed=17)

    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(left.cell.array, right.cell.array)
        np.testing.assert_allclose(left.positions, right.positions)
        assert left.info["Config_type"] == right.info["Config_type"]


def test_perturbations_change_with_the_seed_and_return_a_flat_sequence(
    tmp_path,
):
    source = tmp_path / "structure.xyz"
    write(
        source,
        [
            Atoms(
                "H2",
                positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]],
                cell=[5.0, 5.0, 5.0],
                pbc=True,
            ),
            Atoms(
                "He",
                positions=[[0.0, 0.0, 0.0]],
                cell=[4.0, 4.0, 4.0],
                pbc=True,
            ),
        ],
        format="extxyz",
    )

    first = perturb(source, count=2, seed=17)
    second = perturb(source, count=2, seed=18)

    assert len(first) == 4
    assert len(second) == 4
    assert any(
        not np.allclose(left.positions, right.positions)
        for left, right in zip(first, second, strict=True)
    )
    assert all("max_displacement" in frame.info["Config_type"] for frame in first)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"count": 0}, "count"),
        ({"cell_perturbation": -0.1}, "cell_perturbation"),
        ({"max_displacement": -0.1}, "max_displacement"),
        ({"seed": -1}, "seed"),
    ],
)
def test_perturb_rejects_invalid_requests(tmp_path, kwargs, message):
    source = tmp_path / "structure.xyz"
    write(
        source,
        Atoms("He", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True),
        format="extxyz",
    )

    with pytest.raises(PerturbError, match=message):
        perturb(source, **kwargs)


def test_perturb_cli_adapter_writes_the_flat_result(tmp_path):
    source = tmp_path / "structure.xyz"
    output = tmp_path / "nested" / "perturbed.xyz"
    write(
        source,
        Atoms("He", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True),
        format="extxyz",
    )

    run_perturb(
        SimpleNamespace(
            model_path=str(source),
            cell_pert_fraction=0.02,
            max_displacement=0.05,
            num=3,
            seed=11,
            out_file_path=str(output),
            append=False,
        )
    )

    frames = read(output, index=":", format="extxyz")
    assert len(frames) == 3
    assert all("seed 11" in frame.info["Config_type"] for frame in frames)


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
