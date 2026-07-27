from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from NepTrain import utils
from NepTrain.core.gpumd.utils import get_dump_interval
from NepTrain.core.perturb.run import perturb
from NepTrain.core.select.filter import (
    adjust_reasonable,
    calculate_pairwise_distances,
)
from NepTrain.core.select.select import (
    compute_min_bond_lengths,
    filter_by_bonds,
)


def test_min_bond_lengths_use_periodic_minimum_image():
    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    assert compute_min_bond_lengths(atoms)[("H", "H")] == pytest.approx(0.2)


def test_reference_bond_filter_rejects_unknown_element_pairs():
    reference = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    candidate = Atoms(
        "HHe",
        positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )

    good, bad = filter_by_bonds([candidate], reference)

    assert good == []
    assert bad == [candidate]


def test_pairwise_distances_handle_skew_cells_and_unwrapped_positions():
    cell = np.array(
        [[3.0, 0.0, 0.0], [1.2, 2.7, 0.0], [0.4, 0.3, 2.5]]
    )
    positions = np.array([[0.1, 0.2, 0.3], [7.4, -2.0, 5.1]])
    atoms = Atoms("H2", positions=positions, cell=cell, pbc=True)

    actual = calculate_pairwise_distances(cell, positions, fractional=False)

    np.testing.assert_allclose(actual, atoms.get_all_distances(mic=True))


def test_reasonable_filter_detects_overlap_across_periodic_boundary():
    atoms = Atoms(
        "H2",
        positions=[[0.05, 0.0, 0.0], [3.95, 0.0, 0.0]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )

    assert not adjust_reasonable(atoms, coefficient=0.7)


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


def test_structure_iteration_does_not_swallow_keyboard_interrupt(tmp_path):
    source = tmp_path / "structure.xyz"
    write(source, Atoms("H", positions=[[0.0, 0.0, 0.0]]))

    @utils.iter_path_to_atoms(["*.xyz"], show_progress=False)
    def interrupted(_atoms):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted(source)


def test_fail_fast_structure_iteration_rejects_unreadable_input(tmp_path):
    source = tmp_path / "broken.xyz"
    source.write_text("not an xyz file\n", encoding="utf-8")

    @utils.iter_path_to_atoms(["*.xyz"], show_progress=False, fail_fast=True)
    def identity(atoms):
        return atoms

    with pytest.raises(Exception):
        identity(source)


def test_invalid_gpumd_dump_interval_is_not_silently_defaulted(tmp_path):
    run_input = tmp_path / "run.in"
    run_input.write_text("dump_thermo not-an-integer\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid dump_thermo interval"):
        get_dump_interval(run_input)
