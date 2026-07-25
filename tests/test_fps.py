from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from NepTrain.core.fps import hierarchical_farthest_point_sampling


def _atoms(symbols: str) -> Atoms:
    return Atoms(symbols, positions=np.zeros((len(Atoms(symbols)), 3)))


def test_sqrt_quotas_and_soft_strata_balance():
    structures = [_atoms("Fe")] * 9 + [_atoms("FeO")] * 4
    descriptors = np.asarray(
        [[float(value), 0.0] for value in range(1, 10)]
        + [[0.0, float(value)] for value in range(1, 5)]
    )
    ids = [f"fe-{value}" for value in range(9)] + [
        f"feo-{value}" for value in range(4)
    ]
    strata = ["cold", "hot"] * 4 + ["cold"] + ["cold", "hot"] * 2

    result = hierarchical_farthest_point_sampling(
        structures,
        descriptors,
        ids,
        strata,
        budget=5,
    )

    assert result.groups[("Fe",)].initial_quota == 3
    assert result.groups[("Fe", "O")].initial_quota == 2
    assert result.groups[("Fe",)].selected_count == 3
    assert result.groups[("Fe", "O")].selected_count == 2
    assert result.counts_by_stratum == {"cold": 3, "hot": 2}


def test_budget_smaller_than_element_group_count_is_rejected():
    with pytest.raises(ValueError, match="smaller than the number"):
        hierarchical_farthest_point_sampling(
            [_atoms("Fe"), _atoms("O")],
            np.asarray([[0.0], [1.0]]),
            ["fe", "o"],
            ["a", "b"],
            budget=1,
        )


def test_references_warm_start_only_the_same_element_set():
    result = hierarchical_farthest_point_sampling(
        [_atoms("Fe"), _atoms("Fe"), _atoms("FeO")],
        np.asarray([[0.0], [10.0], [100.0]]),
        ["fe-near", "fe-far", "feo"],
        ["s", "s", "s"],
        budget=2,
        min_novelty=0.1,
        reference_structures=[_atoms("Fe"), _atoms("FeO")],
        reference_descriptors=np.asarray([[0.0], [1000.0]]),
    )

    assert result.groups[("Fe",)].reference_count == 1
    assert result.groups[("Fe", "O")].reference_count == 1
    assert "fe-far" in result.selected_ids
    assert "fe-near" not in result.selected_ids
    assert "feo" in result.selected_ids


def test_unused_quota_is_redistributed_without_crossing_novelty_gate():
    result = hierarchical_farthest_point_sampling(
        [_atoms("Fe")] * 3 + [_atoms("O")] * 3,
        np.asarray([[0.0], [0.0], [0.0], [2.0], [4.0], [8.0]]),
        ["fe-0", "fe-1", "fe-2", "o-2", "o-4", "o-8"],
        ["a", "a", "a", "a", "b", "c"],
        budget=4,
        min_novelty=0.01,
        reference_structures=[_atoms("Fe"), _atoms("O")],
        reference_descriptors=np.asarray([[0.0], [0.0]]),
    )

    assert result.groups[("Fe",)].initial_quota == 2
    assert result.groups[("Fe",)].selected_count == 0
    assert result.groups[("O",)].selected_count == 3
    assert len(result.selected_ids) == 3
    assert all(value.startswith("o-") for value in result.selected_ids)


def test_selection_is_stable_under_candidate_reordering():
    structures = [
        _atoms("Fe"),
        _atoms("Fe"),
        _atoms("Fe"),
        _atoms("FeO"),
        _atoms("FeO"),
    ]
    descriptors = np.asarray([[1.0], [3.0], [9.0], [2.0], [8.0]])
    ids = ["fe-b", "fe-a", "fe-c", "feo-b", "feo-a"]
    strata = ["cold", "hot", "cold", "hot", "cold"]
    first = hierarchical_farthest_point_sampling(
        structures,
        descriptors,
        ids,
        strata,
        budget=4,
    )
    order = [4, 2, 0, 3, 1]
    second = hierarchical_farthest_point_sampling(
        [structures[index] for index in order],
        descriptors[order],
        [ids[index] for index in order],
        [strata[index] for index in order],
        budget=4,
    )

    assert first.selected_ids == second.selected_ids
    assert first.selected_novelty == pytest.approx(second.selected_novelty)
    assert {
        ids[index] for index in first.selected_indices
    } == set(first.selected_ids)
    reordered_ids = [ids[index] for index in order]
    assert {
        reordered_ids[index] for index in second.selected_indices
    } == set(second.selected_ids)
    assert first.groups == second.groups


def test_remaining_novelty_and_group_report_include_unselected_candidates():
    result = hierarchical_farthest_point_sampling(
        [_atoms("Fe")] * 3,
        np.asarray([[0.0], [2.0], [10.0]]),
        ["a", "b", "c"],
        ["cold", "hot", "hot"],
        budget=1,
        reference_structures=[_atoms("Fe")],
        reference_descriptors=np.asarray([[0.0]]),
    )

    report = result.groups[("Fe",)]
    assert report.selected_ids == ("c",)
    assert report.counts_by_stratum == {"hot": 1}
    assert report.remaining_novelty > 0.0
    assert result.remaining_novelty == report.remaining_novelty
