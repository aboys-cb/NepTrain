import numpy as np
import pytest
from ase import Atoms

from NepTrain.core.descriptor_features import reduce_atomic_descriptors


def test_elementwise_reduction_keeps_species_channels_separate():
    structures = [
        Atoms("FeO", positions=[[0, 0, 0], [1, 0, 0]]),
        Atoms("FeO", positions=[[0, 0, 0], [1, 0, 0]]),
    ]
    atomic = np.asarray([[2.0], [-2.0], [3.0], [-3.0]])

    global_mean = reduce_atomic_descriptors(
        structures, atomic, reduction="global_mean"
    )
    elementwise = reduce_atomic_descriptors(
        structures,
        atomic,
        reduction="elementwise_mean_std",
        elements=("Fe", "O"),
    )

    np.testing.assert_allclose(global_mean, [[0.0], [0.0]])
    np.testing.assert_allclose(
        elementwise,
        [[2.0, 0.0, -2.0, 0.0], [3.0, 0.0, -3.0, 0.0]],
    )


def test_elementwise_reduction_uses_stable_zero_channels_for_missing_elements():
    structures = [
        Atoms("Fe2", positions=[[0, 0, 0], [1, 0, 0]]),
        Atoms("O", positions=[[0, 0, 0]]),
    ]
    atomic = np.asarray([[1.0], [3.0], [7.0]])

    reduced = reduce_atomic_descriptors(
        structures,
        atomic,
        reduction="elementwise_mean_std",
        elements=("Fe", "O"),
    )

    np.testing.assert_allclose(reduced, [[2.0, 1.0, 0.0, 0.0], [0.0, 0.0, 7.0, 0.0]])


def test_atomic_descriptor_row_count_must_match_atoms():
    with pytest.raises(ValueError, match="one row per atom"):
        reduce_atomic_descriptors(
            [Atoms("Fe2")],
            np.asarray([[1.0]]),
            reduction="elementwise_mean_std",
        )
