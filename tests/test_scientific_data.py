import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from NepTrain.core.scientific_data import (
    ScientificDataError,
    structure_id,
    validate_labeled_frames,
)


def _frame(*, pbc: bool = True) -> Atoms:
    atoms = Atoms(
        "Fe2",
        positions=[[0, 0, 0], [2.2, 0, 0]],
        cell=[6, 6, 6],
        pbc=pbc,
    )
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-1.0,
        forces=np.zeros((2, 3)),
    )
    atoms.info["virial"] = np.zeros((3, 3))
    return atoms


def test_structure_identity_includes_periodic_boundary_conditions():
    periodic = _frame(pbc=True)
    isolated = _frame(pbc=False)

    assert structure_id(periodic) != structure_id(isolated)


def test_structure_identity_includes_nonzero_collinear_initial_moments():
    up = _frame()
    down = _frame()
    ordinary = _frame()
    up.set_initial_magnetic_moments([2.0, 2.0])
    down.set_initial_magnetic_moments([-2.0, -2.0])

    assert structure_id(up) != structure_id(down)
    assert structure_id(up) != structure_id(ordinary)


def test_label_validation_rejects_missing_or_nonfinite_scientific_fields():
    valid = _frame()
    assert validate_labeled_frames([valid]) == [structure_id(valid)]

    missing_virial = _frame()
    missing_virial.info.pop("virial")
    with pytest.raises(ScientificDataError, match="virial is missing"):
        validate_labeled_frames([missing_virial])

    bad_force = _frame()
    bad_force.calc.results["forces"][0, 0] = np.nan
    with pytest.raises(ScientificDataError, match="forces contain non-finite"):
        validate_labeled_frames([bad_force])


def test_gpumd_force_array_is_supported_but_conflicts_fail_closed():
    frame = _frame()
    forces = frame.calc.results.pop("forces")
    frame.set_array("force", forces)
    assert validate_labeled_frames([frame]) == [structure_id(frame)]

    frame.calc.results["forces"] = np.ones((2, 3))
    with pytest.raises(ScientificDataError, match="conflicting force labels"):
        validate_labeled_frames([frame])
