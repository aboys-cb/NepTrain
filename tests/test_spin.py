from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from NepTrain.core.spin import (
    SpinDataError,
    canonicalize_spin_arrays,
    collect_mforce_from_results,
    prepare_spin_for_dft,
    spin_from_lammps,
    spin_to_lammps,
    validate_spin_dataset,
)


def test_extxyz_roundtrip_uses_singular_spin_and_mforce(tmp_path: Path):
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [1, 1, 1]], cell=[3, 3, 3], pbc=True)
    spins = np.asarray([[1.0, 2.0, 3.0], [-2.0, 0.5, 0.25]])
    mforces = np.asarray([[0.1, 0.2, 0.3], [-0.2, 0.0, 0.1]])
    atoms.set_array("spin", spins)
    atoms.set_array("mforce", mforces)
    path = tmp_path / "spin.xyz"
    write(path, atoms, format="extxyz")

    header = path.read_text(encoding="utf-8").splitlines()[1]
    assert "spin:R:3" in header
    assert "mforce:R:3" in header
    restored = read(path, format="extxyz")
    validate_spin_dataset([restored], require_mforce=True)
    np.testing.assert_allclose(restored.arrays["spin"], spins)
    np.testing.assert_allclose(restored.arrays["mforce"], mforces)


def test_spin_lammps_roundtrip_preserves_evolving_magnitude():
    spins = np.asarray([[1.0, 2.0, 2.0], [-2.0, 0.0, 0.0]])
    converted = spin_to_lammps(spins)
    np.testing.assert_allclose(converted.magnitude, [3.0, 2.0])
    np.testing.assert_allclose(
        spin_from_lammps(converted.direction, converted.magnitude), spins
    )


def test_spin_labels_are_mandatory_for_training():
    atoms = Atoms("Fe", positions=[[0, 0, 0]])
    atoms.set_array("spin", np.asarray([[1.0, 0.0, 0.0]]))
    with pytest.raises(SpinDataError, match="mandatory mforce"):
        validate_spin_dataset([atoms], require_mforce=True)


@pytest.mark.parametrize(
    "alias",
    [
        "mforces",
        "force_mag",
        "forces_mag",
        "magnetic_force",
        "magnetic_forces",
    ],
)
def test_mforce_aliases_require_explicit_migration(alias: str):
    atoms = Atoms("Fe", positions=[[0, 0, 0]])
    spin = np.asarray([[1.0, 0.0, 0.0]])
    mforce = np.asarray([[0.1, 0.2, 0.3]])
    atoms.set_array("spin", spin)
    atoms.set_array(alias, mforce)

    with pytest.raises(SpinDataError, match="require explicit migration"):
        validate_spin_dataset([atoms], require_mforce=True)

    canonicalize_spin_arrays(atoms)
    validate_spin_dataset([atoms], require_mforce=True)
    assert alias not in atoms.arrays
    np.testing.assert_allclose(atoms.arrays["mforce"], mforce)


def test_conflicting_mforce_alias_is_rejected():
    atoms = Atoms("Fe", positions=[[0, 0, 0]])
    atoms.set_array("spin", np.asarray([[1.0, 0.0, 0.0]]))
    atoms.set_array("mforce", np.asarray([[0.1, 0.2, 0.3]]))
    atoms.set_array("force_mag", np.asarray([[0.3, 0.2, 0.1]]))

    with pytest.raises(
        SpinDataError, match="ambiguous mforce: mforce and force_mag differ"
    ):
        canonicalize_spin_arrays(atoms)


def test_spin_is_forwarded_to_dft_and_mforce_is_collected():
    atoms = Atoms("Fe", positions=[[0, 0, 0]])
    spin = np.asarray([[1.0, 2.0, 3.0]])
    mforce = np.asarray([[0.1, 0.2, 0.3]])
    atoms.set_array("spin", spin)
    atoms.set_array("mforce", np.asarray([[9.0, 9.0, 9.0]]))
    assert prepare_spin_for_dft(atoms)
    assert "mforce" not in atoms.arrays
    np.testing.assert_allclose(atoms.get_initial_magnetic_moments(), spin)
    collect_mforce_from_results(atoms, {"mforces": mforce})
    np.testing.assert_allclose(atoms.arrays["mforce"], mforce)
