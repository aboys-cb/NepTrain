from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from NepTrain.core.dft import LabelRequest, LabelingError, label
from NepTrain.core.scientific_data import (
    labeled_input_structure_ids,
    structure_id,
)
from NepTrain.core.select import farthest_point_sampling
from NepTrain.core.smoke import SmokeError, run_smoke
from NepTrain.core.spin import validate_spin_dataset


def test_toy_label_adapter_preserves_spin_and_produces_all_labels(tmp_path: Path):
    atoms = Atoms(
        "Fe2",
        positions=[[1.0, 1.0, 1.0], [3.45, 1.0, 1.0]],
        cell=[7.0, 7.0, 7.0],
        pbc=True,
    )
    atoms.set_array("spin", np.asarray([[1.7, 0.0, 0.0], [0.0, 1.6, 0.2]]))
    source = tmp_path / "input.xyz"
    output = tmp_path / "labeled.xyz"
    write(source, atoms, format="extxyz")

    result = label(
        LabelRequest(source, output, tmp_path / "work", options={"profile": "spin"}),
        "toy",
    )

    assert len(result.frames) == 1
    restored = read(output, format="extxyz")
    validate_spin_dataset([restored], require_mforce=True)
    assert restored.calc.results["forces"].shape == (2, 3)
    assert np.asarray(restored.info["virial"]).shape == (3, 3)
    assert np.isfinite(restored.get_potential_energy())


def test_label_publication_is_atomic_when_an_adapter_returns_invalid_data(
    tmp_path: Path, monkeypatch
):
    atoms = Atoms(
        "Fe",
        positions=[[0.0, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    source = tmp_path / "input.xyz"
    output = tmp_path / "labeled.xyz"
    write(source, atoms, format="extxyz")
    output.write_bytes(b"previous validated publication\n")
    before = output.read_bytes()

    def invalid_adapter(request):
        invalid = read(request.source)
        write(request.output_file, invalid, format="extxyz")
        return [invalid]

    monkeypatch.setattr(
        "NepTrain.core.dft.toy.run_toy_teacher",
        invalid_adapter,
    )
    with pytest.raises(LabelingError, match="scientific data contract"):
        label(LabelRequest(source, output, tmp_path / "work"), "toy")

    assert output.read_bytes() == before


def test_label_append_preserves_input_order_and_rejects_duplicate_ownership(
    tmp_path: Path,
):
    first = Atoms(
        "Fe2",
        positions=[[0.0, 0.0, 0.0], [2.45, 0.0, 0.0]],
        cell=[7.0, 7.0, 7.0],
        pbc=True,
    )
    second = first.copy()
    second.positions[1, 1] = 0.2
    first_source = tmp_path / "first.xyz"
    second_source = tmp_path / "second.xyz"
    output = tmp_path / "labeled.xyz"
    write(first_source, first, format="extxyz")
    write(second_source, second, format="extxyz")

    label(LabelRequest(first_source, output, tmp_path / "first-work"), "toy")
    label(
        LabelRequest(
            second_source,
            output,
            tmp_path / "second-work",
            append=True,
        ),
        "toy",
    )

    restored = read(output, index=":", format="extxyz")
    assert labeled_input_structure_ids(restored) == [
        structure_id(first),
        structure_id(second),
    ]
    before = output.read_bytes()
    with pytest.raises(LabelingError, match="already present"):
        label(
            LabelRequest(
                second_source,
                output,
                tmp_path / "retry-work",
                append=True,
            ),
            "toy",
        )
    assert output.read_bytes() == before


def test_fps_is_deterministic_without_a_reference_and_never_repeats():
    points = np.asarray([[0.0], [1.0], [1.0], [2.0]])
    first = farthest_point_sampling(points, 10, min_dist=0.0)
    second = farthest_point_sampling(points, 10, min_dist=0.0)
    assert first == second
    assert len(first) == len(set(first))
    assert len(first) == 3


def test_all_toy_smoke_profiles_pass(tmp_path: Path):
    ordinary = run_smoke(tmp_path / "ordinary", profile="ordinary")
    spin = run_smoke(tmp_path / "spin", profile="spin")
    recovery = run_smoke(tmp_path / "recovery", profile="recovery")

    assert ordinary.passed
    assert spin.passed
    assert recovery.passed
    assert spin.derivative_mforce_max_error is not None
    assert recovery.recovery_match is True


def test_smoke_refuses_to_delete_the_working_directory():
    with pytest.raises(SmokeError, match="unsafe smoke output"):
        run_smoke(Path.cwd(), force=True)
