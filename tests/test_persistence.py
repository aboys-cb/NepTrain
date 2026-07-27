import json
from pathlib import Path

import pytest

from NepTrain.core.persistence import atomic_write_json


def test_atomic_write_json_creates_parent_and_replaces_existing_file(
    tmp_path: Path,
):
    destination = tmp_path / "state" / "controller.json"
    destination.parent.mkdir()
    destination.write_text('{"stale": true}\n', encoding="utf-8")

    result = atomic_write_json(destination, {"state": "running", "step": 2})

    assert result == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "state": "running",
        "step": 2,
    }
    assert destination.read_text(encoding="utf-8").endswith("\n")
    assert not destination.with_suffix(".json.tmp").exists()


def test_atomic_write_json_rejects_nan_without_touching_existing_file(
    tmp_path: Path,
):
    destination = tmp_path / "state.json"
    destination.write_text('{"state": "safe"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Out of range float values"):
        atomic_write_json(destination, {"loss": float("nan")})

    assert destination.read_text(encoding="utf-8") == '{"state": "safe"}\n'
