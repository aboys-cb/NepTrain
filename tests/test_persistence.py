import json
import errno
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


def test_atomic_write_json_retries_transient_filesystem_write_errors(
    tmp_path: Path, monkeypatch,
):
    destination = tmp_path / "controller.json"
    original = Path.write_text
    attempts = 0

    def flaky_write(path, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError(errno.EFAULT, "Bad address")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", flaky_write)

    atomic_write_json(destination, {"state": "running"})

    assert attempts == 3
    assert json.loads(destination.read_text()) == {"state": "running"}


def test_atomic_write_json_does_not_retry_permanent_write_errors(
    tmp_path: Path, monkeypatch,
):
    attempts = 0

    def full_filesystem(_path, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Path, "write_text", full_filesystem)

    with pytest.raises(OSError, match="No space left"):
        atomic_write_json(tmp_path / "controller.json", {"state": "running"})

    assert attempts == 1
