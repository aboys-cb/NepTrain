import hashlib
import json
from pathlib import Path

import pytest

from NepTrain.core.content_addressing import canonical_sha256, file_sha256


def test_file_sha256_matches_standard_digest_across_read_blocks(
    tmp_path: Path,
):
    payload = (b"neptrain-content-addressing\n" * 50_000) + b"tail"
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(payload)

    assert file_sha256(artifact) == hashlib.sha256(payload).hexdigest()


def test_canonical_sha256_is_independent_of_mapping_order():
    left = {"route": "npt", "conditions": {"pressure": 0, "temperature": 500}}
    right = {"conditions": {"temperature": 500, "pressure": 0}, "route": "npt"}
    expected = hashlib.sha256(
        json.dumps(
            left,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()

    assert canonical_sha256(left) == expected
    assert canonical_sha256(right) == expected


def test_canonical_sha256_rejects_non_finite_values():
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_sha256({"loss": float("nan")})
