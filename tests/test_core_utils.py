from pathlib import Path

import pytest

from NepTrain.core import utils as core_utils


def test_environment_check_only_requires_potcar_for_vasp(tmp_path: Path):
    core_utils.check_env(commands=())

    missing = tmp_path / "missing-potcar"
    with pytest.raises(FileNotFoundError, match="pseudopotential root"):
        core_utils.check_env(
            potcar_path=missing,
            require_potcar=True,
            commands=(),
        )
