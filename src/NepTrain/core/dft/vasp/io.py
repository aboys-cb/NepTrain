"""Project-specific ASE VASP input setup."""

from __future__ import annotations

import os

from ase.calculators.vasp import Vasp

class VaspInput(Vasp):
    """ASE VASP calculator configured with NepTrain POTCAR conventions."""

    def __init__(self, *args, **kwargs):
        pp_path = kwargs.pop("pp_path", None)
        if not pp_path:
            raise ValueError("VASP pseudopotential root is required")
        super().__init__(*args, **kwargs)
        self.input_params["setups"] = {"base": "recommended"}
        self.input_params["pp"] = ""
        os.environ[self.VASP_PP_PATH] = os.path.expanduser(str(pp_path))


__all__ = ["VaspInput"]
