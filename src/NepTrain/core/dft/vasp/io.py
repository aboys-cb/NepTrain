"""Project-specific ASE VASP input setup."""

from __future__ import annotations

import os

from ase.calculators.vasp import Vasp

from NepTrain import Config


class VaspInput(Vasp):
    """ASE VASP calculator configured with NepTrain POTCAR conventions."""

    def __init__(self, *args, **kwargs):
        pp_path = kwargs.pop("pp_path", None)
        super().__init__(*args, **kwargs)
        setups = {"base": "recommended"}
        for option in Config.options("potcar"):
            element = option.capitalize()
            configured = Config.get("potcar", option).strip()
            if configured.startswith(element):
                setups[element] = configured[len(element) :]
        self.input_params["setups"] = setups
        self.input_params["pp"] = ""
        configured = pp_path or Config.get("environ", "potcar_path")
        os.environ[self.VASP_PP_PATH] = os.path.expanduser(str(configured))


__all__ = ["VaspInput"]
