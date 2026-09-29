"""Project-specific ASE VASP input setup."""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
import threading
import warnings

from ase.config import ASEEnvDeprecationWarning
from ase.calculators.vasp import Vasp


_PP_PATH_LOCK = threading.RLock()


class VaspInput(Vasp):
    """ASE VASP calculator configured with NepTrain POTCAR conventions."""

    def __init__(self, *args, **kwargs):
        pp_path = kwargs.pop("pp_path", None)
        if not pp_path:
            raise ValueError("VASP pseudopotential root is required")
        super().__init__(*args, **kwargs)
        self._neptrain_pp_path = os.path.expanduser(str(pp_path))
        self.input_params["setups"] = {"base": "recommended"}
        self.input_params["pp"] = ""

    def _build_pp_list(self, atoms, setups=None, special_setups=()):
        """Let ASE resolve POTCARs against this calculator's pinned root."""

        with _PP_PATH_LOCK:
            previous = os.environ.get(self.VASP_PP_PATH)
            os.environ[self.VASP_PP_PATH] = self._neptrain_pp_path
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ASEEnvDeprecationWarning)
                    return super()._build_pp_list(
                        atoms,
                        setups=setups,
                        special_setups=special_setups,
                    )
            finally:
                if previous is None:
                    os.environ.pop(self.VASP_PP_PATH, None)
                else:
                    os.environ[self.VASP_PP_PATH] = previous


__all__ = ["VaspInput"]


def read_vasp_input(calculator, filename):
    """Preserve complete DeltaSpin values through ASE's custom-key interface."""
    text = Path(filename).read_text(encoding="utf-8")
    ordinary = []
    custom = {}
    for line in text.replace("\\\n", " ").splitlines():
        for assignment in re.split(r"[#!]", line, maxsplit=1)[0].split(";"):
            key, separator, value = assignment.partition("=")
            key = key.strip().lower()
            if not separator:
                if assignment.strip():
                    ordinary.append(assignment)
                continue
            if key in {"ldeltaspin", "m_deltaspin", "ediff_rho"} or key.startswith("deltaspin_"):
                custom[key] = value.strip()
            else:
                ordinary.append(assignment)
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".INCAR", encoding="utf-8") as handle:
        handle.write("\n".join(ordinary) + "\n")
        handle.flush()
        calculator.read_incar(handle.name)
    if custom:
        calculator.set(custom=custom)
