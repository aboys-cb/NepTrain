"""VASP 6 DeltaSpin vector constraints and active-site effective labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from ...spin import SpinDataError, spin_constraint_mask


class DeltaSpinError(ValueError):
    pass


def numbers(value: str) -> np.ndarray:
    """Read INCAR numeric lists, including VASP's count*value notation."""
    result = []
    for token in value.split():
        parts = token.split("*")
        if len(parts) == 1:
            count, number = 1, parts[0]
        elif len(parts) == 2:
            count, number = int(parts[0]), parts[1]
            if count <= 0:
                raise DeltaSpinError("INCAR repetition counts must be positive")
        else:
            raise DeltaSpinError(f"invalid INCAR number: {token}")
        result.extend([float(number.replace("D", "E").replace("d", "e"))] * count)
    array = np.asarray(result, dtype=float)
    if not np.isfinite(array).all():
        raise DeltaSpinError("DeltaSpin input contains non-finite values")
    return array


def custom_parameters(calculator) -> dict[str, str]:
    return dict(getattr(calculator, "input_params", {}).get("custom") or {})


def enabled(calculator) -> bool:
    value = str(custom_parameters(calculator).get("ldeltaspin", ".FALSE.")).lower()
    if value not in {"t", "true", ".true.", "f", "false", ".false."}:
        raise DeltaSpinError("LDELTASPIN must be .TRUE. or .FALSE.")
    return value in {"t", "true", ".true."}


def validate_input(calculator) -> None:
    params = custom_parameters(calculator)
    if not calculator.bool_params.get("lnoncollinear"):
        raise DeltaSpinError("VASP DeltaSpin requires LNONCOLLINEAR=.TRUE.")
    if int(params.get("deltaspin_constraint_mode", "0")) != 0:
        raise DeltaSpinError("spin/mforce labeling requires DELTASPIN_CONSTRAINT_MODE=0 (full vector)")
    if int(params.get("deltaspin_moment_def", "0")) not in {0, 1, 2}:
        raise DeltaSpinError("DELTASPIN_MOMENT_DEF must be 0, 1, or 2")
    if "deltaspin_components" in params and not np.array_equal(
        numbers(params["deltaspin_components"]), [1, 1, 1]
    ):
        raise DeltaSpinError("spin/mforce labeling requires DELTASPIN_COMPONENTS=1 1 1")
    if "deltaspin_atoms" in params:
        mask = numbers(params["deltaspin_atoms"])
        if not np.isin(mask, [0, 1]).all() or not mask.any():
            raise DeltaSpinError("DELTASPIN_ATOMS must be a nonempty binary active-atom mask")
        if not mask.all() and int(params.get("deltaspin_moment_def", "0")) != 0:
            raise DeltaSpinError("partial atom constraints currently support moment definition 0 only")
    saxis = calculator.list_float_params.get("saxis")
    if saxis is not None and not np.array_equal(saxis, [0, 0, 1]):
        raise DeltaSpinError("VASP DeltaSpin labeling currently requires SAXIS=0 0 1")
    if "deltaspin_tol" in params:
        tolerance = numbers(params["deltaspin_tol"])
        if tolerance.shape != (1,) or tolerance[0] <= 0:
            raise DeltaSpinError("DELTASPIN_TOL must be positive")


def prepare_input(calculator, atoms):
    """Group atoms once so targets and ASE's POSCAR share exactly one order."""
    from .io import prepare_vector_moments

    params = custom_parameters(calculator)
    if "spin_active" in atoms.arrays or "deltaspin_atoms" not in params:
        try:
            mask = spin_constraint_mask(atoms)
        except SpinDataError as error:
            raise DeltaSpinError(str(error)) from error
        # A per-frame mask owns this field, just as spin owns M_DELTASPIN.
        params["deltaspin_atoms"] = " ".join(str(int(v)) for v in mask)
        calculator.set(custom=params)
    if "deltaspin_atoms" in params and len(numbers(params["deltaspin_atoms"])) != len(atoms):
        raise DeltaSpinError("DELTASPIN_ATOMS must have one entry per input atom; omit it for automatic generation")
    validate_input(calculator)
    mask = numbers(params["deltaspin_atoms"]).astype(int)
    ordered, resort = prepare_vector_moments(calculator, atoms, atoms.arrays["spin"])
    mask = mask[np.argsort(resort)]
    params = custom_parameters(calculator)
    params.update(
        ldeltaspin=".TRUE.",
        deltaspin_atoms=" ".join(str(v) for v in mask),
        deltaspin_components="1 1 1",
        m_deltaspin=params["magmom"],
    )
    calculator.set(custom=params)
    ordered.arrays.pop("mforce", None)
    ordered.arrays.pop("spin_projected", None)
    return ordered, resort


@dataclass(frozen=True)
class DeltaSpinResult:
    spin: np.ndarray
    mforce: np.ndarray
    moment_def: int
    max_error: float
    tolerance: float
    active: np.ndarray
    projected: np.ndarray


def read_result(path: Path, symbols: list[str], targets: np.ndarray, *, active=None) -> DeltaSpinResult:
    """Accept one fully converged single point; never fill missing labels with zero."""
    active = np.ones(len(symbols), dtype=int) if active is None else np.asarray(active)
    if active.shape != (len(symbols),) or not np.isin(active, [0, 1]).all() or not active.any():
        raise DeltaSpinError("invalid result constraint mask")
    active = active.astype(bool)
    targets = np.asarray(targets)
    if targets.shape != (len(symbols), 3) or not np.isfinite(targets).all():
        raise DeltaSpinError("invalid spin target shape or non-finite targets")
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.count("DeltaSpin final status") != 1 or text.count("DeltaSpin final constrained observables") != 1:
        raise DeltaSpinError("OUTCAR must contain exactly one DeltaSpin final status and observables block")
    block = text.split("DeltaSpin final status", 1)[1]
    status, table_text = block.split("DeltaSpin final constrained observables", 1)
    for name in ("SCF convergence", "Moment constraint"):
        if not re.search(rf"^\s*{name}\s*:\s*reached\s*$", status, re.MULTILINE):
            raise DeltaSpinError(f"DeltaSpin {name} was not reached")

    def scalar(label):
        match = re.search(rf"{label}\s*:\s*(\S+)", status)
        if not match:
            raise DeltaSpinError(f"DeltaSpin status is missing {label}")
        return float(numbers(match.group(1))[0])

    max_error = scalar("Maximum moment error")
    tolerance = scalar("Requested tolerance")
    if tolerance <= 0 or max_error < 0 or max_error > tolerance:
        raise DeltaSpinError("DeltaSpin maximum moment error exceeds the requested tolerance")
    definition = re.search(r"DELTASPIN_MOMENT_DEF\s*=\s*([012])\b", table_text)
    if not definition:
        raise DeltaSpinError("DeltaSpin output is missing a supported DELTASPIN_MOMENT_DEF")
    if "magnetic force = spin_lambda for L=E+lambda.(M-M_target)" not in table_text:
        raise DeltaSpinError("DeltaSpin output is missing the magnetic-force sign convention")

    def table(header, indices):
        lines = table_text.splitlines()
        matches = [i for i, line in enumerate(lines) if line.split() == header.split()]
        if len(matches) != 1:
            raise DeltaSpinError(f"missing or ambiguous DeltaSpin table: {header}")
        start = matches[0] + 1
        if start >= len(lines) or set(lines[start].strip()) != {"-"}:
            raise DeltaSpinError("malformed DeltaSpin table separator")
        rows = []
        for line in lines[start + 1:]:
            fields = line.split()
            if not fields or not fields[0].isdigit():
                break
            if len(rows) >= len(indices):
                raise DeltaSpinError("extra DeltaSpin rows")
            index = indices[len(rows)]
            flags = [str(int(active[index]))] * 3
            if (len(fields) != len(header.split()) or int(fields[0]) != index + 1
                    or fields[1] != symbols[index] or fields[2:5] != flags):
                raise DeltaSpinError("DeltaSpin table has incomplete constraints or wrong atom order")
            rows.append(numbers(" ".join(fields[5:8])))
        if len(rows) != len(indices):
            raise DeltaSpinError("DeltaSpin table is missing required labels")
        return np.asarray(rows)

    spin = np.zeros((len(symbols), 3))
    spin[active] = table("ion elem cx cy cz Mx My Mz |M|", np.flatnonzero(active))
    mforce = table("ion elem cx cy cz MFx MFy MFz", np.arange(len(symbols)))
    if np.any(mforce[~active] != 0):
        raise DeltaSpinError("inactive constraint multipliers must be zero")
    if np.max(np.abs(spin[active] - targets[active])) > tolerance * (1 + 1e-4) + 1e-10:
        raise DeltaSpinError("DeltaSpin final moments do not match active input targets")
    projected = spin.copy()
    if not active.all():
        if int(definition.group(1)) != 0:
            raise DeltaSpinError("partial-mask diagnostics require projector moment definition 0")
        for k, axis in enumerate("xyz"):
            marker = f"magnetization ({axis})"
            if marker not in text:
                raise DeltaSpinError("missing final induced-moment diagnostic")
            rows = []
            for line in text.rsplit(marker, 1)[1].splitlines():
                fields = line.split()
                if len(fields) == 5 and fields[0].isdigit():
                    if int(fields[0]) != len(rows) + 1:
                        raise DeltaSpinError("induced-moment atom order mismatch")
                    rows.append(numbers(fields[-1])[0])
                elif rows:
                    break
            if len(rows) != len(symbols):
                raise DeltaSpinError("incomplete induced-moment diagnostic")
            projected[~active, k] = np.asarray(rows)[~active]
    return DeltaSpinResult(spin, mforce, int(definition.group(1)), max_error, tolerance, active, projected)
