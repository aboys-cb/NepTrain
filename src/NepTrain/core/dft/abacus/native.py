"""Native ABACUS case rendering, execution, and result collection."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixCartesian
from ase.data import atomic_masses, atomic_numbers
from ase.units import Bohr

from ...content_addressing import file_sha256
from ...spin import prepare_spin_for_dft
from ..attempts import new_attempt_directory
from .resources import validate_abacus_resources


PARSER_VERSION = "neptrain-abacus-running-scf-v2"
KBAR_ANGSTROM3_TO_EV = 0.0006241509125883258
_FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")


class NativeAbacusError(RuntimeError):
    """Raised when a native ABACUS case cannot produce a valid label."""


@dataclass(frozen=True)
class NativeAbacusRequest:
    work_dir: Path
    resource_dir: Path
    resource_manifest: Path
    command: str
    input_parameters: Mapping[str, str]
    use_gamma: bool
    kpoint_mode: str
    kspacing: float | None
    ka: tuple[int, int, int]
    flat_single_case: bool = False


@dataclass(frozen=True)
class ParsedAbacusResult:
    energy: float
    forces: np.ndarray
    stress_kbar: np.ndarray
    magnetization: np.ndarray | None
    mforce: np.ndarray | None
    scf_iterations: int | None


def run_native_abacus(
    atoms: Atoms, request: NativeAbacusRequest, *, case_index: int
) -> Atoms:
    """Run one fresh ABACUS attempt and return a normalized labeled frame."""

    input_frame = atoms.copy()
    spin_frame = prepare_spin_for_dft(input_frame)
    parameters = dict(request.input_parameters)
    electronic_mode = validate_abacus_spin_contract(
        parameters,
        spin_frame=spin_frame,
    )
    basis_type = str(parameters.get("basis_type", "pw")).strip().lower()
    resource_provenance, pp_files, orb_files = validate_abacus_resources(
        request.resource_dir,
        request.resource_manifest,
        input_frame,
        require_orbitals=basis_type == "lcao",
    )
    case_dir = new_attempt_directory(
        request.work_dir,
        case_index,
        input_frame.get_chemical_formula(),
        flat_single_case=request.flat_single_case,
    )
    ordered_indices = _render_case(
        case_dir,
        input_frame,
        parameters,
        resource_dir=request.resource_dir,
        pp_files=pp_files,
        orb_files=orb_files,
        use_gamma=request.use_gamma,
        kpoint_mode=request.kpoint_mode,
        kspacing=request.kspacing,
        ka=request.ka,
        resource_provenance=resource_provenance,
    )
    completed = subprocess.run(
        request.command,
        shell=True,
        cwd=case_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_path = case_dir / "abacus.stdout"
    stderr_path = case_dir / "abacus.stderr"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        _write_result_manifest(
            case_dir,
            command=request.command,
            returncode=completed.returncode,
            status="process_failed",
            electronic_mode=electronic_mode,
        )
        detail = _tail(completed.stderr or completed.stdout)
        raise NativeAbacusError(
            f"ABACUS failed with exit code {completed.returncode} in {case_dir}: "
            f"{detail}"
        )

    log_path: Path | None = None
    try:
        log_path = _running_scf_log(case_dir, parameters)
        parsed = parse_running_scf(log_path, expected_atoms=len(input_frame))
        if spin_frame and parsed.magnetization is None:
            raise NativeAbacusError(
                f"ABACUS spin result is missing Total Magnetism: {log_path}"
            )
        if spin_frame and parsed.mforce is None:
            raise NativeAbacusError(
                f"ABACUS spin result is missing mandatory magnetic force: {log_path}"
            )
    except NativeAbacusError:
        _write_result_manifest(
            case_dir,
            command=request.command,
            returncode=completed.returncode,
            status="result_invalid",
            log_path=log_path,
            electronic_mode=electronic_mode,
        )
        raise
    forces = _restore_atom_order(parsed.forces, ordered_indices)
    frame = input_frame.copy()
    frame.arrays.pop("initial_magmoms", None)
    frame.calc = SinglePointCalculator(
        frame,
        energy=parsed.energy,
        forces=forces,
    )
    frame.info["virial"] = (
        parsed.stress_kbar * frame.get_volume() * KBAR_ANGSTROM3_TO_EV
    )
    frame.info.setdefault("Config_type", "NepTrain scf ")
    frame.info["Weight"] = 1.0
    frame.info["dft_electronic_mode"] = electronic_mode
    if spin_frame:
        magnetization = _restore_atom_order(
            parsed.magnetization, ordered_indices
        )
        mforce = _restore_atom_order(parsed.mforce, ordered_indices)
        target_spin = np.asarray(input_frame.arrays["spin"], dtype=np.float64)
        frame.set_array("spin", magnetization)
        frame.set_array("mforce", mforce)
        frame.info["spin_constraint_rms_uB"] = float(
            np.sqrt(
                np.sum(np.square(magnetization - target_spin))
                / len(target_spin)
            )
        )
    _write_result_manifest(
        case_dir,
        command=request.command,
        returncode=completed.returncode,
        status="completed",
        log_path=log_path,
        parsed=parsed,
        electronic_mode=electronic_mode,
    )
    return frame


def parse_running_scf(path: Path, *, expected_atoms: int) -> ParsedAbacusResult:
    """Parse the final SCF label from an ABACUS running log."""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    energy = _last_energy(lines)
    converged = any(
        "SCF IS CONVERGED" in line
        or "charge density convergence is achieved" in line
        for line in lines
    )
    iterations = _last_scf_iteration(lines)
    if energy is None:
        raise NativeAbacusError(f"ABACUS total energy is missing: {path}")
    if not converged:
        suffix = f" after {iterations} iterations" if iterations is not None else ""
        raise NativeAbacusError(f"ABACUS SCF did not converge{suffix}: {path}")
    forces = _last_vector_table(lines, "TOTAL-FORCE", expected_atoms)
    if forces is None:
        raise NativeAbacusError(f"ABACUS force table is missing or incomplete: {path}")
    stress = _last_matrix_table(lines, "TOTAL-STRESS", rows=3)
    if stress is None:
        raise NativeAbacusError(f"ABACUS stress table is missing or incomplete: {path}")
    magnetization = _last_vector_table(lines, "Total Magnetism (uB)", expected_atoms)
    mforce = _last_magnetic_force_table(lines, expected_atoms)
    return ParsedAbacusResult(
        energy=energy,
        forces=forces,
        stress_kbar=stress,
        magnetization=magnetization,
        mforce=mforce,
        scf_iterations=iterations,
    )


def _configure_spin_parameters(parameters: dict[str, str]) -> None:
    """Select ABACUS full-vector DeltaSpin for canonical spin:R:3 input."""

    direction_only = str(parameters.pop("sc_direction_only", "0")).strip().lower()
    if direction_only not in {"", "0", "false", "f", "no", "off"}:
        raise NativeAbacusError(
            "ABACUS direction-only spin constraints are incompatible with "
            "variable-magnitude spin labels"
        )
    parameters.update(
        {
            "nspin": "4",
            "noncolin": "1",
            "sc_mag_switch": "1",
            "symmetry": "0",
        }
    )


def _parameter_enabled(value: object) -> bool:
    return str(value or "0").strip().lower() not in {
        "",
        "0",
        "false",
        "f",
        "no",
        "off",
    }


def validate_abacus_spin_contract(
    parameters: dict[str, str],
    *,
    spin_frame: bool,
) -> str:
    """Separate electronic spin polarization from spin-force labeling."""

    if spin_frame:
        _configure_spin_parameters(parameters)
        return "constrained_vector_spin_force"
    if _parameter_enabled(parameters.get("sc_mag_switch")):
        raise NativeAbacusError(
            "ABACUS sc_mag_switch requires canonical spin:R:3 input; "
            "otherwise magnetic forces would be discarded"
        )
    nspin = str(parameters.get("nspin", "1")).strip()
    noncollinear = _parameter_enabled(parameters.get("noncolin"))
    if noncollinear or nspin == "4":
        return "noncollinear_spin_polarized"
    if nspin == "2":
        return "collinear_spin_polarized"
    return "non_spin_polarized"


def _render_case(
    case_dir: Path,
    atoms: Atoms,
    parameters: dict[str, str],
    *,
    resource_dir: Path,
    pp_files: Mapping[str, str],
    orb_files: Mapping[str, str],
    use_gamma: bool,
    kpoint_mode: str,
    kspacing: float | None,
    ka: tuple[int, int, int],
    resource_provenance: Mapping[str, object],
) -> tuple[int, ...]:
    if str(parameters.get("smearing_method", "")).strip().lower() == "gau":
        parameters["smearing_method"] = "gaussian"
    parameters["pseudo_dir"] = str(resource_dir)
    if str(parameters.get("basis_type", "pw")).strip().lower() == "lcao":
        parameters["orbital_dir"] = str(resource_dir)
    parameters["cal_force"] = "1"
    parameters["cal_stress"] = "1"
    template_kspacing = parameters.get("kspacing")
    active_kspacing = kspacing if kspacing is not None else template_kspacing
    if kpoint_mode == "kpoints":
        active_kspacing = None
    elif kpoint_mode == "kspacing" and active_kspacing is None:
        raise NativeAbacusError(
            "ABACUS kpoint_mode=kspacing requires kspacing in the request or INPUT"
        )
    if active_kspacing is not None:
        parameters["kspacing"] = str(active_kspacing)
    else:
        parameters.pop("kspacing", None)
    _write_input(case_dir / "INPUT", parameters)
    basis_type = str(parameters.get("basis_type", "pw")).strip().lower()
    active_orb_files = orb_files if basis_type == "lcao" else {}
    ordered_indices = _write_stru(
        case_dir / "STRU",
        atoms,
        pp_files,
        active_orb_files,
    )
    if active_kspacing is None:
        a, b, c, *_ = atoms.cell.cellpar()
        grid = (
            max(1, int(np.ceil(ka[0] / a))),
            max(1, int(np.ceil(ka[1] / b))),
            max(1, int(np.ceil(ka[2] / c))),
        )
        _write_kpt(case_dir / "KPT", grid, gamma=use_gamma)
    _write_input_manifest(
        case_dir,
        resource_dir,
        pp_files,
        active_orb_files,
        resource_provenance=resource_provenance,
    )
    return ordered_indices


def _write_input(path: Path, parameters: Mapping[str, str]) -> None:
    rows = ["INPUT_PARAMETERS"]
    rows.extend(f"{key} {value}" for key, value in parameters.items() if value != "")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_stru(
    path: Path,
    atoms: Atoms,
    pp_files: Mapping[str, str],
    orb_files: Mapping[str, str],
) -> tuple[int, ...]:
    symbols = atoms.get_chemical_symbols()
    species = list(dict.fromkeys(symbols))
    ordered_indices = tuple(
        index for element in species for index, symbol in enumerate(symbols) if symbol == element
    )
    rows = ["ATOMIC_SPECIES"]
    for element in species:
        mass = atomic_masses[atomic_numbers[element]]
        rows.append(f"{element} {mass:.12g} {pp_files[element]}")
    if orb_files:
        rows.extend(["", "NUMERICAL_ORBITAL"])
        rows.extend(orb_files[element] for element in species)
    rows.extend(["", "LATTICE_CONSTANT", f"{1.0 / Bohr:.16g}", "", "LATTICE_VECTORS"])
    rows.extend(_format_vector(vector) for vector in np.asarray(atoms.cell))
    mobility = _constraint_mobility(atoms)
    positions = np.asarray(atoms.positions)
    spin = (
        np.asarray(atoms.arrays["spin"], dtype=np.float64)
        if "spin" in atoms.arrays
        else None
    )
    rows.extend(["", "ATOMIC_POSITIONS", "Cartesian"])
    for element in species:
        indices = [index for index in ordered_indices if symbols[index] == element]
        rows.extend(["", element, "0.0", str(len(indices))])
        for index in indices:
            row = f"{_format_vector(positions[index])} m {_format_int_vector(mobility[index])}"
            if spin is not None:
                row += f" mag {_format_vector(spin[index])} sc 1 1 1"
            rows.append(row)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return ordered_indices


def _write_kpt(path: Path, grid: Sequence[int], *, gamma: bool) -> None:
    mode = "Gamma" if gamma else "MP"
    path.write_text(
        "K_POINTS\n0\n" + mode + "\n" + " ".join(map(str, (*grid, 0, 0, 0))) + "\n",
        encoding="utf-8",
    )


def _write_input_manifest(
    case_dir: Path,
    resource_dir: Path,
    pp_files: Mapping[str, str],
    orb_files: Mapping[str, str],
    *,
    resource_provenance: Mapping[str, object],
) -> None:
    resources = {}
    for element, filename in pp_files.items():
        pseudo = resource_dir / filename
        record = {
            "pseudopotential": str(pseudo),
            "pseudopotential_sha256": file_sha256(pseudo),
        }
        if element in orb_files:
            orbital = resource_dir / orb_files[element]
            record.update(
                {
                    "orbital": str(orbital),
                    "orbital_sha256": file_sha256(orbital),
                }
            )
        resources[element] = record
    (case_dir / "abacus-input.json").write_text(
        json.dumps(
            {
                "backend": "abacus",
                "resource_dir": str(resource_dir),
                "manifest": dict(resource_provenance),
                "resources": resources,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _constraint_mobility(atoms: Atoms) -> np.ndarray:
    mobility = np.ones((len(atoms), 3), dtype=int)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            mobility[constraint.get_indices()] = 0
        elif isinstance(constraint, FixCartesian):
            indices = np.asarray(constraint.get_indices(), dtype=int)
            mask = np.asarray(constraint.mask, dtype=bool)
            if mask.ndim == 1:
                mobility[np.ix_(indices, np.where(mask)[0])] = 0
            else:
                for index, atom_mask in zip(indices, mask):
                    mobility[index, atom_mask] = 0
    return mobility


def _running_scf_log(case_dir: Path, parameters: Mapping[str, str]) -> Path:
    suffix = str(parameters.get("suffix", "ABACUS"))
    preferred = case_dir / f"OUT.{suffix}" / "running_scf.log"
    if preferred.is_file():
        return preferred
    candidates = sorted(case_dir.glob("OUT.*/running_scf.log"))
    if len(candidates) == 1:
        return candidates[0]
    raise NativeAbacusError(f"ABACUS running_scf.log was not produced in {case_dir}")


def _last_energy(lines: Sequence[str]) -> float | None:
    patterns = (
        re.compile(rf"!FINAL_ETOT_IS\s+({_FLOAT_RE.pattern})\s+eV"),
        re.compile(rf"final etot is\s+({_FLOAT_RE.pattern})\s+eV", re.I),
    )
    energy = None
    for line in lines:
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                energy = float(match.group(1))
    return energy


def _last_scf_iteration(lines: Sequence[str]) -> int | None:
    iteration = None
    for line in lines:
        for pattern in (
            re.compile(r"\bELEC\s*=\s*(\d+)"),
            re.compile(r"#ELEC ITER#\s*(\d+)"),
        ):
            match = pattern.search(line)
            if match:
                iteration = int(match.group(1))
                break
        parts = line.split()
        if (
            len(parts) >= 7
            and re.fullmatch(r"[A-Za-z]+\d+", parts[0])
            and len(_floats(" ".join(parts[1:]))) >= 6
        ):
            iteration = int(re.search(r"\d+", parts[0]).group(0))
    return iteration


def _last_vector_table(
    lines: Sequence[str], title: str, expected_rows: int
) -> np.ndarray | None:
    for start in _title_indices(lines, title):
        rows = []
        for line in lines[start + 1 :]:
            stripped = line.strip()
            if not stripped or set(stripped) <= {"-"}:
                if rows:
                    break
                continue
            values = _floats(stripped)
            if len(values) < 3:
                if rows:
                    break
                continue
            rows.append(values[-3:])
            if len(rows) == expected_rows:
                return np.asarray(rows, dtype=float)
    return None


def _last_matrix_table(
    lines: Sequence[str], title: str, *, rows: int
) -> np.ndarray | None:
    for start in _title_indices(lines, title):
        matrix = []
        for line in lines[start + 1 :]:
            values = _floats(line)
            if len(values) >= 3:
                matrix.append(values[:3])
                if len(matrix) == rows:
                    return np.asarray(matrix, dtype=float)
            elif matrix:
                break
    return None


def _last_magnetic_force_table(
    lines: Sequence[str], expected_rows: int
) -> np.ndarray | None:
    for start in _title_indices(lines, "Magnetic force (eV/uB)"):
        rows = []
        for line in lines[start + 1 :]:
            stripped = line.strip()
            if not stripped or set(stripped) <= {"-"}:
                if rows:
                    break
                continue
            parts = stripped.split()
            values = _floats(stripped)
            if len(values) >= 3:
                rows.append(values[-3:])
            else:
                scalar_values = (
                    _floats(" ".join(parts[1:]))
                    if len(parts) > 1
                    else values
                )
                if len(scalar_values) == 1:
                    rows.append([0.0, 0.0, scalar_values[0]])
                elif rows:
                    break
            if len(rows) == expected_rows:
                return np.asarray(rows, dtype=float)
    return None


def _title_indices(lines: Sequence[str], title: str) -> list[int]:
    return [index for index, line in reversed(list(enumerate(lines))) if title in line]


def _restore_atom_order(values: np.ndarray, ordered_indices: Sequence[int]) -> np.ndarray:
    restored = np.empty_like(values)
    restored[np.asarray(ordered_indices, dtype=int)] = values
    return restored


def _write_result_manifest(
    case_dir: Path,
    *,
    command: str,
    returncode: int,
    status: str,
    log_path: Path | None = None,
    parsed: ParsedAbacusResult | None = None,
    electronic_mode: str | None = None,
) -> None:
    inputs = {}
    for name in ("INPUT", "STRU", "KPT", "abacus-input.json"):
        path = case_dir / name
        if path.is_file():
            inputs[name] = file_sha256(path)
    payload = {
        "backend": "abacus",
        "command": command,
        "returncode": returncode,
        "status": status,
        "parser_version": PARSER_VERSION,
        "input_sha256": inputs,
        "electronic_mode": electronic_mode,
        "spin_force_labels": electronic_mode == "constrained_vector_spin_force",
    }
    if log_path is not None:
        payload["running_scf_log"] = str(log_path)
        payload["running_scf_sha256"] = file_sha256(log_path)
    if parsed is not None:
        payload["result"] = {
            "energy_eV": parsed.energy,
            "atom_count": len(parsed.forces),
            "has_magnetization": parsed.magnetization is not None,
            "has_mforce": parsed.mforce is not None,
            "scf_iterations": parsed.scf_iterations,
        }
    (case_dir / "abacus-result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _floats(line: str) -> list[float]:
    return [float(value) for value in _FLOAT_RE.findall(line)]


def _format_vector(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.16g}" for value in values)


def _format_int_vector(values: Sequence[int]) -> str:
    return " ".join(str(int(value)) for value in values)


def _tail(text: str, lines: int = 20) -> str:
    selected = text.strip().splitlines()[-lines:]
    return " | ".join(selected) if selected else "no process output"


__all__ = [
    "NativeAbacusError",
    "NativeAbacusRequest",
    "ParsedAbacusResult",
    "parse_running_scf",
    "run_native_abacus",
    "validate_abacus_spin_contract",
]
