"""Non-magnetic VASP single-point execution and result normalization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import ase
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.vasp import Vasp

from ...content_addressing import file_sha256
from ..attempts import new_attempt_directory
from .io import VaspInput
from .resources import validate_vasp_resources


class NativeVaspError(RuntimeError):
    """Raised when VASP cannot produce a valid non-magnetic label."""


@dataclass(frozen=True)
class NativeVaspRequest:
    work_dir: Path
    resource_dir: Path
    resource_manifest: Path
    command: str
    input_file: Path
    use_gamma: bool
    kpoint_mode: str
    kspacing: float | None
    ka: tuple[int, int, int]
    flat_single_case: bool = False


def run_native_vasp(
    atoms: Atoms, request: NativeVaspRequest, *, case_index: int
) -> Atoms:
    """Run one fresh VASP single-point attempt and return a portable frame."""

    electronic_mode = None
    case_dir = new_attempt_directory(
        request.work_dir,
        case_index,
        atoms.get_chemical_formula(),
        flat_single_case=request.flat_single_case,
    )
    resource_provenance = None
    try:
        resource_provenance = validate_vasp_resources(
            request.resource_dir,
            request.resource_manifest,
            atoms,
        )
        calculator = VaspInput(pp_path=request.resource_dir)
        calculator.read_incar(request.input_file)
        electronic_mode = _validate_single_point_input(calculator)
        validate_vasp_structure(atoms, electronic_mode=electronic_mode)
        settings = {
            "directory": str(case_dir),
            "command": request.command,
            "pp": "",
            "setups": {
                "base": "minimal",
                **resource_provenance["ase_setups"],
            },
        }
        template_kspacing = calculator.float_params.get("kspacing")
        active_kspacing = (
            request.kspacing
            if request.kspacing is not None
            else template_kspacing
        )
        if request.kpoint_mode == "kpoints":
            active_kspacing = None
        elif request.kpoint_mode == "kspacing" and active_kspacing is None:
            raise NativeVaspError(
                "VASP kpoint_mode=kspacing requires KSPACING in the request or INCAR"
            )
        if request.kpoint_mode == "auto" and template_kspacing is not None:
            # read_incar() already loaded KSPACING and KGAMMA.  Do not rewrite
            # either value: in auto mode the user input remains authoritative.
            pass
        elif active_kspacing is not None:
            settings.update(kspacing=active_kspacing, kgamma=request.use_gamma)
        else:
            a, b, c, *_ = atoms.cell.cellpar()
            settings.update(
                kpts=(
                    max(1, int(np.ceil(request.ka[0] / a))),
                    max(1, int(np.ceil(request.ka[1] / b))),
                    max(1, int(np.ceil(request.ka[2] / c))),
                ),
                gamma=request.use_gamma,
                kspacing=None,
            )
        calculator.set(**settings)
        calculator.calculate(atoms, ("energy", "forces", "stress"))
        if not calculator.converged:
            raise NativeVaspError(f"VASP electronic SCF did not converge in {case_dir}")
        energy, forces, stress = _validated_results(calculator.results, len(atoms))
    except Exception as error:
        _write_manifest(
            case_dir,
            command=request.command,
            status="failed",
            error=str(error),
            electronic_mode=electronic_mode,
            resources=resource_provenance,
        )
        raise

    frame = atoms.copy()
    frame.calc = SinglePointCalculator(frame, energy=energy, forces=forces)
    frame.info["virial"] = _stress_to_virial(stress, frame.get_volume())
    frame.info["dft_electronic_mode"] = electronic_mode
    frame.info.setdefault("Config_type", "NepTrain scf ")
    frame.info["Weight"] = 1.0
    _write_manifest(
        case_dir,
        command=request.command,
        status="completed",
        energy=energy,
        atom_count=len(frame),
        electronic_mode=electronic_mode,
        resources=resource_provenance,
    )
    return frame


def _validate_single_point_input(calculator: VaspInput) -> str:
    ibrion = calculator.int_params.get("ibrion")
    nsw = calculator.int_params.get("nsw")
    ispin = calculator.int_params.get("ispin")
    if ibrion not in {None, -1} or nsw not in {None, 0}:
        raise NativeVaspError(
            "VASP labeling requires a fixed-geometry single point: IBRION=-1 and NSW=0"
        )
    if ispin not in {None, 1, 2}:
        raise NativeVaspError(
            "VASP labeling supports non-spin-polarized ISPIN=1 or collinear "
            "ISPIN=2 ordinary energy/force labels"
        )
    if calculator.bool_params.get("lnoncollinear"):
        raise NativeVaspError("VASP labeling forbids LNONCOLLINEAR")
    if calculator.bool_params.get("lsorbit"):
        raise NativeVaspError("VASP labeling forbids LSORBIT")
    magmom = calculator.list_float_params.get("magmom")
    if (
        ispin != 2
        and magmom is not None
        and any(abs(float(value)) > 0.0 for value in magmom)
    ):
        raise NativeVaspError(
            "nonzero MAGMOM requires collinear ISPIN=2"
        )
    return "collinear_spin_polarized" if ispin == 2 else "non_spin_polarized"


def validate_vasp_input_file(input_file: str | Path) -> str:
    """Validate the VASP physics contract without launching VASP."""

    calculator = Vasp()
    try:
        calculator.read_incar(str(Path(input_file).expanduser().resolve()))
    except Exception as error:
        raise NativeVaspError(f"cannot parse VASP INCAR {input_file}: {error}") from error
    return _validate_single_point_input(calculator)


def validate_vasp_structure(
    atoms: Atoms,
    *,
    electronic_mode: str | None = None,
) -> None:
    """Reject magnetic structure inputs that VASP cannot label faithfully."""

    if "spin" in atoms.arrays:
        raise NativeVaspError(
            "VASP production labeling does not produce spin/mforce labels; "
            "use ABACUS DeltaSpin"
        )
    initial = atoms.arrays.get("initial_magmoms")
    if initial is None:
        return
    values = np.asarray(initial, dtype=float)
    if values.ndim != 1 and np.any(np.abs(values) > 0.0):
        raise NativeVaspError(
            "VASP labeling forbids noncollinear initial magnetic moments"
        )
    if (
        np.any(np.abs(values) > 0.0)
        and electronic_mode not in {None, "collinear_spin_polarized"}
    ):
        raise NativeVaspError(
            "nonzero initial magnetic moments require collinear ISPIN=2"
        )


def _validated_results(
    results: dict, expected_atoms: int
) -> tuple[float, np.ndarray, np.ndarray]:
    missing = [key for key in ("energy", "forces", "stress") if key not in results]
    if missing:
        raise NativeVaspError("VASP result is missing: " + ", ".join(missing))
    energy = float(results["energy"])
    forces = np.asarray(results["forces"], dtype=float)
    stress = np.asarray(results["stress"], dtype=float)
    if forces.shape != (expected_atoms, 3):
        raise NativeVaspError(
            f"VASP force shape must be ({expected_atoms}, 3), got {forces.shape}"
        )
    if stress.shape not in {(6,), (3, 3)}:
        raise NativeVaspError(f"VASP stress shape must be (6,) or (3, 3), got {stress.shape}")
    if not np.isfinite(energy) or not np.all(np.isfinite(forces)) or not np.all(
        np.isfinite(stress)
    ):
        raise NativeVaspError("VASP result contains non-finite values")
    return energy, forces, stress


def _stress_to_virial(stress: np.ndarray, volume: float) -> np.ndarray:
    if stress.shape == (3, 3):
        return -stress * volume
    xx, yy, zz, yz, xz, xy = stress
    return -volume * np.asarray(
        [(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)], dtype=float
    )


def _write_manifest(
    case_dir: Path,
    *,
    command: str,
    status: str,
    error: str | None = None,
    energy: float | None = None,
    atom_count: int | None = None,
    electronic_mode: str | None = None,
    resources: dict | None = None,
) -> None:
    inputs = {}
    for name in ("INCAR", "POSCAR", "KPOINTS", "POTCAR"):
        path = case_dir / name
        if path.is_file():
            inputs[name] = file_sha256(path)
    payload = {
        "backend": "vasp",
        "command": command,
        "status": status,
        "parser": "ase.calculators.vasp",
        "ase_version": ase.__version__,
        "input_sha256": inputs,
        "electronic_mode": electronic_mode,
        "spin_force_labels": False,
    }
    outputs = {}
    for name in ("vasprun.xml", "OUTCAR"):
        path = case_dir / name
        if path.is_file():
            outputs[name] = file_sha256(path)
    if outputs:
        payload["output_sha256"] = outputs
    if resources is not None:
        payload["pseudopotentials"] = resources
    if error is not None:
        payload["error"] = error
    if energy is not None:
        payload["result"] = {"energy_eV": energy, "atom_count": atom_count}
    (case_dir / "vasp-result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


__all__: Sequence[str] = (
    "NativeVaspError",
    "NativeVaspRequest",
    "run_native_vasp",
    "validate_vasp_input_file",
    "validate_vasp_structure",
)
