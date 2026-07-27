"""Deterministic development smoke built around the production data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from .labeling import LabelRequest, label
from .dft.toy import ToyTeacher
from .fps import farthest_point_sampling
from .spin import SpinDataError, validate_spin_dataset
from .toy_workflow import toy_base_frame, toy_candidate_frames, toy_features


class SmokeError(RuntimeError):
    """Raised when a smoke contract fails."""


@dataclass(frozen=True)
class SmokeReport:
    profile: str
    seed: int
    candidates: int
    selected: int
    max_selected: int
    deterministic_selection: bool
    label_roundtrip: bool
    derivative_force_max_error: float
    derivative_virial_max_error: float
    derivative_mforce_max_error: float | None
    remaining_novelty: float
    selected_energy_span_fraction: float
    recovery_match: bool | None
    passed: bool


def _assert_safe_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    protected = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in protected or len(resolved.parts) < 3:
        raise SmokeError(f"refusing unsafe smoke output path: {resolved}")
    return resolved


def _derivative_probe(
    teacher: ToyTeacher, atoms: Atoms, epsilon: float = 1.0e-6
) -> tuple[float, float, float | None]:
    _, forces, virial, mforce = teacher.calculate(atoms)
    force_errors = []
    for atom_index in range(len(atoms)):
        for axis in range(3):
            plus = atoms.copy()
            minus = atoms.copy()
            plus.positions[atom_index, axis] += epsilon
            minus.positions[atom_index, axis] -= epsilon
            e_plus = teacher.calculate(plus)[0]
            e_minus = teacher.calculate(minus)[0]
            numeric = -(e_plus - e_minus) / (2.0 * epsilon)
            force_errors.append(abs(numeric - forces[atom_index, axis]))

    virial_errors = []
    for axis in range(3):
        plus = atoms.copy()
        minus = atoms.copy()
        plus_cell = np.asarray(atoms.cell).copy()
        minus_cell = np.asarray(atoms.cell).copy()
        plus_cell[axis] *= 1.0 + epsilon
        minus_cell[axis] *= 1.0 - epsilon
        plus.set_cell(plus_cell, scale_atoms=True)
        minus.set_cell(minus_cell, scale_atoms=True)
        e_plus = teacher.calculate(plus)[0]
        e_minus = teacher.calculate(minus)[0]
        numeric = -(e_plus - e_minus) / (2.0 * epsilon)
        virial_errors.append(abs(numeric - virial[axis, axis]))

    mforce_error = None
    if mforce is not None:
        spin_errors = []
        for atom_index in range(len(atoms)):
            for axis in range(3):
                plus = atoms.copy()
                minus = atoms.copy()
                plus_spin = np.asarray(plus.arrays["spin"]).copy()
                minus_spin = np.asarray(minus.arrays["spin"]).copy()
                plus_spin[atom_index, axis] += epsilon
                minus_spin[atom_index, axis] -= epsilon
                plus.set_array("spin", plus_spin)
                minus.set_array("spin", minus_spin)
                e_plus = teacher.calculate(plus)[0]
                e_minus = teacher.calculate(minus)[0]
                numeric = -(e_plus - e_minus) / (2.0 * epsilon)
                spin_errors.append(abs(numeric - mforce[atom_index, axis]))
        mforce_error = float(max(spin_errors, default=0.0))
    return (
        float(max(force_errors, default=0.0)),
        float(max(virial_errors, default=0.0)),
        mforce_error,
    )


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recovery_probe(root: Path, frames: list[Atoms], profile: str) -> bool:
    direct_input = root / "recovery-direct-input.xyz"
    direct_output = root / "recovery-direct.xyz"
    resumed_output = root / "recovery-resumed.xyz"
    ase_write(direct_input, frames, format="extxyz")
    label(
        LabelRequest(direct_input, direct_output, root / "direct", settings={"profile": profile}),
        "toy",
    )

    midpoint = len(frames) // 2
    first = root / "recovery-first.xyz"
    second = root / "recovery-second.xyz"
    ase_write(first, frames[:midpoint], format="extxyz")
    ase_write(second, frames[midpoint:], format="extxyz")
    label(
        LabelRequest(first, resumed_output, root / "resumed-1", settings={"profile": profile}),
        "toy",
    )
    checkpoint = root / "recovery.json"
    checkpoint.write_text(json.dumps({"completed": midpoint}, sort_keys=True), encoding="utf-8")
    completed = json.loads(checkpoint.read_text(encoding="utf-8"))["completed"]
    if completed != midpoint:
        return False
    label(
        LabelRequest(second, resumed_output, root / "resumed-2", append=True, settings={"profile": profile}),
        "toy",
    )
    return _hash_file(direct_output) == _hash_file(resumed_output)


def run_smoke(
    output_dir: str | Path,
    *,
    profile: str = "ordinary",
    seed: int = 20260721,
    max_selected: int = 8,
    force: bool = False,
) -> SmokeReport:
    if profile not in {"ordinary", "spin", "recovery"}:
        raise SmokeError("smoke profile must be ordinary, spin, or recovery")
    if max_selected < 1:
        raise SmokeError("max_selected must be at least 1")
    teacher_profile = "spin" if profile in {"spin", "recovery"} else "ordinary"
    root = _assert_safe_output(Path(output_dir))
    if root.exists():
        if not force:
            raise SmokeError(f"smoke output already exists: {root}; pass --force to replace it")
        shutil.rmtree(root)
    root.mkdir(parents=True)

    seed_frame = toy_base_frame(teacher_profile == "spin")
    candidates = toy_candidate_frames(teacher_profile, seed)
    candidate_file = root / "candidates.xyz"
    ase_write(candidate_file, candidates, format="extxyz")
    truth_file = root / "teacher-truth.xyz"
    truth = label(
        LabelRequest(candidate_file, truth_file, root / "teacher", settings={"profile": teacher_profile}),
        "toy",
    ).frames

    all_frames = [seed_frame, *candidates]
    feature_matrix = toy_features(all_frames, teacher_profile)
    seed_features = feature_matrix[:1]
    candidate_features = feature_matrix[1:]
    selected = farthest_point_sampling(
        candidate_features,
        budget=max_selected,
        min_novelty=0.0,
        reference_descriptors=seed_features,
    ).selected_indices
    repeated = farthest_point_sampling(
        candidate_features,
        budget=max_selected,
        min_novelty=0.0,
        reference_descriptors=seed_features,
    ).selected_indices
    selected_frames = [candidates[index] for index in selected]
    selected_input = root / "selected-input.xyz"
    selected_output = root / "selected-labels.xyz"
    ase_write(selected_input, selected_frames, format="extxyz")
    selected_result = label(
        LabelRequest(selected_input, selected_output, root / "selected", settings={"profile": teacher_profile}),
        "toy",
    )
    restored = ase_read(selected_output, index=":", format="extxyz")
    restored = restored if isinstance(restored, list) else [restored]
    validate_spin_dataset(restored, require_mforce=True)
    label_roundtrip = len(restored) == len(selected) and all(
        frame.calc is not None
        and "energy" in frame.calc.results
        and np.isfinite(frame.calc.results["energy"])
        and np.asarray(frame.calc.results.get("forces", [])).shape == (len(frame), 3)
        and np.asarray(frame.info.get("virial", [])).shape == (3, 3)
        for frame in restored
    )

    missing_mforce_rejected = True
    if teacher_profile == "spin":
        broken = restored[0].copy()
        broken.arrays.pop("mforce", None)
        try:
            validate_spin_dataset([broken], require_mforce=True)
        except SpinDataError:
            pass
        else:
            missing_mforce_rejected = False

    teacher = ToyTeacher(teacher_profile)
    force_error, virial_error, mforce_error = _derivative_probe(teacher, seed_frame)
    selected_features = candidate_features[list(selected)]
    reference = np.vstack([seed_features, selected_features])
    remaining = np.linalg.norm(candidate_features[:, None, :] - reference[None, :, :], axis=2).min(axis=1)
    energies = np.asarray([frame.get_potential_energy() for frame in truth])
    selected_energies = energies[list(selected)]
    energy_span = float(np.ptp(energies))
    span_fraction = float(np.ptp(selected_energies) / energy_span) if energy_span > 0 else 1.0

    recovery_match = None
    if profile == "recovery":
        recovery_match = _recovery_probe(root, candidates[:10], teacher_profile)

    passed = (
        selected == repeated
        and len(selected_result.frames) == len(selected)
        and label_roundtrip
        and missing_mforce_rejected
        and force_error < 1.0e-7
        and virial_error < 1.0e-7
        and (mforce_error is None or mforce_error < 1.0e-7)
        and recovery_match is not False
    )
    report = SmokeReport(
        profile=profile,
        seed=seed,
        candidates=len(candidates),
        selected=len(selected),
        max_selected=max_selected,
        deterministic_selection=selected == repeated,
        label_roundtrip=label_roundtrip and missing_mforce_rejected,
        derivative_force_max_error=force_error,
        derivative_virial_max_error=virial_error,
        derivative_mforce_max_error=mforce_error,
        remaining_novelty=float(remaining.max(initial=0.0)),
        selected_energy_span_fraction=span_fraction,
        recovery_match=recovery_match,
        passed=passed,
    )
    (root / "smoke-report.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not passed:
        raise SmokeError(f"{profile} smoke failed; see {root / 'smoke-report.json'}")
    return report


__all__ = [
    "SmokeError",
    "SmokeReport",
    "run_smoke",
]
