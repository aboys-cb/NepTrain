"""LAMMPS input, execution, and trajectory conversion."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Mapping

import numpy as np
from ase import Atoms
from ase.calculators.lammps.coordinatetransform import Prism
from ase.data import atomic_masses, atomic_numbers
from ase.io import write as ase_write

from ..nep.calculator import resolve_backend
from ..persistence import atomic_write_json
from ..spin import SPIN_KEY, spin_from_lammps, spin_to_lammps, validate_spin_structure
from .health import TrajectoryHealthPolicy, classify_trajectory


_VARIABLE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")
_COMPUTE_PROPERTY = re.compile(
    r"^\s*compute\s+(\S+)\s+\S+\s+property/atom\s+(.+?)\s*$",
    re.MULTILINE,
)
_FIX_HALT = re.compile(
    r"Fix halt condition for fix-id (?P<fix_id>\S+) met on step "
    r"(?P<step>\d+) with value (?P<value>\S+)"
)
_MAIN_RUN = re.compile(
    r"^(?P<indent>[ \t]*)run[ \t]+{{\s*steps\s*}}[ \t]*$",
    re.MULTILINE,
)
_HALT_REASONS = {
    "neptrain_halt_volume_low": "volume_ratio_below_min",
    "neptrain_halt_volume_high": "volume_ratio_above_max",
    "neptrain_halt_force": "max_force_above_limit",
}


class LammpsError(RuntimeError):
    pass


@dataclass(frozen=True)
class _HaltEvent:
    fix_id: str
    step: int
    value: str

    @property
    def reason_code(self) -> str:
        return _HALT_REASONS.get(self.fix_id, "lammps_fix_halt")

    def to_dict(self) -> dict[str, object]:
        return {
            "fix_id": self.fix_id,
            "step": self.step,
            "value": self.value,
            "reason_code": self.reason_code,
        }


def _halt_commands(
    policy: TrajectoryHealthPolicy,
    *,
    reference_volume: float,
    interval: int,
) -> str:
    """Render online checks that LAMMPS can evaluate without duplicating MIC logic."""

    commands: list[str] = []
    if (
        policy.min_volume_ratio is not None
        or policy.max_volume_ratio is not None
    ):
        commands.append("variable neptrain_volume equal vol")
    if policy.min_volume_ratio is not None:
        threshold = reference_volume * policy.min_volume_ratio
        commands.append(
            "fix neptrain_halt_volume_low all halt "
            f"{interval} v_neptrain_volume < {threshold:.17g} "
            "error soft message yes"
        )
    if policy.max_volume_ratio is not None:
        threshold = reference_volume * policy.max_volume_ratio
        commands.append(
            "fix neptrain_halt_volume_high all halt "
            f"{interval} v_neptrain_volume > {threshold:.17g} "
            "error soft message yes"
        )
    if policy.max_force is not None:
        commands.extend(
            [
                "variable neptrain_force_norm atom "
                "sqrt(fx*fx+fy*fy+fz*fz)",
                "compute neptrain_max_force all reduce max "
                "v_neptrain_force_norm",
                "variable neptrain_max_force_value equal "
                "c_neptrain_max_force",
                "fix neptrain_halt_force all halt "
                f"{interval} v_neptrain_max_force_value > "
                f"{policy.max_force:.17g} "
                "error soft message yes",
            ]
        )
    return "\n".join(commands)


def _parse_halt_event(text: str) -> _HaltEvent | None:
    match = _FIX_HALT.search(text)
    if match is None:
        return None
    return _HaltEvent(
        fix_id=match.group("fix_id"),
        step=int(match.group("step")),
        value=match.group("value"),
    )


def _with_halt_placeholder(template: str) -> str:
    """Upgrade an existing single-run template without rewriting custom logic."""

    if "halt_commands" in _VARIABLE.findall(template):
        return template
    matches = list(_MAIN_RUN.finditer(template))
    if len(matches) != 1:
        return template
    match = matches[0]
    replacement = (
        f"{match.group('indent')}{{{{ halt_commands }}}}\n{match.group(0)}"
    )
    return template[: match.start()] + replacement + template[match.end() :]


def render_template(template: str, variables: Mapping[str, object]) -> str:
    required = set(_VARIABLE.findall(template))
    missing = sorted(required.difference(variables))
    if missing:
        raise LammpsError(f"missing LAMMPS template variables: {', '.join(missing)}")

    def replace(match: re.Match[str]) -> str:
        value = variables[match.group(1)]
        return "" if value is None else str(value)

    rendered = _VARIABLE.sub(replace, template)
    unresolved = _VARIABLE.findall(rendered)
    if unresolved:
        raise LammpsError(f"unresolved LAMMPS template variables: {unresolved}")
    return rendered


def compute_property_columns(rendered_input: str) -> dict[str, str]:
    """Map dump column names to the property named by compute property/atom."""

    result: dict[str, str] = {}
    for match in _COMPUTE_PROPERTY.finditer(rendered_input):
        compute_id, properties = match.groups()
        for index, property_name in enumerate(properties.split(), start=1):
            result[f"c_{compute_id}[{index}]"] = property_name
    return result


def write_lammps_data(
    path: Path,
    atoms: Atoms,
    elements: tuple[str, ...],
    *,
    spin: bool,
) -> Prism:
    if not np.all(atoms.pbc):
        raise LammpsError("LAMMPS NEP MD currently requires fully periodic structures")
    symbols = atoms.get_chemical_symbols()
    unknown = sorted(set(symbols).difference(elements))
    if unknown:
        raise LammpsError(f"elements absent from model: {', '.join(unknown)}")

    prism = Prism(np.asarray(atoms.cell), pbc=atoms.pbc)
    xhi, yhi, zhi, xy, xz, yz = prism.get_lammps_prism()
    positions = prism.vector_to_lammps(atoms.positions, wrap=True)
    type_map = {element: index + 1 for index, element in enumerate(elements)}
    lines = [
        "LAMMPS data file generated by NepTrain",
        "",
        f"{len(atoms)} atoms",
        f"{len(elements)} atom types",
        "",
        f"0.0 {xhi:.17g} xlo xhi",
        f"0.0 {yhi:.17g} ylo yhi",
        f"0.0 {zhi:.17g} zlo zhi",
        f"{xy:.17g} {xz:.17g} {yz:.17g} xy xz yz",
        "",
        "Masses",
        "",
    ]
    for index, element in enumerate(elements, start=1):
        lines.append(f"{index} {atomic_masses[atomic_numbers[element]]:.17g}")
    lines.extend(["", f"Atoms # {'spin' if spin else 'atomic'}", ""])

    if spin:
        validate_spin_structure(atoms, require_mforce=False)
        spin_data = spin_to_lammps(atoms.arrays[SPIN_KEY])
        directions = prism.vector_to_lammps(spin_data.direction)
        for atom_id, (symbol, position, direction, magnitude) in enumerate(
            zip(symbols, positions, directions, spin_data.magnitude), start=1
        ):
            row = [
                atom_id,
                type_map[symbol],
                *position,
                *direction,
                magnitude,
                0,
                0,
                0,
            ]
            lines.append(" ".join(f"{value:.17g}" if isinstance(value, float | np.floating) else str(value) for value in row))
    else:
        for atom_id, (symbol, position) in enumerate(zip(symbols, positions), start=1):
            row = [atom_id, type_map[symbol], *position, 0, 0, 0]
            lines.append(" ".join(f"{value:.17g}" if isinstance(value, float | np.floating) else str(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return prism


def _box_from_dump(bounds: list[list[float]], triclinic: bool) -> tuple[np.ndarray, np.ndarray]:
    if triclinic:
        xlo_bound, xhi_bound, xy = bounds[0]
        ylo_bound, yhi_bound, xz = bounds[1]
        zlo_bound, zhi_bound, yz = bounds[2]
        xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
        xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max(0.0, yz)
    else:
        xlo, xhi = bounds[0][:2]
        ylo, yhi = bounds[1][:2]
        zlo_bound, zhi_bound = bounds[2][:2]
        xy = xz = yz = 0.0
    origin = np.asarray([xlo, ylo, zlo_bound], dtype=np.float64)
    cell = np.asarray(
        [[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi_bound - zlo_bound]],
        dtype=np.float64,
    )
    return origin, cell


def read_lammps_dump(
    path: Path,
    prism: Prism,
    elements: tuple[str, ...],
    *,
    spin: bool,
    property_columns: Mapping[str, str] | None = None,
    allow_incomplete_tail: bool = False,
) -> list[Atoms]:
    if not path.is_file():
        raise LammpsError(f"LAMMPS dump does not exist: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    frames: list[Atoms] = []
    cursor = 0
    while cursor < len(lines):
        if lines[cursor] != "ITEM: TIMESTEP":
            cursor += 1
            continue
        if cursor + 8 >= len(lines):
            if allow_incomplete_tail:
                break
            raise LammpsError("invalid LAMMPS dump: incomplete frame header")
        step = int(lines[cursor + 1])
        if lines[cursor + 2] != "ITEM: NUMBER OF ATOMS":
            raise LammpsError("invalid LAMMPS dump: NUMBER OF ATOMS header missing")
        count = int(lines[cursor + 3])
        box_header = lines[cursor + 4]
        if not box_header.startswith("ITEM: BOX BOUNDS"):
            raise LammpsError("invalid LAMMPS dump: BOX BOUNDS header missing")
        triclinic = all(token in box_header.split() for token in ("xy", "xz", "yz"))
        bounds = [list(map(float, lines[cursor + 5 + i].split())) for i in range(3)]
        atom_header_index = cursor + 8
        if not lines[atom_header_index].startswith("ITEM: ATOMS"):
            raise LammpsError("invalid LAMMPS dump: ATOMS header missing")
        frame_end = atom_header_index + 1 + count
        if frame_end > len(lines):
            if allow_incomplete_tail:
                break
            raise LammpsError("invalid LAMMPS dump: incomplete atom rows")
        columns = lines[atom_header_index].split()[2:]
        rows = [lines[atom_header_index + 1 + i].split() for i in range(count)]
        raw_index = {name: position for position, name in enumerate(columns)}
        index = dict(raw_index)
        for dump_column, property_name in (property_columns or {}).items():
            if dump_column in raw_index:
                if property_name in index and index[property_name] != raw_index[dump_column]:
                    raise LammpsError(
                        f"ambiguous dump property {property_name}: direct and computed columns differ"
                    )
                index[property_name] = raw_index[dump_column]
        required = {"id", "type", "x", "y", "z"}
        if not required.issubset(index):
            raise LammpsError("LAMMPS dump is missing id/type/x/y/z columns")
        rows.sort(key=lambda row: int(row[index["id"]]))
        origin, lammps_cell = _box_from_dump(bounds, triclinic)
        lammps_positions = np.asarray(
            [[float(row[index[key]]) for key in ("x", "y", "z")] for row in rows]
        ) - origin
        symbols = [elements[int(row[index["type"]]) - 1] for row in rows]
        frame = Atoms(
            symbols,
            positions=prism.vector_to_ase(lammps_positions),
            cell=prism.vector_to_ase(lammps_cell),
            pbc=True,
        )
        frame.info["Config_type"] = f"lammps-step-{step}"
        frame.info["lammps_step"] = step
        frame.info["md_step"] = step
        force_columns = ("fx", "fy", "fz")
        if all(name in index for name in force_columns):
            forces = np.asarray(
                [[float(row[index[name]]) for name in force_columns] for row in rows]
            )
            frame.set_array("nep_force", prism.vector_to_ase(forces))
        if spin:
            spin_columns = ("spx", "spy", "spz")
            if not all(name in index for name in (*spin_columns, "sp")):
                raise LammpsError(
                    "spin dump must expose sp, spx, spy, and spz through compute property/atom"
                )
            direction = np.asarray(
                [[float(row[index[name]]) for name in spin_columns] for row in rows]
            )
            magnitude = np.asarray([float(row[index["sp"]]) for row in rows])
            physical_spin = spin_from_lammps(direction, magnitude)
            frame.set_array(SPIN_KEY, prism.vector_to_ase(physical_spin))
            mforce_columns = ("fmx", "fmy", "fmz")
            if all(name in index for name in mforce_columns):
                mforces = np.asarray(
                    [[float(row[index[name]]) for name in mforce_columns] for row in rows]
                )
                frame.set_array("mforce", prism.vector_to_ase(mforces))
        frames.append(frame)
        cursor = frame_end
    if not frames:
        raise LammpsError(f"LAMMPS produced no readable frames in {path}")
    return frames


@dataclass(frozen=True)
class LammpsRunResult:
    trajectory: Path
    input_file: Path
    log_file: Path
    backend: str
    pair_style: str
    completed: bool = True
    last_step: int | None = None
    failure_code: str | None = None
    failure_reason: str | None = None
    health_report: Path | None = None


def _failure_reason(returncode: int, stderr: str, stdout: str) -> str:
    stderr_lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    stdout_lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    detail = next(
        (line for line in reversed([*stderr_lines, *stdout_lines]) if "ERROR:" in line),
        stderr_lines[-1]
        if stderr_lines
        else stdout_lines[-1]
        if stdout_lines
        else "no output detail",
    )
    return f"exit code {returncode}: {detail[:500]}"


def run_lammps(
    *,
    atoms: Atoms,
    model_file: Path,
    output_dir: Path,
    output_file: Path,
    template: str,
    variables: Mapping[str, object],
    inference_backend: str,
    lmp_command: str,
    mpiexec: str,
    mpi_ranks: int,
    spin: bool,
    pre_failure_frames: int = 2,
    bad_tail_frames: int = 1,
    health_policy: TrajectoryHealthPolicy | None = None,
) -> LammpsRunResult:
    try:
        from nep_adapters import inspect_model
    except ImportError as error:  # pragma: no cover
        raise LammpsError("NEPAdapters is required for LAMMPS MD") from error

    info = inspect_model(model_file)
    if spin != info.supports("spin"):
        raise LammpsError(
            f"spin={spin} does not match model capability spin={info.supports('spin')}"
        )
    selected = resolve_backend(model_file, inference_backend)
    pair_style = "nep/gpu/kk" if selected == "cuda" else "nep/cpu"
    atom_style = "spin/kk" if spin and selected == "cuda" else "spin" if spin else "atomic"
    fix_suffix = "/kk" if selected == "cuda" else ""
    output_dir.mkdir(parents=True, exist_ok=True)
    local_model = output_dir / "nep.txt"
    if model_file.resolve() != local_model.resolve():
        local_model.write_bytes(model_file.read_bytes())
    data_file = output_dir / "structure.data"
    prism = write_lammps_data(data_file, atoms, info.elements, spin=spin)
    dump_file = output_dir / "dump.lammpstrj"
    input_file = output_dir / "lammps.in"
    log_file = output_dir / "log.lammps"
    template_variables = dict(variables)
    policy = health_policy or TrajectoryHealthPolicy()
    halt_interval = max(
        1,
        min(
            100,
            int(template_variables.get("dump_interval", 100)),
            int(template_variables.get("steps", 100)),
        ),
    )
    template_variables.update(
        atom_style=atom_style,
        structure_file=data_file.name,
        model_file=local_model.name,
        pair_style=pair_style,
        elements=" ".join(info.elements),
        fix_suffix=fix_suffix,
        trajectory_file=dump_file.name,
        halt_commands=_halt_commands(
            policy,
            reference_volume=float(abs(atoms.get_volume())),
            interval=halt_interval,
        ),
    )
    rendered_input = render_template(
        _with_halt_placeholder(template), template_variables
    )
    input_file.write_text(rendered_input, encoding="utf-8")

    command: list[str] = []
    if mpi_ranks > 1:
        if not mpiexec:
            raise LammpsError("mpiexec is required when mpi_ranks > 1")
        command.extend(shlex.split(mpiexec))
        command.extend(["-np", str(mpi_ranks)])
    command.extend(shlex.split(lmp_command))
    if selected == "cuda":
        command.extend(["-k", "on", "g", "1", "-sf", "kk"])
    command.extend(["-in", input_file.name, "-log", log_file.name])
    environment = os.environ.copy()
    completed = subprocess.run(
        command,
        cwd=output_dir,
        env=environment,
        text=True,
        capture_output=True,
    )
    (output_dir / "lammps.stdout").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "lammps.stderr").write_text(completed.stderr, encoding="utf-8")
    grids = re.findall(r"(\d+) by (\d+) by (\d+) MPI processor grid", completed.stdout)
    if completed.returncode == 0 and mpi_ranks > 1:
        if not grids:
            raise LammpsError("could not verify the LAMMPS MPI processor grid")
        observed = int(grids[0][0]) * int(grids[0][1]) * int(grids[0][2])
        if observed != mpi_ranks:
            raise LammpsError(
                f"requested {mpi_ranks} MPI ranks but LAMMPS reported a {observed}-rank grid"
            )
    log_text = (
        log_file.read_text(encoding="utf-8", errors="replace")
        if log_file.is_file()
        else ""
    )
    halt_event = _parse_halt_event(f"{completed.stdout}\n{log_text}")
    subprocess_completed = completed.returncode == 0
    run_completed = subprocess_completed and halt_event is None
    if halt_event is not None:
        failure_reason = (
            f"{halt_event.reason_code}: fix {halt_event.fix_id} halted LAMMPS "
            f"at step {halt_event.step} (value {halt_event.value})"
        )
        if not subprocess_completed:
            failure_reason += "; " + _failure_reason(
                completed.returncode, completed.stderr, completed.stdout
            )
    elif not subprocess_completed:
        failure_reason = _failure_reason(
            completed.returncode, completed.stderr, completed.stdout
        )
    else:
        failure_reason = None
    try:
        frames = read_lammps_dump(
            dump_file,
            prism,
            info.elements,
            spin=spin,
            property_columns=compute_property_columns(rendered_input),
            allow_incomplete_tail=not run_completed,
        )
    except LammpsError as error:
        if run_completed:
            raise
        raise LammpsError(
            f"LAMMPS failed ({failure_reason}) and produced no recoverable trajectory: {error}"
        ) from error
    health = classify_trajectory(
        frames,
        atoms,
        process_completed=run_completed,
        policy=policy,
        pre_failure_frames=pre_failure_frames,
        bad_tail_frames=bad_tail_frames,
    )
    for frame, window in zip(frames, health.windows):
        frame.info.update(
            md_window=window,
            md_completed=health.trajectory_completed,
        )
    health_payload = health.to_dict()
    health_payload["process_failure_reason"] = failure_reason
    health_payload["halt"] = (
        halt_event.to_dict() if halt_event is not None else None
    )
    health_path = output_dir / "trajectory-health.json"
    atomic_write_json(health_path, health_payload)
    result_failure_code = None
    result_failure_reason = None
    if health.first_bad_frame is not None:
        result_failure_code = "trajectory_health"
        result_failure_reason = (
            f"trajectory health failed at step {health.first_bad_step}: "
            + ", ".join(health.reason_codes)
        )
        if failure_reason is not None:
            result_failure_reason += f"; LAMMPS also failed ({failure_reason})"
    elif halt_event is not None:
        result_failure_code = "trajectory_halt"
        result_failure_reason = failure_reason
    elif not run_completed:
        result_failure_code = "lammps_nonzero_exit"
        result_failure_reason = failure_reason
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ase_write(output_file, frames, format="extxyz")
    return LammpsRunResult(
        output_file,
        input_file,
        log_file,
        selected,
        pair_style,
        completed=health.trajectory_completed,
        last_step=int(frames[-1].info["lammps_step"]),
        failure_code=result_failure_code,
        failure_reason=result_failure_reason,
        health_report=health_path,
    )
