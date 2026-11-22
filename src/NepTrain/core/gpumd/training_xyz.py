"""Stage GPUMD XYZ labels without reserializing the atoms or other fields."""

from math import isfinite
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from ..structures import VIRIAL_ORDER_KEY


# Match complete fields so a virial-looking string inside other metadata is ignored.
_FIELD = re.compile(
    r"""([^\s=]+)\s*=\s*("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|"""
    r"""\{(?:\\.|[^}\\])*\}|\[(?:\\.|[^\]\\])*\]|\S+)"""
)
_TRANSPOSE = (0, 3, 6, 1, 4, 7, 2, 5, 8)


def _gpumd_header(header: str) -> str:
    fields = [(match[1].lower(), match) for match in _FIELD.finditer(header)]
    virials = [match for key, match in fields if key == "virial"]
    orders = [match for _, match in fields if match[1] == VIRIAL_ORDER_KEY]
    if not virials and not orders:
        return header
    if len(virials) != 1 or len(orders) > 1:
        raise ValueError("expected one virial and at most one virial order per frame")
    if orders and orders[0][2].strip("\"'") != "row_major":
        raise ValueError(f"invalid {VIRIAL_ORDER_KEY}: {orders[0][2]}")
    match = virials[0]
    values = match[2].strip("\"'{}[]").replace(",", " ").split()
    if len(values) != 9 or not all(isfinite(float(value)) for value in values):
        raise ValueError("virial must contain nine finite components")
    if not orders:
        values = [values[index] for index in _TRANSPOSE]
    text = f'virial="{" ".join(values)}"'
    if not orders:
        text += f" {VIRIAL_ORDER_KEY}=row_major"
    return header[:match.start()] + text + header[match.end():]


def stage_gpumd_xyz(source: Path, target: Path) -> None:
    """Export ASE column order to GPUMD row order; keep marked row order intact."""
    with NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        changed = False
        with source.open(encoding="utf-8", newline="") as input_xyz, temporary.open(
            "w", encoding="utf-8", newline=""
        ) as output:
            frame = 0
            for line in input_xyz:
                output.write(line)
                if not line.strip():
                    continue
                frame += 1
                try:
                    count = int(line)
                    header = next(input_xyz, "")
                    if count < 1 or not header:
                        raise ValueError("missing XYZ header or invalid atom count")
                    converted = _gpumd_header(header)
                    changed |= converted != header
                    output.write(converted)
                    for _ in range(count):
                        atom = next(input_xyz, "")
                        if not atom.strip():
                            raise ValueError("incomplete XYZ atom block")
                        output.write(atom)
                except ValueError as error:
                    raise ValueError(f"{source}, frame {frame}: {error}") from error
        if source.resolve() != target.resolve():
            temporary.replace(target)
        elif changed:
            raise ValueError(f"cannot convert {source} in place; use a separate GPUMD output directory")
    finally:
        temporary.unlink(missing_ok=True)
