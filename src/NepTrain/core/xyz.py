"""ASE I/O for NepTrain datasets with GPUMD row-major virials."""

from copy import copy
import re

import numpy as np
from ase import Atoms
from ase.io import iread, read, write
from ase.io.extxyz import key_val_str_to_dict
from ase.io.formats import filetype


# Match complete fields so virial-like text in other metadata is ignored.
_FIELD = re.compile(
    r"""([^\s=]+)\s*=\s*("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|"""
    r"""\{(?:\\.|[^}\\])*\}|\[(?:\\.|[^\]\\])*\]|\S+)"""
)


def _read_header(header):
    info = key_val_str_to_dict(header)
    keys = [key for key in info if key.lower() == "virial"]
    if keys:
        if header.lower().count("virial") > 1:
            count = sum(match[1].lower() == "virial" for match in _FIELD.finditer(header))
            if count != 1:
                raise ValueError("expected one virial per frame")
        key = keys[0]
        virial = np.asarray(info[key], dtype=float)
        if virial.size != 9 or not np.isfinite(virial).all():
            raise ValueError("virial must contain nine finite components")
        if key != "virial":
            del info[key]
        # ASE interprets lowercase virial as a column-major matrix.
        info["virial"] = virial.T.reshape(3, 3)
    return info


def read_atoms(path, index=None, *, format=None):
    """Read structures, decoding row-major virials in EXTXYZ inputs."""
    format = format or filetype(str(path))
    if format == "extxyz":
        return read(path, index=index, format=format, properties_parser=_read_header)
    return read(path, index=index, format=format)


def iread_extxyz(path, index=":"):
    """Stream EXTXYZ structures with row-major virials."""
    return iread(path, index=index, format="extxyz", properties_parser=_read_header)


def _encode_frame(atoms):
    has_virial = "virial" in atoms.info
    active = atoms.arrays.get("spin_active")
    encode_active = active is not None and (
        active.dtype.kind == "b"
        or (active.dtype.kind == "f" and np.isin(active, (0, 1)).all())
    )
    if not has_virial and not encode_active:
        return atoms
    frame = copy(atoms)
    if has_virial:
        # A flat vector is written by ASE without reordering its entries.
        frame.info = {**atoms.info, "virial": np.asarray(atoms.info["virial"]).ravel()}
    if encode_active:
        frame.arrays = {**atoms.arrays, "spin_active": active.astype(np.int32)}
    return frame


def write_extxyz(path, frames, *, append=False):
    """Write EXTXYZ with row-major virials and integer spin masks."""
    if isinstance(frames, Atoms):
        frames = (frames,)
    write(path, (_encode_frame(frame) for frame in frames), format="extxyz", append=append)
