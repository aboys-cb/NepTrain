"""Content-addressed VASP pseudopotential resource contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping

from ase import Atoms

from ...content_addressing import file_sha256


class VaspResourceError(RuntimeError):
    """Raised before VASP launch when POTCAR provenance is incomplete."""


_PROTOCOL = "neptrain.vasp-resources.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VaspResourceError(
            f"cannot read VASP resource manifest {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise VaspResourceError(
            f"VASP resource manifest {path} must contain an object"
        )
    manifest = dict(value)
    if manifest.get("protocol") != _PROTOCOL:
        raise VaspResourceError(
            f"VASP resource manifest {path} must use protocol {_PROTOCOL}"
        )
    for field in ("family", "release"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise VaspResourceError(
                f"VASP resource manifest {path} requires non-empty {field}"
            )
    elements = manifest.get("elements")
    if not isinstance(elements, Mapping) or not elements:
        raise VaspResourceError(
            f"VASP resource manifest {path} requires an elements mapping"
        )
    normalized: dict[str, dict[str, str]] = {}
    for symbol, raw in elements.items():
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][a-z]?", symbol):
            raise VaspResourceError(
                f"invalid element key in VASP resource manifest: {symbol!r}"
            )
        if not isinstance(raw, Mapping):
            raise VaspResourceError(
                f"VASP resource manifest element {symbol} must be an object"
            )
        unknown = sorted(set(raw) - {"path", "sha256", "titel"})
        if unknown:
            raise VaspResourceError(
                f"VASP resource manifest element {symbol} has unknown fields: "
                + ", ".join(unknown)
            )
        relative = str(raw.get("path", ""))
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.name != "POTCAR"
            or len(pure.parts) != 2
            or (
                pure.parent.name != symbol
                and not pure.parent.name.startswith(symbol + "_")
            )
        ):
            raise VaspResourceError(
                f"VASP resource path for {symbol} must be "
                f"{symbol}/POTCAR or {symbol}_<setup>/POTCAR under the "
                "configured resource root"
            )
        sha256 = str(raw.get("sha256", "")).lower()
        if _SHA256.fullmatch(sha256) is None:
            raise VaspResourceError(
                f"VASP resource manifest element {symbol} requires a SHA256"
            )
        titel = str(raw.get("titel", "")).strip()
        if not titel:
            raise VaspResourceError(
                f"VASP resource manifest element {symbol} requires an exact TITEL"
            )
        normalized[symbol] = {
            "path": pure.as_posix(),
            "sha256": sha256,
            "titel": titel,
        }
    manifest["elements"] = normalized
    return manifest


def vasp_element_order(atoms: Atoms) -> tuple[str, ...]:
    """Return the exact first-appearance order used by ASE's POTCAR builder."""

    return tuple(dict.fromkeys(atoms.get_chemical_symbols()))


def validate_vasp_manifest_elements(
    manifest_path: str | Path,
    element_orders: Iterable[Iterable[str]],
) -> dict[str, Any]:
    """Validate manifest syntax and coverage without accessing the resource root."""

    path = Path(manifest_path).expanduser().resolve()
    manifest = _read_manifest(path)
    required = {
        str(symbol)
        for order in element_orders
        for symbol in order
    }
    missing = sorted(required - set(manifest["elements"]))
    if missing:
        raise VaspResourceError(
            "VASP resource manifest is missing elements: " + ", ".join(missing)
        )
    return {
        "protocol": _PROTOCOL,
        "manifest_sha256": file_sha256(path),
        "family": manifest["family"],
        "release": manifest["release"],
        "required_elements": sorted(required),
    }


def vasp_resource_files(
    manifest_path: str | Path,
) -> tuple[dict[str, str], ...]:
    """Return validated immutable files for preflight checks."""

    manifest = _read_manifest(Path(manifest_path).expanduser().resolve())
    return tuple(
        {
            "element": symbol,
            "path": record["path"],
            "sha256": record["sha256"],
        }
        for symbol, record in sorted(manifest["elements"].items())
    )


def _potcar_identity(path: Path) -> tuple[str, str]:
    titel = None
    element = None
    try:
        with path.open("rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if titel is None and "TITEL" in line and "=" in line:
                    titel = line.split("=", 1)[1].strip()
                if element is None:
                    match = re.search(r"\bVRHFIN\s*=\s*([A-Z][a-z]?)\s*:", line)
                    if match:
                        element = match.group(1)
                if titel is not None and element is not None:
                    break
    except OSError as error:
        raise VaspResourceError(f"cannot read POTCAR {path}: {error}") from error
    if titel is None or element is None:
        raise VaspResourceError(
            f"POTCAR {path} is missing TITEL or VRHFIN identity metadata"
        )
    return titel, element


def validate_vasp_resources(
    resource_root: str | Path,
    manifest_path: str | Path,
    atoms: Atoms,
) -> dict[str, Any]:
    """Validate ordered POTCAR files, exact versions, and hashes for one frame."""

    root = Path(resource_root).expanduser()
    if not root.is_dir():
        raise VaspResourceError(
            f"VASP pseudopotential root does not exist: {root}"
        )
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_manifest(manifest_file)
    order = vasp_element_order(atoms)
    validate_vasp_manifest_elements(manifest_file, [order])
    records = []
    combined = hashlib.sha256()
    family = str(manifest["family"])
    for symbol in order:
        expected = manifest["elements"][symbol]
        potcar = root / expected["path"]
        if not potcar.is_file():
            raise VaspResourceError(
                f"VASP resource for {symbol} does not exist: {potcar}"
            )
        actual_sha256 = file_sha256(potcar)
        if actual_sha256 != expected["sha256"]:
            raise VaspResourceError(
                f"VASP resource hash mismatch for {symbol}: {potcar}"
            )
        titel, actual_symbol = _potcar_identity(potcar)
        if actual_symbol != symbol:
            raise VaspResourceError(
                f"POTCAR order/identity mismatch: requested {symbol}, "
                f"but {potcar} declares {actual_symbol}"
            )
        if titel != expected["titel"]:
            raise VaspResourceError(
                f"POTCAR version mismatch for {symbol}: expected TITEL "
                f"{expected['titel']!r}, got {titel!r}"
            )
        if family not in titel:
            raise VaspResourceError(
                f"POTCAR family mismatch for {symbol}: {titel!r} does not "
                f"declare {family!r}"
            )
        with potcar.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                combined.update(block)
        records.append(
            {
                "element": symbol,
                "path": expected["path"],
                "sha256": actual_sha256,
                "titel": titel,
                "setup_suffix": potcar.parent.name[len(symbol) :],
            }
        )
    return {
        "protocol": _PROTOCOL,
        "manifest_sha256": file_sha256(manifest_file),
        "family": family,
        "release": manifest["release"],
        "element_order": list(order),
        "potcars": records,
        "ase_setups": {
            record["element"]: record["setup_suffix"]
            for record in records
        },
        "combined_sha256": combined.hexdigest(),
    }


__all__ = [
    "VaspResourceError",
    "validate_vasp_manifest_elements",
    "validate_vasp_resources",
    "vasp_resource_files",
    "vasp_element_order",
]
