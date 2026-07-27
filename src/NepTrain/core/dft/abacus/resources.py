"""Content-addressed ABACUS pseudopotential and orbital resources."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping

from ase import Atoms

from ...content_addressing import file_sha256


class AbacusResourceError(RuntimeError):
    """Raised when ABACUS resources do not match their pinned manifest."""


_PROTOCOL = "neptrain.abacus-resources.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _resource_record(
    value: Any,
    *,
    element: str,
    kind: str,
) -> dict[str, str] | None:
    if value is None and kind == "orbital":
        return None
    if not isinstance(value, Mapping):
        raise AbacusResourceError(
            f"ABACUS manifest {element}.{kind} must be an object"
        )
    if set(value) != {"path", "sha256"}:
        raise AbacusResourceError(
            f"ABACUS manifest {element}.{kind} must define only path and sha256"
        )
    relative = str(value["path"])
    pure = PurePosixPath(relative)
    expected_suffix = ".upf" if kind == "pseudopotential" else ".orb"
    if (
        not relative
        or pure.is_absolute()
        or ".." in pure.parts
        or pure.suffix.lower() != expected_suffix
    ):
        raise AbacusResourceError(
            f"ABACUS manifest {element}.{kind}.path must be a relative "
            f"{expected_suffix} file"
        )
    sha256 = str(value["sha256"]).lower()
    if _SHA256.fullmatch(sha256) is None:
        raise AbacusResourceError(
            f"ABACUS manifest {element}.{kind} requires a SHA256"
        )
    return {"path": pure.as_posix(), "sha256": sha256}


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AbacusResourceError(
            f"cannot read ABACUS resource manifest {path}: {error}"
        ) from error
    if not isinstance(raw, Mapping) or raw.get("protocol") != _PROTOCOL:
        raise AbacusResourceError(
            f"ABACUS resource manifest {path} must use protocol {_PROTOCOL}"
        )
    release = raw.get("release")
    if not isinstance(release, str) or not release.strip():
        raise AbacusResourceError(
            f"ABACUS resource manifest {path} requires a release"
        )
    elements = raw.get("elements")
    if not isinstance(elements, Mapping) or not elements:
        raise AbacusResourceError(
            f"ABACUS resource manifest {path} requires an elements mapping"
        )
    normalized = {}
    for element, value in elements.items():
        if (
            not isinstance(element, str)
            or re.fullmatch(r"[A-Z][a-z]?", element) is None
            or not isinstance(value, Mapping)
        ):
            raise AbacusResourceError(
                f"invalid ABACUS resource element record: {element!r}"
            )
        if set(value) - {"pseudopotential", "orbital"}:
            raise AbacusResourceError(
                f"ABACUS resource element {element} has unknown fields"
            )
        normalized[element] = {
            "pseudopotential": _resource_record(
                value.get("pseudopotential"),
                element=element,
                kind="pseudopotential",
            ),
            "orbital": _resource_record(
                value.get("orbital"),
                element=element,
                kind="orbital",
            ),
        }
    return {
        "protocol": _PROTOCOL,
        "release": release,
        "elements": normalized,
    }


def validate_abacus_manifest_elements(
    manifest_path: str | Path,
    element_orders: Iterable[Iterable[str]],
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = _read_manifest(path)
    required = {
        str(element)
        for order in element_orders
        for element in order
    }
    missing = sorted(required - set(manifest["elements"]))
    if missing:
        raise AbacusResourceError(
            "ABACUS resource manifest is missing elements: " + ", ".join(missing)
        )
    return {
        "protocol": _PROTOCOL,
        "manifest_sha256": file_sha256(path),
        "release": manifest["release"],
        "required_elements": sorted(required),
    }


def abacus_resource_files(
    manifest_path: str | Path,
) -> tuple[dict[str, str], ...]:
    """Return every pinned file for local or remote doctor probes."""

    manifest = _read_manifest(Path(manifest_path).expanduser().resolve())
    records = []
    for element, resources in sorted(manifest["elements"].items()):
        for kind in ("pseudopotential", "orbital"):
            record = resources.get(kind)
            if record is not None:
                records.append(
                    {
                        "element": element,
                        "kind": kind,
                        "path": record["path"],
                        "sha256": record["sha256"],
                    }
                )
    return tuple(records)


def _declared_element(path: Path, *, orbital: bool) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as error:
        raise AbacusResourceError(f"cannot read ABACUS resource {path}") from error
    pattern = (
        r"\bElement\s+([A-Z][a-z]?)\b"
        if orbital
        else r"\belement\s*=\s*[\"']([A-Z][a-z]?)[\"']"
    )
    match = re.search(pattern, text, flags=re.IGNORECASE if not orbital else 0)
    if match is None:
        raise AbacusResourceError(
            f"ABACUS resource {path} does not declare its element"
        )
    value = match.group(1)
    return value[0].upper() + value[1:].lower()


def validate_abacus_resources(
    resource_root: str | Path,
    manifest_path: str | Path,
    atoms: Atoms,
    *,
    require_orbitals: bool,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    root = Path(resource_root).expanduser()
    if not root.is_dir():
        raise AbacusResourceError(
            f"ABACUS resource directory does not exist: {root}"
        )
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_manifest(manifest_file)
    elements = tuple(dict.fromkeys(atoms.get_chemical_symbols()))
    validate_abacus_manifest_elements(manifest_file, [elements])
    pp_files: dict[str, str] = {}
    orb_files: dict[str, str] = {}
    records = {}
    for element in elements:
        expected = manifest["elements"][element]
        kinds = [("pseudopotential", False)]
        if require_orbitals:
            kinds.append(("orbital", True))
        record = {}
        for kind, orbital in kinds:
            file_record = expected.get(kind)
            if file_record is None:
                raise AbacusResourceError(
                    f"ABACUS resource manifest is missing {kind} for {element}"
                )
            resource = root / file_record["path"]
            if not resource.is_file():
                raise AbacusResourceError(
                    f"ABACUS {kind} for {element} does not exist: {resource}"
                )
            if file_sha256(resource) != file_record["sha256"]:
                raise AbacusResourceError(
                    f"ABACUS {kind} hash mismatch for {element}: {resource}"
                )
            declared = _declared_element(resource, orbital=orbital)
            if declared != element:
                raise AbacusResourceError(
                    f"ABACUS {kind} identity mismatch: requested {element}, "
                    f"but {resource} declares {declared}"
                )
            record[kind] = dict(file_record)
            if orbital:
                orb_files[element] = file_record["path"]
            else:
                pp_files[element] = file_record["path"]
        records[element] = record
    provenance = {
        "protocol": _PROTOCOL,
        "manifest_sha256": file_sha256(manifest_file),
        "release": manifest["release"],
        "element_order": list(elements),
        "resources": records,
    }
    return provenance, pp_files, orb_files


__all__ = [
    "AbacusResourceError",
    "abacus_resource_files",
    "validate_abacus_manifest_elements",
    "validate_abacus_resources",
]
