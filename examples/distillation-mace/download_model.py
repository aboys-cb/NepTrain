from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen


URL = (
    "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/"
    "2023-12-10-mace-128-L0_energy_epoch-249.model"
)
OUTPUT = Path(__file__).with_name("mace-mp-0-small.model")
SHA256 = "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.is_file():
        actual = _sha256(OUTPUT)
        if actual != SHA256:
            raise SystemExit(
                f"existing model SHA256 mismatch: expected {SHA256}, got {actual}"
            )
        print(f"Using existing {OUTPUT.name} ({actual})")
        return
    temporary = OUTPUT.with_suffix(".model.part")
    digest = hashlib.sha256()
    with urlopen(URL, timeout=60) as response, temporary.open("wb") as handle:
        while block := response.read(1024 * 1024):
            handle.write(block)
            digest.update(block)
    actual = digest.hexdigest()
    if actual != SHA256:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"model SHA256 mismatch: expected {SHA256}, got {actual}"
        )
    temporary.replace(OUTPUT)
    print(f"Downloaded {OUTPUT.name} ({actual})")


if __name__ == "__main__":
    main()
