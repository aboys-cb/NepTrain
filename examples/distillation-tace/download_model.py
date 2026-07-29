from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen


REVISION = "aa4a36a9f6a4d68d3090c0b568a7d42b8b4ef7d9"
URL = (
    "https://huggingface.co/xvzemin/tace-foundations/resolve/"
    f"{REVISION}/TACE-OAM-7M.pt"
)
OUTPUT = Path(__file__).with_name("TACE-OAM-7M.pt")
SHA256 = "87f6b23c40a5b29c3b0d2520bca104d73c653866e82541919db94979c7f6511c"


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
    temporary = OUTPUT.with_suffix(".pt.part")
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
