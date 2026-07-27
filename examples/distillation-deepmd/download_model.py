from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess


MODEL_NAME = "DPA-3.2-5M"
MODEL_FILE = "DPA-3.2-5M.pt"


def main() -> None:
    root = Path(__file__).parent
    cache = root / ".teacher-cache"
    subprocess.run(
        [
            "dp",
            "pretrained",
            "download",
            MODEL_NAME,
            "--cache-dir",
            str(cache),
        ],
        check=True,
    )
    matches = list(cache.rglob(MODEL_FILE))
    if len(matches) != 1:
        raise SystemExit(
            f"expected one downloaded {MODEL_FILE} under {cache}, "
            f"found {len(matches)}"
        )
    source = matches[0]
    output = root / MODEL_FILE
    shutil.copy2(source, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Saved {output.name}")
    print(f"SHA256 {digest}")


if __name__ == "__main__":
    main()
