"""Internal torchrun entry point; never execute a workflow stage on each rank."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) != 2 or int(os.environ.get("WORLD_SIZE", "1")) < 2:
        raise RuntimeError("This worker must be launched by NepTrain through torchrun.")
    import torch
    from torchnep import train_nep_sharded

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError("Distributed TorchNEP training requires one visible GPU per rank.")
    torch.cuda.set_device(local_rank)
    options = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    train_nep_sharded(**options)


if __name__ == "__main__":
    main()
