"""Training seam with GPUMD and TorchNEP adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil

from ase.io import read as ase_read
import numpy as np

from .spin import validate_spin_dataset


class TrainingError(RuntimeError):
    """Raised when training fails or does not produce its promised artifacts."""


@dataclass(frozen=True)
class TrainingRequest:
    config_file: Path
    train_file: Path
    output_dir: Path
    test_file: Path | None = None
    restart_file: Path | None = None
    finetune_file: Path | None = None
    continue_steps: int = 10_000
    device: str = "cuda"
    torch_backend: str = "auto"
    precision: str = "float32"
    use_compile: bool = False
    seed: int = 20260723


@dataclass(frozen=True)
class TrainingResult:
    backend: str
    best_model: Path
    final_model: Path | None
    checkpoint: Path | None


def _validate_training_data(path: Path) -> bool:
    frames = ase_read(path, index=":", format="extxyz")
    _, spin_frames = validate_spin_dataset(frames, require_mforce=True)
    return spin_frames > 0


def _validate_adapter_model(path: Path, *, expects_spin: bool) -> None:
    try:
        import nep_adapters
    except ImportError as error:  # pragma: no cover - required dependency path
        raise TrainingError(
            "NEPAdapters is required to validate the trained model"
        ) from error
    try:
        info = nep_adapters.inspect_model(path)
    except Exception as error:
        raise TrainingError(
            "training produced a model that NEPAdapters cannot load; "
            "check that the trainer's model format and spin descriptor match "
            "the installed NEPAdapters runtime"
        ) from error
    if info.supports("spin") != expects_spin:
        raise TrainingError(
            "trained model spin capability does not match the training dataset"
        )


def _train_gpumd(request: TrainingRequest) -> TrainingResult:
    if request.finetune_file is not None:
        raise TrainingError("GPUMD training does not support finetune_file")
    from .nep.io import RunInput

    run = RunInput(str(request.train_file), str(request.config_file), str(request.test_file) if request.test_file else None)
    run.set_restart(
        str(request.restart_file) if request.restart_file else None,
        request.continue_steps,
    )
    run.calculate(str(request.output_dir))
    model = request.output_dir / "nep.txt"
    if not model.is_file():
        raise TrainingError("GPUMD training completed without nep.txt")
    return TrainingResult(
        backend="gpumd",
        best_model=model,
        final_model=model,
        checkpoint=(request.output_dir / "nep.restart") if (request.output_dir / "nep.restart").is_file() else None,
    )


def _train_torchnep(request: TrainingRequest) -> TrainingResult:
    if request.restart_file is not None and request.finetune_file is not None:
        raise TrainingError("TorchNEP restart_file and finetune_file are mutually exclusive")
    if request.test_file is not None and request.test_file.is_file():
        raise TrainingError(
            "TorchNEP currently owns its split through nep.in; remove training.test_path"
        )
    try:
        from torchnep import train_nep
    except ImportError as error:  # pragma: no cover - installation failure path
        raise TrainingError(
            "TorchNEP backend requested but torchnep is not installed"
        ) from error
    import torch

    random.seed(request.seed)
    np.random.seed(request.seed % (2**32))
    torch.manual_seed(request.seed)
    torch.cuda.manual_seed_all(request.seed)

    request.output_dir.mkdir(parents=True, exist_ok=True)
    train_nep(
        config_file=str(request.config_file),
        data_file=str(request.train_file),
        output_dir=str(request.output_dir),
        device=request.device,
        precision=request.precision,
        backend=request.torch_backend,
        use_compile=request.use_compile,
        restart=True,
        finetune_from=str(request.finetune_file) if request.finetune_file else None,
        resume_from=str(request.restart_file) if request.restart_file else None,
    )
    best = request.output_dir / "nep_best.txt"
    if not best.is_file():
        raise TrainingError("TorchNEP training completed without nep_best.txt")
    canonical = request.output_dir / "nep.txt"
    shutil.copy2(best, canonical)
    final = request.output_dir / "nep_final.txt"
    checkpoint = request.output_dir / "checkpoint.pt"
    return TrainingResult(
        backend="torchnep",
        best_model=canonical,
        final_model=final if final.is_file() else None,
        checkpoint=checkpoint if checkpoint.is_file() else None,
    )


def train(request: TrainingRequest, backend: str) -> TrainingResult:
    """Run the selected Adapter through the stable training Interface."""

    if not request.train_file.is_file():
        raise TrainingError(f"training data does not exist: {request.train_file}")
    if not request.config_file.is_file():
        raise TrainingError(f"training config does not exist: {request.config_file}")
    expects_spin = _validate_training_data(request.train_file)
    if backend == "gpumd":
        result = _train_gpumd(request)
    elif backend == "torchnep":
        result = _train_torchnep(request)
    else:
        raise TrainingError("training backend must be gpumd or torchnep")
    _validate_adapter_model(result.best_model, expects_spin=expects_spin)
    return result
