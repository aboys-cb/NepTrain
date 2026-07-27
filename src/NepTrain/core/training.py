"""Training seam with GPUMD and TorchNEP adapters."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
from typing import Mapping

from ase.data import atomic_numbers
from ase.io import read as ase_read
import numpy as np

from .reporting import build_training_report
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
    outputs: Mapping[str, Path] = field(default_factory=dict)


def _validate_training_data(path: Path) -> tuple[bool, tuple[str, ...]]:
    frames = ase_read(path, index=":", format="extxyz")
    _, spin_frames = validate_spin_dataset(frames, require_mforce=True)
    symbols = {
        symbol
        for frame in frames
        for symbol in frame.get_chemical_symbols()
    }
    if not symbols:
        raise TrainingError("training data contains no atoms")
    elements = tuple(sorted(symbols, key=atomic_numbers.__getitem__))
    return spin_frames > 0, elements


def _replace_gpumd_keyword(
    lines: list[str],
    keyword: str,
    value: str,
) -> None:
    pattern = re.compile(
        rf"^([ \t]*{re.escape(keyword)}[ \t]+)"
        r"([^#\r\n]*?)"
        r"([ \t]*(?:#.*)?)"
        r"(\r?\n)?$"
    )
    matches = [
        (index, pattern.match(line))
        for index, line in enumerate(lines)
        if pattern.match(line) is not None
    ]
    if len(matches) > 1:
        raise TrainingError(
            f"GPUMD training config defines {keyword!r} more than once"
        )
    if not matches:
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += "\n"
        lines.append(f"{keyword} {value}\n")
        return
    index, match = matches[0]
    assert match is not None
    newline = match.group(4) or ""
    lines[index] = f"{match.group(1)}{value}{match.group(3)}{newline}"


def _prepare_gpumd_inputs(
    request: TrainingRequest,
    elements: tuple[str, ...],
) -> None:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    lines = request.config_file.read_text(encoding="utf-8").splitlines(
        keepends=True
    )
    if not any(
        re.match(r"^[ \t]*type[ \t]+", line)
        for line in lines
    ):
        _replace_gpumd_keyword(
            lines,
            "type",
            f"{len(elements)} {' '.join(elements)}",
        )
    if not any(
        re.match(r"^[ \t]*generation[ \t]+", line)
        for line in lines
    ):
        _replace_gpumd_keyword(lines, "generation", "100000")
    if request.restart_file is not None:
        _replace_gpumd_keyword(
            lines, "generation", str(request.continue_steps)
        )
        _replace_gpumd_keyword(lines, "lambda_1", "0")
    (request.output_dir / "nep.in").write_text(
        "".join(lines), encoding="utf-8"
    )

    sources = [(request.train_file, request.output_dir / "train.xyz")]
    if request.test_file is not None:
        sources.append((request.test_file, request.output_dir / "test.xyz"))
    if request.restart_file is not None:
        sources.append(
            (request.restart_file, request.output_dir / "nep.restart")
        )
    for source, target in sources:
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)


def _process_failure_reason(
    returncode: int,
    stderr: Path,
    stdout: Path,
) -> str:
    lines: list[str] = []
    for path in (stderr, stdout):
        if path.is_file():
            lines.extend(
                line.strip()
                for line in path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if line.strip()
            )
    detail = lines[-1] if lines else "no output detail"
    return f"exit code {returncode}: {detail}"


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


def _train_gpumd(
    request: TrainingRequest,
    elements: tuple[str, ...],
) -> TrainingResult:
    if request.finetune_file is not None:
        raise TrainingError("GPUMD training does not support finetune_file")
    if request.test_file is not None and not request.test_file.is_file():
        raise TrainingError(f"test data does not exist: {request.test_file}")
    if request.restart_file is not None and not request.restart_file.is_file():
        raise TrainingError(
            f"GPUMD restart does not exist: {request.restart_file}"
        )
    if request.continue_steps <= 0:
        raise TrainingError("continue_steps must be positive")
    _prepare_gpumd_inputs(request, elements)
    model = request.output_dir / "nep.txt"
    model.unlink(missing_ok=True)
    stdout = request.output_dir / "nep.out"
    stderr = request.output_dir / "nep.err"
    command = shlex.split(os.environ.get("NEPTRAIN_NEP_COMMAND", "nep"))
    if not command:
        raise TrainingError("NEPTRAIN_NEP_COMMAND must not be empty")
    try:
        with stdout.open("w", encoding="utf-8") as stdout_handle, stderr.open(
            "w", encoding="utf-8", buffering=1
        ) as stderr_handle:
            completed = subprocess.run(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                cwd=request.output_dir,
                check=False,
            )
    except OSError as error:
        raise TrainingError(
            f"could not start GPUMD NEP trainer {command[0]!r}: {error}"
        ) from error
    if completed.returncode != 0:
        raise TrainingError(
            "GPUMD NEP training failed with "
            + _process_failure_reason(
                completed.returncode, stderr, stdout
            )
        )
    if not model.is_file():
        raise TrainingError("GPUMD training completed without nep.txt")
    outputs = {
        path.name: path
        for path in sorted(request.output_dir.glob("*.out"))
        if path.is_file()
    }
    output_log = request.output_dir / "output.log"
    if output_log.is_file():
        outputs[output_log.name] = output_log
    return TrainingResult(
        backend="gpumd",
        best_model=model,
        final_model=model,
        checkpoint=(request.output_dir / "nep.restart") if (request.output_dir / "nep.restart").is_file() else None,
        outputs=outputs,
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
    outputs = {
        path.name: path
        for path in sorted(request.output_dir.glob("*.out"))
        if path.is_file()
    }
    for name in ("output.log", "checkpoint_stage1.pt"):
        path = request.output_dir / name
        if path.is_file():
            outputs[name] = path
    return TrainingResult(
        backend="torchnep",
        best_model=canonical,
        final_model=final if final.is_file() else None,
        checkpoint=checkpoint if checkpoint.is_file() else None,
        outputs=outputs,
    )


def train(request: TrainingRequest, backend: str) -> TrainingResult:
    """Run the selected Adapter through the stable training Interface."""

    if not request.train_file.is_file():
        raise TrainingError(f"training data does not exist: {request.train_file}")
    if not request.config_file.is_file():
        raise TrainingError(f"training config does not exist: {request.config_file}")
    expects_spin, elements = _validate_training_data(request.train_file)
    if backend == "gpumd":
        result = _train_gpumd(request, elements)
    elif backend == "torchnep":
        result = _train_torchnep(request)
    else:
        raise TrainingError("training backend must be gpumd or torchnep")
    _validate_adapter_model(result.best_model, expects_spin=expects_spin)
    report = build_training_report(
        request.output_dir,
        backend=result.backend,
        loss_path=result.outputs.get("loss.out"),
    )
    outputs = dict(result.outputs)
    outputs[report.report.name] = report.report
    if report.chart is not None:
        outputs[report.chart.name] = report.chart
    return replace(result, outputs=outputs)
