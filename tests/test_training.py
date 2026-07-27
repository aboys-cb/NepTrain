from pathlib import Path
import subprocess
import sys
import types

from ase import Atoms
from ase.io import write

import pytest

from NepTrain.core.training import TrainingError, TrainingRequest, train


def _ordinary_model_adapter():
    return types.SimpleNamespace(
        inspect_model=lambda _path: types.SimpleNamespace(
            supports=lambda capability: capability != "spin"
        )
    )


def test_gpumd_training_adapter_prepares_inputs_and_collects_outputs(
    tmp_path: Path, monkeypatch
):
    train_file = tmp_path / "source.xyz"
    write(
        train_file,
        Atoms(
            "FeO",
            positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
            cell=[4, 4, 4],
            pbc=True,
        ),
        format="extxyz",
    )
    test_file = tmp_path / "test-source.xyz"
    write(
        test_file,
        Atoms("Fe", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True),
        format="extxyz",
    )
    config_file = tmp_path / "source.in"
    config_file.write_text("cutoff 6 4\n", encoding="utf-8")
    captured = {}

    def fake_run(command, *, stdout, stderr, cwd, check):
        del stderr, check
        directory = Path(cwd)
        captured["command"] = command
        captured["config"] = (directory / "nep.in").read_text(
            encoding="utf-8"
        )
        assert (directory / "train.xyz").is_file()
        assert (directory / "test.xyz").is_file()
        stdout.write("training complete\n")
        stdout.flush()
        (directory / "nep.txt").write_text("nep4 2 O Fe\n", encoding="utf-8")
        (directory / "nep.restart").write_bytes(b"checkpoint")
        (directory / "loss.out").write_text(
            "0 4 0.1 0.2 2 3 4 2.5 3.5 4.5\n"
            "10 2 0.1 0.2 1 2 3 1.5 2.5 3.5\n",
            encoding="utf-8",
        )
        (directory / "output.log").write_text(
            "trainer log\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv("NEPTRAIN_NEP_COMMAND", "srun nep")
    monkeypatch.setattr("NepTrain.core.training.subprocess.run", fake_run)
    monkeypatch.setitem(sys.modules, "nep_adapters", _ordinary_model_adapter())
    output = tmp_path / "output"

    result = train(
        TrainingRequest(
            config_file=config_file,
            train_file=train_file,
            test_file=test_file,
            output_dir=output,
        ),
        "gpumd",
    )

    assert captured["command"] == ["srun", "nep"]
    assert "type 2 O Fe\n" in captured["config"]
    assert "generation 100000\n" in captured["config"]
    assert result.best_model == output / "nep.txt"
    assert result.checkpoint == output / "nep.restart"
    assert set(result.outputs) == {
        "loss.out",
        "nep.out",
        "output.log",
        "training-convergence.png",
        "training-report.json",
    }


def test_gpumd_restart_overrides_only_restart_owned_config(
    tmp_path: Path, monkeypatch
):
    train_file = tmp_path / "train.xyz"
    write(
        train_file,
        Atoms("Fe", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True),
        format="extxyz",
    )
    config_file = tmp_path / "nep.in"
    config_file.write_text(
        "type 1 Fe\n"
        "generation 100000 # original duration\n"
        "lambda_1 0.1 # regularization\n",
        encoding="utf-8",
    )
    restart = tmp_path / "previous.restart"
    restart.write_bytes(b"previous checkpoint")
    captured = {}

    def fake_run(command, *, stdout, stderr, cwd, check):
        del stdout, stderr, check
        directory = Path(cwd)
        captured["config"] = (directory / "nep.in").read_text(
            encoding="utf-8"
        )
        captured["restart"] = (directory / "nep.restart").read_bytes()
        (directory / "nep.txt").write_text("nep4 1 Fe\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("NepTrain.core.training.subprocess.run", fake_run)
    monkeypatch.setitem(sys.modules, "nep_adapters", _ordinary_model_adapter())

    train(
        TrainingRequest(
            config_file=config_file,
            train_file=train_file,
            output_dir=tmp_path / "output",
            restart_file=restart,
            continue_steps=2500,
        ),
        "gpumd",
    )

    assert "generation 2500 # original duration" in captured["config"]
    assert "lambda_1 0 # regularization" in captured["config"]
    assert captured["restart"] == b"previous checkpoint"


def test_gpumd_training_failure_reports_process_detail_and_rejects_stale_model(
    tmp_path: Path, monkeypatch
):
    train_file = tmp_path / "train.xyz"
    write(
        train_file,
        Atoms("Fe", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True),
        format="extxyz",
    )
    config_file = tmp_path / "nep.in"
    config_file.write_text("type 1 Fe\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    stale = output / "nep.txt"
    stale.write_text("stale model\n", encoding="utf-8")

    def fake_run(command, *, stdout, stderr, cwd, check):
        del stdout, cwd, check
        stderr.write("ERROR: invalid training input\n")
        stderr.flush()
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr("NepTrain.core.training.subprocess.run", fake_run)

    with pytest.raises(
        TrainingError,
        match="exit code 2: ERROR: invalid training input",
    ):
        train(
            TrainingRequest(
                config_file=config_file,
                train_file=train_file,
                output_dir=output,
            ),
            "gpumd",
        )

    assert not stale.exists()


def test_gpumd_training_rejects_missing_restart_before_launch(
    tmp_path: Path, monkeypatch
):
    train_file = tmp_path / "train.xyz"
    write(
        train_file,
        Atoms("Fe", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True),
        format="extxyz",
    )
    config_file = tmp_path / "nep.in"
    config_file.write_text("type 1 Fe\n", encoding="utf-8")
    launched = []
    monkeypatch.setattr(
        "NepTrain.core.training.subprocess.run",
        lambda *args, **kwargs: launched.append((args, kwargs)),
    )

    with pytest.raises(TrainingError, match="restart does not exist"):
        train(
            TrainingRequest(
                config_file=config_file,
                train_file=train_file,
                output_dir=tmp_path / "output",
                restart_file=tmp_path / "missing.restart",
            ),
            "gpumd",
        )

    assert launched == []


def test_torchnep_best_model_becomes_canonical_nep_txt(tmp_path: Path, monkeypatch):
    train_file = tmp_path / "train.xyz"
    write(train_file, Atoms("Fe", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True))
    config_file = tmp_path / "nep.in"
    config_file.write_text("type 1 Fe\n", encoding="utf-8")
    finetune = tmp_path / "previous.pt"
    finetune.write_bytes(b"previous")
    captured = {}
    seeds = []

    def fake_train_nep(**kwargs):
        captured.update(kwargs)
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "nep_best.txt").write_text("nep4 1 Fe\n", encoding="utf-8")
        (output / "nep_final.txt").write_text("nep4 1 Fe\n", encoding="utf-8")
        (output / "checkpoint.pt").write_bytes(b"checkpoint")
        (output / "checkpoint_stage1.pt").write_bytes(b"stage 1")
        (output / "loss.out").write_text(
            "0 4 0.1 0.2 2 3 4 2.5 3.5 4.5\n"
            "10 2 0.1 0.2 1 2 3 1.5 2.5 3.5\n",
            encoding="utf-8",
        )
        (output / "force_train.out").write_text("forces\n", encoding="utf-8")
        (output / "output.log").write_text("training log\n", encoding="utf-8")

    monkeypatch.setitem(sys.modules, "torchnep", types.SimpleNamespace(train_nep=fake_train_nep))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            manual_seed=lambda value: seeds.append(("cpu", value)),
            cuda=types.SimpleNamespace(
                manual_seed_all=lambda value: seeds.append(("cuda", value))
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "nep_adapters",
        types.SimpleNamespace(
            inspect_model=lambda _path: types.SimpleNamespace(
                supports=lambda capability: capability != "spin"
            )
        ),
    )
    output = tmp_path / "out"
    result = train(
        TrainingRequest(
            config_file=config_file,
            train_file=train_file,
            output_dir=output,
            finetune_file=finetune,
        ),
        "torchnep",
    )
    assert result.best_model == output / "nep.txt"
    assert result.best_model.read_text(encoding="utf-8") == "nep4 1 Fe\n"
    assert result.checkpoint == output / "checkpoint.pt"
    assert set(result.outputs) == {
        "checkpoint_stage1.pt",
        "force_train.out",
        "loss.out",
        "output.log",
        "training-convergence.png",
        "training-report.json",
    }
    assert captured["finetune_from"] == str(finetune)
    assert captured["resume_from"] is None
    assert seeds == [("cpu", 20260723), ("cuda", 20260723)]


def test_training_rejects_model_that_nepadapters_cannot_load(tmp_path: Path, monkeypatch):
    train_file = tmp_path / "train.xyz"
    write(train_file, Atoms("Fe", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True))
    config_file = tmp_path / "nep.in"
    config_file.write_text("type 1 Fe\n", encoding="utf-8")

    def fake_train_nep(**kwargs):
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "nep_best.txt").write_text("incompatible\n", encoding="utf-8")

    def reject_model(_path):
        raise RuntimeError("unsupported descriptor")

    monkeypatch.setitem(sys.modules, "torchnep", types.SimpleNamespace(train_nep=fake_train_nep))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            manual_seed=lambda _value: None,
            cuda=types.SimpleNamespace(manual_seed_all=lambda _value: None),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "nep_adapters",
        types.SimpleNamespace(inspect_model=reject_model),
    )

    with pytest.raises(TrainingError, match="NEPAdapters cannot load"):
        train(
            TrainingRequest(
                config_file=config_file,
                train_file=train_file,
                output_dir=tmp_path / "out",
            ),
            "torchnep",
        )
