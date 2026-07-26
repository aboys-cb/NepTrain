from pathlib import Path
import sys
import types

from ase import Atoms
from ase.io import write

import pytest

from NepTrain.core.training import TrainingError, TrainingRequest, train


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
        (output / "loss.out").write_text("loss\n", encoding="utf-8")
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
