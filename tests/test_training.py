import json
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
        (directory / "energy_train.out").write_text("1.02 1.0\n1.52 1.5\n")
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
        "training-parity-train-report.json",
        "energy_train.out",
        "training-parity-train.png",
    }
    parity = json.loads(result.outputs["training-parity-train-report.json"].read_text())
    assert parity["panels"]["energy"]["rmse"] == pytest.approx(0.02)
    assert (
        result.outputs["training-parity-train.png"].read_bytes().startswith(b"\x89PNG")
    )


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
                device_count=lambda: 1,
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
        "training-parity-train-report.json",
    }
    assert captured["finetune_from"] == str(finetune)
    assert captured["resume_from"] is None
    assert captured["backend"] == "auto"
    assert seeds == [("cpu", 20260723), ("cuda", 20260723)]


@pytest.mark.parametrize("backend", ["auto", "bmm"])
def test_torchnep_current_api_receives_run_seed_without_legacy_backend(
    tmp_path, monkeypatch, caplog, backend
):
    from NepTrain.core.training import _train_torchnep

    received = {}

    def current_train_nep(
        config_file, data_file, output_dir, device, precision, use_compile,
        restart, finetune_from, resume_from, run_seed=None,
    ):
        received["seed"] = run_seed
        received["resume"] = resume_from
        (Path(output_dir) / "nep_best.txt").write_text("nep4 1 Al\n")

    monkeypatch.setitem(sys.modules, "torchnep", types.SimpleNamespace(train_nep=current_train_nep))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        manual_seed=lambda _: None,
        cuda=types.SimpleNamespace(device_count=lambda: 1, manual_seed_all=lambda _: None),
    ))
    request = TrainingRequest(
        config_file=tmp_path / "nep.in", train_file=tmp_path / "train.xyz",
        output_dir=tmp_path / "out", torch_backend=backend, seed=1234,
        restart_file=tmp_path / "checkpoint.pt",
    )
    result = _train_torchnep(request)
    assert received == {"seed": 1234, "resume": str(request.restart_file)}
    assert result.best_model.read_text() == "nep4 1 Al\n"
    assert ("selects its execution backend automatically" in caplog.text) == (backend != "auto")


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
            cuda=types.SimpleNamespace(device_count=lambda: 1, manual_seed_all=lambda _value: None),
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


@pytest.mark.parametrize("warm_start", [None, "restart_file", "finetune_file"])
def test_torchnep_multigpu_launches_only_trainer_and_collects_once(tmp_path, monkeypatch, warm_start):
    from NepTrain.core.training import _train_torchnep

    def single(config_file, data_file, output_dir, device, precision, use_compile,
               restart, finetune_from, resume_from, run_seed=None,
               use_gpumd_qscaler=False, print_interval=1, prediction_interval=100):
        pytest.fail("Multi-GPU training must not call the single-device trainer")

    def sharded(config_file, data_file, output_dir, precision, use_compile,
                restart, finetune_from, resume_from, run_seed=None,
                use_gpumd_qscaler=True, print_interval=10, prediction_interval=20):
        pytest.fail("The stage worker must not itself become a DDP rank")

    monkeypatch.setitem(sys.modules, "torchnep", types.SimpleNamespace(
        train_nep=single, train_nep_sharded=sharded,
    ))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        manual_seed=lambda _: None,
        cuda=types.SimpleNamespace(device_count=lambda: 4, manual_seed_all=lambda _: None),
    ))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,4,6,7")
    calls = []
    output = tmp_path / "training output"

    def launch(command, *, check):
        calls.append(command)
        assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
        assert "--nproc-per-node=4" in command
        assert "--max-restarts=0" in command
        assert command[-2] == "NepTrain.core.torchnep_worker"
        assert "stage-worker" not in command
        assert not check
        options = json.loads(Path(command[-1]).read_text())
        assert "device" not in options
        assert "backend" not in options  # Current TorchNEP selects this internally.
        assert options["run_seed"] == 17
        assert options["use_gpumd_qscaler"] is False
        assert options["print_interval"] == 1
        assert options["prediction_interval"] == 100
        assert options["resume_from"] == (str(tmp_path / "previous.pt") if warm_start == "restart_file" else None)
        assert options["finetune_from"] == (str(tmp_path / "previous.pt") if warm_start == "finetune_file" else None)
        assert __import__('os').environ['CUDA_VISIBLE_DEVICES'] == '2,4,6,7'
        assert not (output / "nep.txt").exists()
        (output / "nep_best.txt").write_text("best model")
        (output / "nep_final.txt").write_text("final model")
        (output / "checkpoint.pt").write_bytes(b"checkpoint")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", launch)
    request = TrainingRequest(tmp_path / "nep.in", tmp_path / "train.xyz", output,
                              seed=17, **({warm_start: tmp_path / "previous.pt"} if warm_start else {}))
    result = _train_torchnep(request)
    assert len(calls) == 1
    assert result.best_model.read_text() == "best model"
    assert result.final_model.read_text() == "final model"
    assert result.checkpoint.read_bytes() == b"checkpoint"
    assert not Path(calls[0][-1]).exists()


def test_torchnep_rank_failure_does_not_publish_partial_model(tmp_path, monkeypatch):
    from NepTrain.core.training import _train_torchnep

    monkeypatch.setitem(sys.modules, "torchnep", types.SimpleNamespace(
        train_nep=lambda **kw: None, train_nep_sharded=lambda **kw: None,
    ))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        manual_seed=lambda _: None,
        cuda=types.SimpleNamespace(device_count=lambda: 2, manual_seed_all=lambda _: None),
    ))
    def fail(command, **kwargs):
        options = json.loads(Path(command[-1]).read_text())
        (Path(options['output_dir']) / 'nep_best.txt').write_text('partial')
        return subprocess.CompletedProcess(command, 1)
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(TrainingError, match='distributed training failed'):
        _train_torchnep(TrainingRequest(tmp_path / 'nep.in', tmp_path / 'train.xyz', tmp_path / 'out'))
    assert not (tmp_path / 'out/nep.txt').exists()
    assert not list((tmp_path / 'out').glob('.torchnep-ddp-*'))


@pytest.mark.parametrize('device,count', [('cuda', 1), ('cuda:2', 4), ('cpu', 4), ('mps', 4)])
def test_torchnep_single_device_selection(tmp_path, monkeypatch, device, count):
    from NepTrain.core.training import _train_torchnep

    received = []
    def single(**kw):
        received.append(kw['device'])
        (Path(kw['output_dir']) / 'nep_best.txt').write_text('best')
    monkeypatch.setitem(sys.modules, 'torchnep', types.SimpleNamespace(train_nep=single))
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(
        manual_seed=lambda _: None,
        cuda=types.SimpleNamespace(device_count=lambda: count, manual_seed_all=lambda _: None),
    ))
    _train_torchnep(TrainingRequest(tmp_path / 'nep.in', tmp_path / 'train.xyz', tmp_path / 'out', device=device))
    assert received == [device]


@pytest.mark.parametrize('environment,error', [
    ({'WORLD_SIZE': '4'}, 'without torchrun'),
    ({'SLURM_JOB_NUM_NODES': '2'}, 'one node'),
    ({}, 'no train_nep_sharded'),
])
def test_torchnep_rejects_unsupported_launch_before_training(tmp_path, monkeypatch, environment, error):
    from NepTrain.core.training import _train_torchnep

    monkeypatch.setitem(sys.modules, 'torchnep', types.SimpleNamespace(train_nep=lambda **kw: pytest.fail('launched')))
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 4)))
    for key in ('WORLD_SIZE', 'SLURM_JOB_NUM_NODES'):
        monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(TrainingError, match=error):
        _train_torchnep(TrainingRequest(tmp_path / 'nep.in', tmp_path / 'train.xyz', tmp_path / 'out'))


def test_distributed_worker_binds_local_rank_and_only_calls_sharded_api(tmp_path, monkeypatch):
    from NepTrain.core import torchnep_worker

    payload = {'run_seed': 12, 'resume_from': '/checkpoint.pt', 'output_dir': str(tmp_path)}
    request = tmp_path / 'request.json'
    request.write_text(json.dumps(payload))
    calls = []
    monkeypatch.setattr(sys, 'argv', ['torchnep_worker', str(request)])
    monkeypatch.setenv('WORLD_SIZE', '4')
    monkeypatch.setenv('LOCAL_RANK', '2')
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 4,
        set_device=lambda rank: calls.append(rank),
    )))
    monkeypatch.setitem(sys.modules, 'torchnep', types.SimpleNamespace(
        train_nep_sharded=lambda **kw: calls.append(kw),
    ))
    torchnep_worker.main()
    assert calls == [2, payload]


@pytest.mark.parametrize("contents", [None, "", " \n\t"])
def test_gpumd_skips_unavailable_test_and_stale_copy(tmp_path, monkeypatch, caplog, contents):
    train_file = tmp_path / "train.xyz"
    write(train_file, Atoms("Fe", cell=[4, 4, 4], pbc=True), format="extxyz")
    config = tmp_path / "nep.in"
    config.write_text("cutoff 6 4\n")
    optional = tmp_path / "optional.xyz"
    if contents is not None:
        optional.write_text(contents)
    output = tmp_path / "output"
    output.mkdir()
    (output / "test.xyz").write_text("stale test from previous run")

    def fake_run(command, *, stdout, stderr, cwd, check):
        assert not (Path(cwd) / "test.xyz").exists()
        (Path(cwd) / "nep.txt").write_text("nep4 1 Fe\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("NepTrain.core.training.subprocess.run", fake_run)
    monkeypatch.setitem(sys.modules, "nep_adapters", _ordinary_model_adapter())
    result = train(TrainingRequest(config, train_file, output, test_file=optional), "gpumd")
    assert result.best_model.is_file()
    assert "Training test dataset skipped" in caplog.text
