from pathlib import Path

from ase import Atoms
from ase.io import write

from NepTrain.core.train.run import NepTrainWorker


def _worker(tmp_path: Path) -> NepTrainWorker:
    worker = NepTrainWorker()
    worker.config = {
        "work_path": str(tmp_path / "cache"),
        "generation": 1,
        "training": {
            "backend": "torchnep",
            "config_path": str(tmp_path / "nep.in"),
            "test_path": None,
            "restart": True,
            "restart_steps": 20,
            "device": "cuda",
            "torch_backend": "auto",
            "precision": "float32",
            "use_compile": True,
        },
        "md": {
            "backend": "lammps",
            "duration_ps_every_generation": [1],
            "temperatures": [300],
            "structures": str(tmp_path / "structure.xyz"),
            "template_path": str(tmp_path / "lammps.in"),
            "timestep": 0.001,
            "ensemble": "nvt",
            "pressure": 0,
            "dump_interval": 10,
            "inference_backend": "cpu",
            "spin": True,
            "spin_temperature": 400,
            "spin_alpha": 0.02,
            "spin_seed": 7,
            "midpoint_iter": 3,
            "lmp": "lmp",
            "mpiexec": "mpirun",
            "mpi_ranks": 2,
            "plugin_path": "/plugins",
        },
    }
    return worker


def test_worker_builds_selected_training_and_md_commands(tmp_path: Path):
    worker = _worker(tmp_path)
    (tmp_path / "nep.in").write_text("type 1 Fe\n", encoding="utf-8")
    (tmp_path / "lammps.in").write_text("run {{ steps }}\n", encoding="utf-8")
    structure = Atoms("Fe", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
    write(tmp_path / "structure.xyz", structure)
    Path(worker.last_improved_train_xyz_file).write_bytes((tmp_path / "structure.xyz").read_bytes())
    Path(worker.nep_nep_txt_file).write_text("nep4 1 Fe\n", encoding="utf-8")

    training_command = worker.build_nep_params()
    assert "NepTrain nep --backend torchnep" in training_command
    assert "--device cuda" in training_command
    assert "--compile" in training_command

    md_command = worker.build_gpumd_params(tmp_path / "structure.xyz", 300, 0)
    assert "NepTrain md" in md_command
    assert "--backend lammps" in md_command
    assert "--spin-temperature 400" in md_command
    assert "--mpi-ranks 2" in md_command
