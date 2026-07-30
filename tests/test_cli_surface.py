from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from ruamel.yaml import YAML

from NepTrain.cli.cli import (
    _doctor_command_tools,
    _doctor_resource_probe,
    _doctor_target_requirements,
    _print_manual_status,
    run_manual_label_command,
    run_manual_md_command,
    run_doctor,
    run_model_worker_command,
)
from NepTrain.core.execution import ExecutionTarget


def _help(*arguments):
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    return subprocess.run(
        [sys.executable, "-m", "NepTrain.cli.cli", *arguments],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def test_doctor_resource_probe_checks_every_pinned_file(tmp_path):
    resource_root = tmp_path / "potpaw"
    resource = resource_root / "Fe" / "POTCAR"
    resource.parent.mkdir(parents=True)
    resource.write_bytes(b"pinned-potcar\n")
    record = {
        "path": "Fe/POTCAR",
        "sha256": hashlib.sha256(resource.read_bytes()).hexdigest(),
    }
    script = _doctor_resource_probe(resource_root, [record])

    passed = subprocess.run(
        ["bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
        check=False,
    )
    assert passed.returncode == 0, passed.stderr

    resource.write_bytes(b"drifted-potcar\n")
    failed = subprocess.run(
        ["bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode != 0
    assert "resource hash mismatch" in failed.stderr


def test_primary_help_only_shows_the_new_product_surface():
    completed = _help("--help")
    assert completed.returncode == 0
    assert "{train,md,label,select,perturb,workflow,task,data,doctor,smoke}" in completed.stdout
    for removed in ("vasp", "gpumd", "migrate"):
        assert f"    {removed} " not in completed.stdout
    assert "stage-worker" not in completed.stdout
    assert "stage-verify" not in completed.stdout
    assert "manual-worker" not in completed.stdout
    assert "model-worker" not in completed.stdout


@pytest.mark.parametrize(
    ("adapter", "module_name"),
    [
        ("mace", "NepTrain.runners.mace"),
        ("deepmd", "NepTrain.runners.deepmd"),
        ("tace", "NepTrain.runners.tace"),
    ],
)
def test_internal_model_worker_dispatches_to_the_selected_adapter(
    adapter,
    module_name,
    monkeypatch,
):
    captured = {}

    def fake_label_frames(*arguments, **options):
        captured["arguments"] = arguments
        captured["options"] = options

    monkeypatch.setattr(f"{module_name}.label_frames", fake_label_frames)
    result = run_model_worker_command(
        SimpleNamespace(
            adapter=adapter,
            head="OMol25",
            model="teacher.model",
            input="input.xyz",
            output="output.xyz",
            device="cpu",
            precision="float64",
            fidelity_index=2,
        )
    )

    assert result == 0
    assert captured["arguments"] == (
        "teacher.model",
        "input.xyz",
        "output.xyz",
    )
    assert captured["options"]["device"] == "cpu"
    assert captured["options"]["precision"] == "float64"
    if adapter == "deepmd":
        assert captured["options"]["head"] == "OMol25"
    elif adapter == "tace":
        assert captured["options"]["fidelity_index"] == 2
    else:
        assert "head" not in captured["options"]


def test_doctor_detects_tace_command_for_model_labeling():
    config = {
        "training": {"backend": "gpumd"},
        "md": {"backend": "gpumd"},
        "labeling": {
            "backend": "model",
            "runner": "neptrain model-worker tace --fidelity-index 0",
        },
        "execution": {
            "stage_targets": {
                "training": "teacher",
                "sampling": "teacher",
                "labeling": "teacher",
                "analysis": "teacher",
            },
            "sampling_route_targets": {},
        },
    }
    target = ExecutionTarget(
        name="teacher",
        executor="process",
        command="neptrain",
    )

    tools, packages, roles = _doctor_target_requirements(
        config,
        "teacher",
        target,
    )

    assert "tace-eval" in tools
    assert packages == []
    assert "labeling" in roles


def test_doctor_extracts_launcher_and_payload_commands():
    assert _doctor_command_tools("mpirun -n 8 vasp_std") == [
        "mpirun",
        "vasp_std",
    ]
    assert _doctor_command_tools("srun --exclusive gpumd") == [
        "srun",
        "gpumd",
    ]


def test_doctor_infers_all_project_target_requirements():
    config = {
        "training": {"backend": "torchnep"},
        "md": {"backend": "lammps"},
        "labeling": {
            "backend": "model",
            "runner": "neptrain model-worker deepmd --head OMol25",
        },
        "execution": {
            "stage_targets": {
                "training": "gpu",
                "sampling": "gpu",
                "labeling": "gpu",
                "analysis": "gpu",
            },
            "sampling_route_targets": {},
        },
    }
    target = ExecutionTarget(
        name="gpu",
        executor="slurm",
        command="neptrain",
        partition="gpu",
        cpus_per_task=4,
    )

    tools, packages, roles = _doctor_target_requirements(
        config,
        "gpu",
        target,
    )

    assert tools == [
        "lmp",
        "mpirun",
        "neptrain",
        "sacct",
        "sbatch",
        "squeue",
    ]
    assert packages == ["deepmd.calculator", "torchnep"]
    assert roles == ["analysis", "labeling", "sampling", "training"]


def test_workflow_commands_are_grouped():
    completed = _help("workflow", "--help")
    assert completed.returncode == 0
    assert "{init,run,status,resume,extend,stop}" in completed.stdout


def test_workflow_stop_exposes_explicit_job_preservation():
    completed = _help("workflow", "stop", "--help")
    assert completed.returncode == 0
    assert "--keep-jobs" in completed.stdout
    assert "--cancel-jobs" not in completed.stdout


def test_md_cli_keeps_template_owned_parameters_out_of_the_interface():
    completed = _help("md", "--help")
    assert completed.returncode == 0
    assert "--seed" in completed.stdout
    for removed in (
        "--plugin-path",
        "--timestep",
        "--tdamp",
        "--pdamp",
        "--dump-interval",
        "--spin-alpha",
        "--spin-seed",
        "--midpoint-iter",
    ):
        assert removed not in completed.stdout


def test_perturb_cli_names_the_two_physical_amplitudes_explicitly():
    completed = _help("perturb", "--help")
    assert completed.returncode == 0
    assert "--cell-perturbation" in completed.stdout
    assert "--max-displacement" in completed.stdout
    assert "--seed" in completed.stdout


def test_label_cli_rejects_competing_kpoint_overrides():
    completed = _help(
        "label",
        "input.xyz",
        "--kspacing",
        "0.2",
        "--ka",
        "4,4,4",
    )

    assert completed.returncode == 2
    assert "not allowed with argument" in completed.stderr


def test_workflow_init_returns_success_for_console_entrypoint(tmp_path):
    completed = _help(
        "workflow",
        "init",
        "--profile",
        "local",
        "--directory",
        str(tmp_path),
    )

    assert completed.returncode == 0
    assert (tmp_path / "project.yaml").is_file()


def test_workflow_run_recognizes_directory_as_prepared_workflow(tmp_path):
    completed = _help("workflow", "run", str(tmp_path))

    assert completed.returncode == 2
    assert "prepared workflow manifest does not exist" in completed.stderr
    assert "requires a project YAML file" not in completed.stderr


def test_manual_commands_offer_human_output_and_explicit_json():
    for command in ("train", "md", "label"):
        completed = _help(command, "--help")
        assert completed.returncode == 0
        assert "--json" in completed.stdout
    completed = _help("task", "status", "--help")
    assert completed.returncode == 0
    assert "--json" in completed.stdout


def test_select_exposes_production_sampling_controls():
    completed = _help("select", "--help")

    assert completed.returncode == 0
    assert "--base" in completed.stdout
    assert "--nep" in completed.stdout
    assert "--max-selected" in completed.stdout
    assert "--min-novelty" in completed.stdout
    assert "--descriptor-reduction" in completed.stdout
    assert "--report" in completed.stdout
    assert "--pca" not in completed.stdout
    assert "--umap" not in completed.stdout


def test_manual_status_is_human_readable_by_default(capsys):
    value = {
        "operation_id": "dft-abc",
        "kind": "label",
        "state": "submitted",
        "job_id": "123",
        "completed": 0,
        "total": 4,
        "run_directory": "/tmp/dft-run",
        "reason": "0/4 shards completed",
        "errors": [],
    }

    _print_manual_status(value)

    output = capsys.readouterr().out
    assert "Task: label (dft-abc)" in output
    assert "State: submitted" in output
    assert "Progress: 0/4" in output
    assert "Next: neptrain task wait /tmp/dft-run" in output
    assert not output.lstrip().startswith("{")


def test_manual_status_json_is_opt_in(capsys):
    value = {"kind": "label", "state": "complete"}

    _print_manual_status(value, json_output=True)

    assert json.loads(capsys.readouterr().out) == value


def _manual_project(tmp_path: Path) -> Path:
    (tmp_path / "route-a.in").write_text("run {{ steps }}\n", encoding="utf-8")
    (tmp_path / "route-b.in").write_text("run {{ steps }}\n", encoding="utf-8")
    value = {
        "schema_version": 8,
        "training": {
            "backend": "torchnep",
            "initial_path": "./train.xyz",
            "config_path": "./nep.in",
        },
        "md": {"backend": "lammps", "spin": False},
        "sampling": {
            "routes": [
                {
                    "id": "route-a",
                    "structures": ["./a.xyz"],
                    "template_path": "./route-a.in",
                    "conditions": {"temperature_path": [300]},
                },
                {
                    "id": "route-b",
                    "structures": ["./b.xyz"],
                    "template_path": "./route-b.in",
                    "conditions": {
                        "temperature_path": [400, 800],
                        "pressure": 2.5,
                    },
                    "progression": {
                        "steps": {
                            "smoke_passed": 123,
                            "short_stable": 456,
                            "long_stable": 789,
                            "production_ready": 1000,
                        },
                        "replicas": {
                            "smoke_passed": 1,
                            "short_stable": 1,
                            "long_stable": 2,
                            "production_ready": 3,
                        },
                    },
                },
            ]
        },
        "labeling": {
            "backend": "toy",
            "kpoint_mode": "kpoints",
            "kpoints": 4,
            "structures_per_job": 3,
            "max_concurrent": 9,
        },
        "workflow": {"id": "manual-probe", "max_model_generations": 1},
        "execution": {
            "stage_targets": {
                "training": "local",
                "sampling": "local",
                "labeling": "local",
                "analysis": "local",
            },
            "sampling_route_targets": {"route-b": "route-worker"},
            "targets": {
                "local": {"executor": "process"},
                "route-worker": {"executor": "process"},
            },
        },
    }
    path = tmp_path / "project.yaml"
    with path.open("w", encoding="utf-8") as handle:
        YAML().dump(value, handle)
    return path


def test_doctor_reads_backends_and_route_targets_from_project(
    tmp_path,
    monkeypatch,
    capsys,
):
    project = _manual_project(tmp_path)
    yaml = YAML()
    with project.open(encoding="utf-8") as handle:
        value = yaml.load(handle)
    value["training"]["backend"] = "gpumd"
    value["md"]["backend"] = "gpumd"
    for target in value["execution"]["targets"].values():
        target["command"] = sys.executable
        target["environment"] = {
            "NEPTRAIN_NEP_COMMAND": sys.executable,
            "NEPTRAIN_GPUMD_COMMAND": sys.executable,
        }
    with project.open("w", encoding="utf-8") as handle:
        yaml.dump(value, handle)
    monkeypatch.setattr("importlib.util.find_spec", lambda _package: object())
    value["notifications"] = {
        "feishu": {
            "webhook": (
                "https://open.feishu.cn/open-apis/bot/v2/hook/test-token"
            ),
            "secret": "test-secret",
        }
    }
    with project.open("w", encoding="utf-8") as handle:
        yaml.dump(value, handle)
    from NepTrain.core.notifications import DeliveryResult

    monkeypatch.setattr(
        "NepTrain.core.notifications.doctor_probe",
        lambda *_args, **_kwargs: DeliveryResult(True, "success"),
    )

    run_doctor(
        SimpleNamespace(
            project=str(project),
            training_backend=None,
            md_backend=None,
            inference_backend=None,
            model=None,
            structure=None,
            lmp="lmp",
            mpiexec="mpirun",
            mpi_ranks=1,
        )
    )

    output = capsys.readouterr().out
    assert "CHECK training=gpumd md=gpumd inference=auto" in output
    assert "roles=analysis,labeling,sampling,training" in output
    assert "roles=sampling" in output
    assert "OK Feishu webhook signature and delivery" in output
    assert "Doctor completed successfully." in output


def test_manual_md_inherits_the_selected_route_and_route_target(
    tmp_path, monkeypatch, capsys
):
    project = _manual_project(tmp_path)
    captured = {}

    def fake_prepare(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("NepTrain.core.manual.prepare_md", fake_prepare)
    monkeypatch.setattr(
        "NepTrain.core.manual.submit_operation",
        lambda *_args, **_kwargs: {"kind": "md", "state": "complete"},
    )
    run_manual_md_command(
        SimpleNamespace(
            project=str(project),
            target=None,
            route="route-b",
            maturity="short_stable",
            input="input.xyz",
            backend=None,
            model="nep.txt",
            temperature=None,
            output="trajectory.xyz",
            workdir=None,
            steps=None,
            pressure=None,
            ensemble=None,
            template=None,
            spin=None,
            spin_temperature=None,
            inference_backend=None,
            lmp=None,
            mpiexec=None,
            mpi_ranks=None,
            pre_failure_frames=None,
            bad_tail_frames=None,
            max_concurrent=None,
            force=False,
            wait=False,
            poll_interval=10,
            json=True,
        )
    )

    assert captured["target"].name == "route-worker"
    assert captured["temperatures"] == [400, 800]
    assert captured["steps"] == 456
    assert captured["seed"] == 12345
    assert captured["pressure"] == 2.5
    assert captured["template_path"] == str((tmp_path / "route-b.in").resolve())
    assert json.loads(capsys.readouterr().out)["state"] == "complete"


def test_manual_dft_inherits_project_parallelism_and_normalizes_scalar_kpoints(
    tmp_path, monkeypatch, capsys
):
    project = _manual_project(tmp_path)
    captured = {}

    def fake_prepare(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("NepTrain.core.manual.prepare_labeling", fake_prepare)
    monkeypatch.setattr(
        "NepTrain.core.manual.submit_operation",
        lambda *_args, **_kwargs: {"kind": "label", "state": "complete"},
    )
    run_manual_label_command(
        SimpleNamespace(
            project=str(project),
            target=None,
            input="input.xyz",
            backend=None,
            output="labeled.xyz",
            workdir=None,
            kspacing=None,
            ka=None,
            gamma=None,
            dft_input=None,
            resources=None,
            potcar_manifest=None,
            resource_manifest=None,
            cpus=None,
            structures_per_job=None,
            max_concurrent=None,
            teacher_profile=None,
            model=None,
            model_name=None,
            runner=None,
            device=None,
            precision=None,
            force=False,
            wait=False,
            poll_interval=10,
            json=True,
        )
    )

    assert captured["structures_per_job"] == 3
    assert captured["max_concurrent"] == 9
    assert captured["ka"] == [4, 4, 4]
    assert json.loads(capsys.readouterr().out)["state"] == "complete"


def test_local_toy_label_json_stdout_is_one_clean_document(tmp_path):
    source = tmp_path / "input.xyz"
    ase_write(
        source,
        Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
        format="extxyz",
    )
    completed = _help(
        "label",
        str(source),
        "--backend",
        "toy",
        "--workdir",
        str(tmp_path / "run"),
        "--output",
        str(tmp_path / "labeled.xyz"),
        "--json",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["state"] == "complete"
    assert payload["completed"] == payload["total"] == 1
    assert "[NepTrain]" not in completed.stdout


def test_json_output_rejects_nonfinite_values_without_partial_stdout(capsys):
    with pytest.raises(SystemExit, match="stable JSON output"):
        _print_manual_status(
            {"protocol": "fixture.v1", "metric": float("nan")},
            json_output=True,
        )

    assert capsys.readouterr().out == ""


def test_smoke_with_iteration_emits_one_versioned_json_document(tmp_path):
    completed = _help(
        "smoke",
        "--profile",
        "ordinary",
        "--output",
        str(tmp_path / "smoke"),
        "--iterations",
        "1",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["protocol"] == "neptrain.smoke.v1"
    assert payload["smoke"]["passed"] is True
    assert payload["iteration"]["generations_completed"] == 1


def test_spin_migration_is_explicit_atomic_and_json_clean(tmp_path):
    source = tmp_path / "legacy.xyz"
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    atoms.set_array("spins", np.asarray([[1.0, 0.0, 0.0]]))
    atoms.set_array("mforces", np.asarray([[0.1, 0.2, 0.3]]))
    ase_write(source, atoms, format="extxyz")
    output = tmp_path / "canonical.xyz"

    completed = _help(
        "data",
        "migrate-spin",
        str(source),
        str(output),
        "--json",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["spin_frames"] == 1
    restored = ase_read(output)
    assert "spin" in restored.arrays
    assert "mforce" in restored.arrays
    assert "spins" not in restored.arrays
    assert "mforces" not in restored.arrays
