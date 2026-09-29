import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from NepTrain.core.gpumd.io import RunInput
from NepTrain.core.md import MdError, MdRequest, run_md
from NepTrain.core.md.dump import adaptive_dump_interval


def _atoms() -> Atoms:
    return Atoms(
        "Fe2",
        positions=[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]],
        cell=[6.0, 6.0, 6.0],
        pbc=True,
    )


def _request(tmp_path: Path, **overrides) -> MdRequest:
    model = tmp_path / "nep.txt"
    model.write_text("fake model\n", encoding="utf-8")
    values = {
        "atoms": _atoms(),
        "model_file": model,
        "output_dir": tmp_path / "run",
        "output_file": tmp_path / "trajectory.xyz",
        "temperature": 500.0,
        "steps": 25,
        "seed": 9,
    }
    values.update(overrides)
    return MdRequest(**values)


def _write_dump(directory: Path, frames: list[Atoms], times: list[float]) -> None:
    for frame, time_fs in zip(frames, times):
        frame.info["Time"] = time_fs
    ase_write(directory / "dump.xyz", frames, format="extxyz")


@pytest.mark.parametrize(
    ("steps", "expected"),
    [(25, 1), (5_000, 50), (20_000, 200), (100_000, 1000), (500_000, 1000)],
)
def test_adaptive_dump_interval(steps, expected):
    assert adaptive_dump_interval(steps) == expected


def test_default_gpumd_nvt_input_uses_temperature_steps_and_seed(
    tmp_path: Path, monkeypatch
):
    captured = {}

    def fake_run(command, *, stdout, stderr, cwd, check):
        del stdout, stderr, check
        captured["command"] = command
        captured["input"] = (Path(cwd) / "run.in").read_text(encoding="utf-8")
        _write_dump(Path(cwd), [_atoms()], [25.0])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("NepTrain.core.gpumd.io.subprocess.run", fake_run)
    result = run_md(_request(tmp_path), "gpumd")

    assert captured["command"] == ["gpumd"]
    assert "velocity 500.0 seed 9" in captured["input"]
    assert "ensemble nvt_nhc 500.0 500.0 100" in captured["input"]
    assert "time_step 1.0" in captured["input"]
    assert "run 25" in captured["input"]
    assert result.completed is True
    assert result.last_step == 25
    assert result.health_report is not None
    frames = ase_read(result.trajectory, index=":")
    assert frames[0].info["gpumd_step"] == 25
    assert frames[0].info["md_window"] == "stable_prefix"


def test_default_gpumd_nve_input_keeps_velocity_initialisation(
    tmp_path: Path, monkeypatch
):
    captured = {}

    def fake_run(command, *, stdout, stderr, cwd, check):
        del stdout, stderr, check
        captured["input"] = (Path(cwd) / "run.in").read_text(encoding="utf-8")
        _write_dump(Path(cwd), [_atoms()], [25.0])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("NepTrain.core.gpumd.io.subprocess.run", fake_run)
    result = run_md(_request(tmp_path, ensemble="nve"), "gpumd")

    assert "velocity 500.0 seed 9" in captured["input"]
    assert "ensemble nve" in captured["input"]
    assert result.completed is True


@pytest.mark.parametrize(
    ("ensemble", "rendered_ensemble"),
    [
        ("nve", "nve"),
        ("nvt_nhc 100 {{ temperature }} 100", "nvt_nhc 100 500.0 100"),
        ("user_defined_ensemble {{ temperature }}", "user_defined_ensemble 500.0"),
    ],
)
def test_custom_gpumd_template_renders_without_ensemble_inference(
    tmp_path: Path, monkeypatch, ensemble: str, rendered_ensemble: str
):
    template = tmp_path / "route.in"
    template.write_text(
        "# {{ route_id }} replica {{ replica }} {{ route_fingerprint }}\n"
        "potential {{ model_file }}\n"
        "velocity {{ temperature }} seed {{ seed }}\n"
        f"ensemble {ensemble}\n"
        "time_step {{ timestep_fs }}\n"
        "dump_exyz {{ dump_interval }} 0 1\n"
        "run {{ steps }}\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_run(command, *, stdout, stderr, cwd, check):
        captured["input"] = (Path(cwd) / "run.in").read_text(encoding="utf-8")
        assert (Path(cwd) / "model.xyz").is_file()
        assert (Path(cwd) / "nep.txt").read_text() == "fake model\n"
        _write_dump(Path(cwd), [_atoms()], [25.0])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("NepTrain.core.gpumd.io.subprocess.run", fake_run)
    result = run_md(
        _request(
            tmp_path, template_path=template, ensemble="custom-route",
            replica=3, route_id="route_b", route_fingerprint="abc123",
        ),
        "gpumd",
    )

    assert captured["input"] == (
        "# route_b replica 3 abc123\n"
        "potential nep.txt\n"
        "velocity 500.0 seed 9\n"
        f"ensemble {rendered_ensemble}\n"
        "time_step 1.0\n"
        "dump_exyz 1 0 1\n"
        "run 25\n"
    )
    assert result.completed is True
    assert result.last_step == 25


def test_gpumd_template_preserves_fixed_values_comments_and_multiple_stages(tmp_path):
    text = (
        "# User owns all commands, including stages and force output.\n"
        "potential custom-nep.txt\n"
        "velocity 50 seed 17\n"
        "ensemble nvt_nhc 50 900 100  # Heating ramp\n"
        "time_step 2\n"
        "dump_exyz 1000 0 0\n"
        "run 100000\n\n"
        "ensemble npt_scr 900 600 100 1 2 3 4 5 6 100 101 102 103 104 105 1000\n"
        "run {{ steps }}\n"
    )
    template = tmp_path / "route.in"
    template.write_text(text, encoding="utf-8")
    run = RunInput(tmp_path / "nep.txt")
    run.read_run(template)
    run.configure(temperature=500, pressure=2.5, steps=25, timestep_fs=1, seed=9)
    output = tmp_path / "rendered.in"
    run.write_run(output)

    assert output.read_text() == text.replace("{{ steps }}", "25")
    assert run.dump_interval() == 1000
    assert run.timestep_fs() == 2
    # Rendering again uses the original placeholders, not the previous result.
    run.configure(temperature=600, pressure=3, steps=50, timestep_fs=1, seed=10)
    run.write_run(output)
    assert output.read_text() == text.replace("{{ steps }}", "50")


def test_gpumd_template_only_replaces_explicit_pressure_components(tmp_path):
    template = tmp_path / "route.in"
    template.write_text(
        "ensemble npt_scr 300 {{ temperature }} 100 "
        "{{ pressure }} 2 3 4 5 6 100 101 102 103 104 105 1000\n",
        encoding="utf-8",
    )
    run = RunInput(tmp_path / "nep.txt")
    run.read_run(template)
    run.configure(temperature=500, pressure=2.5, steps=25, timestep_fs=1, seed=9)
    output = tmp_path / "rendered.in"
    run.write_run(output)
    assert output.read_text() == (
        "ensemble npt_scr 300 500 100 "
        "2.5 2 3 4 5 6 100 101 102 103 104 105 1000\n"
    )


def test_gpumd_template_does_not_inject_missing_commands(tmp_path):
    template = tmp_path / "route.in"
    text = "# Commands need not contain a recognized ensemble.\nrun {{ steps }}\n"
    template.write_text(text, encoding="utf-8")
    run = RunInput(tmp_path / "nep.txt")
    run.read_run(template)
    run.configure(temperature=500, pressure=0, steps=25, timestep_fs=1, seed=9)
    output = tmp_path / "rendered.in"
    run.write_run(output)
    assert output.read_text() == text.replace("{{ steps }}", "25")


def test_gpumd_missing_template_variable_fails_before_launch(tmp_path, monkeypatch):
    template = tmp_path / "route.in"
    template.write_text("ensemble custom {{ unknown_temperature }}\n")

    def unexpected_run(*args, **kwargs):
        pytest.fail("GPUMD must not launch with unresolved placeholders")

    monkeypatch.setattr("NepTrain.core.gpumd.io.subprocess.run", unexpected_run)
    with pytest.raises(MdError, match="missing GPUMD template variables: unknown_temperature"):
        run_md(_request(tmp_path, template_path=template), "gpumd")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("run 25\n", "must define dump_exyz"),
        ("dump_exyz 0 0 1\n", "interval must be positive"),
        ("dump_exyz bad 0 1\n", "integer interval"),
        ("dump_exyz 1 0 1 0 1\n", "separated dump_exyz"),
        ("dump_exyz 1 0 1\n", "time_step must be positive"),
    ],
)
def test_gpumd_checks_trajectory_contract_before_launch(tmp_path, text, message):
    template = tmp_path / "route.in"
    template.write_text(text)
    with pytest.raises(MdError, match=message):
        run_md(_request(tmp_path, template_path=template), "gpumd")


def test_default_gpumd_npt_keeps_pressure_mapping(tmp_path):
    run = RunInput(tmp_path / "nep.txt")
    run.use_default(
        ensemble="npt", temperature=500, pressure=2.5, steps=25,
        timestep_fs=1, seed=9,
    )
    output = tmp_path / "rendered.in"
    run.write_run(output)
    assert (
        "ensemble npt_scr 500 500 100 2.5 2.5 2.5 0 0 0 100 100 100 100 100 100 1000\n"
        in output.read_text()
    )


def test_gpumd_health_quarantines_a_physically_bad_tail(
    tmp_path: Path, monkeypatch
):
    frames = [_atoms(), _atoms(), _atoms()]
    frames[-1].positions[1] = [1.1, 1.0, 1.0]

    def fake_run(command, *, stdout, stderr, cwd, check):
        del stdout, stderr, check
        _write_dump(Path(cwd), frames, [10.0, 20.0, 25.0])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("NepTrain.core.gpumd.io.subprocess.run", fake_run)
    result = run_md(_request(tmp_path), "gpumd")

    assert result.completed is False
    assert result.failure_code == "trajectory_health"
    health = json.loads(result.health_report.read_text(encoding="utf-8"))
    assert health["first_bad_step"] == 25
    written = ase_read(result.trajectory, index=":")
    assert [frame.info["md_window"] for frame in written] == [
        "pre_failure",
        "pre_failure",
        "bad_tail",
    ]


def test_gpumd_dump_forces_feed_the_shared_health_policy(
    tmp_path: Path, monkeypatch
):
    frames = [_atoms(), _atoms()]
    for frame in frames:
        frame.set_array("forces", np.zeros((2, 3)))
    frames[-1].arrays["forces"][0, 0] = 101.0

    def fake_run(command, *, stdout, stderr, cwd, check):
        del stdout, stderr, check
        _write_dump(Path(cwd), frames, [10.0, 20.0])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("NepTrain.core.gpumd.io.subprocess.run", fake_run)
    result = run_md(_request(tmp_path), "gpumd")

    assert result.completed is False
    health = json.loads(result.health_report.read_text(encoding="utf-8"))
    assert health["reason_codes"] == ["max_force_above_limit"]
    assert "max_force" not in health["unavailable_thresholds"]


def test_failed_gpumd_run_recovers_frames_and_marks_fallback_tail(
    tmp_path: Path, monkeypatch
):
    def fake_run(command, *, stdout, stderr, cwd, check):
        del stdout, check
        stderr.write("GPUMD failure\n")
        stderr.flush()
        _write_dump(Path(cwd), [_atoms(), _atoms(), _atoms()], [10.0, 20.0, 25.0])
        with (Path(cwd) / "dump.xyz").open("a", encoding="utf-8") as handle:
            handle.write("2\nTime=30.0 Properties=species:S:1:pos:R:3\nFe 0 0 0\n")
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr("NepTrain.core.gpumd.io.subprocess.run", fake_run)
    result = run_md(_request(tmp_path), "gpumd")

    assert result.completed is False
    assert result.failure_code == "gpumd_nonzero_exit"
    assert "GPUMD failure" in result.failure_reason
    frames = ase_read(result.trajectory, index=":")
    assert [frame.info["md_window"] for frame in frames] == [
        "pre_failure",
        "pre_failure",
        "bad_tail",
    ]
