import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from NepTrain.core.gpumd.io import GpumdInputError, RunInput
from NepTrain.core.md import MdRequest, run_md


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


def test_custom_gpumd_npt_template_receives_scalar_pressure(
    tmp_path: Path, monkeypatch
):
    template = tmp_path / "route.in"
    template.write_text(
        "\n".join(
            [
                "potential nep.txt",
                "velocity 50",
                (
                    "ensemble npt_scr 50 50 100 "
                    "0 0 0 0 0 0 100 101 102 103 104 105 1000"
                ),
                "time_step 2",
                "dump_exyz 1000 0 0",
                "run 100000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_run(command, *, stdout, stderr, cwd, check):
        del stdout, stderr, check
        captured["input"] = (Path(cwd) / "run.in").read_text(encoding="utf-8")
        _write_dump(Path(cwd), [_atoms()], [50.0])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("NepTrain.core.gpumd.io.subprocess.run", fake_run)
    result = run_md(
        _request(tmp_path, template_path=template, pressure=2.5),
        "gpumd",
    )

    assert (
        "ensemble npt_scr 500.0 500.0 100 "
        "2.5 2.5 2.5 0 0 0 100 101 102 103 104 105 1000"
        in captured["input"]
    )
    assert "velocity 500.0 seed 9" in captured["input"]
    assert "dump_exyz 25 0 1" in captured["input"]
    assert "time_step 2" in captured["input"]
    assert result.completed is True


@pytest.mark.parametrize(
    ("controls", "expected"),
    [
        ("0 100 1000", "2.5 100 1000"),
        (
            "0 0 0 100 101 102 1000",
            "2.5 2.5 2.5 100 101 102 1000",
        ),
    ],
)
def test_gpumd_npt_pressure_forms_are_updated_without_losing_coupling(
    tmp_path: Path, controls: str, expected: str
):
    template = tmp_path / "run.in"
    template.write_text(
        "\n".join(
            [
                "potential old-nep.txt",
                "velocity 50",
                f"ensemble npt_scr 50 50 100 {controls}",
                "time_step 1",
                "dump_exyz 10 0 0",
                "run 100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run = RunInput(tmp_path / "nep.txt")
    run.read_run(template)
    run.configure(
        temperature=500,
        pressure=2.5,
        steps=25,
        timestep_fs=1,
        seed=9,
    )
    rendered = tmp_path / "rendered.in"
    run.write_run(rendered)

    assert f"ensemble npt_scr 500 500 100 {expected}" in rendered.read_text(
        encoding="utf-8"
    )


def test_gpumd_rejects_an_ambiguous_npt_pressure_form(tmp_path: Path):
    template = tmp_path / "run.in"
    template.write_text(
        "ensemble npt_scr 50 50 100 0 0 0 0\nrun 10\n",
        encoding="utf-8",
    )
    run = RunInput(tmp_path / "nep.txt")
    run.read_run(template)

    with pytest.raises(GpumdInputError, match="pressure controls"):
        run.configure(
            temperature=500,
            pressure=2.5,
            steps=25,
            timestep_fs=1,
            seed=9,
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
