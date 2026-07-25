from importlib.resources import files
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.calculators.lammps.coordinatetransform import Prism
from ase.io import read as ase_read
from ase.io import write as ase_write

from NepTrain.core.md.lammps import (
    compute_property_columns,
    read_lammps_dump,
    render_template,
    run_lammps,
    write_lammps_data,
)
from NepTrain.core.md import MdRequest, run_md


def test_compute_property_order_drives_dump_meaning():
    columns = compute_property_columns(
        "compute spin all property/atom sp spx spy spz fmx fmy fmz fx fy fz"
    )
    assert columns["c_spin[1]"] == "sp"
    assert columns["c_spin[2]"] == "spx"
    assert columns["c_spin[4]"] == "spz"
    assert columns["c_spin[7]"] == "fmz"


def test_custom_template_receives_route_variables_without_ensemble_inference(
    tmp_path: Path, monkeypatch
):
    template = tmp_path / "custom.in"
    template.write_text(
        "{{ route_id }} {{ replica }} {{ temperature }} {{ steps }}\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_lammps(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            trajectory=kwargs["output_file"],
            backend="cpu",
            pair_style="nep/cpu",
            completed=True,
            last_step=25,
            failure_code=None,
            failure_reason=None,
            health_report=None,
        )

    monkeypatch.setattr("NepTrain.core.md.run.run_lammps", fake_lammps)
    request = MdRequest(
        atoms=Atoms("Fe", positions=[[0, 0, 0]]),
        model_file=tmp_path / "nep.txt",
        output_dir=tmp_path / "run",
        output_file=tmp_path / "run" / "trajectory.xyz",
        temperature=500,
        pressure=2,
        steps=25,
        seed=9,
        replica=3,
        ensemble="mc-md-custom",
        template_path=template,
        route_id="route_b",
        route_fingerprint="f" * 64,
    )

    result = run_md(request, "lammps")

    assert result.completed is True
    assert captured["variables"] == {
        "temperature": 500,
        "spin_temperature": None,
        "pressure": 2,
        "steps": 25,
        "seed": 9,
        "replica": 3,
        "route_id": "route_b",
        "route_fingerprint": "f" * 64,
    }


def test_read_spin_dump_reconstructs_full_vector(tmp_path: Path):
    dump = tmp_path / "traj.dump"
    dump.write_text(
        """ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS xy xz yz pp pp pp
0 4 0
0 4 0
0 4 0
ITEM: ATOMS id type x y z c_spin[1] c_spin[2] c_spin[3] c_spin[4] c_spin[5] c_spin[6] c_spin[7]
1 1 1 2 3 2.5 0.6 0.8 0.0 0.1 0.2 0.3
""",
        encoding="utf-8",
    )
    mapping = compute_property_columns(
        "compute spin all property/atom sp spx spy spz fmx fmy fmz fx fy fz"
    )
    frames = read_lammps_dump(
        dump,
        Prism(np.eye(3) * 4),
        ("Fe",),
        spin=True,
        property_columns=mapping,
    )
    np.testing.assert_allclose(frames[0].arrays["spin"], [[1.5, 2.0, 0.0]])
    np.testing.assert_allclose(frames[0].arrays["mforce"], [[0.1, 0.2, 0.3]])
    assert frames[0].info["lammps_step"] == 100


def _ordinary_dump(steps: list[int], *, incomplete_tail: bool = False) -> str:
    frames = []
    for step in steps:
        frames.append(
            f"""ITEM: TIMESTEP
{step}
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS xy xz yz pp pp pp
0 4 0
0 4 0
0 4 0
ITEM: ATOMS id type x y z fx fy fz
1 1 {1 + step / 1000.0} 1 1 0 0 0
"""
        )
    if incomplete_tail:
        frames.append(
            "ITEM: TIMESTEP\n50\nITEM: NUMBER OF ATOMS\n1\n"
        )
    return "".join(frames)


def test_failed_lammps_run_recovers_complete_frames_and_quarantines_tail(
    tmp_path: Path, monkeypatch
):
    model = tmp_path / "model.txt"
    model.write_text("fake model\n", encoding="utf-8")
    atoms = Atoms("Fe", positions=[[1, 1, 1]], cell=[4, 4, 4], pbc=True)
    monkeypatch.setitem(
        sys.modules,
        "nep_adapters",
        SimpleNamespace(
            inspect_model=lambda _path: SimpleNamespace(
                elements=("Fe",), supports=lambda capability: capability == "none"
            )
        ),
    )
    monkeypatch.setattr(
        "NepTrain.core.md.lammps.resolve_backend", lambda _model, _backend: "cpu"
    )

    def fake_run(command, *, cwd, env, text, capture_output):
        del command, env, text, capture_output
        (Path(cwd) / "dump.lammpstrj").write_text(
            _ordinary_dump([0, 10, 20, 30, 40], incomplete_tail=True),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="ERROR: Lost atoms\n", stderr=""
        )

    monkeypatch.setattr("NepTrain.core.md.lammps.subprocess.run", fake_run)
    output = tmp_path / "trajectory.xyz"
    ase_write(output, atoms, format="extxyz")
    result = run_lammps(
        atoms=atoms,
        model_file=model,
        output_dir=tmp_path / "run",
        output_file=output,
        template="run {{ steps }}\n",
        variables={"steps": 100},
        inference_backend="cpu",
        lmp_command="lmp",
        mpiexec="mpirun",
        mpi_ranks=1,
        spin=False,
        pre_failure_frames=2,
        bad_tail_frames=1,
    )
    frames = read_lammps_dump(
        tmp_path / "run" / "dump.lammpstrj",
        Prism(np.eye(3) * 4),
        ("Fe",),
        spin=False,
        allow_incomplete_tail=True,
    )
    recovered = ase_read(output, index=":", format="extxyz")

    assert len(frames) == len(recovered) == 5
    assert [frame.info["md_window"] for frame in recovered] == [
        "stable_prefix",
        "stable_prefix",
        "pre_failure",
        "pre_failure",
        "bad_tail",
    ]
    assert result.completed is False
    assert result.last_step == 40
    assert result.failure_code == "lammps_nonzero_exit"
    assert "Lost atoms" in result.failure_reason
    assert result.health_report is not None
    assert result.health_report.is_file()


def test_write_spin_data_uses_direction_then_magnitude(tmp_path: Path):
    atoms = Atoms("Fe", positions=[[1, 1, 1]], cell=[4, 4, 4], pbc=True)
    atoms.set_array("spin", np.asarray([[0.0, 3.0, 4.0]]))
    path = tmp_path / "structure.data"
    write_lammps_data(path, atoms, ("Fe",), spin=True)
    atom_line = path.read_text(encoding="utf-8").split("Atoms # spin\n\n", 1)[1].splitlines()[0]
    values = list(map(float, atom_line.split()))
    np.testing.assert_allclose(values[5:9], [0.0, 0.6, 0.8, 5.0])


def test_default_spin_template_matches_documented_compute_order():
    template = files("NepTrain.core.md").joinpath("templates/spin-nvt.in").read_text()
    assert "property/atom sp spx spy spz fmx fmy fmz fx fy fz" in template
    assert "velocity all create {{ temperature }} {{ seed }}" in template
    assert "stemp {{ spin_temperature }} seed {{ seed }}" in template
    rendered = render_template(
        "run {{ steps }}\nvariable x equal {{ temperature }}\n",
        {"steps": 10, "temperature": 300},
    )
    assert rendered == "run 10\nvariable x equal 300\n"
