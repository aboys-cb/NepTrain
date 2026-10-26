from importlib.resources import files
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
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
from NepTrain.core.spin import validate_spin_structure


def test_compute_property_order_drives_dump_meaning():
    columns = compute_property_columns(
        "compute spin all property/atom sp spx spy spz fmx fmy fmz fx fy fz"
    )
    assert columns["c_spin[1]"] == "sp"
    assert columns["c_spin[2]"] == "spx"
    assert columns["c_spin[4]"] == "spz"
    assert columns["c_spin[7]"] == "fmz"


@pytest.mark.parametrize("ids", [(2, 1), (2, 2), (3, 1)])
def test_dump_preserves_site_constraints_by_atom_id(tmp_path, ids):
    reference = Atoms("FeO", cell=[4, 4, 4], pbc=True)
    reference.set_array("spin", np.array([[0., 0, 2], [0, 0, .04]]))
    reference.set_array("spin_active", np.array([1, 0]))
    dump = tmp_path / "traj.dump"
    dump.write_text(
        "ITEM: TIMESTEP\n10\nITEM: NUMBER OF ATOMS\n2\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 4\n0 4\n0 4\n"
        "ITEM: ATOMS id type x y z sp spx spy spz fmx fmy fmz\n"
        f"{ids[0]} 2 2 2 2 0.04 0 0 1 .1 .2 .3\n"
        f"{ids[1]} 1 0 0 0 2 0 0 1 .4 .5 .6\n"
    )
    from NepTrain.core.md.lammps import LammpsError
    if ids != (2, 1):
        with pytest.raises(LammpsError, match="cannot restore spin_active"):
            read_lammps_dump(dump, Prism(reference.cell), ("Fe", "O"), spin=True, reference_atoms=reference)
        return
    frame = read_lammps_dump(dump, Prism(reference.cell), ("Fe", "O"), spin=True, reference_atoms=reference)[0]
    np.testing.assert_array_equal(frame.arrays["spin_active"], [1, 0])
    assert frame.get_chemical_symbols() == ["Fe", "O"]
    np.testing.assert_allclose(frame.arrays["nep_mforce"][1], [.1, .2, .3])
    assert "mforce" not in frame.arrays  # Model predictions are not DFT multipliers.
    assert validate_spin_structure(frame, require_mforce=False)
    path = tmp_path / "trajectory.xyz"
    ase_write(path, frame, format="extxyz")
    np.testing.assert_array_equal(ase_read(path).arrays["spin_active"], [1, 0])


def test_custom_template_receives_route_variables_without_ensemble_inference(
    tmp_path: Path, monkeypatch
):
    template = tmp_path / "custom.in"
    template.write_text(
        "{{ route_id }} {{ replica }} {{ temperature }} {{ steps }}\n"
        "timestep {{ timestep_ps }} # {{ timestep_fs }} fs\n"
        "timestep 0.002\n",
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
        timestep=0.005,
        template_path=template,
        route_id="route_b",
        route_fingerprint="f" * 64,
    )

    result = run_md(request, "lammps")

    assert result.completed is True
    rendered = render_template(captured["template"], captured["variables"])
    assert "timestep 0.005 # 5.0 fs" in rendered
    assert "timestep 0.002" in rendered
    assert captured["variables"] == {
        "temperature": 500,
        "spin_temperature": None,
        "pressure": 2,
        "timestep_ps": 0.005,
        "timestep_fs": 5.0,
        "steps": 25,
        "seed": 9,
        "replica": 3,
        "route_id": "route_b",
        "route_fingerprint": "f" * 64,
        "dump_interval": 1,
    }


def test_default_lammps_dump_interval_grows_with_trajectory_length():
    template = files("NepTrain.core.md").joinpath("templates/nvt.in").read_text()

    rendered = render_template(
        template,
        {
            "atom_style": "atomic",
            "structure_file": "structure.data",
            "pair_style": "nep/cpu",
            "model_file": "nep.txt",
            "elements": "Fe",
            "temperature": 600,
            "timestep_ps": 0.005,
            "seed": 7,
            "fix_suffix": "",
            "trajectory_file": "dump.lammpstrj",
            "steps": 500_000,
            "dump_interval": 1000,
            "halt_commands": "fix stop all halt 100 v_bad != 0 error soft",
        },
    )

    assert "timestep 0.005" in rendered
    assert "dump trajectory all custom 1000 dump.lammpstrj" in rendered
    assert "fix stop all halt 100" in rendered


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
    np.testing.assert_allclose(frames[0].arrays["nep_mforce"], [[0.1, 0.2, 0.3]])
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


def test_soft_fix_halt_is_not_misclassified_as_completed(tmp_path: Path, monkeypatch):
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
            _ordinary_dump([0, 10, 20]),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "Fix halt condition for fix-id neptrain_halt_force "
                "met on step 25 with value 101\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("NepTrain.core.md.lammps.subprocess.run", fake_run)
    output = tmp_path / "trajectory.xyz"
    result = run_lammps(
        atoms=atoms,
        model_file=model,
        output_dir=tmp_path / "run",
        output_file=output,
        template="run {{ steps }}\n",
        variables={"steps": 100, "dump_interval": 10},
        inference_backend="cpu",
        lmp_command="lmp",
        mpiexec="mpirun",
        mpi_ranks=1,
        spin=False,
    )

    rendered = result.input_file.read_text(encoding="utf-8")
    health = result.health_report.read_text(encoding="utf-8")
    recovered = ase_read(output, index=":", format="extxyz")

    assert "variable neptrain_volume equal vol" in rendered
    assert "fix neptrain_halt_volume_low all halt 10 v_neptrain_volume" in rendered
    assert "compute neptrain_max_force all reduce max" in rendered
    assert "v_neptrain_max_force_value > 100" in rendered
    assert result.completed is False
    assert result.failure_code == "trajectory_halt"
    assert "max_force_above_limit" in result.failure_reason
    assert '"reason_code": "max_force_above_limit"' in health
    assert [frame.info["md_window"] for frame in recovered] == [
        "pre_failure",
        "pre_failure",
        "bad_tail",
    ]


def test_write_spin_data_uses_direction_then_magnitude(tmp_path: Path):
    atoms = Atoms("Fe", positions=[[1, 1, 1]], cell=[4, 4, 4], pbc=True)
    atoms.set_array("spin", np.asarray([[0.0, 3.0, 4.0]]))
    path = tmp_path / "structure.data"
    write_lammps_data(path, atoms, ("Fe",), spin=True)
    atom_line = path.read_text(encoding="utf-8").split("Atoms # spin\n\n", 1)[1].splitlines()[0]
    values = list(map(float, atom_line.split()))
    np.testing.assert_allclose(values[5:9], [0.0, 0.6, 0.8, 5.0])


def test_write_spin_data_preserves_nonmagnetic_elements(tmp_path: Path):
    atoms = Atoms(
        "FeOY",
        positions=[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
        cell=[[0, 4, 0], [-4, 1, 0], [0, 0, 4]],
        pbc=True,
    )
    spins = np.asarray([[0.0, 3.0, 4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    atoms.set_array("spin", spins)
    path = tmp_path / "structure.data"

    prism = write_lammps_data(path, atoms, ("Fe", "O", "Y"), spin=True)
    rows = path.read_text().split("Atoms # spin\n\n", 1)[1].splitlines()
    values = np.asarray([list(map(float, row.split())) for row in rows])
    directions, magnitudes = values[:, 5:8], values[:, 8]

    assert np.isfinite(values).all()
    np.testing.assert_array_equal(values[:, 1], [1, 2, 3])
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    np.testing.assert_array_equal(magnitudes, [5.0, 0.0, 0.0])
    np.testing.assert_allclose(
        prism.vector_to_ase(directions * magnitudes[:, None]), spins
    )


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


def test_spin_range_failure_reaches_task_result_and_trajectory_windows(tmp_path, monkeypatch):
    import json
    from NepTrain.core.md_policy import TrajectoryHealthPolicy

    model = tmp_path / 'model.txt'
    model.write_text('fake model\n')
    atoms = Atoms('FeO', positions=[[1, 1, 1], [4, 1, 1]], cell=[8, 8, 8], pbc=True)
    atoms.set_array('spin', np.array([[0., 0., 3.], [0., 0., 0.]]))
    frames = [atoms.copy() for _ in range(3)]
    for i, frame in enumerate(frames):
        frame.info['lammps_step'] = i * 10
    frames[1].arrays['spin'][0] = [0., 0., .4]
    monkeypatch.setitem(sys.modules, 'nep_adapters', SimpleNamespace(
        inspect_model=lambda _: SimpleNamespace(elements=('Fe', 'O'), supports=lambda _: True)))
    monkeypatch.setattr('NepTrain.core.md.lammps.resolve_backend', lambda *_: 'cpu')
    monkeypatch.setattr('NepTrain.core.md.lammps.read_lammps_dump', lambda *a, **kw: frames)
    monkeypatch.setattr('NepTrain.core.md.lammps.subprocess.run',
                        lambda *a, **kw: subprocess.CompletedProcess([], 0, stdout='', stderr=''))
    result = run_lammps(
        atoms=atoms, model_file=model, output_dir=tmp_path / 'run',
        output_file=tmp_path / 'trajectory.xyz', template='run {{ steps }}\n',
        variables={'steps': 30}, inference_backend='cpu', lmp_command='lmp',
        mpiexec='mpirun', mpi_ranks=1, spin=True,
        health_policy=TrajectoryHealthPolicy.from_mapping({
            'spin_magnitude_by_element': {'Fe': {'min': 1.}}}),
    )
    assert not result.completed
    assert result.failure_code == 'trajectory_health'
    assert 'Fe atom #1: spin magnitude 0.4 μB below minimum 1 μB' in result.failure_reason
    health = json.loads(result.health_report.read_text())
    assert health['first_bad_spin_violation_counts'] == {'Fe': 1}
    saved = ase_read(result.trajectory, index=':')
    assert [f.info['md_window'] for f in saved] == ['pre_failure', 'bad_tail', 'bad_tail']
