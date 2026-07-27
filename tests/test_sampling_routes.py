from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read as ase_read
from ase.io import write as ase_write

from NepTrain.core.dft.toy import ToyTeacher
from NepTrain.core.iteration import (
    GenerationController,
    GenerationPlan,
    StageContext,
)
from NepTrain.core.md import MdResult
from NepTrain.core.sampling_route import load_sampling_routes
from NepTrain.core.scenario import ScenarioLadder
from NepTrain.core.toy_workflow import toy_candidate_frames, toy_raw_features
from NepTrain.core.training import TrainingResult
from NepTrain.core.workflow import WorkflowError, prepare_workflow
from NepTrain.core.workflow_iteration import (
    PredictionEvaluation,
    WorkflowIterationAdapter,
    WorkflowRuntime,
)


def _progression(smoke: int = 10) -> dict:
    return {
        "steps": {
            "smoke_passed": smoke,
            "short_stable": smoke * 4,
            "long_stable": smoke * 16,
            "production_ready": smoke * 64,
        },
        "replicas": {
            "smoke_passed": 1,
            "short_stable": 1,
            "long_stable": 2,
            "production_ready": 3,
        },
    }


def _sampling(routes: list[dict], *, maximum: int = 4) -> dict:
    return {
        "routes": routes,
        "candidate_pool": {
            "pre_failure_frames": 2,
            "bad_tail_frames": 1,
            "health": {},
        },
        "selection": {
            "max_selected": maximum,
            "novelty": "auto",
        },
    }


def _route(
    route_id: str,
    structure: Path,
    template: Path,
    temperatures: list[float],
    *,
    smoke: int = 10,
) -> dict:
    return {
        "id": route_id,
        "structures": [str(structure)],
        "template_path": str(template),
        "conditions": {
            "temperature_path": temperatures,
            "production_temperatures": temperatures,
            "pressure": 0.0,
        },
        "progression": _progression(smoke),
    }


def test_one_explore_stage_schedules_two_explicit_routes(tmp_path: Path):
    structure_a = tmp_path / "a.xyz"
    structure_b = tmp_path / "b.xyz"
    template_a = tmp_path / "a.in"
    template_b = tmp_path / "b.in"
    initial = tmp_path / "initial.xyz"
    model = tmp_path / "nep.txt"
    ase_write(structure_a, Atoms("Fe", positions=[[0, 0, 0]]))
    ase_write(structure_b, Atoms("Ni", positions=[[1, 0, 0]]))
    ase_write(initial, ToyTeacher("ordinary").label(Atoms("Fe", positions=[[0, 0, 0]])))
    template_a.write_text("route A {{ temperature }}\n", encoding="utf-8")
    template_b.write_text("route B {{ temperature }}\n", encoding="utf-8")
    model.write_text("model\n", encoding="utf-8")
    captured = []

    def fake_md(request, backend):
        captured.append(request)
        frame = request.atoms.copy()
        frame.positions += 0.01 * len(captured)
        frame.info["md_window"] = "stable_prefix"
        request.output_dir.mkdir(parents=True, exist_ok=True)
        ase_write(request.output_file, [frame], format="extxyz")
        return MdResult(backend, request.output_file, request.output_dir, "cpu")

    config = {
        "training": {"backend": "gpumd"},
        "md": {"backend": "lammps", "spin": False},
        "sampling": _sampling(
            [
                _route("route_a", structure_a, template_a, [300, 600, 900]),
                _route(
                    "route_b",
                    structure_b,
                    template_b,
                    [500, 1000, 1500],
                    smoke=20,
                ),
            ]
        ),
        "labeling": {"backend": "toy"},
    }
    adapter = WorkflowIterationAdapter(
        config,
        initial_training=initial,
        runtime=WorkflowRuntime(md=fake_md),
    )
    outcome = adapter._explore(
        StageContext(
            generation=1,
            generation_dir=tmp_path,
            plan=GenerationPlan(1, 11, 4),
            artifacts={"model": model, "training_input": initial},
            previous_artifacts={},
            stage_dir=tmp_path / "explore",
        )
    )

    assert [(item.route_id, item.temperature, item.steps) for item in captured] == [
        ("route_a", 300.0, 10),
        ("route_b", 500.0, 20),
    ]
    assert [item.template_path for item in captured] == [template_a, template_b]
    plan = json.loads(outcome.artifacts["scenario_plan"].read_text())
    assert {item["route_id"] for item in plan["routes"]} == {
        "route_a",
        "route_b",
    }
    frames = ase_read(outcome.artifacts["candidates"], index=":")
    assert {frame.info["route_id"] for frame in frames} == {
        "route_a",
        "route_b",
    }
    assert all(frame.info["route_fingerprint"] for frame in frames)
    assert all(frame.info["template_sha256"] for frame in frames)
    assert all(frame.info["structure_hash"] for frame in frames)
    assert all(frame.info["sampling_model_sha256"] for frame in frames)


def test_same_structure_in_two_routes_has_isolated_scenario_identity(tmp_path: Path):
    structure = tmp_path / "same.xyz"
    template_a = tmp_path / "a.in"
    template_b = tmp_path / "b.in"
    ase_write(structure, Atoms("Fe", positions=[[0, 0, 0]]))
    template_a.write_text("A\n", encoding="utf-8")
    template_b.write_text("B\n", encoding="utf-8")
    routes = load_sampling_routes(
        _sampling(
            [
                _route("route_a", structure, template_a, [300]),
                _route("route_b", structure, template_b, [300]),
            ]
        ),
        base_dir=tmp_path,
    )
    structure_hash = "structure-sha"
    attempts = []
    for route in routes:
        ladder = ScenarioLadder.from_sampling(
            {
                "conditions": route.conditions,
                "progression": route.progression,
            }
        )
        attempts.append(
            ladder.schedule(
                [structure_hash],
                route_id=route.route_id,
                route_fingerprint=route.fingerprint,
                pressure=0.0,
                generation=1,
                seed=7,
                limit=1,
                model_id="model-sha",
            )[0]
        )
    assert attempts[0].scenario_id != attempts[1].scenario_id
    assert attempts[0].attempt_id != attempts[1].attempt_id

    ladder_a = ScenarioLadder.from_sampling(
        {
            "conditions": routes[0].conditions,
            "progression": routes[0].progression,
        }
    )
    history_a = ladder_a.record(
        [attempts[0]],
        completed={attempts[0].attempt_id: True},
        diagnostic_accepted={attempts[0].attempt_id: None},
        validation_accepted=None,
        model_improved=False,
        novelty_converged=False,
        final_model_id="model-sha",
    )
    assert history_a["counts_by_maturity"] == {"smoke_passed": 1}
    ladder_b = ScenarioLadder.from_sampling(
        {
            "conditions": routes[1].conditions,
            "progression": routes[1].progression,
        }
    )
    repeated_b = ladder_b.schedule(
        [structure_hash],
        route_id=routes[1].route_id,
        route_fingerprint=routes[1].fingerprint,
        pressure=0.0,
        generation=1,
        seed=7,
        limit=1,
        model_id="model-sha",
    )[0]
    assert repeated_b.target_level == "smoke_passed"


def test_route_fingerprint_is_canonical_and_content_addressed(tmp_path: Path):
    structure = tmp_path / "same.xyz"
    template = tmp_path / "route.in"
    ase_write(structure, Atoms("Fe", positions=[[0, 0, 0]]))
    template.write_text("first\n", encoding="utf-8")
    implicit = _route("route", structure, template, [300])
    implicit["conditions"].pop("production_temperatures")
    explicit = _route("route", structure, template, [300.0])
    first = load_sampling_routes(
        _sampling([implicit]), base_dir=tmp_path
    )[0]
    second = load_sampling_routes(
        _sampling([explicit]), base_dir=tmp_path
    )[0]
    assert first.fingerprint == second.fingerprint

    template.write_text("second\n", encoding="utf-8")
    changed = load_sampling_routes(
        _sampling([explicit]), base_dir=tmp_path
    )[0]
    assert changed.fingerprint != first.fingerprint

    condition_changed = _route("route", structure, template, [300, 600])
    changed_conditions = load_sampling_routes(
        _sampling([condition_changed]), base_dir=tmp_path
    )[0]
    assert changed_conditions.fingerprint != changed.fingerprint


def test_preparation_reuses_identical_routes_and_rejects_template_drift(
    tmp_path: Path,
):
    initial = tmp_path / "initial.xyz"
    structure = tmp_path / "structure.xyz"
    template = tmp_path / "route.in"
    training_config = tmp_path / "nep.in"
    initial_frame = Atoms(
        "Fe",
        positions=[[0, 0, 0]],
        cell=[4, 4, 4],
        pbc=True,
    )
    initial_frame.calc = SinglePointCalculator(
        initial_frame,
        energy=-1.0,
        forces=np.zeros((1, 3)),
    )
    initial_frame.info["virial"] = np.zeros((3, 3))
    ase_write(initial, initial_frame, format="extxyz")
    ase_write(structure, Atoms("Fe", positions=[[0, 0, 0]]))
    template.write_text("run {{ steps }}\n", encoding="utf-8")
    training_config.write_text("type 1 Fe\n", encoding="utf-8")
    project = tmp_path / "project.yaml"
    project.write_text(
        f"""
schema_version: 8
training:
  backend: gpumd
  initial_path: {initial}
  config_path: {training_config}
md:
  backend: lammps
  spin: false
sampling:
  routes:
    - id: route
      structures: [{structure}]
      template_path: {template}
      conditions:
        temperature_path: [300]
        production_temperatures: [300]
        pressure: 0
      progression:
        steps: {{smoke_passed: 10, short_stable: 40, long_stable: 160, production_ready: 640}}
        replicas: {{smoke_passed: 1, short_stable: 1, long_stable: 2, production_ready: 3}}
  candidate_pool: {{pre_failure_frames: 2, bad_tail_frames: 1, health: {{}}}}
  selection: {{max_selected: 4, novelty: auto}}
labeling:
  backend: toy
workflow:
  id: route-resume
  max_model_generations: 1
execution:
  stage_targets: {{training: local, sampling: local, labeling: local, analysis: local}}
  sampling_route_targets: {{}}
  targets:
    local: {{executor: process}}
""",
        encoding="utf-8",
    )
    output = tmp_path / "workflow"
    first = prepare_workflow(project, initial, output)
    second = prepare_workflow(project, initial, output)
    assert first.manifest == second.manifest
    assert json.loads(first.manifest.read_text())["sampling_routes"]

    template.write_text("run {{ steps }} # changed\n", encoding="utf-8")
    with pytest.raises(WorkflowError, match="preparation changed"):
        prepare_workflow(project, initial, output)


def test_workflow_generation_runs_without_evaluation(tmp_path: Path):
    teacher = ToyTeacher("ordinary")
    initial = tmp_path / "initial.xyz"
    structure = tmp_path / "structure.xyz"
    template = tmp_path / "route.in"
    training_config = tmp_path / "nep.in"
    initial_frames = [
        teacher.label(toy_candidate_frames("ordinary", 811, 2)[index])
        for index in range(2)
    ]
    ase_write(initial, initial_frames, format="extxyz")
    ase_write(structure, initial_frames[0], format="extxyz")
    template.write_text("run {{ steps }}\n", encoding="utf-8")
    training_config.write_text("type 1 Fe\n", encoding="utf-8")
    train_count = 0

    def fake_train(request, backend):
        nonlocal train_count
        train_count += 1
        request.output_dir.mkdir(parents=True, exist_ok=True)
        model = request.output_dir / "nep.txt"
        model.write_text(f"model {train_count}\n", encoding="utf-8")
        return TrainingResult(backend, model, None, None)

    def fake_md(request, backend):
        request.output_dir.mkdir(parents=True, exist_ok=True)
        frames = toy_candidate_frames(
            "ordinary",
            900 + request.replica,
            3,
            temperatures=(request.temperature,),
        )
        for frame in frames:
            frame.info["md_window"] = "stable_prefix"
        ase_write(request.output_file, frames, format="extxyz")
        return MdResult(backend, request.output_file, request.output_dir, "cpu")

    def finite_predict(_model, _frames, _backend):
        return PredictionEvaluation(
            {
                "energy_rmse": 0.5,
                "force_rmse": 0.5,
                "virial_rmse": 0.5,
            }
        )

    adapter = WorkflowIterationAdapter(
        {
            "training": {
                "backend": "gpumd",
                "config_path": str(training_config),
            },
            "md": {"backend": "lammps", "spin": False},
            "sampling": _sampling(
                [_route("route", structure, template, [300])],
                maximum=2,
            ),
            "labeling": {"backend": "toy"},
        },
        initial_training=initial,
        runtime=WorkflowRuntime(
            train=fake_train,
            md=fake_md,
            descriptors=lambda _model, frames: toy_raw_features(
                frames, "ordinary"
            ),
            predict=finite_predict,
        ),
    )
    summary = GenerationController(
        tmp_path / "workflow", "optional-evaluation"
    ).run_generation(
        GenerationPlan(1, 17, 2),
        adapter,
    )
    signals = json.loads(summary.artifacts["signals"].read_text())

    assert summary.accepted is True
    assert signals["evaluation_configured"] is False
    assert signals["validation_accepted"] is None
    assert signals["workflow_converged"] is False
    assert "validation_path" not in signals
    assert signals["candidate_validation_metrics"] is None
