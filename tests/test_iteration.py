import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from NepTrain.core.iteration import (
    GenerationController,
    GenerationPlan,
    IterationError,
    StageContext,
    StageOutcome,
    progressive_plans,
    stratified_farthest_point_sampling,
)
from NepTrain.core.md import MdResult
from NepTrain.core.scientific_data import bind_labeled_frames_to_inputs
from NepTrain.core.toy_iteration import (
    ToyGenerationPlan,
    ToyIterationAdapter,
    run_toy_iteration_smoke,
)
from NepTrain.core.toy_workflow import toy_candidate_frames, toy_raw_features
from NepTrain.core.training import TrainingResult
from NepTrain.core.workflow_iteration import (
    PredictionEvaluation,
    WorkflowIterationAdapter,
    WorkflowIterationError,
    WorkflowRuntime,
    _batched_descriptors,
    _acquisition_convergence_status,
    _nep_prediction_evaluation,
    _evaluation_quality,
    _structure_descriptors,
)
from NepTrain.core.generation_policy import (
    ACTIVE_LEARNING_ACQUISITION_STAGES,
    ACTIVE_LEARNING_FINALIZATION_STAGES,
    ACTIVE_LEARNING_GENERATION_PROTOCOL,
    ADAPTIVE_GENERATION_PROTOCOL,
    generation_stage_sequence,
    stage_for_role,
)
from NepTrain.core.reporting import ParitySeries
from NepTrain.core.dft.toy import ToyTeacher
from NepTrain.core.labeling import LabelResult
from NepTrain.core.candidate_pool import (
    CandidatePoolError,
    regular_batch_minimum,
    validate_candidate_pool,
    write_candidate_pool,
)
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.calculators.singlepoint import SinglePointCalculator


def test_prediction_evaluation_uses_per_atom_energy_and_virial(
    tmp_path, monkeypatch
):
    frame = Atoms("Fe2", positions=[[0, 0, 0], [2, 0, 0]])
    frame.calc = SinglePointCalculator(
        frame,
        energy=20.0,
        forces=np.zeros((2, 3)),
    )
    frame.info["virial"] = np.full(6, 8.0)

    class FakeCalculator:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def calculate(self, _frames, *, mean_virial):
            assert mean_virial is True
            return (
                np.asarray([22.0]),
                [np.zeros((2, 3))],
                np.full((1, 6), 5.0),
            )

    monkeypatch.setattr(
        "NepTrain.core.workflow_iteration.Nep3Calculator",
        FakeCalculator,
    )

    result = _nep_prediction_evaluation(
        tmp_path / "nep.txt", [frame], "cpu"
    )

    assert result.metrics["energy_rmse"] == pytest.approx(1.0)
    assert result.metrics["virial_rmse"] == pytest.approx(1.0)
    assert result.comparisons["energy"].unit == "eV/atom"
    assert result.comparisons["virial"].unit == "eV/atom"
    np.testing.assert_allclose(result.comparisons["energy"].reference, [10.0])
    np.testing.assert_allclose(result.comparisons["virial"].reference, 4.0)


def test_acquisition_convergence_requires_every_attempt_and_a_streak():
    policy = {
        "acquisition_max_rmse": {
            "energy_rmse": 0.003,
            "force_rmse": 0.10,
            "virial_rmse": 0.03,
        },
        "consecutive_generations": 2,
    }
    diagnostic = {
        "current_model_energy_rmse": 0.002,
        "current_model_force_rmse": 0.08,
        "current_model_virial_rmse": 0.02,
        "attempt_metrics": {
            "hot": {
                "energy_rmse": 0.0025,
                "force_rmse": 0.09,
                "virial_rmse": 0.025,
            }
        },
    }

    converged = _acquisition_convergence_status(
        diagnostic, policy, {"acquisition_convergence_streak": 1}
    )
    assert converged["acquisition_accepted"] is True
    assert converged["acquisition_convergence_streak"] == 2
    assert converged["acquisition_converged"] is True

    diagnostic["attempt_metrics"]["hot"]["force_rmse"] = 0.11
    rejected = _acquisition_convergence_status(diagnostic, policy, converged)
    assert rejected["acquisition_accepted"] is False
    assert rejected["acquisition_convergence_streak"] == 0
    assert rejected["acquisition_converged"] is False


def test_r2_convergence_requires_aggregate_groups_outliers_and_enough_labels():
    policy = {
        "acquisition_min_r2": {
            "energy_r2": 0.95,
            "force_r2": 0.95,
        },
        "group_min_force_r2": 0.90,
        "max_outlier_fraction": 0.05,
        "min_selected": 50,
        "consecutive_generations": 1,
    }
    diagnostic = {
        "current_model_energy_r2": 0.98,
        "current_model_force_r2": 0.97,
        "element_force_r2": {"Ce": 0.94, "Fe": 0.96},
        "condition_force_r2": {"R=main|T=600|P=0": 0.93},
        "outlier_fraction": {"energy": 0.01, "force": 0.03},
        "evaluated_count": 100,
    }

    result = _acquisition_convergence_status(diagnostic, policy, {})
    assert result["acquisition_accepted"] is True
    assert result["acquisition_converged"] is True

    diagnostic["condition_force_r2"]["R=main|T=600|P=0"] = 0.89
    rejected = _acquisition_convergence_status(diagnostic, policy, {})
    assert rejected["acquisition_groups_accepted"] is False
    assert rejected["acquisition_converged"] is False


def test_evaluation_quality_keeps_element_force_information():
    frames = [
        Atoms("CeFe", positions=[[0, 0, 0], [2, 0, 0]]),
        Atoms("Fe2", positions=[[0, 0, 0], [2, 0, 0]]),
    ]
    reference = np.arange(12, dtype=float)
    evaluation = PredictionEvaluation(
        {"force_rmse": 0.1},
        {
            "force": ParitySeries(
                reference,
                reference + 0.01,
                "eV/A",
            )
        },
    )

    quality = _evaluation_quality(evaluation, frames)

    assert set(quality["element_force_r2"]) == {"Ce", "Fe"}
    assert quality["r2"]["force_r2"] > 0.99
    assert quality["outlier_fraction"]["force"] == 0.0


def test_adaptive_generation_finalizes_in_a_train_evaluate_only_generation(
    tmp_path,
):
    calls = []

    class AdaptiveAdapter:
        def run_stage(self, stage, context):
            calls.append((context.generation, context.generation_kind, stage))
            path = context.work_dir / f"{stage}-{context.generation}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
            metrics = {}
            if context.generation_kind == "acquisition" and stage == "merge":
                metrics = {
                    "accepted": True,
                    "generation_disposition": "finalize",
                    "workflow_converged": False,
                }
            if context.generation_kind == "finalization" and stage == "evaluate":
                metrics = {
                    "accepted": True,
                    "generation_disposition": "finalize",
                    "workflow_converged": True,
                }
            return StageOutcome(
                {f"g{context.generation}_{stage}": path}, metrics
            )

    controller = GenerationController(
        tmp_path / "adaptive",
        "adaptive",
        generation_protocol=ADAPTIVE_GENERATION_PROTOCOL,
    )
    adapter = AdaptiveAdapter()
    first = controller.run_generation(GenerationPlan(1, 1, 2), adapter)
    second = controller.run_generation(GenerationPlan(2, 2, 2), adapter)

    assert first.accepted is True
    assert second.accepted is True
    assert calls == [
        (1, "acquisition", "train"),
        (1, "acquisition", "evaluate"),
        (1, "acquisition", "explore"),
        (1, "acquisition", "select"),
        (1, "acquisition", "label"),
        (1, "acquisition", "diagnose"),
        (1, "acquisition", "merge"),
        (2, "finalization", "train"),
        (2, "finalization", "evaluate"),
    ]


def test_active_learning_v3_uses_unambiguous_public_stage_names(tmp_path):
    calls = []

    class Adapter:
        def run_stage(self, stage, context):
            calls.append(
                (
                    context.generation,
                    stage,
                    tuple(context.stage_input["stage_sequence"]),
                )
            )
            path = context.work_dir / f"{stage}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
            metrics = {}
            if context.generation_kind == "acquisition" and stage == "update":
                metrics = {
                    "accepted": True,
                    "generation_disposition": "finalize",
                }
            elif context.generation_kind == "finalization" and stage == "validate":
                metrics = {"accepted": True, "workflow_converged": True}
            return StageOutcome({f"g{context.generation}_{stage}": path}, metrics)

    controller = GenerationController(
        tmp_path / "active-learning-v3",
        "active-learning-v3",
        generation_protocol=ACTIVE_LEARNING_GENERATION_PROTOCOL,
    )
    adapter = Adapter()
    controller.run_generation(GenerationPlan(1, 1, 2), adapter)
    controller.run_generation(GenerationPlan(2, 2, 2), adapter)

    assert [stage for generation, stage, _ in calls if generation == 1] == list(
        ACTIVE_LEARNING_ACQUISITION_STAGES
    )
    assert [stage for generation, stage, _ in calls if generation == 2] == list(
        ACTIVE_LEARNING_FINALIZATION_STAGES
    )
    assert all(
        sequence
        == (
            ACTIVE_LEARNING_ACQUISITION_STAGES
            if generation == 1
            else ACTIVE_LEARNING_FINALIZATION_STAGES
        )
        for generation, _, sequence in calls
    )


def test_active_learning_v3_dispatches_evaluate_separately_from_validate():
    calls = []

    class Adapter(WorkflowIterationAdapter):
        def __init__(self):
            pass

        def _acquisition_evaluate(self, context):
            calls.append("validate-implementation")
            return StageOutcome({}, {})

        def _acquisition_diagnose(self, context):
            calls.append("evaluate-implementation")
            return StageOutcome({}, {})

        def _acquisition_merge(self, context):
            calls.append("update-implementation")
            return StageOutcome({}, {"accepted": True})

    adapter = Adapter()
    for stage in ("validate", "evaluate", "update"):
        adapter.run_stage(
            stage,
            StageContext(
                generation=1,
                generation_dir=Path("."),
                plan=GenerationPlan(1, 1, 2),
                artifacts={},
                previous_artifacts={},
                generation_kind="acquisition",
                stage_sequence=ACTIVE_LEARNING_ACQUISITION_STAGES,
            ),
        )

    assert calls == [
        "validate-implementation",
        "evaluate-implementation",
        "update-implementation",
    ]


def test_stage_roles_disambiguate_v2_and_v3_evaluate():
    v2 = {
        "kind": "acquisition",
        "stage_sequence": [
            "train",
            "evaluate",
            "explore",
            "select",
            "label",
            "diagnose",
            "merge",
        ],
    }
    v3 = {
        "kind": "acquisition",
        "stage_sequence": list(ACTIVE_LEARNING_ACQUISITION_STAGES),
    }

    assert stage_for_role(v2, "validate") == "evaluate"
    assert stage_for_role(v2, "evaluate") == "diagnose"
    assert stage_for_role(v2, "update") == "merge"
    assert stage_for_role(v3, "validate") == "validate"
    assert stage_for_role(v3, "evaluate") == "evaluate"
    assert stage_for_role(v3, "update") == "update"

    with pytest.raises(
        ValueError,
        match="generation kind and stage sequence are inconsistent",
    ):
        generation_stage_sequence(
            {
                "kind": "acquisition",
                "stage_sequence": [
                    "train",
                    "validate",
                    "explore",
                    "select",
                    "label",
                    "diagnose",
                    "update",
                ],
            }
        )


def test_rejected_finalization_reopens_from_train(tmp_path):
    class Adapter:
        def __init__(self):
            self.accept_final = False

        def run_stage(self, stage, context):
            path = context.work_dir / f"{stage}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(stage + "\n", encoding="utf-8")
            metrics = {}
            if context.generation_kind == "acquisition" and stage == "merge":
                metrics = {
                    "accepted": True,
                    "generation_disposition": "finalize",
                }
            elif context.generation_kind == "finalization" and stage == "evaluate":
                metrics = {"accepted": self.accept_final}
            return StageOutcome(
                {f"g{context.generation}_{stage}": path}, metrics
            )

    adapter = Adapter()
    controller = GenerationController(
        tmp_path / "recover-final",
        "recover-final",
        generation_protocol=ADAPTIVE_GENERATION_PROTOCOL,
    )
    controller.run_generation(GenerationPlan(1, 1, 2), adapter)
    plan = GenerationPlan(2, 2, 2)
    rejected = controller.run_generation(plan, adapter)
    assert rejected.accepted is False

    controller.reopen_rejected(plan)

    assert controller.next_stage(plan) == "train"


@pytest.mark.parametrize(
    ("protocol", "stages", "evaluation_stage", "update_stage"),
    [
        (
            ADAPTIVE_GENERATION_PROTOCOL,
            (
                "train",
                "evaluate",
                "explore",
                "select",
                "label",
                "diagnose",
                "merge",
            ),
            "diagnose",
            "merge",
        ),
        (
            ACTIVE_LEARNING_GENERATION_PROTOCOL,
            ACTIVE_LEARNING_ACQUISITION_STAGES,
            "evaluate",
            "update",
        ),
    ],
)
def test_real_adapter_acquisition_trains_before_md_and_defers_completion(
    tmp_path,
    protocol,
    stages,
    evaluation_stage,
    update_stage,
):
    teacher = ToyTeacher("ordinary")
    initial = tmp_path / "initial.xyz"
    structures = tmp_path / "structures.xyz"
    training_config = tmp_path / "nep.in"
    ase_write(
        initial,
        [teacher.label(toy_candidate_frames("ordinary", 71, 1)[0])],
        format="extxyz",
    )
    ase_write(
        structures,
        toy_candidate_frames("ordinary", 72, 1),
        format="extxyz",
    )
    training_config.write_text("type 1 Fe\n", encoding="utf-8")

    def fake_train(request, backend):
        request.output_dir.mkdir(parents=True, exist_ok=True)
        model = request.output_dir / "nep.txt"
        model.write_text("trained\n", encoding="utf-8")
        return TrainingResult(backend, model, None, None)

    def fake_md(request, backend):
        request.output_dir.mkdir(parents=True, exist_ok=True)
        frames = [request.atoms.copy(), request.atoms.copy()]
        for index, frame in enumerate(frames, start=1):
            frame.positions[0, 0] += index * 0.01
            frame.info["lammps_step"] = index * 10
        ase_write(request.output_file, frames, format="extxyz")
        return MdResult(
            backend,
            request.output_file,
            request.output_dir,
            "cpu",
            completed=True,
            last_step=20,
        )

    def perfect_predict(_model, frames, _backend):
        energy = np.asarray(
            [frame.get_potential_energy() / len(frame) for frame in frames]
        )
        force = np.concatenate([frame.get_forces() for frame in frames])
        virial = np.asarray(
            [frame.info["virial"] / len(frame) for frame in frames]
        )
        return PredictionEvaluation(
            {"energy_rmse": 0.0, "force_rmse": 0.0, "virial_rmse": 0.0},
            {
                "energy": ParitySeries(energy, energy, "eV/atom"),
                "force": ParitySeries(force, force, "eV/A"),
                "virial": ParitySeries(virial, virial, "eV/atom"),
            },
        )

    adapter = WorkflowIterationAdapter(
        {
            "training": {
                "backend": "gpumd",
                "config_path": str(training_config),
            },
            "md": {"backend": "lammps", "spin": False},
            "sampling": _sampling(
                structures=structures,
                template=training_config,
                max_selected=2,
            ),
            "labeling": {"backend": "toy"},
            "workflow": {
                "convergence": {
                    "acquisition_min_r2": {
                        "energy_r2": 0.95,
                        "force_r2": 0.95,
                    },
                    "group_min_force_r2": 0.90,
                    "max_outlier_fraction": 0.05,
                    "min_selected": 1,
                    "consecutive_generations": 1,
                }
            },
        },
        initial_training=initial,
        runtime=WorkflowRuntime(
            train=fake_train,
            md=fake_md,
            descriptors=lambda _model, frames: toy_raw_features(
                frames, "ordinary"
            ),
            predict=perfect_predict,
        ),
    )
    summary = GenerationController(
        tmp_path / "adaptive-real",
        "adaptive-real",
        generation_protocol=protocol,
    ).run_generation(GenerationPlan(1, 7, 2), adapter)

    assert tuple(summary.metrics) == stages
    assert summary.metrics[evaluation_stage]["current_model_force_r2"] == 1.0
    assert summary.metrics[update_stage]["workflow_converged"] is False
    assert summary.metrics[update_stage]["generation_disposition"] in {
        "continue",
        "finalize",
    }
    assert "model_training_set" in summary.artifacts
    assert "training_set" in summary.artifacts


@pytest.mark.parametrize(
    ("stage_sequence", "validation_stage"),
    [
        (("train", "evaluate"), "evaluate"),
        (ACTIVE_LEARNING_FINALIZATION_STAGES, "validate"),
    ],
)
def test_real_finalization_trains_merged_dataset_and_emits_no_sampling_work(
    tmp_path,
    stage_sequence,
    validation_stage,
):
    merged = tmp_path / "merged.xyz"
    previous_model = tmp_path / "previous-nep.txt"
    training_config = tmp_path / "nep.in"
    teacher = ToyTeacher("ordinary")
    ase_write(
        merged,
        [teacher.label(toy_candidate_frames("ordinary", 81, 1)[0])],
        format="extxyz",
    )
    previous_model.write_text("previous\n", encoding="utf-8")
    training_config.write_text("type 1 Fe\n", encoding="utf-8")
    requests = []

    def fake_train(request, backend):
        requests.append(request)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        model = request.output_dir / "nep.txt"
        model.write_text("final\n", encoding="utf-8")
        return TrainingResult(backend, model, None, None)

    adapter = WorkflowIterationAdapter(
        {
            "training": {
                "backend": "gpumd",
                "config_path": str(training_config),
            },
            "md": {"spin": False},
            "workflow": {},
        },
        initial_training=None,
        active_stage="train",
        active_generation_kind="finalization",
        runtime=WorkflowRuntime(train=fake_train),
    )
    plan = GenerationPlan(2, 9, 2)
    train = adapter.run_stage(
        "train",
        StageContext(
            generation=2,
            generation_dir=tmp_path / "generation-2",
            plan=plan,
            artifacts={},
            previous_artifacts={
                "training_set": merged,
                "activated_model": previous_model,
            },
            stage_dir=tmp_path / "train",
            generation_kind="finalization",
            stage_sequence=stage_sequence,
        ),
    )
    evaluate = adapter.run_stage(
        validation_stage,
        StageContext(
            generation=2,
            generation_dir=tmp_path / "generation-2",
            plan=plan,
            artifacts=train.artifacts,
            previous_artifacts={},
            stage_dir=tmp_path / validation_stage,
            generation_kind="finalization",
            stage_sequence=stage_sequence,
        ),
    )

    assert requests[0].train_file == merged
    assert evaluate.metrics["accepted"] is True
    assert evaluate.metrics["workflow_converged"] is True
    assert evaluate.artifacts["training_set"] == merged
    assert not {"candidates", "selected_input", "labeled"}.intersection(
        evaluate.artifacts
    )


def test_finalization_does_not_complete_when_independent_validation_fails(
    tmp_path,
):
    teacher = ToyTeacher("ordinary")
    merged = tmp_path / "merged.xyz"
    validation = tmp_path / "validation.xyz"
    previous_model = tmp_path / "previous-nep.txt"
    training_config = tmp_path / "nep.in"
    ase_write(
        merged,
        [teacher.label(toy_candidate_frames("ordinary", 91, 1)[0])],
        format="extxyz",
    )
    validation_frame = toy_candidate_frames("ordinary", 92, 1)[0]
    validation_frame.positions[0, 0] += 0.25
    ase_write(validation, [teacher.label(validation_frame)], format="extxyz")
    previous_model.write_text("previous\n", encoding="utf-8")
    training_config.write_text("type 1 Fe\n", encoding="utf-8")

    def fake_train(request, backend):
        request.output_dir.mkdir(parents=True, exist_ok=True)
        model = request.output_dir / "nep.txt"
        model.write_text("final\n", encoding="utf-8")
        return TrainingResult(backend, model, None, None)

    def failing_predict(_model, frames, _backend):
        force = np.concatenate([frame.get_forces() for frame in frames])
        return PredictionEvaluation(
            {"energy_rmse": 0.2, "force_rmse": 0.5},
            {
                "force": ParitySeries(force, force + 0.5, "eV/A"),
            },
        )

    config = {
        "training": {
            "backend": "gpumd",
            "config_path": str(training_config),
        },
        "evaluation": {
            "validation_path": str(validation),
            "max_rmse": {"energy_rmse": 0.01, "force_rmse": 0.1},
        },
        "md": {"spin": False},
        "workflow": {},
    }
    train_adapter = WorkflowIterationAdapter(
        config,
        initial_training=None,
        active_stage="train",
        active_generation_kind="finalization",
        runtime=WorkflowRuntime(train=fake_train, predict=failing_predict),
    )
    plan = GenerationPlan(2, 9, 2)
    train = train_adapter.run_stage(
        "train",
        StageContext(
            generation=2,
            generation_dir=tmp_path / "generation-2",
            plan=plan,
            artifacts={},
            previous_artifacts={
                "training_set": merged,
                "activated_model": previous_model,
            },
            stage_dir=tmp_path / "train",
            generation_kind="finalization",
            stage_sequence=("train", "evaluate"),
        ),
    )
    evaluate_adapter = WorkflowIterationAdapter(
        config,
        initial_training=None,
        active_stage="evaluate",
        active_generation_kind="finalization",
        runtime=WorkflowRuntime(train=fake_train, predict=failing_predict),
    )
    outcome = evaluate_adapter.run_stage(
        "evaluate",
        StageContext(
            generation=2,
            generation_dir=tmp_path / "generation-2",
            plan=plan,
            artifacts=train.artifacts,
            previous_artifacts={},
            stage_dir=tmp_path / "evaluate",
            generation_kind="finalization",
            stage_sequence=("train", "evaluate"),
        ),
    )

    assert outcome.metrics["accepted"] is False
    assert outcome.metrics["workflow_converged"] is False


def test_descriptor_backend_receives_every_frame_in_bounded_batches(tmp_path):
    frames = toy_candidate_frames("ordinary", 501, 4101)
    calls = []

    def describe(_model, batch):
        calls.append(len(batch))
        return np.zeros((len(batch), 3))

    values = _batched_descriptors(describe, tmp_path / "nep.txt", frames)

    assert calls == [4096, 5]
    assert values.shape == (4101, 3)


def test_elementwise_workflow_descriptors_use_atomic_backend(tmp_path):
    frames = [
        Atoms("FeO", positions=[[0, 0, 0], [1, 0, 0]]),
        Atoms("FeO", positions=[[0, 0, 0], [1, 0, 0]]),
    ]

    def forbidden_global(*_args):
        raise AssertionError("global descriptor backend must not be used")

    def atomic(_model, _frames):
        return np.asarray([[2.0], [-2.0], [3.0], [-3.0]])

    values = _structure_descriptors(
        WorkflowRuntime(
            descriptors=forbidden_global,
            atomic_descriptors=atomic,
        ),
        tmp_path / "nep.txt",
        frames,
        reduction="elementwise_mean_std",
        elements=("Fe", "O"),
    )

    np.testing.assert_allclose(
        values,
        [[2.0, 0.0, -2.0, 0.0], [3.0, 0.0, -3.0, 0.0]],
    )


def _sampling(
    temperatures=(300.0,),
    *,
    structures,
    template,
    steps=(100, 400, 1600, 6400),
    max_selected=8,
):
    return {
        "routes": [
            {
                "id": "default",
                "structures": [str(structures)],
                "template_path": str(template),
                "conditions": {
                    "temperature_path": list(temperatures),
                    "production_temperatures": list(temperatures),
                    "pressure": 0.0,
                },
                "progression": {
                    "steps": dict(
                        zip(
                            (
                                "smoke_passed",
                                "short_stable",
                                "long_stable",
                                "production_ready",
                            ),
                            steps,
                        )
                    ),
                    "replicas": {
                        "smoke_passed": 1,
                        "short_stable": 1,
                        "long_stable": 2,
                        "production_ready": 3,
                    },
                },
            }
        ],
        "candidate_pool": {
            "pre_failure_frames": 2,
            "bad_tail_frames": 1,
            "health": {},
        },
        "selection": {
            "max_selected": max_selected,
            "novelty": "auto",
        },
    }


def _with_route(frames):
    for frame in frames:
        frame.info.setdefault("route_id", "default")
        frame.info.setdefault("route_fingerprint", "f" * 64)
    return frames


def test_progression_keeps_the_user_selection_limit_constant():
    plans = progressive_plans(4, max_selected=100)

    assert [plan.max_selected for plan in plans] == [100, 100, 100, 100]


def test_excluded_composition_is_not_scheduled_for_md(tmp_path: Path):
    excluded = Atoms(
        "Al2Co3Fe6Ni3Ta2",
        positions=np.arange(48, dtype=float).reshape(16, 3) * 0.1,
        cell=[20, 20, 20],
        pbc=True,
    )
    allowed = Atoms(
        "Al3Co3Fe4Ni3Ta3",
        positions=np.arange(48, dtype=float).reshape(16, 3) * 0.11,
        cell=[20, 20, 20],
        pbc=True,
    )
    structures = tmp_path / "structures.extxyz"
    model = tmp_path / "nep.txt"
    template = tmp_path / "template.in"
    training = tmp_path / "training.extxyz"
    ase_write(structures, [excluded, allowed], format="extxyz")
    ase_write(training, [allowed], format="extxyz")
    model.write_text("model\n", encoding="utf-8")
    template.write_text("template\n", encoding="utf-8")
    sampling = _sampling(
        structures=structures,
        template=template,
        max_selected=4,
    )
    sampling["excluded_compositions"] = ["Al2Co3Fe6Ni3Ta2"]
    adapter = WorkflowIterationAdapter(
        {
            "training": {"backend": "gpumd"},
            "md": {"backend": "lammps", "spin": False},
            "sampling": sampling,
            "labeling": {"backend": "toy"},
        },
        initial_training=training,
    )
    context = StageContext(
        generation=1,
        generation_dir=tmp_path,
        plan=GenerationPlan(1, 7, 4),
        artifacts={"model": model},
        previous_artifacts={},
    )

    attempts = adapter.plan_explore_attempts(context)

    assert len(attempts) == 1


def test_regular_label_floor_is_derived_from_the_user_maximum():
    assert regular_batch_minimum(1) == 1
    assert regular_batch_minimum(100) == 50
    assert regular_batch_minimum(200) == 100


def test_candidate_pool_rejects_a_different_sampling_model(tmp_path: Path):
    first_model = tmp_path / "model-a.txt"
    second_model = tmp_path / "model-b.txt"
    first_model.write_text("a\n", encoding="utf-8")
    second_model.write_text("b\n", encoding="utf-8")
    candidates = _with_route(toy_candidate_frames("ordinary", 811, 2))
    candidates_path = tmp_path / "candidates.xyz"
    manifest = tmp_path / "candidate-pool.json"
    write_candidate_pool(
        candidates_path,
        manifest,
        candidates,
        generation=3,
        model_path=first_model,
        requested_md_runs=4,
        available_md_runs=4,
        scheduled_md_runs=4,
        failed_md_runs=0,
    )

    with pytest.raises(CandidatePoolError, match="another sampling model"):
        validate_candidate_pool(
            ase_read(candidates_path, index=":"),
            manifest,
            generation=3,
            model_path=second_model,
        )


def test_stratified_fps_is_input_order_independent_with_stratum_anchors():
    points = np.asarray(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]
    )
    candidate_ids = ["a1", "a2", "a3", "b1", "b2", "b3"]
    strata = ["A", "A", "A", "B", "B", "B"]
    plan = GenerationPlan(1, 1, 4)
    first = stratified_farthest_point_sampling(
        points, np.asarray([[0.0, 0.0]]), candidate_ids, strata, plan
    )
    order = [5, 2, 4, 1, 3, 0]
    second = stratified_farthest_point_sampling(
        points[order],
        np.asarray([[0.0, 0.0]]),
        [candidate_ids[index] for index in order],
        [strata[index] for index in order],
        plan,
    )

    assert first.selected_ids == second.selected_ids
    assert first.counts_by_stratum == {"A": 3, "B": 1}


def test_zero_min_novelty_rejects_candidates_duplicating_reference():
    plan = GenerationPlan(1, 1, 1, selection_novelty_threshold=0.0)
    result = stratified_farthest_point_sampling(
        np.asarray([[1.0, 1.0], [1.0, 1.0]]),
        np.asarray([[1.0, 1.0]]),
        ["candidate-a", "candidate-b"],
        ["T=10", "T=10"],
        plan,
    )

    assert result.selected_ids == ()


def test_two_generation_toy_workflow_is_deterministic_and_resumable(tmp_path: Path):
    report = run_toy_iteration_smoke(
        tmp_path / "iteration",
        profile="spin",
        generations=2,
        max_selected=8,
    )

    assert report.passed
    assert report.generations_completed == 2
    assert report.max_selected == (8, 8)
    assert report.steps == (100, 100)
    assert report.selected_counts == (8, 8)
    assert report.training_counts[1] > report.training_counts[0]
    assert report.coverage_p95[1] < report.coverage_p95[0]
    assert report.scenario_steps == ((100,), (100, 400))
    assert report.maturity_counts == (
        {"smoke_passed": 1},
        {"smoke_passed": 1, "short_stable": 1},
    )
    assert report.deterministic_selection
    assert report.resume_reused_artifacts
    ledger = json.loads(
        (tmp_path / "iteration/workflow/workflow-ledger.json").read_text(
            encoding="utf-8"
        )
    )
    assert ledger["generations"]["2"]["stages"]["train"]["metrics"][
        "reused_previous_model"
    ] is True


def test_toy_generations_keep_all_temperature_strata_visible(tmp_path: Path):
    report = run_toy_iteration_smoke(
        tmp_path / "iteration-three",
        profile="spin",
        generations=3,
        max_selected=8,
    )

    assert report.passed
    assert report.steps == (100, 100, 100)
    assert set(report.strata_counts[2]) == {
        "T=300|P=0",
        "T=500|P=0",
        "T=700|P=0",
    }
    assert all(value >= 1 for value in report.strata_counts[2].values())
    assert report.strata_counts[2]["T=700|P=0"] > report.strata_counts[2][
        "T=300|P=0"
    ]
    assert report.scenario_steps[2] == (100, 400, 1600)
    assert report.maturity_counts[2] == {
        "smoke_passed": 1,
        "short_stable": 1,
        "long_stable": 1,
    }


def test_controller_rejects_completed_artifact_drift(tmp_path: Path):
    root = tmp_path / "drift"
    run_toy_iteration_smoke(root, profile="spin", generations=1, seed=17)
    plan = (ToyGenerationPlan(1, 17, 8),)
    adapter = ToyIterationAdapter(
        profile="spin",
        initial_training=root / "initial-train.xyz",
        validation=root / "validation.xyz",
    )
    controller = GenerationController(root / "workflow", "toy-spin-17")
    summary = controller.run_workflow(plan, adapter)[0]
    summary.artifacts["selection_result"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(IterationError, match="artifact drifted"):
        GenerationController(root / "workflow", "toy-spin-17").run_workflow(plan, adapter)


def test_ledger_records_plan_and_stage_artifact_hashes(tmp_path: Path):
    run_toy_iteration_smoke(tmp_path / "iteration", generations=1)
    ledger = json.loads(
        (tmp_path / "iteration/workflow/workflow-ledger.json").read_text(encoding="utf-8")
    )

    generation = ledger["generations"]["1"]
    assert generation["complete"] is True
    assert generation["accepted"] is True
    assert list(generation["stages"]) == [
        "diagnose",
        "evaluate",
        "explore",
        "label",
        "merge",
        "retrain",
        "select",
        "train",
    ]
    assert all(
        artifact["sha256"]
        for stage in generation["stages"].values()
        for artifact in stage["artifacts"].values()
    )


def test_rejected_generation_stops_before_next_plan(tmp_path: Path):
    class RejectingAdapter:
        def run_stage(self, stage, context):
            artifact = context.generation_dir / f"{stage}.json"
            artifact.write_text("{}\n", encoding="utf-8")
            metrics = {"accepted": False} if stage == "evaluate" else {}
            return StageOutcome({f"{stage}_artifact": artifact}, metrics)

    plans = progressive_plans(2)
    summaries = GenerationController(
        tmp_path / "rejected", "rejected"
    ).run_workflow(plans, RejectingAdapter())

    assert len(summaries) == 1
    assert summaries[0].accepted is False
    assert not (tmp_path / "rejected/Generation-2").exists()


def test_rejected_generation_can_reopen_from_retrain_with_history(tmp_path: Path):
    class RecoveringAdapter:
        evaluations = 0

        def run_stage(self, stage, context):
            attempt = self.evaluations + 1
            artifact = context.generation_dir / f"{stage}-attempt-{attempt}.json"
            artifact.write_text("{}\n", encoding="utf-8")
            metrics = {}
            if stage == "evaluate":
                self.evaluations += 1
                metrics["accepted"] = self.evaluations > 1
            return StageOutcome({f"{stage}_artifact": artifact}, metrics)

    plan = progressive_plans(1)[0]
    root = tmp_path / "recover-rejected"
    adapter = RecoveringAdapter()
    controller = GenerationController(root, "recover-rejected")

    first = controller.run_generation(plan, adapter)
    assert first.accepted is False
    controller.reopen_rejected(plan)
    assert controller.next_stage(plan) == "retrain"

    recovered = controller.run_generation(plan, adapter)
    assert recovered.accepted is True
    ledger = json.loads((root / "workflow-ledger.json").read_text())
    generation = ledger["generations"]["1"]
    assert generation["recovery_attempts"][0]["from_stage"] == "retrain"
    assert generation["recovery_attempts"][0]["stages"]["evaluate"][
        "metrics"
    ]["accepted"] is False
    assert generation["stages"]["evaluate"]["metrics"]["accepted"] is True


def test_incomplete_generation_can_restart_from_reached_stage(tmp_path: Path):
    class Adapter:
        def run_stage(self, stage, context):
            artifact = context.generation_dir / f"{stage}.json"
            artifact.write_text("{}\n", encoding="utf-8")
            return StageOutcome({f"{stage}_artifact": artifact}, {})

    plan = progressive_plans(1)[0]
    root = tmp_path / "restart-incomplete"
    controller = GenerationController(root, "restart-incomplete")
    adapter = Adapter()
    for expected in ("train", "explore", "select", "label"):
        assert controller.next_stage(plan) == expected
        controller.run_stage(plan, adapter)

    assert controller.next_stage(plan) == "diagnose"
    controller.reopen_from(plan, from_stage="select")
    assert controller.next_stage(plan) == "select"

    ledger = json.loads((root / "workflow-ledger.json").read_text())
    generation = ledger["generations"]["1"]
    assert set(generation["stages"]) == {"train", "explore"}
    recovery = generation["recovery_attempts"][0]
    assert recovery["from_stage"] == "select"
    assert set(recovery["stages"]) == {"select", "label"}


def test_controller_rejects_plan_change_after_generation_started(tmp_path: Path):
    class AcceptingAdapter:
        def run_stage(self, stage, context):
            artifact = context.generation_dir / f"{stage}.json"
            artifact.write_text("{}\n", encoding="utf-8")
            metrics = {"accepted": True} if stage == "evaluate" else {}
            return StageOutcome({f"{stage}_artifact": artifact}, metrics)

    controller = GenerationController(tmp_path / "plan-drift", "plan-drift")
    original = GenerationPlan(1, 1, 2)
    controller.run_workflow((original,), AcceptingAdapter())
    changed = GenerationPlan(
        1, 1, 2, completion_coverage_threshold=0.1
    )

    with pytest.raises(IterationError, match="plan changed"):
        controller.run_workflow((changed,), AcceptingAdapter())


def test_route_identity_change_starts_fresh_scenario_history(tmp_path: Path):
    initial = tmp_path / "initial.xyz"
    template = tmp_path / "lammps.in"
    ase_write(
        initial,
        Atoms("Fe", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True),
        format="extxyz",
    )
    template.write_text("run {{ steps }}\n", encoding="utf-8")
    adapter = WorkflowIterationAdapter(
        {
            "training": {"backend": "gpumd", "config_path": str(template)},
            "md": {"backend": "lammps", "spin": False},
            "sampling": _sampling(
                (300.0,), structures=initial, template=template
            ),
            "labeling": {"backend": "toy"},
        },
        initial_training=initial,
    )
    route = adapter.routes[0]

    def write_json(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    histories, _ = adapter._record_route_histories(
        StageContext(
            generation=2,
            generation_dir=tmp_path,
            plan=GenerationPlan(2, 2, 1),
            artifacts={
                "scenario_plan": write_json(
                    "scenario-plan.json",
                    {
                        "version": 3,
                        "routes": [
                            {
                                "route_id": route.route_id,
                                "route_fingerprint": route.fingerprint,
                                "template_sha256": route.template_sha256,
                                "structure_ids": [],
                                "pressure": 0.0,
                                "attempts": [],
                                "completed": {},
                            }
                        ],
                    },
                ),
                "selection_result": write_json(
                    "selection-result.json",
                    {"remaining_novelty_by_condition": {}},
                ),
            },
            previous_artifacts={
                "scenario_maturity": write_json(
                    "previous-scenario-maturity.json",
                    {
                        "version": 3,
                        "routes": {
                            route.route_id: {
                                "route_fingerprint": "former-route",
                                "history": {"must_not_be_reused": True},
                            }
                        },
                    },
                )
            },
            stage_dir=tmp_path / "evaluate",
        ),
        diagnostic={"attempt_accepted": {}},
        validation_metrics=None,
        evidence_validation=None,
        validation_accepted=None,
        model_improved=True,
        novelty_converged=False,
        final_model_id="model-v2",
    )

    record = histories["routes"][route.route_id]
    assert record["route_fingerprint"] == route.fingerprint
    assert "must_not_be_reused" not in record["history"]

    attempts = adapter.plan_explore_attempts(
        StageContext(
            generation=2,
            generation_dir=tmp_path,
            plan=GenerationPlan(2, 2, 1),
            artifacts={"model": initial},
            previous_artifacts={
                "scenario_maturity": write_json(
                    "planning-scenario-maturity.json",
                    {
                        "version": 3,
                        "routes": {
                            route.route_id: {
                                "route_fingerprint": "former-route",
                                "history": {"must_not_be_reused": True},
                            }
                        },
                    },
                )
            },
            stage_dir=tmp_path / "explore",
        )
    )
    assert attempts
    assert {item["route_fingerprint"] for item in attempts} == {
        route.fingerprint
    }


def test_controller_runs_one_resource_stage_at_a_time(tmp_path: Path):
    class AcceptingAdapter:
        def run_stage(self, stage, context):
            artifact = context.generation_dir / f"{stage}.json"
            artifact.write_text("{}\n", encoding="utf-8")
            metrics = {"accepted": True} if stage == "evaluate" else {}
            return StageOutcome({f"{stage}_artifact": artifact}, metrics)

    plan = progressive_plans(1)[0]
    root = tmp_path / "stage-by-stage"
    for expected in (
        "train",
        "explore",
        "select",
        "label",
        "diagnose",
        "merge",
        "retrain",
        "evaluate",
    ):
        controller = GenerationController(root, "split-resources")
        assert controller.next_stage(plan) == expected
        result = controller.run_stage(plan, AcceptingAdapter(), expected)
        assert result.stage == expected
    assert result.generation_complete
    assert result.accepted is True
    assert GenerationController(root, "split-resources").next_stage(plan) is None


def test_controller_rejects_out_of_order_resource_stage(tmp_path: Path):
    plan = progressive_plans(1)[0]
    controller = GenerationController(tmp_path / "out-of-order", "out-of-order")

    with pytest.raises(IterationError, match="expects stage train, not explore"):
        controller.run_stage(plan, object(), "explore")


def test_stale_controller_cannot_repeat_a_completed_stage(tmp_path: Path):
    class Adapter:
        def run_stage(self, stage, context):
            artifact = context.generation_dir / f"{stage}.json"
            artifact.write_text("{}\n", encoding="utf-8")
            return StageOutcome({f"{stage}_artifact": artifact})

    plan = progressive_plans(1)[0]
    root = tmp_path / "stale-controller"
    first = GenerationController(root, "stale-controller")
    stale = GenerationController(root, "stale-controller")
    first.run_stage(plan, Adapter(), "train")

    with pytest.raises(IterationError, match="expects stage explore, not train"):
        stale.run_stage(plan, Adapter(), "train")


def test_workflow_adapter_connects_real_stage_contracts_with_toy_teacher(tmp_path: Path):
    initial_frames = [
        ToyTeacher("spin").label(frame)
        for frame in toy_candidate_frames("spin", 7, 3)
    ]
    initial = tmp_path / "initial.xyz"
    structure = tmp_path / "structure.xyz"
    validation = tmp_path / "validation.xyz"
    ase_write(initial, initial_frames, format="extxyz")
    ase_write(structure, initial_frames[0], format="extxyz")
    validation_frames = [
        ToyTeacher("spin").label(frame)
        for frame in toy_candidate_frames("spin", 991, 2)
    ]
    ase_write(validation, validation_frames, format="extxyz")
    config_file = tmp_path / "nep.in"
    config_file.write_text(
        "type 1 Fe\nspin_descriptor spin_nep_lite\nlr 0.003\n",
        encoding="utf-8",
    )

    calls = []
    training_requests = []

    def fake_train(request, backend):
        calls.append(("train", backend, request.device))
        training_requests.append(request)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        model = request.output_dir / "nep.txt"
        checkpoint = request.output_dir / "checkpoint.pt"
        model.write_text(
            f"fake spin_nep_lite model {len(training_requests)}\n",
            encoding="utf-8",
        )
        checkpoint.write_text("checkpoint\n", encoding="utf-8")
        return TrainingResult(backend, model, None, checkpoint)

    def fake_md(request, backend):
        calls.append(("md", backend, request.temperature, request.steps))
        request.output_dir.mkdir(parents=True, exist_ok=True)
        frames = toy_candidate_frames(
            "spin", int(request.temperature), 4, temperatures=(request.temperature,)
        )
        if request.temperature == 500.0:
            for index, frame in enumerate(frames):
                frame.info["md_window"] = (
                    "bad_tail" if index == 3 else "pre_failure" if index >= 1 else "stable_prefix"
                )
                frame.info["md_completed"] = False
        ase_write(request.output_file, frames, format="extxyz")
        health_report = None
        if request.temperature == 500.0:
            health_report = request.output_dir / "trajectory-health.json"
            health_report.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "trajectory_completed": False,
                        "first_bad_step": 30,
                        "reason_codes": ["min_distance_ratio_below_min"],
                    }
                ),
                encoding="utf-8",
            )
        return MdResult(
            backend,
            request.output_file,
            request.output_dir,
            "cpu",
            "nep/cpu",
            completed=request.temperature != 500.0,
            last_step=40 if request.temperature == 300.0 else 30,
            failure_code=None if request.temperature == 300.0 else "lammps_nonzero_exit",
            failure_reason=None if request.temperature == 300.0 else "lost atoms",
            health_report=health_report,
        )

    def fake_descriptors(_model, frames):
        return toy_raw_features(frames, "spin")

    def fake_predict(model, frames, backend):
        calls.append(("predict", backend, len(frames)))
        candidate = model.read_text(encoding="utf-8").strip().endswith("2")
        scale = 1.0 if candidate or len(frames) != 3 else 10.0
        return PredictionEvaluation(
            {
                "energy_rmse": 0.1 * scale,
                "force_rmse": 0.2 * scale,
                "virial_rmse": 0.3 * scale,
                "mforce_rmse": 0.4 * scale,
            },
            {
                "energy": ParitySeries(
                    np.asarray([0.0, 1.0]),
                    np.asarray([0.1, 0.9]),
                    "eV",
                )
            },
        )

    config = {
        "training": {
            "backend": "torchnep",
            "config_path": str(config_file),
            "device": "cuda",
        },
        "md": {
            "backend": "lammps",
            "spin": True,
            "mpi_ranks": 1,
        },
        "sampling": _sampling(
            (300.0, 500.0),
            structures=structure,
            template=config_file,
            steps=(40, 160, 640, 2560),
            max_selected=3,
        ),
        "labeling": {"backend": "toy"},
        "evaluation": {
            "inference_backend": "cpu",
            "validation_path": str(validation),
            "max_rmse": {
                "energy_rmse": 0.5,
                "force_rmse": 0.5,
                "virial_rmse": 0.5,
                "mforce_rmse": 0.5,
            },
        },
        "workflow": {},
    }
    config["sampling"]["selection"]["novelty"] = {
        "selection_threshold": 0.0,
        "completion_threshold": 0.0,
    }
    runtime = WorkflowRuntime(
        train=fake_train,
        md=fake_md,
        descriptors=fake_descriptors,
        predict=fake_predict,
    )
    adapter = WorkflowIterationAdapter(
        config, initial_training=initial, runtime=runtime
    )
    plan = GenerationPlan(1, 19, 3)
    summary = GenerationController(
        tmp_path / "workflow", "real-contract-toy-labels"
    ).run_generation(plan, adapter)

    assert summary.accepted
    assert summary.metrics["train"]["backend"] == "torchnep"
    assert summary.metrics["explore"]["candidate_count"] == 4
    assert summary.metrics["explore"]["source_count"] == 1
    assert summary.metrics["explore"]["completed_source_count"] == 1
    assert summary.metrics["explore"]["failed_source_count"] == 0
    assert summary.metrics["select"]["selected_count"] == 3
    assert summary.metrics["select"]["candidate_count_after_deduplication"] == 4
    assert summary.metrics["label"]["backend"] == "toy"
    assert summary.metrics["diagnose"]["current_model_mforce_rmse"] == 4.0
    assert summary.metrics["merge"]["added_count"] == 3
    assert summary.metrics["retrain"]["training_count"] == 6
    assert summary.artifacts["retraining_config"].read_text(encoding="utf-8").endswith(
        "lr 0.0003\n"
    )
    assert summary.metrics["evaluate"]["added_training_count"] == 3
    assert summary.metrics["evaluate"]["mforce_rmse"] == 0.4
    assert "evaluation_parity" in summary.artifacts
    assert "evaluation_parity_report" in summary.artifacts
    assert summary.metrics["evaluate"]["model_trained_on_current_labels"] is True
    assert summary.metrics["explore"]["scenario_targets"] == ["smoke_passed"]
    assert summary.metrics["evaluate"]["scenario_counts_by_maturity"] == {
        "untested": 1,
    }
    assert "scenario_maturity" in summary.artifacts
    assert "md_attempts" in summary.artifacts
    md_attempts = json.loads(summary.artifacts["md_attempts"].read_text())
    assert all(item["completed"] for item in md_attempts["attempts"])
    assert len([call for call in calls if call[0] == "train"]) == 2
    assert len([call for call in calls if call[0] == "predict"]) == 6
    assert calls[0] == ("train", "torchnep", "cuda")
    assert training_requests[0].config_file == config_file
    assert training_requests[1].config_file.name == "torchnep-finetune.in"
    assert "lr 0.0003\n" in training_requests[1].config_file.read_text(
        encoding="utf-8"
    )
    assert ("md", "lammps", 300.0, 40) in calls
    assert ("md", "lammps", 500.0, 40) not in calls

    flat_md_root = tmp_path / "flat-md-output"
    flat_md_context = StageContext(
        generation=1,
        generation_dir=flat_md_root,
        plan=plan,
        artifacts={"model": summary.artifacts["model"]},
        previous_artifacts={},
        stage_dir=flat_md_root,
        flat_output=True,
    )
    attempts = adapter.plan_explore_attempts(flat_md_context)
    assert len(attempts) == 1
    adapter.run_stage(
        "explore",
        StageContext(
            generation=1,
            generation_dir=flat_md_root,
            plan=plan,
            artifacts={"model": summary.artifacts["model"]},
            previous_artifacts={},
            stage_dir=flat_md_root,
            stage_input={"attempt_ids": [attempts[0]["attempt_id"]]},
            flat_output=True,
        ),
    )
    assert (flat_md_root / "trajectory.xyz").is_file()
    assert not (flat_md_root / "calculations").exists()
    assert not (flat_md_root / "md").exists()

    overlap_config = {
        **config,
        "evaluation": {
            **config["evaluation"],
            "validation_path": str(initial),
        },
    }
    overlap_adapter = WorkflowIterationAdapter(
        overlap_config, initial_training=initial, runtime=runtime
    )
    with pytest.raises(WorkflowIterationError, match="overlaps"):
        GenerationController(
            tmp_path / "overlap-workflow", "overlap"
        ).run_generation(plan, overlap_adapter)

    unsafe_config = {
        **config,
        "evaluation": {
            "validation_path": str(validation),
            "max_rmse": {},
        },
    }
    with pytest.raises(WorkflowIterationError, match="mforce_rmse"):
        WorkflowIterationAdapter(
            unsafe_config, initial_training=initial, runtime=runtime
        )

    fallback_config = {
        **config,
        "training": {
            **config["training"],
            "test_path": str(validation),
        },
        "evaluation": {
            key: value
            for key, value in config["evaluation"].items()
            if key != "validation_path"
        },
    }
    with pytest.raises(
        WorkflowIterationError, match="evaluation.validation_path"
    ):
        WorkflowIterationAdapter(
            fallback_config, initial_training=initial, runtime=runtime
        )
    assert all(request.test_file is None for request in training_requests)


def test_torchnep_restart_false_keeps_full_training_config(tmp_path: Path):
    training_input = tmp_path / "train.xyz"
    ase_write(
        training_input,
        Atoms("Fe", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True),
        format="extxyz",
    )
    config_file = tmp_path / "nep.in"
    config_file.write_text("type 1 Fe\nlr 0.01\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    requests = []

    def fake_train(request, backend):
        requests.append(request)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        model = request.output_dir / "nep.txt"
        model.write_text("nep4 1 Fe\n", encoding="utf-8")
        return TrainingResult(backend, model, None, None)

    adapter = WorkflowIterationAdapter(
        {
            "training": {
                "backend": "torchnep",
                "config_path": str(config_file),
                "restart": False,
            }
        },
        initial_training=training_input,
        runtime=WorkflowRuntime(train=fake_train),
        active_stage="retrain",
    )
    stage_dir = tmp_path / "retrain"
    _, frame_count, selected_config = adapter._execute_training(
        StageContext(
            generation=1,
            generation_dir=tmp_path / "generation",
            plan=GenerationPlan(1, 7, 1),
            artifacts={},
            previous_artifacts={},
            stage_dir=stage_dir,
        ),
        training_input=training_input,
        role="retraining",
        warm_start=checkpoint,
    )

    assert frame_count == 1
    assert selected_config == config_file
    assert requests[0].config_file == config_file
    assert requests[0].finetune_file is None
    assert not (stage_dir / "retraining" / "torchnep-finetune.in").exists()


def test_workflow_selection_deduplicates_frames_and_keeps_pre_failure(
    tmp_path: Path,
):
    initial_frames = [
        ToyTeacher("spin").label(frame)
        for frame in toy_candidate_frames("spin", 11, 2)
    ]
    initial = tmp_path / "initial.xyz"
    validation = tmp_path / "validation.xyz"
    ase_write(initial, initial_frames, format="extxyz")
    ase_write(
        validation,
        [ToyTeacher("spin").label(toy_candidate_frames("spin", 12, 1)[0])],
        format="extxyz",
    )
    config_file = tmp_path / "nep.in"
    config_file.write_text(
        "type 1 Fe\nspin_descriptor spin_nep_lite\n", encoding="utf-8"
    )
    config = {
        "training": {"backend": "torchnep", "config_path": str(config_file)},
        "md": {"backend": "lammps", "spin": True},
        "sampling": _sampling(
            (100.0,),
            structures=initial,
            template=config_file,
            max_selected=20,
        ),
        "labeling": {"backend": "toy"},
        "evaluation": {
            "validation_path": str(validation),
            "max_rmse": {
                "energy_rmse": 1.0,
                "force_rmse": 1.0,
                "mforce_rmse": 1.0,
            },
        },
    }
    config["sampling"]["selection"]["novelty"] = {
        "selection_threshold": 0.0,
        "completion_threshold": 0.0,
    }
    runtime = WorkflowRuntime(
        descriptors=lambda _model, frames: toy_raw_features(frames, "spin")
    )
    adapter = WorkflowIterationAdapter(
        config, initial_training=initial, runtime=runtime
    )

    first, repeated = toy_candidate_frames("spin", 91, 2)
    first.info.update(
        source_id="healthy", temperature=100.0, pressure=0.0, md_window="stable_prefix"
    )
    stable_duplicate = repeated.copy()
    stable_duplicate.info.update(
        source_id="healthy",
        temperature=100.0,
        pressure=0.0,
        md_window="stable_prefix",
        route_id="stable-route",
        route_fingerprint="e" * 64,
    )
    preferred_duplicate = repeated.copy()
    preferred_duplicate.info.update(
        source_id="failed",
        temperature=100.0,
        pressure=0.0,
        md_window="pre_failure",
        route_id="failure-route",
        route_fingerprint="d" * 64,
    )
    candidates_path = tmp_path / "candidates.xyz"
    ase_write(
        candidates_path,
        [first, stable_duplicate, preferred_duplicate],
        format="extxyz",
    )
    model = tmp_path / "nep.txt"
    model.write_text("fake\n", encoding="utf-8")
    manifest = tmp_path / "candidate-pool.json"
    write_candidate_pool(
        candidates_path,
        manifest,
        _with_route(ase_read(candidates_path, index=":")),
        generation=1,
        model_path=model,
        requested_md_runs=4,
        available_md_runs=2,
        scheduled_md_runs=2,
        failed_md_runs=1,
    )
    context = StageContext(
        generation=1,
        generation_dir=tmp_path,
        plan=GenerationPlan(1, 1, 20),
        artifacts={
            "candidates": candidates_path,
            "candidate_pool_manifest": manifest,
            "training_input": initial,
            "model": model,
        },
        previous_artifacts={},
    )

    outcome = adapter._select(context)
    selected = ase_read(outcome.artifacts["selected_input"], index=":")

    assert outcome.metrics["candidate_count_after_deduplication"] == 2
    assert outcome.metrics["duplicate_candidate_count"] == 1
    assert outcome.metrics["selected_count"] == 2
    assert outcome.metrics["configured_max_selected"] == 20
    assert "pre_failure" in {frame.info["md_window"] for frame in selected}
    assert {
        frame.info["route_id"]
        for frame in selected
        if frame.info["md_window"] == "pre_failure"
    } == {"failure-route"}


def test_no_novel_candidates_skip_label_backend_and_retraining(tmp_path: Path):
    frame = ToyTeacher("ordinary").label(
        toy_candidate_frames("ordinary", 41, 1)[0]
    )
    initial = tmp_path / "initial.xyz"
    candidates = tmp_path / "candidates.xyz"
    model = tmp_path / "nep.txt"
    manifest = tmp_path / "candidate-pool.json"
    ase_write(initial, [frame], format="extxyz")
    candidate = frame.copy()
    candidate.info.update(
        route_id="default",
        route_fingerprint="f" * 64,
        temperature=300.0,
        pressure=0.0,
        source_id="same-structure",
    )
    ase_write(candidates, [candidate], format="extxyz")
    model.write_text("model\n", encoding="utf-8")
    write_candidate_pool(
        candidates,
        manifest,
        [candidate],
        generation=1,
        model_path=model,
        requested_md_runs=1,
        available_md_runs=1,
        scheduled_md_runs=1,
        failed_md_runs=0,
    )
    config_file = tmp_path / "nep.in"
    config_file.write_text("type 1 Fe\n", encoding="utf-8")
    config = {
        "training": {"backend": "torchnep", "config_path": str(config_file)},
        "md": {"backend": "lammps", "spin": False},
        "sampling": _sampling(
            (300.0,), structures=initial, template=config_file
        ),
        "labeling": {"backend": "vasp"},
    }

    def forbidden_label(*_args, **_kwargs):
        raise AssertionError("label backend must not run for an empty batch")

    def forbidden_train(*_args, **_kwargs):
        raise AssertionError("training backend must not run without new labels")

    adapter = WorkflowIterationAdapter(
        config,
        initial_training=initial,
        runtime=WorkflowRuntime(
            label=forbidden_label,
            train=forbidden_train,
            descriptors=lambda _model, frames: toy_raw_features(
                frames, "ordinary"
            ),
        ),
    )
    plan = GenerationPlan(1, 1, 10)
    selected = adapter._select(
        StageContext(
            1,
            tmp_path,
            plan,
            {
                "candidates": candidates,
                "candidate_pool_manifest": manifest,
                "training_input": initial,
                "model": model,
            },
            {},
        )
    )
    assert selected.metrics["selected_count"] == 0
    assert selected.metrics["batch_kind"] == "coverage_complete"
    assert selected.metrics["descriptor_reduction"] == "global_mean"
    assert selected.metrics["descriptor_elements"] == ["Fe"]

    label_dir = tmp_path / "label"
    label_dir.mkdir()
    labeled = adapter._label(
        StageContext(
            1,
            tmp_path,
            plan,
            {**selected.artifacts, "model": model},
            {},
            stage_dir=label_dir,
        )
    )
    assert labeled.metrics["skipped"] is True
    assert labeled.artifacts["labeled"].stat().st_size == 0

    diagnose_dir = tmp_path / "diagnose"
    diagnose_dir.mkdir()
    diagnosed = adapter._diagnose(
        StageContext(
            1,
            tmp_path,
            plan,
            {"labeled": labeled.artifacts["labeled"], "model": model},
            {},
            stage_dir=diagnose_dir,
        )
    )
    merge_dir = tmp_path / "merge"
    merge_dir.mkdir()
    merged = adapter._merge(
        StageContext(
            1,
            tmp_path,
            plan,
            {
                "training_input": initial,
                "labeled": labeled.artifacts["labeled"],
            },
            {},
            stage_dir=merge_dir,
        )
    )
    attempts = tmp_path / "md-attempts.json"
    attempts.write_text(
        json.dumps({"attempts": [{"completed": True}]}), encoding="utf-8"
    )
    retrain_dir = tmp_path / "retrain"
    retrain_dir.mkdir()
    retrained = adapter._retrain(
        StageContext(
            1,
            tmp_path,
            plan,
            {
                "training_input": initial,
                "training_set": merged.artifacts["training_set"],
                "model": model,
                "acquisition_signals": diagnosed.artifacts[
                    "acquisition_signals"
                ],
                "md_attempts": attempts,
                "label_provenance": labeled.artifacts["label_provenance"],
            },
            {},
            stage_dir=retrain_dir,
        )
    )
    assert retrained.metrics["retrained"] is False
    assert retrained.metrics["model_updated"] is False


@pytest.mark.parametrize(
    "descriptor_reduction",
    ["global_mean", "elementwise_mean_std"],
)
def test_workflow_selection_describes_every_unique_valid_dump_frame(
    tmp_path: Path,
    descriptor_reduction: str,
):
    initial = tmp_path / "initial.xyz"
    candidates_path = tmp_path / "candidates.xyz"
    model = tmp_path / "nep.txt"
    initial_frame = ToyTeacher("ordinary").label(
        toy_candidate_frames("ordinary", 401, 1)[0]
    )
    candidates = _with_route(toy_candidate_frames("ordinary", 402, 20))
    for index, frame in enumerate(candidates):
        frame.info.update(
            source_id=f"source-{index % 2}",
            temperature=300.0,
            pressure=0.0,
            md_window="stable_prefix",
        )
    ase_write(initial, [initial_frame], format="extxyz")
    ase_write(candidates_path, candidates, format="extxyz")
    model.write_text("fake\n", encoding="utf-8")
    manifest = tmp_path / "candidate-pool.json"
    write_candidate_pool(
        candidates_path,
        manifest,
        candidates,
        generation=1,
        model_path=model,
        requested_md_runs=4,
        available_md_runs=4,
        scheduled_md_runs=4,
        failed_md_runs=0,
    )

    descriptor_counts = []
    atomic_descriptor_counts = []

    def descriptors(_model, frames):
        descriptor_counts.append(len(frames))
        return toy_raw_features(frames, "ordinary")

    def atomic_descriptors(_model, frames):
        atomic_descriptor_counts.append(len(frames))
        rows = toy_raw_features(frames, "ordinary")
        return np.vstack(
            [
                np.repeat(row[None, :], len(frame), axis=0)
                for frame, row in zip(frames, rows, strict=True)
            ]
        )

    sampling = _sampling(
        structures=initial,
        template=initial,
        max_selected=4,
    )
    sampling["selection"]["descriptor_reduction"] = descriptor_reduction
    adapter = WorkflowIterationAdapter(
        {
            "training": {"backend": "gpumd"},
            "md": {"backend": "lammps", "spin": False},
            "sampling": sampling,
            "labeling": {"backend": "toy"},
            "evaluation": {
                "validation_path": str(initial),
                "max_rmse": {"energy_rmse": 1.0, "force_rmse": 1.0},
            },
        },
        initial_training=initial,
        runtime=WorkflowRuntime(
            descriptors=descriptors,
            atomic_descriptors=atomic_descriptors,
        ),
    )
    outcome = adapter._select(
        StageContext(
            generation=1,
            generation_dir=tmp_path,
            plan=GenerationPlan(1, 1, 4),
            artifacts={
                "candidates": candidates_path,
                "candidate_pool_manifest": manifest,
                "training_input": initial,
                "model": model,
            },
            previous_artifacts={},
        )
    )

    if descriptor_reduction == "global_mean":
        assert descriptor_counts == [20, 1]
        assert atomic_descriptor_counts == []
    else:
        assert descriptor_counts == []
        assert atomic_descriptor_counts == [20, 1]
    assert outcome.metrics["candidate_count_before_deduplication"] == 20
    assert outcome.metrics["candidate_count_after_deduplication"] == 20
    assert outcome.metrics["selected_count"] == 4
    assert outcome.metrics["descriptor_reduction"] == descriptor_reduction
    assert outcome.metrics["descriptor_elements"] == ["Fe"]


def test_explore_accumulates_same_model_md_waves_to_the_derived_floor(
    tmp_path: Path,
):
    initial = tmp_path / "initial.xyz"
    validation = tmp_path / "validation.xyz"
    structures = tmp_path / "structures.xyz"
    config_file = tmp_path / "nep.in"
    model = tmp_path / "nep.txt"
    teacher = ToyTeacher("ordinary")
    ase_write(
        initial,
        [teacher.label(toy_candidate_frames("ordinary", 901, 1)[0])],
        format="extxyz",
    )
    ase_write(
        validation,
        [teacher.label(toy_candidate_frames("ordinary", 902, 1)[0])],
        format="extxyz",
    )
    starts = toy_candidate_frames("ordinary", 903, 3)
    ase_write(structures, starts, format="extxyz")
    config_file.write_text("type 1 Fe\n", encoding="utf-8")
    model.write_text("active model\n", encoding="utf-8")
    md_calls = []

    def one_frame_md(request, backend):
        md_calls.append(request.seed)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        frame = request.atoms.copy()
        frame.info["lammps_step"] = request.steps
        ase_write(request.output_file, [frame], format="extxyz")
        return MdResult(
            backend,
            request.output_file,
            request.output_dir,
            "cpu",
            "nep/cpu",
            completed=True,
            last_step=request.steps,
        )

    adapter = WorkflowIterationAdapter(
        {
            "training": {"backend": "gpumd", "config_path": str(config_file)},
            "md": {
                "backend": "lammps",
                "spin": False,
            },
            "sampling": _sampling(
                (300.0,),
                structures=structures,
                template=config_file,
                max_selected=4,
            ),
            "labeling": {"backend": "toy"},
            "evaluation": {
                "validation_path": str(validation),
                "max_rmse": {"energy_rmse": 1.0, "force_rmse": 1.0},
            },
        },
        initial_training=initial,
        runtime=WorkflowRuntime(md=one_frame_md),
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

    assert len(md_calls) == 3
    assert outcome.metrics["unique_candidate_count"] == 3
    assert outcome.metrics["regular_batch_minimum"] == 2
    manifest = json.loads(
        outcome.artifacts["candidate_pool_manifest"].read_text()
    )
    assert manifest["model_sha256"] == outcome.metrics["sampling_model_sha256"]
    assert manifest["scheduled_md_runs"] == 3
    assert manifest["available_md_runs"] == 3


def test_next_md_round_uses_the_newly_published_model(tmp_path: Path):
    teacher = ToyTeacher("ordinary")
    initial = tmp_path / "initial.xyz"
    validation = tmp_path / "validation.xyz"
    structures = tmp_path / "structures.xyz"
    config_file = tmp_path / "nep.in"
    ase_write(
        initial,
        [teacher.label(toy_candidate_frames("ordinary", 921, 1)[0])],
        format="extxyz",
    )
    ase_write(
        validation,
        [teacher.label(toy_candidate_frames("ordinary", 922, 1)[0])],
        format="extxyz",
    )
    ase_write(
        structures,
        toy_candidate_frames("ordinary", 923, 1),
        format="extxyz",
    )
    config_file.write_text("type 1 Fe\n", encoding="utf-8")
    training_calls = []
    md_models = []

    def versioned_train(request, backend):
        version = len(training_calls) + 1
        training_calls.append(version)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        model = request.output_dir / "nep.txt"
        model.write_text(f"model-{version}\n", encoding="utf-8")
        return TrainingResult(backend, model, None, None)

    def model_recording_md(request, backend):
        md_models.append(request.model_file.read_text(encoding="utf-8").strip())
        request.output_dir.mkdir(parents=True, exist_ok=True)
        frames = toy_candidate_frames(
            "ordinary",
            930 + len(md_models),
            2,
            temperatures=(request.temperature,),
        )
        ase_write(request.output_file, frames, format="extxyz")
        return MdResult(
            backend,
            request.output_file,
            request.output_dir,
            "cpu",
            "nep/cpu",
            completed=True,
            last_step=request.steps,
        )

    def model_sensitive_errors(model, _frames, _backend):
        value = (
            0.5
            if model.read_text(encoding="utf-8").strip() == "model-2"
            else 2.0
        )
        return PredictionEvaluation(
            {"energy_rmse": value, "force_rmse": value}
        )

    adapter = WorkflowIterationAdapter(
        {
            "training": {"backend": "gpumd", "config_path": str(config_file)},
            "md": {
                "backend": "lammps",
                "spin": False,
            },
            "sampling": _sampling(
                (300.0,),
                structures=structures,
                template=config_file,
                max_selected=2,
            ),
            "labeling": {"backend": "toy"},
            "evaluation": {
                "validation_path": str(validation),
                "max_rmse": {"energy_rmse": 1.0, "force_rmse": 1.0},
            },
        },
        initial_training=initial,
        runtime=WorkflowRuntime(
            train=versioned_train,
            md=model_recording_md,
            descriptors=lambda _model, frames: toy_raw_features(
                frames, "ordinary"
            ),
            predict=model_sensitive_errors,
        ),
    )
    controller = GenerationController(tmp_path / "workflow", "model-handoff")
    first = controller.run_generation(
        GenerationPlan(1, 31, 2), adapter
    )
    controller.run_generation(
        GenerationPlan(2, 32, 2), adapter
    )

    assert first.metrics["evaluate"]["candidate_activation_accepted"] is True
    assert first.artifacts["activated_model"].read_text().strip() == "model-2"
    assert md_models == ["model-1", "model-2"]


@pytest.mark.parametrize(
    ("candidate_error", "candidate_accepted", "active_model_text"),
    (
        (0.5, True, "candidate"),
        (3.0, False, "parent"),
    ),
)
def test_candidate_model_must_pass_activation_before_the_next_round(
    tmp_path: Path,
    candidate_error: float,
    candidate_accepted: bool,
    active_model_text: str,
):
    teacher = ToyTeacher("ordinary")
    initial = tmp_path / "initial.xyz"
    labeled = tmp_path / "labeled.xyz"
    training_set = tmp_path / "train.xyz"
    validation = tmp_path / "validation.xyz"
    initial_frames = [
        teacher.label(toy_candidate_frames("ordinary", 941, 1)[0])
    ]
    labeled_frames = [
        teacher.label(toy_candidate_frames("ordinary", 942, 1)[0])
    ]
    ase_write(initial, initial_frames, format="extxyz")
    ase_write(labeled, labeled_frames, format="extxyz")
    ase_write(
        training_set,
        [*initial_frames, *labeled_frames],
        format="extxyz",
    )
    ase_write(
        validation,
        [teacher.label(toy_candidate_frames("ordinary", 943, 1)[0])],
        format="extxyz",
    )
    config_file = tmp_path / "nep.in"
    config_file.write_text("type 1 Fe\n", encoding="utf-8")
    parent = tmp_path / "parent.nep"
    candidate = tmp_path / "candidate.nep"
    parent.write_text("parent\n", encoding="utf-8")
    candidate.write_text("candidate\n", encoding="utf-8")
    parent_checkpoint = tmp_path / "parent.pt"
    candidate_checkpoint = tmp_path / "candidate.pt"
    parent_checkpoint.write_text("parent checkpoint\n", encoding="utf-8")
    candidate_checkpoint.write_text("candidate checkpoint\n", encoding="utf-8")

    def errors(model, _frames, _backend):
        value = (
            candidate_error
            if model.read_text(encoding="utf-8").strip() == "candidate"
            else 2.0
        )
        return PredictionEvaluation(
            {"energy_rmse": value, "force_rmse": value}
        )

    adapter = WorkflowIterationAdapter(
        {
            "training": {"backend": "gpumd", "config_path": str(config_file)},
            "md": {"backend": "lammps", "spin": False},
            "sampling": _sampling(
                (300.0,),
                structures=initial,
                template=config_file,
            ),
            "labeling": {"backend": "toy"},
            "evaluation": {
                "validation_path": str(validation),
                "max_rmse": {"energy_rmse": 1.0, "force_rmse": 1.0},
            },
        },
        initial_training=initial,
        runtime=WorkflowRuntime(predict=errors),
    )
    parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()

    def write_json(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    work_dir = tmp_path / "evaluate"
    work_dir.mkdir()
    outcome = adapter._evaluate(
        StageContext(
            generation=1,
            generation_dir=tmp_path,
            plan=GenerationPlan(1, 1, 2),
            artifacts={
                "training_input": initial,
                "training_set": training_set,
                "labeled": labeled,
                "model": parent,
                "checkpoint": parent_checkpoint,
                "retrained_model": candidate,
                "retrained_checkpoint": candidate_checkpoint,
                "retraining_decision": write_json(
                    "retraining-decision.json", {"retrained": True}
                ),
                "model_lineage": write_json(
                    "model-lineage.json",
                    {
                        "generation": 1,
                        "parent_model_sha256": parent_sha,
                        "candidate_model_sha256": candidate_sha,
                        "model_updated": True,
                        "trained_on_current_labels": True,
                        "training_dataset_sha256": hashlib.sha256(
                            training_set.read_bytes()
                        ).hexdigest(),
                        "training_count": 2,
                        "pending_label_count": 0,
                    },
                ),
                "acquisition_signals": write_json(
                    "acquisition-signals.json",
                    {
                        "attempt_accepted": {},
                        "attempt_metrics": {},
                    },
                ),
                "selection_result": write_json(
                    "selection-result.json", {"remaining_novelty": 0.0}
                ),
                "scenario_plan": write_json(
                        "scenario-plan.json",
                        {
                            "version": 3,
                            "model_id": parent_sha,
                            "routes": [
                                {
                                    "route_id": "default",
                                    "route_fingerprint": (
                                        adapter.routes[0].fingerprint
                                    ),
                                    "template_sha256": (
                                        adapter.routes[0].template_sha256
                                    ),
                                    "structure_ids": [],
                                    "pressure": 0.0,
                                    "attempts": [],
                                    "completed": {},
                                }
                            ],
                        },
                ),
            },
            previous_artifacts={},
            stage_dir=work_dir,
        )
    )

    assert outcome.metrics["accepted"] is True
    assert (
        outcome.metrics["candidate_activation_accepted"]
        is candidate_accepted
    )
    assert (
        outcome.artifacts["activated_model"].read_text().strip()
        == active_model_text
    )
    expected_checkpoint = (
        candidate_checkpoint if candidate_accepted else parent_checkpoint
    )
    assert outcome.artifacts["activated_checkpoint"] == expected_checkpoint
    next_round = adapter._train(
        StageContext(
            generation=2,
            generation_dir=tmp_path,
            plan=GenerationPlan(2, 2, 2),
            artifacts={},
            previous_artifacts={
                "training_set": training_set,
                "activated_model": outcome.artifacts["activated_model"],
                "activated_checkpoint": outcome.artifacts[
                    "activated_checkpoint"
                ],
                "active_model_lineage": outcome.artifacts[
                    "active_model_lineage"
                ],
            },
            stage_dir=tmp_path / "next-round",
        )
    )
    assert next_round.artifacts["model"] == outcome.artifacts["activated_model"]
    assert next_round.artifacts["checkpoint"] == expected_checkpoint


def test_retrain_updates_only_when_diagnostics_require_a_new_model(
    tmp_path: Path,
):
    initial_frames = [
        ToyTeacher("ordinary").label(
            toy_candidate_frames("ordinary", 31, 1)[0]
        )
    ]
    initial = tmp_path / "initial.xyz"
    validation = tmp_path / "validation.xyz"
    ase_write(initial, initial_frames, format="extxyz")
    ase_write(
        validation,
        [
            ToyTeacher("ordinary").label(
                toy_candidate_frames("ordinary", 32, 1)[0]
            )
        ],
        format="extxyz",
    )
    config_file = tmp_path / "nep.in"
    config_file.write_text("type 1 Fe\n", encoding="utf-8")

    training_calls = []

    def continuation_train(request, backend):
        training_calls.append((request, backend))
        request.output_dir.mkdir(parents=True, exist_ok=True)
        trained_model = request.output_dir / "nep.txt"
        trained_model.write_text("continued model\n", encoding="utf-8")
        return TrainingResult(backend, trained_model, None, None)

    adapter = WorkflowIterationAdapter(
        {
            "training": {
                "backend": "torchnep",
                "config_path": str(config_file),
            },
            "md": {"backend": "lammps", "spin": False},
            "sampling": _sampling(
                (300.0,),
                structures=initial,
                template=config_file,
            ),
            "labeling": {"backend": "toy"},
            "evaluation": {
                "validation_path": str(validation),
                "max_rmse": {
                    "energy_rmse": 1.0,
                    "force_rmse": 1.0,
                },
            },
        },
        initial_training=initial,
        runtime=WorkflowRuntime(train=continuation_train),
    )
    model = tmp_path / "nep.txt"
    model.write_text("model\n", encoding="utf-8")
    diagnostic = tmp_path / "acquisition-signals.json"
    diagnostic.write_text(
        json.dumps({"diagnostic_accepted": True}), encoding="utf-8"
    )
    attempts = tmp_path / "md-attempts.json"
    attempts.write_text(
        json.dumps({"attempts": [{"completed": True}]}), encoding="utf-8"
    )
    label_provenance = tmp_path / "label-provenance.json"
    label_provenance.write_text(
        json.dumps({"backend": "toy", "origin": "development"}),
        encoding="utf-8",
    )
    work_dir = tmp_path / "retrain"
    work_dir.mkdir()
    outcome = adapter.run_stage(
        "retrain",
        StageContext(
            generation=1,
            generation_dir=tmp_path,
            plan=GenerationPlan(1, 1, 2),
            artifacts={
                "training_input": initial,
                "training_set": initial,
                "model": model,
                "acquisition_signals": diagnostic,
                "md_attempts": attempts,
                "label_provenance": label_provenance,
            },
            previous_artifacts={},
            stage_dir=work_dir,
        ),
    )

    assert outcome.metrics["retrained"] is False
    assert outcome.metrics["backend"] == "reuse"
    assert outcome.artifacts["retrained_model"] == model
    lineage = json.loads(outcome.artifacts["model_lineage"].read_text())
    assert lineage["trained_on_current_labels"] is False
    assert lineage["model_updated"] is False
    assert lineage["parent_model_sha256"] == lineage["candidate_model_sha256"]
    assert len(training_calls) == 0
    reused_lineage = tmp_path / "reused-active-lineage.json"
    reused_lineage.write_text(
        json.dumps(
            {
                "generation": 1,
                "active_model_sha256": lineage["candidate_model_sha256"],
            }
        ),
        encoding="utf-8",
    )
    reused = adapter._train(
        StageContext(
            generation=2,
            generation_dir=tmp_path,
            plan=GenerationPlan(2, 2, 2),
            artifacts={},
            previous_artifacts={
                "training_set": initial,
                "activated_model": outcome.artifacts["retrained_model"],
                "active_model_lineage": reused_lineage,
            },
            stage_dir=tmp_path / "next-reuse",
        )
    )
    assert reused.artifacts["model"] == model

    diagnostic.write_text(
        json.dumps({"diagnostic_accepted": False}), encoding="utf-8"
    )
    expanded_training = tmp_path / "expanded-train.xyz"
    ase_write(
        expanded_training,
        [
            *initial_frames,
            ToyTeacher("ordinary").label(
                toy_candidate_frames("ordinary", 33, 1)[0]
            ),
        ],
        format="extxyz",
    )
    update_dir = tmp_path / "update"
    update_dir.mkdir()
    updated = adapter.run_stage(
        "retrain",
        StageContext(
            generation=2,
            generation_dir=tmp_path,
            plan=GenerationPlan(2, 2, 2),
                artifacts={
                    "training_input": initial,
                    "training_set": expanded_training,
                "model": model,
                "acquisition_signals": diagnostic,
                "md_attempts": attempts,
                "label_provenance": label_provenance,
            },
            previous_artifacts={},
            stage_dir=update_dir,
        ),
    )

    assert updated.metrics["retrained"] is True
    assert updated.metrics["backend"] == "torchnep"
    assert updated.artifacts["retrained_model"].read_text() == "continued model\n"
    updated_lineage = json.loads(
        updated.artifacts["model_lineage"].read_text()
    )
    assert updated_lineage["trained_on_current_labels"] is True
    assert updated_lineage["model_updated"] is True
    assert (
        updated_lineage["parent_model_sha256"]
        != updated_lineage["candidate_model_sha256"]
    )
    assert len(training_calls) == 1
    activated_lineage = tmp_path / "updated-active-lineage.json"
    activated_lineage.write_text(
        json.dumps(
            {
                "generation": 2,
                "active_model_sha256": updated_lineage[
                    "candidate_model_sha256"
                ],
            }
        ),
        encoding="utf-8",
    )
    next_updated = adapter._train(
        StageContext(
            generation=3,
            generation_dir=tmp_path,
            plan=GenerationPlan(3, 3, 2),
            artifacts={},
            previous_artifacts={
                "training_set": initial,
                "activated_model": updated.artifacts["retrained_model"],
                "active_model_lineage": activated_lineage,
            },
            stage_dir=tmp_path / "next-update",
        )
    )
    assert (
        next_updated.artifacts["model"]
        == updated.artifacts["retrained_model"]
    )


@pytest.mark.parametrize("backend", ["vasp", "abacus"])
@pytest.mark.parametrize("kpoint_mode", ["auto", "kpoints"])
def test_workflow_label_routes_production_dft_through_label_interface(
    tmp_path: Path, backend: str, kpoint_mode: str, monkeypatch
):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    initial = tmp_path / "initial.xyz"
    validation = tmp_path / "validation.xyz"
    selected_input = tmp_path / "selected.xyz"
    config_file = tmp_path / "nep.in"
    input_file = tmp_path / ("INCAR" if backend == "vasp" else "INPUT")
    resource_dir = tmp_path / "dft-resources"
    resource_dir.mkdir()
    config_file.write_text("type 1 Fe\n", encoding="utf-8")
    input_file.write_text("test input\n", encoding="utf-8")
    ase_write(
        initial,
        [ToyTeacher("ordinary").label(toy_candidate_frames("ordinary", 71, 1)[0])],
        format="extxyz",
    )
    ase_write(
        validation,
        [ToyTeacher("ordinary").label(toy_candidate_frames("ordinary", 72, 1)[0])],
        format="extxyz",
    )
    ase_write(
        selected_input,
        toy_candidate_frames("ordinary", 73, 2),
        format="extxyz",
    )
    calls = []

    def fake_label(request, selected_backend):
        inputs = ase_read(request.source, index=":")
        frames = [
            ToyTeacher("ordinary").label(frame)
            for frame in inputs
        ]
        bind_labeled_frames_to_inputs(inputs, frames)
        ase_write(request.output_file, frames, format="extxyz")
        calls.append((request, selected_backend))
        return LabelResult(selected_backend, request.output_file, tuple(frames))

    dft_options = {
        "backend": backend,
        "input_path": str(input_file),
        "resource_path": str(resource_dir),
        "gamma_centered": True,
    }
    if kpoint_mode == "kpoints":
        dft_options.update(kpoint_mode="kpoints", kpoints=[2, 3, 4])

    adapter = WorkflowIterationAdapter(
        {
            "training": {"backend": "gpumd", "config_path": str(config_file)},
            "md": {"backend": "lammps", "spin": False},
            "sampling": _sampling(
                structures=initial,
                template=config_file,
            ),
            "labeling": dft_options,
            "evaluation": {
                "validation_path": str(validation),
                "max_rmse": {"energy_rmse": 1.0, "force_rmse": 1.0},
            },
        },
        initial_training=initial,
        runtime=WorkflowRuntime(label=fake_label),
    )
    generation_dir = tmp_path / "Generation-1"
    generation_dir.mkdir()

    outcome = adapter._label(
        StageContext(
            generation=1,
            generation_dir=generation_dir,
            plan=GenerationPlan(1, 1, 2),
            artifacts={"selected_input": selected_input},
            previous_artifacts={},
        )
    )

    request, selected_backend = calls[0]
    assert selected_backend == backend
    assert request.settings["input_file"] == input_file
    assert request.settings["resource_dir"] == resource_dir
    assert request.settings["n_cpu"] == 4
    assert request.settings["use_gamma"] is True
    assert request.settings["kpoint_mode"] == kpoint_mode
    assert request.settings["ka"] == (
        (2, 3, 4) if kpoint_mode == "kpoints" else (1, 1, 1)
    )
    assert request.settings["kspacing"] is None
    assert outcome.metrics == {
        "backend": backend,
        "origin": None,
        "labeled_count": 2,
    }
