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
from NepTrain.core.toy_iteration import (
    ToyIterationAdapter,
    run_toy_iteration_smoke,
)
from NepTrain.core.toy_workflow import toy_candidate_frames, toy_raw_features
from NepTrain.core.training import TrainingResult
from NepTrain.core.workflow_iteration import (
    WorkflowIterationAdapter,
    WorkflowIterationError,
    WorkflowRuntime,
)
from NepTrain.core.dft.toy import ToyTeacher
from NepTrain.core.dft import LabelResult
from ase.io import read as ase_read
from ase.io import write as ase_write


def _sampling(
    temperatures=(300.0,),
    *,
    steps=(100, 400, 1600, 6400),
    md_runs=4,
    candidate_target=24,
    frame_stride=1,
):
    return {
        "mode": "auto",
        "conditions": {
            "temperature_path": list(temperatures),
            "production_temperatures": list(temperatures),
            "pressure": 0.0,
            "spin_temperature": "auto",
        },
        "progression": {
            "md_runs_per_iteration": md_runs,
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
        },
        "candidate_pool": {
            "target": candidate_target,
            "growth": 1.0,
            "frame_stride": frame_stride,
            "pre_failure_frames": 2,
            "bad_tail_frames": 1,
            "health": {},
        },
        "selection": {
            "method": "fps",
            "dft_budget": 8,
            "minimum_dft_budget": 4,
            "budget_decay": 0.75,
            "min_novelty": 0.0,
        },
    }


def test_progression_grows_exploration_and_reduces_dft_budget():
    plans = progressive_plans(
        4,
        initial_budget=12,
        minimum_budget=4,
        candidate_growth=2.0,
    )

    assert [plan.steps for plan in plans] == [100, 100, 100, 100]
    assert [plan.dft_budget for plan in plans] == [12, 9, 7, 6]
    assert [len(plan.temperatures) for plan in plans] == [3, 3, 3, 3]
    assert all(
        later.candidate_count > earlier.candidate_count
        for earlier, later in zip(plans, plans[1:])
    )


def test_stratified_fps_is_input_order_independent_and_balanced():
    points = np.asarray(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]
    )
    candidate_ids = ["a1", "a2", "a3", "b1", "b2", "b3"]
    strata = ["A", "A", "A", "B", "B", "B"]
    plan = GenerationPlan(1, 1, 6, 4, 100, (300.0,))
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
    assert first.counts_by_stratum == {"A": 2, "B": 2}


def test_zero_min_novelty_allows_duplicate_smoke_candidates():
    plan = GenerationPlan(1, 1, 2, 1, 2, (10.0,), min_novelty=0.0)
    result = stratified_farthest_point_sampling(
        np.asarray([[1.0, 1.0], [1.0, 1.0]]),
        np.asarray([[1.0, 1.0]]),
        ["candidate-a", "candidate-b"],
        ["T=10", "T=10"],
        plan,
    )

    assert result.selected_ids == ("candidate-a",)


def test_two_generation_toy_workflow_is_deterministic_and_resumable(tmp_path: Path):
    report = run_toy_iteration_smoke(
        tmp_path / "iteration",
        profile="spin",
        generations=2,
        dft_budget=8,
    )

    assert report.passed
    assert report.generations_completed == 2
    assert report.budgets == (8, 6)
    assert report.steps == (100, 100)
    assert report.selected_counts == (8, 6)
    assert report.training_counts[1] > report.training_counts[0]
    assert report.coverage_p95[1] < report.coverage_p95[0]
    assert report.scenario_steps == ((100,), (100, 400))
    assert report.maturity_counts == (
        {"smoke_passed": 1},
        {"short_stable": 1, "smoke_passed": 1},
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
        dft_budget=8,
    )

    assert report.passed
    assert report.steps == (100, 100, 100)
    assert set(report.strata_counts[2]) == {
        "T=300|P=0",
        "T=500|P=0",
        "T=700|P=0",
    }
    assert max(report.strata_counts[2].values()) - min(
        report.strata_counts[2].values()
    ) <= 1
    assert report.scenario_steps[2] == (100, 400, 1600)
    assert report.maturity_counts[2] == {
        "long_stable": 1,
        "short_stable": 1,
        "smoke_passed": 1,
    }


def test_controller_rejects_completed_artifact_drift(tmp_path: Path):
    root = tmp_path / "drift"
    run_toy_iteration_smoke(root, profile="spin", generations=1, seed=17)
    plan = progressive_plans(1, seed=17)
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


def test_controller_rejects_plan_change_after_generation_started(tmp_path: Path):
    class AcceptingAdapter:
        def run_stage(self, stage, context):
            artifact = context.generation_dir / f"{stage}.json"
            artifact.write_text("{}\n", encoding="utf-8")
            metrics = {"accepted": True} if stage == "evaluate" else {}
            return StageOutcome({f"{stage}_artifact": artifact}, metrics)

    controller = GenerationController(tmp_path / "plan-drift", "plan-drift")
    original = GenerationPlan(1, 1, 4, 2, 100, (300.0,))
    controller.run_workflow((original,), AcceptingAdapter())
    changed = GenerationPlan(1, 1, 4, 2, 200, (300.0,))

    with pytest.raises(IterationError, match="plan changed"):
        controller.run_workflow((changed,), AcceptingAdapter())


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
        model.write_text("fake spin_nep_lite model\n", encoding="utf-8")
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

    def fake_predict(_model, frames, backend):
        calls.append(("predict", backend, len(frames)))
        scale = 10.0 if len(frames) == 3 else 1.0
        return {
            "energy_rmse": 0.1 * scale,
            "force_rmse": 0.2 * scale,
            "virial_rmse": 0.3 * scale,
            "mforce_rmse": 0.4 * scale,
        }

    config = {
        "training": {
            "backend": "torchnep",
            "config_path": str(config_file),
            "device": "cuda",
        },
        "md": {
            "backend": "lammps",
            "structures": str(structure),
            "spin": True,
            "mpi_ranks": 1,
        },
        "sampling": _sampling(
            (300.0, 500.0),
            steps=(40, 160, 640, 2560),
            md_runs=2,
            candidate_target=6,
            frame_stride=1,
        ),
        "dft": {"software": "toy", "teacher_profile": "spin"},
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
    runtime = WorkflowRuntime(
        train=fake_train,
        md=fake_md,
        descriptors=fake_descriptors,
        predict=fake_predict,
    )
    adapter = WorkflowIterationAdapter(
        config, initial_training=initial, runtime=runtime
    )
    plan = GenerationPlan(1, 19, 6, 3, 40, (300.0, 500.0), frame_stride=1)
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
    assert summary.metrics["select"]["candidate_count_after_thinning"] == 4
    assert summary.metrics["label"]["backend"] == "toy"
    assert summary.metrics["diagnose"]["current_model_mforce_rmse"] == 4.0
    assert summary.metrics["merge"]["added_count"] == 3
    assert summary.metrics["retrain"]["training_count"] == 6
    assert summary.artifacts["retraining_config"].read_text(encoding="utf-8").endswith(
        "lr 0.0003\n"
    )
    assert summary.metrics["evaluate"]["added_training_count"] == 3
    assert summary.metrics["evaluate"]["mforce_rmse"] == 0.4
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
    assert len([call for call in calls if call[0] == "predict"]) == 3
    assert calls[0] == ("train", "torchnep", "cuda")
    assert training_requests[0].config_file == config_file
    assert training_requests[1].config_file.name == "torchnep-finetune.in"
    assert "lr 0.0003\n" in training_requests[1].config_file.read_text(
        encoding="utf-8"
    )
    assert ("md", "lammps", 300.0, 40) in calls
    assert ("md", "lammps", 500.0, 40) not in calls

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
    fallback_summary = GenerationController(
        tmp_path / "fallback-workflow", "fallback"
    ).run_generation(
        plan,
        WorkflowIterationAdapter(
            fallback_config, initial_training=initial, runtime=runtime
        ),
    )
    assert fallback_summary.accepted
    assert all(request.test_file is None for request in training_requests)


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
        "sampling": _sampling((100.0,), candidate_target=3),
        "dft": {"software": "toy", "teacher_profile": "spin"},
        "evaluation": {
            "validation_path": str(validation),
            "max_rmse": {
                "energy_rmse": 1.0,
                "force_rmse": 1.0,
                "mforce_rmse": 1.0,
            },
        },
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
        source_id="healthy", temperature=100.0, pressure=0.0, md_window="stable_prefix"
    )
    preferred_duplicate = repeated.copy()
    preferred_duplicate.info.update(
        source_id="failed", temperature=100.0, pressure=0.0, md_window="pre_failure"
    )
    candidates_path = tmp_path / "candidates.xyz"
    ase_write(
        candidates_path,
        [first, stable_duplicate, preferred_duplicate],
        format="extxyz",
    )
    model = tmp_path / "nep.txt"
    model.write_text("fake\n", encoding="utf-8")
    context = StageContext(
        generation=1,
        generation_dir=tmp_path,
        plan=GenerationPlan(1, 1, 3, 20, 10, (100.0,), frame_stride=1),
        artifacts={
            "candidates": candidates_path,
            "training_input": initial,
            "model": model,
        },
        previous_artifacts={},
    )

    outcome = adapter._select(context)
    selected = ase_read(outcome.artifacts["selected_input"], index=":")

    assert outcome.metrics["candidate_count_after_thinning"] == 2
    assert outcome.metrics["duplicate_candidate_count"] == 1
    assert outcome.metrics["selected_count"] == 2
    assert outcome.metrics["configured_dft_budget"] == 20
    assert outcome.metrics["effective_dft_budget"] == 4
    assert "pre_failure" in {frame.info["md_window"] for frame in selected}


def test_candidate_cap_spreads_each_source_over_full_trajectory():
    def frames(source: str, count: int):
        result = []
        for step in range(count):
            frame = toy_candidate_frames("ordinary", step + 100, 1)[0]
            frame.info.update(source_id=source, lammps_step=step)
            result.append(frame)
        return result

    first = frames("first", 10)
    second = frames("second", 10)
    selected = WorkflowIterationAdapter._balanced_cap(
        [("first", first), ("second", second)], 6
    )

    by_source = {
        source: [frame.info["lammps_step"] for item_source, _, frame in selected if item_source == source]
        for source in ("first", "second")
    }
    assert by_source == {"first": [0, 4, 9], "second": [0, 4, 9]}


def test_candidate_cap_prioritizes_pre_failure_frames():
    frames = toy_candidate_frames("ordinary", 301, 6)
    for step, frame in enumerate(frames):
        frame.info.update(
            lammps_step=step,
            md_window="pre_failure" if step >= 4 else "stable_prefix",
        )

    selected = WorkflowIterationAdapter._balanced_cap([("failed", frames)], 2)

    assert [frame.info["lammps_step"] for _, _, frame in selected] == [4, 5]


def test_retrain_is_skipped_when_current_model_passes_new_dft_diagnostics(
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

    def unexpected_train(_request, _backend):
        raise AssertionError("training should be skipped")

    adapter = WorkflowIterationAdapter(
        {
            "training": {
                "backend": "torchnep",
                "config_path": str(config_file),
            },
            "md": {"backend": "lammps", "spin": False},
            "sampling": _sampling((300.0,)),
            "dft": {"backend": "toy"},
            "evaluation": {
                "validation_path": str(validation),
                "max_rmse": {
                    "energy_rmse": 1.0,
                    "force_rmse": 1.0,
                },
            },
        },
        initial_training=initial,
        runtime=WorkflowRuntime(train=unexpected_train),
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
    work_dir = tmp_path / "retrain"
    work_dir.mkdir()
    outcome = adapter.run_stage(
        "retrain",
        StageContext(
            generation=1,
            generation_dir=tmp_path,
            plan=GenerationPlan(1, 1, 4, 2, 10, (300.0,)),
            artifacts={
                "training_set": initial,
                "model": model,
                "acquisition_signals": diagnostic,
                "md_attempts": attempts,
            },
            previous_artifacts={},
            stage_dir=work_dir,
        ),
    )

    assert outcome.metrics["retrained"] is False
    assert outcome.metrics["backend"] == "reuse"
    assert outcome.artifacts["retrained_model"] == model

    training_calls = []

    def continuation_train(request, backend):
        training_calls.append((request, backend))
        request.output_dir.mkdir(parents=True, exist_ok=True)
        trained_model = request.output_dir / "nep.txt"
        trained_model.write_text("continued model\n", encoding="utf-8")
        return TrainingResult(backend, trained_model, None, None)

    adapter.runtime = WorkflowRuntime(train=continuation_train)
    previous_signals = tmp_path / "previous-signals.json"
    previous_signals.write_text(
        json.dumps(
            {"production_ready": True, "validation_accepted": False}
        ),
        encoding="utf-8",
    )
    continuation_dir = tmp_path / "continuation"
    continuation_dir.mkdir()
    continued = adapter.run_stage(
        "retrain",
        StageContext(
            generation=2,
            generation_dir=tmp_path,
            plan=GenerationPlan(2, 2, 4, 2, 10, (300.0,)),
            artifacts={
                "training_set": initial,
                "model": model,
                "acquisition_signals": diagnostic,
                "md_attempts": attempts,
            },
            previous_artifacts={"signals": previous_signals},
            stage_dir=continuation_dir,
        ),
    )

    assert continued.metrics["retrained"] is True
    assert "global validation" in continued.metrics["reason"]
    assert len(training_calls) == 1


@pytest.mark.parametrize("backend", ["vasp", "abacus"])
def test_workflow_label_routes_production_dft_through_label_interface(
    tmp_path: Path, backend: str
):
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
        frames = [
            ToyTeacher("ordinary").label(frame)
            for frame in ase_read(request.source, index=":")
        ]
        ase_write(request.output_file, frames, format="extxyz")
        calls.append((request, selected_backend))
        return LabelResult(selected_backend, request.output_file, tuple(frames))

    adapter = WorkflowIterationAdapter(
        {
            "training": {"backend": "gpumd", "config_path": str(config_file)},
            "md": {"backend": "lammps", "spin": False},
            "sampling": _sampling(),
                "dft": {
                    "backend": backend,
                    "n_cpu": 4,
                    "input_path": str(input_file),
                "resource_path": str(resource_dir),
                "kpoints_use_gamma": True,
                "use_k_stype": "kpoints",
                "kpoints": [2, 3, 4],
            },
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
            plan=GenerationPlan(1, 1, 2, 2, 10, (300.0,)),
            artifacts={"selected_input": selected_input},
            previous_artifacts={},
        )
    )

    request, selected_backend = calls[0]
    assert selected_backend == backend
    assert request.input_file == input_file
    assert request.resource_dir == resource_dir
    assert request.n_cpu == 4
    assert request.use_gamma is True
    assert request.kpoint_mode == "kpoints"
    assert request.ka == (2, 3, 4)
    assert request.kspacing is None
    assert outcome.metrics == {"backend": backend, "labeled_count": 2}
