"""Exercise real v3 workflow decisions using deterministic local backend doubles.
No real NEP training, MD integration, or DFT is performed.
The adapter, labeling, selection, progression, persistence and reporting are real.
"""

import json
from pathlib import Path
from typing import Any

from ase.io import read, write
from .dft.toy import ToyTeacher
from .toy_workflow import toy_candidate_frames, toy_raw_features
from .training import TrainingResult
from .md import MdResult
from .iteration import GenerationController, GenerationPlan
from .generation_policy import ACTIVE_LEARNING_GENERATION_PROTOCOL
from .workflow_iteration import (
    WorkflowIterationAdapter,
    WorkflowRuntime,
    PredictionEvaluation,
)
from .workflow import _generation_science
from .notifications import _generation_event
from .config import validate_config
from .persistence import atomic_write_json
from .smoke import SmokeError, _assert_safe_output


def _run_case(root: Path, case: str, profile: str, seed: int) -> dict[str, Any]:
    root.mkdir(parents=True)
    initial = root / "initial.xyz"
    structures = root / "structures.xyz"
    template = root / "nep.in"
    template.write_text(
        "type 1 Fe\n" + ("spin_mode 3\n" if profile == "spin" else ""),
        encoding="utf-8",
    )
    teacher = ToyTeacher(profile)
    labels = [teacher.label(f) for f in toy_candidate_frames(profile, seed, 4)]
    for f in labels:
        f.info["smoke_test"] = True
    write(initial, labels, format="extxyz")
    write(structures, toy_candidate_frames(profile, seed + 1, 1), format="extxyz")
    events = []
    current = {"generation": 0, "stage": None}

    def train(request, backend):
        request.output_dir.mkdir(parents=True, exist_ok=True)
        model = request.output_dir / "nep.txt"
        model.write_text("backend-double: deterministic model\n")
        return TrainingResult(backend, model, None, None)

    def md(request, backend):
        request.output_dir.mkdir(parents=True, exist_ok=True)
        # Deduplicate replicas to four labels per generation in the evidence-shortfall case.
        frame_seed = (
            seed + 100 + current["generation"]
            if case == "insufficient_labels"
            else request.seed
        )
        count = 4 if case == "insufficient_labels" else 8
        frames = toy_candidate_frames(
            profile, frame_seed, count, temperatures=(request.temperature,)
        )
        for i, f in enumerate(frames):
            f.info["lammps_step"] = (i + 1) * request.steps // count
        failed = case == "md_recovery" and current["generation"] == 1
        write(request.output_file, frames, format="extxyz")
        events.append(
            {
                "generation": current["generation"],
                "steps": request.steps,
                "temperature": request.temperature,
                "replica": request.replica,
                "seed": request.seed,
                "completed": not failed,
            }
        )
        return MdResult(
            backend,
            request.output_file,
            request.output_dir,
            "cpu",
            completed=not failed,
            last_step=request.steps,
            failure_code="injected_failure" if failed else None,
            failure_reason="deterministic recovery probe" if failed else None,
        )

    def predict(model, frames, backend):
        test = bool(frames and frames[0].info.get("smoke_test"))
        if test and case == "broken_test":
            raise RuntimeError("injected auxiliary predictor failure")
        scale = 100 if test else 1
        return PredictionEvaluation(
            {
                "energy_rmse": 0.005 * scale,
                "force_rmse": 0.05 * scale,
                "virial_rmse": 0.02 * scale,
                **({"mforce_rmse": 0.03 * scale} if profile == "spin" else {}),
            }
        )

    config = {
        "schema_version": 8,
        "training": {
            "backend": "gpumd",
            "initial_path": str(initial),
            "config_path": str(template),
        },
        "md": {"backend": "lammps", "spin": profile == "spin"},
        "sampling": {
            "routes": [
                {
                    "id": "main",
                    "structures": [str(structures)],
                    "template_path": str(template),
                    "conditions": {
                        "temperature_path": [300.0],
                        "production_temperatures": [300.0],
                        "pressure": 0.0,
                    },
                    "progression": {
                        "steps": {
                            "smoke_passed": 100,
                            "short_stable": 400,
                            "long_stable": 1600,
                            "production_ready": 6400,
                        },
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
                "max_selected": 8,
                "novelty": {"selection_threshold": 0.0, "completion_threshold": 0.0},
            },
        },
        "labeling": {"backend": "toy"},
        "workflow": {
            "max_model_generations": 8,
            "convergence": {
                "acquisition_max_rmse": {
                    "energy_rmse": 0.01,
                    "force_rmse": 0.1,
                    "virial_rmse": 0.1,
                    **({"mforce_rmse": 0.1} if profile == "spin" else {}),
                },
                "min_selected": 8 if case == "insufficient_labels" else 1,
                "consecutive_generations": 6 if case == "long_streak" else 2,
            },
        },
    }
    config["execution"] = {
        "stage_targets": {
            name: "local" for name in ("training", "sampling", "labeling", "analysis")
        },
        "targets": {"local": {"executor": "process"}},
    }
    if case in {"overlap_test", "bad_test", "broken_test"}:
        test_file = initial
        if case != "overlap_test":
            test_file = root / "test.xyz"
            test_frames = [
                teacher.label(f) for f in toy_candidate_frames(profile, seed + 2, 4)
            ]
            for f in test_frames:
                f.info["smoke_test"] = True
            write(test_file, test_frames, format="extxyz")
        config["evaluation"] = {
            "validation_path": str(test_file),
            "max_rmse": {"energy_rmse": 0.01, "force_rmse": 0.1},
        }
    if case in {"missing_test", "empty_test"}:
        test_file = root / "optional-test.xyz"
        if case == "empty_test":
            test_file.write_text(" \n\t", encoding="utf-8")
        config["evaluation"] = {"validation_path": str(test_file)}
    validate_config(config)
    atomic_write_json(root / "config.json", config)
    stage_calls = []

    class Adapter(WorkflowIterationAdapter):
        def run_stage(self, stage, context):
            stage_calls.append((context.generation, stage))
            current.update(generation=context.generation, stage=stage)
            return super().run_stage(stage, context)

    adapter = Adapter(
        config,
        initial_training=initial,
        runtime=WorkflowRuntime(
            train=train,
            md=md,
            descriptors=lambda model, frames: toy_raw_features(frames, profile),
            predict=predict,
        ),
    )
    controller = GenerationController(
        root / "workflow", case, generation_protocol=ACTIVE_LEARNING_GENERATION_PROTOCOL
    )
    rows = []
    notifications = []
    sciences = []
    training_counts_match = []
    budget = 4 if case == "insufficient_labels" else 8
    for generation in range(1, budget + 2):
        plan = GenerationPlan(generation, seed + generation, 8)
        summary = controller.run_generation(plan, adapter)
        ledger = json.loads(controller.ledger_path.read_text())
        record = ledger["generations"][str(generation)]
        science = _generation_science(
            {"generation": generation, "max_selected": 8}, record
        )
        actual_training_count = len(read(summary.artifacts["training_set"], index=":"))
        training_counts_match.append(
            science["training"]["after_count"] == actual_training_count
        )
        decision = summary.metrics.get("update", summary.metrics.get("validate", {}))
        rows.append(
            {
                "generation": generation,
                "kind": record["kind"],
                "stages": list(summary.metrics),
                "selected": summary.metrics.get("select", {}).get("selected_count"),
                "training_count": actual_training_count,
                "production_ready": decision.get("production_ready"),
                "streak": decision.get("acquisition_convergence_streak"),
                "disposition": decision.get("generation_disposition"),
                "converged": decision.get("workflow_converged"),
                "reasons": decision.get("convergence_reasons", []),
                "warnings": summary.metrics.get("validate", {}).get(
                    "validation_warnings", []
                ),
                "maturity": decision.get("scenario_counts_by_maturity", {}),
                "test_overlap_count": summary.metrics.get("validate", {}).get(
                    "validation_overlap_count"
                ),
                "validation_accepted": summary.metrics.get("validate", {}).get(
                    "validation_accepted"
                ),
            }
        )
        notifications.append(
            _generation_event(
                case, budget, {"generation": generation, "max_selected": 8}, record
            ).text
        )
        sciences.append(science)
        if decision.get("workflow_converged") or (
            generation >= budget
            and decision.get("generation_disposition") != "finalize"
        ):
            break
    stage_count = len(stage_calls)
    controller.run_generation(plan, adapter)
    (root / "notifications.txt").write_text(
        "\n\n".join(notifications), encoding="utf-8"
    )
    atomic_write_json(root / "md-events.json", events)
    atomic_write_json(root / "science.json", sciences)
    result = {
        "generations": rows,
        "generations_completed": len(rows),
        "converged": bool(rows[-1]["converged"]),
        "resume_reused_artifacts": len(stage_calls) == stage_count,
        "training_counts_match_artifacts": all(training_counts_match),
    }
    atomic_write_json(root / "result.json", result)
    return result


def run_workflow_smoke(
    output_dir: str | Path, *, profile: str = "ordinary", seed: int = 20260721
) -> dict[str, Any]:
    """Run fixed v3 decision scenarios; report checks without claiming physical accuracy."""
    if profile not in {"ordinary", "spin"}:
        raise SmokeError("workflow smoke profile must be ordinary or spin")
    root = _assert_safe_output(Path(output_dir))
    if root.exists():
        raise SmokeError(f"workflow smoke output already exists: {root}")
    root.mkdir(parents=True)
    cases = {
        case: _run_case(root / case, case, profile, seed)
        for case in (
            "no_test",
            "missing_test",
            "empty_test",
            "overlap_test",
            "bad_test",
            "broken_test",
            "md_recovery",
            "long_streak",
            "insufficient_labels",
        )
    }

    def decisions(case):
        return [
            (row["kind"], row["selected"], row["streak"], row["disposition"])
            for row in cases[case]["generations"]
        ]

    streak = cases["long_streak"]["generations"]
    recovery = cases["md_recovery"]["generations"]
    checks = {
        "training_counts_match_artifacts": all(
            case["training_counts_match_artifacts"] for case in cases.values()
        ),
        "expected_completion": all(
            case["converged"] == (name != "insufficient_labels")
            for name, case in cases.items()
        ),
        "optional_test_does_not_change_decisions": all(
            decisions(name) == decisions("no_test")
            for name in ("overlap_test", "bad_test", "broken_test", "missing_test", "empty_test")
        ),
        "overlap_reported": all(
            row["test_overlap_count"] == 4 and row["validation_accepted"] is None
            for row in cases["overlap_test"]["generations"]
        ),
        "test_failures_reported": all(
            row["warnings"]
            for name in ("bad_test", "broken_test", "missing_test", "empty_test")
            for row in cases[name]["generations"]
        ),
        "resume_reused_artifacts": all(
            case["resume_reused_artifacts"] for case in cases.values()
        ),
        "md_recovery_does_not_skip_maturity": (
            recovery[0]["maturity"] == {"untested": 1}
            and recovery[1]["maturity"] == {"smoke_passed": 1}
            and len(recovery) == 6
        ),
        "production_probes_extend_streak": (
            len(streak) == 7
            and streak[3]["production_ready"] is True
            and streak[3]["disposition"] == "continue"
            and streak[5]["disposition"] == "finalize"
        ),
        "insufficient_labels_do_not_converge": all(
            row["selected"] == 4 and row["streak"] == 0 and row["reasons"]
            for row in cases["insufficient_labels"]["generations"]
        ),
        "finalization_has_no_sampling": all(
            case["generations"][-1]["stages"] == ["train", "validate"]
            for name, case in cases.items()
            if name != "insufficient_labels"
        ),
    }
    report = {
        "protocol": "neptrain.workflow-smoke.v1",
        "generation_protocol": ACTIVE_LEARNING_GENERATION_PROTOCOL,
        "backend_mode": "deterministic_doubles",
        "profile": profile,
        "seed": seed,
        "checks": checks,
        "cases": cases,
        "passed": all(checks.values()),
    }
    atomic_write_json(root / "workflow-smoke-report.json", report)
    if not report["passed"]:
        raise SmokeError(
            f'workflow smoke failed; see {root / "workflow-smoke-report.json"}'
        )
    return report
