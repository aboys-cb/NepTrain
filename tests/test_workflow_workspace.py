from __future__ import annotations

import json
from pathlib import Path

import pytest

from NepTrain.core.workflow_workspace import WorkflowWorkspace
from NepTrain.core.iteration import (
    GenerationController,
    GenerationPlan,
    IterationError,
    StageOutcome,
)


class _PublishingAdapter:
    def __init__(self, *, complete_artifacts: bool = True):
        self.complete_artifacts = complete_artifacts

    def run_stage(self, stage, context):
        path = context.work_dir / f"{stage}.txt"
        path.write_text(
            json.dumps(stage) + "\n" if stage == "evaluate" else stage + "\n",
            encoding="utf-8",
        )
        artifacts = {f"{stage}_evidence": path}
        metrics = {}
        if stage == "merge":
            artifacts["training_set"] = path
        elif stage == "retrain":
            artifacts["retrained_model"] = path
        elif stage == "evaluate":
            metrics = {
                "accepted": True,
                "energy_rmse": 0.01,
                "force_rmse": 0.02,
                "virial_rmse": 0.03,
            }
            if self.complete_artifacts:
                artifacts["signals"] = path
                artifacts["activated_model"] = path
        return StageOutcome(artifacts=artifacts, metrics=metrics)


def test_workspace_hides_machine_state_and_publishes_accepted_results(tmp_path: Path):
    workspace = WorkflowWorkspace.create(tmp_path / "al-nonmag")
    controller = GenerationController(workspace.root, "al-nonmag")
    plan = GenerationPlan(1, 7, 2)

    summary = controller.run_generation(plan, _PublishingAdapter())

    assert summary.accepted is True
    assert workspace.ledger == workspace.root / ".neptrain" / "ledger.json"
    assert workspace.version == 3
    assert workspace.tasks_dir == workspace.root / ".neptrain" / "jobs"
    assert not (workspace.internal_dir / "tasks").exists()
    assert not (workspace.internal_dir / "locks").exists()
    assert workspace.generation_dir(1).name == "0001"
    assert (workspace.generation_dir(1) / "md/explore.txt").is_file()
    assert (workspace.generation_dir(1) / "dft/label.txt").is_file()
    assert (workspace.generation_dir(1) / "retrain/retrain.txt").is_file()
    assert json.loads((workspace.results_dir / "nep.txt").read_text()) == "evaluate"
    assert (workspace.results_dir / "train.xyz").read_text() == "merge\n"
    assert json.loads((workspace.results_dir / "metrics.json").read_text()) == "evaluate"
    assert "最新验收代：1" in (workspace.results_dir / "summary.md").read_text()


def test_result_publication_failure_does_not_accept_generation(tmp_path: Path):
    workspace = WorkflowWorkspace.create(tmp_path / "broken")
    controller = GenerationController(workspace.root, "broken")
    plan = GenerationPlan(1, 7, 2)

    with pytest.raises(IterationError, match="missing publishable artifacts"):
        controller.run_generation(plan, _PublishingAdapter(complete_artifacts=False))

    ledger = json.loads(workspace.ledger.read_text(encoding="utf-8"))
    generation = ledger["generations"]["1"]
    assert "complete" not in generation
    assert "evaluate" not in generation["stages"]
    assert not (workspace.generation_dir(1) / "summary.json").exists()


def test_workspace_rejects_legacy_layout(tmp_path: Path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "workflow-manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="workflow does not exist"):
        WorkflowWorkspace.locate(legacy)


def test_workspace_rejects_previous_layout_versions(tmp_path: Path):
    root = tmp_path / "existing-v2"
    internal = root / ".neptrain"
    internal.mkdir(parents=True)
    (internal / "layout.json").write_text(
        json.dumps({"layout": "neptrain.workflow", "version": 2}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported workflow layout"):
        WorkflowWorkspace.locate(root)
