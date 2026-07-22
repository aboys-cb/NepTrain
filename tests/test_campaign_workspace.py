from __future__ import annotations

import json
from pathlib import Path

import pytest

from NepTrain.core.campaign_workspace import CampaignWorkspace
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
        return StageOutcome(artifacts=artifacts, metrics=metrics)


def test_workspace_hides_machine_state_and_publishes_accepted_results(tmp_path: Path):
    workspace = CampaignWorkspace.create(tmp_path / "al-nonmag")
    controller = GenerationController(workspace.root, "al-nonmag")
    plan = GenerationPlan(1, 7, 4, 2, 10, (300.0,))

    summary = controller.run_generation(plan, _PublishingAdapter())

    assert summary.accepted is True
    assert workspace.ledger == workspace.root / ".neptrain" / "ledger.json"
    assert workspace.generation_dir(1).name == "0001"
    assert (workspace.generation_dir(1) / "sampling/md/explore.txt").is_file()
    assert (workspace.generation_dir(1) / "labeling/label.txt").is_file()
    assert (workspace.generation_dir(1) / "training/retrain/retrain.txt").is_file()
    assert (workspace.results_dir / "nep.txt").read_text() == "retrain\n"
    assert (workspace.results_dir / "train.xyz").read_text() == "merge\n"
    assert json.loads((workspace.results_dir / "metrics.json").read_text()) == "evaluate"
    assert "最新验收代：1" in (workspace.results_dir / "summary.md").read_text()


def test_result_publication_failure_does_not_accept_generation(tmp_path: Path):
    workspace = CampaignWorkspace.create(tmp_path / "broken")
    controller = GenerationController(workspace.root, "broken")
    plan = GenerationPlan(1, 7, 4, 2, 10, (300.0,))

    with pytest.raises(IterationError, match="missing publishable artifacts"):
        controller.run_generation(plan, _PublishingAdapter(complete_artifacts=False))

    ledger = json.loads(workspace.ledger.read_text(encoding="utf-8"))
    generation = ledger["generations"]["1"]
    assert "complete" not in generation
    assert "evaluate" not in generation["stages"]
    assert not (workspace.generation_dir(1) / "summary.json").exists()
