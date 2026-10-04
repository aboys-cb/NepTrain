"""Compatibility fixtures for persisted workflow protocols."""

import pytest


@pytest.fixture
def legacy_workflow_protocol(monkeypatch):
    # These executor doubles and stage assertions describe saved legacy_v1 runs.
    # Simulate their original preparation; production now creates only v3 runs.
    monkeypatch.setattr(
        "NepTrain.core.workflow.ACTIVE_LEARNING_GENERATION_PROTOCOL", "legacy_v1"
    )
