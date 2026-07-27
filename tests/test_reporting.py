import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from NepTrain.core.reporting import (
    ParitySeries,
    build_evaluation_report,
    build_parity_report,
    build_training_report,
)


def _assert_png(path: Path) -> None:
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    pixels = mpimg.imread(path)
    assert pixels.shape[0] > 0
    assert pixels.shape[1] > 0


def test_training_report_builds_deterministic_matplotlib_png(tmp_path: Path):
    loss = tmp_path / "loss.out"
    loss.write_text(
        "\n".join(
            [
                "0 10 0.1 0.2 3 4 5 3.5 4.5 5.5",
                "10 4 0.1 0.2 1 2 3 1.5 2.5 3.5",
                "20 2 0.1 0.2 0.5 1 2 0.7 1.2 2.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = build_training_report(
        tmp_path,
        backend="torchnep",
        loss_path=loss,
    )

    report = json.loads(artifacts.report.read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    assert report["renderer"].startswith("matplotlib-")
    assert report["parsed_rows"] == 3
    assert report["chart"] == "training-convergence.png"
    assert "energy_train" in report["series"]
    assert artifacts.chart is not None
    _assert_png(artifacts.chart)
    repeated_dir = tmp_path / "repeated"
    repeated_dir.mkdir()
    repeated_loss = repeated_dir / "loss.out"
    repeated_loss.write_bytes(loss.read_bytes())
    repeated = build_training_report(
        repeated_dir,
        backend="torchnep",
        loss_path=repeated_loss,
    )
    assert repeated.chart is not None
    assert repeated.chart.read_bytes() == artifacts.chart.read_bytes()


def test_training_report_records_unavailable_loss_without_fake_chart(
    tmp_path: Path,
):
    loss = tmp_path / "loss.out"
    loss.write_text("trainer header only\n", encoding="utf-8")

    artifacts = build_training_report(
        tmp_path,
        backend="gpumd",
        loss_path=loss,
    )

    report = json.loads(artifacts.report.read_text(encoding="utf-8"))
    assert report["status"] == "unavailable"
    assert report["parsed_rows"] == 0
    assert artifacts.chart is None
    assert not (tmp_path / "training-convergence.png").exists()


def test_evaluation_report_normalises_metrics_by_threshold(tmp_path: Path):
    artifacts = build_evaluation_report(
        tmp_path,
        metrics={"energy_rmse": 0.04, "force_rmse": 0.25},
        thresholds={"energy_rmse": 0.05, "force_rmse": 0.2},
        parent_metrics={"energy_rmse": 0.06, "force_rmse": 0.3},
    )

    report = json.loads(artifacts.report.read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    assert report["series"] == ["energy_rmse", "force_rmse"]
    assert report["chart"] == "evaluation-metrics.png"
    assert artifacts.chart is not None
    _assert_png(artifacts.chart)


def test_evaluation_report_serialises_non_finite_metrics_as_null(
    tmp_path: Path,
):
    artifacts = build_evaluation_report(
        tmp_path,
        metrics={"energy_rmse": float("nan")},
        thresholds={"energy_rmse": 0.05},
    )

    report = json.loads(artifacts.report.read_text(encoding="utf-8"))
    assert report["status"] == "unavailable"
    assert report["metrics"]["energy_rmse"] is None
    assert artifacts.chart is None


def test_parity_report_covers_validation_observables(tmp_path: Path):
    reference = np.linspace(-2.0, 2.0, 12)
    artifacts = build_parity_report(
        tmp_path,
        series={
            "energy": ParitySeries(
                reference,
                reference + 0.02,
                "eV",
            ),
            "force": ParitySeries(
                reference,
                reference - 0.04,
                "eV/Å",
            ),
            "virial": ParitySeries(
                reference,
                reference + 0.06,
                "eV",
            ),
        },
        source={
            "validation_sha256": "a" * 64,
            "candidate_model_sha256": "b" * 64,
        },
    )

    report = json.loads(artifacts.report.read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    assert set(report["panels"]) == {"energy", "force", "virial"}
    assert report["panels"]["force"]["finite_pairs"] == 12
    assert report["panels"]["force"]["sampling"] == "all"
    assert report["chart"] == "evaluation-parity.png"
    assert artifacts.chart is not None
    _assert_png(artifacts.chart)
