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


def test_training_parity_uses_prediction_first_files_and_records_model(tmp_path):
    from NepTrain.core.reporting import build_training_parity_reports
    from NepTrain.core.content_addressing import file_sha256

    model = tmp_path / "nep_final.txt"
    model.write_text("final model\n")
    files = {}
    for name, width in (("energy", 1), ("force", 3), ("virial", 6), ("mforce", 3)):
        reference = np.arange(4 * width).reshape(4, width) / 10
        path = tmp_path / f"{name}_train.out"
        np.savetxt(path, np.column_stack([reference + 0.02, reference]))
        files[path.name] = path
    artifacts = build_training_parity_reports(
        tmp_path, outputs=files, backend="torchnep", model=model
    )
    report = json.loads(artifacts["training-parity-train-report.json"].read_text())
    assert report["status"] == "ready"
    assert report["source"]["prediction_model_sha256"] == file_sha256(model)
    assert "final-epoch model" in report["title"]
    assert set(report["panels"]) == {"energy", "force", "virial", "mforce"}
    assert report["panels"]["energy"]["reference_range"] == [0, 0.3]
    np.testing.assert_allclose(
        report["panels"]["energy"]["predicted_range"], [0.02, 0.32]
    )
    np.testing.assert_allclose([p["rmse"] for p in report["panels"].values()], 0.02)
    _assert_png(artifacts["training-parity-train.png"])


def test_training_parity_reports_bad_files_without_inventing_predictions(tmp_path):
    from NepTrain.core.reporting import build_training_parity_reports

    bad = tmp_path / "energy_train.out"
    bad.write_text("1 2 3\n")
    artifacts = build_training_parity_reports(
        tmp_path, outputs={bad.name: bad}, backend="gpumd", model=None
    )
    report = json.loads(artifacts["training-parity-train-report.json"].read_text())
    assert report["status"] == "unavailable"
    assert "expected 2 columns" in report["source"]["warnings"][0]
    assert "training-parity-train.png" not in artifacts


def test_selection_pca_uses_one_basis_and_preserves_selected_indices(tmp_path):
    from NepTrain.core.reporting import build_selection_pca_report

    points = np.random.default_rng(24).normal(size=(40, 5))
    original = points.copy()
    selected = [3, 18, 32]
    result = build_selection_pca_report(
        tmp_path,
        descriptors=points,
        selected_indices=selected,
        candidate_ids=[str(i) for i in range(40)],
    )
    report = json.loads(result.report.read_text())
    coordinates = np.asarray(report["coordinates"])
    np.testing.assert_allclose(
        coordinates, (points - report["center"]) @ np.asarray(report["components"]).T
    )
    np.testing.assert_array_equal(points, original)
    assert coordinates.shape == (40, 2)
    assert report["selected_indices"] == selected
    assert report["candidate_ids"][18] == "18"
    assert 0 < sum(report["explained_variance_ratio"]) <= 1
    repeat = build_selection_pca_report(
        tmp_path / "repeat",
        descriptors=points,
        selected_indices=selected,
        candidate_ids=[str(i) for i in range(40)],
    )
    assert result.chart.read_bytes() == repeat.chart.read_bytes()
    _assert_png(result.chart)


def test_selection_pca_handles_one_point_constant_data_and_empty_pool(tmp_path):
    from NepTrain.core.reporting import build_selection_pca_report

    for n in (0, 1, 4):
        result = build_selection_pca_report(
            tmp_path / str(n),
            descriptors=np.ones((n, 3)),
            selected_indices=[0] if n else [],
            candidate_ids=[str(i) for i in range(n)],
        )
        report = json.loads(result.report.read_text())
        if n:
            np.testing.assert_array_equal(report["coordinates"], np.zeros((n, 2)))
            assert report["explained_variance_ratio"] == [0, 0]
            _assert_png(result.chart)
        else:
            assert result.chart is None and report["status"] == "unavailable"
