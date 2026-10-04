"""Matplotlib workflow reports for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np

from .content_addressing import file_sha256


_LOSS_COLUMNS = (
    "step",
    "total_loss",
    "l1",
    "l2",
    "energy_train",
    "force_train",
    "virial_train",
    "energy_test",
    "force_test",
    "virial_test",
)
_TRAINING_SERIES = (
    ("total_loss", "#202124", "-"),
    ("energy_train", "#0072B2", "-"),
    ("energy_test", "#0072B2", "--"),
    ("force_train", "#D55E00", "-"),
    ("force_test", "#D55E00", "--"),
    ("virial_train", "#009E73", "-"),
    ("virial_test", "#009E73", "--"),
)
_MATPLOTLIB_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": "#3c4043",
    "axes.labelcolor": "#3c4043",
    "axes.titlecolor": "#202124",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": "#e8eaed",
    "grid.linewidth": 0.8,
    "grid.alpha": 1.0,
    "xtick.color": "#5f6368",
    "ytick.color": "#5f6368",
    "legend.frameon": False,
    "savefig.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
}


@dataclass(frozen=True)
class ReportArtifacts:
    """Machine-readable report plus an optional human-readable chart."""

    report: Path
    chart: Path | None


@dataclass(frozen=True)
class ParitySeries:
    """One reference/prediction comparison at a declared physical unit."""

    reference: np.ndarray
    predicted: np.ndarray
    unit: str

    def __post_init__(self) -> None:
        reference = np.asarray(self.reference, dtype=np.float64).reshape(-1)
        predicted = np.asarray(self.predicted, dtype=np.float64).reshape(-1)
        if reference.shape != predicted.shape:
            raise ValueError(
                "parity reference and prediction arrays must have equal shape"
            )
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "predicted", predicted)


def _atomic_write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return path


def _save_figure(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with matplotlib.rc_context(_MATPLOTLIB_STYLE):
        figure.savefig(
            temporary,
            format="png",
            metadata={"Software": "NepTrain"},
        )
    os.replace(temporary, path)
    return path


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    return _atomic_write(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _json_number(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _read_loss(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw_line in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            values = [float(token) for token in line.split()]
        except ValueError:
            continue
        if len(values) >= 2 and all(
            math.isfinite(value) for value in values
        ):
            rows.append(values)
    return rows


def _training_figure(
    rows: Sequence[Sequence[float]],
    backend: str,
) -> tuple[Figure, list[str]]:
    maximum_columns = max(len(row) for row in rows)
    available = _LOSS_COLUMNS[:maximum_columns]
    plotted: list[
        tuple[str, str, str, list[float], list[float]]
    ] = []
    for name, color, linestyle in _TRAINING_SERIES:
        if name not in available:
            continue
        index = available.index(name)
        values = [
            (row[0], row[index])
            for row in rows
            if len(row) > index and row[index] > 0
        ]
        if len(values) >= 2:
            plotted.append(
                (
                    name,
                    color,
                    linestyle,
                    [value[0] for value in values],
                    [value[1] for value in values],
                )
            )
    if not plotted:
        raise ValueError("loss.out has no plottable positive loss series")

    with matplotlib.rc_context(_MATPLOTLIB_STYLE):
        figure = Figure(figsize=(10, 6.2), layout="constrained")
        axis = figure.subplots()
        for name, color, linestyle, steps, values in plotted:
            axis.plot(
                steps,
                values,
                label=name,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
            )
        axis.set_yscale("log")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Loss (log scale)")
        axis.set_title("Training convergence", loc="left", pad=30)
        axis.text(
            0.0,
            1.015,
            f"Backend: {backend}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            color="#5f6368",
            fontsize=10,
        )
        axis.grid(True, which="major", axis="both")
        axis.grid(False, which="minor")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(
            loc="upper left",
            ncols=2,
            frameon=True,
            framealpha=0.9,
            edgecolor="none",
        )
    return figure, [name for name, *_ in plotted]


def build_training_report(
    output_dir: Path,
    *,
    backend: str,
    loss_path: Path | None,
) -> ReportArtifacts:
    """Build a deterministic report without fabricating missing loss data."""

    report_path = output_dir / "training-report.json"
    chart_path = output_dir / "training-convergence.png"
    report: dict[str, Any] = {
        "version": 1,
        "kind": "training",
        "renderer": f"matplotlib-{matplotlib.__version__}",
        "backend": backend,
        "status": "unavailable",
        "chart": None,
    }
    if loss_path is None or not loss_path.is_file():
        report["reason"] = "loss.out was not produced"
        return ReportArtifacts(_write_json(report_path, report), None)

    report["source"] = {
        "name": loss_path.name,
        "sha256": file_sha256(loss_path),
    }
    rows = _read_loss(loss_path)
    report["parsed_rows"] = len(rows)
    if len(rows) < 2:
        report["reason"] = "loss.out contains fewer than two numeric rows"
        return ReportArtifacts(_write_json(report_path, report), None)
    try:
        figure, series = _training_figure(rows, backend)
    except ValueError as error:
        report["reason"] = str(error)
        return ReportArtifacts(_write_json(report_path, report), None)
    _save_figure(figure, chart_path)
    report.update(
        status="ready",
        chart=chart_path.name,
        series=series,
        first_step=rows[0][0],
        last_step=rows[-1][0],
    )
    return ReportArtifacts(_write_json(report_path, report), chart_path)


def _evaluation_values(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
    parent_metrics: Mapping[str, float] | None,
) -> tuple[list[str], list[float], list[float | None]]:
    names = [
        name
        for name in thresholds
        if name in metrics
        and math.isfinite(float(metrics[name]))
        and math.isfinite(float(thresholds[name]))
        and float(thresholds[name]) > 0
    ]
    if not names:
        raise ValueError(
            "no finite evaluation metrics have positive thresholds"
        )
    candidate = [
        float(metrics[name]) / float(thresholds[name]) for name in names
    ]
    parent = [
        (
            float(parent_metrics[name]) / float(thresholds[name])
            if parent_metrics is not None
            and name in parent_metrics
            and math.isfinite(float(parent_metrics[name]))
            else None
        )
        for name in names
    ]
    return names, candidate, parent


def _evaluation_figure(
    names: Sequence[str],
    candidate: Sequence[float],
    parent: Sequence[float | None],
) -> Figure:
    has_parent = any(value is not None for value in parent)
    height = max(3.8, 0.72 * len(names) + 2.0)
    positions = list(range(len(names)))
    maximum = max(
        1.1,
        *candidate,
        *(value for value in parent if value is not None),
    )
    maximum *= 1.18
    with matplotlib.rc_context(_MATPLOTLIB_STYLE):
        figure = Figure(figsize=(10, height), layout="constrained")
        axis = figure.subplots()
        if has_parent:
            parent_values = [
                0.0 if value is None else value for value in parent
            ]
            axis.barh(
                [value - 0.17 for value in positions],
                parent_values,
                height=0.28,
                color="#9aa0a6",
                edgecolor="#5f6368",
                linewidth=0.6,
            )
            candidate_positions = [
                value + 0.17 for value in positions
            ]
            candidate_height = 0.28
        else:
            candidate_positions = positions
            candidate_height = 0.5
        colors = [
            "#0072B2" if value <= 1.0 else "#D55E00"
            for value in candidate
        ]
        bars = axis.barh(
            candidate_positions,
            candidate,
            height=candidate_height,
            color=colors,
            edgecolor="#3c4043",
            linewidth=0.6,
        )
        for bar, ratio in zip(bars, candidate, strict=True):
            if ratio > 1.0:
                bar.set_hatch("//")
            axis.text(
                ratio + maximum * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{ratio:.3g}×",
                va="center",
                ha="left",
                color="#3c4043",
            )
        axis.axvline(
            1.0,
            color="#3c4043",
            linestyle="--",
            linewidth=1.5,
        )
        axis.set_yticks(positions, labels=names)
        axis.invert_yaxis()
        axis.set_xlim(0, maximum)
        axis.set_xlabel("Metric / configured threshold")
        axis.set_title(
            "Validation metrics versus thresholds",
            loc="left",
            pad=18,
        )
        axis.text(
            0.0,
            1.035,
            (
                "Gray: parent; colored/hatched: candidate; "
                "dashed line: configured threshold (1×)"
                if has_parent
                else (
                    "Colored/hatched: candidate; dashed line: "
                    "configured threshold (1×)"
                )
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            color="#5f6368",
            fontsize=10,
        )
        axis.grid(True, axis="x")
        axis.grid(False, axis="y")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    return figure


def build_evaluation_report(
    output_dir: Path,
    *,
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
    parent_metrics: Mapping[str, float] | None = None,
    suffix: str = "",
) -> ReportArtifacts:
    """Compare validation metrics to configured acceptance thresholds."""

    report_path = output_dir / f"evaluation-report{suffix}.json"
    chart_path = output_dir / f"evaluation-metrics{suffix}.png"
    report: dict[str, Any] = {
        "version": 1,
        "kind": "evaluation",
        "renderer": f"matplotlib-{matplotlib.__version__}",
        "status": "unavailable",
        "chart": None,
        "metrics": {
            name: _json_number(value) for name, value in metrics.items()
        },
        "thresholds": {
            name: _json_number(value) for name, value in thresholds.items()
        },
        "parent_metrics": (
            {
                name: _json_number(value)
                for name, value in parent_metrics.items()
            }
            if parent_metrics is not None
            else None
        ),
    }
    try:
        names, candidate, parent = _evaluation_values(
            metrics,
            thresholds,
            parent_metrics,
        )
    except ValueError as error:
        report["reason"] = str(error)
        return ReportArtifacts(_write_json(report_path, report), None)
    figure = _evaluation_figure(names, candidate, parent)
    _save_figure(figure, chart_path)
    report.update(status="ready", chart=chart_path.name, series=names)
    return ReportArtifacts(_write_json(report_path, report), chart_path)


def _parity_sample(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    maximum: int = 20_000,
    outliers: int = 200,
) -> np.ndarray:
    if len(reference) <= maximum:
        return np.arange(len(reference), dtype=np.int64)
    outlier_count = min(outliers, maximum // 4)
    regular_count = maximum - outlier_count
    regular = np.linspace(
        0,
        len(reference) - 1,
        regular_count,
        dtype=np.int64,
    )
    residual = np.abs(predicted - reference)
    extreme = np.argpartition(residual, -outlier_count)[-outlier_count:]
    selected = np.unique(np.concatenate((regular, extreme)))
    return np.sort(selected[:maximum])


def _parity_figure(
    series: Mapping[str, ParitySeries],
    title: str,
) -> tuple[Figure, dict[str, dict[str, Any]]]:
    prepared: list[
        tuple[str, ParitySeries, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    panels: dict[str, dict[str, Any]] = {}
    for name, values in series.items():
        finite = np.isfinite(values.reference) & np.isfinite(
            values.predicted
        )
        reference = values.reference[finite]
        predicted = values.predicted[finite]
        if len(reference) < 1:
            continue
        selected = _parity_sample(reference, predicted)
        rmse = float(
            np.sqrt(np.mean(np.square(predicted - reference)))
        )
        panels[name] = {
            "unit": values.unit,
            "total_pairs": int(len(values.reference)),
            "finite_pairs": int(len(reference)),
            "plotted_pairs": int(len(selected)),
            "rmse": rmse,
            "reference_range": [float(reference.min()), float(reference.max())],
            "predicted_range": [float(predicted.min()), float(predicted.max())],
            "sampling": (
                "all"
                if len(selected) == len(reference)
                else "evenly_spaced_plus_largest_residuals"
            ),
        }
        prepared.append(
            (name, values, reference, predicted, selected)
        )
    if not prepared:
        raise ValueError("no parity series contains finite pairs")

    columns = 1 if len(prepared) == 1 else 2
    rows = math.ceil(len(prepared) / columns)
    with matplotlib.rc_context(_MATPLOTLIB_STYLE):
        figure = Figure(
            figsize=(10, 4.1 * rows + 0.7),
            layout="constrained",
        )
        axes = np.asarray(
            figure.subplots(rows, columns, squeeze=False)
        )
        figure.suptitle(
            title,
            fontsize=16,
            fontweight="bold",
        )
        display_names = {
            "energy": "Energy",
            "force": "Force components",
            "virial": "Virial components",
            "mforce": "Magnetic-force components",
        }
        colors = {
            "energy": "#65ad65",
            "force": "#9476b8",
            "virial": "#e8a35a",
            "mforce": "#5266bd",
        }
        for axis, (
            name,
            values,
            reference,
            predicted,
            selected,
        ) in zip(axes.flat, prepared, strict=False):
            x = reference[selected]
            y = predicted[selected]
            lower = float(min(reference.min(), predicted.min()))
            upper = float(max(reference.max(), predicted.max()))
            span = upper - lower
            padding = max(span * 0.05, abs(upper) * 1e-6, 1e-12)
            lower -= padding
            upper += padding
            axis.scatter(
                x,
                y,
                s=9,
                alpha=0.35,
                color=colors.get(name, "#0072B2"),
                edgecolors="none",
                rasterized=len(selected) > 5_000,
            )
            axis.plot(
                [lower, upper],
                [lower, upper],
                color="#3c4043",
                linestyle="--",
                linewidth=1.3,
            )
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_aspect("equal", adjustable="box")
            unit = f" ({values.unit})" if values.unit else ""
            axis.set_xlabel(f"Reference{unit}")
            axis.set_ylabel(f"Prediction{unit}")
            axis.set_title(
                display_names.get(name, name),
                loc="left",
                pad=10,
            )
            panel = panels[name]
            metric_scale = 1000 if values.unit.startswith("eV") else 1
            metric_unit = (
                values.unit.replace("eV", "meV", 1)
                if metric_scale == 1000
                else values.unit
            )
            axis.text(
                0.03,
                0.97,
                (
                    f"n={panel['finite_pairs']:,}\n"
                    f"RMSE={panel['rmse'] * metric_scale:.4g} {metric_unit}"
                ),
                transform=axis.transAxes,
                ha="left",
                va="top",
                color="#3c4043",
                bbox={
                    "facecolor": "#ffffff",
                    "edgecolor": "none",
                    "alpha": 0.85,
                    "pad": 3,
                },
            )
            residual = np.abs(predicted - reference) * metric_scale
            inset = axis.inset_axes([0.61, 0.10, 0.34, 0.28])
            limit = float(np.quantile(residual, 0.99)) * 1.05
            limit = max(limit, 1e-12)
            counts, edges = np.histogram(residual, bins=30, range=(0, limit))
            density = counts / max(float(counts.max()), 1.0)
            inset.stairs(
                density, edges, fill=True, color=colors.get(name, "#0072B2"), alpha=0.65
            )
            inset.set_xlim(0, limit)
            inset.set_yticks([])
            inset.set_xlabel(f"Absolute error ({metric_unit})", fontsize=7, labelpad=2)
            inset.xaxis.set_major_locator(MaxNLocator(3))
            inset.tick_params(axis="x", labelsize=7)
            inset.grid(False)
            panel["error_histogram"] = {
                "range": [0.0, limit],
                "unit": metric_unit,
                "counts": counts.tolist(),
                "edges": edges.tolist(),
                "outside_range": int(np.sum(residual > limit)),
            }
            axis.grid(False)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        for axis in axes.flat[len(prepared):]:
            axis.set_visible(False)
    return figure, panels


def build_parity_report(
    output_dir: Path,
    *,
    series: Mapping[str, ParitySeries],
    source: Mapping[str, Any] | None = None,
    suffix: str = "",
    stem: str = "evaluation-parity",
    title: str = "Auxiliary test: reference versus prediction",
) -> ReportArtifacts:
    """Plot reference/prediction agreement without conflating dataset roles."""

    report_path = output_dir / f"{stem}-report{suffix}.json"
    chart_path = output_dir / f"{stem}{suffix}.png"
    report: dict[str, Any] = {
        "version": 1,
        "kind": "evaluation_parity",
        "title": title,
        "renderer": f"matplotlib-{matplotlib.__version__}",
        "status": "unavailable",
        "chart": None,
        "source": dict(source or {}),
    }
    try:
        figure, panels = _parity_figure(series, title)
    except ValueError as error:
        report["reason"] = str(error)
        return ReportArtifacts(_write_json(report_path, report), None)
    _save_figure(figure, chart_path)
    report.update(
        status="ready",
        chart=chart_path.name,
        panels=panels,
    )
    return ReportArtifacts(_write_json(report_path, report), chart_path)


def build_training_parity_reports(
    output_dir: Path,
    *,
    outputs: Mapping[str, Path],
    backend: str,
    model: Path | None,
) -> dict[str, Path]:
    """Reuse NEP prediction files: predicted columns precede reference columns."""
    artifacts = {}
    for split in ("train", "test"):
        if split == "test" and not any(
            f"{name}_test.out" in outputs
            for name in ("energy", "force", "virial", "mforce")
        ):
            continue
        series = {}
        sources = {}
        warnings = []
        for name, width, unit in (
            ("energy", 1, "eV/atom"),
            ("force", 3, "eV/Å"),
            ("virial", 6, "eV/atom"),
            ("mforce", 3, "eV/μB"),
        ):
            path = outputs.get(f"{name}_{split}.out")
            if path is None:
                continue
            try:
                values = np.loadtxt(path, ndmin=2)
                if values.shape[1] != width * 2:
                    raise ValueError(
                        f"expected {width * 2} columns, got {values.shape[1]}"
                    )
                predicted, reference = values[:, :width], values[:, width:]
                if name == "virial":
                    # NEP uses -1e6 for a missing virial reference.
                    reference = np.where(reference == -1e6, np.nan, reference)
                series[name] = ParitySeries(reference, predicted, unit)
                sources[path.name] = file_sha256(path)
            except (OSError, ValueError) as error:
                warnings.append(f"{path.name}: {error}")
        role = "final-epoch model" if backend == "torchnep" else "trainer output"
        report = build_parity_report(
            output_dir,
            series=series,
            stem=f"training-parity-{split}",
            title=f"{'Training' if split == 'train' else 'Test'} set · {role}",
            source={
                "dataset_role": split,
                "backend": backend,
                "prediction_model_role": role,
                "prediction_model_sha256": (
                    file_sha256(model)
                    if model is not None and model.is_file()
                    else None
                ),
                "files": sources,
                "column_order": "prediction_then_reference",
                "warnings": warnings,
            },
        )
        artifacts[report.report.name] = report.report
        if report.chart is not None:
            artifacts[report.chart.name] = report.chart
    return artifacts


def build_selection_pca_report(
    output_dir: Path,
    *,
    descriptors: np.ndarray,
    selected_indices: Sequence[int],
    candidate_ids: Sequence[str],
    source: Mapping[str, Any] | None = None,
    stem: str = "selection-pca",
) -> ReportArtifacts:
    """Visualize eligible candidates in a shared PCA basis, without changing FPS."""
    report_path = output_dir / f"{stem}-report.json"
    chart_path = output_dir / f"{stem}.png"
    points = np.asarray(descriptors, dtype=np.float64)
    selected = np.asarray(selected_indices, dtype=int)
    report = {
        "version": 1,
        "kind": "selection_pca",
        "status": "unavailable",
        "chart": None,
        "source": dict(source or {}),
        "candidate_ids": list(candidate_ids),
        "selected_indices": selected.tolist(),
        "preprocessing": "centered reduced candidate descriptors; no feature rescaling",
        "scope": (source or {}).get(
            "candidate_scope",
            "eligible candidate pool after training-overlap removal and deduplication",
        ),
    }
    if len(points) == 0:
        report["reason"] = "no eligible candidate structures"
        return ReportArtifacts(_write_json(report_path, report), None)
    if (
        points.ndim != 2
        or len(points) != len(candidate_ids)
        or points.shape[1] == 0
        or not np.isfinite(points).all()
    ):
        raise ValueError("PCA requires finite descriptor rows matching candidate_ids")
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= len(points)):
        raise ValueError("selected indices are outside the candidate pool")
    # Bound the PCA fit cost and plot density; every selected structure is still projected and shown.
    fit_indices = np.linspace(0, len(points) - 1, min(len(points), 20_000), dtype=int)
    fit = points[fit_indices]
    center = fit.mean(axis=0)
    centered = fit - center
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    components = np.zeros((2, points.shape[1]))
    count = min(2, len(vectors))
    components[:count] = vectors[:count]
    for vector in components:
        if vector[np.argmax(np.abs(vector))] < 0:
            vector *= -1
    coordinates = (points - center) @ components.T
    variance = singular**2
    total = float(variance.sum())
    ratios = np.zeros(2)
    if total > 0:
        ratios[:count] = variance[:count] / total
    with matplotlib.rc_context(_MATPLOTLIB_STYLE):
        figure = Figure(figsize=(8, 6.2), layout="constrained")
        axis = figure.subplots()
        axis.scatter(
            coordinates[fit_indices, 0],
            coordinates[fit_indices, 1],
            s=25,
            color="#1f77b4",
            alpha=0.45,
            label="All candidate structures",
        )
        axis.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=9,
            color="#ff7f0e",
            label="Selected structures",
            zorder=3,
        )
        axis.set_xlabel(f"PC1 ({ratios[0]:.1%} variance)")
        axis.set_ylabel(f"PC2 ({ratios[1]:.1%} variance)")
        axis.set_title(
            f"Selection coverage · {len(selected)} / {len(points)} structures",
            loc="left",
        )
        if len(fit_indices) < len(points):
            axis.text(
                0.99,
                0.01,
                f"Background / PCA fit: {len(fit_indices):,} evenly spaced candidates",
                transform=axis.transAxes,
                ha="right",
                fontsize=8,
            )
        axis.grid(False)
        axis.legend(loc="best", frameon=True)
    _save_figure(figure, chart_path)
    report.update(
        status="ready",
        chart=chart_path.name,
        candidate_count=len(points),
        selected_count=len(selected),
        coordinates=coordinates.tolist(),
        components=components.tolist(),
        center=center.tolist(),
        explained_variance_ratio=ratios.tolist(),
        fit_indices=fit_indices.tolist(),
        rank=int(
            np.sum(singular > (singular[0] * max(centered.shape) * np.finfo(float).eps))
        ),
        note="PCA is a display projection, not the group-normalized FPS distance or a convergence criterion",
    )
    return ReportArtifacts(_write_json(report_path, report), chart_path)


__all__ = [
    "ParitySeries",
    "ReportArtifacts",
    "build_evaluation_report",
    "build_parity_report",
    "build_training_report",
    "build_training_parity_reports",
    "build_selection_pca_report",
]
