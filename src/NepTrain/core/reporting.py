"""Matplotlib workflow reports for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
from matplotlib.figure import Figure
import numpy as np


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
    "svg.fonttype": "none",
    "svg.hashsalt": "neptrain-report-v1",
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
            format="svg",
            metadata={"Date": None, "Creator": "NepTrain"},
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
    chart_path = output_dir / "training-convergence.svg"
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
        "sha256": sha256(loss_path.read_bytes()).hexdigest(),
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
    chart_path = output_dir / f"evaluation-metrics{suffix}.svg"
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
        if len(reference) < 2:
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
        raise ValueError(
            "no parity series contains at least two finite pairs"
        )

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
            "Validation parity: reference versus candidate",
            fontsize=16,
            fontweight="bold",
        )
        display_names = {
            "energy": "Energy",
            "force": "Force components",
            "virial": "Virial components",
            "mforce": "Magnetic-force components",
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
                color="#0072B2",
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
            axis.set_ylabel(f"Candidate prediction{unit}")
            axis.set_title(
                display_names.get(name, name),
                loc="left",
                pad=10,
            )
            panel = panels[name]
            axis.text(
                0.03,
                0.97,
                (
                    f"n={panel['finite_pairs']:,}\n"
                    f"RMSE={panel['rmse']:.4g}"
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
            axis.grid(True, which="major")
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
) -> ReportArtifacts:
    """Plot reference/prediction agreement for independent validation data."""

    report_path = output_dir / f"evaluation-parity-report{suffix}.json"
    chart_path = output_dir / f"evaluation-parity{suffix}.svg"
    report: dict[str, Any] = {
        "version": 1,
        "kind": "evaluation_parity",
        "renderer": f"matplotlib-{matplotlib.__version__}",
        "status": "unavailable",
        "chart": None,
        "source": dict(source or {}),
    }
    try:
        figure, panels = _parity_figure(series)
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


__all__ = [
    "ParitySeries",
    "ReportArtifacts",
    "build_evaluation_report",
    "build_parity_report",
    "build_training_report",
]
