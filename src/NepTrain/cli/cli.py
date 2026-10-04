#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/24 14:33
# @Author  : 兵
# @email    : 1747193328@qq.com
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import unicodedata
from NepTrain import __version__
from NepTrain.core.config import (
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_STRUCTURES_PER_LABEL_JOB,
    DEFAULT_STRUCTURES_PER_MODEL_JOB,
)


def _print_json(value):
    try:
        payload = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SystemExit(
            f"NepTrain: error: cannot serialize stable JSON output: {error}"
        ) from error
    print(payload)


def init_template(args):
    from NepTrain.core.template import init_template as implementation
    return implementation(args)


def run_perturb(args):
    from NepTrain.core.perturb import run_perturb as implementation
    return implementation(args)


def run_select(args):
    from NepTrain.core.select import run_select as implementation
    return implementation(args)


def run_smoke_command(args):
    from dataclasses import asdict

    from NepTrain.core.smoke import run_smoke

    report = run_smoke(
        args.output,
        profile=args.profile,
        seed=args.seed,
        max_selected=args.max_selected,
        force=args.force,
    )
    result = {"protocol": "neptrain.smoke.v1", "smoke": asdict(report)}
    if args.iterations:
        from NepTrain.core.toy_iteration import run_toy_iteration_smoke

        iteration = run_toy_iteration_smoke(
            Path(args.output) / "iteration",
            profile="spin" if args.profile == "recovery" else args.profile,
            generations=args.iterations,
            seed=args.seed,
            max_selected=args.max_selected,
            force=args.force,
        )
        result["iteration"] = asdict(iteration)
    if args.workflow:
        from NepTrain.core.workflow_smoke import run_workflow_smoke

        result["workflow"] = run_workflow_smoke(
            Path(args.output) / "workflow",
            profile="spin" if args.profile == "recovery" else args.profile,
            seed=args.seed,
        )
    _print_json(result)


_STATE_LABELS = {
    "prepared": "待启动",
    "running": "运行中",
    "waiting": "等待中",
    "degraded": "连接异常",
    "paused": "已暂停",
    "complete": "已完成",
    "failed": "失败",
    "rejected": "验收未通过",
    "stalled": "已停滞",
    "budget_exhausted": "代次预算耗尽",
    "coverage_exhausted": "采样覆盖耗尽",
    "damaged": "状态损坏",
}
_STAGE_LABELS = {
    "train": "训练",
    "validate": "模型检查",
    "explore": "采样",
    "select": "选样",
    "label": "标注",
    "diagnose": "诊断",
    "merge": "合并训练集",
    "retrain": "重新训练",
    "evaluate": "评估",
    "update": "更新训练集",
}


def _updated_text(value):
    if not value:
        return "暂无时间记录"
    try:
        observed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        seconds = max(0, int((now - observed.astimezone(timezone.utc)).total_seconds()))
        if seconds < 5:
            age = "刚刚"
        elif seconds < 60:
            age = f"{seconds} 秒前"
        elif seconds < 3600:
            age = f"{seconds // 60} 分钟前"
        else:
            age = f"{seconds // 3600} 小时前"
        return f"{observed.astimezone().strftime('%H:%M:%S')}（{age}）"
    except (TypeError, ValueError):
        return str(value)


def _ps(value):
    return f"{float(value):.4g}"


_MATURITY_LABELS = {
    "untested": "未测试",
    "smoke_passed": "短试跑",
    "short_stable": "短程",
    "long_stable": "长程",
    "production_ready": "生产",
}


def _sampling_cell(cell):
    temperature = f"{float(cell['temperature']):g} K"
    state = cell["state"]
    if state == "pending":
        return f"{temperature} ○ 未开始"
    if state == "attempted":
        return f"{temperature} 已尝试（缺少完成证据）"
    labels = {
        "complete": "轨迹完成",
        "failed": "有失败",
        "active": "执行中",
        "collected": "执行结束，待科学阶段确认",
    }
    parts = [f"{temperature} {labels.get(state, state)}"]
    if cell.get("total"):
        prefix = "执行完成" if cell.get("live") else "轨迹正常"
        parts.append(f"{prefix} {cell['completed']}/{cell['total']}")
        for key, label in (
            ("running", "运行"),
            ("waiting", "排队"),
            ("failed", "失败"),
            ("unknown", "状态未知"),
        ):
            if cell.get(key):
                parts.append(f"{label} {cell[key]}")
    if cell.get("live"):
        current, upper = cell.get("current_ps"), cell.get("current_ps_max")
        target, lower = cell.get("target_ps"), cell.get("target_ps_min")
        if current is not None and target is not None:
            progress = (
                _ps(current)
                if upper is None or upper == current
                else f"{_ps(current)}–{_ps(upper)}"
            )
            goal = (
                _ps(target)
                if lower is None or lower == target
                else f"{_ps(lower)}–{_ps(target)}"
            )
            parts.append(
                f"已读轨迹 {progress}/{goal} ps（{cell.get('readable', 1)}/{cell['total']} 可读）"
            )
        else:
            parts.append("轨迹时间暂不可读")
    levels = [
        _MATURITY_LABELS.get(level, level)
        for level in cell.get("target_levels", [])
        if level != "unknown"
    ]
    if levels:
        parts.append("目标：" + "/".join(levels))
    maturity = cell.get("maturities", {})
    if maturity:
        parts.append(
            "已记录等级："
            + "、".join(
                f"{_MATURITY_LABELS.get(k, k)} {v}" for k, v in maturity.items()
            )
        )
    coverage = cell.get("production_coverage")
    if coverage:
        model = str(cell.get("production_model_sha256") or "未知")[:8]
        parts.append(
            f"生产覆盖 {coverage['verified']}/{coverage['total']}（模型 {model}）"
        )
    return " | ".join(parts)


def _metric_cell(value, previous, *, scale):
    if value is None:
        return "-"
    number = float(value) * scale
    text = f"{number:.4g}"
    if previous is None or float(previous) == 0:
        return text
    change = (float(value) - float(previous)) / abs(float(previous)) * 100
    if abs(change) < 0.5:
        return text
    arrow = "↓" if change < 0 else "↑"
    return f"{text} {arrow}{abs(change):.0f}%"


def _display_width(value):
    return sum(
        2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        for character in str(value)
    )


def _table(rows):
    widths = [
        max(_display_width(row[index]) for row in rows)
        for index in range(len(rows[0]))
    ]
    for row in rows:
        print(
            "  ".join(
                str(value) + " " * (widths[index] - _display_width(value))
                for index, value in enumerate(row)
            ).rstrip()
        )


def _generation_state(generation, status):
    state = generation["state"]
    if state == "accepted":
        return "完成"
    if state == "rejected":
        return "未通过"
    if state == "not_started":
        return "未开始"
    if generation["generation"] == status.generation and status.stage:
        stage = _STAGE_LABELS.get(status.stage, status.stage)
        state = getattr(status, "state", "running")
        suffix = {
            "failed": "失败",
            "paused": "已暂停",
            "degraded": "连接异常",
            "waiting": "等待中",
            "stalled": "停滞",
            "damaged": "状态损坏",
            "prepared": "待开始",
        }.get(state, "中")
        return f"{stage}{suffix}"
    return "进行中"


def _print_precision(status, *, details=False):
    acquisition = status.precision_basis != "validation"
    metric_key = "acquisition_rmse" if acquisition else "validation_rmse"
    accepted_key = (
        "acquisition_accepted" if acquisition else "validation_thresholds_met"
    )
    print()
    observed = [
        g
        for g in status.generations
        if any(v is not None for v in g["quality"].get(metric_key, {}).values())
        and g.get("kind") != "finalization"
    ]
    for generation in status.generations:
        if (
            acquisition
            and generation.get("kind") == "finalization"
            and generation["state"] != "not_started"
        ):
            print(
                f"G{generation['generation']} 最终训练：{_generation_state(generation, status)}（本代不采样）"
            )
    if not observed:
        print("精度变化：暂无新增结构的训练前预测结果")
        return
    visible = observed if details else observed[-3:]
    print(
        "新增结构预测精度（训练前 RMSE）："
        if acquisition
        else "辅助测试精度（仅供参考）："
    )
    names = [
        (key, title)
        for key, title in (
            ("energy_rmse", "E/meV·atom⁻¹"),
            ("force_rmse", "F/meV·Å⁻¹"),
            ("virial_rmse", "V/meV·atom⁻¹"),
            ("mforce_rmse", "M/meV/μB"),
        )
        if any(g["quality"][metric_key].get(key) is not None for g in visible)
    ]
    rows = [("代", "状态", "新标签", *[title for _, title in names], "本轮判据")]
    previous = {}
    for generation in visible:
        quality = generation["quality"]
        accepted = quality.get(accepted_key)
        rows.append(
            (
                f"G{generation['generation']}",
                _generation_state(generation, status),
                (
                    quality.get("acquisition_count")
                    if quality.get("acquisition_count") is not None
                    else "未知"
                ),
                *[
                    _metric_cell(
                        quality[metric_key].get(key),
                        None if acquisition else previous.get(key),
                        scale=1000,
                    )
                    for key, _ in names
                ],
                (
                    "通过"
                    if accepted is True
                    else "未通过" if accepted is False else "未判定"
                ),
            )
        )
        previous.update(
            {
                key: value
                for key, value in quality[metric_key].items()
                if value is not None
            }
        )
    _table(rows)
    if acquisition:
        print("每代评估结构不同，跨代误差变化不直接代表模型提升。")
    if len(visible) < len(observed):
        print("仅显示最近 3 次评估；--details 查看历史及辅助 test。")


def _print_acceptance(status):
    policy = getattr(status, "convergence_policy", None) or {}
    evaluated = [
        g
        for g in status.generations
        if g["quality"].get("acquisition_count") is not None
    ]
    if not policy or not evaluated:
        return
    latest = evaluated[-1]
    quality = latest["quality"]
    rows = [("判据", "当前值", "要求", "结果")]

    def add(label, value, threshold, *, minimum=False, scale=1, unit=""):
        if threshold is None:
            return
        passed = value is not None and (
            value >= threshold if minimum else value <= threshold
        )
        rows.append(
            (
                label,
                "缺数据" if value is None else f"{value * scale:.4g}{unit}",
                f"{'≥' if minimum else '≤'}{threshold * scale:.4g}{unit}",
                "待确认" if value is None else "达标" if passed else "未达标",
            )
        )

    names = {
        "energy": ("能量", " meV/atom"),
        "force": ("力", " meV/Å"),
        "virial": ("Virial", " meV/atom"),
        "mforce": ("磁力", " meV/μB"),
    }
    for metric, limit in policy.get("acquisition_max_rmse", {}).items():
        label, unit = names[metric.removesuffix("_rmse")]
        add(
            label + " RMSE",
            quality["acquisition_rmse"].get(metric),
            limit,
            scale=1000,
            unit=unit,
        )
        attempts = quality.get("attempt_metrics", {})
        if attempts:
            worst_id = max(
                attempts,
                key=lambda key: (
                    attempts[key].get(metric)
                    if attempts[key].get(metric) is not None
                    else float("inf")
                ),
            )
            worst = attempts[worst_id].get(metric)
            if worst is None or worst > limit:
                add(
                    f"最差轨迹 {worst_id[:8]} {label} RMSE",
                    worst,
                    limit,
                    scale=1000,
                    unit=unit,
                )
    for metric, limit in policy.get("acquisition_min_r2", {}).items():
        add(
            names[metric.removesuffix("_r2")][0] + " R²",
            quality["acquisition_r2"].get(metric),
            limit,
            minimum=True,
        )
    for key, label in (
        ("element_force_r2", "最差元素"),
        ("condition_force_r2", "最差温压条件"),
    ):
        values = quality.get(key, {})
        worst = min(values, key=values.get) if values else None
        add(
            f"{label} {worst or '未知'} 力 R²",
            values.get(worst),
            policy.get("group_min_force_r2"),
            minimum=True,
        )
    outliers = quality.get("outlier_fraction", {})
    worst = max(outliers, key=outliers.get) if outliers else None
    add(
        f"异常残差比例 {worst or ''}",
        outliers.get(worst),
        policy.get("max_outlier_fraction"),
        scale=100,
        unit="%",
    )
    add(
        "有效新标签",
        quality.get("acquisition_count"),
        policy.get("min_selected", 1),
        minimum=True,
    )
    if quality.get("acquisition_convergence_streak") is not None:
        add(
            "连续达标代数",
            quality["acquisition_convergence_streak"],
            quality.get(
                "acquisition_convergence_required",
                policy.get("consecutive_generations", 1),
            ),
            minimum=True,
        )
    print(f"\n最近评估判据 · G{latest['generation']}：")
    _table(rows)


def _print_current_progress(status, generation):
    record = next(
        (g for g in status.generations if g["generation"] == generation), None
    )
    if not record:
        return
    print(f"\n本轮进展 · G{generation}：")
    if record.get("kind") == "finalization":
        print("最终训练（本代不采样）")
    else:
        completed = record.get("completed_stages", ())
        stages = ("train", "explore", "select", "label", "evaluate", "update")
        if "diagnose" in completed or "merge" in completed or "retrain" in completed:
            stages = (
                "train",
                "explore",
                "select",
                "label",
                "diagnose",
                "merge",
                "retrain",
                "evaluate",
            )
        stages = record.get("stage_sequence", stages)
        cells = []
        for stage in stages:
            label = _STAGE_LABELS[stage]
            if stage in completed:
                label += " ✓"
            elif stage == status.stage and generation == status.generation:
                label = (
                    _generation_state(record, status)
                    if record["state"] != "not_started"
                    else label + "待开始"
                )
            else:
                label += "待开始"
            cells.append(label)
        line = ""
        for cell in cells:
            candidate = f"{line} → {cell}" if line else cell
            if _display_width(candidate) > 88 and line:
                print(line + " →")
                line = cell
            else:
                line = candidate
        if line:
            print(line)
    sampling = record.get("sampling", {})
    parts = []
    for key, label in (
        ("candidate_count", "候选"),
        ("candidate_count_after_deduplication", "排重后候选"),
        ("selected_count", "选中"),
        ("labeled_count", "已标注"),
    ):
        if sampling.get(key) is not None:
            parts.append(f"{label} {sampling[key]}")
    if parts:
        print(" → ".join(parts))
    training = record.get("training", {})
    before, after = training.get("before_count"), training.get("merged_count")
    if before is not None:
        print(
            f"训练集：{before} → {after}（新增 {training.get('added_count', 0)}）"
            if after is not None
            else f"训练集：{before} 帧（本轮新标签尚未合并）"
        )
    progress = getattr(status, "training_progress", None)
    if progress:
        label = "epoch" if progress.get("backend") == "torchnep" else "训练步"
        print(
            f"训练记录：{label} {progress['step']:g} | loss {progress['loss']:.4g} | 日志更新 {_updated_text(progress['updated_at'])}"
        )
    current_jobs = [job for job in status.jobs if job.get("current")]
    if current_jobs:
        counts = {}
        for job in current_jobs:
            state = str(job["state"]).upper()
            label = {
                "COMPLETED": "执行完成",
                "RUNNING": "运行",
                "PENDING": "排队",
                "QUEUED": "排队",
                "SUBMITTED": "排队",
                "FAILED": "失败",
                "SKIPPED": "跳过",
                "UNKNOWN": "未知",
            }.get(state, state)
            counts[label] = counts.get(label, 0) + 1
        print(
            "当前任务："
            + " | ".join(f"{label} {count}" for label, count in counts.items())
            + "（执行状态，尚不代表科学验收）"
        )


def _print_reports(status, *, details=False):
    available = [g for g in status.generations if g.get("reports")]
    if not available:
        return
    generation = available[-1]
    reports = generation["reports"]
    labels = {
        "training-parity-train": "训练集对角线",
        "training-parity-test": "训练器 test 对角线",
        "acquisition-parity": "新标签对角线",
        "selection-pca": "选样 PCA",
        "training": "训练曲线",
        "evaluation-parity": "辅助 test 对角线",
    }
    shown = False
    for report in reports:
        name = Path(report["report"]).name.removesuffix("-report.json")
        if not details and name in {"training-parity-test", "evaluation-parity"}:
            continue
        if not shown:
            print(f"\n图片 · G{generation['generation']}：")
            shown = True
        label = labels.get(name, name)
        if report.get("chart"):
            print(f"  {label}：{report['chart']}")
        else:
            reason = report.get("reason") or "没有有效绘图数据"
            if reason in {
                "no parity series contains finite pairs",
                "no eligible candidate structures",
            }:
                reason = (
                    "没有有效的新标签预测数据"
                    if name == "acquisition-parity"
                    else (
                        "没有可选候选结构"
                        if name == "selection-pca"
                        else "训练器未提供有效的预测输出"
                    )
                )
            print(f"  {label}：未出图（{reason}）")


def _compact_job_ids(jobs):
    values = [str(job["job_id"]) for job in jobs if job.get("job_id")]
    if not values:
        return "-"
    unique = list(dict.fromkeys(values))
    if len(unique) == 1:
        return unique[0]
    if all(value.isdigit() for value in unique):
        numbers = sorted(int(value) for value in unique)
        if numbers == list(range(numbers[0], numbers[-1] + 1)):
            return f"{numbers[0]}–{numbers[-1]}"
    preview = ", ".join(unique[:3])
    return preview + (f", …（共 {len(unique)} 个）" if len(unique) > 3 else "")


def _print_job_batches(jobs):
    groups = {}
    for job in jobs:
        key = (
            job.get("generation"),
            job.get("stage") or Path(job["script"]).name,
            job.get("attempt"),
            job.get("target"),
        )
        groups.setdefault(key, []).append(job)
    print()
    print("执行批次：")
    if not groups:
        print("  暂无执行任务")
        return
    state_labels = {
        "COMPLETED": "完成",
        "RUNNING": "运行",
        "PENDING": "等待",
        "SUBMITTED": "等待",
        "SUBMITTING": "等待",
        "LAUNCHING": "等待",
        "QUEUED": "等待",
        "NOT_SUBMITTED": "等待",
        "FAILED": "失败",
        "CANCELLED": "取消",
        "CANCELLING": "取消中",
        "SKIPPED": "跳过",
        "UNKNOWN": "未知",
    }
    for (generation, stage, attempt, target), batch in groups.items():
        counts = {}
        for job in batch:
            label = state_labels.get(str(job["state"]).upper(), str(job["state"]))
            counts[label] = counts.get(label, 0) + 1
        parts = [f"{len(batch)} 个任务"]
        ordered_labels = (
            "完成",
            "运行",
            "等待",
            "失败",
            "取消",
            "取消中",
            "跳过",
            "未知",
        )
        for label in ordered_labels:
            if counts.get(label):
                parts.append(f"{label} {counts[label]}")
        known = {"完成", "运行", "等待", "失败", "取消", "取消中", "跳过", "未知"}
        parts.extend(
            f"{label} {count}"
            for label, count in sorted(counts.items())
            if label not in known
        )
        prefix = f"G{generation} " if generation is not None else ""
        if target:
            prefix += f"[{target}] "
        print(
            f"  {prefix}{_STAGE_LABELS.get(str(stage), stage)} "
            f"{attempt}：{' | '.join(parts)} | Job {_compact_job_ids(batch)}"
        )


def _print_workflow_status(status, *, show_jobs: bool = True, details: bool = False):
    state = _STATE_LABELS.get(status.state, status.state)
    generation = status.generation or status.completed_generations
    stage = _STAGE_LABELS.get(status.stage, status.stage) if status.stage else None
    location = (
        f"第 {generation}/{status.total_generations} 代"
        if generation
        else f"0/{status.total_generations} 代"
    )
    if stage:
        suffix = (
            "中"
            if status.state in {"running", "waiting", "degraded"}
            else ""
        )
        location += f" | {stage}{suffix}"
    print(f"NepTrain · {status.workflow_id}")
    print(f"路径：{status.project_path}")
    print(f"状态：{state} | {location}")
    print(f"控制器心跳：{_updated_text(status.updated_at)}（不代表计算进度更新）")
    observations = [job.get("observed_at") for job in status.jobs if job.get("current")]
    known = [value for value in observations if value]
    if observations:
        print(
            f"任务状态：控制器缓存；最早检查 {_updated_text(min(known)) if known else '暂无记录'}（{len(known)}/{len(observations)} 有检查时间）"
        )
    if getattr(status, "convergence_configured", None) is False:
        print("收敛：未启用自动收敛；完成采样覆盖或用尽预算不代表精度达标。")
    if status.state in {
        "degraded",
        "paused",
        "failed",
        "rejected",
        "stalled",
        "budget_exhausted",
        "coverage_exhausted",
        "damaged",
    }:
        print(f"原因：{status.reason}")

    _print_current_progress(status, generation)
    if status.sampling_routes:
        print()
        print("采样进度：")
        total_failed = 0
        for route in status.sampling_routes:
            print(
                f"  {route['route_id']} | P={route.get('pressure', 0):g}（模板压力单位）"
            )
            for cell in route["temperatures"]:
                print("    " + _sampling_cell(cell))
            total_failed += int(route.get("failed", 0))
        if total_failed:
            print(f"异常：{total_failed} 条采样轨迹失败，失败证据已保留")

    _print_precision(status, details=details)
    _print_acceptance(status)
    visible_generations = [g for g in status.generations if g["state"] != "not_started"]
    for generation_record in (
        visible_generations if details else visible_generations[-1:]
    ):
        quality = generation_record["quality"]
        test_metrics = quality.get("validation_rmse", {})
        if (
            details
            and status.precision_basis != "validation"
            and any(value is not None for value in test_metrics.values())
        ):
            values = ", ".join(
                f"{name}={float(value) * 1000:.4g}"
                for name, value in test_metrics.items()
                if value is not None
            )
            print(
                f"G{generation_record['generation']} 辅助测试（仅供参考，E/V: meV/atom，F: meV/Å，M: meV/μB）：{values}"
            )
        for warning in quality.get("validation_warnings", []):
            print(f"提示 G{generation_record['generation']}：{warning}")
    decisions = [
        generation
        for generation in status.generations
        if generation["quality"].get("convergence_reasons")
        or generation["quality"].get("generation_disposition")
    ]
    if decisions:
        latest = decisions[-1]
        quality = latest["quality"]
        print()
        if quality.get("workflow_converged"):
            print("收敛判断：采样判据已通过，最终模型训练完成。")
        elif quality.get("finalization_pending"):
            print("收敛判断：采样精度与生产覆盖已通过，进入最终训练。")
        else:
            print(f"收敛判断（G{latest['generation']}）：尚未收敛")
        for reason in quality.get("convergence_reasons", []):
            print(f"  · {reason}")
    _print_reports(status, details=details)
    if show_jobs:
        print("\n任务状态来自控制器上次记录；历史失败可能已由后续重试恢复。")
        _print_job_batches(status.jobs)
    if status.notifications:
        notification = status.notifications
        notification_state = {
            "configured": "已配置",
            "ok": "正常",
            "degraded": "异常",
        }.get(notification["state"], notification["state"])
        print()
        print(
            "通知：飞书"
            f"{notification_state} | 成功 {notification['delivered']} | "
            f"失败 {notification['failed']}"
        )
        if notification.get("last_error"):
            print(f"通知错误：{notification['last_error']}")
    if status.state in {"failed", "rejected", "stalled", "damaged"}:
        root = Path(status.project_path)
        print(f"日志：{root / 'logs'}")
        if generation:
            print(f"本代计算与报告：{root / 'generations' / f'{generation:04d}'}")
        failed_jobs = [job for job in status.jobs if str(job["state"]).upper() in {"FAILED", "SKIPPED"}]
        for job in failed_jobs[-3:]:
            if job.get("detail"):
                print(f"  {job.get('stage', '')} / Job {job.get('job_id')}: {job['detail']}")
            if job.get("bundle"):
                print(f"  任务目录（stdout.log / output）：{job['bundle']}")
        if status.state in {"failed", "rejected"}:
            print("先修复日志中的原因，再执行恢复；resume 会重试未完成阶段。")
            print("若需更改固定的结构、模板或策略，请修改原始项目并用新的 --output 目录运行。")
    result_root = Path(status.project_path) / "results"
    available_results = [result_root / name for name in ("nep.txt", "train.xyz")
                         if (result_root / name).is_file()]
    if available_results:
        print("结果（流程已完成）：" if status.state == "complete" else "当前结果（尚未确认流程收敛）：")
        for path in available_results:
            print(f"  {path}")
        print(f"逐代报告：{Path(status.project_path) / 'generations'}")
    if status.next_action and status.state not in {"running", "waiting", "degraded"}:
        print(f"下一步：{status.next_action}")


def _print_workflow_control(payload, *, json_output=False):
    if json_output:
        _print_json(payload)
        return
    action = payload.get("action", "")
    labels = {"prepare": "已准备，尚未启动", "start": "已启动", "resume": "已恢复",
              "restart": "已重新启动", "restart_preview": "重算预览（未执行）",
              "repair": "已修复", "noop": "无需重复执行", "stop": "已停止"}
    name = payload.get("workflow_id") or Path(payload.get("project", "workflow")).name
    print(f"NepTrain · {name}")
    if action:
        print(f"操作：{labels.get(action, action)}")
    root = payload.get("project")
    if root:
        print(f"路径：{root}")
    if "total_model_generations" in payload:
        print(f"采样代预算：{payload['previous_model_generations']} → "
              f"{payload['total_model_generations']}（增加 {payload['added_generations']} 代）")
    for key, label in (("reused_stages", "保留阶段"), ("restarted_stages", "重算阶段")):
        if key in payload:
            print(f"{label}：{', '.join(payload[key]) or '无'}")
    if "retried_tasks" in payload:
        print(f"任务：保留 {payload.get('preserved_tasks', 0)}，重试 {payload['retried_tasks']}")
    if isinstance(payload.get("current_execution"), dict):
        execution = payload["current_execution"]
        print(f"当前任务：{execution.get('action', '-')}；{execution.get('detail', '')}")
    if "controller_exit_code" in payload:
        print(f"Controller 退出码：{payload['controller_exit_code']}")
    if root:
        print(f"查看状态：neptrain workflow status {shlex.quote(root)} --jobs")
        print(f"日志目录：{Path(root) / 'logs'}")
    if payload.get("next_action"):
        print(f"下一步：{payload['next_action']}")


def run_project_command(args):
    """Start from either a project YAML file or a prepared workflow."""

    from NepTrain.core.workflow import (
        WorkflowError,
        prepare_workflow,
        start_workflow,
    )
    from NepTrain.core.controller import start_controller

    project = Path(args.project).expanduser()
    try:
        if project.is_dir():
            invalid = [
                option
                for option, value in (
                    ("--initial-training", args.initial_training),
                    ("--output", args.output),
                    ("--workflow-id", args.workflow_id),
                    ("--prepare-only", args.prepare_only),
                )
                if value
            ]
            if invalid:
                raise WorkflowError(
                    f"{', '.join(invalid)} can only be used with a project "
                    "YAML file"
                )
            result = start_workflow(
                project,
                foreground=getattr(args, "foreground", False),
                poll_interval=getattr(args, "poll_interval", None),
            )
            payload = _workflow_resume_payload(result)
        else:
            initial_training = args.initial_training
            output = args.output
            if not initial_training or not output:
                from NepTrain.core.config import ConfigError, load_config

                try:
                    config, _ = load_config(project)
                except ConfigError as error:
                    raise WorkflowError(
                        f"invalid project configuration: {error}"
                    ) from error
                value = config.get("training", {}).get("initial_path")
                if value and not initial_training:
                    path = Path(value).expanduser()
                    initial_training = str(
                        (project.parent / path).resolve()
                        if not path.is_absolute()
                        else path.resolve()
                    )
                if not output:
                    workflow_id = str(
                        config.get("workflow", {}).get("id", "")
                    ).strip()
                    if workflow_id:
                        output = str((project.parent / workflow_id).resolve())
            if not initial_training or not output:
                raise WorkflowError(
                    "starting from a project file requires "
                    "training.initial_path (or --initial-training) and "
                    "workflow.id (or --output)"
                )
            preparation = prepare_workflow(
                project,
                initial_training,
                output,
                workflow_id=args.workflow_id,
            )
            payload = {
                "protocol": "neptrain.workflow-control.v1",
                "workflow_id": preparation.workflow_id,
                "action": "prepare" if args.prepare_only else "start",
                "project": str(preparation.output_dir),
                "manifest": str(preparation.manifest),
            }
            if args.prepare_only:
                payload["next_action"] = (
                    f"neptrain workflow run "
                    f"{shlex.quote(str(preparation.output_dir))}"
                )
            else:
                try:
                    controller_result = start_controller(
                        preparation.output_dir,
                        foreground=getattr(args, "foreground", False),
                        poll_interval=getattr(args, "poll_interval", None),
                    )
                except Exception as error:
                    raise WorkflowError(str(error)) from error
                if getattr(args, "foreground", False):
                    payload["controller_exit_code"] = controller_result
                else:
                    payload["controller_pid"] = controller_result
    except WorkflowError:
        raise
    _print_workflow_control(payload, json_output=getattr(args, "json", False))
    return payload.get("controller_exit_code", 0)


def run_status_command(args):
    from dataclasses import asdict
    from NepTrain.core.workflow import WorkflowError, workflow_status

    try:
        status = workflow_status(args.project)
    except WorkflowError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    if args.json:
        _print_json(
            {
                "protocol": "neptrain.workflow-status.v1",
                **asdict(status),
            }
        )
    else:
        _print_workflow_status(
            status, show_jobs=args.jobs, details=getattr(args, "details", False)
        )


def run_resume_command(args):
    from NepTrain.core.workflow import WorkflowError, resume_workflow

    try:
        result = resume_workflow(args.project)
    except WorkflowError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    payload = _workflow_resume_payload(result)
    _print_workflow_control(payload, json_output=getattr(args, "json", False))


def run_restart_command(args):
    from dataclasses import asdict

    from NepTrain.core.workflow import WorkflowError, restart_workflow
    from NepTrain.core.workflow_workspace import WorkflowWorkspace

    try:
        result = restart_workflow(
            args.project,
            generation=args.generation,
            from_stage=args.from_stage,
            task_scope=args.tasks,
            dry_run=args.dry_run,
            foreground=args.foreground,
            poll_interval=args.poll_interval,
        )
    except WorkflowError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    payload = asdict(result)
    payload["protocol"] = "neptrain.workflow-restart.v1"
    payload["manifest"] = str(result.manifest)
    payload["project"] = str(WorkflowWorkspace.locate(result.manifest).root)
    if result.controller_pid is None:
        payload.pop("controller_pid")
    if result.controller_exit_code is None:
        payload.pop("controller_exit_code")
    _print_workflow_control(payload, json_output=getattr(args, "json", False))


def _workflow_resume_payload(result):
    from dataclasses import asdict
    from NepTrain.core.workflow_workspace import WorkflowWorkspace

    payload = asdict(result)
    payload["protocol"] = "neptrain.workflow-control.v1"
    payload["manifest"] = str(result.manifest)
    payload["project"] = str(WorkflowWorkspace.locate(result.manifest).root)
    if result.controller_pid is not None:
        payload.pop("controller_exit_code", None)
    elif result.controller_exit_code is not None:
        payload.pop("controller_pid", None)
    else:
        payload.pop("controller_pid", None)
        payload.pop("controller_exit_code", None)
    return payload


def run_extend_command(args):
    from NepTrain.core.workflow import WorkflowError, extend_workflow, workflow_status
    from NepTrain.core.workflow_workspace import WorkflowWorkspace
    from NepTrain.core.config import load_config

    try:
        workspace = WorkflowWorkspace.locate(args.project)
        previous_config, _ = load_config(workspace.project_file)
        previous_total = int(previous_config["workflow"]["max_model_generations"])
        preparation = extend_workflow(args.project, args.generations)
    except WorkflowError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    from NepTrain.core.config import load_config
    config, _ = load_config(preparation.config_file)
    total = int(config["workflow"]["max_model_generations"])
    _print_workflow_control(
        {
            "protocol": "neptrain.workflow-extend.v1",
            "workflow_id": preparation.workflow_id,
            "total_model_generations": total,
            "previous_model_generations": previous_total,
            "added_generations": total - previous_total,
            "project": str(preparation.output_dir),
            "next_action": workflow_status(preparation.output_dir).next_action,
        },
        json_output=getattr(args, "json", False),
    )


def run_stop_command(args):
    from NepTrain.core.controller import ControllerError, stop_workflow

    try:
        result = stop_workflow(
            args.project, cancel_jobs=bool(args.cancel_jobs)
        )
    except ControllerError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    _print_workflow_control(
        {
            "protocol": "neptrain.workflow-stop.v1",
            **result,
            "action": "stop",
        },
        json_output=getattr(args, "json", False),
    )


def run_controller_command(args):
    from NepTrain.core.controller import run_controller

    return run_controller(args.project, poll_interval=args.poll_interval)


def run_stage_worker_command(args):
    from NepTrain.core.execution import run_stage_worker

    return run_stage_worker(args.bundle)


def run_stage_verify_command(args):
    from NepTrain.core.execution import verify_stage_task

    verify_stage_task(args.bundle)
    return 0


def run_stage_verify_many_command(args):
    from NepTrain.core.execution import verify_stage_tasks

    verify_stage_tasks(args.bundles)
    return 0


def _doctor_resource_contract(config, project):
    """Resolve the authoritative resource contract for a labeling Adapter."""

    backend = str(config.get("labeling", {}).get("backend", "vasp"))
    if backend in {"model", "toy"}:
        return None
    labeling = config["labeling"]
    execution = config["execution"]
    target_name = str(execution["stage_targets"]["labeling"])
    target = execution["targets"][target_name]
    resource_root = (
        target.get("labeling_resource_path")
        or labeling.get("resource_path")
    )
    if not resource_root:
        raise ValueError("labeling Adapter has no configured resource root")
    if not target.get("labeling_resource_path"):
        candidate = Path(str(resource_root)).expanduser()
        if not candidate.is_absolute():
            resource_root = str((project.parent / candidate).resolve())
    if backend == "vasp":
        from NepTrain.core.dft.vasp.resources import vasp_resource_files

        manifest = Path(
            str(labeling["potcar_manifest_path"])
        ).expanduser()
        label = "VASP POTCAR"
        records = vasp_resource_files(
            manifest if manifest.is_absolute() else project.parent / manifest
        )
    elif backend == "abacus":
        from NepTrain.core.dft.abacus.resources import abacus_resource_files

        manifest = Path(
            str(labeling["resource_manifest_path"])
        ).expanduser()
        label = "ABACUS pseudopotential/orbital"
        records = abacus_resource_files(
            manifest if manifest.is_absolute() else project.parent / manifest
        )
    else:  # validated schema owns this invariant
        raise ValueError(f"unsupported labeling backend: {backend}")
    return target_name, str(resource_root), label, records


def _doctor_resource_probe(resource_root, records):
    lines = [
        "set -eo pipefail",
        f"resource_root={shlex.quote(str(resource_root))}",
        'case "$resource_root" in "~/"*) '
        'resource_root="$HOME/${resource_root#~/}";; esac',
        'test -d "$resource_root" || { '
        'echo "resource root is missing: $resource_root" >&2; exit 3; }',
        "if command -v sha256sum >/dev/null; then",
        "  file_sha256() { sha256sum \"$1\" | awk '{print $1}'; }",
        "elif command -v shasum >/dev/null; then",
        "  file_sha256() { shasum -a 256 \"$1\" | awk '{print $1}'; }",
        "else",
        '  echo "sha256sum or shasum is required" >&2',
        "  exit 4",
        "fi",
    ]
    for record in records:
        relative = str(record["path"])
        expected = str(record["sha256"])
        lines.extend(
            [
                f"resource_path=\"$resource_root\"/{shlex.quote(relative)}",
                'test -f "$resource_path" || { '
                'echo "resource file is missing: $resource_path" >&2; exit 5; }',
                'actual=$(file_sha256 "$resource_path")',
                f'test "$actual" = {shlex.quote(expected)} || {{ '
                'echo "resource hash mismatch: $resource_path" >&2; exit 6; }',
            ]
        )
    return "\n".join(lines) + "\n"


def _doctor_command_tools(command: str) -> list[str]:
    """Return the launcher and payload executables from a configured command."""

    import shlex

    tokens = shlex.split(command)
    if not tokens:
        return []
    tools = [tokens[0]]
    for token in reversed(tokens[1:]):
        if (
            token.startswith("-")
            or token.replace(".", "", 1).isdigit()
            or ("=" in token and "/" not in token)
        ):
            continue
        if token not in tools:
            tools.append(token)
        break
    return tools


def _doctor_target_requirements(config, target_name, target, *, base_dir=None):
    """Resolve the commands and Python packages used on one project target."""

    execution = config["execution"]
    stage_targets = execution["stage_targets"]
    roles = {
        role
        for role, name in stage_targets.items()
        if str(name) == target_name
    }
    if target_name in {
        str(name)
        for name in execution.get("sampling_route_targets", {}).values()
    }:
        roles.add("sampling")

    tools = []
    packages = []

    def add_command(command):
        tools.extend(_doctor_command_tools(command))

    if target.executor == "slurm":
        tools.extend(["sbatch", "squeue", "sacct"])
    add_command(target.command)
    environment = target.environment
    if "training" in roles:
        backend = str(config["training"]["backend"])
        if backend == "gpumd":
            add_command(environment.get("NEPTRAIN_NEP_COMMAND", "nep"))
        else:
            packages.append("torchnep")
    if "sampling" in roles:
        backend = str(config["md"]["backend"])
        if backend == "gpumd":
            add_command(environment.get("NEPTRAIN_GPUMD_COMMAND", "gpumd"))
        else:
            add_command(environment.get("NEPTRAIN_LMP_COMMAND", "lmp"))
            if (target.cpus_per_task or 1) > 1:
                add_command(environment.get("NEPTRAIN_MPIEXEC", "mpirun"))
    if "labeling" in roles:
        labeling = config["labeling"]
        backend = str(labeling.get("backend", "vasp"))
        if backend == "vasp":
            from NepTrain.core.dft.vasp.io import default_vasp_command

            command = environment.get("NEPTRAIN_VASP_COMMAND")
            if command is None:
                input_file = labeling.get("input_path")
                if input_file is not None:
                    input_file = Path(input_file).expanduser()
                    if not input_file.is_absolute() and base_dir is not None:
                        input_file = Path(base_dir) / input_file
                command = default_vasp_command(input_file)
            add_command(command)
        elif backend == "abacus":
            add_command(
                environment.get(
                    "NEPTRAIN_ABACUS_COMMAND",
                    "mpirun -n 1 abacus",
                )
            )
        elif backend == "model":
            runner = shlex.split(str(labeling["runner"]))
            if runner:
                tools.append(runner[0])
            if len(runner) >= 3 and runner[:2] == [
                "neptrain",
                "model-worker",
            ]:
                if runner[2] == "mace":
                    packages.append("mace.calculators")
                elif runner[2] == "deepmd":
                    packages.append("deepmd.calculator")
                elif runner[2] == "tace":
                    tools.append("tace-eval")
    return (
        sorted(set(tools)),
        sorted(set(packages)),
        sorted(roles),
    )


def _doctor_target_probe(target, tools, packages, setup_path):
    import shlex

    lines = ["set -eo pipefail"]
    if setup_path is not None:
        if setup_path.is_file():
            lines.append(setup_path.read_text(encoding="utf-8"))
        else:
            setup_text = str(setup_path)
            setup_source = (
                f'"$HOME"/{shlex.quote(setup_text[2:])}'
                if setup_text.startswith("~/")
                else shlex.quote(setup_text)
            )
            lines.extend(
                [
                    f"test -r {setup_source} || {{ "
                    f"echo {shlex.quote(f'missing setup script: {setup_path}')} "
                    ">&2; exit 2; }",
                    f"source {setup_source}",
                ]
            )
    for key, value in target.environment.items():
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.extend(["set +e", "status=0"])
    required_tools = [*tools]
    if packages and "python" not in required_tools:
        required_tools.append("python")
    for tool in required_tools:
        lines.append(
            f"command -v {shlex.quote(tool)} >/dev/null || {{ "
            f"echo {shlex.quote(f'missing command: {tool}')} >&2; status=1; }}"
        )
    for package in packages:
        lines.append(
            "python -c "
            + shlex.quote(
                "import importlib, sys; importlib.import_module(sys.argv[1])"
            )
            + " "
            + shlex.quote(package)
            + " >/dev/null 2>&1 || { "
            + f"echo {shlex.quote(f'missing Python package: {package}')} >&2; "
            + "status=1; }"
        )
    lines.append('exit "$status"')
    return "\n".join(lines) + "\n"


def _doctor_run_probe(target, script, *, timeout_message):
    import subprocess

    command = (
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            target.host,
            "bash",
            "-s",
        ]
        if target.host
        else ["bash", "-s"]
    )
    try:
        return subprocess.run(
            command,
            input=script,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            command,
            124,
            stdout="",
            stderr=timeout_message,
        )


def run_doctor(args):
    import importlib.util
    import os
    import shlex
    import shutil
    import subprocess
    import tempfile

    failures = []
    package_status = {}
    config = None
    project = None
    if args.project:
        from NepTrain.core.config import ConfigError, load_config

        project = Path(args.project).expanduser().resolve()
        try:
            config, _ = load_config(project)
        except ConfigError as error:
            raise SystemExit(
                f"NepTrain: error: invalid project configuration: {error}"
            ) from error

    training_backend = (
        args.training_backend
        or (str(config["training"]["backend"]) if config else "gpumd")
    )
    md_backend = (
        args.md_backend
        or (str(config["md"]["backend"]) if config else "gpumd")
    )
    selected_inference = (
        args.inference_backend
        or (
            str(config["md"].get("inference_backend", "auto"))
            if config
            else "cpu"
        )
    )
    print(
        "CHECK "
        f"training={training_backend} md={md_backend} "
        f"inference={selected_inference}"
    )

    for package in ("nep_adapters", "ase"):
        available = importlib.util.find_spec(package) is not None
        package_status[package] = available
        print(f"{'OK' if available else 'FAIL'} package {package}")
        if not available:
            failures.append(package)

    model_info = None
    if config is not None:
        from NepTrain.core.project_checks import check_project_inputs

        print("项目输入检查（FAIL 必须修复，WARN 为提示）：")
        for level, detail in check_project_inputs(config, project.parent):
            print(f"{level} {detail}")
            if level == "FAIL":
                failures.append(detail)
        from NepTrain.core.execution import ExecutionTarget

        checked_config = {
            **config,
            "training": {
                **config["training"],
                "backend": training_backend,
            },
            "md": {
                **config["md"],
                "backend": md_backend,
                "inference_backend": selected_inference,
            },
        }
        try:
            resource_contract = _doctor_resource_contract(config, project)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            resource_contract = None
            failures.append("labeling resource contract")
            print(f"FAIL 标注资源清单：{error}")
            print("  补齐资源清单中的元素、文件路径和真实 SHA256；参见对应后端示例。")
        labeling = config.get("labeling", {})
        labeling_backend = str(labeling.get("backend", "vasp"))
        if labeling_backend == "model":
            teacher = Path(str(labeling["model_path"])).expanduser()
            if not teacher.is_absolute():
                teacher = (project.parent / teacher).resolve()
            available = teacher.is_file()
            print(f"{'OK' if available else 'FAIL'} teacher model {teacher}")
            if not available:
                failures.append(f"teacher model {teacher}")
        resolved_targets = {}
        for name, raw_target in config["execution"]["targets"].items():
            value = dict(raw_target)
            raw_setup = value.get("setup_script")
            setup_path = None
            if raw_setup:
                setup_text = str(raw_setup)
                candidate = Path(setup_text).expanduser()
                local_candidate = (
                    (project.parent / candidate).resolve()
                    if not candidate.is_absolute()
                    else candidate
                )
                if local_candidate.is_file():
                    value["setup_script"] = str(local_candidate)
                    setup_path = local_candidate
                else:
                    setup_path = Path(setup_text)
            target = ExecutionTarget.from_mapping(str(name), value)
            resolved_targets[str(name)] = target
            try:
                tools, packages, roles = _doctor_target_requirements(
                    checked_config,
                    str(name),
                    target,
                    base_dir=project.parent,
                )
                probe = _doctor_target_probe(
                    target,
                    tools,
                    packages,
                    setup_path,
                )
                completed = _doctor_run_probe(
                    target,
                    probe,
                    timeout_message="probe timed out after 30s",
                )
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                failures.append(f"execution target {name}")
                print(f"FAIL execution target {name}: {error}")
                print("  检查该 target 的 setup_script、command、host 和标注输入；修复后重跑 doctor。")
                continue
            available = completed.returncode == 0
            location = target.host or "local"
            role_text = ",".join(roles) if roles else "unused"
            print(
                f"{'OK' if available else 'FAIL'} execution target {name} "
                f"({target.executor} on {location}; roles={role_text})"
            )
            if not available:
                failures.append(f"execution target {name}")
                detail = (completed.stderr or completed.stdout).strip()
                if detail:
                    print(f"  {detail}")
        if resource_contract is not None:
            target_name, resource_root, label, records = resource_contract
            target = resolved_targets[target_name]
            completed = _doctor_run_probe(
                target,
                _doctor_resource_probe(resource_root, records),
                timeout_message="resource probe timed out after 30s",
            )
            available = completed.returncode == 0
            location = target.host or "local"
            print(
                f"{'OK' if available else 'FAIL'} {label} resources "
                f"on {target_name} ({location})"
            )
            if not available:
                failures.append(f"{label} resources on {target_name}")
                detail = (completed.stderr or completed.stdout).strip()
                if detail:
                    print(f"  {detail}")
    else:
        if training_backend == "torchnep":
            available = importlib.util.find_spec("torchnep") is not None
            print(f"{'OK' if available else 'FAIL'} package torchnep")
            if not available:
                failures.append("torchnep")
        else:
            for tool in _doctor_command_tools(
                os.environ.get("NEPTRAIN_NEP_COMMAND", "nep")
            ):
                available = shutil.which(tool) is not None
                print(f"{'OK' if available else 'FAIL'} GPUMD trainer {tool}")
                if not available:
                    failures.append(tool)
        if md_backend == "gpumd":
            for tool in _doctor_command_tools(
                os.environ.get("NEPTRAIN_GPUMD_COMMAND", "gpumd")
            ):
                available = shutil.which(tool) is not None
                print(f"{'OK' if available else 'FAIL'} GPUMD MD {tool}")
                if not available:
                    failures.append(tool)

    if args.model and package_status["nep_adapters"]:
        from nep_adapters import inspect_model
        from NepTrain.core.nep.calculator import resolve_backend

        model_info = inspect_model(args.model)
        selected_inference = resolve_backend(args.model, selected_inference)
        print(
            f"OK model type={model_info.model_type} elements={','.join(model_info.elements)} "
            f"backend={selected_inference}"
        )
    if config is None and md_backend == "lammps":
        lmp_tokens = shlex.split(args.lmp)
        executable = shutil.which(lmp_tokens[0]) if lmp_tokens else None
        print(f"{'OK' if executable else 'FAIL'} LAMMPS executable {args.lmp}")
        if not executable:
            failures.append("lammps")
        else:
            probe_command = [*lmp_tokens, "-h"]
            try:
                completed = subprocess.run(
                    probe_command,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                completed = subprocess.CompletedProcess(
                    probe_command,
                    124,
                    stdout="",
                    stderr="LAMMPS help probe timed out after 30s",
                )
            help_text = completed.stdout + completed.stderr
            pair = "nep/gpu/kk" if selected_inference == "cuda" else "nep/cpu"
            pair_available = pair in help_text
            print(f"{'OK' if pair_available else 'FAIL'} LAMMPS pair style {pair}")
            if not pair_available:
                failures.append(pair)
    if (
        config is None
        and args.structure
        and args.model
        and md_backend == "lammps"
        and not failures
    ):
        from ase.io import read as ase_read
        from NepTrain.core.md import MdRequest, run_md

        atoms = ase_read(args.structure, index=0)
        spin = bool(model_info and model_info.supports("spin"))
        with tempfile.TemporaryDirectory(prefix="neptrain-doctor-") as directory:
            root = Path(directory)
            run_md(
                MdRequest(
                    atoms=atoms,
                    model_file=Path(args.model),
                    output_dir=root / "run",
                    output_file=root / "trajectory.xyz",
                    temperature=300,
                    spin_temperature=300 if spin else None,
                    steps=1,
                    timestep=0.0001,
                    spin=spin,
                    inference_backend=selected_inference,
                    lmp_command=args.lmp,
                    mpiexec=args.mpiexec,
                    mpi_ranks=args.mpi_ranks,
                ),
                "lammps",
            )
        print(f"OK real LAMMPS smoke at mpi_ranks={args.mpi_ranks}")
    if config is not None:
        feishu = config.get("notifications", {}).get("feishu", {})
        if feishu:
            from NepTrain.core.notifications import doctor_probe

            workflow_id = str(
                config.get("workflow", {}).get("id") or project.stem
            )
            result = doctor_probe(
                feishu,
                workflow_id=workflow_id,
                project_path=project,
            )
            print(
                f"{'OK' if result.ok else 'FAIL'} Feishu webhook "
                "signature and delivery"
            )
            if not result.ok:
                failures.append("Feishu webhook")
                print(f"  {result.detail}")
    if failures:
        print(f"检查结束：{len(failures)} 项必须修复；按上面的路径和提示处理后重跑 doctor。")
        raise SystemExit("Doctor failed: " + ", ".join(failures))
    print("Doctor completed successfully.")

def build_perturb(subparsers):
    parser = subparsers.add_parser(
        "perturb",
        help="Generate perturbed structures.",
    )
    parser.set_defaults(func=run_perturb)
    parser.add_argument(
        "model_path",
        help="Structure file or directory containing XYZ/VASP structures.",
    )
    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default=20,
        help="Number of perturbations generated per input structure.",
    )
    parser.add_argument(
        "--cell-perturbation",
        "--cell",
        "-c",
        dest="cell_pert_fraction",
        type=float,
        default=0.03,
        help="Maximum cell deformation fraction, default 0.03.",
    )
    parser.add_argument(
        "--max-displacement",
        "--distance",
        "-d",
        type=float,
        dest="max_displacement",
        default=0.1,
        help="Maximum Cartesian displacement amplitude in Å, default 0.1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible perturbations, default 42.",
    )
    parser.add_argument(
        "--out",
        "-o",
        dest="out_file_path",
        default="./perturb.xyz",
        help="Output extxyz path, default ./perturb.xyz.",
    )
    parser.add_argument(
        "--append",
        "-a",
        action="store_true",
        help="Append to an existing output file.",
    )

def build_doctor(subparsers):
    parser = subparsers.add_parser("doctor", help="Check selected runtime capabilities.")
    parser.set_defaults(func=run_doctor)
    parser.add_argument(
        "--training-backend",
        choices=["gpumd", "torchnep"],
        help="Override project training.backend; default gpumd without --project.",
    )
    parser.add_argument(
        "--md-backend",
        choices=["gpumd", "lammps"],
        help="Override project md.backend; default gpumd without --project.",
    )
    parser.add_argument(
        "--inference-backend",
        choices=["auto", "cpu", "cuda"],
        help=(
            "Override project md.inference_backend; default cpu without "
            "--project."
        ),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--structure", default=None)
    parser.add_argument("--lmp", default="lmp")
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument(
        "--project",
        default=None,
        help="Check every execution target in a project YAML.",
    )

def build_smoke(subparsers):
    parser = subparsers.add_parser(
        "smoke",
        help="Run the deterministic Toy Teacher workflow-development smoke.",
    )
    parser.set_defaults(func=run_smoke_command)
    parser.add_argument(
        "--profile", choices=["ordinary", "spin", "recovery"], default="ordinary"
    )
    parser.add_argument("--output", default="./outputs/smoke")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max-selected", type=int, default=8)
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Also run a resumable progressive Toy workflow for this many generations.",
    )
    parser.add_argument(
        "--workflow",
        action="store_true",
        help="Also run the fixed active-learning decision/recovery suite with deterministic backend doubles (no real MD or DFT).",
    )
    parser.add_argument("--force", action="store_true")


def build_select(subparsers):
    parser_select = subparsers.add_parser(
        "select",
        help="Select structures manually with the production FPS policy.",
    )
    parser_select.set_defaults(func=run_select)

    parser_select.add_argument(
        "trajectory_paths",
        nargs="+",
        help="Candidate extxyz trajectory files.",
    )
    parser_select.add_argument(
        "--base",
        "-base",
        help="Optional reference extxyz dataset used to warm-start FPS.",
    )
    parser_select.add_argument(
        "--nep",
        "-nep",
        help="NEP model used for descriptors; without it SOAP is used.",
    )
    parser_select.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="NEPAdapters descriptor backend, default auto.",
    )
    parser_select.add_argument(
        "--descriptor-reduction",
        choices=("global_mean", "elementwise_mean_std"),
        default="global_mean",
        help=(
            "Structure descriptor reduction: historical global_mean or "
            "element-preserving elementwise_mean_std."
        ),
    )
    parser_select.add_argument(
        "--max-selected",
        "-max",
        type=int,
        default=20,
        help="Maximum number of selected structures, default 20.",
    )
    parser_select.add_argument(
        "--min-novelty",
        "--min-distance",
        "--min_distance",
        "-d",
        type=float,
        dest="min_novelty",
        default=0.01,
        help="Strict normalized descriptor novelty threshold, default 0.01.",
    )
    parser_select.add_argument(
        "--filter",
        "-f",
        type=float,
        const=0.6,
        nargs="?",
        default=False,
        help="Reject short bonds using this covalent-radius coefficient.",
    )
    parser_select.add_argument(
        "--rejected-out",
        help="Optional extxyz output for structures rejected by --filter.",
    )
    parser_select.add_argument(
        "--out",
        "-o",
        dest="out_file_path",
        default="./selected.xyz",
        help="Selected extxyz output, default ./selected.xyz.",
    )
    parser_select.add_argument(
        "--report",
        help="Selection JSON report; defaults beside --out.",
    )

    group = parser_select.add_argument_group("SOAP parameters")
    group.add_argument("--r-cut", "--r_cut", "-r", type=float, default=6.0)
    group.add_argument("--n-max", "--n_max", "-n", type=int, default=8)
    group.add_argument("--l-max", "--l_max", "-l", type=int, default=6)

def _print_manual_status(value, *, json_output=False):
    if json_output:
        _print_json(value)
        return
    kind = value.get("kind")
    operation_id = value.get("operation_id")
    if kind or operation_id:
        task = str(kind or "task")
        if operation_id:
            task += f" ({operation_id})"
        print(f"Task: {task}")
    labels = (
        ("state", "State"),
        ("scheduler_state", "Scheduler"),
        ("job_id", "Job"),
        ("run_directory", "Run directory"),
        ("remote_directory", "Remote directory"),
        ("result", "Result"),
        ("reason", "Reason"),
    )
    for key, label in labels:
        item = value.get(key)
        if item not in (None, "", [], {}):
            print(f"{label}: {item}")
    completed = value.get("completed")
    total = value.get("total")
    if completed is not None and total is not None:
        print(f"Progress: {completed}/{total}")
    errors = value.get("errors")
    if errors:
        print("Errors:")
        for error in errors:
            if isinstance(error, dict):
                index = error.get("index")
                prefix = "collection" if index is None else f"job {index}"
                print(f"- {prefix}: {error.get('error', '')}")
            else:
                print(f"- {error}")
    logs = value.get("logs")
    if logs:
        print("Logs:")
        for log in logs:
            print(f"- {log}")
    run_directory = value.get("run_directory")
    state = str(value.get("state", "")).lower()
    next_action = value.get("next_action")
    if next_action:
        print(f"Next: {next_action}")
    elif run_directory and state in {"prepared", "submitted", "running", "unknown"}:
        print(f"Next: neptrain task wait {shlex.quote(str(run_directory))}")


def _manual_project(project):
    if not project:
        return {}, Path.cwd()
    from NepTrain.core.config import load_config

    path = Path(project).expanduser().resolve()
    config, _ = load_config(path)
    return config, path.parent


def _project_path(base, value):
    if value is None:
        return None
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def _manual_sampling_route(project, base, route_id):
    from NepTrain.core.manual import ManualTaskError

    if not project:
        if route_id:
            raise ManualTaskError("--route requires --project")
        return None
    routes = list(project.get("sampling", {}).get("routes") or [])
    if route_id:
        matches = [route for route in routes if route.get("id") == route_id]
        if not matches:
            available = ", ".join(str(route.get("id")) for route in routes)
            raise ManualTaskError(
                f"unknown sampling route {route_id!r}; available routes: {available}"
            )
        selected = dict(matches[0])
    elif len(routes) == 1:
        selected = dict(routes[0])
    else:
        raise ManualTaskError(
            "project defines multiple sampling routes; select one with --route"
        )
    selected["template_path"] = _project_path(base, selected["template_path"])
    return selected


def run_manual_train_command(args):
    from NepTrain.core.manual import (
        prepare_training,
        submit_operation,
        target_from_project,
    )

    project, base = _manual_project(args.project)
    settings = project.get("training", {})
    target = target_from_project(args.project, args.target, route="training")
    operation = prepare_training(
        args.input,
        backend=args.backend or settings.get("backend", "torchnep"),
        config_file=args.config
        or _project_path(base, settings.get("config_path"))
        or "./nep.in",
        output=args.output,
        workdir=args.workdir,
        target=target,
        test_file=args.test or _project_path(base, settings.get("test_path")),
        restart_file=args.restart,
        device=args.device or settings.get("device", "cuda"),
        torch_backend=args.torch_backend
        or settings.get("torch_backend", "auto"),
        precision=args.precision or settings.get("precision", "float32"),
        use_compile=(
            args.use_compile
            if args.use_compile is not None
            else bool(settings.get("use_compile", False))
        ),
        seed=args.seed if args.seed is not None else int(settings.get("seed", 20260723)),
        force=args.force,
    )
    _print_manual_status(
        submit_operation(
            operation, wait=args.wait, poll_interval=args.poll_interval
        ),
        json_output=args.json,
    )


def run_manual_md_command(args):
    from NepTrain.core.manual import (
        prepare_md,
        submit_operation,
        target_from_project,
    )

    project, base = _manual_project(args.project)
    settings = project.get("md", {})
    sampling = project.get("sampling", {})
    candidate_pool = sampling.get("candidate_pool", {})
    route = _manual_sampling_route(project, base, args.route)
    route_id = None if route is None else str(route["id"])
    target = target_from_project(
        args.project,
        args.target,
        route="sampling",
        sampling_route_id=route_id,
    )
    conditions = dict((route or {}).get("conditions") or {})
    from NepTrain.core.sampling_route import normalized_progression

    progression = normalized_progression((route or {}).get("progression"))
    operation = prepare_md(
        args.input,
        backend=args.backend or settings.get("backend", "lammps"),
        model_file=args.model,
        temperatures=(
            args.temperature
            if args.temperature is not None
            else conditions.get("temperature_path", [300.0])
        ),
        output=args.output,
        workdir=args.workdir,
        target=target,
        steps=(
            args.steps
            if args.steps is not None
            else progression["steps"][args.maturity]
        ),
        seed=(
            getattr(args, "seed", None)
            if getattr(args, "seed", None) is not None
            else int(project.get("workflow", {}).get("seed", 12345))
        ),
        pressure=(
            args.pressure
            if args.pressure is not None
            else float(conditions.get("pressure", 0.0))
        ),
        ensemble=args.ensemble or "nvt",
        template_path=args.template or (route or {}).get("template_path"),
        spin=args.spin
        if args.spin is not None
        else bool(settings.get("spin", False)),
        spin_temperature=args.spin_temperature
        if args.spin_temperature is not None
        else None,
        inference_backend=args.inference_backend
        or settings.get("inference_backend", "auto"),
        lmp=args.lmp or "lmp",
        mpiexec=args.mpiexec or "mpirun",
        mpi_ranks=args.mpi_ranks
        if args.mpi_ranks is not None
        else 1,
        pre_failure_frames=(
            args.pre_failure_frames
            if args.pre_failure_frames is not None
            else int(candidate_pool.get("pre_failure_frames", 2))
        ),
        bad_tail_frames=(
            args.bad_tail_frames
            if args.bad_tail_frames is not None
            else int(candidate_pool.get("bad_tail_frames", 1))
        ),
        health=dict(candidate_pool.get("health") or {}),
        max_concurrent=(
            args.max_concurrent
            if args.max_concurrent is not None
            else DEFAULT_MAX_CONCURRENT
        ),
        force=args.force,
    )
    _print_manual_status(
        submit_operation(
            operation, wait=args.wait, poll_interval=args.poll_interval
        ),
        json_output=args.json,
    )


def run_manual_label_command(args):
    from NepTrain.core.manual import (
        prepare_labeling,
        submit_operation,
        target_from_project,
    )

    project, base = _manual_project(args.project)
    settings = project.get("labeling", {})
    target = target_from_project(args.project, args.target, route="labeling")
    if args.resources is not None:
        resource_dir = args.resources
    elif target.labeling_resource_path:
        resource_dir = None
    else:
        resource_dir = _project_path(base, settings.get("resource_path"))
    if args.kspacing is not None:
        kpoint_mode = "kspacing"
        kspacing = args.kspacing
    elif args.ka is not None:
        kpoint_mode = "kpoints"
        kspacing = None
    else:
        kpoint_mode = settings.get("kpoint_mode", "auto")
        kspacing = (
            settings.get("kspacing")
            if kpoint_mode == "kspacing"
            else None
        )
    raw_ka = args.ka if args.ka is not None else settings.get("kpoints", [1, 1, 1])
    if isinstance(raw_ka, int):
        ka = [raw_ka, raw_ka, raw_ka]
    else:
        ka = list(raw_ka)
        if len(ka) == 1:
            ka *= 3
    backend = args.backend or settings.get("backend", "vasp")
    operation = prepare_labeling(
        args.input,
        backend=backend,
        output=args.output,
        workdir=args.workdir,
        target=target,
        input_file=args.dft_input
        or _project_path(base, settings.get("input_path")),
        resource_dir=resource_dir,
        resource_manifest=(
            (
                args.potcar_manifest
                or _project_path(base, settings.get("potcar_manifest_path"))
            )
            if backend == "vasp"
            else (
                args.resource_manifest
                or _project_path(base, settings.get("resource_manifest_path"))
            )
        ),
        n_cpu=args.cpus,
        use_gamma=(
            args.gamma
            if args.gamma is not None
            else bool(settings.get("gamma_centered", False))
        ),
        kpoint_mode=kpoint_mode,
        kspacing=kspacing,
        ka=ka,
        structures_per_job=(
            args.structures_per_job
            if args.structures_per_job is not None
            else int(
                settings.get(
                    "structures_per_job",
                    DEFAULT_STRUCTURES_PER_MODEL_JOB
                    if backend == "model"
                    else DEFAULT_STRUCTURES_PER_LABEL_JOB,
                )
            )
        ),
        max_concurrent=(
            args.max_concurrent
            if args.max_concurrent is not None
            else int(
                settings.get("max_concurrent", DEFAULT_MAX_CONCURRENT)
            )
        ),
        teacher_profile=args.teacher_profile or "ordinary",
        model_file=args.model
        or _project_path(base, settings.get("model_path")),
        model_name=args.model_name or settings.get("model_name"),
        runner=args.runner or settings.get("runner"),
        device=args.device or settings.get("device", "cuda"),
        precision=args.precision
        or settings.get("precision", "float32"),
        force=args.force,
    )
    _print_manual_status(
        submit_operation(
            operation, wait=args.wait, poll_interval=args.poll_interval
        ),
        json_output=args.json,
    )


def run_task_command(args):
    from NepTrain.core.manual import (
        cancel_operation,
        load_operation,
        operation_logs,
        refresh_operation,
        retry_failed,
        wait_operation,
    )

    operation = load_operation(args.run)
    if args.task_action == "status":
        value = refresh_operation(operation)
    elif args.task_action == "wait":
        value = wait_operation(operation, poll_interval=args.poll_interval)
    elif args.task_action == "retry":
        value = retry_failed(operation)
    elif args.task_action == "cancel":
        value = cancel_operation(operation)
    elif args.task_action == "logs":
        value = {
            "protocol": "neptrain.manual-logs.v1",
            "operation_id": operation.operation_id,
            "kind": operation.kind,
            "run_directory": str(operation.root),
            "logs": operation_logs(operation),
        }
    else:  # pragma: no cover - argparse owns this invariant
        raise ValueError(args.task_action)
    _print_manual_status(value, json_output=args.json)


def run_manual_worker_command(args):
    from NepTrain.core.manual import run_manual_worker

    return run_manual_worker(args.run, args.index)


def run_model_worker_command(args):
    from NepTrain.core.labeling.interface import LabelingError

    try:
        if args.adapter == "mace":
            from NepTrain.runners.mace import label_frames

            label_frames(
                args.model,
                args.input,
                args.output,
                device=args.device,
                precision=args.precision,
            )
        elif args.adapter == "deepmd":
            from NepTrain.runners.deepmd import label_frames

            label_frames(
                args.model,
                args.input,
                args.output,
                device=args.device,
                precision=args.precision,
                head=args.head,
            )
        else:
            from NepTrain.runners.tace import label_frames

            label_frames(
                args.model,
                args.input,
                args.output,
                device=args.device,
                precision=args.precision,
                fidelity_index=args.fidelity_index,
            )
    except (OSError, RuntimeError, ValueError) as error:
        raise LabelingError(str(error)) from error
    return 0


def run_spin_migration_command(args):
    from NepTrain.core.spin import migrate_spin_dataset

    result = migrate_spin_dataset(
        args.input,
        args.output,
        force=args.force,
    )
    if args.json:
        _print_json(result)
    else:
        print(
            f"Migrated {result['spin_frames']}/{result['frames']} spin frames "
            f"to {result['output']}"
        )
        print(f"Removed legacy fields: {result['legacy_fields_removed']}")


def _add_execution_options(parser):
    parser.add_argument(
        "--project",
        help="Schema-v8 project providing reusable execution targets.",
    )
    parser.add_argument("--target", help="Execution target name from project.yaml.")
    parser.add_argument("--workdir", help="Durable run directory.")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for submitted Slurm work and publish the final result.",
    )
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the machine-readable task record.",
    )


def _parse_ka(value):
    values = [item.strip() for item in str(value).split(",")]
    if len(values) == 1:
        values *= 3
    if len(values) != 3 or any(not item.isdigit() for item in values):
        raise argparse.ArgumentTypeError("--ka must be one integer or x,y,z")
    return [int(item) for item in values]


def build_manual_train(subparsers):
    parser = subparsers.add_parser(
        "train", help="Train one NEP model locally or on a configured target."
    )
    parser.set_defaults(func=run_manual_train_command)
    parser.add_argument("input", help="Labeled training extxyz.")
    parser.add_argument("--backend", choices=["gpumd", "torchnep"])
    parser.add_argument("--config")
    parser.add_argument("--test")
    parser.add_argument("--restart")
    parser.add_argument("--device")
    parser.add_argument("--torch-backend", choices=["auto", "loop", "bmm"])
    parser.add_argument("--precision", choices=["float32", "float64"])
    parser.add_argument(
        "--compile",
        dest="use_compile",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", "-o", default="./nep.txt")
    parser.add_argument("--force", action="store_true")
    _add_execution_options(parser)


def build_manual_md(subparsers):
    parser = subparsers.add_parser(
        "md", help="Run GPUMD or LAMMPS over structures and temperatures."
    )
    parser.set_defaults(func=run_manual_md_command)
    parser.add_argument("input", help="Structure file, extxyz, or directory.")
    parser.add_argument("--backend", choices=["gpumd", "lammps"])
    parser.add_argument("--model", default="./nep.txt")
    parser.add_argument("--temperature", type=float, nargs="+")
    parser.add_argument("--pressure", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ensemble", choices=["nve", "nvt", "npt"])
    parser.add_argument("--template")
    parser.add_argument(
        "--route",
        help=(
            "Sampling route whose template, conditions, progression, and "
            "route-specific target provide defaults."
        ),
    )
    parser.add_argument(
        "--maturity",
        choices=[
            "smoke_passed",
            "short_stable",
            "long_stable",
            "production_ready",
        ],
        default="smoke_passed",
        help="Route progression level used when --steps is omitted.",
    )
    parser.add_argument(
        "--spin", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--spin-temperature", type=float)
    parser.add_argument("--inference-backend", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lmp")
    parser.add_argument("--mpiexec")
    parser.add_argument("--mpi-ranks", type=int)
    parser.add_argument("--pre-failure-frames", type=int)
    parser.add_argument("--bad-tail-frames", type=int)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--output", "-o", default="./trajectory.xyz")
    parser.add_argument("--force", action="store_true")
    _add_execution_options(parser)


def build_manual_label(subparsers):
    parser = subparsers.add_parser(
        "label",
        aliases=["dft"],
        help=(
            "Label structures with VASP, ABACUS, a teacher model, or the "
            "development Adapter."
        ),
    )
    parser.set_defaults(func=run_manual_label_command)
    parser.add_argument("input", help="Structure file, extxyz, or directory.")
    parser.add_argument(
        "--backend",
        choices=["vasp", "abacus", "model", "toy"],
    )
    parser.add_argument("--teacher-profile", choices=["ordinary", "spin"])
    parser.add_argument("--input-file", dest="dft_input")
    parser.add_argument("--resources")
    parser.add_argument(
        "--potcar-manifest",
        help=(
            "Local JSON manifest pinning each VASP POTCAR path, SHA256, "
            "TITEL, family, and release."
        ),
    )
    parser.add_argument(
        "--resource-manifest",
        help=(
            "Local JSON manifest pinning ABACUS pseudopotential and orbital "
            "paths and SHA256 values."
        ),
    )
    parser.add_argument("--cpus", type=int)
    parser.add_argument(
        "--gamma", action=argparse.BooleanOptionalAction, default=None
    )
    kpoints = parser.add_mutually_exclusive_group()
    kpoints.add_argument("--kspacing", type=float)
    kpoints.add_argument("--ka", type=_parse_ka)
    parser.add_argument("--structures-per-job", type=int)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--model", help="Fine-tuned teacher model file.")
    parser.add_argument(
        "--model-name",
        help="Stable teacher family/name recorded in label provenance.",
    )
    parser.add_argument(
        "--runner",
        help=(
            "Installed runner command implementing the NepTrain model-label "
            "protocol."
        ),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--precision", choices=["float32", "float64"])
    parser.add_argument("--output", "-o", default="./labeled.xyz")
    parser.add_argument("--force", action="store_true")
    _add_execution_options(parser)


def build_task_commands(subparsers):
    parser = subparsers.add_parser(
        "task", help="Inspect and control a detached manual step."
    )
    actions = parser.add_subparsers(dest="task_action", required=True)
    action_help = {
        "status": "Refresh scheduler state and collect completed results.",
        "logs": "List scheduler log files for the run.",
        "retry": "Resubmit only failed Slurm array elements.",
        "cancel": "Cancel the active Slurm job for the run.",
    }
    for name, help_text in action_help.items():
        command = actions.add_parser(name, help=help_text)
        command.set_defaults(func=run_task_command)
        command.add_argument("run", help="Manual run directory.")
        command.add_argument("--json", action="store_true")
    wait = actions.add_parser(
        "wait", help="Wait until the run completes, fails, or is cancelled."
    )
    wait.set_defaults(func=run_task_command)
    wait.add_argument("run", help="Manual run directory.")
    wait.add_argument("--poll-interval", type=float, default=10.0)
    wait.add_argument("--json", action="store_true")


def build_data_commands(subparsers):
    parser = subparsers.add_parser(
        "data",
        help="Validate or explicitly migrate scientific dataset contracts.",
    )
    actions = parser.add_subparsers(dest="data_action", required=True)
    migrate = actions.add_parser(
        "migrate-spin",
        help="Rewrite legacy spins/mforces aliases to spin/mforce atomically.",
    )
    migrate.set_defaults(func=run_spin_migration_command)
    migrate.add_argument("input")
    migrate.add_argument("output")
    migrate.add_argument("--force", action="store_true")
    migrate.add_argument("--json", action="store_true")


def build_workflow_commands(subparsers):
    parser = subparsers.add_parser(
        "workflow", help="Prepare and control an automated active-learning workflow."
    )
    actions = parser.add_subparsers(dest="workflow_action", required=True)
    init = actions.add_parser("init", help="Create a strict schema-v8 project.")
    init.set_defaults(func=init_template)
    init.add_argument("--profile", choices=["local", "slurm"], default="slurm")
    init.add_argument("--ensemble", choices=["npt", "nvt"], default="npt")
    init.add_argument("--spin", action="store_true")
    init.add_argument(
        "--dft-backend",
        choices=["vasp", "abacus"],
        default="vasp",
    )
    init.add_argument("--directory", default=".")
    init.add_argument("--force", action="store_true")

    run = actions.add_parser(
        "run",
        help=(
            "Create a workflow from project YAML, or start an existing "
            "prepared workflow directory."
        ),
    )
    run.set_defaults(func=run_project_command)
    run.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    run.add_argument("project", help="Project YAML or prepared workflow directory.")
    run.add_argument("--initial-training")
    run.add_argument("--output")
    run.add_argument("--workflow-id")
    run.add_argument("--prepare-only", action="store_true")
    run.add_argument("--foreground", action="store_true")
    run.add_argument("--poll-interval", type=float)

    status = actions.add_parser(
        "status", help="Show scientific progress and current execution state."
    )
    status.set_defaults(func=run_status_command)
    status.add_argument("project")
    status.add_argument("--json", action="store_true")
    status.add_argument(
        "--details",
        action="store_true",
        help="Show all evaluated generations and optional test diagnostics.",
    )
    status.add_argument(
        "--jobs",
        action="store_true",
        help="Show execution tasks grouped by generation, stage, and attempt.",
    )

    resume = actions.add_parser(
        "resume", help="Start or restart an existing workflow controller."
    )
    resume.set_defaults(func=run_resume_command)
    resume.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    resume.add_argument("project")

    restart = actions.add_parser(
        "restart",
        help="Restart the latest unfinished generation from an explicit stage.",
    )
    restart.set_defaults(func=run_restart_command)
    restart.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    restart.add_argument("project")
    restart.add_argument("--generation", type=int, required=True)
    restart.add_argument(
        "--from",
        dest="from_stage",
        required=True,
        help="First stage to execute again.",
    )
    restart.add_argument(
        "--tasks",
        choices=["failed", "all"],
        default="failed",
        help=(
            "For a retained task group, retry only failed tasks (default) "
            "or rebuild every task."
        ),
    )
    restart.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the restart plan without changing workflow state.",
    )
    restart.add_argument("--foreground", action="store_true")
    restart.add_argument("--poll-interval", type=float)

    extend = actions.add_parser(
        "extend", help="Increase the maximum model-generation budget."
    )
    extend.set_defaults(func=run_extend_command)
    extend.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    extend.add_argument("project")
    extend.add_argument("generations", type=int, metavar="TOTAL_GENERATIONS",
                        help="New total sampling-generation budget, not the number to add (e.g. 10 -> 15 adds 5).")

    stop = actions.add_parser(
        "stop", help="Stop the controller and cancel its current compute jobs."
    )
    stop.set_defaults(func=run_stop_command)
    stop.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    stop.add_argument("project")
    cancellation = stop.add_mutually_exclusive_group()
    cancellation.set_defaults(cancel_jobs=True)
    cancellation.add_argument(
        "--keep-jobs",
        dest="cancel_jobs",
        action="store_false",
        help="Stop only the controller and keep current compute jobs running.",
    )


def build_internal_commands(subparsers):
    controller = subparsers.add_parser("controller", help=argparse.SUPPRESS)
    subparsers._choices_actions.pop()
    controller.set_defaults(func=run_controller_command)
    controller.add_argument("project")
    controller.add_argument("--poll-interval", type=float)

    worker = subparsers.add_parser("stage-worker", help=argparse.SUPPRESS)
    subparsers._choices_actions.pop()
    worker.set_defaults(func=run_stage_worker_command)
    worker.add_argument("bundle")

    verifier = subparsers.add_parser("stage-verify", help=argparse.SUPPRESS)
    subparsers._choices_actions.pop()
    verifier.set_defaults(func=run_stage_verify_command)
    verifier.add_argument("bundle")

    verifier_many = subparsers.add_parser(
        "stage-verify-many",
        help=argparse.SUPPRESS,
    )
    subparsers._choices_actions.pop()
    verifier_many.set_defaults(func=run_stage_verify_many_command)
    verifier_many.add_argument("bundles", nargs="+")

    manual = subparsers.add_parser("manual-worker", help=argparse.SUPPRESS)
    subparsers._choices_actions.pop()
    manual.set_defaults(func=run_manual_worker_command)
    manual.add_argument("run")
    manual.add_argument("index", type=int)

    model = subparsers.add_parser("model-worker", help=argparse.SUPPRESS)
    subparsers._choices_actions.pop()
    model.set_defaults(func=run_model_worker_command)
    model.add_argument("adapter", choices=["mace", "deepmd", "tace"])
    model.add_argument("--head")
    model.add_argument("--fidelity-index", type=int)
    model.add_argument("--model", required=True)
    model.add_argument("--input", required=True)
    model.add_argument("--output", required=True)
    model.add_argument("--device", choices=["cpu", "cuda"], required=True)
    model.add_argument(
        "--precision",
        choices=["float32", "float64"],
        required=True,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="neptrain",
        description=(
            "Run individual NEP training, MD, labeling and sampling steps, "
            "or compose the same steps into an automated workflow."
        ),
    )
    parser.add_argument("-v", "--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar=(
            "{train,md,label,select,perturb,workflow,task,data,doctor,smoke}"
        ),
    )
    build_manual_train(subparsers)
    build_manual_md(subparsers)
    build_manual_label(subparsers)
    build_select(subparsers)
    build_perturb(subparsers)
    build_workflow_commands(subparsers)
    build_task_commands(subparsers)
    build_data_commands(subparsers)
    build_doctor(subparsers)
    build_smoke(subparsers)
    build_internal_commands(subparsers)
    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:
        pass
    args = parser.parse_args()
    try:
        result = args.func(args)
        return result if type(result) is int else None
    except Exception as error:
        from NepTrain.core.workflow import WorkflowError
        from NepTrain.core.config import ConfigError
        from NepTrain.core.controller import ControllerError
        from NepTrain.core.execution import ExecutionError
        from NepTrain.core.iteration import IterationError
        from NepTrain.core.manual import ManualTaskError

        lightweight_errors = (
            WorkflowError,
            ConfigError,
            ControllerError,
            ExecutionError,
            IterationError,
            ManualTaskError,
            OSError,
        )
        scientific_error_names = {
            "LabelingError",
            "MdError",
            "SmokeError",
            "SpinDataError",
            "SelectionError",
            "TrainingError",
            "WorkflowIterationError",
        }
        is_scientific_error = (
            type(error).__module__.startswith("NepTrain.core.")
            and type(error).__name__ in scientific_error_names
        )
        if isinstance(error, lightweight_errors) or is_scientific_error:
            parser.exit(2, f"NepTrain: error: {error}\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
