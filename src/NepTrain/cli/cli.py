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
    "explore": "采样",
    "select": "选样",
    "label": "标注",
    "diagnose": "诊断",
    "merge": "合并训练集",
    "retrain": "重新训练",
    "evaluate": "评估",
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


def _sampling_cell(cell):
    temperature = f"{float(cell['temperature']):g} K"
    state = cell["state"]
    if state == "complete":
        return f"{temperature} ✓"
    if state == "pending":
        return f"{temperature} ○"
    marker = "●" if state == "active" else "⚠"
    current = cell.get("current_ps")
    target = cell.get("target_ps")
    progress = (
        f" {_ps(current)}/{_ps(target)} ps"
        if current is not None and target is not None
        else "（ps 暂不可读）"
    )
    count = (
        f"（{cell['completed']}/{cell['total']} 条轨迹完成）"
        if cell.get("total")
        else ""
    )
    return f"{temperature} {marker}{progress}{count}"


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


def _r2_cell(value):
    return "-" if value is None else f"{float(value):.4f}"


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
        return f"{_STAGE_LABELS.get(status.stage, status.stage)}中"
    return "进行中"


def _print_precision(status):
    print()
    if status.precision_basis not in {"validation", "acquisition"}:
        print("精度变化：暂无可比较数据（未配置独立验证集）")
        return
    acquisition = status.precision_basis == "acquisition"
    print("新增 DFT 预测精度（训练前）：" if acquisition else "验证集精度：")
    rows = [
        (
            "代",
            "状态",
            "E/meV·atom⁻¹",
            "F/meV·Å⁻¹",
            "V/meV·atom⁻¹",
            "M/meV/μB",
            "验收",
        )
    ]
    previous = {
        "energy_rmse": None,
        "force_rmse": None,
        "virial_rmse": None,
        "mforce_rmse": None,
    }
    for generation in status.generations:
        metrics = generation["quality"][
            "acquisition_rmse" if acquisition else "validation_rmse"
        ]
        accepted = generation["quality"].get(
            "acquisition_accepted" if acquisition else "accepted"
        )
        rows.append(
            (
                f"G{generation['generation']}",
                _generation_state(generation, status),
                _metric_cell(
                    metrics["energy_rmse"],
                    previous["energy_rmse"],
                    scale=1000,
                ),
                _metric_cell(
                    metrics["force_rmse"],
                    previous["force_rmse"],
                    scale=1000,
                ),
                _metric_cell(
                    metrics["virial_rmse"],
                    previous["virial_rmse"],
                    scale=1000,
                ),
                _metric_cell(
                    metrics["mforce_rmse"],
                    previous["mforce_rmse"],
                    scale=1000,
                ),
                (
                    "通过"
                    if accepted is True
                    else "未通过"
                    if accepted is False
                    else "等待"
                ),
            )
        )
        for name, value in metrics.items():
            if value is not None:
                previous[name] = value
    _table(rows)
    if acquisition and any(
        value is not None
        for generation in status.generations
        for value in generation["quality"].get("acquisition_r2", {}).values()
    ):
        print()
        print("新增 DFT 相关性与决策：")
        r2_rows = [("代", "E R²", "F R²", "V R²", "M R²", "下一步")]
        for generation in status.generations:
            quality = generation["quality"]
            values = quality.get("acquisition_r2", {})
            disposition = quality.get("generation_disposition")
            r2_rows.append(
                (
                    f"G{generation['generation']}",
                    _r2_cell(values.get("energy_r2")),
                    _r2_cell(values.get("force_r2")),
                    _r2_cell(values.get("virial_r2")),
                    _r2_cell(values.get("mforce_r2")),
                    (
                        "最终训练"
                        if disposition == "finalize"
                        else "继续采样"
                        if disposition == "continue"
                        else "等待"
                    ),
                )
            )
        _table(r2_rows)


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
    for (generation, stage, attempt), batch in groups.items():
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
        print(
            f"  {prefix}{_STAGE_LABELS.get(str(stage), stage)} "
            f"{attempt}：{' | '.join(parts)} | Job {_compact_job_ids(batch)}"
        )


def _print_workflow_status(status, *, show_jobs: bool = True):
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
    print(f"更新：{_updated_text(status.updated_at)}")
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

    if status.sampling_routes:
        print()
        print("采样进度：")
        multiple = len(status.sampling_routes) > 1
        total_failed = 0
        for route in status.sampling_routes:
            prefix = f"{route['route_id']}：" if multiple else ""
            line = " → ".join(
                _sampling_cell(cell) for cell in route["temperatures"]
            )
            print(f"{prefix}{line}")
            total_failed += int(route.get("failed", 0))
        if total_failed:
            print(f"异常：{total_failed} 条采样轨迹失败，失败证据已保留")

    _print_precision(status)
    if show_jobs:
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
    if status.next_action and status.state not in {"running", "waiting", "degraded"}:
        print(f"下一步：{status.next_action}")


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
                if value:
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
    _print_json(payload)


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
        _print_workflow_status(status, show_jobs=args.jobs)


def run_resume_command(args):
    from NepTrain.core.workflow import WorkflowError, resume_workflow

    try:
        result = resume_workflow(args.project)
    except WorkflowError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    payload = _workflow_resume_payload(result)
    _print_json(payload)


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
    from NepTrain.core.workflow import WorkflowError, extend_workflow

    try:
        preparation = extend_workflow(args.project, args.generations)
    except WorkflowError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    _print_json(
        {
            "protocol": "neptrain.workflow-extend.v1",
            "workflow_id": preparation.workflow_id,
            "total_model_generations": len(preparation.plans),
            "project": str(preparation.output_dir),
        }
    )


def run_stop_command(args):
    from NepTrain.core.controller import ControllerError, stop_workflow

    try:
        result = stop_workflow(
            args.project, cancel_jobs=bool(args.cancel_jobs)
        )
    except ControllerError as error:
        raise SystemExit(f"NepTrain: error: {error}") from error
    _print_json(
        {
            "protocol": "neptrain.workflow-stop.v1",
            **result,
        }
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


def _doctor_target_requirements(config, target_name, target):
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
            add_command(
                environment.get(
                    "NEPTRAIN_VASP_COMMAND",
                    "mpirun -n 1 vasp_std",
                )
            )
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
            raise SystemExit(
                f"NepTrain: error: invalid labeling resource contract: {error}"
            ) from error
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
            tools, packages, roles = _doctor_target_requirements(
                checked_config,
                str(name),
                target,
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
        "--jobs",
        action="store_true",
        help="Show execution tasks grouped by generation, stage, and attempt.",
    )

    resume = actions.add_parser(
        "resume", help="Start or restart an existing workflow controller."
    )
    resume.set_defaults(func=run_resume_command)
    resume.add_argument("project")

    extend = actions.add_parser(
        "extend", help="Increase the maximum model-generation budget."
    )
    extend.set_defaults(func=run_extend_command)
    extend.add_argument("project")
    extend.add_argument("generations", type=int)

    stop = actions.add_parser(
        "stop", help="Stop the controller and cancel its current compute jobs."
    )
    stop.set_defaults(func=run_stop_command)
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
