"""User-facing filesystem interface for a NepTrain workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Mapping


_LAYOUT_VERSION = 2
_STAGE_DIRECTORIES = {
    "train": "training/bootstrap",
    "explore": "sampling/md",
    "select": "sampling/selection",
    "label": "labeling",
    "diagnose": "evaluation/acquisition",
    "merge": "dataset",
    "retrain": "training/retrain",
    "evaluate": "evaluation/post-train",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


@dataclass(frozen=True)
class WorkflowWorkspace:
    """Own every durable path in one workflow directory.

    Layout v2 keeps the machine state below ``.neptrain`` and exposes only
    inputs, generation evidence, logs, and accepted results at the project
    root. Legacy layouts are rejected instead of being silently migrated.
    """

    root: Path
    version: int = _LAYOUT_VERSION

    @classmethod
    def create(cls, root: str | Path) -> "WorkflowWorkspace":
        workspace = cls(Path(root).expanduser().resolve())
        if workspace.root.exists() and any(workspace.root.iterdir()):
            raise ValueError(
                f"workflow output directory is not empty: {workspace.root}"
            )
        workspace.root.mkdir(parents=True, exist_ok=True)
        for directory in (
            workspace.inputs_dir,
            workspace.results_dir,
            workspace.generations_dir,
            workspace.logs_dir,
            workspace.plans_dir,
            workspace.jobs_dir,
            workspace.tasks_dir,
            workspace.locks_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        _write_json(
            workspace.layout_file,
            {
                "layout": "neptrain.workflow",
                "version": _LAYOUT_VERSION,
                "user_directories": ["inputs", "results", "generations", "logs"],
                "internal_directory": ".neptrain",
            },
        )
        (workspace.root / "README.md").write_text(
            "# NepTrain workflow\n\n"
            "- `project.yaml`：本次运行采用的完整配置快照。\n"
            "- `inputs/`：进入 workflow 的输入快照。\n"
            "- `results/`：最新通过验收的模型、训练集和指标。\n"
            "- `generations/`：按代组织的采样、标注、训练和评价证据。\n"
            "- `logs/`：Controller 与执行后端日志。\n"
            "- `.neptrain/`：账本、计划、可移植任务和执行状态，通常无需编辑。\n"
            "\n常用命令：`neptrain workflow status .`、"
            "`neptrain workflow resume .`、`neptrain workflow stop .`。\n"
            "流程作废并需要取消当前计算任务时使用："
            "`neptrain workflow stop . --cancel-jobs`。\n",
            encoding="utf-8",
        )
        return workspace

    @classmethod
    def locate(cls, path: str | Path) -> "WorkflowWorkspace":
        candidate = Path(path).expanduser().resolve()
        if candidate.is_file():
            if candidate.name == "manifest.json" and candidate.parent.name == ".neptrain":
                candidate = candidate.parent.parent
        layout = candidate / ".neptrain" / "layout.json"
        if layout.is_file():
            value = json.loads(layout.read_text(encoding="utf-8"))
            if value.get("layout") != "neptrain.workflow" or int(
                value.get("version", 0)
            ) != _LAYOUT_VERSION:
                raise ValueError(f"unsupported workflow layout: {layout}")
            return cls(candidate)
        raise FileNotFoundError(f"NepTrain workflow does not exist: {candidate}")

    @property
    def internal_dir(self) -> Path:
        return self.root / ".neptrain"

    @property
    def layout_file(self) -> Path:
        return self.internal_dir / "layout.json"

    @property
    def manifest(self) -> Path:
        return self.internal_dir / "manifest.json"

    @property
    def project_file(self) -> Path:
        return self.root / "project.yaml"

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def generations_dir(self) -> Path:
        return self.root / "generations"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def plans_dir(self) -> Path:
        return self.internal_dir / "plans"

    @property
    def jobs_dir(self) -> Path:
        return self.internal_dir / "jobs"

    @property
    def tasks_dir(self) -> Path:
        return self.internal_dir / "tasks"

    @property
    def locks_dir(self) -> Path:
        return self.internal_dir / "locks"

    @property
    def manifest_lock(self) -> Path:
        return self.locks_dir / "manifest.lock"

    @property
    def ledger(self) -> Path:
        return self.internal_dir / "ledger.json"

    @property
    def ledger_lock(self) -> Path:
        return self.locks_dir / "ledger.lock"

    @property
    def controller_file(self) -> Path:
        return self.internal_dir / "controller.json"

    @property
    def controller_lock(self) -> Path:
        return self.locks_dir / "controller.lock"

    @property
    def controller_pid(self) -> Path:
        return self.internal_dir / "controller.pid"

    @property
    def controller_log(self) -> Path:
        return self.logs_dir / "controller.log"

    @property
    def controller_root(self) -> Path:
        return self.root

    def generation_dir(self, generation: int) -> Path:
        return self.generations_dir / f"{generation:04d}"

    def stage_dir(self, generation: int, stage: str) -> Path:
        try:
            relative = _STAGE_DIRECTORIES[stage]
        except KeyError as error:
            raise ValueError(f"unknown workflow stage: {stage}") from error
        return self.generation_dir(generation) / relative

    def snapshot_inputs(
        self, config: Mapping[str, Any], initial_training: Path
    ) -> tuple[dict[str, Any], Path]:
        """Create the portable user input view used by all generated jobs."""

        snapshot = json.loads(json.dumps(config))
        initial_snapshot = self.inputs_dir / "initial-train.xyz"
        _copy_file(initial_training, initial_snapshot)

        def copy_path(section: str, key: str, relative: str) -> None:
            value = snapshot.get(section, {}).get(key)
            if value in {None, "", "auto"}:
                return
            source = Path(value)
            if not source.is_file():
                return
            target = self.inputs_dir / f"{relative}{source.suffix}"
            _copy_file(source, target)
            snapshot[section][key] = str(target.relative_to(self.root))

        copy_path("training", "config_path", "training/nep")
        copy_path("training", "test_path", "validation/training-test")
        copy_path("md", "structures", "md/structures")
        copy_path("md", "template_path", "md/template")
        copy_path("dft", "input_path", "dft/input")
        copy_path("evaluation", "validation_path", "validation/validation")
        for name, profile in snapshot.get("execution", {}).get(
            "targets", {}
        ).items():
            value = profile.get("setup_script")
            if value in {None, ""}:
                continue
            source = Path(value)
            if not source.is_file():
                continue
            target = self.inputs_dir / "platform" / f"{name}.sh"
            _copy_file(source, target)
            profile["setup_script"] = str(target.relative_to(self.root))
        snapshot.setdefault("training", {})["initial_path"] = str(
            initial_snapshot.relative_to(self.root)
        )
        return snapshot, initial_snapshot

    def publish_generation(
        self, generation: int, generation_record: Mapping[str, Any]
    ) -> None:
        """Publish a stable human-facing summary and latest accepted results."""

        stages = generation_record.get("stages", {})
        summary = {
            "generation": generation,
            "accepted": generation_record.get("accepted"),
            "complete": bool(generation_record.get("complete")),
            "metrics": {
                stage: dict(record.get("metrics", {}))
                for stage, record in stages.items()
            },
        }
        accepted = generation_record.get("accepted") is True
        artifacts = {}
        required = {}
        if accepted:
            artifacts = {
                name: Path(record["path"])
                for stage in stages.values()
                for name, record in stage.get("artifacts", {}).items()
            }
            required = {
                "retrained_model": self.results_dir / "nep.txt",
                "training_set": self.results_dir / "train.xyz",
                "signals": self.results_dir / "metrics.json",
            }
            missing = [name for name in required if name not in artifacts]
            if missing:
                raise ValueError(
                    "accepted workflow generation is missing publishable artifacts: "
                    + ", ".join(missing)
                )
        _write_json(self.generation_dir(generation) / "summary.json", summary)
        if not accepted:
            return
        for name, target in required.items():
            _copy_file(artifacts[name], target)
        _write_json(self.results_dir / "summary.json", summary)
        evaluation = summary["metrics"].get("evaluate", {})
        lines = [
            f"# NepTrain workflow 结果\n",
            f"- 最新验收代：{generation}",
            f"- Energy RMSE：{evaluation.get('energy_rmse', 'n/a')}",
            f"- Force RMSE：{evaluation.get('force_rmse', 'n/a')}",
            f"- Virial RMSE：{evaluation.get('virial_rmse', 'n/a')}",
            "",
            "最终模型：`nep.txt`",
            "最终训练集：`train.xyz`",
            "完整指标：`metrics.json`",
            "",
        ]
        (self.results_dir / "summary.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )


__all__ = ["WorkflowWorkspace"]
