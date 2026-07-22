"""User-facing filesystem interface for a NepTrain campaign."""

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
class CampaignWorkspace:
    """Own every durable path in one campaign directory.

    Layout v2 keeps the machine state below ``.neptrain`` and exposes only
    inputs, generation evidence, logs, and accepted results at the project
    root.  ``locate`` can still open a v1 campaign so completed campaigns are
    not stranded, but newly created workspaces always use v2.
    """

    root: Path
    version: int = _LAYOUT_VERSION

    @classmethod
    def create(cls, root: str | Path) -> "CampaignWorkspace":
        workspace = cls(Path(root).expanduser().resolve())
        if workspace.root.exists() and any(workspace.root.iterdir()):
            raise ValueError(
                f"campaign output directory is not empty: {workspace.root}"
            )
        workspace.root.mkdir(parents=True, exist_ok=True)
        for directory in (
            workspace.inputs_dir,
            workspace.results_dir,
            workspace.generations_dir,
            workspace.logs_dir,
            workspace.plans_dir,
            workspace.jobs_dir,
            workspace.locks_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        _write_json(
            workspace.layout_file,
            {
                "layout": "neptrain.campaign",
                "version": _LAYOUT_VERSION,
                "user_directories": ["inputs", "results", "generations", "logs"],
                "internal_directory": ".neptrain",
            },
        )
        (workspace.root / "README.md").write_text(
            "# NepTrain campaign\n\n"
            "- `project.yaml`：本次运行采用的完整配置快照。\n"
            "- `inputs/`：进入 campaign 的输入快照。\n"
            "- `results/`：最新通过验收的模型、训练集和指标。\n"
            "- `generations/`：按代组织的采样、标注、训练和评价证据。\n"
            "- `logs/`：Slurm 标准输出。\n"
            "- `.neptrain/`：账本、计划和作业脚本等内部状态，通常无需编辑。\n"
            "\n常用命令：`NepTrain status .`、`NepTrain resume .`。\n",
            encoding="utf-8",
        )
        return workspace

    @classmethod
    def locate(cls, path: str | Path) -> "CampaignWorkspace":
        candidate = Path(path).expanduser().resolve()
        if candidate.is_file():
            if candidate.name == "manifest.json" and candidate.parent.name == ".neptrain":
                candidate = candidate.parent.parent
            elif candidate.name == "campaign-manifest.json":
                return cls(candidate.parent, version=1)
        layout = candidate / ".neptrain" / "layout.json"
        if layout.is_file():
            value = json.loads(layout.read_text(encoding="utf-8"))
            if value.get("layout") != "neptrain.campaign" or int(
                value.get("version", 0)
            ) != _LAYOUT_VERSION:
                raise ValueError(f"unsupported campaign layout: {layout}")
            return cls(candidate)
        if (candidate / "campaign-manifest.json").is_file():
            return cls(candidate, version=1)
        raise FileNotFoundError(f"NepTrain campaign does not exist: {candidate}")

    @property
    def internal_dir(self) -> Path:
        return self.root / ".neptrain" if self.version == 2 else self.root

    @property
    def layout_file(self) -> Path:
        return self.internal_dir / "layout.json"

    @property
    def manifest(self) -> Path:
        return (
            self.internal_dir / "manifest.json"
            if self.version == 2
            else self.root / "campaign-manifest.json"
        )

    @property
    def project_file(self) -> Path:
        return self.root / ("project.yaml" if self.version == 2 else "job.resolved.yaml")

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def generations_dir(self) -> Path:
        return self.root / "generations" if self.version == 2 else self.root / "state"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def plans_dir(self) -> Path:
        return self.internal_dir / "plans" if self.version == 2 else self.root / "plans"

    @property
    def jobs_dir(self) -> Path:
        return self.internal_dir / "jobs" if self.version == 2 else self.root / "jobs"

    @property
    def locks_dir(self) -> Path:
        return self.internal_dir / "locks" if self.version == 2 else self.root

    @property
    def manifest_lock(self) -> Path:
        return (
            self.locks_dir / "manifest.lock"
            if self.version == 2
            else self.root / ".campaign-manifest.lock"
        )

    @property
    def ledger(self) -> Path:
        return (
            self.internal_dir / "ledger.json"
            if self.version == 2
            else self.root / "state" / "campaign-ledger.json"
        )

    @property
    def ledger_lock(self) -> Path:
        return (
            self.locks_dir / "ledger.lock"
            if self.version == 2
            else self.root / "state" / ".campaign-ledger.lock"
        )

    @property
    def controller_root(self) -> Path:
        return self.root if self.version == 2 else self.root / "state"

    def generation_dir(self, generation: int) -> Path:
        if self.version == 2:
            return self.generations_dir / f"{generation:04d}"
        return self.generations_dir / f"Generation-{generation}"

    def stage_dir(self, generation: int, stage: str) -> Path:
        if self.version != 2:
            return self.generation_dir(generation)
        try:
            relative = _STAGE_DIRECTORIES[stage]
        except KeyError as error:
            raise ValueError(f"unknown campaign stage: {stage}") from error
        return self.generation_dir(generation) / relative

    def snapshot_inputs(
        self, config: Mapping[str, Any], initial_training: Path
    ) -> tuple[dict[str, Any], Path]:
        """Create the portable user input view used by all generated jobs."""

        if self.version != 2:
            return json.loads(json.dumps(config)), initial_training
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
        copy_path("dft", "incar_path", "dft/input")
        copy_path("evaluation", "validation_path", "validation/validation")
        for resource in ("training", "cpu", "dft"):
            profile = snapshot.get("campaign", {}).get("slurm", {}).get(
                resource, {}
            )
            value = profile.get("setup_script")
            if value in {None, ""}:
                continue
            source = Path(value)
            if not source.is_file():
                continue
            target = self.inputs_dir / "platform" / f"{resource}.sh"
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

        if self.version != 2:
            return
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
                    "accepted campaign generation is missing publishable artifacts: "
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
            f"# NepTrain campaign 结果\n",
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


__all__ = ["CampaignWorkspace"]
