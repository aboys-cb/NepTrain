"""User-facing filesystem interface for a NepTrain workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from .content_addressing import canonical_sha256, file_sha256
from .persistence import atomic_write_json


_LAYOUT_VERSION = 4
_STAGE_DIRECTORIES = {
    "train": "train",
    "explore": "md",
    "select": "select",
    "label": "label",
    "diagnose": "diagnose",
    "merge": "dataset",
    "retrain": "retrain",
    "evaluate": "evaluate",
}
_PUBLICATION_FILENAMES = (
    "nep.txt",
    "train.xyz",
    "metrics.json",
    "summary.json",
    "summary.md",
)


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def _copy_path(source: Path, target: Path) -> None:
    if source.is_file():
        _copy_file(source, target)
        return
    if not source.is_dir():
        raise FileNotFoundError(f"workflow input does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    temporary.replace(target)


def _atomic_symlink(target: str, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.link-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    temporary.replace(link)


@dataclass(frozen=True)
class WorkflowWorkspace:
    """Own every durable path in one workflow directory.

    Layout v3 keeps the machine state below ``.neptrain`` and exposes only
    inputs, generation evidence, logs, and accepted results at the project
    root. Test-phase layouts are intentionally not migrated or retained.
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
            workspace.tasks_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        (workspace.results_dir / "accepted").mkdir()
        for name in (
            "nep.txt",
            "train.xyz",
            "metrics.json",
            "summary.json",
            "summary.md",
        ):
            _atomic_symlink(f"current/{name}", workspace.results_dir / name)
        atomic_write_json(
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
            "`stop` 默认同时取消当前计算任务；仅停止 Controller 时使用："
            "`neptrain workflow stop . --keep-jobs`。\n",
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
            version = int(value.get("version", 0))
            if value.get("layout") != "neptrain.workflow" or version != _LAYOUT_VERSION:
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
    def tasks_dir(self) -> Path:
        return self.internal_dir / "jobs"

    @property
    def locks_dir(self) -> Path:
        return self.internal_dir / "locks"

    @property
    def manifest_lock(self) -> Path:
        return self.internal_dir / ".manifest.lock"

    @property
    def ledger(self) -> Path:
        return self.internal_dir / "ledger.json"

    @property
    def ledger_lock(self) -> Path:
        return self.internal_dir / ".ledger.lock"

    @property
    def controller_file(self) -> Path:
        return self.internal_dir / "controller.json"

    @property
    def controller_lock(self) -> Path:
        return self.internal_dir / ".controller.lock"

    @property
    def controller_pid(self) -> Path:
        return self.internal_dir / "controller.pid"

    @property
    def controller_log(self) -> Path:
        return self.logs_dir / "controller.log"

    @property
    def notification_state(self) -> Path:
        return self.internal_dir / "notifications.json"

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
        for route in snapshot.get("sampling", {}).get("routes", []):
            route_root = (
                self.inputs_dir / "sampling" / "routes" / str(route["id"])
            )
            template_source = Path(route["template_path"])
            template_target = (
                route_root / f"template{template_source.suffix}"
            )
            _copy_path(template_source, template_target)
            route["template_path"] = str(
                template_target.relative_to(self.root)
            )
            copied_structures = []
            for index, raw in enumerate(route["structures"]):
                source = Path(raw)
                target = route_root / "structures" / str(index)
                if source.is_file():
                    target = target.with_suffix(source.suffix)
                _copy_path(source, target)
                copied_structures.append(
                    str(target.relative_to(self.root))
                )
            route["structures"] = copied_structures
        copy_path("labeling", "input_path", "labeling/input")
        copy_path(
            "labeling",
            "potcar_manifest_path",
            "labeling/vasp-resources",
        )
        copy_path(
            "labeling",
            "resource_manifest_path",
            "labeling/abacus-resources",
        )
        copy_path("labeling", "model_path", "labeling/teacher-model")
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

    @staticmethod
    def _generation_summary(
        generation: int,
        generation_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        stages = generation_record.get("stages", {})
        return {
            "generation": generation,
            "accepted": generation_record.get("accepted"),
            "complete": bool(generation_record.get("complete")),
            "metrics": {
                stage: dict(record.get("metrics", {}))
                for stage, record in stages.items()
            },
        }

    @staticmethod
    def _summary_markdown(summary: Mapping[str, Any]) -> str:
        generation = int(summary["generation"])
        evaluation = summary["metrics"].get("evaluate", {})
        lines = ["# NepTrain workflow 结果", ""]
        if evaluation.get("evaluation_configured") is False:
            lines.extend(
                [
                    f"- 最新完成代：{generation}",
                    "- 独立 evaluation：未配置",
                ]
            )
        else:
            lines.extend(
                [
                    f"- 最新验收代：{generation}",
                    f"- Energy RMSE：{evaluation.get('energy_rmse', 'n/a')}",
                    f"- Force RMSE：{evaluation.get('force_rmse', 'n/a')}",
                    f"- Virial RMSE：{evaluation.get('virial_rmse', 'n/a')}",
                ]
            )
        lines.extend(
            [
                "",
                "最终模型：`nep.txt`",
                "最终训练集：`train.xyz`",
                "完整指标：`metrics.json`",
                "",
            ]
        )
        return "\n".join(lines)

    def prepare_generation_publication(
        self, generation: int, generation_record: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Prepare an immutable accepted result without changing ``current``."""

        stages = generation_record.get("stages", {})
        summary = self._generation_summary(generation, generation_record)
        accepted = generation_record.get("accepted") is True
        if not accepted:
            return None

        artifacts = {
            name: record
            for stage in stages.values()
            for name, record in stage.get("artifacts", {}).items()
        }
        required = {
            "activated_model": "nep.txt",
            "training_set": "train.xyz",
            "signals": "metrics.json",
        }
        missing = [name for name in required if name not in artifacts]
        if missing:
            raise ValueError(
                "accepted workflow generation is missing publishable artifacts: "
                + ", ".join(missing)
            )
        sources = {}
        for name in required:
            record = artifacts[name]
            source = Path(record["path"])
            if not source.is_file() or file_sha256(source) != record["sha256"]:
                raise ValueError(
                    f"accepted workflow artifact drifted before publication: {source}"
                )
            sources[name] = {
                "path": str(source),
                "sha256": record["sha256"],
            }
        content_sha256 = canonical_sha256(
            {
                "generation": generation,
                "sources": sources,
                "summary": summary,
            }
        )
        accepted_root = self.results_dir / "accepted"
        accepted_root.mkdir(exist_ok=True)
        final = accepted_root / f"g{generation:04d}-{content_sha256[:12]}"
        if final.exists():
            valid = False
            try:
                existing = json.loads(
                    (final / "publication.json").read_text(encoding="utf-8")
                )
                valid = (
                    existing.get("protocol") == "neptrain.accepted-result.v1"
                    and int(existing.get("generation", -1)) == generation
                    and existing.get("content_sha256") == content_sha256
                    and set(existing.get("files", {}))
                    == set(_PUBLICATION_FILENAMES)
                    and all(
                        record.get("path") == name
                        and (final / name).is_file()
                        and (final / name).stat().st_size
                        == int(record["size"])
                        and file_sha256(final / name)
                        == record["sha256"]
                        for name, record in existing.get("files", {}).items()
                    )
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                valid = False
            if not valid:
                damaged_root = self.results_dir / "damaged"
                damaged_root.mkdir(exist_ok=True)
                quarantine = damaged_root / f"{final.name}-{os.getpid()}"
                suffix = 1
                while quarantine.exists():
                    quarantine = damaged_root / (
                        f"{final.name}-{os.getpid()}-{suffix}"
                    )
                    suffix += 1
                final.replace(quarantine)
        if not final.exists():
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f".g{generation:04d}-building-",
                    dir=accepted_root,
                )
            )
            # Accepted publications are immutable, but shared workflow groups
            # must still be able to inspect and recover the controller.
            temporary.chmod(0o2770)
            try:
                for artifact_name, filename in required.items():
                    _copy_file(
                        Path(sources[artifact_name]["path"]),
                        temporary / filename,
                    )
                atomic_write_json(temporary / "summary.json", summary)
                (temporary / "summary.md").write_text(
                    self._summary_markdown(summary),
                    encoding="utf-8",
                )
                files = {
                    filename: {
                        "path": filename,
                        "sha256": file_sha256(temporary / filename),
                        "size": (temporary / filename).stat().st_size,
                    }
                    for filename in _PUBLICATION_FILENAMES
                }
                atomic_write_json(
                    temporary / "publication.json",
                    {
                        "protocol": "neptrain.accepted-result.v1",
                        "generation": generation,
                        "content_sha256": content_sha256,
                        "sources": sources,
                        "files": files,
                    },
                )
                temporary.replace(final)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        manifest = final / "publication.json"
        value = json.loads(manifest.read_text(encoding="utf-8"))
        if value.get("content_sha256") != content_sha256:
            raise ValueError(f"accepted publication collision: {final}")
        record = {
            "protocol": "neptrain.accepted-result.v1",
            "generation": generation,
            "content_sha256": content_sha256,
            "directory": str(final),
            "manifest": {
                "path": str(manifest),
                "sha256": file_sha256(manifest),
            },
            "files": {
                name: {
                    **file_record,
                    "path": str(final / file_record["path"]),
                }
                for name, file_record in value["files"].items()
            },
        }
        issues = self.publication_issues(record, check_projection=False)
        if issues:
            raise ValueError("; ".join(issues))
        return record

    def activate_generation(
        self,
        generation: int,
        generation_record: Mapping[str, Any],
    ) -> None:
        """Repair human projections and atomically activate one publication."""

        summary = self._generation_summary(generation, generation_record)
        atomic_write_json(self.generation_dir(generation) / "summary.json", summary)
        if generation_record.get("accepted") is not True:
            return
        publication = generation_record.get("publication")
        if not isinstance(publication, Mapping):
            raise ValueError(
                f"accepted generation {generation} has no publication record"
            )
        issues = self.publication_issues(publication, check_projection=False)
        if issues:
            raise ValueError("; ".join(issues))
        directory = Path(publication["directory"])
        current = self.results_dir / "current"
        if not current.is_symlink() or current.resolve() != directory.resolve():
            if not current.is_symlink() and current.exists():
                raise ValueError("results/current exists but is not a symlink")
            # On a new or legacy workspace, establish the pointer first.  Fixed
            # links either do not exist yet or still contain the same accepted
            # generation that is being migrated.
            if not current.exists() and not current.is_symlink():
                _atomic_symlink(
                    os.path.relpath(directory, self.results_dir),
                    current,
                )
        for name in publication["files"]:
            _atomic_symlink(f"current/{name}", self.results_dir / name)
        if current.resolve() != directory.resolve():
            _atomic_symlink(
                os.path.relpath(directory, self.results_dir),
                current,
            )
        issues = self.publication_issues(publication, check_projection=True)
        if issues:
            raise ValueError("; ".join(issues))

    def publication_issues(
        self,
        publication: Mapping[str, Any],
        *,
        check_projection: bool,
    ) -> list[str]:
        issues: list[str] = []
        directory = Path(str(publication.get("directory", "")))
        manifest_record = publication.get("manifest", {})
        manifest = Path(str(manifest_record.get("path", "")))
        if (
            not directory.is_dir()
            or not manifest.is_file()
            or file_sha256(manifest) != manifest_record.get("sha256")
        ):
            issues.append(f"accepted publication metadata drifted: {directory}")
            return issues
        try:
            manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            issues.append(f"accepted publication metadata is unreadable: {manifest}")
            return issues
        publication_files = publication.get("files", {})
        manifest_files = manifest_value.get("files", {})
        if (
            manifest_value.get("protocol") != "neptrain.accepted-result.v1"
            or manifest_value.get("content_sha256")
            != publication.get("content_sha256")
            or set(publication_files) != set(_PUBLICATION_FILENAMES)
            or set(manifest_files) != set(_PUBLICATION_FILENAMES)
        ):
            issues.append(
                f"accepted publication file contract drifted: {directory}"
            )
            return issues
        for name in _PUBLICATION_FILENAMES:
            record = publication_files[name]
            manifest_file = manifest_files[name]
            path = Path(str(record.get("path", "")))
            if (
                manifest_file.get("path") != name
                or manifest_file.get("sha256") != record.get("sha256")
                or int(manifest_file.get("size", -1))
                != int(record.get("size", -1))
                or not path.is_file()
                or path.stat().st_size != int(record.get("size", -1))
                or file_sha256(path) != record.get("sha256")
            ):
                issues.append(f"accepted result drifted: {name} ({path})")
        if not check_projection:
            return issues
        current = self.results_dir / "current"
        if not current.is_symlink() or current.resolve() != directory.resolve():
            issues.append("results/current does not select the ledger publication")
            return issues
        for name in publication.get("files", {}):
            visible = self.results_dir / name
            if not visible.is_symlink() or visible.resolve() != directory / name:
                issues.append(f"results/{name} is not the current publication")
        return issues


__all__ = ["WorkflowWorkspace"]
