"""Read-only input checks for a user's next workflow run."""

from pathlib import Path
from typing import Any, Mapping

from .scientific_data import optional_dataset_issue
from .workflow import (
    _optional_labeled_dataset_issue,
    _resolved_config,
    _sampling_frames,
    _validate_labeled_dataset_for_preparation,
)

def check_project_inputs(
    config: Mapping[str, Any], base_dir: Path
) -> list[tuple[str, str]]:
    """Collect independent input problems instead of stopping at the first one."""
    resolved = _resolved_config(config, base_dir)
    checks: list[tuple[str, str]] = []
    spin = bool(config.get("md", {}).get("spin", False))

    def dataset(value, role, *, optional=False):
        path = Path(value) if value else None
        if optional:
            issue = _optional_labeled_dataset_issue(path, role=role, expect_spin=spin)
            checks.append(("WARN", issue) if issue else ("OK", f"{role}: {path}"))
            return
        issue = optional_dataset_issue(path, role=role)
        if issue:
            checks.append(
                (
                    "FAIL",
                    f"{role}: {path or 'not configured'}; provide a labeled extxyz dataset or correct the path.",
                )
            )
            return
        try:
            _validate_labeled_dataset_for_preparation(path, role=role, expect_spin=spin)
        except Exception as error:
            checks.append(("FAIL", f"{role}: {path}; {error}"))
        else:
            checks.append(("OK", f"{role}: {path}"))

    def required_file(value, role):
        path = Path(value) if value else None
        issue = optional_dataset_issue(path, role=role)
        if issue:
            checks.append(
                (
                    "FAIL",
                    f"{role}: {path or 'not configured'}; create the file or correct the configured path.",
                )
            )
        else:
            checks.append(("OK", f"{role}: {path}"))

    training = resolved.get("training", {})
    initial = training.get("initial_path")
    if initial and not Path(initial).is_absolute():
        initial = str((base_dir / initial).resolve())
    dataset(initial, "training.initial_path")
    required_file(training.get("config_path"), "training.config_path")
    for section, key in (("training", "test_path"), ("evaluation", "validation_path")):
        dataset(resolved.get(section, {}).get(key), f"{section}.{key}", optional=True)
    for route in resolved.get("sampling", {}).get("routes", []):
        role = f"sampling.routes[{route['id']}]"
        required_file(route.get("template_path"), f"{role}.template_path")
        try:
            frames = _sampling_frames({"sampling": {"routes": [route]}})
        except Exception as error:
            checks.append(
                (
                    "FAIL",
                    f"{role}.structures: {error}; provide readable structures at the configured path.",
                )
            )
        else:
            checks.append(("OK", f"{role}.structures: {len(frames)} structures"))
    labeling = resolved.get("labeling", {})
    if labeling.get("backend") in {"vasp", "abacus"}:
        required_file(labeling.get("input_path"), "labeling.input_path")
    for name, target in config.get("execution", {}).get("targets", {}).items():
        if "REPLACE" in str(target):
            checks.append(
                (
                    "FAIL",
                    f"execution.targets.{name} still contains REPLACE; set the actual partition, resource paths, and environment.",
                )
            )
    if not config.get("workflow", {}).get("convergence"):
        checks.append(
            (
                "WARN",
                "Convergence checks are disabled (active_learning_v4). Budget or coverage exhaustion does not establish accuracy.",
            )
        )
    return checks
