"""Keep shipped runtime copy English while preserving user-provided Unicode."""

from __future__ import annotations

import ast
from pathlib import Path
import re

import pytest

from NepTrain.core.notifications import _workflow_identity
from NepTrain.core.template import init_project

ROOT = Path(__file__).resolve().parents[1]
NON_ENGLISH_COPY = re.compile(
    r"[\u3400-\u9fff\u3001\u3002\uff01\uff08\uff09\uff0c\uff1a\uff1b]"
)


def test_runtime_strings_and_example_templates_use_english():
    """Scan literals, not author credits or user files; README/docs stay bilingual."""
    problems = []
    for root in (ROOT / "src/NepTrain", ROOT / "examples"):
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".py":
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    if (
                        isinstance(node, ast.Constant)
                        and isinstance(node.value, str)
                        and NON_ENGLISH_COPY.search(node.value)
                    ):
                        problems.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}: {node.value!r}"
                        )
            elif path.suffix in {".in", ".yaml", ".json", ".sh"} or path.name in {
                "INCAR",
                "INCAR.deltaspin",
                "INPUT",
            }:
                for number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), 1
                ):
                    if NON_ENGLISH_COPY.search(line):
                        problems.append(f"{path.relative_to(ROOT)}:{number}: {line}")
    assert not problems, "Non-English runtime copy:\n" + "\n".join(problems)


@pytest.mark.parametrize("backend", ["vasp", "abacus"])
def test_generated_project_copy_is_english(tmp_path, capsys, backend):
    root = tmp_path / backend
    init_project("local", root, dft_backend=backend)
    output = capsys.readouterr().out
    assert "Project created:" in output
    assert "Next: neptrain doctor --project" in output
    assert not NON_ENGLISH_COPY.search(output)
    for path in root.rglob("*"):
        if path.is_file():
            assert not NON_ENGLISH_COPY.search(path.read_text()), path


def test_english_output_preserves_user_names_and_paths(tmp_path, capsys):
    root = tmp_path / "磁性项目 Fe"
    init_project("local", root)
    output = capsys.readouterr().out
    assert str(root) in output
    assert "Project created:" in output
    identity = _workflow_identity("铁合金", root)
    assert identity == ("Workflow: 铁合金", f"Path: {root.resolve()}")


def test_empty_test_history_is_not_presented_as_missing_sampling_data(capsys):
    from types import SimpleNamespace
    from NepTrain.cli.cli import _print_precision

    _print_precision(SimpleNamespace(precision_basis="validation", generations=()))
    text = capsys.readouterr().out
    assert text.strip() == "Optional test: No results available yet"


def test_job_summary_omits_unavailable_attempt_name(capsys):
    from NepTrain.cli.cli import _print_job_batches

    _print_job_batches(
        [{"generation": 1, "stage": "train", "state": "RUNNING", "script": "train.sh"}]
    )
    text = capsys.readouterr().out
    assert "G1 Training: Jobs: 1" in text
    assert "None" not in text
