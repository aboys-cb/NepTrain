from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex

import pytest

from NepTrain.cli import cli

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "README.zh-CN.md",
    *sorted((ROOT / "examples").rglob("README*.md")),
    *sorted((ROOT / "docs/source").rglob("*.md")),
)


def _bash_commands(text: str) -> list[str]:
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        logical = ""
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            logical += (" " if logical else "") + line.removesuffix("\\").strip()
            if raw_line.rstrip().endswith("\\"):
                continue
            if logical.startswith("neptrain "):
                commands.append(logical)
            logical = ""
    return commands


@pytest.fixture(scope="module")
def documented_parser():
    """Capture the actual CLI parser before dispatch; never submit documented jobs."""

    class ParserCaptured(Exception):
        pass

    captured = []

    def capture(parser, *args, **kwargs):
        captured.append(parser)
        raise ParserCaptured

    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("_ARGCOMPLETE", raising=False)
        patch.setattr(argparse.ArgumentParser, "parse_args", capture)
        with pytest.raises(ParserCaptured):
            cli.main()
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize(
    "document", DOCUMENTS, ids=lambda path: str(path.relative_to(ROOT))
)
def test_documented_commands_parse(documented_parser, document):
    for command in _bash_commands(document.read_text(encoding="utf-8")):
        # shlex handles shell comments but does not run substitutions or commands.
        tokens = shlex.split(command, comments=True)
        try:
            documented_parser.parse_args(tokens[1:])
        except SystemExit as error:
            assert error.code == 0, f"{document.relative_to(ROOT)}: {command}"


def test_direct_vasp_examples_pin_the_potcar_manifest():
    for document in DOCUMENTS:
        for command in _bash_commands(document.read_text(encoding="utf-8")):
            if (
                "neptrain label " in command
                and "--backend vasp" in command
                and "--resources" in command
                and "--project" not in command
            ):
                assert "--potcar-manifest" in command, command


def test_distillation_training_checks_match_manual_output_layout(
    documented_parser, tmp_path, monkeypatch
):
    """Exercise the manual task layout without running an external trainer."""
    from NepTrain.core.training import TrainingResult

    document = ROOT / "examples/distillation-mace/README.md"
    command = next(
        command
        for command in _bash_commands(document.read_text())
        if command.startswith("neptrain train ")
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "labeled.xyz").write_text("backend fixture\n")
    (tmp_path / "nep-smoke.in").write_text("backend fixture\n")

    def fake_train(request, backend):
        model = request.output_dir / "nep.txt"
        model.write_text("backend fixture\n")
        # Only stand in for the backend; preparation, worker, and publication are real.
        for name in ("training-report.json", "training-convergence.png"):
            (request.output_dir / name).write_text("backend artifact\n")
        return TrainingResult(backend, model, None, None)

    monkeypatch.setattr("NepTrain.core.training.train", fake_train)
    args = documented_parser.parse_args(shlex.split(command)[1:])
    args.func(args)
    for readme in sorted((ROOT / "examples").glob("distillation-*/README*.md")):
        checks = re.findall(
            r"^test -s (student-\S+)$", readme.read_text(), re.MULTILINE
        )
        assert checks, readme
        for relative in checks:
            path = tmp_path / relative
            assert path.is_file() and path.stat().st_size, (readme, relative)
