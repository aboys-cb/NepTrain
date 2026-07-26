from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "docs/source/command/manual.md",
    ROOT / "docs/source/command/workflow.md",
)
NESTED = {"workflow", "task", "data"}


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


def test_documented_cli_options_exist_on_the_documented_command():
    commands = [
        command
        for document in DOCUMENTS
        for command in _bash_commands(document.read_text(encoding="utf-8"))
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    help_by_surface = {}
    for command in commands:
        tokens = shlex.split(command)
        surface = tuple(tokens[1 : 3 if tokens[1] in NESTED else 2])
        if surface not in help_by_surface:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "NepTrain.cli.cli",
                    *surface,
                    "--help",
                ],
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            )
            assert completed.returncode == 0, (surface, completed.stderr)
            help_by_surface[surface] = completed.stdout
        documented_options = {
            token.split("=", 1)[0]
            for token in tokens
            if token.startswith("--")
        }
        missing = sorted(
            option
            for option in documented_options
            if option not in help_by_surface[surface]
        )
        assert not missing, f"{command}: undocumented parser option(s) {missing}"


def test_direct_vasp_examples_pin_the_potcar_manifest():
    for document in DOCUMENTS:
        for command in _bash_commands(document.read_text(encoding="utf-8")):
            if (
                "neptrain dft " in command
                and "--backend vasp" in command
                and "--resources" in command
                and "--project" not in command
            ):
                assert "--potcar-manifest" in command, command
