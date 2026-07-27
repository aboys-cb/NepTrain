"""Selection helpers with the CLI implementation loaded only on demand."""

from __future__ import annotations

def run_select(args):
    from .run import run_select as implementation

    return implementation(args)


__all__ = ["run_select"]
