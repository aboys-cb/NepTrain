"""Selection helpers with the CLI implementation loaded only on demand."""

from __future__ import annotations

from .select import farthest_point_sampling, filter_by_bonds, process_trajectory, select_structures


def run_select(args):
    from .run import run_select as implementation

    return implementation(args)


__all__ = [
    "farthest_point_sampling",
    "filter_by_bonds",
    "process_trajectory",
    "run_select",
    "select_structures",
]
