"""Reduced chemical-composition identities used by sampling policy."""

from __future__ import annotations

from functools import reduce
from math import gcd
from typing import Mapping

from ase import Atoms
from ase.formula import Formula


CompositionKey = tuple[tuple[str, int], ...]


def reduced_composition(value: Atoms | str) -> CompositionKey:
    """Return a canonical stoichiometric ratio for a structure or formula."""

    counts: Mapping[str, int]
    if isinstance(value, Atoms):
        counts = Formula.from_list(value.get_chemical_symbols()).count()
    else:
        counts = Formula(value).count()
    positive = {symbol: int(count) for symbol, count in counts.items() if count > 0}
    if not positive or len(positive) != len(counts):
        raise ValueError("chemical composition must contain positive atom counts")
    divisor = reduce(gcd, positive.values())
    return tuple(
        (symbol, count // divisor) for symbol, count in sorted(positive.items())
    )


__all__ = ["CompositionKey", "reduced_composition"]
