"""NEP training and NEPAdapters calculation helpers."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "Nep3Calculator": ("NepTrain.core.nep.calculator", "Nep3Calculator"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
