"""NepTrain core modules.

Heavy optional dependencies are imported only when their command is used.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "run_perturb": ("NepTrain.core.perturb", "run_perturb"),
    "init_template": ("NepTrain.core.template", "init_template"),
    "run_select": ("NepTrain.core.select", "run_select"),
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
