"""Explicit variable substitution shared by MD input templates."""

from __future__ import annotations

import re
from typing import Mapping


VARIABLE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


def render_template(
    template: str, variables: Mapping[str, object], *, backend: str
) -> str:
    missing = sorted(set(VARIABLE.findall(template)).difference(variables))
    if missing:
        raise ValueError(f"missing {backend} template variables: {', '.join(missing)}")

    def replace(match: re.Match[str]) -> str:
        value = variables[match.group(1)]
        return "" if value is None else str(value)

    rendered = VARIABLE.sub(replace, template)
    unresolved = VARIABLE.findall(rendered)
    if unresolved:
        raise ValueError(f"unresolved {backend} template variables: {unresolved}")
    return rendered
