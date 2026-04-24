# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jinja2 template rendering for AIPerf configuration."""

from __future__ import annotations

from typing import Any

import jinja2

from aiperf.config.loader.env_vars import substitute_env_vars
from aiperf.config.loader.errors import ConfigurationError

# Fields to skip when rendering Jinja2 templates (they contain Jinja2 templates themselves
# that are rendered at request time by the template endpoint, not at config load time)
SKIP_TEMPLATE_FIELDS = {"template", "body", "payload_template"}


def build_template_context(data: dict[str, Any]) -> dict[str, Any]:
    """Build context for Jinja2 template rendering.

    Creates a flattened context that allows both:
    - Direct access: ``{{ concurrency }}``
    - Dot notation access: ``{{ phases.test.concurrency }}``

    The ``variables`` section values are added at the top level for easy access.
    """
    context: dict[str, Any] = {}

    def flatten(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}.{key}" if prefix else key
                context[new_key] = value
                if not prefix:
                    context[key] = value
                flatten(value, new_key)
        elif isinstance(obj, list):
            context[prefix] = obj
            for i, item in enumerate(obj):
                flatten(item, f"{prefix}.{i}")

    flatten(data)

    if "variables" in data and isinstance(data["variables"], dict):
        for key, value in data["variables"].items():
            context[key] = value

    return context


def _coerce_rendered(rendered: str) -> Any:
    """Coerce a rendered Jinja2 string to bool/int/float when possible."""
    if rendered.lower() == "true":
        return True
    if rendered.lower() == "false":
        return False
    try:
        return int(rendered)
    except ValueError:
        pass
    try:
        return float(rendered)
    except ValueError:
        pass
    return rendered


def _render_template_string(
    data: str,
    context: dict[str, Any],
    current_path: str,
) -> Any:
    """Render a single Jinja2 template string and coerce its output."""
    field_name = current_path.split(".")[-1] if current_path else ""
    if field_name in SKIP_TEMPLATE_FIELDS:
        return data

    if "{{" not in data or "}}" not in data:
        return data

    try:
        template = jinja2.Template(data)
        rendered = template.render(**context)
    except jinja2.TemplateError as e:
        raise ConfigurationError(
            f"Jinja2 template error at '{current_path}': {e}",
            context=f"Template: {data}",
        ) from e

    return _coerce_rendered(rendered)


def render_jinja2_templates(
    data: Any,
    context: dict[str, Any],
    current_path: str = "",
) -> Any:
    """Recursively render Jinja2 ``{{ ... }}`` template strings in config data.

    Processes strings containing ``{{ ... }}`` patterns and evaluates them
    using the provided context. Results are auto-converted to appropriate
    types (int, float, bool, or string).

    Skips fields in SKIP_TEMPLATE_FIELDS (endpoint payload templates that
    are rendered at request time, not config load time).
    """
    if isinstance(data, str):
        return _render_template_string(data, context, current_path)

    if isinstance(data, dict):
        return {
            k: render_jinja2_templates(
                v, context, f"{current_path}.{k}" if current_path else k
            )
            for k, v in data.items()
        }

    if isinstance(data, list):
        return [
            render_jinja2_templates(item, context, f"{current_path}.{i}")
            for i, item in enumerate(data)
        ]

    return data


def expand_config_dict(
    data: dict[str, Any],
    *,
    substitute_env: bool = True,
) -> dict[str, Any]:
    """Apply env var substitution and Jinja2 expansion to a raw config dict.

    Mirrors the expansion pipeline in ``load_config_from_string()``. Use this
    when you already have a parsed dict (e.g., from a Kubernetes CRD spec)
    rather than a YAML string. The ``variables`` key is removed after rendering.

    Order:
        1. ``${VAR}`` / ``${VAR:default}`` substitution from ``os.environ``
        2. Jinja2 ``{{ expr }}`` rendering using the dict itself as context
        3. ``variables`` key removed (it was only needed for Jinja2 context)

    Args:
        data: Raw config dict to expand (mutated copy is returned).
        substitute_env: If False, skip env var substitution.

    Returns:
        New dict with all expansions applied.

    Raises:
        MissingEnvironmentVariableError: If a required env var (no default) is absent.
        ConfigurationError: If a Jinja2 template fails to render.
    """
    if substitute_env:
        data = substitute_env_vars(data)
    context = build_template_context(data)
    data = render_jinja2_templates(data, context)
    data.pop("variables", None)
    return data
