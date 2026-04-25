# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jinja2 template rendering for AIPerf configuration."""

from __future__ import annotations

from typing import Any

import jinja2
from jinja2 import meta

from aiperf.config.loader.env_vars import substitute_env_vars
from aiperf.config.loader.errors import ConfigurationError

# Fields to skip when rendering Jinja2 templates (they contain Jinja2 templates themselves
# that are rendered at request time by the template endpoint, not at config load time)
SKIP_TEMPLATE_FIELDS = {"template", "body", "payload_template"}

# Dotted-path prefixes whose entire subtree is skipped during load-time rendering.
# Matched against ``current_path`` with a trailing ``.`` so siblings (e.g. ``artifacts.user``)
# do not match. Used for content rendered at run-time with run-time-only context
# (epoch, job_name, artifact_dir, ...), e.g. ``artifacts.user_files``.
SKIP_TEMPLATE_PATH_PREFIXES = ("artifacts.user_files",)

# Strict undefined surfaces typo'd / missing variables as ConfigurationError at load time
# rather than silently rendering empty strings that downstream parsers must catch.
# keep_trailing_newline=True preserves terminal '\n' so rendered artifact file contents
# (artifacts.user_files) survive unchanged instead of jinja's default one-newline strip.
_JINJA_ENV = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    autoescape=False,
    keep_trailing_newline=True,
)


def build_template_context(data: dict[str, Any]) -> dict[str, Any]:
    """Build context for Jinja2 template rendering.

    Creates a flattened context that allows both:
    - Direct access: ``{{ concurrency }}``
    - Dot notation access: ``{{ phases.test.concurrency }}``

    The ``variables`` section values are added at the top level for easy access.
    Cross-references inside the ``variables`` block (e.g.
    ``total_concurrency: "{{ a * b }}"``) are resolved in dependency order before
    being added to the context, so later expressions like ``{{ total_concurrency }}``
    see the computed value rather than the raw template.
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
        # Variables can reference any flattened context entry except other
        # variables-block names — those resolve in dep order via _resolve_variables_block.
        rest_ctx = {
            k: v for k, v in context.items() if k.split(".", 1)[0] != "variables"
        }
        # Also strip top-level keys that mirror variable names so a raw template
        # value isn't picked up before the dep graph resolves it.
        for key in data["variables"]:
            rest_ctx.pop(key, None)
        resolved = _resolve_variables_block(data["variables"], rest_ctx)
        for key, value in resolved.items():
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


def _collect_variable_refs(value: Any, candidate_names: set[str]) -> set[str]:
    """Walk ``value`` and return the subset of ``candidate_names`` it references.

    Recurses through dict/list values; only string leaves containing ``{{`` are
    parsed. Names that are syntactically referenced but absent from
    ``candidate_names`` are ignored here — they resolve from the surrounding
    context (or surface as ``ConfigurationError`` later during render).
    """
    refs: set[str] = set()
    if isinstance(value, str) and "{{" in value:
        try:
            ast = _JINJA_ENV.parse(value)
        except jinja2.TemplateError:
            return refs  # malformed template; let render() raise the real error
        refs |= meta.find_undeclared_variables(ast) & candidate_names
    elif isinstance(value, dict):
        for v in value.values():
            refs |= _collect_variable_refs(v, candidate_names)
    elif isinstance(value, list):
        for v in value:
            refs |= _collect_variable_refs(v, candidate_names)
    return refs


def _resolve_variables_block(
    variables: dict[str, Any],
    base_context: dict[str, Any],
) -> dict[str, Any]:
    """Resolve cross-references inside the ``variables`` block in dep order.

    A variable's value may reference any other variable (regardless of YAML
    order) and any top-level config field via ``base_context``. Resolution
    proceeds in dependency order using a Kahn-style topological pass; cycles
    raise ``ConfigurationError`` listing the participating names.

    Args:
        variables: Raw ``variables`` dict from the user config.
        base_context: Flattened context built from the rest of the config
            (everything except the variables block).

    Returns:
        New dict with each variable's value rendered and type-coerced via
        the same ``_coerce_rendered`` rules used for global templates.
    """
    if not variables:
        return {}
    var_names = set(variables)
    pending = {
        name: _collect_variable_refs(value, var_names)
        for name, value in variables.items()
    }
    resolved: dict[str, Any] = {}
    while pending:
        ready = sorted(name for name, deps in pending.items() if not deps)
        if not ready:
            raise ConfigurationError(
                "Circular reference among variables: "
                + ", ".join(sorted(pending.keys())),
                context="Each variable's value transitively depends on another in the cycle.",
            )
        for name in ready:
            ctx = {**base_context, **resolved}
            resolved[name] = render_jinja2_templates(
                variables[name], ctx, current_path=f"variables.{name}"
            )
            del pending[name]
            for deps in pending.values():
                deps.discard(name)
    return resolved


def _path_is_skipped(current_path: str) -> bool:
    """True if ``current_path`` is at or under any SKIP_TEMPLATE_PATH_PREFIXES entry.

    Matches exact prefix or prefix followed by ``.`` so that ``artifacts.user_files``
    matches ``artifacts.user_files``, ``artifacts.user_files.0.content.k`` but NOT
    ``artifacts.user_files_other``.
    """
    for prefix in SKIP_TEMPLATE_PATH_PREFIXES:
        if current_path == prefix or current_path.startswith(prefix + "."):
            return True
    return False


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
        template = _JINJA_ENV.from_string(data)
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
    are rendered at request time, not config load time) and entire subtrees
    under SKIP_TEMPLATE_PATH_PREFIXES (e.g. ``artifacts.user_files``,
    rendered at run start with run-time-only context).
    """
    if _path_is_skipped(current_path):
        return data

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
    rather than a YAML string. The ``variables`` key is preserved on the
    returned dict so run-time renderers (e.g. ``artifacts.user_files``) can
    resolve it again later.

    Order:
        1. ``${VAR}`` / ``${VAR:default}`` substitution from ``os.environ``
        2. Jinja2 ``{{ expr }}`` rendering using the dict itself as context

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
    return data
