# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
YAML configuration loading for AIPerf (schema 2.x).

Loads and validates benchmark YAML with optional environment-variable
substitution and Jinja2 rendering. Body fields (models, endpoint, datasets,
phases, ...) live under the ``benchmark:`` key; envelope-level keys include
``variables``, ``sweep``, ``multi_run``, and ``random_seed``.

Exports include ``load_config``, ``load_config_from_string``, ``load_config_dict``,
``load_config_from_env``, ``validate_config_file``, ``merge_configs``, and helpers
for env substitution and Jinja expansion.

Example:
    >>> from aiperf.config import load_config
    >>> config = load_config("benchmark.yaml")
    >>> print(config.models)
    >>> print(config.phases[0].name)

Environment variables in YAML (processed before Jinja when enabled):

    ${VAR}         - required; raises ``MissingEnvironmentVariableError`` if unset
    ${VAR:default} - optional default
    ${VAR:}        - optional, default empty string

Jinja2 (``{{ ... }}``) uses a flattened context: names from the top-level
``variables:`` block, plus paths into the loaded document. Phases are typically
a *list* of objects with a ``name`` field, so named phases are addressable as
``phases.<name>.<field>`` (e.g. ``phases.profiling.concurrency``) as well as
``benchmark.phases.<name>.<field>``.

Illustrative YAML:

    variables:
      base_concurrency: 16
    benchmark:
      phases:
        - name: profiling
          type: concurrency
          concurrency: "{{ base_concurrency }}"
          requests: "{{ base_concurrency * 100 }}"

Cross-references inside ``variables:`` are resolved in dependency order;
cycles raise ``ConfigurationError`` with the participating names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.config.loader.env_vars import (
    ENV_VAR_PATTERN,
    substitute_env_vars,
)
from aiperf.config.loader.errors import (
    ConfigurationError,
    MissingEnvironmentVariableError,
)
from aiperf.config.loader.jinja import (
    build_template_context,
    expand_config_dict,
    render_jinja2_templates,
)

if TYPE_CHECKING:
    from aiperf.config.loader.core import (
        dump_config,
        load_config,
        load_config_dict,
        load_config_from_env,
        load_config_from_string,
        merge_configs,
        save_config,
        validate_config_file,
    )
    from aiperf.config.loader.plan import (
        build_benchmark_plan,
        load_benchmark_plan,
    )

# Defer the heavy `core` and `plan` pulls (which import the
# top-level `aiperf.config.config` Pydantic graph) until first
# attribute access. Importing a sibling loader submodule like
# ``aiperf.config.loader.duration`` from inside the config package's
# own initialization would otherwise trigger a circular import via
# ``loader.core`` -> ``aiperf.config.config`` -> ``aiperf.config.artifacts``.
_LAZY_EXPORTS = {
    "dump_config": "aiperf.config.loader.core",
    "load_config": "aiperf.config.loader.core",
    "load_config_dict": "aiperf.config.loader.core",
    "load_config_from_env": "aiperf.config.loader.core",
    "load_config_from_string": "aiperf.config.loader.core",
    "merge_configs": "aiperf.config.loader.core",
    "save_config": "aiperf.config.loader.core",
    "validate_config_file": "aiperf.config.loader.core",
    "build_benchmark_plan": "aiperf.config.loader.plan",
    "load_benchmark_plan": "aiperf.config.loader.plan",
}


def __getattr__(name: str):  # noqa: D401
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    # Constants
    "ENV_VAR_PATTERN",
    # Exceptions
    "ConfigurationError",
    "MissingEnvironmentVariableError",
    # Core loading functions
    "build_benchmark_plan",
    "load_benchmark_plan",
    "load_config",
    "load_config_from_env",
    "load_config_dict",
    "load_config_from_string",
    "dump_config",
    "save_config",
    "validate_config_file",
    "merge_configs",
    "substitute_env_vars",
    # Jinja2 rendering
    "build_template_context",
    "expand_config_dict",
    "render_jinja2_templates",
]
