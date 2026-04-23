# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - YAML Loader

This module provides functions for loading AIPerf configuration from
YAML files with support for environment variable substitution and
Jinja2 template rendering.

Key Features:
    - YAML file loading with validation
    - Environment variable substitution (${VAR} syntax)
    - Default value support (${VAR:default} syntax)
    - Jinja2 template rendering ({{ expr }} syntax) with self-reference
    - Detailed error messages for configuration issues

Example Usage:
    >>> from aiperf.config import load_config
    >>> config = load_config("benchmark.yaml")
    >>> print(config.models)

    With environment variables:
    >>> # YAML: api_key: ${OPENAI_API_KEY}
    >>> import os
    >>> os.environ["OPENAI_API_KEY"] = "sk-..."
    >>> config = load_config("benchmark.yaml")

    With Jinja2 templates:
    >>> # YAML:
    >>> # variables:
    >>> #   base_concurrency: 16
    >>> # phases:
    >>> #   test:
    >>> #     concurrency: "{{ base_concurrency }}"
    >>> #     requests: "{{ base_concurrency * 100 }}"

Environment Variable Syntax:
    ${VAR}           - Required variable, error if not set
    ${VAR:default}   - Optional with default value
    ${VAR:}          - Optional with empty string default

Jinja2 Template Syntax:
    {{ var }}                    - Variable from 'variables' section
    {{ phases.test.concurrency }} - Self-reference to config values
    {{ var * 2 }}                - Expression evaluation
    {{ var | int }}              - Filter application
"""

from __future__ import annotations

from aiperf.config.loader.core import (
    dump_config,
    load_config,
    load_config_from_env,
    load_config_from_string,
    merge_configs,
    save_config,
    validate_config_file,
)
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
from aiperf.config.loader.plan import (
    build_benchmark_plan,
    load_benchmark_plan,
)

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
