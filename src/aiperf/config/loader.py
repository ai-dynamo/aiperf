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

import os
import re
from pathlib import Path
from typing import Any

import jinja2
import yaml

from aiperf.config.benchmark import BenchmarkPlan
from aiperf.config.config import AIPerfConfig, BenchmarkConfig

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

# Regex pattern for environment variable substitution
# Matches: ${VAR} or ${VAR:default}
ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")

# Fields to skip when rendering Jinja2 templates (they contain Jinja2 templates themselves
# that are rendered at request time by the template endpoint, not at config load time)
SKIP_TEMPLATE_FIELDS = {"template", "body", "payload_template"}


class ConfigurationError(Exception):
    """
    Exception raised for configuration loading errors.

    Attributes:
        message: Human-readable error description.
        file_path: Path to the configuration file (if applicable).
        context: Additional context about the error.
    """

    def __init__(
        self,
        message: str,
        file_path: Path | str | None = None,
        context: str | None = None,
    ):
        self.message = message
        self.file_path = file_path
        self.context = context

        parts = [message]
        if file_path:
            parts.append(f"File: {file_path}")
        if context:
            parts.append(f"Context: {context}")

        super().__init__("\n".join(parts))


class MissingEnvironmentVariableError(ConfigurationError):
    """
    Exception raised when a required environment variable is not set.

    Attributes:
        variable_name: Name of the missing variable.
        file_path: Path to the configuration file.
    """

    def __init__(
        self,
        variable_name: str,
        file_path: Path | str | None = None,
    ):
        self.variable_name = variable_name
        super().__init__(
            f"Required environment variable '{variable_name}' is not set",
            file_path=file_path,
            context="Use ${{VAR:default}} syntax to provide a default value",
        )


def substitute_env_vars(
    value: Any,
    file_path: Path | str | None = None,
) -> Any:
    """
    Recursively substitute environment variables in configuration values.

    Processes strings, lists, and dictionaries recursively, replacing
    ${VAR} and ${VAR:default} patterns with environment variable values.

    Args:
        value: Configuration value to process. Can be string, list, dict,
            or any other type (non-string/list/dict types pass through unchanged).
        file_path: Path to config file for error messages.

    Returns:
        Value with environment variables substituted.

    Raises:
        MissingEnvironmentVariableError: If a required variable (no default)
            is not set in the environment.

    Examples:
        >>> os.environ["MY_VAR"] = "hello"
        >>> substitute_env_vars("${MY_VAR}")
        'hello'

        >>> substitute_env_vars("${UNSET:default_value}")
        'default_value'

        >>> substitute_env_vars({"key": "${MY_VAR}"})
        {'key': 'hello'}

        >>> substitute_env_vars(["${MY_VAR}", "static"])
        ['hello', 'static']
    """
    if isinstance(value, str):
        return _substitute_string(value, file_path)
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v, file_path) for k, v in value.items()}
    elif isinstance(value, list):
        return [substitute_env_vars(item, file_path) for item in value]
    else:
        # Pass through non-string/dict/list values unchanged
        return value


def _substitute_string(
    text: str,
    file_path: Path | str | None = None,
) -> str:
    """
    Substitute environment variables in a single string.

    Args:
        text: String containing ${VAR} or ${VAR:default} patterns.
        file_path: Path to config file for error messages.

    Returns:
        String with variables substituted.

    Raises:
        MissingEnvironmentVariableError: If required variable not set.
    """

    def replace_match(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)

        env_value = os.environ.get(var_name)

        if env_value is not None:
            return env_value
        elif default is not None:
            # Default was specified (even if empty string)
            return default
        else:
            # No value and no default - error
            raise MissingEnvironmentVariableError(var_name, file_path)

    return ENV_VAR_PATTERN.sub(replace_match, text)


# =============================================================================
# JINJA2 TEMPLATE RENDERING
# =============================================================================


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
        field_name = current_path.split(".")[-1] if current_path else ""
        if field_name in SKIP_TEMPLATE_FIELDS:
            return data

        if "{{" in data and "}}" in data:
            try:
                template = jinja2.Template(data)
                rendered = template.render(**context)

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
            except jinja2.TemplateError as e:
                raise ConfigurationError(
                    f"Jinja2 template error at '{current_path}': {e}",
                    context=f"Template: {data}",
                ) from e
        return data

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


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================


def load_config(
    file_path: Path | str,
    *,
    substitute_env: bool = True,
) -> AIPerfConfig:
    """
    Load and validate AIPerf configuration from a YAML file.

    This is the primary function for loading configuration files. It reads
    the YAML file, optionally substitutes environment variables, and
    validates the configuration against the schema.

    Args:
        file_path: Path to the YAML configuration file.
        substitute_env: Whether to process environment variable substitution.
            Defaults to True.

    Returns:
        Validated AIPerfConfig object.

    Raises:
        ConfigurationError: If the file cannot be read or parsed.
        MissingEnvironmentVariableError: If a required env var is missing.
        pydantic.ValidationError: If the configuration fails validation.

    Example:
        >>> config = load_config("benchmark.yaml")
        >>> print(config.models)
        ['meta-llama/Llama-3.1-8B-Instruct']

        >>> print(list(config.phases.keys())[0])
        'warmup'
    """
    file_path = Path(file_path)

    # Check file exists
    if not file_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {file_path}",
            file_path=file_path,
        )

    if not file_path.is_file():
        raise ConfigurationError(
            f"Path is not a file: {file_path}",
            file_path=file_path,
        )

    # Read file contents
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigurationError(
            f"Failed to read configuration file: {e}",
            file_path=file_path,
        ) from e

    # Load and validate
    return load_config_from_string(
        content,
        file_path=file_path,
        substitute_env=substitute_env,
    )


def load_config_from_string(
    yaml_content: str,
    *,
    file_path: Path | str | None = None,
    substitute_env: bool = True,
) -> AIPerfConfig:
    """
    Load and validate AIPerf configuration from a YAML string.

    Useful for programmatic configuration or testing without files.

    Args:
        yaml_content: YAML configuration as a string.
        file_path: Optional file path for error messages.
        substitute_env: Whether to process environment variable substitution.

    Returns:
        Validated AIPerfConfig object.

    Raises:
        ConfigurationError: If the YAML cannot be parsed.
        MissingEnvironmentVariableError: If a required env var is missing.
        pydantic.ValidationError: If the configuration fails validation.

    Example:
        >>> yaml_str = '''
        ... models:
        ...   - llama-3-8b
        ... endpoint:
        ...   urls: ["http://localhost:8000/v1/chat/completions"]
        ... datasets:
        ...   main:
        ...     type: synthetic
        ...     entries: 100
        ... phases:
        ...   default:
        ...     type: concurrency
        ...     dataset: main
        ...     duration: 10
        ...     concurrency: 1
        ... '''
        >>> config = load_config_from_string(yaml_str)
    """
    # Parse YAML
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Invalid YAML syntax: {e}",
            file_path=file_path,
        ) from e

    if data is None:
        raise ConfigurationError(
            "Configuration file is empty",
            file_path=file_path,
        )

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Configuration must be a YAML mapping, got {type(data).__name__}",
            file_path=file_path,
        )

    # Substitute environment variables
    if substitute_env:
        data = substitute_env_vars(data, file_path)

    # Render Jinja2 templates (after env vars, before validation)
    context = build_template_context(data)
    data = render_jinja2_templates(data, context)
    data.pop("variables", None)

    # Validate and create config
    try:
        return AIPerfConfig.model_validate(data)
    except Exception as e:
        # Re-raise with file context if not already a ConfigurationError
        if isinstance(e, ConfigurationError):
            raise
        # Pydantic ValidationError has good messages, just add file context
        if file_path:
            raise ConfigurationError(
                f"Configuration validation failed: {e}",
                file_path=file_path,
            ) from e
        raise


def dump_config(
    config: AIPerfConfig,
    *,
    exclude_defaults: bool = True,
    exclude_none: bool = True,
) -> str:
    """
    Dump an AIPerfConfig object to YAML string.

    Useful for generating configuration templates or debugging.

    Args:
        config: The configuration object to dump.
        exclude_defaults: Exclude fields that have default values.
        exclude_none: Exclude fields that are None.

    Returns:
        YAML string representation of the configuration.

    Example:
        >>> config = AIPerfConfig(...)
        >>> print(dump_config(config))
    """
    data = config.model_dump(
        exclude_defaults=exclude_defaults,
        exclude_none=exclude_none,
        by_alias=True,
        mode="json",  # Use JSON-compatible types
    )
    return yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


def save_config(
    config: AIPerfConfig,
    file_path: Path | str,
    *,
    exclude_defaults: bool = True,
    exclude_none: bool = True,
) -> None:
    """
    Save an AIPerfConfig object to a YAML file.

    Args:
        config: The configuration object to save.
        file_path: Path to the output YAML file.
        exclude_defaults: Exclude fields that have default values.
        exclude_none: Exclude fields that are None.

    Raises:
        OSError: If the file cannot be written.

    Example:
        >>> config = AIPerfConfig(...)
        >>> save_config(config, "output.yaml")
    """
    file_path = Path(file_path)
    yaml_content = dump_config(
        config,
        exclude_defaults=exclude_defaults,
        exclude_none=exclude_none,
    )
    file_path.write_text(yaml_content, encoding="utf-8")


def validate_config_file(file_path: Path | str) -> list[str]:
    """
    Validate a configuration file and return any warnings.

    Unlike load_config, this function collects warnings rather than
    raising exceptions immediately, making it useful for linting.

    Args:
        file_path: Path to the YAML configuration file.

    Returns:
        List of warning messages (empty if no issues).

    Raises:
        ConfigurationError: If the file has fatal errors.

    Example:
        >>> warnings = validate_config_file("benchmark.yaml")
        >>> for w in warnings:
        ...     print(f"Warning: {w}")
    """
    warnings: list[str] = []

    # Load the config (will raise on fatal errors)
    config = load_config(file_path)

    # Check for potential issues

    # Warn if no profiling load configs
    profiling_phases = config.get_profiling_phases()
    if not profiling_phases:
        warnings.append(
            "All phases have exclude_from_results=True. Final results will be empty."
        )

    # Warn if streaming disabled but TTFT goodput set
    if config.slos:
        if config.slos.time_to_first_token and not config.endpoint.streaming:
            warnings.append(
                "slos.time_to_first_token is set but streaming is disabled. "
                "TTFT measurement requires streaming=true."
            )
        if config.slos.inter_token_latency and not config.endpoint.streaming:
            warnings.append(
                "slos.inter_token_latency is set but streaming is disabled. "
                "ITL measurement requires streaming=true."
            )

    # Warn if prefill_concurrency set without streaming
    for name, phase in config.phases.items():
        if phase.prefill_concurrency and not config.endpoint.streaming:
            warnings.append(
                f"Load config '{name}' has prefill_concurrency set but "
                "streaming is disabled. Prefill concurrency requires streaming=true."
            )

    return warnings


def load_config_from_env() -> AIPerfConfig:
    """
    Load AIPerf configuration from environment variables.

    This function is used by child processes to deserialize the configuration
    that was passed from the parent process via environment variables.

    The configuration is expected to be serialized as JSON in the
    AIPERF_CONFIG environment variable.

    Returns:
        AIPerfConfig object.

    Raises:
        ConfigurationError: If the config cannot be loaded from environment.

    Example:
        >>> # In parent process:
        >>> os.environ["AIPERF_CONFIG"] = config.model_dump_json()
        >>>
        >>> # In child process:
        >>> config = load_config_from_env()
    """
    import orjson

    config_json = os.environ.get("AIPERF_CONFIG")
    if config_json is None:
        raise ConfigurationError(
            "AIPERF_CONFIG environment variable not set. "
            "This function is meant to be called from child processes "
            "that receive configuration from the parent process."
        )

    try:
        data = orjson.loads(config_json)
        return AIPerfConfig.model_validate(data)
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load configuration from environment: {e}"
        ) from e


def merge_configs(
    base: AIPerfConfig,
    override: dict[str, Any],
) -> AIPerfConfig:
    """
    Merge override values into a base configuration.

    Useful for applying CLI overrides to a file-based configuration.

    Args:
        base: The base configuration.
        override: Dictionary of override values.

    Returns:
        New AIPerfConfig with merged values.

    Example:
        >>> config = load_config("benchmark.yaml")
        >>> config = merge_configs(config, {"random_seed": 123})
    """
    base_dict = base.model_dump(exclude_none=True)

    def deep_merge(base_dict: dict, override_dict: dict) -> dict:
        """Recursively merge override into base."""
        result = base_dict.copy()
        for key, value in override_dict.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    merged = deep_merge(base_dict, override)
    return AIPerfConfig.model_validate(merged)


def build_benchmark_plan(config: AIPerfConfig) -> BenchmarkPlan:
    """Build a BenchmarkPlan from a validated AIPerfConfig.

    Expands sweep variations and extracts multi_run settings.
    If no sweep, returns a plan with a single config.

    Args:
        config: Validated AIPerfConfig (may contain sweep + multi_run).

    Returns:
        BenchmarkPlan with expanded configs and execution preferences.
    """
    from aiperf.config.sweep import SweepVariation, expand_sweep

    # Dump to dict, excluding sweep and multi_run (those are plan-level).
    # exclude_none/exclude_unset as safety net: annotated_types (Ge/Gt/Le/Lt)
    # handles None natively, but these flags protect against any future Field(gt=)
    # regressions that would break round-trip validation.
    config_dict = config.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    sweep_dict = config_dict.pop("sweep", None)
    multi_run = config_dict.pop("multi_run", {})

    # Re-inject sweep for expand_sweep to process
    if sweep_dict is not None:
        config_dict["sweep"] = sweep_dict

    # Expand sweep variations
    expanded = expand_sweep(config_dict)

    configs = []
    variations = []
    for variation_dict, variation_meta in expanded:
        variation_dict.pop("sweep", None)
        variation_dict.pop("multi_run", None)

        # Re-render Jinja2 for this variation so sweep-overridden values
        # propagate to any templates that reference them
        context = build_template_context(variation_dict)
        variation_dict = render_jinja2_templates(variation_dict, context)
        variation_dict.pop("variables", None)

        benchmark_config = BenchmarkConfig.model_validate(variation_dict)
        configs.append(benchmark_config)
        variations.append(variation_meta)

    # If no sweep produced variations, ensure we have a default variation
    if not variations:
        variations = [SweepVariation(index=0, label="base", values={})]

    plan_kwargs: dict[str, Any] = dict(
        configs=configs,
        variations=variations,
        trials=multi_run.get("num_runs", 1),
        cooldown_seconds=multi_run.get("cooldown_seconds", 0.0),
        confidence_level=multi_run.get("confidence_level", 0.95),
        set_consistent_seed=multi_run.get("set_consistent_seed", True),
        disable_warmup_after_first=multi_run.get("disable_warmup_after_first", True),
    )
    for key in (
        "convergence_metric",
        "convergence_mode",
        "convergence_threshold",
        "convergence_stat",
    ):
        if key in multi_run and multi_run[key] is not None:
            plan_kwargs[key] = multi_run[key]
    return BenchmarkPlan(**plan_kwargs)


def load_benchmark_plan(
    file_path: Path | str,
    *,
    substitute_env: bool = True,
) -> BenchmarkPlan:
    """Load a YAML config file and return a BenchmarkPlan.

    This is the new primary entry point for the orchestrator.
    Parses YAML -> AIPerfConfig -> expands sweep -> BenchmarkPlan.

    Args:
        file_path: Path to the YAML configuration file.
        substitute_env: Whether to process environment variable substitution.

    Returns:
        BenchmarkPlan with expanded configs and execution preferences.
    """
    config = load_config(file_path, substitute_env=substitute_env)
    return build_benchmark_plan(config)
