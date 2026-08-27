# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core AIPerf configuration loading functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.config.loader.env_vars import substitute_env_vars
from aiperf.config.loader.errors import ConfigurationError
from aiperf.config.loader.jinja import (
    build_template_context,
    render_jinja2_templates,
)

if TYPE_CHECKING:
    from aiperf.config.config import AIPerfConfig

_logger = AIPerfLogger(__name__)


class _DuplicateKeyError(ValueError):
    """Internal: raised by ``_StrictSafeLoader`` when a mapping has dupes."""


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of last-wins.

    PyYAML's default ``SafeLoader.construct_mapping`` silently overwrites
    earlier values when a key appears more than once in the same mapping.
    Combined with the module-level ``_construct_mapping_no_dupes``
    constructor (registered below), this subclass raises
    ``_DuplicateKeyError`` instead so copy-paste typos surface as a config
    error rather than silent metric drift.
    """


def _construct_mapping_no_dupes(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise _DuplicateKeyError(
                f"Duplicate YAML key {key!r} at line "
                f"{key_node.start_mark.line + 1}, column "
                f"{key_node.start_mark.column + 1}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping_no_dupes
)


def _load_yaml_strict(yaml_content: str) -> Any:
    """``yaml.safe_load`` with duplicate-key detection.

    Cycle detection is handled upstream by trapping ``RecursionError`` at
    the call site; PyYAML resolves cyclic anchor graphs without complaint
    and the recursion explodes only when downstream code walks the result.
    """
    return yaml.load(yaml_content, Loader=_StrictSafeLoader)


# Maximum nesting depth for any value walked during config expansion.
# Exceeding this raises ``ConfigurationError`` instead of ``RecursionError``.
_MAX_CONFIG_NESTING_DEPTH = 256


def _assert_string_keys(data: Any, path: str = "") -> None:
    """Recursively assert all dict keys in ``data`` are strings.

    Raises ``ConfigurationError`` on the first non-string key. YAML allows
    int/float/bool/null keys; AIPerf's path-based renderer assumes strings,
    so non-string keys would otherwise blow up downstream as a bare
    ``AttributeError``.
    """
    if isinstance(data, dict):
        for k, v in data.items():
            if not isinstance(k, str):
                where = f" at {path!r}" if path else " at config root"
                raise ConfigurationError(
                    f"YAML key {k!r} is not a string{where}; only string "
                    "keys are allowed in AIPerf configs."
                )
            _assert_string_keys(v, f"{path}.{k}" if path else k)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            _assert_string_keys(item, f"{path}.{i}" if path else str(i))


# Body fields that must live under `benchmark:` in the envelope shape.
# Includes both snake_case (canonical) and camelCase (legacy YAML alias) and
# the singular ``dataset`` shorthand (auto-promoted to ``datasets: [...]``).
_BODY_KEYS = frozenset(
    {
        "model",
        "models",
        "endpoint",
        "dataset",
        "datasets",
        "phases",
        "artifacts",
        "slos",
        "tokenizer",
        "gpu_telemetry",
        "gpuTelemetry",
        "server_metrics",
        "serverMetrics",
        "runtime",
        "logging",
        "metrics",
        "accuracy",
    }
)


def _auto_migrate_flat_shape(
    data: dict[str, Any], file_path: Path | str | None
) -> None:
    """Wrap pre-restructure flat-shape body keys under ``benchmark:`` in place.

    The schema-2.0 redesign moved body fields (models, endpoint, datasets, ...)
    under a ``benchmark:`` envelope key. Older YAMLs (chaos fixtures, Helm
    templates, user docs) still use the flat shape. To preserve cyclopts-era
    parity, auto-migrate at load time and emit a one-line deprecation warning
    pointing to the migrate-config tutorial. Idempotent: a no-op when the
    envelope shape is already in use.

    The migration covers the body-keys-to-benchmark wrap plus the singular
    ``dataset:`` -> ``datasets: [...]`` promotion. Inner schema migrations
    (e.g. ``isl: 16`` -> ``isl: {mean: 16}``) are not handled here - point
    users at ``tools/migrate_config_yaml.py`` for those.
    """
    body_present = sorted(_BODY_KEYS & set(data.keys()))
    if not body_present:
        return
    body: dict[str, Any] = {}
    for k in body_present:
        value = data.pop(k)
        if k == "dataset":
            # Singular form: promote to a single-element ``datasets`` list.
            entry = dict(value) if isinstance(value, dict) else value
            if isinstance(entry, dict) and "name" not in entry:
                entry.setdefault("name", "main")
            body.setdefault(
                "datasets", [entry] if not isinstance(entry, list) else entry
            )
        else:
            body[k] = value
    if "benchmark" in data and isinstance(data["benchmark"], dict):
        for k, v in body.items():
            data["benchmark"].setdefault(k, v)
    else:
        data["benchmark"] = body
    fp = file_path or "<stdin>"
    migrate_target = file_path or "<path>"
    _logger.warning(
        f"Config {fp} uses pre-restructure flat shape (top-level keys: {body_present}); "
        f"auto-migrated to envelope shape under `benchmark:`. Run "
        f"`uv run python tools/migrate_config_yaml.py {migrate_target} --in-place` to "
        f"make the change permanent. See docs/tutorials/migrating-config.md."
    )


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

        >>> print(config.phases[0].name)
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


def _parse_yaml_mapping(
    yaml_content: str,
    file_path: Path | str | None,
) -> dict[str, Any]:
    """Parse a YAML string and ensure it represents a mapping."""
    try:
        data = _load_yaml_strict(yaml_content)
    except _DuplicateKeyError as e:
        raise ConfigurationError(
            f"Duplicate key in configuration: {e}",
            file_path=file_path,
        ) from e
    except RecursionError as e:
        raise ConfigurationError(
            "YAML parser exceeded recursion limit (cyclic aliases or "
            "excessive nesting).",
            file_path=file_path,
        ) from e
    except yaml.constructor.ConstructorError as e:
        # PyYAML's cycle/recursion detection raises ConstructorError with
        # version-dependent wording (e.g. "found unconstructable recursive
        # node" on 6.0.x, plain "recursion" elsewhere). Catching the type
        # directly avoids relying on the exception's text. Cycles that slip
        # past PyYAML are caught downstream by ``_detect_cycles_or_depth``.
        raise ConfigurationError(
            "Cyclic YAML aliases are not supported.",
            file_path=file_path,
        ) from e
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

    try:
        _assert_string_keys(data)
    except ConfigurationError as e:
        # Preserve our string-key error but attach the file path.
        raise ConfigurationError(
            e.message, file_path=file_path, context=e.context
        ) from e

    return data


def _push_container_children(
    node: dict | list,
    depth: int,
    *,
    seen: set[int],
    stack: list[tuple[Any, int]],
    file_path: Path | str | None,
) -> None:
    """Mark a dict/list node as visited and enqueue its container children.

    Raises ``ConfigurationError`` if ``node`` has already been seen (cyclic
    YAML alias). Scalars are skipped; only nested dicts/lists need traversal.
    """
    if id(node) in seen:
        raise ConfigurationError(
            "Cyclic YAML aliases are not supported.",
            file_path=file_path,
        )
    seen.add(id(node))
    children = node.values() if isinstance(node, dict) else node
    for v in children:
        if isinstance(v, (dict, list)):
            stack.append((v, depth + 1))


def _detect_cycles_or_depth(
    data: Any,
    file_path: Path | str | None,
    *,
    limit: int = _MAX_CONFIG_NESTING_DEPTH,
) -> None:
    """Walk ``data`` iteratively to surface cyclic aliases or excess depth.

    Raises ``ConfigurationError`` with a clear message on either condition,
    rather than letting the recursive Jinja walker explode with a bare
    ``RecursionError``. Identity-tracked via ``id(node)`` since cyclic YAML
    aliases produce graphs where the same dict/list object appears twice.
    """
    seen: set[int] = set()
    # Each stack entry: (node, current_depth)
    stack: list[tuple[Any, int]] = [(data, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > limit:
            raise ConfigurationError(
                f"Configuration nested too deeply (>{limit} levels). "
                "If this is intentional, please file an issue.",
                file_path=file_path,
            )
        if isinstance(node, (dict, list)):
            _push_container_children(
                node, depth, seen=seen, stack=stack, file_path=file_path
            )


def _expand_with_recursion_guard(
    data: dict[str, Any],
    file_path: Path | str | None,
    *,
    substitute_env: bool,
) -> dict[str, Any]:
    """Run env-var + Jinja2 expansion via ``_expand_capture_pre_jinja``.

    Cyclic aliases and pathologically deep configs both manifest as
    ``RecursionError`` from the recursive renderer. ``_detect_cycles_or_depth``
    catches both with a precise message before that happens; the
    ``RecursionError`` -> ``ConfigurationError`` trap now lives in
    ``_expand_capture_pre_jinja`` (which this function delegates to) as a
    defense in depth in case some future structure slips past the explicit
    check.
    """
    expanded, _raw = _expand_capture_pre_jinja(
        data, file_path, substitute_env=substitute_env
    )
    return expanded


def _expand_capture_pre_jinja(
    data: dict[str, Any],
    file_path: Path | str | None,
    *,
    substitute_env: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Variant of ``_expand_with_recursion_guard`` that also returns the
    pre-Jinja, post-env-var dict.

    Sweep expansion needs templates intact so per-variation re-rendering can
    pick up swept ``variables.*`` overrides. Returning the pre-Jinja dict
    alongside the post-Jinja dict lets the loader stash it on the resolved
    ``AIPerfConfig`` for ``build_benchmark_plan`` to consume. The post-Jinja
    dict is what feeds Pydantic validation (templates would fail int/float
    field validation), so both are needed.
    """
    import copy as _copy

    _detect_cycles_or_depth(data, file_path)
    try:
        if substitute_env:
            data = substitute_env_vars(data, file_path)
        # Snapshot AFTER env-var substitution so per-variation re-rendering
        # sees the same env-resolved values, but BEFORE Jinja so `{{ var }}`
        # references still exist verbatim.
        pre_jinja = _copy.deepcopy(data)
        context = build_template_context(data)
        return render_jinja2_templates(data, context), pre_jinja
    except RecursionError as e:
        raise ConfigurationError(
            "Configuration expansion exceeded recursion limit (cyclic "
            "aliases or excessive nesting).",
            file_path=file_path,
        ) from e


def _validate_config_dict(
    data: dict[str, Any],
    file_path: Path | str | None,
) -> AIPerfConfig:
    """Validate a fully-expanded config dict into an AIPerfConfig.

    Threads ``source_dir`` (parent of ``file_path``, when present) into the
    Pydantic validation context so envelope-level pre-validators can resolve
    sibling-file references (e.g. the bare-string ``plot:`` form).
    """
    source_dir: Path | None = None
    if file_path is not None:
        source_dir = Path(file_path).resolve().parent
    from aiperf.config.config import AIPerfConfig

    try:
        return AIPerfConfig.model_validate(data, context={"source_dir": source_dir})
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        if file_path:
            raise ConfigurationError(
                f"Configuration validation failed: {e}",
                file_path=file_path,
            ) from e
        raise


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
        ... benchmark:
        ...   models:
        ...     - llama-3-8b
        ...   endpoint:
        ...     urls: ["http://localhost:8000/v1/chat/completions"]
        ...   datasets:
        ...     - name: main
        ...       type: synthetic
        ...       entries: 100
        ...   phases:
        ...     - name: profiling
        ...       type: concurrency
        ...       duration: 10
        ...       concurrency: 1
        ... '''
        >>> config = load_config_from_string(yaml_str)
    """
    data = _parse_yaml_mapping(yaml_content, file_path)
    _auto_migrate_flat_shape(data, file_path)

    expanded, pre_jinja = _expand_capture_pre_jinja(
        data, file_path, substitute_env=substitute_env
    )
    config = _validate_config_dict(expanded, file_path)
    # Stash the pre-Jinja envelope so build_benchmark_plan can re-render
    # `{{ var }}` body fields against each variation's variables block.
    config._raw_envelope = pre_jinja
    return config


def load_config_dict(
    file_path: Path | str,
    *,
    substitute_env: bool = True,
) -> dict[str, Any]:
    """Load a YAML config file and return the expanded dict, skipping validation.

    Returns the post-env-var, post-Jinja2 dict so callers can deep-merge CLI
    overrides on top before running ``AIPerfConfig.model_validate``. Used by
    ``resolve_config`` when the user combines ``--config <yaml>`` with CLI
    flags.

    Raises ``ConfigurationError`` if the file is missing/unreadable; YAML/
    Jinja errors propagate from the underlying loaders.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {file_path}", file_path=file_path
        )
    if not file_path.is_file():
        raise ConfigurationError(
            f"Path is not a file: {file_path}", file_path=file_path
        )
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigurationError(
            f"Failed to read configuration file: {e}", file_path=file_path
        ) from e

    data = _parse_yaml_mapping(content, file_path)
    _auto_migrate_flat_shape(data, file_path)
    return _expand_with_recursion_guard(data, file_path, substitute_env=substitute_env)


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
    # The sweep block's `type:` field is a discriminator with a default
    # ("grid"/"scenarios"/"adaptive_search"). With `exclude_defaults=True`
    # Pydantic strips it, but the discriminated union on reload requires
    # it. Force-inject it so dump -> reload round-trips for every sweep
    # template.
    if config.sweep is not None and isinstance(data.get("sweep"), dict):
        data["sweep"].setdefault("type", config.sweep.type)
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


def _validate_sweep_expansion(config: AIPerfConfig, file_path: Path | str) -> None:
    """Exercise sweep expansion so path/shape errors surface at validate time.

    ``aiperf profile`` runs the same pipeline; doing it here gives the cheap
    pre-flight check parity. ``pydantic.ValidationError`` (extra-field rejects
    from re-validating the post-expansion benchmark) and ``ValueError`` (from
    ``_set_nested_value`` on an unresolvable path) both get rewrapped as
    ``ConfigurationError`` so the CLI's uniform "Error: <msg>" handler fires
    instead of a bare traceback.
    """
    from pydantic import ValidationError

    from aiperf.config.loader.plan import build_benchmark_plan

    try:
        build_benchmark_plan(config)
    except ConfigurationError:
        raise
    except (ValidationError, ValueError, TypeError) as e:
        raise ConfigurationError(
            f"Sweep expansion failed during validation: {e}",
            file_path=file_path,
        ) from e


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

    if config.sweep is not None:
        _validate_sweep_expansion(config, file_path)

    # Check for potential issues

    # Warn if streaming disabled but TTFT goodput set, and reject unknown
    # SLO metric names against the metric registry. SLOsConfig is a plain
    # `dict[str, float]` (see slos.SLOsConfig), so keys must be looked
    # up dict-style — never as attributes.
    bench = config.benchmark
    if bench.slos:
        from aiperf.metrics.metric_registry import MetricRegistry

        unknown = [
            tag for tag in bench.slos if MetricRegistry.get_class_or_none(tag) is None
        ]
        if unknown:
            known = sorted(MetricRegistry.all_tags())
            raise ConfigurationError(
                f"Unknown SLO metric(s): {sorted(unknown)}. "
                f"SLO keys must match a registered metric tag. "
                f"Known tags: {known}"
            )

        if not bench.endpoint.streaming:
            if "time_to_first_token" in bench.slos:
                warnings.append(
                    "slos.time_to_first_token is set but streaming is disabled. "
                    "TTFT measurement requires streaming=true."
                )
            if "inter_token_latency" in bench.slos:
                warnings.append(
                    "slos.inter_token_latency is set but streaming is disabled. "
                    "ITL measurement requires streaming=true."
                )

    # Warn if prefill_concurrency set without streaming
    for phase in bench.phases:
        if phase.prefill_concurrency and not bench.endpoint.streaming:
            warnings.append(
                f"Load config '{phase.name}' has prefill_concurrency set but "
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
    import os

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
        from aiperf.config.config import AIPerfConfig

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
    from aiperf.config.config import AIPerfConfig

    return AIPerfConfig.model_validate(merged)
