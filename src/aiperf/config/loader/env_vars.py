# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment variable substitution for AIPerf configuration."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from aiperf.config.loader.errors import (
    ConfigurationError,
    MissingEnvironmentVariableError,
)

# Regex pattern for environment variable substitution
# Matches: ${VAR} or ${VAR:default}
ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")
WHOLE_ENV_VAR_PATTERN = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*(?::[^}]*)?\}$")

# Matches an unterminated ${ ... opener (no closing '}' before end-of-string or newline).
# Used to surface ``${UNTERMINATED_VAR_NAME`` as a load-time error rather than
# silently passing the literal through.
_UNTERMINATED_ENV_VAR_PATTERN = re.compile(r"\$\{[^}\n]*$")

# ``$${`` escapes a literal ``${``: the opener is swapped for this sentinel
# BEFORE substitution (so the escaped form is never substituted) and swapped
# back to ``${`` AFTER the unterminated-opener check (so the restored literal
# does not trip that error). NUL bytes cannot appear in YAML scalars, so the
# sentinel cannot collide with user content.
_ESCAPED_OPENER = "$${"
_ESCAPED_OPENER_SENTINEL = "\x00aiperf-escaped-env-opener\x00"


def substitute_env_vars(
    value: Any,
    file_path: Path | str | None = None,
) -> Any:
    """
    Recursively substitute environment variables in configuration values.

    Processes strings, lists, and dictionaries recursively, replacing
    ${VAR} and ${VAR:default} patterns with environment variable values.
    A ``$${`` opener escapes substitution and yields a literal ``${``.

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

        >>> substitute_env_vars("$${MY_VAR}")
        '${MY_VAR}'
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


def _coerce_scalar_string(value: str) -> bool | int | float | str:
    """Coerce env-var-only substitutions using the same scalar rules as Jinja."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _substitute_string(
    text: str,
    file_path: Path | str | None = None,
) -> bool | int | float | str:
    """
    Substitute environment variables in a single string.

    Args:
        text: String containing ${VAR} or ${VAR:default} patterns.
            ``$${`` escapes substitution and renders a literal ``${``.
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

    # Protect ``$${`` escapes so the forward regex cannot substitute the
    # trailing ``${VAR}`` inside them (the escape's whole point).
    protected = text.replace(_ESCAPED_OPENER, _ESCAPED_OPENER_SENTINEL)
    substituted = ENV_VAR_PATTERN.sub(replace_match, protected)
    # After substitution, scan the residue for an unterminated ``${`` opener.
    # The forward regex requires a closing ``}``, so ``${UNTERMINATED_VAR_NAME``
    # would otherwise pass through silently as a literal string. The check
    # runs while escapes are still sentinels, so an escaped literal never
    # trips it.
    if _UNTERMINATED_ENV_VAR_PATTERN.search(substituted):
        shown = substituted.replace(_ESCAPED_OPENER_SENTINEL, _ESCAPED_OPENER)
        raise ConfigurationError(
            f"Unterminated environment variable reference in {shown!r}: "
            f"'${{' opener has no closing '}}'. Either close the brace or "
            f"escape the literal as '$${{'.",
            file_path=file_path,
        )
    substituted = substituted.replace(_ESCAPED_OPENER_SENTINEL, "${")
    if WHOLE_ENV_VAR_PATTERN.fullmatch(text):
        return _coerce_scalar_string(substituted)
    return substituted
