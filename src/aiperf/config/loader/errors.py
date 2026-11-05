# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exception classes for AIPerf configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aiperf.common.redact import redact_text, redact_url


class ConfigurationError(Exception):
    """
    Exception raised for configuration loading errors.

    The message and context are redacted, because they often quote config
    values such as a presigned dataset URL.

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
        self.message = redact_text(message)
        self.file_path = file_path
        self.context = redact_text(context) if context else context

        parts = [self.message]
        if file_path:
            parts.append(f"File: {file_path}")
        if self.context:
            parts.append(f"Context: {self.context}")

        super().__init__("\n".join(parts))


def describe_validation_error(error: ValidationError) -> str:
    """Render a ValidationError without echoing credential-bearing input.

    Pydantic truncates a long ``input_value`` to its first and last characters,
    so the tail of a presigned signature can appear with no parameter name
    beside it for text redaction to find. When any failing input carries a
    credential, the inputs are left out of the message altogether.
    """
    details = error.errors(include_url=False)
    if not any(_carries_credential(detail.get("input")) for detail in details):
        return str(error)
    count = len(details)
    lines = [f"{count} validation error{'s' if count != 1 else ''} for {error.title}"]
    for detail in details:
        lines.append(".".join(str(part) for part in detail["loc"]))
        lines.append(f"  {detail['msg']} [type={detail['type']}]")
    return "\n".join(lines)


def _carries_credential(value: Any) -> bool:
    if isinstance(value, str):
        return redact_url(value) != value
    if isinstance(value, dict):
        return any(
            _carries_credential(k) or _carries_credential(v) for k, v in value.items()
        )
    if isinstance(value, list | tuple | set):
        return any(_carries_credential(item) for item in value)
    return False


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
            context="Use ${VAR:default} syntax to provide a default value",
        )
