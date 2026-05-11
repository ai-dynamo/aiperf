# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exception classes for AIPerf configuration loading."""

from __future__ import annotations

from pathlib import Path


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
            context="Use ${VAR:default} syntax to provide a default value",
        )
