# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging configuration models.

Split out of ``runtime.py`` so each config section lives in its own file.
Re-exported via :mod:`aiperf.config`.

Note: this module is named ``logging`` to mirror the YAML section name. It
does not shadow the stdlib ``logging`` module for any caller using absolute
imports (``import logging`` resolves to the stdlib; ``aiperf.config.logging``
is fully qualified). Inside this file we avoid importing the stdlib module
entirely to dodge the ambiguity.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field

from aiperf.common.enums import AIPerfLogLevel
from aiperf.config.base import BaseConfig


class LoggingConfig(BaseConfig):
    """Logging configuration for verbosity and debug settings."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    level: Annotated[
        AIPerfLogLevel,
        Field(
            default=AIPerfLogLevel.INFO,
            description="Global logging verbosity level (loguru-style severity ladder; "
            "TRACE most verbose, CRITICAL least). NOTICE and SUCCESS are loguru-specific "
            "intermediate levels (between INFO and WARNING) and have no equivalent in "
            "the Python stdlib `logging` module — pick INFO/WARNING/ERROR if you want "
            "stdlib-compatible severities.",
        ),
    ]


__all__ = [
    "LoggingConfig",
]
