# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.aiperf_logger import (
    _CRITICAL,
    _DEBUG,
    _ERROR,
    _INFO,
    _NOTICE,
    _SUCCESS,
    _TRACE,
    _WARNING,
)
from aiperf.common.enums.base import CaseInsensitiveStrEnum


class AIPerfLogLevel(CaseInsensitiveStrEnum):
    """Log levels for AIPerfLogger."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    NOTICE = "NOTICE"
    WARNING = "WARNING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @property
    def level(self) -> int:
        """Get the integer level equivalent."""
        return _LEVEL_MAP[self]


# Map the enum values to the integer levels
_LEVEL_MAP = {
    AIPerfLogLevel.TRACE: _TRACE,
    AIPerfLogLevel.DEBUG: _DEBUG,
    AIPerfLogLevel.INFO: _INFO,
    AIPerfLogLevel.NOTICE: _NOTICE,
    AIPerfLogLevel.WARNING: _WARNING,
    AIPerfLogLevel.SUCCESS: _SUCCESS,
    AIPerfLogLevel.ERROR: _ERROR,
    AIPerfLogLevel.CRITICAL: _CRITICAL,
}
