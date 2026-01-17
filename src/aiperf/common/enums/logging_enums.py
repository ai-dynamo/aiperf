# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class AIPerfLogLevel(CaseInsensitiveStrEnum):
    """Logging levels for AIPerf output verbosity."""

    TRACE = "TRACE"
    """Most verbose. Logs all operations including ZMQ messages and internal state changes."""

    DEBUG = "DEBUG"
    """Detailed debugging information. Logs function calls and important state transitions."""

    INFO = "INFO"
    """General informational messages. Default level showing benchmark progress and results."""

    NOTICE = "NOTICE"
    """Important informational messages that are more significant than INFO but not warnings."""

    WARNING = "WARNING"
    """Warning messages for potentially problematic situations that don't prevent execution."""

    SUCCESS = "SUCCESS"
    """Success messages for completed operations and milestones."""

    ERROR = "ERROR"
    """Error messages for failures that prevent specific operations but allow continued execution."""

    CRITICAL = "CRITICAL"
    """Critical errors that may cause the benchmark to fail or produce invalid results."""
