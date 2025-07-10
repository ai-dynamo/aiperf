# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base import CaseInsensitiveStrEnum


class SystemState(CaseInsensitiveStrEnum):
    """State of the system as a whole.

    This is used to track the state of the system as a whole, and is used to
    determine what actions to take when a signal is received.
    """

    INITIALIZING = "initializing"
    """The system is initializing. This is the initial state."""

    CONFIGURING = "configuring"
    """The system is configuring services."""

    READY = "ready"
    """The system is ready to start profiling. This is a temporary state that should be
    followed by PROFILING."""

    PROFILING = "profiling"
    """The system is running a profiling run."""

    PROCESSING = "processing"
    """The system is processing results."""

    STOPPING = "stopping"
    """The system is stopping."""

    SHUTDOWN = "shutdown"
    """The system is shutting down. This is the final state."""
