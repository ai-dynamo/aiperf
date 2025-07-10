# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base import CaseInsensitiveStrEnum


class CommandType(CaseInsensitiveStrEnum):
    """List of commands that the SystemController can send to component services."""

    SHUTDOWN = "shutdown"
    """A command sent to shutdown a service. This will stop the service gracefully
    no matter what state it is in."""

    PROCESS_RECORDS = "process_records"
    """A command sent to process records. This will process the records and return
    the services to their pre-record processing state."""

    PROFILE_CONFIGURE = "profile_configure"
    """A command sent to configure a service in preparation for a profile run. This will
    override the current configuration."""

    PROFILE_START = "profile_start"
    """A command sent to indicate that a service should begin profiling using the
    current configuration."""

    PROFILE_STOP = "profile_stop"
    """A command sent to indicate that a service should stop doing profile related
    work, as the profile run is complete."""

    PROFILE_CANCEL = "profile_cancel"
    """A command sent to cancel a profile run. This will stop the current profile run and
    process the partial results."""


class CommandResponseStatus(CaseInsensitiveStrEnum):
    """Status of a command response."""

    SUCCESS = "success"
    FAILURE = "failure"
