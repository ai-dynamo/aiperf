# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class CommandType(CaseInsensitiveStrEnum):
    REALTIME_METRICS = "realtime_metrics"
    PROCESS_RECORDS = "process_records"
    PROFILE_CANCEL = "profile_cancel"
    PROFILE_CONFIGURE = "profile_configure"
    PROFILE_START = "profile_start"
    REGISTER_SERVICE = "register_service"
    SHUTDOWN = "shutdown"
    SHUTDOWN_WORKERS = "shutdown_workers"
    SPAWN_WORKERS = "spawn_workers"


class CommandResponseStatus(CaseInsensitiveStrEnum):
    ACKNOWLEDGED = "acknowledged"
    FAILURE = "failure"
    SUCCESS = "success"
    UNHANDLED = "unhandled"  # The command was received but not handled by any hook
