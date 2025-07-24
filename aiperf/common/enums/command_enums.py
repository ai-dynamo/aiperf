# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class CommandType(CaseInsensitiveStrEnum):
    SHUTDOWN = "shutdown"
    PROCESS_RECORDS = "process_records"
    CONFIGURE_PROFILING = "configure_profiling"
    START_PROFILING = "start_profiling"
    SPAWN_WORKERS = "spawn_workers"
    SHUTDOWN_WORKERS = "shutdown_workers"


class CommandResponseStatus(CaseInsensitiveStrEnum):
    ACKNOWLEDGED = "acknowledged"
    SUCCESS = "success"
    FAILURE = "failure"
