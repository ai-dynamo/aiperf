# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class SystemState(CaseInsensitiveStrEnum):
    """State of the system as a whole.

    This is used to track the state of the system as a whole, and is used to
    determine what actions to take when a signal is received.
    """

    CREATED = "created"
    INITIALIZING = "initializing"
    STARTING_SERVICES = "starting_services"
    CONFIGURING_SERVICES = "configuring_services"
    PROFILING = "profiling"
    PROCESSING_RESULTS = "processing_results"
    EXPORTING_DATA = "exporting_data"
    STOPPING = "stopping"
    SHUTDOWN = "shutdown"
