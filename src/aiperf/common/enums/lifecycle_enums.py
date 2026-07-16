# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum
from aiperf.common.enums.enums import LifecycleState as LifecycleState
from aiperf.common.enums.enums import (
    ServiceRegistrationStatus as ServiceRegistrationStatus,
)
from aiperf.common.enums.enums import SystemState as SystemState
from aiperf.common.enums.enums import WorkerStatus as WorkerStatus


class WorkerStartupState(CaseInsensitiveStrEnum):
    """The current startup lifecycle state of a worker service."""

    STARTING = "starting"
    WAITING_FOR_DATASET = "waiting_for_dataset"
    ROUTER_PROBING = "router_probing"
    READY = "ready"
    SHUTTING_DOWN = "shutting_down"
