# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from pydantic import Field

from aiperf.common.enums import (
    LifecycleState,
    ServiceRegistrationStatus,
)
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.types import ServiceTypeT


class ServiceInfo(AIPerfBaseModel):
    service_id: str = Field(..., description="The ID of the service")
    service_type: ServiceTypeT = Field(..., description="The type of the service")
    state: LifecycleState = Field(..., description="The current state of the service")


class ServiceRunInfo(ServiceInfo):
    """Base model for tracking service run information."""

    registration_status: ServiceRegistrationStatus = Field(
        ...,
        description="The registration status of the service",
    )
    first_seen: int | None = Field(
        default_factory=time.time_ns,
        description="The first time the service was seen",
    )
    last_seen: int | None = Field(
        default_factory=time.time_ns,
        description="The last time the service was seen",
    )
