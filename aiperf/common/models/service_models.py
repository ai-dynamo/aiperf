# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from pydantic import BaseModel, Field

from aiperf.common.enums import (
    ServiceRegistrationStatus,
    ServiceState,
    ServiceType,
)


class ServiceRunInfo(BaseModel):
    """Base model for tracking service run information."""

    service_type: ServiceType = Field(
        ...,
        description="The type of service",
    )
    registration_status: ServiceRegistrationStatus = Field(
        ...,
        description="The registration status of the service",
    )
    service_id: str = Field(
        ...,
        description="The ID of the service",
    )
    first_seen: int | None = Field(
        default_factory=time.time_ns,
        description="The first time the service was seen",
    )
    last_seen: int | None = Field(
        default_factory=time.time_ns,
        description="The last time the service was seen",
    )
    state: ServiceState = Field(
        default=ServiceState.UNKNOWN,
        description="The current state of the service",
    )
