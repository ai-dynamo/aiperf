# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import msgspec

from aiperf.common.enums import (
    LifecycleState,
    ServiceRegistrationStatus,
)
from aiperf.common.models.base_models import PydanticStructMixin
from aiperf.common.types import ServiceTypeT


class ServiceRunInfo(
    PydanticStructMixin,
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
):
    """Tracks a service's registration + lifecycle identity.

    Mutable: the service registry rewrites ``registration_status``,
    ``last_seen_ns``, ``state``, and ``pod_name`` over the service's
    lifetime. No ``frozen``.
    """

    service_type: ServiceTypeT
    registration_status: ServiceRegistrationStatus
    service_id: str
    first_seen_ns: int | None = None
    last_seen_ns: int | None = None
    state: LifecycleState = LifecycleState.CREATED
    pod_name: str | None = None
    pod_index: str | None = None
