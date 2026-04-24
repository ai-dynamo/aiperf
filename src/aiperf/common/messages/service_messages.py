# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

import msgspec

from aiperf.common.enums import LifecycleState, MessageType
from aiperf.common.memory_tracker import MemoryPhase
from aiperf.common.messages.base_messages import Message
from aiperf.common.models.error_models import ErrorDetails


class BaseServiceMessage(Message, kw_only=True, omit_defaults=True):
    """Any message originating from a specific service; requires ``service_id``."""

    service_id: str


class BaseStatusMessage(BaseServiceMessage, kw_only=True, omit_defaults=True):
    """Lifecycle status message — ``request_ns`` defaults to ``time.time_ns``."""

    state: LifecycleState
    # ServiceType is an ExtensibleStrEnum with a custom metaclass; msgspec
    # cannot build a decoder for ``ServiceType | str``. Storing the raw ``str``
    # on the wire preserves equality (enum members compare against str values).
    service_type: str
    request_ns: int = msgspec.field(default_factory=time.time_ns)  # type: ignore[assignment]


class StatusMessage(BaseStatusMessage, kw_only=True, tag=MessageType.STATUS.value):
    """Service status report."""


class RegistrationMessage(
    BaseStatusMessage, kw_only=True, tag=MessageType.REGISTRATION.value
):
    """Service self-registration."""


class HeartbeatMessage(
    BaseStatusMessage, kw_only=True, tag=MessageType.HEARTBEAT.value
):
    """Service heartbeat."""


class MemoryReportMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.MEMORY_REPORT.value
):
    """Self-reported memory snapshot from a service process."""

    pid: int
    service_type: str
    phase: MemoryPhase
    pss_bytes: int
    rss_bytes: int | None = None
    uss_bytes: int | None = None
    shared_bytes: int | None = None


class ConnectionProbeMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.CONNECTION_PROBE.value
):
    """ZMQ slow-joiner self-echo probe."""


class BaseServiceErrorMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.SERVICE_ERROR.value
):
    """Service-level error envelope."""

    error: ErrorDetails
