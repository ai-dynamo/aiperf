# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    SerializeAsAny,
)

from aiperf.common.enums import (
    MessageType,
    NotificationType,
    ServiceState,
    ServiceType,
)
from aiperf.common.messages.base_messages import Message
from aiperf.common.models import (
    ErrorDetails,
)


class BaseServiceMessage(Message):
    """Base message that is sent from a service. Requires a service_id field to specify
    the service that sent the message."""

    service_id: str = Field(
        ...,
        description="ID of the service sending the message",
    )


class BaseStatusMessage(BaseServiceMessage):
    """Base message containing status data.
    This message is sent by a service to the system controller to report its status.
    """

    # override request_ns to be auto-filled if not provided
    request_ns: int | None = Field(
        default=time.time_ns(),
        description="Timestamp of the request",
    )
    state: ServiceState = Field(
        ...,
        description="Current state of the service",
    )
    service_type: ServiceType = Field(
        ...,
        description="Type of service",
    )


class StatusMessage(BaseStatusMessage):
    """Message containing status data.
    This message is sent by a service to the system controller to report its status.
    """

    message_type: Literal[MessageType.STATUS] = MessageType.STATUS


class RegistrationMessage(BaseStatusMessage):
    """Message containing registration data.
    This message is sent by a service to the system controller to register itself.
    """

    message_type: Literal[MessageType.REGISTRATION] = MessageType.REGISTRATION

    state: ServiceState = ServiceState.READY


class HeartbeatMessage(BaseStatusMessage):
    """Message containing heartbeat data.
    This message is sent by a service to the system controller to indicate that it is
    still running.
    """

    message_type: Literal[MessageType.HEARTBEAT] = MessageType.HEARTBEAT


class NotificationMessage(BaseServiceMessage):
    """Message containing a notification from a service. This is used to notify other services of events."""

    message_type: Literal[MessageType.NOTIFICATION] = MessageType.NOTIFICATION

    notification_type: NotificationType = Field(
        ...,
        description="The type of notification",
    )

    data: SerializeAsAny[BaseModel | None] = Field(
        default=None,
        description="Data to send with the notification",
    )


class BaseServiceErrorMessage(BaseServiceMessage):
    """Base message containing error data."""

    message_type: Literal[MessageType.SERVICE_ERROR] = MessageType.SERVICE_ERROR

    error: ErrorDetails = Field(..., description="Error information")
