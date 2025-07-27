# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

from pydantic import (
    BaseModel,
    Field,
    SerializeAsAny,
)

from aiperf.common.enums import (
    LifecycleState,
    MessageType,
    NotificationType,
)
from aiperf.common.messages.base_messages import Message
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.types import MessageTypeT, ServiceTypeT


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
    state: LifecycleState = Field(
        ...,
        description="Current state of the service",
    )
    service_type: ServiceTypeT = Field(
        ...,
        description="Type of service",
    )


class StatusMessage(BaseStatusMessage):
    """Message containing status data.
    This message is sent by a service to the system controller to report its status.
    """

    message_type: MessageTypeT = MessageType.STATUS


class RegistrationMessage(BaseStatusMessage):
    """Message containing registration data.
    This message is sent by a service to the system controller to register itself.
    """

    message_type: MessageTypeT = MessageType.REGISTRATION


class HeartbeatMessage(BaseStatusMessage):
    """Message containing heartbeat data.
    This message is sent by a service to the system controller to indicate that it is
    still running.
    """

    message_type: MessageTypeT = MessageType.HEARTBEAT


class NotificationMessage(BaseServiceMessage):
    """Message containing a notification from a service. This is used to notify other services of events."""

    message_type: MessageTypeT = MessageType.NOTIFICATION

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

    message_type: MessageTypeT = MessageType.SERVICE_ERROR

    error: ErrorDetails = Field(..., description="Error information")
