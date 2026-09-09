# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from typing import Self

from pydantic import Field, model_validator

from aiperf.common.enums import LifecycleState, MessageType
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
        default_factory=time.time_ns,
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


class HeartbeatMessage(BaseStatusMessage):
    """Message containing heartbeat data.
    This message is sent by a service to the system controller to indicate that it is
    still running.
    """

    message_type: MessageTypeT = MessageType.HEARTBEAT


class BaseServiceErrorMessage(BaseServiceMessage):
    """Base message containing error data."""

    message_type: MessageTypeT = MessageType.SERVICE_ERROR

    error: ErrorDetails = Field(..., description="Error information")


class TargetedServiceMessage(BaseServiceMessage):
    """Message that can be targeted to a specific service by id or type.
    If both `target_service_type` and `target_service_id` are None, the message is
    sent to all services that are subscribed to the message type.
    """

    @model_validator(mode="after")
    def validate_target_service(self) -> Self:
        if self.target_service_id is not None and self.target_service_type is not None:
            raise ValueError(
                "Either target_service_id or target_service_type can be provided, but not both"
            )
        return self

    target_service_id: str | None = Field(
        default=None,
        description="ID of the target service to send the message to. "
        "If both `target_service_type` and `target_service_id` are None, the message is "
        "sent to all services that are subscribed to the message type.",
    )
    target_service_type: ServiceTypeT | None = Field(
        default=None,
        description="Type of the service to send the message to. "
        "If both `target_service_type` and `target_service_id` are None, the message is "
        "sent to all services that are subscribed to the message type.",
    )


class ConnectionProbeMessage(TargetedServiceMessage):
    """Message containing a connection probe from a service. This is used to probe the connection to the service."""

    message_type: MessageTypeT = MessageType.CONNECTION_PROBE
