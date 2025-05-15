#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Base Pydantic payload models used for communication between services."""

import time
import uuid
from abc import ABC
from typing import Any, Literal, Union

from pydantic import BaseModel, Field

from aiperf.common.enums import CommandType, MessageType, ServiceState, ServiceType


class BasePayload(BaseModel, ABC):
    """Base model for all payload data. Each payload type must inherit
    from this class, and override the `message_type` field.

    This is used with Pydantic's `discriminator` to allow for polymorphic payloads,
    and automatic type coercion when receiving messages.
    """

    # Note: Literal[MessageType.UNKNOWN] is required due to the way the
    # discriminator is implemented in Pydantic, it requires everything to be a
    # literal.
    message_type: Literal[MessageType.UNKNOWN] = Field(
        ...,
        description="Type of message this payload represents",
    )


class ErrorPayload(BasePayload):
    """Error payload sent by services to report an error."""

    message_type: Literal[MessageType.ERROR] = MessageType.ERROR

    error_code: str = Field(
        ...,
        description="Error code",
    )
    error: str = Field(
        ...,
        description="Error response",
    )
    error_details: dict[str, Any] | None = Field(
        default=None,
        description="Additional details about the error",
    )


class DataPayload(BasePayload):
    """Base model for data payloads with metadata."""

    message_type: Literal[MessageType.DATA] = MessageType.DATA


class StatusPayload(BasePayload):
    """Status payload sent by services to report their current state."""

    message_type: Literal[MessageType.STATUS] = MessageType.STATUS

    state: ServiceState = Field(
        ...,
        description="Current state of the service",
    )
    service_type: ServiceType = Field(
        ...,
        description="Type of service",
    )


class HeartbeatPayload(StatusPayload):
    """Heartbeat payload sent periodically by services."""

    message_type: Literal[MessageType.HEARTBEAT] = MessageType.HEARTBEAT

    state: ServiceState = ServiceState.RUNNING


class RegistrationPayload(StatusPayload):
    """Registration payload sent by services to register themselves."""

    message_type: Literal[MessageType.REGISTRATION] = MessageType.REGISTRATION

    state: ServiceState = ServiceState.READY


class CommandPayload(BasePayload):
    """Command payload sent to services to request an action."""

    message_type: Literal[MessageType.COMMAND] = MessageType.COMMAND

    command: CommandType = Field(
        ...,
        description="Command to execute",
    )
    command_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:8],
        description="Unique identifier for this command",
    )
    require_response: bool = Field(
        default=False,
        description="Whether a response is required for this command",
    )
    target_service_id: str | None = Field(
        default=None,
        description="ID of the target service for this command",
    )


class CreditDropPayload(BasePayload):
    """Credit drop payload sent to services to request a credit drop."""

    message_type: Literal[MessageType.CREDIT_DROP] = MessageType.CREDIT_DROP

    amount: int = Field(
        ...,
        description="Amount of credits to drop",
    )
    timestamp: int = Field(
        default_factory=time.time_ns, description="Timestamp of the credit drop"
    )


class CreditReturnPayload(BasePayload):
    """Credit return payload sent to services to request a credit return."""

    message_type: Literal[MessageType.CREDIT_RETURN] = MessageType.CREDIT_RETURN

    amount: int = Field(
        ...,
        description="Amount of credits to return",
    )


# Only put concrete payload types here, with unique message_type values,
# otherwise the discriminator will complain.
PayloadType = Union[  # noqa: UP007
    DataPayload,
    HeartbeatPayload,
    RegistrationPayload,
    StatusPayload,
    CommandPayload,
    CreditDropPayload,
    CreditReturnPayload,
    ErrorPayload,
]
"""This is a union of all the payload types that can be sent and received.

This is used with Pydantic's `discriminator` to allow for polymorphic payloads,
and automatic type coercion when receiving messages.
"""
