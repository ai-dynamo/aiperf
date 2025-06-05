# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import time
import uuid
from typing import Annotated, Any, Generic, Literal, TypeVar, Union

from pydantic import BaseModel, Field, TypeAdapter

from aiperf.common.enums import CommandType, MessageType, ServiceState, ServiceType

################################################################################
# Payload Models
################################################################################


class ErrorPayload(BaseModel):
    """Exception payload sent by services to report an error."""

    error_code: str | None = Field(
        default=None,
        description="Exception code",
    )
    error_message: str | None = Field(
        default=None,
        description="Exception message",
    )
    error_details: dict[str, Any] | None = Field(
        default=None,
        description="Additional details about the error",
    )


class DataPayload(BaseModel):
    """Base model for data payloads with metadata."""


class StatusPayload(BaseModel):
    """Status payload sent by services to report their current state."""

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

    state: ServiceState = ServiceState.RUNNING


class RegistrationPayload(StatusPayload):
    """Registration payload sent by services to register themselves."""

    state: ServiceState = ServiceState.READY


class CommandPayload(BaseModel):
    """Command payload sent to services to request an action."""

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
    data: BaseModel | None = Field(
        default=None,
        description="Data to send with the command",
    )


class CreditDropPayload(BaseModel):
    """Credit drop payload sent to services to request a credit drop."""

    amount: int = Field(
        ...,
        description="Amount of credits to drop",
    )
    timestamp: int = Field(
        default_factory=time.time_ns, description="Timestamp of the credit drop"
    )


class CreditReturnPayload(BaseModel):
    """Credit return payload sent to services to request a credit return."""

    amount: int = Field(
        ...,
        description="Amount of credits to return",
    )


################################################################################
# Message Models
################################################################################


PayloadT = TypeVar("PayloadT", bound=BaseModel)
"""Type variable used to type hint the payload of a message. Payloads must be a subclass of BaseModel."""


class BaseMessage(BaseModel, Generic[PayloadT]):
    """Base message model with common fields for all messages.
    The payload can be any of the payload types defined above.
    """

    service_id: str | None = Field(
        default=None,
        description="ID of the service sending the message",
    )
    timestamp: int | None = Field(
        default=None,
        description="Time when the message was created",
    )
    request_id: str | None = Field(
        default=None,
        description="ID of the request",
    )
    message_type: Literal[MessageType.UNKNOWN] = Field(
        MessageType.UNKNOWN,
        description="Type of message this message represents",
    )
    payload: PayloadT = Field(
        ...,
        description="Payload of the message",
    )

    def model_dump_json(self, **kwargs) -> str:
        """Serialize the message to JSON.

        This method overrides the default model_dump_json method to exclude
        unset fields.
        """
        return super().model_dump_json(exclude_none=True, **kwargs)


class DataMessage(BaseMessage[DataPayload]):
    """Message containing data."""

    message_type: Literal[MessageType.DATA] = MessageType.DATA  # type: ignore


class HeartbeatMessage(BaseMessage[HeartbeatPayload]):
    """Message containing heartbeat data."""

    message_type: Literal[MessageType.HEARTBEAT] = MessageType.HEARTBEAT  # type: ignore


class RegistrationMessage(BaseMessage[RegistrationPayload]):
    """Message containing registration data."""

    message_type: Literal[MessageType.REGISTRATION] = MessageType.REGISTRATION  # type: ignore


class StatusMessage(BaseMessage[StatusPayload]):
    """Message containing status data."""

    message_type: Literal[MessageType.STATUS] = MessageType.STATUS  # type: ignore


class CommandMessage(BaseMessage[CommandPayload]):
    """Message containing command data."""

    message_type: Literal[MessageType.COMMAND] = MessageType.COMMAND  # type: ignore


class CreditDropMessage(BaseMessage[CreditDropPayload]):
    """Message indicating that a credit has been dropped."""

    message_type: Literal[MessageType.CREDIT_DROP] = MessageType.CREDIT_DROP  # type: ignore


class CreditReturnMessage(BaseMessage[CreditReturnPayload]):
    """Message indicating that a credit has been returned."""

    message_type: Literal[MessageType.CREDIT_RETURN] = MessageType.CREDIT_RETURN  # type: ignore


class ErrorMessage(BaseMessage[ErrorPayload]):
    """Message containing error data."""

    message_type: Literal[MessageType.ERROR] = MessageType.ERROR  # type: ignore


# Discriminated union type
Message = Annotated[
    Union[  # noqa: UP007
        DataMessage,
        HeartbeatMessage,
        RegistrationMessage,
        StatusMessage,
        CommandMessage,
        CreditDropMessage,
        CreditReturnMessage,
        ErrorMessage,
        BaseMessage[PayloadT],
    ],
    Field(discriminator="message_type"),
]
"""Union of all message types. This is used as a type hint when a function
accepts a message as an argument.

The message type is determined by the discriminator field `message_type`. This is
used by the Pydantic `discriminator` argument to determine the type of the
payload automatically when the message is deserialized from a JSON string.

To serialize a message to a JSON string, use the `model_dump_json` method.
To deserialize a message from a JSON string, use the `model_validate_json`
method.

Example:
```python
>>> message = DataMessage(
...     service_id="service_1",
...     request_id="request_1",
...     payload=DataPayload(data="Hello, world!"),
... )
>>> json_string = message.model_dump_json()
>>> print(json_string)
{"payload": {"data": "Hello, world!"}, "service_id": "service_1", "request_id": "request_1", "timestamp": 1716278400000000000, "message_type": "data"}
>>> deserialized_message = MessageTypeAdapter.validate_json(json_string)
>>> print(deserialized_message)
DataMessage(
    message_type=MessageType.DATA,
    payload=DataPayload(data="Hello, world!"),
    service_id="service_1",
    request_id="request_1",
    timestamp=1716278400000000000,
)
>>> print(deserialized_message.payload.data)
Hello, world!
```
"""

# Create a TypeAdapter for JSON validation of messages
MessageTypeAdapter = TypeAdapter(Message)
"""TypeAdapter for JSON validation of messages.
Example:
```python
>>> json_string = '{"payload": {"data": "Hello, world!"}, "service_id": "service_1", "request_id": "request_1", "timestamp": 1716278400000000000, "message_type": "data"}'
>>> message = MessageTypeAdapter.validate_json(json_string)
>>> print(message)
DataMessage(
    message_type=MessageType.DATA,
    payload=DataPayload(data="Hello, world!"),
    service_id="service_1",
    request_id="request_1",
    timestamp=1716278400000000000,
)
```
"""
