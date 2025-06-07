# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import uuid
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, Field, TypeAdapter, model_serializer

from aiperf.common.enums import CommandType, MessageType, ServiceState, ServiceType

################################################################################
# Abstract Base Message Models
################################################################################

EXCLUDE_IF_NONE = "__exclude_if_none__"


def exclude_if_none(field_names: list[str]):
    """Decorator to set the _exclude_if_none_fields class attribute to the set of
    field names that should be excluded if they are None.
    """

    def decorator(model: type["BaseMessage"]) -> type["BaseMessage"]:
        model._exclude_if_none_fields.update(field_names)
        return model

    return decorator


@exclude_if_none(["request_ns", "request_id"])
class BaseMessage(BaseModel):
    """Base message model with common fields for all messages.

    Each message model should inherit from this class, set the message_type field,
    and define its own additional fields.

    Optionally, the @exclude_if_none decorator can be used to specify which fields
    should be excluded from the serialized message if they are None.

    Example:
    ```python
    @exclude_if_none(["some_field"])
    class ExampleMessage(BaseMessage):
        some_field: int | None = Field(default=None)
        other_field: int = Field(default=1)
    ```
    """

    _exclude_if_none_fields: ClassVar[set[str]] = set()
    """Set of field names that should be excluded from the serialized message if they
    are None. This is set by the @exclude_if_none decorator.
    """

    request_ns: int | None = Field(
        default=None,
        description="Timestamp of the request",
    )

    request_id: str | None = Field(
        default=None,
        description="ID of the request",
    )

    @model_serializer
    def _serialize_message(self) -> dict[str, Any]:
        """Serialize the message to a dictionary.

        This method overrides the default serializer to exclude fields that have a
        value of None and have the EXCLUDE_IF_NONE json_schema_extra key set to True.
        """
        return {
            k: v
            for k, v in self
            if not (k in self._exclude_if_none_fields and v is None)
        }


class BaseServiceMessage(BaseMessage):
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


################################################################################
# Concrete Message Models
################################################################################


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

    state: ServiceState = ServiceState.RUNNING


class CommandMessage(BaseServiceMessage):
    """Message containing command data.
    This message is sent by the system controller to a service to command it to do something.
    """

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
    data: BaseModel | None = Field(
        default=None,
        description="Data to send with the command",
    )


class CreditDropMessage(BaseServiceMessage):
    """Message indicating that a credit has been dropped.
    This message is sent by the timing manager to a workers to indicate that credit(s)
    have been dropped.
    """

    message_type: Literal[MessageType.CREDIT_DROP] = MessageType.CREDIT_DROP

    amount: int = Field(
        ...,
        description="Amount of credits that have been dropped",
    )
    credit_drop_ns: int = Field(
        default_factory=time.time_ns, description="Timestamp of the credit drop"
    )


class CreditReturnMessage(BaseServiceMessage):
    """Message indicating that a credit has been returned.
    This message is sent by a worker to the timing manager to indicate that work has
    been completed.
    """

    message_type: Literal[MessageType.CREDIT_RETURN] = MessageType.CREDIT_RETURN

    amount: int = Field(
        ...,
        description="Amount of credits being returned",
    )


class ErrorMessage(BaseServiceMessage):
    """Message containing error data."""

    message_type: Literal[MessageType.ERROR] = MessageType.ERROR

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


# Discriminated union type - only include message types that include a message_type field
Message = Annotated[
    HeartbeatMessage
    | RegistrationMessage
    | StatusMessage
    | CommandMessage
    | CreditDropMessage
    | CreditReturnMessage
    | ErrorMessage,
    Field(discriminator="message_type"),
]
"""Union of all message types. This is used as a type hint when a function
accepts a message as an argument.

The message type is determined by the discriminator field `message_type`. This is
used by the Pydantic `discriminator` argument to determine the type of the
message automatically when the message is deserialized from a JSON string.

To serialize a message to a JSON string, use the `model_dump_json` method.
To deserialize a message from a JSON string, use the `model_validate_json`
method.

Example:
```python
>>> message = StatusMessage(
...     service_id="service_1",
...     request_id="request_1",
...     request_ns=1716278400000000000,
...     state=ServiceState.READY,
...     service_type=ServiceType.TEST,
... )
>>> json_string = message.model_dump_json()
>>> print(json_string)
{"state": "ready", "service_type": "test", "service_id": "service_1", "request_id": "request_1", "request_ns": 1716278400000000000, "message_type": "status"}
>>> deserialized_message = MessageTypeAdapter.validate_json(json_string)
>>> print(deserialized_message)
StatusMessage(
    message_type=MessageType.STATUS,
    state=ServiceState.READY,
    service_type=ServiceType.TEST,
    service_id="service_1",
    request_id="request_1",
    request_ns=1716278400000000000,
)
>>> print(deserialized_message.state)
ready
```
"""

# Create a TypeAdapter for JSON validation of messages
MessageTypeAdapter = TypeAdapter(Message)
"""TypeAdapter for JSON validation of messages.
Example:
```python
>>> json_string = '{"state": "ready", "service_type": "test", "service_id": "service_1", "request_id": "request_1", "request_ns": 1716278400000000000, "message_type": "status"}'
>>> message = MessageTypeAdapter.validate_json(json_string)
>>> print(message)
StatusMessage(
    message_type=MessageType.STATUS,
    state=ServiceState.READY,
    service_type=ServiceType.TEST,
    service_id="service_1",
    request_id="request_1",
    request_ns=1716278400000000000,
)
```
"""
