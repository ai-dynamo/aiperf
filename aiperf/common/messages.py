# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import time
import uuid
from typing import Any, ClassVar, Literal

import orjson
from pydantic import (
    BaseModel,
    Field,
    SerializeAsAny,
    model_serializer,
)

from aiperf.common.dataset_models import Conversation
from aiperf.common.enums import (
    CommandResponseStatus,
    CommandType,
    MessageType,
    NotificationType,
    ServiceState,
    ServiceType,
)
from aiperf.common.record_models import (
    ErrorDetails,
    ErrorDetailsCount,
    MetricResult,
    ParsedResponseRecord,
    RequestRecord,
)

################################################################################
# Abstract Base Message Models
################################################################################

EXCLUDE_IF_NONE = "__exclude_if_none__"


def exclude_if_none(field_names: list[str]):
    """Decorator to set the _exclude_if_none_fields class attribute to the set of
    field names that should be excluded if they are None.
    """

    def decorator(model: type["Message"]) -> type["Message"]:
        model._exclude_if_none_fields.update(field_names)
        return model

    return decorator


@exclude_if_none(["request_ns", "request_id"])
class Message(BaseModel):
    """Base message class for optimized message handling.

    This class provides a base for all messages, including common fields like message_type,
    request_ns, and request_id. It also supports optional field exclusion based on the
    @exclude_if_none decorator.

    Each message model should inherit from this class, set the message_type field,
    and define its own additional fields.
    Optionally, the @exclude_if_none decorator can be used to specify which fields
    should be excluded from the serialized message if they are None.

    Example:
    ```python
    @exclude_if_none(["some_field"])
    class ExampleMessage(Message):
        some_field: int | None = Field(default=None)
        other_field: int = Field(default=1)
    ```
    """

    _exclude_if_none_fields: ClassVar[set[str]] = set()
    """Set of field names that should be excluded from the serialized message if they
    are None. This is set by the @exclude_if_none decorator.
    """

    _message_type_lookup: ClassVar[dict[MessageType, type["Message"]]] = {}

    def __init_subclass__(cls, **kwargs: dict[str, Any]):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "message_type"):
            cls._message_type_lookup[cls.message_type] = cls

    message_type: MessageType | Any = Field(
        ...,
        description="Type of the message",
    )

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

    @classmethod
    def __get_validators__(cls):
        yield cls.from_json

    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Fast deserialization without full validation"""
        data = json.loads(json_str)
        message_type = data.get("message_type")
        if not message_type:
            raise ValueError("Missing message_type")

        # Use cached message type lookup
        message_class = cls._message_type_lookup[message_type]
        if not message_class:
            raise ValueError(f"Unknown message type: {message_type}")

        return message_class(**data)

    def to_json(self) -> str:
        """Fast serialization without full validation"""
        return orjson.dumps(self.__dict__).decode("utf-8")


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


class ProcessRecordsCommandData(BaseModel):
    """Data to send with the process records command."""

    cancelled: bool = Field(
        default=False,
        description="Whether the profile run was cancelled",
    )


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
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this command. If not provided, a random UUID will be generated.",
    )
    require_response: bool = Field(
        default=False,
        description="Whether a response is required for this command",
    )
    target_service_type: ServiceType | None = Field(
        default=None,
        description="Type of the service to send the command to. "
        "If both `target_service_type` and `target_service_id` are None, the command is "
        "sent to all services.",
    )
    target_service_id: str | None = Field(
        default=None,
        description="ID of the target service to send the command to. "
        "If both `target_service_type` and `target_service_id` are None, the command is "
        "sent to all services.",
    )
    data: SerializeAsAny[ProcessRecordsCommandData | BaseModel | None] = Field(
        default=None,
        description="Data to send with the command",
    )


class CommandResponseMessage(BaseServiceMessage):
    """Message containing a command response.
    This message is sent by a component service to the system controller to respond to a command.
    """

    message_type: Literal[MessageType.COMMAND_RESPONSE] = MessageType.COMMAND_RESPONSE

    command: CommandType = Field(
        ...,
        description="Command type that is being responded to",
    )
    command_id: str = Field(
        ..., description="The ID of the command that is being responded to"
    )
    status: CommandResponseStatus = Field(..., description="The status of the command")
    data: SerializeAsAny[BaseModel | None] = Field(
        default=None,
        description="Data to send with the command response if the command succeeded",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="Error information if the command failed",
    )


class CreditDropMessage(BaseServiceMessage):
    """Message indicating that a credit has been dropped.
    This message is sent by the timing manager to workers to indicate that credit(s)
    have been dropped.
    """

    message_type: Literal[MessageType.CREDIT_DROP] = MessageType.CREDIT_DROP

    amount: int = Field(
        ...,
        description="Amount of credits that have been dropped",
    )
    conversation_id: str | None = Field(
        default=None, description="The ID of the conversation, if applicable."
    )
    credit_drop_ns: int | None = Field(
        default=None,
        description="Timestamp of the credit drop, if applicable. None means send ASAP.",
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


class ErrorMessage(Message):
    """Message containing error data."""

    message_type: Literal[MessageType.ERROR] = MessageType.ERROR

    error: ErrorDetails = Field(..., description="Error information")


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


class CreditsCompleteMessage(BaseServiceMessage):
    """Credits complete message sent to System controller to signify all requests have completed."""

    message_type: Literal[MessageType.CREDITS_COMPLETE] = MessageType.CREDITS_COMPLETE
    cancelled: bool = Field(
        default=False,
        description="Whether the profile run was cancelled",
    )


class ConversationRequestMessage(BaseServiceMessage):
    """Message for a conversation request."""

    message_type: Literal[MessageType.CONVERSATION_REQUEST] = (
        MessageType.CONVERSATION_REQUEST
    )

    conversation_id: str | None = Field(
        default=None, description="The session ID of the conversation"
    )


class ConversationResponseMessage(BaseServiceMessage):
    """Message for a conversation response."""

    message_type: Literal[MessageType.CONVERSATION_RESPONSE] = (
        MessageType.CONVERSATION_RESPONSE
    )

    conversation: Conversation = Field(..., description="The conversation data")


class InferenceResultsMessage(BaseServiceMessage):
    """Message for a inference results."""

    message_type: Literal[MessageType.INFERENCE_RESULTS] = MessageType.INFERENCE_RESULTS

    record: SerializeAsAny[RequestRecord] = Field(
        ..., description="The inference results record"
    )


class ParsedInferenceResultsMessage(BaseServiceMessage):
    """Message for a parsed inference results."""

    message_type: Literal[MessageType.PARSED_INFERENCE_RESULTS] = (
        MessageType.PARSED_INFERENCE_RESULTS
    )

    record: SerializeAsAny[ParsedResponseRecord] = Field(
        ..., description="The post process results record"
    )


class ProfileResultsMessage(BaseServiceMessage):
    """Message for profile results."""

    message_type: Literal[MessageType.PROFILE_RESULTS] = MessageType.PROFILE_RESULTS

    records: SerializeAsAny[list[MetricResult]] = Field(
        ..., description="The records of the profile results"
    )
    total: int = Field(
        ...,
        description="The total number of inference requests expected to be made (if known)",
    )
    completed: int = Field(
        ..., description="The number of inference requests completed"
    )
    start_ns: int = Field(
        ..., description="The start time of the profile run in nanoseconds"
    )
    end_ns: int = Field(
        ..., description="The end time of the profile run in nanoseconds"
    )
    was_cancelled: bool = Field(
        default=False,
        description="Whether the profile run was cancelled early",
    )
    errors_by_type: list[ErrorDetailsCount] = Field(
        default_factory=list,
        description="A list of the unique error details and their counts",
    )


class ProfileProgressMessage(BaseServiceMessage):
    """Message for profile progress. Sent by the timing manager to the system controller to report the progress of the profile run."""

    message_type: Literal[MessageType.PROFILE_PROGRESS] = MessageType.PROFILE_PROGRESS

    profile_id: str | None = Field(
        default=None, description="The ID of the current profile"
    )
    start_ns: int = Field(
        ..., description="The start time of the profile run in nanoseconds"
    )
    end_ns: int | None = Field(
        default=None, description="The end time of the profile run in nanoseconds"
    )
    total: int = Field(
        ..., description="The total number of inference requests to be made (if known)"
    )
    completed: int = Field(
        ..., description="The number of inference requests completed"
    )


class ProfileStatsMessage(BaseServiceMessage):
    """Message for profile stats. Sent by the records manager to the system controller to report the stats of the profile run."""

    message_type: Literal[MessageType.PROFILE_STATS] = MessageType.PROFILE_STATS

    error_count: int = Field(default=0, description="The number of errors encountered")
    completed: int = Field(default=0, description="The number of requests completed")
    worker_completed: dict[str, int] = Field(
        default_factory=dict,
        description="Per-worker request completion counts, keyed by worker service_id",
    )
    worker_errors: dict[str, int] = Field(
        default_factory=dict,
        description="Per-worker error counts, keyed by worker service_id",
    )


class DatasetTimingRequest(BaseServiceMessage):
    """Message for a dataset timing request."""

    message_type: Literal[MessageType.DATASET_TIMING_REQUEST] = (
        MessageType.DATASET_TIMING_REQUEST
    )


class DatasetTimingResponse(BaseServiceMessage):
    """Message for a dataset timing response."""

    message_type: Literal[MessageType.DATASET_TIMING_RESPONSE] = (
        MessageType.DATASET_TIMING_RESPONSE
    )

    timing_data: list[tuple[int, str]] = Field(
        ...,
        description="The timing data of the dataset. Tuple of (timestamp, conversation_id)",
    )
