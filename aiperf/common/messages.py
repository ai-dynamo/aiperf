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
from aiperf.common.enums.timing import CreditPhase
from aiperf.common.health_models import ProcessHealth
from aiperf.common.pydantic_utils import ExcludeIfNoneMixin, exclude_if_none
from aiperf.common.record_models import (
    ErrorDetails,
    ParsedResponseRecord,
    RequestRecord,
)
from aiperf.common.worker_models import WorkerPhaseTaskStats

################################################################################
# Abstract Base Message Models
################################################################################


@exclude_if_none(["request_ns", "request_id"])
class Message(ExcludeIfNoneMixin):
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


class RequiresRequestNSMixin(Message):
    """Mixin for messages that require a request_ns field."""

    request_ns: int = Field(  # type: ignore[assignment]
        default_factory=time.time_ns,
        description="Timestamp of the request in nanoseconds",
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


class ConversationRequestMessage(BaseServiceMessage):
    """Message for a conversation request."""

    message_type: Literal[MessageType.CONVERSATION_REQUEST] = (
        MessageType.CONVERSATION_REQUEST
    )

    conversation_id: str | None = Field(
        default=None, description="The session ID of the conversation"
    )
    credit_phase: CreditPhase | None = Field(
        default=None,
        description="The type of credit phase (either warmup or profiling). If not provided, the timing manager will use the default credit phase.",
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


################################################################################
# Credit Messages
################################################################################


class CreditDropMessage(BaseServiceMessage):
    """Message indicating that a credit has been dropped.
    This message is sent by the timing manager to workers to indicate that credit(s)
    have been dropped.
    """

    message_type: Literal[MessageType.CREDIT_DROP] = MessageType.CREDIT_DROP

    phase: CreditPhase = Field(..., description="The type of credit phase")
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

    phase: CreditPhase = Field(
        ...,
        description="The type of credit phase",
    )
    conversation_id: str | None = Field(
        default=None, description="The ID of the conversation, if applicable."
    )
    credit_drop_ns: int | None = Field(
        default=None,
        description="Timestamp of the original credit drop, if applicable.",
    )
    delayed_ns: int | None = Field(
        default=None,
        ge=0,
        description="The number of nanoseconds the credit drop was delayed by, or None if the credit was sent on time. "
        "NOTE: This is only applicable if credit_drop_ns is not None.",
    )
    pre_inference_ns: int | None = Field(
        default=None,
        description="The latency of the credit in nanoseconds from when it was first received to when the inference request was sent.",
        ge=0,
    )

    @property
    def delayed(self) -> bool:
        return self.delayed_ns is not None


class CreditPhaseStartMessage(BaseServiceMessage):
    """Message for credit phase start. Sent by the TimingManager to report that a credit phase has started."""

    message_type: Literal[MessageType.CREDIT_PHASE_START] = (
        MessageType.CREDIT_PHASE_START
    )
    phase: CreditPhase = Field(..., description="The type of credit phase")
    start_ns: int = Field(
        ge=1,
        description="The start time of the credit phase in nanoseconds.",
    )
    total_requests: int | None = Field(
        default=None,
        ge=1,
        description="The total number of expected credits. If None, the phase is not request count based.",
    )
    expected_duration_ns: int | None = Field(
        default=None,
        ge=1,
        description="The expected duration of the credit phase in nanoseconds. If None, the phase is not time based.",
    )


class CreditPhaseProgressMessage(BaseServiceMessage):
    """Sent by the TimingManager to report the progress of a credit phase."""

    message_type: Literal[MessageType.CREDIT_PHASE_PROGRESS] = (
        MessageType.CREDIT_PHASE_PROGRESS
    )
    phase: CreditPhase = Field(..., description="The type of credit phase")
    sent: int = Field(default=0, description="The number of sent credits")
    completed: int = Field(
        default=0,
        description="The number of completed credits (returned from the workers)",
    )


class CreditPhaseSendingCompleteMessage(BaseServiceMessage):
    """Message for credit phase sending complete. Sent by the TimingManager to report that a credit phase has completed sending."""

    message_type: Literal[MessageType.CREDIT_PHASE_SENDING_COMPLETE] = (
        MessageType.CREDIT_PHASE_SENDING_COMPLETE
    )
    phase: CreditPhase = Field(..., description="The type of credit phase")
    sent_end_ns: int | None = Field(
        default=None,
        description="The time of the last sent credit in nanoseconds. If None, the phase has not sent all credits.",
    )


class CreditPhaseCompleteMessage(BaseServiceMessage):
    """Message for credit phase complete. Sent by the TimingManager to report that a credit phase has completed."""

    message_type: Literal[MessageType.CREDIT_PHASE_COMPLETE] = (
        MessageType.CREDIT_PHASE_COMPLETE
    )
    phase: CreditPhase = Field(..., description="The type of credit phase")
    end_ns: int | None = Field(
        default=None,
        ge=1,
        description="The time in which the last credit was returned from the workers in nanoseconds. If None, the phase has not completed.",
    )


class CreditsCompleteMessage(BaseServiceMessage):
    """Credits complete message sent by the TimingManager to the System controller to signify all Credit Phases
    have been completed."""

    message_type: Literal[MessageType.CREDITS_COMPLETE] = MessageType.CREDITS_COMPLETE


################################################################################
# Worker Messages
################################################################################


class WorkerHealthMessage(BaseServiceMessage):
    """Message for a worker health check."""

    message_type: Literal[MessageType.WORKER_HEALTH] = MessageType.WORKER_HEALTH

    process: ProcessHealth = Field(..., description="The health of the worker process")

    # Worker specific fields
    task_stats: dict[CreditPhase, WorkerPhaseTaskStats] = Field(
        ...,
        description="Stats for the tasks that have been sent to the worker, keyed by the credit phase",
    )

    @property
    def total_tasks(self) -> int:
        """The total number of tasks that have been sent to the worker."""
        return sum(task_stats.total for task_stats in self.task_stats.values())

    @property
    def completed_tasks(self) -> int:
        """The number of tasks that have been completed by the worker."""
        return sum(task_stats.completed for task_stats in self.task_stats.values())

    @property
    def failed_tasks(self) -> int:
        """The number of tasks that have failed by the worker."""
        return sum(task_stats.failed for task_stats in self.task_stats.values())

    @property
    def in_progress_tasks(self) -> int:
        """The number of tasks that are currently in progress by the worker."""
        return sum(task_stats.in_progress for task_stats in self.task_stats.values())

    @property
    def error_rate(self) -> float:
        """The error rate of the worker."""
        if self.total_tasks == 0:
            return 0
        return self.failed_tasks / self.total_tasks
