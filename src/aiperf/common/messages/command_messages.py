# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import uuid
from typing import Any, ClassVar

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.common.enums import (
    CommandResponseStatus,
    CommandType,
    LifecycleState,
    MessageType,
)
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import (
    ErrorDetails,
    ProcessRecordsResult,
)
from aiperf.common.types import CommandTypeT, MessageTypeT, ServiceTypeT


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


class CommandMessage(TargetedServiceMessage):
    """Message containing command data with automatic routing by command field.

    Uses AutoRoutedModel for nested routing:
    1. First routes by message_type -> CommandMessage
    2. Then routes by command -> specific command class (e.g., SpawnWorkersCommand)

    This message is sent by the system controller to a service to command it to do something.
    """

    discriminator_field: ClassVar[str] = "command"

    message_type: MessageTypeT = MessageType.COMMAND

    command: CommandTypeT = Field(
        ...,
        description="Command to execute",
    )
    command_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this command. If not provided, a random UUID will be generated.",
    )


class CommandResponse(TargetedServiceMessage):
    """Message containing a command response with automatic routing by status.

    Uses AutoRoutedModel for multi-level routing:
    1. Routes by message_type -> CommandResponse
    2. Routes by status -> specific status class
    3. For success, routes by command -> specific command response

    This enables single-parse deserialization to the most specific response type.
    """

    discriminator_field: ClassVar[str] = "status"

    message_type: MessageTypeT = MessageType.COMMAND_RESPONSE

    command: CommandTypeT = Field(
        ...,
        description="Command type that is being responded to",
    )
    command_id: str = Field(
        ..., description="The ID of the command that is being responded to"
    )
    status: CommandResponseStatus = Field(..., description="The status of the command")


class CommandErrorResponse(CommandResponse):
    status: CommandResponseStatus = CommandResponseStatus.FAILURE
    error: ErrorDetails = Field(
        ...,
        description="Error information if the command failed",
    )

    @classmethod
    def from_command_message(
        cls, command_message: CommandMessage, service_id: str, error: ErrorDetails
    ) -> Self:
        return cls(
            service_id=service_id,
            target_service_id=command_message.service_id,
            command=command_message.command,
            command_id=command_message.command_id,
            error=error,
        )


class CommandSuccessResponse(CommandResponse):
    """Generic command response message when a command succeeds.

    Uses nested discriminator routing: success responses can be further
    specialized by command type (e.g., ProcessRecordsResponse).
    """

    discriminator_field: ClassVar[str] = "command"

    status: CommandResponseStatus = CommandResponseStatus.SUCCESS
    data: Any | None = Field(
        default=None,
        description="The data of the command response",
    )

    @classmethod
    def from_command_message(
        cls, command_message: CommandMessage, service_id: str, data: Any | None = None
    ) -> Self:
        return cls(
            service_id=service_id,
            target_service_id=command_message.service_id,
            command=command_message.command,
            command_id=command_message.command_id,
            data=data,
        )


class CommandAcknowledgedResponse(CommandResponse):
    status: CommandResponseStatus = CommandResponseStatus.ACKNOWLEDGED

    @classmethod
    def from_command_message(
        cls, command_message: CommandMessage, service_id: str
    ) -> Self:
        return cls(
            service_id=service_id,
            target_service_id=command_message.service_id,
            command=command_message.command,
            command_id=command_message.command_id,
        )


class CommandUnhandledResponse(CommandResponse):
    status: CommandResponseStatus = CommandResponseStatus.UNHANDLED

    @classmethod
    def from_command_message(
        cls, command_message: CommandMessage, service_id: str
    ) -> Self:
        return cls(
            service_id=service_id,
            target_service_id=command_message.service_id,
            command=command_message.command,
            command_id=command_message.command_id,
        )


class RealtimeMetricsCommand(CommandMessage):
    command: CommandTypeT = CommandType.REALTIME_METRICS


class StartRealtimeTelemetryCommand(CommandMessage):
    """Command to start the realtime telemetry background task in RecordsManager.

    This command is sent when the user dynamically enables the telemetry dashboard
    by pressing the telemetry option in the UI. This always sets the GPU telemetry
    mode to REALTIME_DASHBOARD.
    """

    command: CommandTypeT = CommandType.START_REALTIME_TELEMETRY


class SpawnWorkersCommand(CommandMessage):
    command: CommandTypeT = CommandType.SPAWN_WORKERS

    num_workers: int = Field(..., description="Number of workers to spawn")


class ShutdownWorkersCommand(CommandMessage):
    command: CommandTypeT = CommandType.SHUTDOWN_WORKERS

    @model_validator(mode="after")
    def validate_worker_ids_or_num_workers(self) -> Self:
        if self.all_workers:
            if self.worker_ids is not None or self.num_workers is not None:
                raise ValueError(
                    "When all_workers is True, worker_ids and num_workers must not be specified"
                )
            return self

        if self.worker_ids is None and self.num_workers is None:
            raise ValueError(
                "Either worker_ids, num_workers, or all_workers must be provided"
            )
        if self.worker_ids is not None and self.num_workers is not None:
            raise ValueError(
                "Either worker_ids or num_workers must be provided, not both"
            )
        return self

    all_workers: bool = Field(
        default=False,
        description="Whether to shutdown all workers. If True, worker_ids and num_workers must be None.",
    )
    worker_ids: list[str] | None = Field(
        default=None,
        description="Specific IDs of the workers to shutdown.",
    )
    num_workers: int | None = Field(
        default=None,
        description="Number of workers to shutdown if worker_ids is not provided.",
    )


class ProcessRecordsCommand(CommandMessage):
    """Data to send with the process records command."""

    command: CommandTypeT = CommandType.PROCESS_RECORDS

    cancelled: bool = Field(
        default=False,
        description="Whether the profile run was cancelled",
    )


class ProfileConfigureCommand(CommandMessage):
    """Data to send with the profile configure command."""

    command: CommandTypeT = CommandType.PROFILE_CONFIGURE

    # TODO: Define this type
    config: Any = Field(..., description="Configuration for the profile")


class ProfileStartCommand(CommandMessage):
    """Command message sent to request services to start profiling."""

    command: CommandTypeT = CommandType.PROFILE_START


class ProfileCompleteCommand(CommandMessage):
    """Command message sent when all records are received and profiling is complete.

    Triggers final scrape of server metrics to capture end state.
    """

    command: CommandTypeT = CommandType.PROFILE_COMPLETE


class ProfileCancelCommand(CommandMessage):
    """Command message sent to request services to cancel profiling."""

    command: CommandTypeT = CommandType.PROFILE_CANCEL


class ShutdownCommand(CommandMessage):
    """Command message sent to request a service to shutdown."""

    command: CommandTypeT = CommandType.SHUTDOWN


class RegisterServiceCommand(CommandMessage):
    """Command message sent from a service to the system controller to register itself."""

    command: CommandTypeT = CommandType.REGISTER_SERVICE

    service_id: str = Field(..., description="The ID of the service to register")
    service_type: ServiceTypeT = Field(
        ..., description="The type of the service to register"
    )
    state: LifecycleState = Field(..., description="The current state of the service")


class ProcessRecordsResponse(CommandSuccessResponse):
    """Response to the process records command."""

    command: CommandTypeT = CommandType.PROCESS_RECORDS

    data: ProcessRecordsResult | None = Field(  # type: ignore[assignment]
        default=None,
        description="The result of the process records command",
    )


class ConnectionProbeMessage(TargetedServiceMessage):
    """Message containing a connection probe from a service. This is used to probe the connection to the service."""

    message_type: MessageTypeT = MessageType.CONNECTION_PROBE
