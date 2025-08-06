# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import uuid
from typing import Any, ClassVar

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import (
    CommandResponseStatus,
    CommandType,
    MessageType,
)
from aiperf.common.enums.service_enums import LifecycleState
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import (
    ErrorDetails,
    ProcessRecordsResult,
    exclude_if_none,
)
from aiperf.common.types import CommandTypeT, MessageTypeT, ServiceTypeT

_logger = AIPerfLogger(__name__)


@exclude_if_none("target_service_id", "target_service_type")
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
    """Message containing command data.
    This message is sent by the system controller to a service to command it to do something.
    """

    _command_type_lookup: ClassVar[dict[CommandTypeT, type["CommandMessage"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "command"):
            cls._command_type_lookup[cls.command] = cls

    message_type: MessageTypeT = MessageType.COMMAND

    command: CommandTypeT = Field(
        ...,
        description="Command to execute",
    )
    command_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this command. If not provided, a random UUID will be generated.",
    )

    @classmethod
    def from_json(cls, json_str: str | bytes | bytearray) -> "CommandMessage":
        """Deserialize a command message from a JSON string, attempting to auto-detect the command type."""
        data = json.loads(json_str)
        command_type = data.get("command")
        if not command_type:
            raise ValueError(f"Missing command: {json_str}")

        # Use cached command type lookup
        command_class = cls._command_type_lookup[command_type]
        if not command_class:
            _logger.debug(
                lambda: f"No command class found for command type: {command_type}"
            )
            # fallback to regular command class
            command_class = cls

        return command_class.model_validate(data)


class CommandResponse(TargetedServiceMessage):
    """Message containing a command response."""

    # Specialized lookup for command response messages by status
    _command_status_lookup: ClassVar[
        dict[CommandResponseStatus, type["CommandResponse"]]
    ] = {}
    # Specialized lookup for command response messages by command type, for success messages
    _command_success_type_lookup: ClassVar[
        dict[CommandTypeT, type["CommandResponse"]]
    ] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if (
            hasattr(cls, "status")
            and cls.status is not None
            and cls.status not in cls._command_status_lookup
        ):
            cls._command_status_lookup[cls.status] = cls
        elif (
            cls.__pydantic_fields__.get("status") is not None
            and cls.__pydantic_fields__.get("status").default
            == CommandResponseStatus.SUCCESS
        ):
            # Cache the specialized lookup by command type for success messages
            cls._command_success_type_lookup[cls.command] = cls

    message_type: MessageTypeT = MessageType.COMMAND_RESPONSE

    command: CommandTypeT = Field(
        ...,
        description="Command type that is being responded to",
    )
    command_id: str = Field(
        ..., description="The ID of the command that is being responded to"
    )
    status: CommandResponseStatus = Field(..., description="The status of the command")

    @classmethod
    def from_json(cls, json_str: str | bytes | bytearray) -> "CommandResponse":
        """Deserialize a command response message from a JSON string, attempting to auto-detect the command response type."""
        data = json.loads(json_str)
        status = data.get("status")
        if not status:
            raise ValueError(f"Missing command response status: {json_str}")
        command = data.get("command")
        if not command:
            raise ValueError(f"Missing command in command response: {json_str}")

        if status not in cls._command_status_lookup:
            raise ValueError(
                f"Unknown command response status: {status}. Valid statuses are: {list(cls._command_status_lookup.keys())}"
            )

        # Use cached command response type lookup by status
        command_response_class = cls._command_status_lookup[status]

        if (
            status == CommandResponseStatus.SUCCESS
            and command in cls._command_success_type_lookup
        ):
            # For success messages, use the specialized lookup by command type if it exists
            command_response_class = cls._command_success_type_lookup[command]

        return command_response_class.model_validate(data)


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
    """Generic command response message when a command succeeds. It should be
    subclassed for specific command types."""

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


class SpawnWorkersCommand(CommandMessage):
    command: CommandTypeT = CommandType.SPAWN_WORKERS

    num_workers: int = Field(..., description="Number of workers to spawn")


@exclude_if_none("worker_ids", "num_workers")
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


class SpawnDatasetProcessorsCommand(CommandMessage):
    command: CommandTypeT = CommandType.SPAWN_DATASET_PROCESSORS

    num_processors: int = Field(
        ..., description="Number of dataset processors to spawn"
    )


@exclude_if_none("processor_ids", "num_processors")
class ShutdownDatasetProcessorsCommand(CommandMessage):
    command: CommandTypeT = CommandType.SHUTDOWN_DATASET_PROCESSORS

    # TODO: check if these are needed
    @model_validator(mode="after")
    def validate_processor_ids_or_num_processors(self) -> Self:
        if self.all_processors:
            if self.processor_ids is not None or self.num_processors is not None:
                raise ValueError(
                    "When all_processors is True, processor_ids and num_processors must not be specified"
                )
            return self

        if self.processor_ids is None and self.num_processors is None:
            raise ValueError(
                "Either processor_ids, num_processors, or all_processors must be provided"
            )
        if self.processor_ids is not None and self.num_processors is not None:
            raise ValueError(
                "Either processor_ids or num_processors must be provided, not both"
            )
        return self

    all_processors: bool = Field(
        default=False,
        description="Whether to shutdown all processors. If True, processor_ids and num_processors must be None.",
    )
    processor_ids: list[str] | None = Field(
        default=None,
        description="Specific IDs of the processors to shutdown.",
    )
    num_processors: int | None = Field(
        default=None,
        description="Number of processors to shutdown if processor_ids is not provided.",
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
