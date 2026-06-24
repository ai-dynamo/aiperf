# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""msgspec-Struct command messages (replaces the prior Pydantic command hierarchy).

Branch ajc/k8s-new moved command transport onto native control_structs.py, but
main still ships a high-level CommandMessage/CommandResponse layer used by
SystemController, BaseService, workers, and timing. Re-expressing it on top of
``msgspec.Struct`` keeps that internal API intact on the new msgspec wire.

Nested discriminator dispatch (``message_type`` -> ``command`` -> response shape)
is now resolved by direct ``isinstance`` checks rather than the previous
auto-routed registry; tests exercise the same shape.
"""

import uuid
from typing import Any

import msgspec
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


class TargetedServiceMessage(BaseServiceMessage, kw_only=True, omit_defaults=True):
    """Message that can be targeted to a specific service by id or type.

    If both ``target_service_type`` and ``target_service_id`` are None, the message
    is broadcast to all services subscribed to ``message_type``. Only one of the
    two targeting fields may be set.
    """

    target_service_id: str | None = None
    # ServiceType is an ExtensibleStrEnum with a custom metaclass; msgspec
    # cannot build a decoder for ``ServiceType | str``. Store the raw ``str``
    # on the wire (enum members compare equal to their str values).
    target_service_type: str | None = None

    def __post_init__(self) -> None:
        if self.target_service_id is not None and self.target_service_type is not None:
            raise ValueError(
                "Either target_service_id or target_service_type can be provided, but not both"
            )


class CommandMessage(
    TargetedServiceMessage, kw_only=True, tag=MessageType.COMMAND.value
):
    """Command request sent from controller to a service (or service-to-controller).

    ``command`` field selects the concrete command class (SpawnWorkersCommand etc.).
    """

    command: str
    command_id: str = msgspec.field(default_factory=lambda: str(uuid.uuid4()))


class CommandResponse(
    TargetedServiceMessage, kw_only=True, tag=MessageType.COMMAND_RESPONSE.value
):
    """Response to a CommandMessage.

    ``status`` picks success/failure/ack/unhandled; ``command`` echoes the
    triggering CommandType so callers can correlate by (command_id, command).
    """

    command: str
    command_id: str
    status: CommandResponseStatus


class CommandErrorResponse(
    CommandResponse, kw_only=True, tag=MessageType.COMMAND_RESPONSE.value
):
    error: ErrorDetails
    status: CommandResponseStatus = CommandResponseStatus.FAILURE

    @classmethod
    def from_command_message(
        cls, command_message: "CommandMessage", service_id: str, error: ErrorDetails
    ) -> Self:
        return cls(
            service_id=service_id,
            target_service_id=command_message.service_id,
            command=command_message.command,
            command_id=command_message.command_id,
            error=error,
        )


class CommandSuccessResponse(
    CommandResponse, kw_only=True, tag=MessageType.COMMAND_RESPONSE.value
):
    """Generic success response. Specialized subclasses (e.g. ProcessRecordsResponse)
    refine ``data`` for specific commands."""

    data: Any | None = None
    status: CommandResponseStatus = CommandResponseStatus.SUCCESS

    @classmethod
    def from_command_message(
        cls,
        command_message: "CommandMessage",
        service_id: str,
        data: Any | None = None,
    ) -> Self:
        return cls(
            service_id=service_id,
            target_service_id=command_message.service_id,
            command=command_message.command,
            command_id=command_message.command_id,
            data=data,
        )


class CommandAcknowledgedResponse(
    CommandResponse, kw_only=True, tag=MessageType.COMMAND_RESPONSE.value
):
    status: CommandResponseStatus = CommandResponseStatus.ACKNOWLEDGED

    @classmethod
    def from_command_message(
        cls, command_message: "CommandMessage", service_id: str
    ) -> Self:
        return cls(
            service_id=service_id,
            target_service_id=getattr(command_message, "service_id", None) or "",
            command=getattr(
                command_message, "command", getattr(command_message, "cmd", "")
            ),
            command_id=getattr(
                command_message, "command_id", getattr(command_message, "cid", "")
            ),
        )


class CommandUnhandledResponse(
    CommandResponse, kw_only=True, tag=MessageType.COMMAND_RESPONSE.value
):
    status: CommandResponseStatus = CommandResponseStatus.UNHANDLED

    @classmethod
    def from_command_message(
        cls, command_message: "CommandMessage", service_id: str
    ) -> Self:
        return cls(
            service_id=service_id,
            target_service_id=command_message.service_id,
            command=command_message.command,
            command_id=command_message.command_id,
        )


class RealtimeMetricsCommand(
    CommandMessage, kw_only=True, tag=MessageType.COMMAND.value
):
    command: str = CommandType.REALTIME_METRICS


class StartRealtimeTelemetryCommand(
    CommandMessage, kw_only=True, tag=MessageType.COMMAND.value
):
    """Command to start the realtime telemetry background task in RecordsManager.

    Sent when the user dynamically enables the telemetry dashboard in the UI.
    Always forces GPU telemetry mode to REALTIME_DASHBOARD.
    """

    command: str = CommandType.START_REALTIME_TELEMETRY


class SpawnWorkersCommand(CommandMessage, kw_only=True, tag=MessageType.COMMAND.value):
    command: str = CommandType.SPAWN_WORKERS
    num_workers: int = 0  # validated > 0 in __post_init__

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_workers <= 0:
            raise ValueError("num_workers must be > 0")


class ShutdownWorkersCommand(
    CommandMessage, kw_only=True, tag=MessageType.COMMAND.value
):
    command: str = CommandType.SHUTDOWN_WORKERS
    all_workers: bool = False
    worker_ids: list[str] | None = None
    num_workers: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.all_workers:
            if self.worker_ids is not None or self.num_workers is not None:
                raise ValueError(
                    "When all_workers is True, worker_ids and num_workers must not be specified"
                )
            return
        if self.worker_ids is None and self.num_workers is None:
            raise ValueError(
                "Either worker_ids, num_workers, or all_workers must be provided"
            )
        if self.worker_ids is not None and self.num_workers is not None:
            raise ValueError(
                "Either worker_ids or num_workers must be provided, not both"
            )


class ProcessRecordsCommand(
    CommandMessage, kw_only=True, tag=MessageType.COMMAND.value
):
    command: str = CommandType.PROCESS_RECORDS
    cancelled: bool = False


class ProfileConfigureCommand(
    CommandMessage, kw_only=True, tag=MessageType.COMMAND.value
):
    """Trigger PROFILE_CONFIGURE in receiving services.

    Carries no payload: every receiving service was spawned with the same
    ``BenchmarkRun`` and reads from ``self.run.cfg`` directly.
    """

    command: str = CommandType.PROFILE_CONFIGURE


class ProfileStartCommand(CommandMessage, kw_only=True, tag=MessageType.COMMAND.value):
    """Command sent to request services to start profiling."""

    command: str = CommandType.PROFILE_START


class ProfileCompleteCommand(
    CommandMessage, kw_only=True, tag=MessageType.COMMAND.value
):
    """Command sent when all records are received and profiling is complete.

    Triggers final scrape of server metrics to capture end state.
    """

    command: str = CommandType.PROFILE_COMPLETE


class ProfileCancelCommand(CommandMessage, kw_only=True, tag=MessageType.COMMAND.value):
    """Command sent to request services to cancel profiling."""

    command: str = CommandType.PROFILE_CANCEL


class ShutdownCommand(CommandMessage, kw_only=True, tag=MessageType.COMMAND.value):
    """Command sent to request a service to shutdown."""

    command: str = CommandType.SHUTDOWN


class RegisterServiceCommand(
    CommandMessage, kw_only=True, tag=MessageType.COMMAND.value
):
    """Command sent from a service to the system controller to register itself."""

    command: str = CommandType.REGISTER_SERVICE
    service_type: str = ""  # ExtensibleStrEnum -> str on the wire
    state: LifecycleState = LifecycleState.CREATED


class ProcessRecordsResponse(
    CommandSuccessResponse, kw_only=True, tag=MessageType.COMMAND_RESPONSE.value
):
    """Response to the process records command."""

    command: str = CommandType.PROCESS_RECORDS
    data: ProcessRecordsResult | None = None  # type: ignore[assignment]
