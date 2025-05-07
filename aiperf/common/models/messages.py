"""Pydantic models for message structures used in inter-service communication."""

import time
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from aiperf.common.enums import MessageType, ServiceState


class BaseMessage(BaseModel):
    """Base message model with common fields for all messages."""

    service_id: str = Field(
        ...,
        description="ID of the service sending the message",
    )
    service_type: str = Field(
        ...,
        description="Type of service sending the message",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Time when the message was created",
    )


class StatusMessage(BaseMessage):
    """Status message sent by services to report their state."""

    message_type: MessageType = MessageType.STATUS
    state: ServiceState = Field(
        ...,
        description="Current state of the service",
    )


class HeartbeatMessage(StatusMessage):
    """Heartbeat message sent periodically by services."""

    message_type: MessageType = MessageType.HEARTBEAT
    state: ServiceState = ServiceState.RUNNING


class CommandMessage(BaseMessage):
    """Command message sent to services to request an action."""

    message_type: MessageType = MessageType.COMMAND
    command_id: str = Field(
        ...,
        description="Unique identifier for this command",
    )
    command: str = Field(
        ...,
        description="Command to execute",
    )
    require_response: bool = Field(
        default=False,
        description="Whether a response is required for this command",
    )
    target_service_id: Optional[str] = Field(
        default=None,
        description="ID of the target service for this command",
    )


class ResponseMessage(BaseMessage):
    """Response message sent in reply to a command."""

    message_type: MessageType = MessageType.RESPONSE
    request_id: str = Field(
        ...,
        description="ID of the command this is responding to",
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response data",
    )


class DataMessage(BaseMessage):
    """Data message for sharing information between services."""

    message_type: MessageType = MessageType.DATA
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data payload",
    )


class RegistrationMessage(BaseMessage):
    """Registration message sent by services to register with the controller."""

    message_type: MessageType = MessageType.REGISTRATION
    state: str = Field(
        default=ServiceState.READY.value,
        description="Current state of the service",
    )


class RegistrationResponseMessage(BaseModel):
    """Response to a registration request."""

    status: str = Field(
        ...,
        description="Status of the registration (ok or error)",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if registration failed",
    )


class CreditMessage(BaseMessage):
    """Credit message sent by the timing manager to authorize a request."""

    message_type: MessageType = MessageType.CREDIT.value
    credit: Dict[str, Any] = Field(
        ...,
        description="Credit data",
    )


class ResultMessage(BaseMessage):
    """Result message sent by workers to report results."""

    message_type: MessageType = MessageType.DATA.value  # Using DATA type for results
    result: Dict[str, Any] = Field(
        ...,
        description="Result data",
    )
