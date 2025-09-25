# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.types import MessageTypeT


class CreditDropMessage(BaseServiceMessage):
    """Message indicating that a credit has been dropped.
    This message is sent by the timing manager to workers to indicate that credit(s)
    have been dropped.
    """

    message_type: MessageTypeT = MessageType.CREDIT_DROP

    phase: CreditPhase = Field(..., description="The type of credit phase")
    conversation_id: str | None = Field(
        default=None, description="The ID of the conversation, if applicable."
    )
    credit_drop_ns: int | None = Field(
        default=None,
        description="Timestamp of the credit drop, if applicable. None means send ASAP.",
    )
    should_cancel: bool = Field(
        default=False,
        description="Whether this request should be cancelled after the specified delay.",
    )
    cancel_after_ns: int = Field(
        default=0,
        ge=0,
        description="Delay in nanoseconds after which the request should be cancelled. Only applicable if should_cancel is True.",
    )


class CreditReturnMessage(BaseServiceMessage):
    """Message indicating that a credit has been returned.
    This message is sent by a worker to the timing manager to indicate that work has
    been completed.
    """

    message_type: MessageTypeT = MessageType.CREDIT_RETURN

    phase: CreditPhase = Field(
        ...,
        description="The Credit Phase of the credit drop. This is so the TimingManager can track the progress of the credit phase.",
    )
    delayed_ns: int | None = Field(
        default=None,
        ge=1,
        description="The number of nanoseconds the credit drop was delayed by, or None if the credit was sent on time. "
        "NOTE: This is only applicable if the original credit_drop_ns was not None.",
    )

    @property
    def delayed(self) -> bool:
        return self.delayed_ns is not None


class CreditPhaseStartMessage(BaseServiceMessage):
    """Message for credit phase start. Sent by the TimingManager to report that a credit phase has started."""

    message_type: MessageTypeT = MessageType.CREDIT_PHASE_START
    phase: CreditPhase = Field(..., description="The type of credit phase")
    start_ns: int = Field(
        ge=1,
        description="The start time of the credit phase in nanoseconds.",
    )
    total_expected_requests: int | None = Field(
        default=None,
        ge=1,
        description="The total number of expected requests. If None, the phase is not request count based.",
    )
    expected_duration_sec: float | None = Field(
        default=None,
        ge=1,
        description="The expected duration of the credit phase in seconds. If None, the phase is not time based.",
    )


class CreditPhaseProgressMessage(BaseServiceMessage):
    """Sent by the TimingManager to report the progress of a credit phase."""

    message_type: MessageTypeT = MessageType.CREDIT_PHASE_PROGRESS
    phase: CreditPhase = Field(..., description="The type of credit phase")
    sent: int = Field(
        ...,
        ge=0,
        description="The number of sent credits",
    )
    completed: int = Field(
        ...,
        ge=0,
        description="The number of completed credits (returned from the workers)",
    )


class CreditPhaseSendingCompleteMessage(BaseServiceMessage):
    """Message for credit phase sending complete. Sent by the TimingManager to report that a credit phase has completed sending."""

    message_type: MessageTypeT = MessageType.CREDIT_PHASE_SENDING_COMPLETE
    phase: CreditPhase = Field(..., description="The type of credit phase")
    sent_end_ns: int = Field(
        ...,
        ge=1,
        description="The time of the last sent credit in nanoseconds.",
    )
    sent: int = Field(
        ...,
        ge=0,
        description="The final number of sent credits.",
    )


class CreditPhaseCompleteMessage(BaseServiceMessage):
    """Message for credit phase complete. Sent by the TimingManager to report that a credit phase has completed."""

    message_type: MessageTypeT = MessageType.CREDIT_PHASE_COMPLETE
    phase: CreditPhase = Field(..., description="The type of credit phase")
    completed: int = Field(
        ...,
        ge=0,
        description="The number of completed credits (returned from the workers). This is the final count of completed credits.",
    )
    end_ns: int = Field(
        ...,
        ge=1,
        description="The time in which the last credit was returned from the workers in nanoseconds",
    )
    timeout_triggered: bool = Field(
        default=False,
        description="Whether this phase completed because a timeout was triggered",
    )


class CreditsCompleteMessage(BaseServiceMessage):
    """Credits complete message sent by the TimingManager to the System controller to signify all Credit Phases
    have been completed."""

    message_type: MessageTypeT = MessageType.CREDITS_COMPLETE
