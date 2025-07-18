# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

from pydantic import (
    Field,
)

from aiperf.common.enums import (
    CreditPhase,
    MessageType,
)
from aiperf.common.messages.service_messages import BaseServiceMessage


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
        ge=1,
        description="The number of nanoseconds the credit drop was delayed by, or None if the credit was sent on time. "
        "NOTE: This is only applicable if credit_drop_ns is not None.",
    )
    pre_inference_ns: int | None = Field(
        default=None,
        description="The latency of the credit in nanoseconds from when it was first received to when the inference request was sent. "
        "This can be used to trace the latency in order to identify bottlenecks or other issues.",
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
