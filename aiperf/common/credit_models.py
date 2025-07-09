# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Literal

from pydantic import Field

from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages import BaseServiceMessage
from aiperf.common.pydantic_utils import AIPerfBaseModel


class CreditPhaseStats(AIPerfBaseModel):
    """Model for phase credit stats. This is used by the TimingManager to track the progress of the credit phases.
    How many credits were dropped and how many were returned, as well as the progress percentage of the phase."""

    type: CreditPhase = Field(..., description="The type of credit phase")
    start_ns: int | None = Field(
        default=None,
        description="The start time of the credit phase in nanoseconds. If None, the phase has not started.",
    )
    end_ns: int | None = Field(
        default=None,
        description="The end time of the credit phase in nanoseconds. If None, the phase has not ended.",
    )
    sent_end_ns: int | None = Field(
        default=None,
        description="The end time of the sent credits in nanoseconds. If None, the phase has not sent all credits.",
    )
    total: int | None = Field(
        default=None,
        description="The total number of expected credits. If None, the phase is not request count based.",
    )
    expected_duration_ns: int | None = Field(
        default=None,
        description="The expected duration of the credit phase in nanoseconds. If None, the phase is not time based.",
    )
    sent: int = Field(default=0, description="The number of sent credits")
    completed: int = Field(
        default=0,
        description="The number of completed credits (returned from the workers)",
    )

    @property
    def is_sending_complete(self) -> bool:
        return self.sent_end_ns is not None

    @property
    def is_complete(self) -> bool:
        return self.is_sending_complete and self.end_ns is not None

    @property
    def is_started(self) -> bool:
        return self.start_ns is not None

    @property
    def in_flight(self) -> int:
        """Calculate the number of in-flight credits (sent but not completed)."""
        return self.sent - self.completed

    @property
    def is_time_based(self) -> bool:
        return self.expected_duration_ns is not None

    @property
    def progress_percent(self) -> float | None:
        if not self.is_started:
            return None

        if self.is_complete:
            return 100

        if self.is_time_based:
            # Time based, so progress is the percentage of time elapsed compared to the duration
            return (
                (time.time_ns() - self.start_ns) / self.expected_duration_ns  # type: ignore
            ) * 100

        elif self.total is not None:
            # Credit count based, so progress is the percentage of credits returned
            return (self.completed / self.total) * 100

        # We don't know the progress
        return None


class PhaseProcessingStats(AIPerfBaseModel):
    """Model for phase processing stats. How many requests were processed and
    how many errors were encountered."""

    processed: int = Field(default=0, description="The number of records processed")
    errors: int = Field(
        default=0, description="The number of record errors encountered"
    )


class RecordsProcessingStatsMessage(BaseServiceMessage):
    """Message for processing stats. Sent by the RecordsManager to report the stats of the profile run.
    This contains the stats for a single credit phase only."""

    message_type: Literal[MessageType.PROCESSING_STATS] = MessageType.PROCESSING_STATS

    request_ns: int = Field(  # type: ignore
        default_factory=time.time_ns,
        description="The timestamp of the request in nanoseconds",
    )
    current_phase: CreditPhase = Field(
        ...,
        description="The current credit phase (either warmup, ramp-up, stabilizing, or steady-state)",
    )
    worker_stats: dict[str, PhaseProcessingStats] = Field(
        default_factory=dict,
        description="The stats for each worker how many requests were processed and how many errors were "
        "encountered, keyed by worker service_id",
    )


class BasePhaseStatsMessage(BaseServiceMessage):
    """Base message for phase stats. Sent by the TimingManager to report stats of a credit phase."""

    # override request_ns to be auto-filled if not provided
    request_ns: int = Field(  # type: ignore
        default_factory=time.time_ns,
        description="The timestamp of the request in nanoseconds",
    )
    phase_stats: CreditPhaseStats = Field(
        ...,
        description="The credit phase stats for the phase",
    )


class CreditPhaseProgressMessage(BasePhaseStatsMessage):
    """Message for credit phase stats. Sent by the TimingManager to report the stats of a credit phase."""

    message_type: Literal[MessageType.CREDIT_PHASE_PROGRESS] = (
        MessageType.CREDIT_PHASE_PROGRESS
    )


class CreditPhaseStartMessage(BasePhaseStatsMessage):
    """Message for credit phase start. Sent by the TimingManager to report that a credit phase has started."""

    message_type: Literal[MessageType.CREDIT_PHASE_START] = (
        MessageType.CREDIT_PHASE_START
    )


class CreditPhaseCompleteMessage(BasePhaseStatsMessage):
    """Message for credit phase complete. Sent by the TimingManager to report that a credit phase has completed."""

    message_type: Literal[MessageType.CREDIT_PHASE_COMPLETE] = (
        MessageType.CREDIT_PHASE_COMPLETE
    )


class CreditPhaseSendingCompleteMessage(BasePhaseStatsMessage):
    """Message for credit phase sending complete. Sent by the TimingManager to report that a credit phase has completed sending."""

    message_type: Literal[MessageType.CREDIT_PHASE_SENDING_COMPLETE] = (
        MessageType.CREDIT_PHASE_SENDING_COMPLETE
    )


class CreditDropMessage(BaseServiceMessage):
    """Message indicating that a credit has been dropped.
    This message is sent by the timing manager to workers to indicate that credit(s)
    have been dropped.
    """

    message_type: Literal[MessageType.CREDIT_DROP] = MessageType.CREDIT_DROP

    credit_phase: CreditPhase = Field(
        ...,
        description="The type of credit phase",
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

    credit_phase: CreditPhase = Field(
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
        description="The number of nanoseconds the credit drop was delayed by, or None if the credit was sent on time.",
    )


class CreditsCompleteMessage(BaseServiceMessage):
    """Credits complete message sent by the TimingManager to the System controller to signify all requests have completed."""

    message_type: Literal[MessageType.CREDITS_COMPLETE] = MessageType.CREDITS_COMPLETE
    cancelled: bool = Field(
        default=False,
        description="Whether the profile run was cancelled",
    )
