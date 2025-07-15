# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from pydantic import Field

from aiperf.common.enums import CreditPhase
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.pydantic_utils import AIPerfBaseModel


class CreditPhaseStats(AIPerfBaseModel):
    """Model for phase credit stats. This is used by the TimingManager to track the progress of the credit phases.
    How many credits were dropped and how many were returned, as well as the progress percentage of the phase."""

    type: CreditPhase = Field(..., description="The type of credit phase")
    start_ns: int = Field(
        default_factory=time.time_ns,
        ge=1,
        description="The start time of the credit phase in nanoseconds.",
    )
    sent_end_ns: int | None = Field(
        default=None,
        description="The time of the last sent credit in nanoseconds. If None, the phase has not sent all credits.",
    )
    end_ns: int | None = Field(
        default=None,
        ge=1,
        description="The time in which the last credit was returned from the workers in nanoseconds. If None, the phase has not completed.",
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
    def should_send(self) -> bool:
        if self.expected_duration_ns:
            return time.time_ns() - self.start_ns <= self.expected_duration_ns
        elif self.total_requests:
            return self.sent < self.total_requests
        raise InvalidStateError("Phase is not time or request count based")

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

        elif self.total_requests is not None:
            # Credit count based, so progress is the percentage of credits returned
            return (self.completed / self.total_requests) * 100

        # We don't know the progress
        return None


class PhaseProcessingStats(AIPerfBaseModel):
    """Model for phase processing stats. How many requests were processed and
    how many errors were encountered."""

    processed: int = Field(
        default=0, description="The number of records processed successfully"
    )
    errors: int = Field(
        default=0, description="The number of record errors encountered"
    )

    @property
    def total_records(self) -> int:
        """The total number of records processed successfully or in error."""
        return self.processed + self.errors
