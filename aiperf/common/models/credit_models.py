# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from pydantic import Field

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.models.base_models import AIPerfBaseModel


class CreditPhaseConfig(AIPerfBaseModel):
    """Model for phase credit config. This is used by the TimingManager to configure the credit phases."""

    type: CreditPhase = Field(..., description="The type of credit phase")
    total_expected_requests: int | None = Field(
        default=None,
        ge=1,
        description="The total number of expected credits. If None, the phase is not request count based.",
    )
    expected_duration_sec: float | None = Field(
        default=None,
        ge=1,
        description="The expected duration of the credit phase in seconds. If None, the phase is not time based.",
    )

    @property
    def is_time_based(self) -> bool:
        return self.expected_duration_sec is not None

    @property
    def is_request_count_based(self) -> bool:
        return self.total_expected_requests is not None

    @property
    def is_valid(self) -> bool:
        """A phase config is valid if it is exactly one of the following:
        - is_time_based (expected_duration_sec is set and > 0)
        - is_request_count_based (total_expected_requests is set and > 0)
        """
        is_time_based = self.is_time_based
        is_request_count_based = self.is_request_count_based
        return (is_time_based and not is_request_count_based) or (
            not is_time_based and is_request_count_based
        )


class CreditPhaseStats(CreditPhaseConfig):
    """Model for phase credit stats. Extends the CreditPhaseConfig fields to track the progress of the credit phases.
    How many credits were dropped and how many were returned, as well as the progress percentage of the phase."""

    start_ns: int | None = Field(
        default=None,
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
    def should_send(self) -> bool:
        """Whether the phase should send more credits."""
        if self.is_time_based:
            return (
                time.time_ns() - (self.start_ns or 0)
                <= (self.expected_duration_sec * NANOS_PER_SECOND)  # type: ignore
            )
        elif self.is_request_count_based:
            return self.sent < self.total_expected_requests  # type: ignore
        raise InvalidStateError("Phase is not time or request count based")

    @property
    def progress_percent(self) -> float | None:
        if self.start_ns is None:
            return None

        if self.is_complete:
            return 100

        if self.is_time_based:
            # Time based, so progress is the percentage of time elapsed compared to the duration

            return (
                (time.time_ns() - self.start_ns)
                / (self.expected_duration_sec * NANOS_PER_SECOND)  # type: ignore
            ) * 100

        elif self.total_expected_requests is not None:
            # Credit count based, so progress is the percentage of credits returned
            return (self.completed / self.total_expected_requests) * 100

        # We don't know the progress
        return None

    @classmethod
    def from_phase_config(cls, phase_config: CreditPhaseConfig) -> "CreditPhaseStats":
        """Create a CreditPhaseStats from a CreditPhaseConfig. This is used to initialize the stats for a phase."""
        return cls(
            type=phase_config.type,
            total_expected_requests=phase_config.total_expected_requests,
            expected_duration_sec=phase_config.expected_duration_sec,
        )


class ProcessingStats(AIPerfBaseModel):
    """Model for phase processing stats. How many requests were processed and
    how many errors were encountered."""

    processed: int = Field(
        default=0, description="The number of records processed successfully"
    )
    errors: int = Field(
        default=0, description="The number of record errors encountered"
    )
    total_expected_requests: int | None = Field(
        default=None,
        description="The total number of expected requests to process. If None, the phase is not request count based.",
    )

    @property
    def total_records(self) -> int:
        """The total number of records processed successfully or in error."""
        return self.processed + self.errors

    @property
    def is_complete(self) -> bool:
        return self.total_records == self.total_expected_requests
