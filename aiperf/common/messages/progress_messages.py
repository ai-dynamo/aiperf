# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field, SerializeAsAny

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ErrorDetailsCount, MetricResult, PhaseProcessingStats
from aiperf.common.types import MessageTypeT


class ProfileProgressMessage(BaseServiceMessage):
    """Message for profile progress. Sent by the timing manager to the system controller to report the progress of the profile run."""

    message_type: MessageTypeT = MessageType.PROFILE_PROGRESS

    profile_id: str | None = Field(
        default=None, description="The ID of the current profile"
    )
    start_ns: int = Field(
        ..., description="The start time of the profile run in nanoseconds"
    )
    end_ns: int | None = Field(
        default=None, description="The end time of the profile run in nanoseconds"
    )
    total: int = Field(
        ..., description="The total number of inference requests to be made (if known)"
    )
    completed: int = Field(
        ..., description="The number of inference requests completed"
    )
    warmup: bool = Field(
        default=False,
        description="Whether this is the warmup phase of the profile run",
    )


class SweepProgressMessage(BaseServiceMessage):
    """Message for sweep progress."""

    # TODO: add profile information

    message_type: MessageTypeT = MessageType.SWEEP_PROGRESS

    sweep_id: str = Field(..., description="The ID of the current sweep")
    sweep_start_ns: int = Field(
        ..., description="The start time of the sweep in nanoseconds"
    )
    end_ns: int | None = Field(
        default=None, description="The end time of the profile run in nanoseconds"
    )


class ProcessingStatsMessage(BaseServiceMessage):
    """Message for processing stats. Sent by the records manager to the system controller to report the stats of the profile run."""

    message_type: MessageTypeT = MessageType.PROCESSING_STATS

    error_count: int = Field(default=0, description="The number of errors encountered")
    completed: int = Field(
        default=0, description="The number of requests processed by the records manager"
    )
    worker_completed: dict[str, int] = Field(
        default_factory=dict,
        description="Per-worker request completion counts, keyed by worker service_id",
    )
    worker_errors: dict[str, int] = Field(
        default_factory=dict,
        description="Per-worker error counts, keyed by worker service_id",
    )


class RecordsProcessingStatsMessage(BaseServiceMessage):
    """Message for processing stats. Sent by the RecordsManager to report the stats of the profile run.
    This contains the stats for a single credit phase only."""

    message_type: MessageTypeT = MessageType.PROCESSING_STATS

    processing_stats: PhaseProcessingStats = Field(
        ..., description="The stats for the credit phase"
    )
    worker_stats: dict[str, PhaseProcessingStats] = Field(
        default_factory=dict,
        description="The stats for each worker how many requests were processed and how many errors were "
        "encountered, keyed by worker service_id",
    )


class ProfileResultsMessage(BaseServiceMessage):
    """Message for profile results."""

    message_type: MessageTypeT = MessageType.PROFILE_RESULTS

    records: SerializeAsAny[list[MetricResult]] = Field(
        ..., description="The records of the profile results"
    )
    total: int = Field(
        ...,
        description="The total number of inference requests expected to be made (if known)",
    )
    completed: int = Field(
        ..., description="The number of inference requests completed"
    )
    start_ns: int = Field(
        ..., description="The start time of the profile run in nanoseconds"
    )
    end_ns: int = Field(
        ..., description="The end time of the profile run in nanoseconds"
    )
    was_cancelled: bool = Field(
        default=False,
        description="Whether the profile run was cancelled early",
    )
    errors_by_type: list[ErrorDetailsCount] = Field(
        default_factory=list,
        description="A list of the unique error details and their counts",
    )
