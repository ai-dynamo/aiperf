# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.enums import MessageType
from aiperf.common.messages.base_messages import RequiresRequestNSMixin
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ProcessingStats
from aiperf.common.models.record_models import ProcessRecordsResult, ProfileResults
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

    processing_stats: ProcessingStats = Field(
        ..., description="The stats for the credit phase"
    )
    worker_stats: dict[str, ProcessingStats] = Field(
        default_factory=dict,
        description="The stats for each worker how many requests were processed and how many errors were "
        "encountered, keyed by worker service_id",
    )


class ProfileResultsMessage(BaseServiceMessage):
    """Message for profile results."""

    message_type: MessageTypeT = MessageType.PROFILE_RESULTS

    profile_results: ProfileResults = Field(..., description="The profile results")


class AllRecordsReceivedMessage(BaseServiceMessage, RequiresRequestNSMixin):
    """This is sent by the RecordsManager to signal that all parsed records have been received, and the final processing stats are available."""

    message_type: MessageTypeT = MessageType.ALL_RECORDS_RECEIVED
    final_processing_stats: ProcessingStats = Field(
        ..., description="The final processing stats for the profile run"
    )


class ProcessRecordsResultMessage(BaseServiceMessage):
    """Message for process records result."""

    message_type: MessageTypeT = MessageType.PROCESS_RECORDS_RESULT

    results: ProcessRecordsResult = Field(..., description="The process records result")
