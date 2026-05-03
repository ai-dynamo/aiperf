# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.enums import MessageType
from aiperf.common.messages.base_messages import RequiresRequestNSMixin
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import (
    PhaseRecordsStats,
    WorkerProcessingStats,
)
from aiperf.common.models.record_models import ProcessRecordsResult, ProfileResults
from aiperf.common.types import MessageTypeT


class RecordsProcessingStatsMessage(BaseServiceMessage):
    """Message for processing stats. Sent by the RecordsManager to report the stats of the profile run.
    This contains the stats for a single credit phase only."""

    message_type: MessageTypeT = MessageType.PROCESSING_STATS

    processing_stats: PhaseRecordsStats = Field(
        ..., description="The stats for the credit phase"
    )
    worker_stats: dict[str, WorkerProcessingStats] = Field(
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
    final_processing_stats: PhaseRecordsStats = Field(
        ..., description="The final processing stats for the profile run"
    )


class ProcessRecordsResultMessage(BaseServiceMessage):
    """Message for process records result."""

    message_type: MessageTypeT = MessageType.PROCESS_RECORDS_RESULT

    results: ProcessRecordsResult = Field(..., description="The process records result")
