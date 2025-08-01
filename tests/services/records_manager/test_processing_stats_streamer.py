# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the ProcessingStatsStreamer class.
"""

from unittest.mock import AsyncMock

import pytest

from aiperf.common.messages.inference_messages import ParsedInferenceResultsMessage
from aiperf.post_processors import (
    ProcessingStatsStreamer,
)
from aiperf.records import RecordsManager


@pytest.fixture
def streamer(records_manager: RecordsManager) -> ProcessingStatsStreamer:
    """Create a ProcessingStatsStreamer instance."""
    return ProcessingStatsStreamer(
        sub_client=records_manager.sub_client,
        pub_client=records_manager.pub_client,
        service_id=records_manager.service_id,
        service_config=records_manager.service_config,
        user_config=records_manager.user_config,
    )


def next_record(
    sample_record: ParsedInferenceResultsMessage,
    time_traveler,
    advance_seconds: float = 1,
) -> ParsedInferenceResultsMessage:
    """Create a next record."""
    time_traveler.advance_time(advance_seconds)
    return sample_record.model_copy(
        update={
            "request": sample_record.record.request.model_copy(
                update={
                    "start_perf_ns": time_traveler.perf_counter_ns(),
                    "timestamp_ns": time_traveler.time_ns(),
                }
            )
        }
    )


@pytest.mark.asyncio
class TestProcessingStatsStreamer:
    """Tests for the ProcessingStatsStreamer class."""

    async def test_basic_functionality(
        self,
        streamer: ProcessingStatsStreamer,
        sample_record: ParsedInferenceResultsMessage,
    ):
        """Test the basic functionality of the ProcessingStatsStreamer class."""
        await streamer.stream_record(sample_record.record)

    async def test_all_records_received(
        self,
        time_traveler,
        streamer: ProcessingStatsStreamer,
        sample_record: ParsedInferenceResultsMessage,
    ):
        """Test the all records received functionality of the ProcessingStatsStreamer class."""

        streamer.final_request_count = 10
        streamer.processing_stats.total_expected_requests = 10
        if hasattr(streamer, "pub_client"):
            streamer.pub_client.publish = AsyncMock()

        for _ in range(10):
            await streamer.stream_record(sample_record.record)
            sample_record = next_record(sample_record, time_traveler)

        await streamer.wait_for_tasks()

        assert streamer.processing_stats.processed == 10
        assert streamer.processing_stats.errors == 0
        assert streamer.processing_stats.total_expected_requests == 10
        assert streamer.processing_stats.total_records == 10

    async def test_report_records_task(
        self,
        time_traveler,
        streamer: ProcessingStatsStreamer,
        sample_record: ParsedInferenceResultsMessage,
    ):
        """Test the report records task functionality of the ProcessingStatsStreamer class."""
        streamer.processing_stats.processed = 0
        streamer.processing_stats.errors = 0
        streamer.processing_stats.total_expected_requests = 10
        if hasattr(streamer, "pub_client"):
            streamer.pub_client.publish = AsyncMock()

        for _ in range(10):
            await streamer.stream_record(sample_record.record)
            sample_record = next_record(sample_record, time_traveler)

        await streamer.wait_for_tasks()
        await streamer._report_records_task()

        assert streamer.processing_stats.processed == 10
        assert streamer.processing_stats.errors == 0
        assert streamer.processing_stats.total_expected_requests == 10
