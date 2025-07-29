# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the streaming post processor base class.
"""

import pytest

from aiperf.common.enums.timing_enums import CreditPhase
from aiperf.common.factories import StreamingPostProcessorFactory
from aiperf.common.messages.inference_messages import ParsedInferenceResultsMessage
from aiperf.common.models import ParsedResponseRecord
from aiperf.services.records_manager.post_processors.streaming_post_processor import (
    BaseStreamingPostProcessor,
)
from aiperf.services.records_manager.records_manager import RecordsManager
from tests.utils.async_test_utils import async_fixture


class MockStreamingPostProcessor(BaseStreamingPostProcessor):
    """Test implementation of StreamingPostProcessor for testing purposes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processed_records: list[ParsedResponseRecord] = []
        self.stream_record_call_count = 0
        self.total_output_tokens = 0
        self.total_input_tokens = 0

    async def stream_record(self, record: ParsedResponseRecord) -> None:
        """Test implementation that stores processed records."""
        self.trace(lambda: f"Got streaming post processor record: {record}")
        self.processed_records.append(record)
        self.stream_record_call_count += 1
        self.total_output_tokens += record.output_token_count or 0
        self.total_input_tokens += record.input_token_count or 0


@pytest.mark.asyncio
class TestStreamingPostProcessorBasicFunctionality:
    """Test basic functionality of the StreamingPostProcessor class."""

    async def test_basic_streaming_functionality(
        self,
        records_manager: RecordsManager,
        sample_record: ParsedInferenceResultsMessage,
    ):
        # Clear the registry to avoid conflicts with other tests
        StreamingPostProcessorFactory._registry.clear()
        StreamingPostProcessorFactory.register("test")(MockStreamingPostProcessor)

        records_manager = await async_fixture(records_manager)
        await records_manager._initialize_streaming_post_processors()
        proc = records_manager.streaming_post_processors[0]
        assert proc.service_id == records_manager.service_id
        assert proc.records_queue.maxsize == 100_000
        assert len(proc.processed_records) == 0
        assert proc.stream_record_call_count == 0
        await proc.wait_for_start()

        for _ in range(10):
            await records_manager._on_parsed_inference_results(
                sample_record,
            )

        # Make sure the records manager has finished streaming the records
        await records_manager.wait_for_tasks()

        # Wait for the records to be processed
        await proc.records_queue.join()

        assert proc.stream_record_call_count == 10
        assert len(proc.processed_records) == 10
        assert proc.processed_records[0].worker_id == "test_worker"
        assert proc.processed_records[0].request.credit_phase == CreditPhase.PROFILING
        assert proc.processed_records[0].input_token_count == 10
        assert proc.processed_records[0].output_token_count == 10
        assert proc.total_output_tokens == 100
        assert proc.total_input_tokens == 100
