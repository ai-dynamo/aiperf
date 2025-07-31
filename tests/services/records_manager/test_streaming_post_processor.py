# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the streaming post processor base class.
"""

import asyncio

import pytest

from aiperf.common.enums import StreamingPostProcessorType
from aiperf.common.enums.timing_enums import CreditPhase
from aiperf.common.factories import StreamingPostProcessorFactory
from aiperf.common.messages.inference_messages import ParsedInferenceResultsMessage
from aiperf.common.models import ParsedResponseRecord
from aiperf.post_processors import (
    BaseStreamingPostProcessor,
)
from aiperf.records import RecordsManager


@pytest.fixture(autouse=True)
def patch_streaming_post_processor_factory():
    StreamingPostProcessorFactory._registry.clear()
    StreamingPostProcessorFactory.register(StreamingPostProcessorType.JSONL)(
        MockStreamingPostProcessor
    )
    yield
    StreamingPostProcessorFactory._registry.clear()


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
        proc = next(
            p
            for p in records_manager.streaming_post_processors
            if isinstance(p, MockStreamingPostProcessor)
        )
        # Test hack: manually start the background processing task
        # This is necessary because the test does not go through the full lifecycle of RecordsManager
        # and its streaming post processors.
        proc._task = asyncio.create_task(proc._stream_records_task())
        assert proc.service_id == records_manager.service_id
        assert proc.records_queue.maxsize == 100_000
        assert len(proc.processed_records) == 0
        assert proc.stream_record_call_count == 0

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
