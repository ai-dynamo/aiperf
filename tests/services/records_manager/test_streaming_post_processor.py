# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the streaming post processor base class.
"""

from collections.abc import AsyncGenerator

import pytest

from aiperf.common.enums.timing_enums import CreditPhase
from aiperf.common.factories import StreamingPostProcessorFactory
from aiperf.common.messages.inference_messages import ParsedInferenceResultsMessage
from aiperf.common.models import ParsedResponseRecord, RequestRecord, ResponseData
from aiperf.common.models.record_models import SSEField, SSEMessage
from aiperf.services.records_manager.post_processors.streaming_post_processor import (
    StreamingPostProcessor,
)
from aiperf.services.records_manager.records_manager import RecordsManager
from tests.utils.async_test_utils import async_fixture


class StreamingPostProcessorTest(StreamingPostProcessor):
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


@pytest.fixture
async def records_manager(
    service_config, user_config
) -> AsyncGenerator[RecordsManager, None]:
    records_manager = RecordsManager(
        service_config=service_config, user_config=user_config
    )
    yield records_manager
    await records_manager.stop()


@pytest.fixture
def sample_record(
    time_traveler, records_manager: RecordsManager
) -> ParsedInferenceResultsMessage:
    return ParsedInferenceResultsMessage(
        service_id=records_manager.service_id,
        record=ParsedResponseRecord(
            worker_id="test_worker",
            request=RequestRecord(
                credit_phase=CreditPhase.PROFILING,
                start_perf_ns=time_traveler.perf_counter_ns(),
                timestamp_ns=time_traveler.time_ns(),
                end_perf_ns=time_traveler.perf_counter_ns() + 100,
                delayed_ns=time_traveler.perf_counter_ns() + 100,
                responses=[
                    SSEMessage(
                        packets=[SSEField(name="data", value='{"test": "test"}')],
                        perf_ns=time_traveler.perf_counter_ns() + 100,
                    )
                ],
            ),
            responses=[
                ResponseData(
                    perf_ns=time_traveler.perf_counter_ns() + 100,
                    raw_text=['{"test": "test"}'],
                    parsed_text=["test"],
                )
            ],
            input_token_count=10,
            output_token_count=10,
        ),
    )


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
        StreamingPostProcessorFactory.register("test")(StreamingPostProcessorTest)

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
