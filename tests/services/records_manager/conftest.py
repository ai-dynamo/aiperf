# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Fixtures for the records manager tests.
"""

import time
from collections.abc import AsyncGenerator

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.messages.inference_messages import ParsedInferenceResultsMessage
from aiperf.common.messages.progress_messages import (
    AllRecordsReceivedMessage,
    PhaseProcessingStats,
)
from aiperf.common.models import (
    ParsedResponseRecord,
    RequestRecord,
    ResponseData,
    SSEField,
    SSEMessage,
)
from aiperf.services.records_manager.records_manager import RecordsManager


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


@pytest.fixture
def sample_final_phase_processing_stats() -> PhaseProcessingStats:
    return PhaseProcessingStats(
        total_expected_requests=10,
        processed=10,
        errors=0,
    )


@pytest.fixture
def sample_all_records_received_message(
    records_manager: RecordsManager,
    sample_final_phase_processing_stats: PhaseProcessingStats,
) -> AllRecordsReceivedMessage:
    return AllRecordsReceivedMessage(
        service_id=records_manager.service_id,
        request_ns=time.time_ns(),
        final_processing_stats=sample_final_phase_processing_stats,
    )
