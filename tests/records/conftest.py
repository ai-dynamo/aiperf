# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for records tests."""

import pytest

from aiperf.common.models import (
    ParsedResponse,
    ParsedResponseRecord,
    RequestRecord,
    TextResponseData,
)

# Constants for test data
DEFAULT_START_TIME_NS = 1_000_000
DEFAULT_FIRST_RESPONSE_NS = 1_050_000
DEFAULT_LAST_RESPONSE_NS = 1_100_000
DEFAULT_INPUT_TOKENS = 5
DEFAULT_OUTPUT_TOKENS = 2


@pytest.fixture
def sample_request_record() -> RequestRecord:
    """Create a sample RequestRecord for testing."""
    return RequestRecord(
        conversation_id="test-conversation",
        turn_index=0,
        model_name="test-model",
        start_perf_ns=DEFAULT_START_TIME_NS,
        timestamp_ns=DEFAULT_START_TIME_NS,
        end_perf_ns=DEFAULT_LAST_RESPONSE_NS,
        error=None,
    )


@pytest.fixture
def sample_parsed_record(sample_request_record: RequestRecord) -> ParsedResponseRecord:
    """Create a valid ParsedResponseRecord for testing."""
    responses = [
        ParsedResponse(
            perf_ns=DEFAULT_FIRST_RESPONSE_NS,
            data=TextResponseData(text="Hello"),
        ),
        ParsedResponse(
            perf_ns=DEFAULT_LAST_RESPONSE_NS,
            data=TextResponseData(text=" world"),
        ),
    ]

    return ParsedResponseRecord(
        request=sample_request_record,
        responses=responses,
        input_token_count=DEFAULT_INPUT_TOKENS,
        output_token_count=DEFAULT_OUTPUT_TOKENS,
    )
