# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.models import ErrorDetails, RequestRecord
from tests.unit.records.conftest import create_invalid_record


def create_request_record(has_error=False, model_name="test-model"):
    """Helper to create request records with various states."""
    record = RequestRecord(conversation_id="cid", turn_index=0, model_name=model_name)

    if has_error:
        record.error = ErrorDetails(
            code=500, message="Server error", type="ServerError"
        )

    return record


def setup_parser_for_error_tests(parser, mock_tokenizer, sample_turn):
    """Common setup for error record tests."""
    parser.get_tokenizer = AsyncMock(return_value=mock_tokenizer)
    parser.get_turn = AsyncMock(return_value=sample_turn)
    parser.extractor = MagicMock()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_config,expected_notes",  # fmt: skip
    [
        ({"no_responses": True}, ["No responses were received"]),
        ({"bad_start_timestamp": True}, ["Start perf ns timestamp is invalid: -1"]),
        (
            {"bad_response_timestamps": [-1]},
            ["Response 0 perf ns timestamp is invalid: -1"],
        ),
        (
            {"bad_start_timestamp": True, "bad_response_timestamps": [-100, 0]},
            [
                "Start perf ns timestamp is invalid: -1",
                "Response 0 perf ns timestamp is invalid: -100",
                "Response 1 perf ns timestamp is invalid: 0",
            ],
        ),
    ],
)
async def test_invalid_records_converted_to_errors(
    setup_inference_parser, sample_turn, invalid_config, expected_notes
):
    """Test that invalid records are converted to error records with appropriate notes."""
    record = create_invalid_record(**invalid_config)
    record.turns = [sample_turn]

    result = await setup_inference_parser.parse_request_record(record)

    # Verify error was created
    assert record.has_error
    assert record.error is not None
    assert record.error.type == "InvalidInferenceResultError"
    assert "Invalid inference result" in record.error.message

    # Verify all expected notes are present
    error_str = str(record.error)
    for note in expected_notes:
        assert note in error_str, (
            f"Expected note '{note}' not found in error: {error_str}"
        )

    # Verify parsed result structure
    assert result.request == record
    assert result.input_token_count == 8  # 8 words in sample_turn
    assert result.responses == []


@pytest.mark.asyncio
async def test_no_content_responses_converted_to_error(
    inference_result_parser, mock_tokenizer, sample_turn
):
    """Test that records with responses but no content are converted to error records."""
    from aiperf.common.models import ParsedResponse

    record = create_invalid_record(no_content_responses=True)
    record.turns = [sample_turn]

    # Mock the parser dependencies
    inference_result_parser.get_tokenizer = AsyncMock(return_value=mock_tokenizer)
    inference_result_parser.get_turn = AsyncMock(return_value=sample_turn)

    # Mock extract_response_data to return parsed responses with no content (data=None)
    # This simulates responses that only contain usage metadata or [DONE] markers
    inference_result_parser.endpoint = MagicMock()
    inference_result_parser.endpoint.extract_response_data = MagicMock(
        return_value=[
            ParsedResponse(perf_ns=1000, data=None),  # Usage-only response
            ParsedResponse(perf_ns=2000, data=None),  # [DONE] marker
        ]
    )

    result = await inference_result_parser.parse_request_record(record)

    # Verify error was created after parsing
    assert record.has_error
    assert record.error is not None
    assert record.error.type == "InvalidInferenceResultError"
    assert (
        "No responses with actual content were received from the server (only usage/metadata, null/empty data, or [DONE] markers)"
        in record.error.message
    )

    # Verify parsed result structure
    assert result.request == record
    assert result.input_token_count == 8  # 8 words in sample_turn
    assert result.responses == []


@pytest.mark.asyncio
async def test_existing_errors_not_overwritten(setup_inference_parser, sample_turn):
    """Test that records with existing errors are not overwritten by create_error_from_invalid."""
    record = create_invalid_record(has_error=True, no_responses=True)
    record.turns = [sample_turn]

    result = await setup_inference_parser.parse_request_record(record)

    # Verify original error preserved
    assert record.error.message == "Original error"
    assert record.error.type == "ServerError"
    assert record.error.code == 500

    # Verify parsed result
    assert result.request == record
    assert result.input_token_count == 8
    assert result.responses == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "record_type",
    ["error", "invalid", "processing_exception"],
)
async def test_error_records_compute_input_tokens(
    inference_result_parser, mock_tokenizer, sample_turn, record_type
):
    """Test that input_token_count is computed for all error scenarios."""
    if record_type == "error":
        record = create_request_record(has_error=True)
    elif record_type == "invalid":
        record = create_invalid_record(no_responses=True)
    else:  # processing_exception
        record = create_request_record()

    # Set turns so that input token count can be computed
    record.turns = [sample_turn]

    setup_parser_for_error_tests(inference_result_parser, mock_tokenizer, sample_turn)

    if record_type == "processing_exception":
        inference_result_parser.extractor.extract_response_data = AsyncMock(
            side_effect=ValueError("Processing failed")
        )

    result = await inference_result_parser.parse_request_record(record)

    assert result.request == record
    assert result.input_token_count == 8
    assert result.responses == []
    assert record.error is not None
