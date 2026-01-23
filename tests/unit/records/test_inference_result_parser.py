# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.models import (
    ErrorDetails,
    ParsedResponse,
    RequestRecord,
    TextResponseData,
    Usage,
)
from tests.unit.records.conftest import create_invalid_record, create_test_request_info


@pytest.fixture
def request_record(sample_turn):
    """Basic request record for testing with sample turn included."""
    return RequestRecord(
        request_info=create_test_request_info(turns=[sample_turn]),
        model_name="test-model",
        turns=[sample_turn],
    )


@pytest.fixture
def spy_tokenizer():
    """Tokenizer spy that tracks encode() calls and returns word-based counts."""
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
    return tokenizer


@pytest.fixture
def server_token_parser(setup_inference_parser):
    """Parser with server token count enabled."""
    setup_inference_parser.user_config.endpoint.use_server_token_count = True
    return setup_inference_parser


def make_parsed_response(
    text: str = "output",
    perf_ns: int = 1000,
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    include_usage: bool = True,
) -> ParsedResponse:
    """Create a ParsedResponse with optional usage data."""
    usage = None
    if include_usage and (prompt_tokens is not None or completion_tokens is not None):
        usage_data: dict = {}
        if prompt_tokens is not None:
            usage_data["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            usage_data["completion_tokens"] = completion_tokens
        if reasoning_tokens is not None:
            usage_data["completion_tokens_details"] = {
                "reasoning_tokens": reasoning_tokens
            }
        usage = Usage(root=usage_data) if usage_data else None

    return ParsedResponse(
        perf_ns=perf_ns,
        data=TextResponseData(text=text) if text else None,
        usage=usage,
    )


def setup_parser_responses(parser, responses: list[ParsedResponse]) -> None:
    """Configure parser to return specific responses."""
    parser.endpoint.extract_response_data = MagicMock(return_value=responses)


@pytest.mark.asyncio
class TestInvalidRecords:
    """Tests for invalid record handling and error conversion."""

    @pytest.mark.parametrize(
        "invalid_config,expected_notes",
        [
            ({"no_responses": True}, ["No responses were received"]),
            ({"bad_start_timestamp": True}, ["Start perf ns timestamp is invalid: -1"]),
            ({"bad_response_timestamps": [-1]}, ["Response 0 perf ns timestamp is invalid: -1"]),
            (
                {"bad_start_timestamp": True, "bad_response_timestamps": [-100, 0]},
                [
                    "Start perf ns timestamp is invalid: -1",
                    "Response 0 perf ns timestamp is invalid: -100",
                    "Response 1 perf ns timestamp is invalid: 0",
                ],
            ),
        ],
        ids=["no_responses", "bad_start", "bad_response_ts", "multiple_errors"],
    )  # fmt: skip
    async def test_converted_to_errors(
        self, setup_inference_parser, sample_turn, invalid_config, expected_notes
    ):
        """Invalid records are converted to error records with appropriate notes."""
        record = create_invalid_record(**invalid_config, turns=[sample_turn])

        result = await setup_inference_parser.parse_request_record(record)

        assert record.has_error
        assert record.error.type == "InvalidInferenceResultError"
        assert "Invalid inference result" in record.error.message

        error_str = str(record.error)
        for note in expected_notes:
            assert note in error_str, (
                f"Expected note '{note}' not found in error: {error_str}"
            )

        assert result.request == record
        assert result.token_counts.input == 8
        assert result.responses == []

    async def test_no_content_responses_converted_to_error(
        self, inference_result_parser, mock_tokenizer, sample_turn
    ):
        """Records with responses but no content are converted to error records."""
        record = create_invalid_record(no_content_responses=True, turns=[sample_turn])

        inference_result_parser.get_tokenizer = AsyncMock(return_value=mock_tokenizer)
        inference_result_parser.get_turn = AsyncMock(return_value=sample_turn)
        inference_result_parser.endpoint = MagicMock()
        setup_parser_responses(
            inference_result_parser,
            [
                ParsedResponse(perf_ns=1000, data=None),
                ParsedResponse(perf_ns=2000, data=None),
            ],
        )

        result = await inference_result_parser.parse_request_record(record)

        assert record.has_error
        assert record.error.type == "InvalidInferenceResultError"
        assert "No responses with actual content" in record.error.message
        assert result.token_counts.input == 8
        assert result.responses == []

    async def test_existing_errors_not_overwritten(
        self, setup_inference_parser, sample_turn
    ):
        """Records with existing errors are not overwritten by create_error_from_invalid."""
        record = create_invalid_record(
            has_error=True, no_responses=True, turns=[sample_turn]
        )

        result = await setup_inference_parser.parse_request_record(record)

        assert record.error.message == "Original error"
        assert record.error.type == "ServerError"
        assert record.error.code == 500
        assert result.token_counts.input == 8
        assert result.responses == []

    @pytest.mark.parametrize(
        "record_type", ["error", "invalid", "processing_exception"]
    )
    async def test_compute_input_tokens(
        self, inference_result_parser, mock_tokenizer, sample_turn, record_type
    ):
        """Input token count is computed for all error scenarios."""
        if record_type == "error":
            record = RequestRecord(
                request_info=create_test_request_info(turns=[sample_turn]),
                model_name="test-model",
                turns=[sample_turn],
                error=ErrorDetails(
                    code=500, message="Server error", type="ServerError"
                ),
            )
        elif record_type == "invalid":
            record = create_invalid_record(no_responses=True, turns=[sample_turn])
        else:
            record = RequestRecord(
                request_info=create_test_request_info(turns=[sample_turn]),
                model_name="test-model",
                turns=[sample_turn],
            )

        inference_result_parser.get_tokenizer = AsyncMock(return_value=mock_tokenizer)
        inference_result_parser.get_turn = AsyncMock(return_value=sample_turn)
        inference_result_parser.extractor = MagicMock()

        if record_type == "processing_exception":
            inference_result_parser.extractor.extract_response_data = AsyncMock(
                side_effect=ValueError("Processing failed")
            )

        result = await inference_result_parser.parse_request_record(record)

        assert result.request == record
        assert result.token_counts.input == 8
        assert result.responses == []
        assert record.error is not None


@pytest.mark.asyncio
class TestServerTokenCount:
    """Tests for --use-server-token-count flag functionality."""

    async def test_uses_server_values(
        self, server_token_parser, request_record, spy_tokenizer
    ):
        """Server token counts are used when flag is enabled."""
        server_token_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)
        setup_parser_responses(
            server_token_parser,
            [
                make_parsed_response(
                    prompt_tokens=150, completion_tokens=50, reasoning_tokens=10
                )
            ],
        )

        result = await server_token_parser.process_valid_record(request_record)

        assert result.token_counts.input == 150
        assert result.token_counts.output == 40  # 50 - 10
        assert result.token_counts.reasoning == 10
        spy_tokenizer.encode.assert_not_called()

    async def test_missing_usage_returns_none(
        self, server_token_parser, request_record
    ):
        """None is returned when server doesn't provide usage."""
        setup_parser_responses(
            server_token_parser, [make_parsed_response(include_usage=False)]
        )

        result = await server_token_parser.process_valid_record(request_record)

        assert result.token_counts.input is None
        assert result.token_counts.output is None
        assert result.token_counts.reasoning is None

    async def test_partial_usage(self, server_token_parser, request_record):
        """Partial usage information is handled correctly."""
        setup_parser_responses(
            server_token_parser, [make_parsed_response(prompt_tokens=150)]
        )

        result = await server_token_parser.process_valid_record(request_record)

        assert result.token_counts.input == 150
        assert result.token_counts.output is None
        assert result.token_counts.reasoning is None

    async def test_streaming_uses_last_value(self, server_token_parser, request_record):
        """Last non-None usage value is used for streaming responses."""
        setup_parser_responses(
            server_token_parser,
            [
                make_parsed_response(text="chunk1", perf_ns=1000, include_usage=False),
                make_parsed_response(
                    text="chunk2", perf_ns=2000, prompt_tokens=150, completion_tokens=20
                ),
                make_parsed_response(
                    text="chunk3", perf_ns=3000, prompt_tokens=150, completion_tokens=50
                ),
            ],
        )

        result = await server_token_parser.process_valid_record(request_record)

        assert result.token_counts.input == 150
        assert result.token_counts.output == 50

    async def test_client_tokenization_when_disabled(
        self, setup_inference_parser, request_record, spy_tokenizer
    ):
        """Client-side tokenization works when flag is disabled."""
        assert not setup_inference_parser.user_config.endpoint.use_server_token_count

        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)
        setup_parser_responses(
            setup_inference_parser,
            [
                make_parsed_response(
                    text="Hello world test", prompt_tokens=999, completion_tokens=999
                )
            ],
        )

        result = await setup_inference_parser.process_valid_record(request_record)

        assert result.token_counts.input == 8
        assert result.token_counts.output == 3
        assert spy_tokenizer.encode.called

    @pytest.mark.parametrize(
        "completion_tokens,reasoning_tokens,expected_output",
        [
            (50, 10, 40),
            (50, None, 50),
            (50, 0, 50),
            (10, 20, 0),
        ],
        ids=["with_reasoning", "no_reasoning", "zero_reasoning", "negative_clamped"],
    )  # fmt: skip
    async def test_output_excludes_reasoning_tokens(
        self,
        setup_inference_parser,
        completion_tokens,
        reasoning_tokens,
        expected_output,
    ):
        """Output count excludes reasoning tokens."""
        responses = [
            make_parsed_response(
                completion_tokens=completion_tokens, reasoning_tokens=reasoning_tokens
            )
        ]
        reasoning_count = setup_inference_parser._extract_server_reasoning_token_count(
            responses
        )
        result = setup_inference_parser._extract_server_output_token_count(
            responses, reasoning_count
        )

        assert result == expected_output

    async def test_warning_when_no_usage_provided(
        self, server_token_parser, request_record
    ):
        """Warning is logged when server provides no usage information."""
        setup_parser_responses(
            server_token_parser, [make_parsed_response(include_usage=False)]
        )

        with patch.object(server_token_parser, "warning") as mock_warning:
            await server_token_parser.process_valid_record(request_record)

            mock_warning.assert_called_once()
            call_args = mock_warning.call_args[0][0]
            assert "Server did not provide token usage information" in call_args


@pytest.mark.asyncio
class TestContextPromptISL:
    """Tests for ISL computation including context prompts."""

    @pytest.mark.parametrize(
        "system_message,user_context_message,expected_tokens",
        [
            ("You are a helpful assistant", None, 13),
            (None, "This is user context for session", 14),
            ("You are a helpful assistant", "This is user context for session", 19),
            (None, None, 8),
            ("", "", 8),
        ],
        ids=[
            "system_only",
            "user_context_only",
            "both_context_messages",
            "no_context",
            "empty_context",
        ],
    )  # fmt: skip
    async def test_isl_with_context_messages(
        self,
        setup_inference_parser,
        sample_turn,
        spy_tokenizer,
        sample_request_info,
        system_message,
        user_context_message,
        expected_tokens,
    ):
        """ISL computation includes context prompts correctly."""
        if system_message is not None:
            sample_request_info.system_message = system_message
        if user_context_message is not None:
            sample_request_info.user_context_message = user_context_message
        sample_request_info.turns = [sample_turn]

        record = RequestRecord(
            model_name="test-model",
            request_info=sample_request_info,
            turns=[sample_turn],
        )
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        result = await setup_inference_parser.compute_input_token_count(record)

        assert result == expected_tokens
        assert spy_tokenizer.encode.call_count == 1

    async def test_isl_context_prompts_for_error_records(
        self, setup_inference_parser, sample_turn, spy_tokenizer, sample_request_info
    ):
        """ISL computation includes context prompts even for error records."""
        sample_request_info.system_message = "You are a helpful assistant"
        sample_request_info.user_context_message = "This is user context for session"
        sample_request_info.turns = [sample_turn]

        record = RequestRecord(
            model_name="test-model",
            request_info=sample_request_info,
            turns=[sample_turn],
            error=ErrorDetails(code=500, message="Server error", type="ServerError"),
        )
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        parsed_record = await setup_inference_parser.parse_request_record(record)

        assert parsed_record.token_counts.input == 19
        assert parsed_record.responses == []
