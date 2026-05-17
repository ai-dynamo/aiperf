# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.models import (
    ErrorDetails,
    ExtractedPayload,
    ParsedResponse,
    RequestRecord,
    TextResponseData,
    Usage,
)
from tests.unit.records.conftest import (
    create_invalid_record,
    create_test_metric_inputs,
)

_SAMPLE_PAYLOAD_BYTES = b'{"messages":[{"role":"user","content":"sample"}]}'


def _wire_sample_extraction(parser, *, texts: list[str] | None = None) -> None:
    """Configure parser endpoint to extract a known list of texts from any payload.

    Default mirrors sample_turn's 8 words so existing assertions hold.
    """
    if texts is None:
        texts = [
            "Hello world",
            " Test case",
            "Another input",
            " Final message",
        ]
    parser.endpoint.extract_payload_inputs = MagicMock(
        return_value=ExtractedPayload(texts=texts)
    )


@pytest.fixture
def request_record(sample_turn):
    """Basic request record for testing with sample turn included."""
    return RequestRecord(
        metric_inputs=create_test_metric_inputs(payload_bytes=_SAMPLE_PAYLOAD_BYTES),
        model_name="test-model",
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
    setup_inference_parser.run.cfg.endpoint.use_server_token_count = True
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
        usage = Usage(usage_data) if usage_data else None

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
        record = create_invalid_record(
            **invalid_config,
            turns=[sample_turn],
            payload_bytes=_SAMPLE_PAYLOAD_BYTES,
        )
        _wire_sample_extraction(setup_inference_parser)

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
        record = create_invalid_record(
            no_content_responses=True,
            turns=[sample_turn],
            payload_bytes=_SAMPLE_PAYLOAD_BYTES,
        )

        inference_result_parser.get_tokenizer = AsyncMock(return_value=mock_tokenizer)
        inference_result_parser.get_turn = AsyncMock(return_value=sample_turn)
        inference_result_parser.endpoint = MagicMock()
        _wire_sample_extraction(inference_result_parser)
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
            has_error=True,
            no_responses=True,
            turns=[sample_turn],
            payload_bytes=_SAMPLE_PAYLOAD_BYTES,
        )
        _wire_sample_extraction(setup_inference_parser)

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
                metric_inputs=create_test_metric_inputs(
                    payload_bytes=_SAMPLE_PAYLOAD_BYTES
                ),
                model_name="test-model",
                error=ErrorDetails(
                    code=500, message="Server error", type="ServerError"
                ),
            )
        elif record_type == "invalid":
            record = create_invalid_record(
                no_responses=True,
                turns=[sample_turn],
                payload_bytes=_SAMPLE_PAYLOAD_BYTES,
            )
        else:
            record = RequestRecord(
                metric_inputs=create_test_metric_inputs(
                    payload_bytes=_SAMPLE_PAYLOAD_BYTES
                ),
                model_name="test-model",
            )

        inference_result_parser.get_tokenizer = AsyncMock(return_value=mock_tokenizer)
        inference_result_parser.get_turn = AsyncMock(return_value=sample_turn)
        inference_result_parser.endpoint = MagicMock()
        _wire_sample_extraction(inference_result_parser)

        if record_type == "processing_exception":
            inference_result_parser.endpoint.extract_response_data = MagicMock(
                side_effect=ValueError("Processing failed")
            )

        result = await inference_result_parser.parse_request_record(record)

        assert result.request == record
        assert result.token_counts.input == 8
        assert result.responses == []
        assert record.error is not None


@pytest.mark.asyncio
class TestAsyncTokenizerEncode:
    """Tests for async _compute_token_count using asyncio.to_thread."""

    async def test_compute_token_count_returns_correct_count(
        self, setup_inference_parser, spy_tokenizer
    ):
        """_compute_token_count returns the token count via async encode."""
        result = await setup_inference_parser._compute_token_count(
            spy_tokenizer, ["Hello world test"]
        )
        assert result == 3
        spy_tokenizer.encode.assert_called_once_with("Hello world test")

    async def test_compute_token_count_with_separator(
        self, setup_inference_parser, spy_tokenizer
    ):
        """Texts are joined with the separator before encoding."""
        result = await setup_inference_parser._compute_token_count(
            spy_tokenizer, ["Hello", "world", "test"], separator=" "
        )
        assert result == 3
        spy_tokenizer.encode.assert_called_once_with("Hello world test")

    async def test_compute_token_count_empty_texts(
        self, setup_inference_parser, spy_tokenizer
    ):
        """Empty text list returns None without calling encode."""
        result = await setup_inference_parser._compute_token_count(spy_tokenizer, [])
        assert result is None
        spy_tokenizer.encode.assert_not_called()

    async def test_compute_token_count_single_text(
        self, setup_inference_parser, spy_tokenizer
    ):
        """Single text with no separator works correctly."""
        result = await setup_inference_parser._compute_token_count(
            spy_tokenizer, ["one"]
        )
        assert result == 1

    async def test_compute_token_count_called_via_compute_input(
        self, setup_inference_parser, spy_tokenizer
    ):
        """_compute_input_token_count delegates to async _compute_token_count."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        result = await setup_inference_parser._compute_input_token_count(
            RequestRecord(model_name="test-model"),
            ExtractedPayload(
                texts=["Hello world", "Test case", "Another input", "Final message"]
            ),
        )

        assert result == 8
        assert spy_tokenizer.encode.call_count == 1

    async def test_client_side_token_counts_uses_async(
        self, setup_inference_parser, spy_tokenizer
    ):
        """_compute_client_side_token_counts calls async _compute_token_count for output/reasoning."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)
        record = RequestRecord(
            metric_inputs=create_test_metric_inputs(),
            model_name="test-model",
        )

        setup_parser_responses(
            setup_inference_parser,
            [make_parsed_response(text="output tokens here")],
        )

        result = await setup_inference_parser._compute_client_side_token_counts(
            record, [make_parsed_response(text="output tokens here")], None
        )

        assert result.output == 3
        assert spy_tokenizer.encode.called


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
        assert not setup_inference_parser.run.cfg.endpoint.use_server_token_count

        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)
        setup_parser_responses(
            setup_inference_parser,
            [
                make_parsed_response(
                    text="Hello world test", prompt_tokens=999, completion_tokens=999
                )
            ],
        )

        result = await setup_inference_parser.process_valid_record(
            request_record,
            payload_inputs=ExtractedPayload(
                texts=["Hello world", " Test case", "Another input", " Final message"]
            ),
        )

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
        token_counts = await setup_inference_parser._compute_server_token_counts(
            responses
        )

        assert token_counts.output == expected_output

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
class TestPayloadBytesISL:
    """Tokenizer reads from ``metric_inputs.payload_bytes`` via the parser's
    single-extraction chokepoint.

    payload_bytes is the canonical wire payload populated by
    ``inference_client`` for every dispatched request (unless the records-side
    mmap client will resolve it). The tokenizer reads ``payload_inputs.texts``
    that ``_extract_payload_inputs`` produced; it does not re-extract.
    """

    async def test_compute_input_uses_payload_inputs_texts(
        self, setup_inference_parser, spy_tokenizer
    ):
        """Parser-extracted ExtractedPayload feeds the tokenizer."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        record = RequestRecord(
            metric_inputs=create_test_metric_inputs(payload_bytes=b'{"messages":[]}'),
            model_name="test-model",
        )

        result = await setup_inference_parser._compute_input_token_count(
            record, ExtractedPayload(texts=["hello world from payload"])
        )

        assert result == 4
        assert spy_tokenizer.encode.call_count == 1

    async def test_compute_input_payload_inputs_none_returns_none(
        self, setup_inference_parser, spy_tokenizer
    ):
        """No payload_inputs (extraction skipped or failed) -> None."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        record = RequestRecord(model_name="test-model")
        result = await setup_inference_parser._compute_input_token_count(record, None)

        assert result is None
        spy_tokenizer.encode.assert_not_called()

    async def test_compute_input_empty_texts_returns_none(
        self, setup_inference_parser, spy_tokenizer
    ):
        """Empty extracted texts -> None (no tokenization)."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        record = RequestRecord(model_name="test-model")
        result = await setup_inference_parser._compute_input_token_count(
            record, ExtractedPayload(texts=[])
        )

        assert result is None
        spy_tokenizer.encode.assert_not_called()

    async def test_compute_input_uses_pretokenised_count_without_texts(
        self, setup_inference_parser, spy_tokenizer
    ):
        """Pre-tokenised input contributes to ISL without tokenizer calls."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        record = RequestRecord(model_name="test-model")
        result = await setup_inference_parser._compute_input_token_count(
            record, ExtractedPayload(pretokenised_token_count=4)
        )

        assert result == 4
        spy_tokenizer.encode.assert_not_called()

    async def test_compute_input_adds_pretokenised_count_to_text_count(
        self, setup_inference_parser, spy_tokenizer
    ):
        """Mixed text and pre-tokenised inputs both contribute to ISL."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        record = RequestRecord(model_name="test-model")
        result = await setup_inference_parser._compute_input_token_count(
            record, ExtractedPayload(texts=["hello world"], pretokenised_token_count=3)
        )

        assert result == 5
        spy_tokenizer.encode.assert_called_once_with("hello world")


@pytest.mark.asyncio
class TestParseRecordSingleExtraction:
    """The parser is THE single chokepoint for payload IO + JSON decode +
    ``extract_payload_inputs``. Every downstream metric reads stashed fields
    off ``ParsedResponseRecord`` instead of re-resolving.
    """

    async def test_parse_record_extracts_payload_inputs_once(
        self, setup_inference_parser, spy_tokenizer
    ):
        """``extract_payload_inputs`` is called exactly once per record even
        though downstream metrics + tokenizer + raw_export all need it."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)
        extracted = ExtractedPayload(texts=["hello world"], image_count=2)
        setup_inference_parser.endpoint.extract_payload_inputs = MagicMock(
            return_value=extracted
        )
        setup_parser_responses(
            setup_inference_parser, [make_parsed_response(text="reply")]
        )

        record = RequestRecord(
            metric_inputs=create_test_metric_inputs(
                payload_bytes=b'{"messages":[{"role":"user","content":"hello world"}]}'
            ),
            model_name="test-model",
        )

        parsed = await setup_inference_parser.parse_request_record(record)

        # Exactly one extraction call regardless of how many consumers want it.
        setup_inference_parser.endpoint.extract_payload_inputs.assert_called_once()
        # Stashed results visible to downstream consumers.
        assert parsed.payload_inputs is extracted
        assert parsed.payload_dict == {
            "messages": [{"role": "user", "content": "hello world"}]
        }

    async def test_parse_record_looks_up_turn_metadata(
        self, setup_inference_parser, spy_tokenizer
    ):
        """The parser populates ``turn_metadata`` from its conversation-indexed turns."""
        from aiperf.common.models import TurnMetadata

        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)
        setup_inference_parser.endpoint.extract_payload_inputs = MagicMock(
            return_value=ExtractedPayload(texts=[])
        )
        setup_parser_responses(
            setup_inference_parser, [make_parsed_response(text="reply")]
        )

        tmeta = TurnMetadata(max_tokens=42, audio_duration_seconds=3.5)
        setup_inference_parser.on_dataset_configured(
            turn_metadata_by_conversation={"cid": (tmeta,)},
            dataset_client=None,
        )

        record = RequestRecord(
            metric_inputs=create_test_metric_inputs(payload_bytes=b'{"messages":[]}'),
            model_name="test-model",
        )

        parsed = await setup_inference_parser.parse_request_record(record)

        assert parsed.turn_metadata is tmeta
        assert parsed.turn_metadata.max_tokens == 42

    async def test_parse_record_invalid_json_degrades_gracefully(
        self, setup_inference_parser, spy_tokenizer
    ):
        """Corrupted payload bytes -> payload_inputs/payload_dict are None, no
        crash, warning logged. The extractor must not even be called."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)
        setup_inference_parser.endpoint.extract_payload_inputs = MagicMock(
            side_effect=AssertionError("must not extract from invalid JSON")
        )
        setup_parser_responses(
            setup_inference_parser, [make_parsed_response(text="reply")]
        )

        record = RequestRecord(
            metric_inputs=create_test_metric_inputs(payload_bytes=b"{invalid json"),
            model_name="test-model",
        )

        parsed = await setup_inference_parser.parse_request_record(record)

        assert parsed.payload_inputs is None
        assert parsed.payload_dict is None
        setup_inference_parser.endpoint.extract_payload_inputs.assert_not_called()
        setup_inference_parser.warning.assert_called()

    async def test_parse_record_mmap_resolution_when_payload_bytes_none(
        self, setup_inference_parser, spy_tokenizer
    ):
        """When ``metric_inputs.payload_bytes`` is None, the parser asks its
        dataset_client for the bytes. Used by the PAYLOAD_BYTES mmap path."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)
        setup_inference_parser.endpoint.extract_payload_inputs = MagicMock(
            return_value=ExtractedPayload(texts=["from mmap"])
        )
        setup_parser_responses(
            setup_inference_parser, [make_parsed_response(text="reply")]
        )

        fake_client = MagicMock()
        fake_client.get_payload_bytes = AsyncMock(
            return_value=b'{"messages":[{"role":"user","content":"from mmap"}]}'
        )
        setup_inference_parser.on_dataset_configured(
            turn_metadata_by_conversation={},
            dataset_client=fake_client,
        )

        record = RequestRecord(
            metric_inputs=create_test_metric_inputs(payload_bytes=None),
            model_name="test-model",
        )

        parsed = await setup_inference_parser.parse_request_record(record)

        fake_client.get_payload_bytes.assert_awaited_once_with("cid", 0)
        assert parsed.payload_inputs is not None
        assert parsed.payload_dict == {
            "messages": [{"role": "user", "content": "from mmap"}]
        }

    async def test_parse_record_no_metric_inputs_skips_extraction(
        self, setup_inference_parser, spy_tokenizer
    ):
        """Records with no metric_inputs (legacy / error pre-transport) get
        ``payload_inputs=None`` cleanly, no crash."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)
        setup_inference_parser.endpoint.extract_payload_inputs = MagicMock(
            side_effect=AssertionError("must not extract without metric_inputs")
        )
        setup_parser_responses(
            setup_inference_parser, [make_parsed_response(text="reply")]
        )

        record = RequestRecord(
            model_name="test-model",
        )
        # metric_inputs intentionally left None.

        parsed = await setup_inference_parser.parse_request_record(record)

        assert parsed.payload_inputs is None
        assert parsed.payload_dict is None
        assert parsed.turn_metadata is None
        setup_inference_parser.endpoint.extract_payload_inputs.assert_not_called()
