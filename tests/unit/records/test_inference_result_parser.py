# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from pytest import param

from aiperf.common.models import (
    ErrorDetails,
    ParsedResponse,
    RequestRecord,
    TextResponse,
    TextResponseData,
    Usage,
)
from aiperf.endpoints.openai_chat import ChatEndpoint
from tests.unit.records.conftest import (
    create_invalid_record,
    create_test_request_info,
    rebuild_payload_bytes,
)


@pytest.fixture
def request_record(sample_turn):
    """Basic request record for testing with sample turn included."""
    return RequestRecord(
        request_info=create_test_request_info(turns=[sample_turn]),
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
        # Stub only the response-extraction side; leave ``extract_payload_inputs``
        # untouched so ISL tokenisation still goes through the real
        # ChatEndpoint installed by the fixture.
        inference_result_parser.endpoint.extract_response_data = MagicMock(
            return_value=[
                ParsedResponse(perf_ns=1000, data=None),
                ParsedResponse(perf_ns=2000, data=None),
            ]
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
        self, setup_inference_parser, spy_tokenizer, sample_turn
    ):
        """compute_input_token_count delegates to async _compute_token_count."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)
        record = RequestRecord(
            request_info=create_test_request_info(turns=[sample_turn]),
            model_name="test-model",
        )

        result = await setup_inference_parser.compute_input_token_count(record)

        assert result == 8
        assert spy_tokenizer.encode.call_count == 1

    async def test_client_side_token_counts_uses_async(
        self, setup_inference_parser, spy_tokenizer
    ):
        """_compute_client_side_token_counts calls async _compute_token_count for output/reasoning."""
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)
        record = RequestRecord(
            request_info=create_test_request_info(turns=[]),
            model_name="test-model",
        )

        setup_parser_responses(
            setup_inference_parser,
            [make_parsed_response(text="output tokens here")],
        )

        result = await setup_inference_parser._compute_client_side_token_counts(
            record, [make_parsed_response(text="output tokens here")]
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

    async def test_first_content_chunk_from_per_chunk_usage(
        self, server_token_parser, request_record
    ):
        """With per-chunk (continuous) usage, first_content_chunk_tokens is the
        cumulative completion_tokens reported on the first content chunk -- the
        real bundled-first-chunk count ITL subtracts."""
        setup_parser_responses(
            server_token_parser,
            [
                make_parsed_response(text="hello world", completion_tokens=20),
                make_parsed_response(text=" more", completion_tokens=50),
            ],
        )

        result = await server_token_parser.process_valid_record(request_record)

        assert result.token_counts.first_content_chunk_tokens == 20
        assert result.token_counts.output == 50  # final total, unchanged

    async def test_first_content_chunk_none_without_per_chunk_usage(
        self, server_token_parser, request_record
    ):
        """Without per-chunk usage (only the final chunk carries usage),
        first_content_chunk_tokens is None so ITL falls back to subtracting one."""
        setup_parser_responses(
            server_token_parser,
            [
                make_parsed_response(text="hello", include_usage=False),
                make_parsed_response(text="", completion_tokens=50),  # final usage-only
            ],
        )

        result = await server_token_parser.process_valid_record(request_record)

        assert result.token_counts.first_content_chunk_tokens is None
        assert result.token_counts.output == 50

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
        # Tokeniser reads payload_bytes only; rebuild after mutations so
        # the wire body reflects the new system/user_context/turns.
        rebuild_payload_bytes(sample_request_info)

        record = RequestRecord(
            model_name="test-model",
            request_info=sample_request_info,
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
        rebuild_payload_bytes(sample_request_info)

        record = RequestRecord(
            model_name="test-model",
            request_info=sample_request_info,
            error=ErrorDetails(code=500, message="Server error", type="ServerError"),
        )
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=spy_tokenizer)

        parsed_record = await setup_inference_parser.parse_request_record(record)

        assert parsed_record.token_counts.input == 19
        assert parsed_record.responses == []


@pytest.mark.asyncio
class TestMultimodalMediaCountsEndToEnd:
    """End-to-end: ``payload_bytes`` → ``InferenceResultParser`` →
    ``ParsedResponseRecord.media_counts``.

    Gap in the existing coverage: ``test_image_metrics.py`` hoists
    ``record.media_counts.images`` directly, bypassing the parser.
    If ``extract_payload_inputs`` miscounts or
    ``inference_result_parser.py``'s media-count wiring (line ~145)
    regresses, the old tests pass while downstream metrics silently
    report zero. These tests drive the real parser.
    """

    @pytest.mark.parametrize(
        "images_in_payload,audios_in_payload,videos_in_payload",
        [
            (0, 0, 0),
            (1, 0, 0),
            (3, 0, 0),
            (2, 1, 1),
            (0, 2, 0),
        ],
        ids=["text_only", "one_image", "three_images", "mixed", "audio_only"],
    )
    async def test_media_counts_from_wire_payload(
        self,
        setup_inference_parser,
        mock_tokenizer,
        sample_request_info,
        images_in_payload,
        audios_in_payload,
        videos_in_payload,
    ):
        """Build a chat-shape payload with a known part count, stash the
        bytes on ``request_info.payload_bytes``, and assert the parsed
        record carries the matching counts."""
        content: list[dict] = [{"type": "text", "text": "describe"}]
        for i in range(images_in_payload):
            content.append({"type": "image_url", "image_url": {"url": f"data:img-{i}"}})
        for i in range(audios_in_payload):
            content.append({"type": "input_audio", "input_audio": {"data": f"a{i}"}})
        for i in range(videos_in_payload):
            content.append({"type": "video_url", "video_url": {"url": f"v{i}"}})
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": content}],
        }

        sample_request_info.payload_bytes = orjson.dumps(payload)
        record = RequestRecord(
            model_name="test-model",
            request_info=sample_request_info,
            start_perf_ns=1000,
            timestamp_ns=1000,
            end_perf_ns=2000,
            status=200,
            responses=[],
        )
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=mock_tokenizer)
        setup_inference_parser.endpoint.extract_response_data = MagicMock(
            return_value=[
                ParsedResponse(perf_ns=1500, data=TextResponseData(text="ok"))
            ]
        )

        parsed_record = await setup_inference_parser.parse_request_record(record)

        assert parsed_record.media_counts.images == images_in_payload
        assert parsed_record.media_counts.audios == audios_in_payload
        assert parsed_record.media_counts.videos == videos_in_payload

    async def test_media_counts_zero_when_payload_bytes_missing(
        self,
        setup_inference_parser,
        mock_tokenizer,
        sample_request_info,
    ):
        """Pre-transport error records (payload_bytes is None) still
        produce a ParsedResponseRecord, with zero media counts — no
        media metric should fire for them."""
        sample_request_info.payload_bytes = None
        record = RequestRecord(
            model_name="test-model",
            request_info=sample_request_info,
            start_perf_ns=1000,
            timestamp_ns=1000,
            end_perf_ns=2000,
            status=200,
            responses=[],
        )
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=mock_tokenizer)
        setup_inference_parser.endpoint.extract_response_data = MagicMock(
            return_value=[
                ParsedResponse(perf_ns=1500, data=TextResponseData(text="ok"))
            ]
        )

        parsed_record = await setup_inference_parser.parse_request_record(record)

        assert parsed_record.media_counts.images == 0
        assert parsed_record.media_counts.audios == 0
        assert parsed_record.media_counts.videos == 0


@pytest.mark.asyncio
class TestMalformedResponseEndToEnd:
    """End-to-end: a malformed/error response body (server crash, proxy error)
    must flow through the parser as a clean failed record, not crash the parser
    or get mislabeled as a parser bug."""

    @pytest.mark.parametrize(
        "body",
        [
            param({"choices": [{"message": {"content": "hi"}}]}, id="missing-object"),
            param(
                {"object": "error", "message": "backend died", "code": 500},
                id="vllm-error-object",
            ),
            param({"error": {"message": "Internal Server Error"}}, id="error-body"),
        ],
    )  # fmt: skip
    async def test_malformed_response_recorded_as_failure_not_parser_crash(
        self, inference_result_parser, sample_turn, body
    ):
        # Real ChatEndpoint (not a MagicMock) so the actual extraction path runs.
        inference_result_parser.endpoint = ChatEndpoint(
            model_endpoint=create_test_request_info().model_endpoint
        )
        inference_result_parser.disable_tokenization = True

        record = RequestRecord(
            model_name="test-model",
            request_info=create_test_request_info(turns=[sample_turn]),
            responses=[TextResponse(perf_ns=1000, text=orjson.dumps(body).decode())],
        )

        # Must not raise (pre-fix this crashed with ValueError in extraction).
        result = await inference_result_parser.parse_request_record(record)

        assert record.has_error
        # Honest cause: "no content from server", NOT a parser-internal ValueError.
        assert record.error.type == "InvalidInferenceResultError"
        assert "No responses with actual content" in str(record.error)
        assert "Unsupported OpenAI object type" not in str(record.error)
        assert result.responses == []


@pytest.mark.asyncio
class TestChatTemplateAwareTokenization:
    """``compute_input_token_count`` prefers the HF chat-template path
    when the payload is chat-shape AND the underlying tokenizer has a
    template configured AND ``--apply-chat-template`` was passed. Falls
    back to bare-text encoding otherwise so completions/embeddings/non-HF
    tokenizers and opt-out runs keep working unchanged.
    """

    @pytest.fixture(autouse=True)
    def _enable_apply_chat_template(self, setup_inference_parser):
        """Enable opt-in flag for every test in this class.

        The chat-template path is gated behind ``--apply-chat-template``;
        these tests exercise that path so they need the flag on. A
        separate test class covers the opt-out (flag-off) behavior.
        """
        setup_inference_parser.run.cfg.tokenizer.apply_chat_template = True

    async def test_chat_template_used_when_available(
        self, setup_inference_parser, sample_turn
    ):
        """When ``apply_chat_template`` returns a token list, its length
        is the reported ISL — not the bare text encode."""
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
        # 17 templated tokens (overhead + role markers + prompt content +
        # generation prompt). Distinct from the bare-text encode of 8 so
        # we can prove the template path was taken.
        tokenizer._tokenizer.apply_chat_template.return_value = list(range(17))
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=tokenizer)

        record = RequestRecord(
            request_info=create_test_request_info(turns=[sample_turn]),
            model_name="test-model",
        )
        result = await setup_inference_parser.compute_input_token_count(record)

        assert result == 17
        tokenizer._tokenizer.apply_chat_template.assert_called_once()
        kwargs = tokenizer._tokenizer.apply_chat_template.call_args.kwargs
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        # Bare encode is NOT called when the template path succeeds.
        tokenizer.encode.assert_not_called()

    async def test_chat_template_messages_passed_with_role_and_content(
        self, setup_inference_parser, sample_turn
    ):
        """The messages list passed to ``apply_chat_template`` carries
        ``role`` + ``content`` for each message in the wire payload."""
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
        tokenizer._tokenizer.apply_chat_template.return_value = [0, 1, 2]
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=tokenizer)

        record = RequestRecord(
            request_info=create_test_request_info(turns=[sample_turn]),
            model_name="test-model",
        )
        await setup_inference_parser.compute_input_token_count(record)

        messages_arg = tokenizer._tokenizer.apply_chat_template.call_args.args[0]
        assert isinstance(messages_arg, list)
        assert all(isinstance(m, dict) for m in messages_arg)
        assert all("role" in m and "content" in m for m in messages_arg)
        assert any(m["role"] == "user" for m in messages_arg)

    async def test_chat_template_counts_tool_text_on_top(
        self, setup_inference_parser, sample_turn
    ):
        """Top-level ``tools`` schemas are absent from the role/content
        messages view, so the templated ISL tokenises them separately and adds
        them on top. Assistant ``tool_calls`` ride in ``messages`` (rendered by
        the template), so they are NOT re-counted via ``tool_texts``."""
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
        tokenizer._tokenizer.apply_chat_template.return_value = list(range(17))
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=tokenizer)

        info = create_test_request_info(turns=[sample_turn])
        payload = orjson.loads(info.payload_bytes)
        payload["messages"].append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            }
        )
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "weather lookup",
                    "parameters": {"type": "object"},
                },
            }
        ]
        info.payload_bytes = orjson.dumps(payload)

        record = RequestRecord(request_info=info, model_name="test-model")
        result = await setup_inference_parser.compute_input_token_count(record)

        # Only the top-level tools schema lands in tool_texts; the assistant
        # tool_calls ride in messages and are counted by the template.
        expected_tool_texts = [
            "get_weather",
            "weather lookup",
            '{"type":"object"}',
        ]
        expected_tool_tokens = len(" ".join(expected_tool_texts).split())
        assert result == 17 + expected_tool_tokens
        tokenizer._tokenizer.apply_chat_template.assert_called_once()

    async def test_chat_template_without_tools_adds_nothing(
        self, setup_inference_parser, sample_turn
    ):
        """No tool text in the payload: the templated count stands alone and
        the bare encoder is never invoked."""
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
        tokenizer._tokenizer.apply_chat_template.return_value = list(range(17))
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=tokenizer)

        record = RequestRecord(
            request_info=create_test_request_info(turns=[sample_turn]),
            model_name="test-model",
        )
        result = await setup_inference_parser.compute_input_token_count(record)

        assert result == 17
        tokenizer.encode.assert_not_called()

    async def test_falls_back_when_apply_chat_template_raises(
        self, setup_inference_parser, sample_turn
    ):
        """Models without a chat template configured raise from
        ``apply_chat_template``; the parser must catch and fall back to
        bare-text encoding rather than surface ``None``."""
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
        tokenizer._tokenizer.apply_chat_template.side_effect = ValueError(
            "Cannot use apply_chat_template() because tokenizer.chat_template is not set"
        )
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=tokenizer)

        record = RequestRecord(
            request_info=create_test_request_info(turns=[sample_turn]),
            model_name="test-model",
        )
        result = await setup_inference_parser.compute_input_token_count(record)

        # Falls back to bare-text encode of 4 joined texts (8 words).
        assert result == 8
        tokenizer.encode.assert_called_once()

    async def test_falls_back_when_no_apply_chat_template_attribute(
        self, setup_inference_parser, sample_turn
    ):
        """Tiktoken / non-HF tokenizers don't expose ``apply_chat_template``
        — must fall back silently to bare-text encode."""
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
        # Replace the auto-MagicMock attribute with a real object that
        # genuinely lacks ``apply_chat_template``.

        class TiktokenLike:
            def encode(self, text):
                return list(range(len(text.split())))

        tokenizer._tokenizer = TiktokenLike()
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=tokenizer)

        record = RequestRecord(
            request_info=create_test_request_info(turns=[sample_turn]),
            model_name="test-model",
        )
        result = await setup_inference_parser.compute_input_token_count(record)

        assert result == 8
        tokenizer.encode.assert_called_once()

    async def test_falls_back_when_template_returns_non_list(
        self, setup_inference_parser, sample_turn
    ):
        """Defensive: if ``apply_chat_template`` returns something other
        than a token-list (string when tokenize=False, mock-by-accident),
        fall back rather than report a meaningless count."""
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
        tokenizer._tokenizer.apply_chat_template.return_value = "not a list"
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=tokenizer)

        record = RequestRecord(
            request_info=create_test_request_info(turns=[sample_turn]),
            model_name="test-model",
        )
        result = await setup_inference_parser.compute_input_token_count(record)

        assert result == 8
        tokenizer.encode.assert_called_once()

    async def test_chat_template_none_short_circuits_no_raise(
        self, setup_inference_parser, sample_turn
    ):
        """HF tokenizers with no chat template carry ``chat_template = None``.
        Skip the call entirely (avoids a per-record raise + format on the
        bare-text fallback path) and go straight to text encoding."""
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
        tokenizer._tokenizer.chat_template = None
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=tokenizer)

        record = RequestRecord(
            request_info=create_test_request_info(turns=[sample_turn]),
            model_name="test-model",
        )
        result = await setup_inference_parser.compute_input_token_count(record)

        assert result == 8
        tokenizer._tokenizer.apply_chat_template.assert_not_called()
        tokenizer.encode.assert_called_once()


@pytest.mark.asyncio
class TestChatTemplateOptOutDefault:
    """Without ``--apply-chat-template`` (the default), the parser must
    skip the chat-template path entirely even when the payload is
    chat-shape AND the tokenizer has a template configured. ISL falls
    back to bare-text encoding so reported counts match the user's
    ``--isl`` rather than the wrapped wire payload.
    """

    async def test_apply_chat_template_off_falls_back_to_bare_encode(
        self, setup_inference_parser, sample_turn
    ):
        """Default config has ``apply_chat_template=False``. Templated
        ISL must NOT be reported even when the tokenizer would happily
        produce one."""
        # Default config has apply_chat_template=False.
        assert setup_inference_parser.run.cfg.tokenizer.apply_chat_template is False

        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda x: list(range(len(x.split())))
        # Tokenizer is fully capable of templating, but we shouldn't call it.
        tokenizer._tokenizer.apply_chat_template.return_value = list(range(17))
        setup_inference_parser.get_tokenizer = AsyncMock(return_value=tokenizer)

        record = RequestRecord(
            request_info=create_test_request_info(turns=[sample_turn]),
            model_name="test-model",
        )
        result = await setup_inference_parser.compute_input_token_count(record)

        # Bare-text encode of 4 joined texts (8 words), NOT the 17 templated tokens.
        assert result == 8
        tokenizer._tokenizer.apply_chat_template.assert_not_called()
        tokenizer.encode.assert_called_once()
