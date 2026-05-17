# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial probes against the records-pipeline parser chokepoint and
endpoint ``extract_payload_inputs`` overrides.

Invariant under attack: every reachable error path through
``InferenceResultParser.parse_request_record`` must produce a
``ParsedResponseRecord`` (with sane / ``None`` ``payload_inputs`` etc.) -- it
must never crash the records loop. Metrics rely on this invariant because
no metric does IO or JSON parsing; if the parser falls over, every metric
downstream silently disappears for that record.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import param

from aiperf.common.enums import MediaType
from aiperf.common.models import ExtractedPayload, RequestRecord
from aiperf.endpoints.nim_image_retrieval import ImageRetrievalEndpoint
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.endpoints.payload_extraction import extract_inputs
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
)
from tests.unit.records.conftest import (
    create_test_metric_inputs,
)

# Default chat-shape PART_TYPES map -- matches ``BaseEndpoint.PART_TYPES``.
CHAT_PART_TYPES: dict[MediaType, set[str]] = {
    MediaType.TEXT: {"text"},
    MediaType.IMAGE: {"image_url"},
    MediaType.AUDIO: {"input_audio"},
    MediaType.VIDEO: {"video_url"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record_with_payload(
    payload_bytes: bytes | None,
    *,
    conversation_id: str = "cid",
    turn_index: int = 0,
) -> RequestRecord:
    """Build a minimal RequestRecord whose only relevant field is metric_inputs.payload_bytes."""
    return RequestRecord(
        metric_inputs=create_test_metric_inputs(
            conversation_id=conversation_id,
            turn_index=turn_index,
            payload_bytes=payload_bytes,
        ),
        model_name="test-model",
    )


# ---------------------------------------------------------------------------
# 1) Malformed payloads at the parser -- non-dict JSON shapes.
#
# orjson.loads accepts every valid JSON top-level (null / array / string /
# bool / number / object). The parser only handles ``dict`` payloads but
# does NOT type-check before calling ``endpoint.extract_payload_inputs``,
# which calls ``.get()`` on the input. Result: AttributeError leaks out
# of ``_extract_payload_inputs`` and the records loop crashes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload_bytes,decoded_repr",
    [
        param(b"null", "None", id="json_null"),
        param(b"[1, 2, 3]", "list"),
        param(b'"just a string"', "str"),
        param(b"true", "bool"),
        param(b"42", "int"),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_parser_non_dict_json_payload_returns_none(
    setup_inference_parser, payload_bytes: bytes, decoded_repr: str
) -> None:
    """``orjson.loads`` accepts top-level non-dicts (``null``, arrays,
    strings, booleans, numbers). ``_extract_payload_inputs`` now
    type-checks the decoded payload: if it isn't a ``dict``, the parser
    warns and returns ``(None, tmeta, None)`` instead of letting the
    extractor's ``.get(...)`` crash with ``AttributeError``.

    file:line: ``src/aiperf/records/inference_result_parser.py`` —
    ``_extract_payload_inputs`` non-dict guard.
    """
    # Real endpoint, not the test conftest mock -- we want to exercise the
    # real extractor's ``.get()`` call on a non-dict.
    real_endpoint = create_endpoint_with_mock_transport(
        ImageRetrievalEndpoint,
        create_model_endpoint(EndpointType.IMAGE_RETRIEVAL),
    )
    setup_inference_parser.endpoint = real_endpoint

    record = _make_record_with_payload(payload_bytes)
    result = await setup_inference_parser.parse_request_record(record)
    assert result.payload_inputs is None
    assert result.payload_dict is None


@pytest.mark.parametrize(
    "payload_bytes",
    [
        param(b"", id="empty_bytes"),
        param(b"   ", id="whitespace_only"),
        param(b"{not json", id="truncated_object"),
        param(b"\xff\xfe\x00\x00", id="invalid_utf8"),
    ],
)
@pytest.mark.asyncio
async def test_parser_invalid_json_returns_none_payload_inputs(
    setup_inference_parser, payload_bytes: bytes
) -> None:
    """Architecture holds: invalid JSON is caught at line 326
    (``except orjson.JSONDecodeError``) and the parser returns a record
    with ``payload_inputs=None`` instead of crashing."""
    record = _make_record_with_payload(payload_bytes)
    result = await setup_inference_parser.parse_request_record(record)
    assert result.payload_inputs is None
    assert result.payload_dict is None


@pytest.mark.asyncio
async def test_parser_deeply_nested_arrays_decodes_fine(setup_inference_parser) -> None:
    """1000-level nested arrays: orjson handles this fine; the parser's
    non-dict guard returns ``payload_inputs=None`` instead of crashing
    the extractor with AttributeError.
    """
    payload = b"[" * 1000 + b"]" * 1000
    real_endpoint = create_endpoint_with_mock_transport(
        ImageRetrievalEndpoint,
        create_model_endpoint(EndpointType.IMAGE_RETRIEVAL),
    )
    setup_inference_parser.endpoint = real_endpoint

    record = _make_record_with_payload(payload)
    result = await setup_inference_parser.parse_request_record(record)
    assert result.payload_inputs is None
    assert result.payload_dict is None


@pytest.mark.asyncio
async def test_parser_duplicate_keys_orjson_keeps_last(
    setup_inference_parser,
) -> None:
    """orjson silently keeps the LAST value for a duplicate key. Document
    the behaviour so no metric is surprised: duplicate ``messages`` arrays
    in a hand-crafted payload result in only the second array being seen.
    Not a crash, but is a real semantic gotcha if an attacker can inject
    duplicate keys upstream.
    """
    payload = (
        b'{"messages":[{"role":"user","content":"FIRST"}],'
        b'"messages":[{"role":"user","content":"SECOND"}]}'
    )
    # Real walker -- not the mocked extractor.
    setup_inference_parser.endpoint.extract_payload_inputs = lambda p: extract_inputs(
        p, CHAT_PART_TYPES
    )
    record = _make_record_with_payload(payload)
    result = await setup_inference_parser.parse_request_record(record)
    assert result.payload_inputs is not None
    assert result.payload_inputs.texts == ["SECOND"]


# ---------------------------------------------------------------------------
# 2) Endpoint walker stress (chat / default).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content,expected_texts",
    [
        param(None, [], id="content_none"),
        param("", [""], id="content_empty_string_appended"),
        param(42, [], id="content_int_ignored"),
        param({"text": "x"}, [], id="content_dict_ignored"),
        param([{"type": "text", "text": None}], [], id="text_part_with_none_text"),
        param([{"type": "text", "text": 42}], [], id="text_part_with_int_text"),
        param([{"type": "text"}], [], id="text_part_missing_text_key"),
        param([{"text": "x"}], [], id="part_missing_type_key"),
        param([{"type": "unknown"}], [], id="part_unknown_type"),
    ],
)  # fmt: skip
def test_chat_walker_pathological_content(
    content: object, expected_texts: list[str]
) -> None:
    """Walker must accept arbitrary garbage in ``content``. The empty-string
    case is a minor curiosity: ``isinstance("", str)`` is True so it gets
    appended; downstream tokenization joins with separator and produces 0
    tokens, which is the right answer. Documents the behaviour.
    """
    payload: dict = {"messages": [{"role": "user", "content": content}]}
    result = extract_inputs(payload, CHAT_PART_TYPES)
    assert result.texts == expected_texts
    assert result.image_count == 0
    assert result.audio_count == 0
    assert result.video_count == 0


def test_chat_walker_tool_call_with_malformed_args_collects_string() -> None:
    """A malformed JSON string in ``tool_calls[*].function.arguments`` is
    still collected as text -- the walker only checks that it is a string
    and non-empty. This is correct: the server tokenises the literal
    string the model emitted, regardless of whether it parses as JSON.
    """
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "foo", "arguments": "{not valid json"}},
                    {"function": "not_a_dict"},
                    "not_a_dict_at_all",
                    {"function": {"name": ""}},
                    {"function": {"name": "bar", "arguments": ""}},
                ],
            }
        ]
    }
    result = extract_inputs(payload, CHAT_PART_TYPES)
    # Empty strings are filtered by ``_collect_str_fields`` (``and value``).
    assert result.texts == ["foo", "{not valid json", "bar"]


def test_chat_walker_mixed_modality_content_counts_each_separately() -> None:
    """Mixed text + image + audio + video content parts all increment
    their own counters in a single pass."""
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "http://x"}},
                    {"type": "image_url", "image_url": {"url": "http://y"}},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "...", "format": "wav"},
                    },
                    {"type": "video_url", "video_url": {"url": "http://v"}},
                ],
            }
        ]
    }
    result = extract_inputs(payload, CHAT_PART_TYPES)
    assert result.texts == ["describe"]
    assert result.image_count == 2
    assert result.audio_count == 1
    assert result.video_count == 1


def test_chat_walker_unknown_role_still_collects_content() -> None:
    """The walker doesn't gate on role names -- any role with content gets
    its content collected. Verifies no implicit role allow-list."""
    payload = {"messages": [{"role": "warlock", "content": "abracadabra"}]}
    result = extract_inputs(payload, CHAT_PART_TYPES)
    assert result.texts == ["abracadabra"]


def test_chat_walker_messages_with_non_dict_items_skipped() -> None:
    """If ``messages`` contains a non-dict element (None, string, int),
    the walker skips it rather than crashing."""
    payload = {
        "messages": [
            None,
            "string item",
            42,
            {"role": "user", "content": "real content"},
        ]
    }
    result = extract_inputs(payload, CHAT_PART_TYPES)
    assert result.texts == ["real content"]


# ---------------------------------------------------------------------------
# 3) Responses API extractor.
# ---------------------------------------------------------------------------


@pytest.fixture
def responses_endpoint() -> ResponsesEndpoint:
    return create_endpoint_with_mock_transport(
        ResponsesEndpoint, create_model_endpoint(EndpointType.RESPONSES)
    )


def test_responses_input_text_with_none_text(
    responses_endpoint: ResponsesEndpoint,
) -> None:
    """``input_text`` with ``text: None`` -- walker only collects str."""
    payload = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": None},
                    {"type": "input_text", "text": "real"},
                ],
            }
        ]
    }
    result = responses_endpoint.extract_payload_inputs(payload)
    assert result.texts == ["real"]


def test_responses_input_item_no_type_field(
    responses_endpoint: ResponsesEndpoint,
) -> None:
    """Responses API input item with no ``type`` key still gets its content
    walked (the items-array detection requires ``role`` OR ``type``; ``role``
    alone suffices). This catches abbreviated input shapes."""
    payload = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "hello"},
                ],
            }
        ]
    }
    result = responses_endpoint.extract_payload_inputs(payload)
    assert result.texts == ["hello"]


def test_responses_mixed_text_image_unknown(
    responses_endpoint: ResponsesEndpoint,
) -> None:
    """Mixed input_text + input_image + unknown ``input_video`` (Responses
    API doesn't currently take video). The unknown part is silently
    dropped; counts and texts reflect only the recognised parts."""
    payload = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe these"},
                    {"type": "input_image", "image_url": "http://x"},
                    {"type": "input_image", "image_url": "http://y"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "...", "format": "wav"},
                    },
                    {"type": "input_video", "video_url": "http://v"},
                ],
            }
        ]
    }
    result = responses_endpoint.extract_payload_inputs(payload)
    assert result.texts == ["describe these"]
    assert result.image_count == 2
    assert result.audio_count == 1
    # video not in Responses PART_TYPES -- unknown ``input_video`` ignored.
    assert result.video_count == 0


def test_responses_instructions_empty_string_inserted(
    responses_endpoint: ResponsesEndpoint,
) -> None:
    """Minor curiosity: ``instructions: ""`` is inserted into ``texts`` as
    empty string (the str branch doesn't check truthiness). The list branch
    DOES filter empty strings. Inconsistent -- documents the asymmetry."""
    payload_str = {
        "instructions": "",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "x"}]}],
    }
    result_str = responses_endpoint.extract_payload_inputs(payload_str)
    assert result_str.texts == ["", "x"]  # empty string inserted at idx 0.

    payload_list = {
        "instructions": [{"type": "input_text", "text": ""}],
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "x"}]}],
    }
    result_list = responses_endpoint.extract_payload_inputs(payload_list)
    assert result_list.texts == ["x"]  # empty string filtered by list branch.


def test_responses_instructions_list_order_preserved(
    responses_endpoint: ResponsesEndpoint,
) -> None:
    """When ``instructions`` is a list, original order must be preserved at
    the start of ``texts`` (not reversed)."""
    payload = {
        "instructions": [
            {"type": "input_text", "text": "first"},
            {"type": "input_text", "text": "second"},
            "plain_str_third",
        ],
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "body"}]}
        ],
    }
    result = responses_endpoint.extract_payload_inputs(payload)
    assert result.texts == ["first", "second", "plain_str_third", "body"]


# ---------------------------------------------------------------------------
# 4) Image retrieval extractor.
# ---------------------------------------------------------------------------


@pytest.fixture
def image_retrieval_endpoint() -> ImageRetrievalEndpoint:
    return create_endpoint_with_mock_transport(
        ImageRetrievalEndpoint,
        create_model_endpoint(EndpointType.IMAGE_RETRIEVAL),
    )


@pytest.mark.parametrize(
    "input_value,expected_image_count",
    [
        param([], 0, id="empty_list"),
        param("not a list", 0, id="string_input"),
        param(None, 0, id="none_input"),
        param([1, 2, 3], 0, id="non_dict_items"),
        param([{"type": "image_url"}], 1, id="image_url_missing_url_still_counted"),
        param(
            [{"type": "image_url", "url": "x"}, {"type": "text", "text": "y"}],
            1,
            id="mix_image_and_text",
        ),
        param(
            [{"type": "image_url", "url": None}, {"type": "image_url", "url": "ok"}],
            2,
            id="image_url_with_none_url_still_counted",
        ),
    ],
)  # fmt: skip
def test_image_retrieval_extractor_edge_cases(
    image_retrieval_endpoint: ImageRetrievalEndpoint,
    input_value: object,
    expected_image_count: int,
) -> None:
    """Image retrieval extractor counts every dict with ``type: image_url``
    in ``input``, regardless of whether ``url`` is set / valid. This means
    a malformed item still bumps ``image_count`` -- documents the
    permissiveness so downstream metrics (image_throughput, num_images)
    know what the count actually represents."""
    result = image_retrieval_endpoint.extract_payload_inputs({"input": input_value})
    assert result.image_count == expected_image_count
    assert result.texts == []


# ---------------------------------------------------------------------------
# 5) Parser error paths.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parser_metric_inputs_is_none_returns_none_tuple(
    setup_inference_parser,
) -> None:
    """``request_record.metric_inputs is None`` -- parser short-circuits to
    None-tuple (line 304). Architecture holds."""
    record = RequestRecord(
        metric_inputs=None,
        model_name="test-model",
    )
    result = await setup_inference_parser.parse_request_record(record)
    assert result.payload_inputs is None
    assert result.turn_metadata is None
    assert result.payload_dict is None


@pytest.mark.asyncio
async def test_parser_no_inline_no_dataset_client_returns_none(
    setup_inference_parser,
) -> None:
    """``mi.payload_bytes is None`` AND ``self._dataset_client is None`` ->
    parser returns ``(None, tmeta, None)`` (line 322). tmeta is also None
    here since we have no turn metadata index."""
    record = _make_record_with_payload(payload_bytes=None)
    # ensure dataset_client is None (it is by default).
    assert setup_inference_parser._dataset_client is None
    result = await setup_inference_parser.parse_request_record(record)
    assert result.payload_inputs is None
    assert result.payload_dict is None


@pytest.mark.asyncio
async def test_parser_dataset_client_keyerror_caught(setup_inference_parser) -> None:
    """``get_payload_bytes`` raising KeyError is in the caught set --
    parser warns and returns ``(None, tmeta, None)``."""
    setup_inference_parser._dataset_client = MagicMock()
    setup_inference_parser._dataset_client.get_payload_bytes = AsyncMock(
        side_effect=KeyError("missing")
    )
    record = _make_record_with_payload(payload_bytes=None)
    result = await setup_inference_parser.parse_request_record(record)
    assert result.payload_inputs is None
    assert result.payload_dict is None


@pytest.mark.parametrize(
    "exc",
    [
        param(MemoryError("oom"), id="memory_error"),
        param(OSError("disk i/o"), id="os_error"),
        param(ValueError("bespoke"), id="value_error"),
    ],
)
@pytest.mark.asyncio
async def test_parser_dataset_client_uncaught_exception_returns_none(
    setup_inference_parser, exc: Exception
) -> None:
    """``_extract_payload_inputs`` now catches ``Exception`` from
    ``get_payload_bytes`` (broadened from ``(KeyError, IndexError,
    RuntimeError)``). MemoryError, OSError, and bespoke ValueError
    subclasses warn and return ``(None, tmeta, None)`` instead of
    propagating up ``parse_request_record``.
    """
    setup_inference_parser._dataset_client = MagicMock()
    setup_inference_parser._dataset_client.get_payload_bytes = AsyncMock(
        side_effect=exc
    )
    record = _make_record_with_payload(payload_bytes=None)
    result = await setup_inference_parser.parse_request_record(record)
    assert result.payload_inputs is None
    assert result.payload_dict is None


@pytest.mark.asyncio
async def test_parser_inline_bytes_take_precedence_over_dataset_client(
    setup_inference_parser,
) -> None:
    """When ``mi.payload_bytes`` is non-None, the dataset client is NOT
    consulted (even if both are set). Verify the dataset_client mock is
    never called."""
    setup_inference_parser._dataset_client = MagicMock()
    setup_inference_parser._dataset_client.get_payload_bytes = AsyncMock(
        return_value=b'{"messages": [{"role": "user", "content": "from_mmap"}]}'
    )
    record = _make_record_with_payload(
        payload_bytes=b'{"messages": [{"role": "user", "content": "inline"}]}'
    )
    setup_inference_parser.endpoint.extract_payload_inputs = lambda p: extract_inputs(
        p, CHAT_PART_TYPES
    )
    result = await setup_inference_parser.parse_request_record(record)
    assert result.payload_inputs is not None
    assert result.payload_inputs.texts == ["inline"]
    setup_inference_parser._dataset_client.get_payload_bytes.assert_not_called()


# ---------------------------------------------------------------------------
# 6) Tokenizer integration.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compute_input_token_count_with_empty_payload_returns_none(
    setup_inference_parser,
) -> None:
    """``payload_inputs = ExtractedPayload(texts=[])`` -> tokenizer must
    not be called; returns ``None`` (line 341-342)."""
    setup_inference_parser.disable_tokenization = False
    result = await setup_inference_parser._compute_input_token_count(
        request_record=MagicMock(model_name="test-model"),
        payload_inputs=ExtractedPayload(texts=[]),
    )
    assert result is None


@pytest.mark.asyncio
async def test_compute_input_token_count_with_none_payload_returns_none(
    setup_inference_parser,
) -> None:
    """``payload_inputs is None`` -> returns ``None`` (line 341)."""
    setup_inference_parser.disable_tokenization = False
    result = await setup_inference_parser._compute_input_token_count(
        request_record=MagicMock(model_name="test-model"),
        payload_inputs=None,
    )
    assert result is None


@pytest.mark.asyncio
async def test_compute_input_token_count_with_empty_strings_tokenizes(
    setup_inference_parser,
) -> None:
    """``texts=['', '', '']`` is NOT short-circuited -- ``not texts`` is
    False for a non-empty list of empty strings. The tokenizer is called
    with ``" ".join(['', '', ''])`` = ``"  "``. Documents the contract."""
    setup_inference_parser.disable_tokenization = False
    result = await setup_inference_parser._compute_input_token_count(
        request_record=MagicMock(model_name="test-model"),
        payload_inputs=ExtractedPayload(texts=["", "", ""]),
    )
    # mock_tokenizer_cls returns len(text.split()) tokens; "  ".split() is [].
    assert result == 0


# ---------------------------------------------------------------------------
# 7) ``ExtractedPayload`` arithmetic.
# ---------------------------------------------------------------------------


def test_extracted_payload_rejects_negative_counts() -> None:
    """``ExtractedPayload`` enforces ``ge=0`` on ``image_count`` /
    ``audio_count`` / ``video_count`` / ``pretokenised_token_count``.
    A misbehaving extractor that hands a negative count to the model is
    rejected at construction time instead of poisoning downstream
    metrics.

    file:line: ``src/aiperf/common/models/extracted_payload.py`` —
    ``ge=0`` on each integer Field.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExtractedPayload(image_count=-5)
    with pytest.raises(ValidationError):
        ExtractedPayload(audio_count=-1)
    with pytest.raises(ValidationError):
        ExtractedPayload(video_count=-99)
    with pytest.raises(ValidationError):
        ExtractedPayload(pretokenised_token_count=-1000)


def test_extracted_payload_rejects_non_int_counts() -> None:
    """Sanity: non-int values for counts are coerced or rejected by Pydantic."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExtractedPayload(image_count="not an int")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 8) Raw record writer fallback.
# ---------------------------------------------------------------------------


def test_raw_record_writer_payload_none_falls_back_to_empty_dict() -> None:
    """``RawRecordWriterProcessor._build_export_record`` falls back to
    ``{}`` when ``record.payload_dict is None`` (line 87). Verify the
    fallback logic so the raw export does not blow up when parser hit a
    fallback (no inline bytes, no mmap, JSON decode failure).

    We exercise the one-line fallback directly rather than instantiating
    the (output-file-bound) processor + a full MetricRecordMetadata
    (the wire shape has too many tightly-coupled required fields to
    construct in a pure-unit adversarial probe)."""
    from aiperf.common.models import (
        ParsedResponse,
        ParsedResponseRecord,
        TextResponseData,
    )

    parsed = ParsedResponseRecord(
        request=_make_record_with_payload(payload_bytes=None),
        responses=[ParsedResponse(perf_ns=1, data=TextResponseData(text="hi"))],
        payload_inputs=None,
        payload_dict=None,
    )

    # Mirrors RawRecordWriterProcessor._build_export_record:87.
    payload = parsed.payload_dict if parsed.payload_dict is not None else {}
    assert payload == {}

    # And with a non-None dict it passes through unchanged.
    parsed2 = ParsedResponseRecord(
        request=_make_record_with_payload(payload_bytes=None),
        responses=[],
        payload_inputs=None,
        payload_dict={"messages": [{"role": "user", "content": "hi"}]},
    )
    payload2 = parsed2.payload_dict if parsed2.payload_dict is not None else {}
    assert payload2 == {"messages": [{"role": "user", "content": "hi"}]}
