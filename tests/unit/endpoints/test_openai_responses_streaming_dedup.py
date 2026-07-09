# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: the Responses-API streaming parser must not double-count
client-side output tokens.

A streaming Responses turn carries the assistant text twice - once as the
chain of ``response.output_text.delta`` events and again, in full, as the
terminal ``response.output_text.done`` event. Before the fix,
``extract_response_data`` emitted a ``TextResponseData`` for BOTH, so
``InferenceResultParser`` tokenised the text twice and reported ~2x OSL /
output-token-throughput. These tests pin the invariant that the streamed
output is tokenised exactly once while the ``done`` event stays the sole
carrier for the no-delta case (non-streaming / server that only emits the
terminal event).
"""

from __future__ import annotations

import orjson
import pytest

from aiperf.common.models import ParsedResponse, RequestRecord, TextResponse
from aiperf.common.models.record_models import (
    ReasoningResponseData,
    TextResponseData,
    ToolCallResponseData,
)
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.harness.fake_tokenizer import FakeTokenizer
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
)

# FakeTokenizer emits one token per 4 characters (round(len(text) / 4)), so
# token counts below are hand-derivable from string lengths.
#   OUTPUT_TEXT  -> 32 chars -> 8 tokens
#   REASONING    -> 16 chars -> 4 tokens
#   TOOL_ARGS    -> 13 chars -> 3 tokens
OUTPUT_DELTAS = ["The quick ", "brown fox ", "jumps over!!"]
OUTPUT_TEXT = "".join(OUTPUT_DELTAS)
EXPECTED_OUTPUT_TOKENS = 8

REASONING_DELTAS = ["Reasoning ", "here.."]
REASONING_TEXT = "".join(REASONING_DELTAS)
EXPECTED_REASONING_TOKENS = 4

TOOL_ARG_DELTAS = ['{"loc":', '"NYC"}']
TOOL_ARGS_TEXT = "".join(TOOL_ARG_DELTAS)
EXPECTED_TOOL_TOKENS = 3


def _record(events: list[dict]) -> RequestRecord:
    """Wrap a list of Responses-API event dicts as an SSE-ordered RequestRecord."""
    responses = [
        TextResponse(perf_ns=i, text=orjson.dumps(event).decode())
        for i, event in enumerate(events)
    ]
    return RequestRecord(responses=responses)


def _client_output_and_reasoning_texts(
    parsed: list[ParsedResponse],
) -> tuple[list[str], list[str]]:
    """Faithful mirror of ``InferenceResultParser._parse_output_and_reasoning_texts``.

    Reasoning items contribute their reasoning to the reasoning bucket and any
    ``content`` to the output bucket; every other data-bearing item contributes
    ``get_text()`` to the output bucket. This is the exact split the production
    token-count path applies to the list ``extract_response_data`` returns.
    """
    output_texts: list[str] = []
    reasoning_texts: list[str] = []
    for response in parsed:
        if not response.data:
            continue
        if isinstance(response.data, ReasoningResponseData):
            if response.data.reasoning:
                reasoning_texts.append(response.data.reasoning)
            if response.data.content:
                output_texts.append(response.data.content)
        else:
            output_texts.append(response.data.get_text())
    return output_texts, reasoning_texts


def _token_count(tokenizer: FakeTokenizer, texts: list[str]) -> int | None:
    """Mirror of ``InferenceResultParser._compute_token_count`` (separator='')."""
    if not texts:
        return None
    return len(tokenizer.encode("".join(texts)))


class TestResponsesStreamingOutputTokenDedup:
    @pytest.fixture
    def endpoint(self) -> ResponsesEndpoint:
        me = create_model_endpoint(EndpointType.RESPONSES, streaming=True)
        return create_endpoint_with_mock_transport(ResponsesEndpoint, me)

    @pytest.fixture
    def tokenizer(self) -> FakeTokenizer:
        return FakeTokenizer()

    def test_streaming_deltas_plus_done_not_double_counted(self, endpoint, tokenizer):
        """Deltas + a terminal ``output_text.done`` must tokenise to the
        delta-only count, NOT twice it.

        This is the core regression: before the fix ``extract_response_data``
        returned the 3 delta ``TextResponseData`` plus a 4th for the ``done``
        event whose ``text`` repeats the full output, so the concatenation
        (and hence the token count) doubled. The dedup keeps the ``done`` as a
        ZERO-TEXT placeholder at its own timestamp (so content-timing metrics
        are unchanged) while contributing no output tokens.
        """
        events = [
            {"type": "response.created", "response": {}},
            {"type": "response.in_progress", "response": {}},
            {"type": "response.output_item.added", "item": {"type": "message"}},
            {"type": "response.content_part.added"},
            *[
                {"type": "response.output_text.delta", "delta": d}
                for d in OUTPUT_DELTAS
            ],
            {"type": "response.output_text.done", "text": OUTPUT_TEXT},
            {"type": "response.content_part.done"},
            {"type": "response.output_item.done", "item": {"type": "message"}},
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 5, "output_tokens": 8}},
            },
        ]
        done_idx = next(
            i for i, e in enumerate(events) if e["type"] == "response.output_text.done"
        )

        parsed = endpoint.extract_response_data(_record(events))

        # The 3 deltas carry text; the redundant ``done`` is kept only as a
        # zero-text placeholder (no doubled tokens), preserving its timestamp.
        text_items = [p for p in parsed if isinstance(p.data, TextResponseData)]
        assert [p.data.text for p in text_items] == [*OUTPUT_DELTAS, ""]
        # RequestLatencyMetric uses content_responses[-1].perf_ns (content =
        # responses with truthy data; the usage-only completed event is
        # excluded). The zero-text placeholder keeps the done timestamp as the
        # last content response, so latency/inter-chunk timing is unchanged.
        last_content = [p for p in parsed if p.data][-1]
        assert last_content.perf_ns == done_idx
        assert last_content.data.get_text() == ""

        output_texts, reasoning_texts = _client_output_and_reasoning_texts(parsed)
        # Reconstructed output is the true text exactly ONCE (not doubled).
        assert "".join(output_texts) == OUTPUT_TEXT
        assert reasoning_texts == []

        output_tokens = _token_count(tokenizer, output_texts)
        deltas_only_tokens = _token_count(tokenizer, OUTPUT_DELTAS)
        assert output_tokens == deltas_only_tokens == EXPECTED_OUTPUT_TOKENS
        # The bug produced double this. Guard against a silent regression.
        assert output_tokens != EXPECTED_OUTPUT_TOKENS * 2

        # The stream-terminal usage is still surfaced for server-token-count mode.
        assert any(p.usage is not None for p in parsed)

    def test_non_streaming_full_response_extracts_output_once(
        self, endpoint, tokenizer
    ):
        """The non-streaming full-response object path still yields the output
        exactly once - this is why the ``done`` handler can't simply be dropped.
        """
        events = [
            {
                "object": "response",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": OUTPUT_TEXT}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 8},
            }
        ]

        parsed = endpoint.extract_response_data(_record(events))

        output_texts, _ = _client_output_and_reasoning_texts(parsed)
        assert "".join(output_texts) == OUTPUT_TEXT
        assert _token_count(tokenizer, output_texts) == EXPECTED_OUTPUT_TOKENS

    def test_done_only_no_deltas_stays_sole_carrier(self, endpoint, tokenizer):
        """A degenerate stream that emits ONLY ``output_text.done`` (no deltas)
        must still count the output once - the ``done`` event is the sole
        carrier here, so suppressing it unconditionally would zero out OSL.
        """
        events = [
            {"type": "response.created", "response": {}},
            {"type": "response.output_text.done", "text": OUTPUT_TEXT},
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 5, "output_tokens": 8}},
            },
        ]

        parsed = endpoint.extract_response_data(_record(events))

        text_items = [p for p in parsed if isinstance(p.data, TextResponseData)]
        assert len(text_items) == 1

        output_texts, _ = _client_output_and_reasoning_texts(parsed)
        assert "".join(output_texts) == OUTPUT_TEXT
        assert _token_count(tokenizer, output_texts) == EXPECTED_OUTPUT_TOKENS

    def test_done_only_part_not_suppressed_by_a_sibling_streamed_part(
        self, endpoint, tokenizer
    ):
        """De-dup must be per ``(output_index, content_index)``, not global.

        A single response can carry multiple output parts. If part 0 streams
        deltas + done and part 1 emits only a ``done`` (deltas dropped under
        load, or the server streamed just one part), a global "saw a delta"
        flag would wrongly suppress part 1's ``done`` and drop its text. Each
        part's output must be counted exactly once: part 0 via its deltas
        (its ``done`` skipped), part 1 via its sole ``done``.
        """
        part0_deltas = ["Alpha ", "beta"]
        part0_text = "".join(part0_deltas)
        part1_text = "Gamma delta epsilon"
        events = [
            {"type": "response.created", "response": {}},
            *[
                {
                    "type": "response.output_text.delta",
                    "delta": d,
                    "output_index": 0,
                    "content_index": 0,
                }
                for d in part0_deltas
            ],
            {
                "type": "response.output_text.done",
                "text": part0_text,
                "output_index": 0,
                "content_index": 0,
            },
            # Part 1: done-only (no deltas) at a different output_index.
            {
                "type": "response.output_text.done",
                "text": part1_text,
                "output_index": 1,
                "content_index": 0,
            },
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 5, "output_tokens": 8}},
            },
        ]

        parsed = endpoint.extract_response_data(_record(events))
        output_texts, _ = _client_output_and_reasoning_texts(parsed)

        # Both parts present, neither doubled: part 0's deltas + part 1's done.
        assert "".join(output_texts) == part0_text + part1_text
        assert _token_count(tokenizer, output_texts) == _token_count(
            tokenizer, [*part0_deltas, part1_text]
        )

    def test_reasoning_alongside_output_counted_separately(self, endpoint, tokenizer):
        """Reasoning deltas and output deltas (plus a redundant output
        ``done``) are bucketed separately: reasoning tokens are not conflated
        with output, and output is still not doubled.
        """
        events = [
            *[
                {"type": "response.reasoning_text.delta", "delta": d}
                for d in REASONING_DELTAS
            ],
            *[
                {"type": "response.output_text.delta", "delta": d}
                for d in OUTPUT_DELTAS
            ],
            {"type": "response.output_text.done", "text": OUTPUT_TEXT},
            {
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 12,
                        "reasoning_tokens": 4,
                    }
                },
            },
        ]

        parsed = endpoint.extract_response_data(_record(events))

        output_texts, reasoning_texts = _client_output_and_reasoning_texts(parsed)
        assert "".join(output_texts) == OUTPUT_TEXT
        assert "".join(reasoning_texts) == REASONING_TEXT

        assert _token_count(tokenizer, output_texts) == EXPECTED_OUTPUT_TOKENS
        assert _token_count(tokenizer, reasoning_texts) == EXPECTED_REASONING_TOKENS

    def test_function_call_arguments_streaming_still_works(self, endpoint, tokenizer):
        """The mirrored sibling: ``function_call_arguments`` deltas are emitted
        once and the terminal ``function_call_arguments.done`` stays structural
        (no extra emission). The output_text dedup must not disturb this path.
        """
        events = [
            {
                "type": "response.output_item.added",
                "item": {"type": "function_call", "name": "get_weather"},
            },
            *[
                {"type": "response.function_call_arguments.delta", "delta": d}
                for d in TOOL_ARG_DELTAS
            ],
            {
                "type": "response.function_call_arguments.done",
                "arguments": TOOL_ARGS_TEXT,
            },
            {"type": "response.completed", "response": {}},
        ]

        parsed = endpoint.extract_response_data(_record(events))

        tool_items = [p for p in parsed if isinstance(p.data, ToolCallResponseData)]
        # Two arg deltas surface; the ``done`` event adds nothing.
        assert len(tool_items) == len(TOOL_ARG_DELTAS)

        output_texts, _ = _client_output_and_reasoning_texts(parsed)
        assert "".join(output_texts) == TOOL_ARGS_TEXT
        assert _token_count(tokenizer, output_texts) == EXPECTED_TOOL_TOKENS
