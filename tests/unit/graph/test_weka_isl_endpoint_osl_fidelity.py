# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ISL recovery from the wire payload when ``Turn.texts`` is empty (Fix 1).

The graph / weka dispatch path wraps the prompt in a ``raw_payload`` (chat
``messages``) with EMPTY ``Turn.texts``, so the parser's turns-walk yields no
tokenizable text and ``InputSequenceLengthMetric`` would be dropped -- ISL
silently absent. Agentx computes ISL by tokenizing the WIRE payload
(``extract_payload_inputs`` over ``payload_bytes``); the fix mirrors that as a
fallback when ``texts`` is empty. These locks are parser-level and independent
of the weka IR shape.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiperf.common.models import RequestRecord, Text, Turn
from aiperf.common.models.record_models import (
    ParsedResponse,
    RecordContext,
    TextResponseData,
)
from aiperf.common.tokenizer import Tokenizer
from aiperf.records.inference_result_parser import InferenceResultParser

# ---------------------------------------------------------------------------
# Fix 1 -- ISL recovered from the wire payload when Turn.texts is empty
# ---------------------------------------------------------------------------


def _raw_payload_record(messages: list[dict], model: str) -> RequestRecord:
    """A record whose only turn is a raw_payload chat body (no texts).

    Mirrors the graph / weka dispatch shape: ``Turn(raw_payload=...)`` with empty
    ``texts``, and ``request_info.payload_bytes`` = the exact wire JSON.
    """
    payload = {"messages": messages, "max_completion_tokens": 25, "model": model}
    return RequestRecord(
        turns=[Turn(role="user", raw_payload=payload)],
        request_info=RecordContext(
            payload_bytes=orjson.dumps(payload),
            credit_num=0,
            credit_phase="profiling",
            conversation_id="conv-0",
            turn_index=0,
            x_request_id="req-0",
            x_correlation_id="t-1#0|deadbeef|t-1|parent_0|profiling",
        ),
        model_name=model,
    )


@pytest.fixture
def parser(benchmark_run) -> InferenceResultParser:
    """A parser with the REAL chat endpoint (so ``extract_payload_inputs`` runs
    for real over the wire payload) but a word-count fake tokenizer (so no HF
    model download is needed). The endpoint's payload extraction is what the
    Fix 1 fallback exercises; the tokenizer is incidental to the count.
    """
    parser = InferenceResultParser(run=benchmark_run)
    fake = MagicMock(spec=Tokenizer)
    fake.encode.side_effect = lambda text: list(range(len(text.split())))
    parser.get_tokenizer = AsyncMock(return_value=fake)
    return parser


@pytest.mark.asyncio
async def test_isl_from_wire_payload_when_texts_empty(
    parser: InferenceResultParser,
) -> None:
    """A raw_payload chat body (empty texts) still yields a tokenized ISL.

    The turns-walk finds no ``texts``; the fallback runs the endpoint's
    ``extract_payload_inputs`` over the wire payload and tokenizes the recovered
    prompt text, so ISL is the wire-prompt token count rather than ``None``.
    """
    record = _raw_payload_record(
        [{"role": "user", "content": "hello world this is the prompt"}],
        parser.model_endpoint.primary_model_name,
    )
    isl = await parser.compute_input_token_count(record)
    assert isl is not None and isl > 0


@pytest.mark.asyncio
async def test_isl_text_turn_path_unchanged(parser: InferenceResultParser) -> None:
    """A turn carrying ``texts`` uses the turns-walk, not the wire fallback.

    Locks that Fix 1 is a strict fallback: when ``texts`` is present the existing
    space-joined tokenization is used (the linear path is not regressed).
    """
    record = RequestRecord(
        turns=[Turn(role="user", texts=[Text(name="p", contents=["hello world"])])],
        model_name=parser.model_endpoint.primary_model_name,
    )
    isl = await parser.compute_input_token_count(record)
    assert isl is not None and isl > 0


@pytest.mark.asyncio
async def test_isl_none_when_no_texts_and_no_payload(
    parser: InferenceResultParser,
) -> None:
    """No texts and no wire payload -> ISL is None (unavailable), not a crash."""
    record = RequestRecord(
        turns=[Turn(role="user")],
        model_name=parser.model_endpoint.primary_model_name,
    )
    isl = await parser.compute_input_token_count(record)
    assert isl is None


# ---------------------------------------------------------------------------
# OSL -- output tokens still counted on the raw_payload (graph/weka) shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_osl_counted_from_response_text_on_raw_payload_record(
    parser: InferenceResultParser,
) -> None:
    """OSL fidelity: the raw_payload record shape yields a real OUTPUT count.

    The graph/weka dispatch shape (empty ``Turn.texts``) must not degrade the
    client-side output tokenization -- OSL is counted from the response text
    (5 words -> 5 tokens under the word-count fake) alongside the
    wire-recovered ISL in the same ``TokenCounts``.
    """
    record = _raw_payload_record(
        [{"role": "user", "content": "hello world this is the prompt"}],
        parser.model_endpoint.primary_model_name,
    )
    responses = [
        ParsedResponse(
            perf_ns=1, data=TextResponseData(text="five words of model output")
        ),
    ]

    counts = await parser._compute_client_side_token_counts(record, responses)

    assert counts.output == 5
    assert counts.input is not None and counts.input > 0


@pytest.mark.asyncio
async def test_osl_concatenates_streamed_response_chunks(
    parser: InferenceResultParser,
) -> None:
    """Streamed chunks are DELTAS: joined with no separator before tokenizing."""
    record = _raw_payload_record(
        [{"role": "user", "content": "hi"}],
        parser.model_endpoint.primary_model_name,
    )
    responses = [
        ParsedResponse(perf_ns=1, data=TextResponseData(text="hello wor")),
        ParsedResponse(perf_ns=2, data=TextResponseData(text="ld once again")),
    ]

    counts = await parser._compute_client_side_token_counts(record, responses)

    # "hello wor" + "ld once again" -> "hello world once again" -> 4 words.
    assert counts.output == 4
