#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory comparison: msgspec Struct vs Pydantic for the full worker data pipeline.

Defines msgspec equivalents of ALL Pydantic models in the hot path:
  SSEMessage, ParsedResponse, RequestRecord, RequestInfo,
  Turn, Text, ModelEndpointInfo, EndpointInfo, etc.

Measures memory side-by-side under identical load including nested
conversation/turn data that workers hold per in-flight request.

Fairness measures:
  - Both sides pre-warm class metadata before measurement
  - Both sides create identical fields and nested data structures
  - Every Pydantic default_factory (list/dict) is matched by an explicit
    fresh mutable on the msgspec side
  - Shared objects (ModelEndpointInfo) are created OUTSIDE the measured block
  - Assertions verify both sides produce the same number of objects

Run: uv run python dev/benchmarks/msgspec_vs_pydantic_memory.py
"""

import time
import tracemalloc
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import orjson
from msgspec import Struct

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models import (
    ParsedResponse,
    RequestRecord,
    SSEMessage,
    TextResponseData,
)
from aiperf.common.models.dataset_models import Text, Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo, SSEField, TokenCounts
from aiperf.endpoints.openai_chat import ChatEndpoint

# =============================================================================
# msgspec Struct equivalents — SSE / response pipeline
# =============================================================================


class MsgspecSSEField(Struct, frozen=True):
    """Equivalent of SSEField (currently a NamedTuple)."""

    name: str
    value: str | None = None


class MsgspecSSEMessage(Struct, kw_only=True):
    """Equivalent of SSEMessage (Pydantic BaseInferenceServerResponse). 2 fields."""

    perf_ns: int
    packets: list[MsgspecSSEField]

    def extract_data_content(self) -> str:
        return "\n".join(p.value for p in self.packets if p.name == "data" and p.value)

    def get_json(self) -> dict[str, Any] | None:
        text = self.extract_data_content()
        if text in ("", "[DONE]"):
            return None
        try:
            return orjson.loads(text)
        except orjson.JSONDecodeError:
            return None

    @classmethod
    def parse(cls, raw: str, perf_ns: int) -> "MsgspecSSEMessage":
        packets = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) < 2:
                packets.append(MsgspecSSEField(name=parts[0].strip()))
            else:
                packets.append(
                    MsgspecSSEField(name=parts[0].strip(), value=parts[1].strip())
                )
        return cls(perf_ns=perf_ns, packets=packets)


class MsgspecTextResponseData(Struct, frozen=True):
    """Equivalent of TextResponseData. 1 field."""

    text: str


class MsgspecReasoningResponseData(Struct, frozen=True):
    """Equivalent of ReasoningResponseData. 2 fields."""

    content: str | None = None
    reasoning: str | None = None


class MsgspecParsedResponse(Struct, kw_only=True):
    """Equivalent of ParsedResponse. All 5 fields."""

    perf_ns: int
    data: MsgspecTextResponseData | MsgspecReasoningResponseData | None = None
    usage: dict[str, Any] | None = None
    sources: dict[str, Any] | list[Any] | None = None
    metadata: dict[str, Any] | None = (
        None  # Builders pass {} to match default_factory=dict
    )


class MsgspecTokenCounts(Struct, kw_only=True):
    """Equivalent of TokenCounts. 3 fields."""

    input: int | None = None
    output: int | None = None
    reasoning: int | None = None


class MsgspecErrorDetails(Struct, kw_only=True):
    """Equivalent of ErrorDetails. 5 fields."""

    code: int | None = None
    type: str | None = None
    message: str = ""
    cause: str | None = None
    details: Any | None = None


# =============================================================================
# msgspec Struct equivalents — nested conversation / endpoint data
# =============================================================================


class MsgspecText(Struct, kw_only=True):
    """Equivalent of Text (Media subclass). 2 fields."""

    name: str = ""
    contents: list[str] = []


class MsgspecMedia(Struct, kw_only=True):
    """Equivalent of Image/Audio/Video (Media subclass). 2 fields."""

    name: str = ""
    contents: list[str] = []


class MsgspecTurn(Struct, kw_only=True):
    """Equivalent of Turn. All 9 fields."""

    model: str | None = None
    role: str | None = None
    timestamp: int | float | None = None
    delay: int | float | None = None
    max_tokens: int | None = None
    texts: list[MsgspecText] = []
    images: list[MsgspecMedia] = []
    audios: list[MsgspecMedia] = []
    videos: list[MsgspecMedia] = []


class MsgspecModelInfo(Struct, kw_only=True):
    """Equivalent of ModelInfo. 2 fields."""

    name: str
    version: str | None = None


class MsgspecModelListInfo(Struct, kw_only=True):
    """Equivalent of ModelListInfo. 2 fields."""

    models: list[MsgspecModelInfo]
    model_selection_strategy: str


class MsgspecEndpointInfo(Struct, kw_only=True):
    """Equivalent of EndpointInfo. All 13 fields."""

    type: str = "openai_chat"
    base_urls: list[str] = []
    custom_endpoint: str | None = None
    url_params: dict[str, Any] | None = None
    streaming: bool = True
    headers: list[tuple[str, str]] = []
    api_key: str | None = None
    ssl_options: dict[str, Any] | None = None
    timeout: float = 600.0
    extra: list[tuple[str, Any]] = []
    use_legacy_max_tokens: bool = False
    use_server_token_count: bool = False
    connection_reuse_strategy: str = "per_worker"


class MsgspecModelEndpointInfo(Struct, kw_only=True):
    """Equivalent of ModelEndpointInfo. 3 fields."""

    models: MsgspecModelListInfo
    endpoint: MsgspecEndpointInfo
    transport: str | None = None


class MsgspecRequestInfo(Struct, kw_only=True):
    """Equivalent of RequestInfo. All 17 fields."""

    model_endpoint: MsgspecModelEndpointInfo
    turns: list[MsgspecTurn]
    turn_index: int = 0
    endpoint_headers: dict[str, str] = {}  # Builders pass {} explicitly
    endpoint_params: dict[str, str] = {}  # Builders pass {} explicitly
    credit_num: int = 0
    credit_phase: str = "profiling"
    cancel_after_ns: int | None = None
    x_request_id: str = ""
    x_correlation_id: str = ""
    conversation_id: str = ""
    system_message: str | None = None
    user_context_message: str | None = None
    drop_perf_ns: int | None = None
    credit_issued_ns: int | None = None
    is_final_turn: bool = True
    url_index: int | None = None


class MsgspecRequestRecord(Struct, kw_only=True):
    """Equivalent of RequestRecord. All 14 fields."""

    request_info: MsgspecRequestInfo | None = None
    request_headers: dict[str, str] | None = None
    model_name: str | None = None
    timestamp_ns: int = 0
    start_perf_ns: int = 0
    end_perf_ns: int | None = None
    recv_start_perf_ns: int | None = None
    status: int | None = None
    responses: list[MsgspecSSEMessage] = []
    error: MsgspecErrorDetails | None = None
    credit_drop_latency: int | None = None
    cancellation_perf_ns: int | None = None
    trace_data: Any | None = None
    turns: list[MsgspecTurn] = []  # Builders pass [] explicitly


# =============================================================================
# Helpers
# =============================================================================


def _make_text(char_count: int) -> str:
    base = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    repetitions = (char_count // len(base)) + 1
    return (base * repetitions)[:char_count]


def _make_openai_sse_chunk(
    content: str, idx: int, finish_reason: str | None = None
) -> str:
    payload = {
        "id": f"chatcmpl-{idx}",
        "object": "chat.completion.chunk",
        "created": 1749678185,
        "model": "test-model",
        "choices": [
            {"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}
        ],
    }
    return f"data: {orjson.dumps(payload).decode()}\n\n"


def _make_openai_usage_chunk(prompt_tokens: int, completion_tokens: int) -> str:
    payload = {
        "id": "chatcmpl-final",
        "object": "chat.completion.chunk",
        "created": 1749678185,
        "model": "test-model",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return f"data: {orjson.dumps(payload).decode()}\n\n"


def _generate_sse_strings(output_tokens: int, tokens_per_chunk: int = 1) -> list[str]:
    chunks = []
    remaining = output_tokens
    idx = 0
    while remaining > 0:
        n = min(tokens_per_chunk, remaining)
        content = _make_text(n * 4)
        finish = "stop" if remaining <= tokens_per_chunk else None
        chunks.append(_make_openai_sse_chunk(content, idx, finish))
        remaining -= n
        idx += 1
    chunks.append(_make_openai_usage_chunk(100, output_tokens))
    chunks.append("data: [DONE]\n\n")
    return chunks


def _snapshot_bytes(func):
    """Run *func* inside a tracemalloc snapshot and return net bytes allocated."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    snap_before = tracemalloc.take_snapshot()
    result = func()
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    stats = snap_after.compare_to(snap_before, "filename")
    return sum(s.size_diff for s in stats if s.size_diff > 0), result


# =============================================================================
# Shared model endpoint objects — created once, shared across requests.
# In production the ModelEndpointInfo is created at startup and referenced
# by every RequestInfo (just a pointer, ~8 bytes). We pre-create both
# Pydantic and msgspec versions outside all measured blocks.
# =============================================================================

_PYDANTIC_MODEL_ENDPOINT = ModelEndpointInfo(
    models=ModelListInfo(
        models=[ModelInfo(name="test-model")],
        model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
    ),
    endpoint=EndpointInfo(),
)

_CHAT_ENDPOINT = ChatEndpoint(model_endpoint=_PYDANTIC_MODEL_ENDPOINT)

_MSGSPEC_MODEL_ENDPOINT = MsgspecModelEndpointInfo(
    models=MsgspecModelListInfo(
        models=[MsgspecModelInfo(name="test-model")],
        model_selection_strategy="round_robin",
    ),
    endpoint=MsgspecEndpointInfo(base_urls=["http://localhost:8000"]),
)


# =============================================================================
# Warmup: pre-allocate class metadata for both Pydantic and msgspec so
# that first-time schema compilation costs are NOT counted in any test.
# =============================================================================


def _warmup_pydantic() -> None:
    """Create and discard one instance of each Pydantic model to trigger schema compilation."""
    _ = SSEMessage.parse("data: warmup", 0)
    _ = ParsedResponse(perf_ns=0, data=TextResponseData(text="w"))
    _ = RequestRecord(
        model_name="w", timestamp_ns=0, start_perf_ns=0, end_perf_ns=0, responses=[]
    )
    _ = TokenCounts(input=0, output=0, reasoning=0)
    _ = Turn(role="user", texts=[Text(name="t", contents=["w"])])
    _ = RequestInfo(
        model_endpoint=_PYDANTIC_MODEL_ENDPOINT,
        turns=[],
        turn_index=0,
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="w",
        x_correlation_id="w",
        conversation_id="w",
    )


def _warmup_msgspec() -> None:
    """Create and discard one instance of each msgspec Struct."""
    _ = MsgspecSSEMessage.parse("data: warmup", 0)
    _ = MsgspecParsedResponse(
        perf_ns=0, data=MsgspecTextResponseData(text="w"), metadata={}
    )
    _ = MsgspecRequestRecord(model_name="w", responses=[], turns=[])
    _ = MsgspecTokenCounts(input=0, output=0, reasoning=0)
    _ = MsgspecErrorDetails(message="warmup")
    _ = MsgspecTurn(
        role="user",
        texts=[MsgspecText(name="t", contents=["w"])],
        images=[],
        audios=[],
        videos=[],
    )
    _ = MsgspecRequestInfo(
        model_endpoint=_MSGSPEC_MODEL_ENDPOINT,
        turns=[],
        turn_index=0,
        endpoint_headers={},
        endpoint_params={},
    )


# Module-level warmup: runs once at import/collection time.
_warmup_pydantic()
_warmup_msgspec()


# =============================================================================
# SSE parsing builders — identical logic on both sides
# =============================================================================


def _build_pydantic_sse_messages(sse_strings: list[str]) -> list[SSEMessage]:
    return [
        SSEMessage.parse(s.strip(), time.perf_counter_ns())
        for s in sse_strings
        if s.strip()
    ]


def _build_msgspec_sse_messages(sse_strings: list[str]) -> list[MsgspecSSEMessage]:
    return [
        MsgspecSSEMessage.parse(s.strip(), time.perf_counter_ns())
        for s in sse_strings
        if s.strip()
    ]


# =============================================================================
# Parsed response builders — identical parsing logic
# =============================================================================


def _build_pydantic_parsed_responses(
    messages: list[SSEMessage], endpoint: ChatEndpoint
) -> list[ParsedResponse]:
    """Use the real ChatEndpoint to parse Pydantic SSEMessages."""
    record = RequestRecord(
        model_name="test-model",
        timestamp_ns=time.time_ns(),
        start_perf_ns=time.perf_counter_ns(),
        end_perf_ns=time.perf_counter_ns(),
        responses=messages,
    )
    return endpoint.extract_response_data(record)


def _parse_msgspec_response(msg: MsgspecSSEMessage) -> MsgspecParsedResponse | None:
    """Mirrors ChatEndpoint.parse_response logic exactly for msgspec types."""
    json_obj = msg.get_json()
    if not json_obj:
        return None

    obj_type = json_obj.get("object")
    if obj_type == "chat.completion.chunk":
        data_key = "delta"
    elif obj_type == "chat.completion":
        data_key = "message"
    else:
        return None

    choices = json_obj.get("choices")
    if not choices:
        usage = json_obj.get("usage")
        return (
            MsgspecParsedResponse(perf_ns=msg.perf_ns, usage=usage, metadata={})
            if usage
            else None
        )

    data = choices[0].get(data_key)
    if not data:
        return None

    content = data.get("content")
    reasoning = data.get("reasoning_content") or data.get("reasoning")
    usage = json_obj.get("usage")

    if not content and not reasoning:
        return (
            MsgspecParsedResponse(perf_ns=msg.perf_ns, usage=usage, metadata={})
            if usage
            else None
        )

    if reasoning:
        resp_data = MsgspecReasoningResponseData(content=content, reasoning=reasoning)
    else:
        resp_data = MsgspecTextResponseData(text=content)

    return MsgspecParsedResponse(
        perf_ns=msg.perf_ns, data=resp_data, usage=usage, metadata={}
    )


def _build_msgspec_parsed_responses(
    messages: list[MsgspecSSEMessage],
) -> list[MsgspecParsedResponse]:
    return [p for msg in messages if (p := _parse_msgspec_response(msg))]


# =============================================================================
# Full record builders — includes nested Turn/Text/RequestInfo data
#
# Both sides create identical nested structures:
#   - RequestInfo with N turns (each with Text prompt data)
#   - RequestRecord.turns = deep copy of turns (mirrors copy_with_stripped_media)
#   - SSE messages + parsed responses
#
# The shared ModelEndpointInfo is created outside the measured block since
# it's a singleton in production (just a pointer in RequestInfo).
# =============================================================================

PROMPT_CHARS = 200  # ~50 tokens, typical single-turn prompt


def _build_pydantic_record_and_parsed(
    sse_strings: list[str],
    endpoint: ChatEndpoint,
    num_turns: int = 1,
    prompt_chars: int = PROMPT_CHARS,
) -> tuple[RequestRecord, list[ParsedResponse]]:
    """Build Pydantic RequestRecord with full nested data + ParsedResponse list."""
    prompt = _make_text(prompt_chars)

    # Turns with text data (request_info.turns)
    turns = [
        Turn(
            role="user",
            max_tokens=256,
            texts=[Text(name="text_input", contents=[prompt])],
            images=[],  # Explicit fresh list to match msgspec/slotted
            audios=[],
            videos=[],
        )
        for _ in range(num_turns)
    ]

    request_info = RequestInfo(
        model_endpoint=endpoint.model_endpoint,
        turns=turns,
        turn_index=0,
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="req-001",
        x_correlation_id="corr-001",
        conversation_id="conv-001",
    )

    # record.turns = deep copy with stripped media (mirrors production behavior)
    record_turns = [
        Turn(
            role=t.role,
            max_tokens=t.max_tokens,
            texts=[Text(name=txt.name, contents=list(txt.contents)) for txt in t.texts],
            images=[],
            audios=[],
            videos=[],
        )
        for t in turns
    ]

    messages = _build_pydantic_sse_messages(sse_strings)
    record = RequestRecord(
        request_info=request_info,
        model_name="test-model",
        timestamp_ns=time.time_ns(),
        start_perf_ns=time.perf_counter_ns(),
        end_perf_ns=time.perf_counter_ns(),
        responses=messages,
        turns=record_turns,
    )
    parsed = endpoint.extract_response_data(record)
    return record, parsed


def _build_msgspec_record_and_parsed(
    sse_strings: list[str],
    num_turns: int = 1,
    prompt_chars: int = PROMPT_CHARS,
) -> tuple[MsgspecRequestRecord, list[MsgspecParsedResponse]]:
    """Build msgspec RequestRecord with full nested data + ParsedResponse list."""
    prompt = _make_text(prompt_chars)

    # Turns with text data (request_info.turns)
    turns = [
        MsgspecTurn(
            role="user",
            max_tokens=256,
            texts=[MsgspecText(name="text_input", contents=[prompt])],
            images=[],  # Explicit fresh list to match Pydantic default_factory
            audios=[],
            videos=[],
        )
        for _ in range(num_turns)
    ]

    request_info = MsgspecRequestInfo(
        model_endpoint=_MSGSPEC_MODEL_ENDPOINT,
        turns=turns,
        turn_index=0,
        endpoint_headers={},  # Fresh dict to match Pydantic default_factory
        endpoint_params={},
        credit_num=0,
        credit_phase="profiling",
        x_request_id="req-001",
        x_correlation_id="corr-001",
        conversation_id="conv-001",
    )

    # record.turns = deep copy with stripped media (mirrors production behavior)
    record_turns = [
        MsgspecTurn(
            role=t.role,
            max_tokens=t.max_tokens,
            texts=[
                MsgspecText(name=txt.name, contents=list(txt.contents))
                for txt in t.texts
            ],
            images=[],
            audios=[],
            videos=[],
        )
        for t in turns
    ]

    messages = _build_msgspec_sse_messages(sse_strings)
    record = MsgspecRequestRecord(
        request_info=request_info,
        model_name="test-model",
        timestamp_ns=time.time_ns(),
        start_perf_ns=time.perf_counter_ns(),
        end_perf_ns=time.perf_counter_ns(),
        responses=messages,
        turns=record_turns,
    )
    parsed = _build_msgspec_parsed_responses(messages)
    return record, parsed


# =============================================================================
# Benchmarks
# =============================================================================


def benchmark_sse_message_comparison() -> None:
    """Compare SSEMessage memory: Pydantic vs msgspec.

    Both sides: parse identical SSE strings -> list[SSEMessage].
    """
    print("\n=== SSE Message Comparison ===")

    cases = [
        (10, "10tok"),
        (50, "50tok"),
        (100, "100tok"),
        (500, "500tok"),
        (1000, "1000tok"),
        (2000, "2000tok"),
    ]

    for output_tokens, _label in cases:
        sse_strings = _generate_sse_strings(output_tokens)

        pydantic_bytes, pydantic_msgs = _snapshot_bytes(
            lambda _s=sse_strings: _build_pydantic_sse_messages(_s)
        )
        msgspec_bytes, msgspec_msgs = _snapshot_bytes(
            lambda _s=sse_strings: _build_msgspec_sse_messages(_s)
        )

        assert len(pydantic_msgs) == len(msgspec_msgs), (
            f"Object count mismatch: pydantic={len(pydantic_msgs)} vs msgspec={len(msgspec_msgs)}"
        )

        savings = pydantic_bytes - msgspec_bytes
        pct = (savings / pydantic_bytes * 100) if pydantic_bytes else 0

        print(
            f"\n  SSEMessage list ({output_tokens} tokens, {len(pydantic_msgs)} messages):"
            f"\n    Pydantic:  {pydantic_bytes:>12,} bytes ({pydantic_bytes / 1024:.1f} KB)"
            f"\n    msgspec:   {msgspec_bytes:>12,} bytes ({msgspec_bytes / 1024:.1f} KB)"
            f"\n    Savings:   {savings:>12,} bytes ({pct:+.1f}%)"
        )


def benchmark_parsed_response_comparison() -> None:
    """Compare ParsedResponse memory: Pydantic vs msgspec.

    Both sides: parse identical SSE messages -> list[ParsedResponse].
    Pydantic uses real ChatEndpoint; msgspec uses equivalent inline logic.
    """
    print("\n=== Parsed Response Comparison ===")

    cases = [
        (10, "10tok"),
        (50, "50tok"),
        (100, "100tok"),
        (500, "500tok"),
        (1000, "1000tok"),
        (2000, "2000tok"),
    ]

    for output_tokens, _label in cases:
        sse_strings = _generate_sse_strings(output_tokens)

        # Pre-build SSE messages outside the measured block
        pydantic_msgs = _build_pydantic_sse_messages(sse_strings)
        msgspec_msgs = _build_msgspec_sse_messages(sse_strings)

        pydantic_bytes, pydantic_parsed = _snapshot_bytes(
            lambda _m=pydantic_msgs: _build_pydantic_parsed_responses(
                _m, _CHAT_ENDPOINT
            )
        )
        msgspec_bytes, msgspec_parsed = _snapshot_bytes(
            lambda _m=msgspec_msgs: _build_msgspec_parsed_responses(_m)
        )

        assert len(pydantic_parsed) == len(msgspec_parsed), (
            f"Object count mismatch: pydantic={len(pydantic_parsed)} vs msgspec={len(msgspec_parsed)}"
        )

        savings = pydantic_bytes - msgspec_bytes
        pct = (savings / pydantic_bytes * 100) if pydantic_bytes else 0

        print(
            f"\n  ParsedResponse list ({output_tokens} tokens, {len(pydantic_parsed)} items):"
            f"\n    Pydantic:  {pydantic_bytes:>12,} bytes ({pydantic_bytes / 1024:.1f} KB)"
            f"\n    msgspec:   {msgspec_bytes:>12,} bytes ({msgspec_bytes / 1024:.1f} KB)"
            f"\n    Savings:   {savings:>12,} bytes ({pct:+.1f}%)"
        )


def benchmark_full_record_comparison() -> None:
    """Compare full record pipeline: Pydantic vs msgspec.

    Both sides create: RequestInfo(turns) + RequestRecord(responses, turns) + ParsedResponses.
    The nested Turn/Text/RequestInfo objects are now included in the measurement.
    """
    print("\n=== Full Record Comparison ===")

    cases = [
        (100, 1, "100tok-1t"),
        (500, 1, "500tok-1t"),
        (1000, 1, "1000tok-1t"),
        (100, 5, "100tok-5t"),
        (500, 5, "500tok-5t"),
        (1000, 5, "1000tok-5t"),
        (500, 10, "500tok-10t"),
    ]

    for output_tokens, num_turns, _label in cases:
        sse_strings = _generate_sse_strings(output_tokens)

        pydantic_bytes, (p_record, p_parsed) = _snapshot_bytes(
            lambda _s=sse_strings, _n=num_turns: _build_pydantic_record_and_parsed(
                _s, _CHAT_ENDPOINT, num_turns=_n
            )
        )
        msgspec_bytes, (m_record, m_parsed) = _snapshot_bytes(
            lambda _s=sse_strings, _n=num_turns: _build_msgspec_record_and_parsed(
                _s, num_turns=_n
            )
        )

        assert len(p_record.responses) == len(m_record.responses), (
            f"SSE count mismatch: pydantic={len(p_record.responses)} vs msgspec={len(m_record.responses)}"
        )
        assert len(p_parsed) == len(m_parsed), (
            f"Parsed count mismatch: pydantic={len(p_parsed)} vs msgspec={len(m_parsed)}"
        )
        assert len(p_record.turns) == len(m_record.turns) == num_turns, (
            f"Turn count mismatch: pydantic={len(p_record.turns)} msgspec={len(m_record.turns)} expected={num_turns}"
        )
        assert p_record.request_info is not None and m_record.request_info is not None

        savings = pydantic_bytes - msgspec_bytes
        pct = (savings / pydantic_bytes * 100) if pydantic_bytes else 0

        print(
            f"\n  Full record ({output_tokens} tokens, {num_turns} turns, {PROMPT_CHARS} char prompt):"
            f"\n    Pydantic:  {pydantic_bytes:>12,} bytes ({pydantic_bytes / 1024:.1f} KB)"
            f"\n    msgspec:   {msgspec_bytes:>12,} bytes ({msgspec_bytes / 1024:.1f} KB)"
            f"\n    Savings:   {savings:>12,} bytes ({pct:+.1f}%)"
            f"\n    Objects:   {len(p_record.responses)} SSE, {len(p_parsed)} parsed, "
            f"{num_turns}x2 turns, 1 RequestInfo"
        )


def benchmark_concurrent_load_comparison() -> None:
    """Compare concurrent load memory: Pydantic vs msgspec at scale.

    Both sides create N x full records (RequestInfo + turns + SSE + parsed).
    """
    print("\n=== Concurrent Load Comparison ===")

    cases = [
        (50, 100, 1, "50c-100tok-1t"),
        (100, 100, 1, "100c-100tok-1t"),
        (50, 500, 1, "50c-500tok-1t"),
        (100, 500, 1, "100c-500tok-1t"),
        (50, 100, 5, "50c-100tok-5t"),
        (100, 100, 5, "100c-100tok-5t"),
        (50, 500, 5, "50c-500tok-5t"),
        (100, 500, 5, "100c-500tok-5t"),
        (100, 1000, 1, "100c-1000tok-1t"),
        (100, 1000, 5, "100c-1000tok-5t"),
    ]

    for concurrency, output_tokens, num_turns, _label in cases:
        sse_strings = _generate_sse_strings(output_tokens)

        def build_pydantic(_s=sse_strings, _n=num_turns, _c=concurrency):
            return [
                _build_pydantic_record_and_parsed(_s, _CHAT_ENDPOINT, num_turns=_n)
                for _ in range(_c)
            ]

        def build_msgspec(_s=sse_strings, _n=num_turns, _c=concurrency):
            return [
                _build_msgspec_record_and_parsed(_s, num_turns=_n) for _ in range(_c)
            ]

        pydantic_bytes, _ = _snapshot_bytes(build_pydantic)
        msgspec_bytes, _ = _snapshot_bytes(build_msgspec)

        savings = pydantic_bytes - msgspec_bytes
        pct = (savings / pydantic_bytes * 100) if pydantic_bytes else 0
        p_per_slot = pydantic_bytes / concurrency
        m_per_slot = msgspec_bytes / concurrency
        p_mb = pydantic_bytes / (1024 * 1024)
        m_mb = msgspec_bytes / (1024 * 1024)

        print(
            f"\n  Concurrent load ({concurrency}x {output_tokens} tokens, {num_turns} turns):"
            f"\n    Pydantic:  {pydantic_bytes:>14,} bytes ({p_mb:.2f} MB)  [{p_per_slot:,.0f}/slot]"
            f"\n    msgspec:   {msgspec_bytes:>14,} bytes ({m_mb:.2f} MB)  [{m_per_slot:,.0f}/slot]"
            f"\n    Savings:   {savings:>14,} bytes ({pct:+.1f}%)"
        )
        assert pydantic_bytes > 0
        assert msgspec_bytes > 0


def benchmark_per_object_overhead() -> None:
    """Measure per-object overhead of Pydantic vs msgspec for individual instances.

    Uses bulk allocation (N=500) to amortize measurement noise and give
    a stable per-object cost.
    """
    print("\n=== Per-Object Overhead ===")

    # SSE message bulk overhead
    n = 500
    raw = "data: Hello world"

    pydantic_bytes, _ = _snapshot_bytes(
        lambda: [SSEMessage.parse(raw, time.perf_counter_ns()) for _ in range(n)]
    )
    msgspec_bytes, _ = _snapshot_bytes(
        lambda: [MsgspecSSEMessage.parse(raw, time.perf_counter_ns()) for _ in range(n)]
    )

    print(
        f"\n  {n} SSEMessage instances (tiny payload):"
        f"\n    Pydantic:  {pydantic_bytes:>10,} bytes ({pydantic_bytes / n:.0f}/item)"
        f"\n    msgspec:   {msgspec_bytes:>10,} bytes ({msgspec_bytes / n:.0f}/item)"
        f"\n    Savings:   {pydantic_bytes - msgspec_bytes:>10,} bytes ({(pydantic_bytes - msgspec_bytes) / n:.0f}/item)"
    )

    # Parsed response bulk overhead
    n = 500

    pydantic_bytes, _ = _snapshot_bytes(
        lambda: [
            ParsedResponse(perf_ns=0, data=TextResponseData(text="Hello"))
            for _ in range(n)
        ]
    )
    msgspec_bytes, _ = _snapshot_bytes(
        lambda: [
            MsgspecParsedResponse(
                perf_ns=0, data=MsgspecTextResponseData(text="Hello"), metadata={}
            )
            for _ in range(n)
        ]
    )

    print(
        f"\n  {n} ParsedResponse instances (tiny payload):"
        f"\n    Pydantic:  {pydantic_bytes:>10,} bytes ({pydantic_bytes / n:.0f}/item)"
        f"\n    msgspec:   {msgspec_bytes:>10,} bytes ({msgspec_bytes / n:.0f}/item)"
        f"\n    Savings:   {pydantic_bytes - msgspec_bytes:>10,} bytes ({(pydantic_bytes - msgspec_bytes) / n:.0f}/item)"
    )

    # Turn bulk overhead
    n = 500
    prompt = _make_text(200)

    pydantic_bytes, _ = _snapshot_bytes(
        lambda: [
            Turn(
                role="user",
                max_tokens=256,
                texts=[Text(name="text_input", contents=[prompt])],
                images=[],
                audios=[],
                videos=[],
            )
            for _ in range(n)
        ]
    )
    msgspec_bytes, _ = _snapshot_bytes(
        lambda: [
            MsgspecTurn(
                role="user",
                max_tokens=256,
                texts=[MsgspecText(name="text_input", contents=[prompt])],
                images=[],
                audios=[],
                videos=[],
            )
            for _ in range(n)
        ]
    )

    print(
        f"\n  {n} Turn instances (with 200-char Text):"
        f"\n    Pydantic:  {pydantic_bytes:>10,} bytes ({pydantic_bytes / n:.0f}/item)"
        f"\n    msgspec:   {msgspec_bytes:>10,} bytes ({msgspec_bytes / n:.0f}/item)"
        f"\n    Savings:   {pydantic_bytes - msgspec_bytes:>10,} bytes ({(pydantic_bytes - msgspec_bytes) / n:.0f}/item)"
    )

    # RequestInfo bulk overhead
    n = 200
    prompt = _make_text(200)

    pydantic_bytes, _ = _snapshot_bytes(
        lambda: [
            RequestInfo(
                model_endpoint=_PYDANTIC_MODEL_ENDPOINT,
                turns=[
                    Turn(
                        role="user",
                        max_tokens=256,
                        texts=[Text(name="t", contents=[prompt])],
                        images=[],
                        audios=[],
                        videos=[],
                    )
                ],
                turn_index=0,
                credit_num=i,
                credit_phase=CreditPhase.PROFILING,
                x_request_id=f"req-{i}",
                x_correlation_id=f"corr-{i}",
                conversation_id=f"conv-{i}",
            )
            for i in range(n)
        ]
    )
    msgspec_bytes, _ = _snapshot_bytes(
        lambda: [
            MsgspecRequestInfo(
                model_endpoint=_MSGSPEC_MODEL_ENDPOINT,
                turns=[
                    MsgspecTurn(
                        role="user",
                        max_tokens=256,
                        texts=[MsgspecText(name="t", contents=[prompt])],
                        images=[],
                        audios=[],
                        videos=[],
                    )
                ],
                turn_index=0,
                endpoint_headers={},
                endpoint_params={},
                credit_num=i,
                credit_phase="profiling",
                x_request_id=f"req-{i}",
                x_correlation_id=f"corr-{i}",
                conversation_id=f"conv-{i}",
            )
            for i in range(n)
        ]
    )

    print(
        f"\n  {n} RequestInfo instances (with 1 Turn + 200-char Text):"
        f"\n    Pydantic:  {pydantic_bytes:>10,} bytes ({pydantic_bytes / n:.0f}/item)"
        f"\n    msgspec:   {msgspec_bytes:>10,} bytes ({msgspec_bytes / n:.0f}/item)"
        f"\n    Savings:   {pydantic_bytes - msgspec_bytes:>10,} bytes ({(pydantic_bytes - msgspec_bytes) / n:.0f}/item)"
    )

    # TokenCounts bulk overhead
    n = 500

    pydantic_bytes, _ = _snapshot_bytes(
        lambda: [TokenCounts(input=128, output=256, reasoning=64) for _ in range(n)]
    )
    msgspec_bytes, _ = _snapshot_bytes(
        lambda: [
            MsgspecTokenCounts(input=128, output=256, reasoning=64) for _ in range(n)
        ]
    )

    print(
        f"\n  {n} TokenCounts instances:"
        f"\n    Pydantic:  {pydantic_bytes:>10,} bytes ({pydantic_bytes / n:.0f}/item)"
        f"\n    msgspec:   {msgspec_bytes:>10,} bytes ({msgspec_bytes / n:.0f}/item)"
        f"\n    Savings:   {pydantic_bytes - msgspec_bytes:>10,} bytes ({(pydantic_bytes - msgspec_bytes) / n:.0f}/item)"
    )

    # SSEField bulk overhead
    n = 1000
    value = _make_text(50)

    namedtuple_bytes, _ = _snapshot_bytes(
        lambda: [SSEField(name="data", value=value) for _ in range(n)]
    )
    msgspec_bytes, _ = _snapshot_bytes(
        lambda: [MsgspecSSEField(name="data", value=value) for _ in range(n)]
    )

    print(
        f"\n  {n} SSEField instances (50 char value):"
        f"\n    NamedTuple: {namedtuple_bytes:>10,} bytes ({namedtuple_bytes / n:.0f}/item)"
        f"\n    msgspec:    {msgspec_bytes:>10,} bytes ({msgspec_bytes / n:.0f}/item)"
        f"\n    Diff:       {namedtuple_bytes - msgspec_bytes:>10,} bytes ({(namedtuple_bytes - msgspec_bytes) / n:.0f}/item)"
    )


# =============================================================================
# MIDDLE GROUND: @dataclass(slots=True) equivalents
#
# Strategy: use slotted dataclasses for hot-path accumulation objects
# (SSEMessage, ParsedResponse, Turn, etc.) inside a single process,
# then convert to Pydantic at ZMQ/export boundaries only.
#
# Pros:
#   - Standard library, zero new dependencies
#   - ~40-50% less memory than Pydantic (slots vs __dict__)
#   - Supports methods, classmethods, properties
#   - Convert to Pydantic at boundary: PydanticModel(**asdict(dc))
#
# Cons:
#   - No cached_property (no __dict__) — use lru_cache or manual caching
#   - No validation on construction — validate at boundaries only
#   - Conversion cost at ZMQ boundary (asdict + model construction)
# =============================================================================


@dataclass(slots=True)
class SlottedSSEField:
    """Slotted dataclass equivalent of SSEField."""

    name: str
    value: str | None = None


@dataclass(slots=True)
class SlottedSSEMessage:
    """Slotted dataclass equivalent of SSEMessage. 2 fields."""

    perf_ns: int
    packets: list[SlottedSSEField] = dc_field(default_factory=list)

    def extract_data_content(self) -> str:
        return "\n".join(p.value for p in self.packets if p.name == "data" and p.value)

    def get_json(self) -> dict[str, Any] | None:
        text = self.extract_data_content()
        if text in ("", "[DONE]"):
            return None
        try:
            return orjson.loads(text)
        except orjson.JSONDecodeError:
            return None

    @classmethod
    def parse(cls, raw: str, perf_ns: int) -> "SlottedSSEMessage":
        packets = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) < 2:
                packets.append(SlottedSSEField(name=parts[0].strip()))
            else:
                packets.append(
                    SlottedSSEField(name=parts[0].strip(), value=parts[1].strip())
                )
        return cls(perf_ns=perf_ns, packets=packets)


@dataclass(slots=True)
class SlottedTextResponseData:
    """Slotted dataclass equivalent of TextResponseData. 1 field."""

    text: str


@dataclass(slots=True)
class SlottedReasoningResponseData:
    """Slotted dataclass equivalent of ReasoningResponseData. 2 fields."""

    content: str | None = None
    reasoning: str | None = None


@dataclass(slots=True)
class SlottedParsedResponse:
    """Slotted dataclass equivalent of ParsedResponse. All 5 fields."""

    perf_ns: int
    data: SlottedTextResponseData | SlottedReasoningResponseData | None = None
    usage: dict[str, Any] | None = None
    sources: Any | None = None
    metadata: dict[str, Any] | None = dc_field(default_factory=dict)


@dataclass(slots=True)
class SlottedTokenCounts:
    """Slotted dataclass equivalent of TokenCounts. 3 fields."""

    input: int | None = None
    output: int | None = None
    reasoning: int | None = None


@dataclass(slots=True)
class SlottedErrorDetails:
    """Slotted dataclass equivalent of ErrorDetails. 5 fields."""

    code: int | None = None
    type: str | None = None
    message: str = ""
    cause: str | None = None
    details: Any | None = None


@dataclass(slots=True)
class SlottedText:
    """Slotted dataclass equivalent of Text. 2 fields."""

    name: str = ""
    contents: list[str] = dc_field(default_factory=list)


@dataclass(slots=True)
class SlottedMedia:
    """Slotted dataclass equivalent of Image/Audio/Video. 2 fields."""

    name: str = ""
    contents: list[str] = dc_field(default_factory=list)


@dataclass(slots=True)
class SlottedTurn:
    """Slotted dataclass equivalent of Turn. All 9 fields."""

    model: str | None = None
    role: str | None = None
    timestamp: int | float | None = None
    delay: int | float | None = None
    max_tokens: int | None = None
    texts: list[SlottedText] = dc_field(default_factory=list)
    images: list[SlottedMedia] = dc_field(default_factory=list)
    audios: list[SlottedMedia] = dc_field(default_factory=list)
    videos: list[SlottedMedia] = dc_field(default_factory=list)


@dataclass(slots=True)
class SlottedRequestInfo:
    """Slotted dataclass equivalent of RequestInfo. All 17 fields."""

    model_endpoint: Any  # Shared reference, not measured
    turns: list[SlottedTurn] = dc_field(default_factory=list)
    turn_index: int = 0
    endpoint_headers: dict[str, str] = dc_field(default_factory=dict)
    endpoint_params: dict[str, str] = dc_field(default_factory=dict)
    credit_num: int = 0
    credit_phase: str = "profiling"
    cancel_after_ns: int | None = None
    x_request_id: str = ""
    x_correlation_id: str = ""
    conversation_id: str = ""
    system_message: str | None = None
    user_context_message: str | None = None
    drop_perf_ns: int | None = None
    credit_issued_ns: int | None = None
    is_final_turn: bool = True
    url_index: int | None = None


@dataclass(slots=True)
class SlottedRequestRecord:
    """Slotted dataclass equivalent of RequestRecord. All 14 fields."""

    request_info: SlottedRequestInfo | None = None
    request_headers: dict[str, str] | None = None
    model_name: str | None = None
    timestamp_ns: int = 0
    start_perf_ns: int = 0
    end_perf_ns: int | None = None
    recv_start_perf_ns: int | None = None
    status: int | None = None
    responses: list[SlottedSSEMessage] = dc_field(default_factory=list)
    error: SlottedErrorDetails | None = None
    credit_drop_latency: int | None = None
    cancellation_perf_ns: int | None = None
    trace_data: Any | None = None
    turns: list[SlottedTurn] = dc_field(default_factory=list)


# Warmup slotted dataclasses (trigger any __init__ compilation)
def _warmup_slotted() -> None:
    _ = SlottedSSEMessage.parse("data: warmup", 0)
    _ = SlottedParsedResponse(perf_ns=0, data=SlottedTextResponseData(text="w"))
    _ = SlottedRequestRecord(model_name="w")
    _ = SlottedTokenCounts(input=0, output=0, reasoning=0)
    _ = SlottedErrorDetails(message="warmup")
    _ = SlottedTurn(role="user", texts=[SlottedText(name="t", contents=["w"])])
    _ = SlottedRequestInfo(model_endpoint=None)


_warmup_slotted()


# =============================================================================
# Slotted dataclass builders
# =============================================================================


def _build_slotted_sse_messages(sse_strings: list[str]) -> list[SlottedSSEMessage]:
    return [
        SlottedSSEMessage.parse(s.strip(), time.perf_counter_ns())
        for s in sse_strings
        if s.strip()
    ]


def _parse_slotted_response(msg: SlottedSSEMessage) -> SlottedParsedResponse | None:
    """Mirrors ChatEndpoint.parse_response for slotted dataclasses."""
    json_obj = msg.get_json()
    if not json_obj:
        return None

    obj_type = json_obj.get("object")
    if obj_type == "chat.completion.chunk":
        data_key = "delta"
    elif obj_type == "chat.completion":
        data_key = "message"
    else:
        return None

    choices = json_obj.get("choices")
    if not choices:
        usage = json_obj.get("usage")
        return (
            SlottedParsedResponse(perf_ns=msg.perf_ns, usage=usage) if usage else None
        )

    data = choices[0].get(data_key)
    if not data:
        return None

    content = data.get("content")
    reasoning = data.get("reasoning_content") or data.get("reasoning")
    usage = json_obj.get("usage")

    if not content and not reasoning:
        return (
            SlottedParsedResponse(perf_ns=msg.perf_ns, usage=usage) if usage else None
        )

    if reasoning:
        resp_data = SlottedReasoningResponseData(content=content, reasoning=reasoning)
    else:
        resp_data = SlottedTextResponseData(text=content)

    return SlottedParsedResponse(perf_ns=msg.perf_ns, data=resp_data, usage=usage)


def _build_slotted_parsed_responses(
    messages: list[SlottedSSEMessage],
) -> list[SlottedParsedResponse]:
    return [p for msg in messages if (p := _parse_slotted_response(msg))]


def _build_slotted_record_and_parsed(
    sse_strings: list[str],
    num_turns: int = 1,
    prompt_chars: int = PROMPT_CHARS,
) -> tuple[SlottedRequestRecord, list[SlottedParsedResponse]]:
    """Build slotted dataclass RequestRecord with full nested data."""
    prompt = _make_text(prompt_chars)

    turns = [
        SlottedTurn(
            role="user",
            max_tokens=256,
            texts=[SlottedText(name="text_input", contents=[prompt])],
            images=[],  # Explicit fresh list to match Pydantic default_factory
            audios=[],
            videos=[],
        )
        for _ in range(num_turns)
    ]

    request_info = SlottedRequestInfo(
        model_endpoint=_MSGSPEC_MODEL_ENDPOINT,  # Shared ref, not measured
        turns=turns,
        turn_index=0,
        endpoint_headers={},  # Fresh dict to match Pydantic default_factory
        endpoint_params={},
        credit_num=0,
        credit_phase="profiling",
        x_request_id="req-001",
        x_correlation_id="corr-001",
        conversation_id="conv-001",
    )

    record_turns = [
        SlottedTurn(
            role=t.role,
            max_tokens=t.max_tokens,
            texts=[
                SlottedText(name=txt.name, contents=list(txt.contents))
                for txt in t.texts
            ],
            images=[],
            audios=[],
            videos=[],
        )
        for t in turns
    ]

    messages = _build_slotted_sse_messages(sse_strings)
    record = SlottedRequestRecord(
        request_info=request_info,
        model_name="test-model",
        timestamp_ns=time.time_ns(),
        start_perf_ns=time.perf_counter_ns(),
        end_perf_ns=time.perf_counter_ns(),
        responses=messages,
        turns=record_turns,
    )
    parsed = _build_slotted_parsed_responses(messages)
    return record, parsed


# =============================================================================
# Three-way comparison helper
# =============================================================================


def _print_three_way(
    label: str,
    pydantic_bytes: int,
    slotted_bytes: int,
    msgspec_bytes: int,
    extra: str = "",
) -> None:
    """Print Pydantic vs Slotted Dataclass vs msgspec comparison."""
    s_pct = (
        ((pydantic_bytes - slotted_bytes) / pydantic_bytes * 100)
        if pydantic_bytes
        else 0
    )
    m_pct = (
        ((pydantic_bytes - msgspec_bytes) / pydantic_bytes * 100)
        if pydantic_bytes
        else 0
    )

    print(
        f"\n  {label}:"
        f"\n    Pydantic:   {pydantic_bytes:>14,} bytes"
        f"\n    Slotted DC: {slotted_bytes:>14,} bytes  ({s_pct:+.1f}% vs pydantic)"
        f"\n    msgspec:    {msgspec_bytes:>14,} bytes  ({m_pct:+.1f}% vs pydantic)"
        + (f"\n    {extra}" if extra else "")
    )


# =============================================================================
# Three-way comparison benchmarks
# =============================================================================


def benchmark_three_way_full_record() -> None:
    """Three-way comparison: Pydantic vs @dataclass(slots=True) vs msgspec.

    Measures the full record pipeline including nested Turn/Text/RequestInfo.
    """
    print("\n=== Three-Way Full Record ===")

    # Full record cases
    full_record_cases = [
        (100, 1, "100tok-1t"),
        (500, 1, "500tok-1t"),
        (1000, 1, "1000tok-1t"),
        (500, 5, "500tok-5t"),
        (1000, 5, "1000tok-5t"),
    ]

    for output_tokens, num_turns, _label in full_record_cases:
        sse_strings = _generate_sse_strings(output_tokens)

        pydantic_bytes, _ = _snapshot_bytes(
            lambda _s=sse_strings, _n=num_turns: _build_pydantic_record_and_parsed(
                _s, _CHAT_ENDPOINT, num_turns=_n
            )
        )
        slotted_bytes, _ = _snapshot_bytes(
            lambda _s=sse_strings, _n=num_turns: _build_slotted_record_and_parsed(
                _s, num_turns=_n
            )
        )
        msgspec_bytes, _ = _snapshot_bytes(
            lambda _s=sse_strings, _n=num_turns: _build_msgspec_record_and_parsed(
                _s, num_turns=_n
            )
        )

        _print_three_way(
            f"Full record ({output_tokens} tok, {num_turns} turns)",
            pydantic_bytes,
            slotted_bytes,
            msgspec_bytes,
        )

    # Concurrent cases
    concurrent_cases = [
        (100, 100, 1, "100c-100tok-1t"),
        (100, 500, 1, "100c-500tok-1t"),
        (100, 1000, 1, "100c-1000tok-1t"),
        (100, 100, 5, "100c-100tok-5t"),
        (100, 500, 5, "100c-500tok-5t"),
        (100, 1000, 5, "100c-1000tok-5t"),
    ]

    for concurrency, output_tokens, num_turns, _label in concurrent_cases:
        sse_strings = _generate_sse_strings(output_tokens)

        def build_pydantic(_s=sse_strings, _n=num_turns, _c=concurrency):
            return [
                _build_pydantic_record_and_parsed(_s, _CHAT_ENDPOINT, num_turns=_n)
                for _ in range(_c)
            ]

        def build_slotted(_s=sse_strings, _n=num_turns, _c=concurrency):
            return [
                _build_slotted_record_and_parsed(_s, num_turns=_n) for _ in range(_c)
            ]

        def build_msgspec(_s=sse_strings, _n=num_turns, _c=concurrency):
            return [
                _build_msgspec_record_and_parsed(_s, num_turns=_n) for _ in range(_c)
            ]

        pydantic_bytes, _ = _snapshot_bytes(build_pydantic)
        slotted_bytes, _ = _snapshot_bytes(build_slotted)
        msgspec_bytes, _ = _snapshot_bytes(build_msgspec)

        p_mb = pydantic_bytes / (1024 * 1024)
        s_mb = slotted_bytes / (1024 * 1024)
        m_mb = msgspec_bytes / (1024 * 1024)

        _print_three_way(
            f"Concurrent {concurrency}x ({output_tokens} tok, {num_turns} turns)",
            pydantic_bytes,
            slotted_bytes,
            msgspec_bytes,
            extra=f"MB: pydantic={p_mb:.1f}  slotted={s_mb:.1f}  msgspec={m_mb:.1f}",
        )
        assert pydantic_bytes > 0
        assert slotted_bytes > 0
        assert msgspec_bytes > 0


def benchmark_three_way_per_object() -> None:
    """Three-way per-object overhead: Pydantic vs slotted dataclass vs msgspec."""
    print("\n=== Three-Way Per-Object Overhead ===")

    # SSE message
    n = 500
    raw = "data: Hello world"

    pydantic_bytes, _ = _snapshot_bytes(
        lambda: [SSEMessage.parse(raw, time.perf_counter_ns()) for _ in range(n)]
    )
    slotted_bytes, _ = _snapshot_bytes(
        lambda: [SlottedSSEMessage.parse(raw, time.perf_counter_ns()) for _ in range(n)]
    )
    msgspec_bytes, _ = _snapshot_bytes(
        lambda: [MsgspecSSEMessage.parse(raw, time.perf_counter_ns()) for _ in range(n)]
    )

    _print_three_way(
        f"{n} SSEMessage instances",
        pydantic_bytes,
        slotted_bytes,
        msgspec_bytes,
        extra=f"Per item: pydantic={pydantic_bytes // n}  slotted={slotted_bytes // n}  msgspec={msgspec_bytes // n}",
    )

    # Parsed response
    n = 500

    pydantic_bytes, _ = _snapshot_bytes(
        lambda: [
            ParsedResponse(perf_ns=0, data=TextResponseData(text="Hello"))
            for _ in range(n)
        ]
    )
    slotted_bytes, _ = _snapshot_bytes(
        lambda: [
            SlottedParsedResponse(perf_ns=0, data=SlottedTextResponseData(text="Hello"))
            for _ in range(n)
        ]
    )
    msgspec_bytes, _ = _snapshot_bytes(
        lambda: [
            MsgspecParsedResponse(
                perf_ns=0, data=MsgspecTextResponseData(text="Hello"), metadata={}
            )
            for _ in range(n)
        ]
    )

    _print_three_way(
        f"{n} ParsedResponse instances",
        pydantic_bytes,
        slotted_bytes,
        msgspec_bytes,
        extra=f"Per item: pydantic={pydantic_bytes // n}  slotted={slotted_bytes // n}  msgspec={msgspec_bytes // n}",
    )

    # Turn with text
    n = 500
    prompt = _make_text(200)

    pydantic_bytes, _ = _snapshot_bytes(
        lambda: [
            Turn(
                role="user",
                max_tokens=256,
                texts=[Text(name="t", contents=[prompt])],
                images=[],
                audios=[],
                videos=[],
            )
            for _ in range(n)
        ]
    )
    slotted_bytes, _ = _snapshot_bytes(
        lambda: [
            SlottedTurn(
                role="user",
                max_tokens=256,
                texts=[SlottedText(name="t", contents=[prompt])],
                images=[],
                audios=[],
                videos=[],
            )
            for _ in range(n)
        ]
    )
    msgspec_bytes, _ = _snapshot_bytes(
        lambda: [
            MsgspecTurn(
                role="user",
                max_tokens=256,
                texts=[MsgspecText(name="t", contents=[prompt])],
                images=[],
                audios=[],
                videos=[],
            )
            for _ in range(n)
        ]
    )

    _print_three_way(
        f"{n} Turn instances (200-char text)",
        pydantic_bytes,
        slotted_bytes,
        msgspec_bytes,
        extra=f"Per item: pydantic={pydantic_bytes // n}  slotted={slotted_bytes // n}  msgspec={msgspec_bytes // n}",
    )

    # RequestInfo
    n = 200
    prompt = _make_text(200)

    pydantic_bytes, _ = _snapshot_bytes(
        lambda: [
            RequestInfo(
                model_endpoint=_PYDANTIC_MODEL_ENDPOINT,
                turns=[
                    Turn(
                        role="user",
                        max_tokens=256,
                        texts=[Text(name="t", contents=[prompt])],
                        images=[],
                        audios=[],
                        videos=[],
                    )
                ],
                turn_index=0,
                credit_num=i,
                credit_phase=CreditPhase.PROFILING,
                x_request_id=f"req-{i}",
                x_correlation_id=f"corr-{i}",
                conversation_id=f"conv-{i}",
            )
            for i in range(n)
        ]
    )
    slotted_bytes, _ = _snapshot_bytes(
        lambda: [
            SlottedRequestInfo(
                model_endpoint=_MSGSPEC_MODEL_ENDPOINT,
                turns=[
                    SlottedTurn(
                        role="user",
                        max_tokens=256,
                        texts=[SlottedText(name="t", contents=[prompt])],
                        images=[],
                        audios=[],
                        videos=[],
                    )
                ],
                turn_index=0,
                endpoint_headers={},
                endpoint_params={},
                credit_num=i,
                credit_phase="profiling",
                x_request_id=f"req-{i}",
                x_correlation_id=f"corr-{i}",
                conversation_id=f"conv-{i}",
            )
            for i in range(n)
        ]
    )
    msgspec_bytes, _ = _snapshot_bytes(
        lambda: [
            MsgspecRequestInfo(
                model_endpoint=_MSGSPEC_MODEL_ENDPOINT,
                turns=[
                    MsgspecTurn(
                        role="user",
                        max_tokens=256,
                        texts=[MsgspecText(name="t", contents=[prompt])],
                        images=[],
                        audios=[],
                        videos=[],
                    )
                ],
                turn_index=0,
                endpoint_headers={},
                endpoint_params={},
                credit_num=i,
                credit_phase="profiling",
                x_request_id=f"req-{i}",
                x_correlation_id=f"corr-{i}",
                conversation_id=f"conv-{i}",
            )
            for i in range(n)
        ]
    )

    _print_three_way(
        f"{n} RequestInfo instances (1 Turn + 200-char text)",
        pydantic_bytes,
        slotted_bytes,
        msgspec_bytes,
        extra=f"Per item: pydantic={pydantic_bytes // n}  slotted={slotted_bytes // n}  msgspec={msgspec_bytes // n}",
    )

    # TokenCounts
    n = 500

    pydantic_bytes, _ = _snapshot_bytes(
        lambda: [TokenCounts(input=128, output=256, reasoning=64) for _ in range(n)]
    )
    slotted_bytes, _ = _snapshot_bytes(
        lambda: [
            SlottedTokenCounts(input=128, output=256, reasoning=64) for _ in range(n)
        ]
    )
    msgspec_bytes, _ = _snapshot_bytes(
        lambda: [
            MsgspecTokenCounts(input=128, output=256, reasoning=64) for _ in range(n)
        ]
    )

    _print_three_way(
        f"{n} TokenCounts instances",
        pydantic_bytes,
        slotted_bytes,
        msgspec_bytes,
        extra=f"Per item: pydantic={pydantic_bytes // n}  slotted={slotted_bytes // n}  msgspec={msgspec_bytes // n}",
    )


if __name__ == "__main__":
    benchmark_sse_message_comparison()
    benchmark_parsed_response_comparison()
    benchmark_full_record_comparison()
    benchmark_concurrent_load_comparison()
    benchmark_per_object_overhead()
    benchmark_three_way_full_record()
    benchmark_three_way_per_object()
