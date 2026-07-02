#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory profiling benchmarks for streaming response parsing under various load conditions.

Measures memory consumed by the full SSE streaming pipeline:
- AsyncSSEStreamReader byte buffer -> SSEMessage parsing
- SSEMessage accumulation in RequestRecord.responses
- Endpoint parsing: SSEMessage -> ParsedResponse (JSON decode + data extraction)
- Full ParsedResponseRecord (parsed responses + token counts)
- Concurrent request accumulation (N in-flight records)

Simulates realistic OpenAI Chat Completions streaming format.

Run: uv run python dev/benchmarks/streaming_response_memory_profile.py
"""

import asyncio
import time
import tracemalloc
import uuid

import orjson

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models import (
    ParsedResponseRecord,
    RequestInfo,
    RequestRecord,
    SSEMessage,
    Text,
    Turn,
)
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import SSEField, TokenCounts
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.transports.sse_utils import AsyncSSEStreamReader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_endpoint() -> ModelEndpointInfo:
    """Create a minimal ModelEndpointInfo for benchmarks."""
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(),
    )


def _make_text(char_count: int) -> str:
    """Generate a repeatable string of the given length."""
    base = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    repetitions = (char_count // len(base)) + 1
    return (base * repetitions)[:char_count]


def _make_openai_sse_chunk(
    content: str,
    chunk_index: int = 0,
    model: str = "test-model",
    finish_reason: str | None = None,
) -> str:
    """Create a realistic OpenAI Chat Completions streaming chunk (SSE format).

    Matches the exact JSON structure returned by vLLM/OpenAI:
    data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"content":"..."}}]}
    """
    payload = {
        "id": f"chatcmpl-{chunk_index}",
        "object": "chat.completion.chunk",
        "created": 1749678185,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {orjson.dumps(payload).decode()}\n\n"


def _make_openai_sse_usage_chunk(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "test-model",
) -> str:
    """Create an OpenAI usage-only final chunk (stream_options include_usage)."""
    payload = {
        "id": "chatcmpl-final",
        "object": "chat.completion.chunk",
        "created": 1749678185,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return f"data: {orjson.dumps(payload).decode()}\n\n"


def _make_openai_done_chunk() -> str:
    """Create the final [DONE] SSE marker."""
    return "data: [DONE]\n\n"


def _generate_streaming_response(
    total_tokens: int,
    tokens_per_chunk: int,
    chars_per_token: int = 4,
    include_usage: bool = False,
    prompt_tokens: int = 100,
) -> list[str]:
    """Generate a realistic sequence of SSE chunks for an OpenAI streaming response.

    Args:
        total_tokens: Total output tokens to generate.
        tokens_per_chunk: Tokens per SSE chunk (typically 1 for streaming).
        chars_per_token: Characters per token (avg ~4 for English).
        include_usage: Whether to include a usage chunk at end.
        prompt_tokens: Prompt token count for usage chunk.

    Returns:
        List of SSE-formatted strings, each ending with \\n\\n.
    """
    chunks = []
    remaining = total_tokens
    idx = 0

    while remaining > 0:
        tokens_in_chunk = min(tokens_per_chunk, remaining)
        content = _make_text(tokens_in_chunk * chars_per_token)
        finish = "stop" if remaining <= tokens_per_chunk else None
        chunks.append(_make_openai_sse_chunk(content, idx, finish_reason=finish))
        remaining -= tokens_in_chunk
        idx += 1

    if include_usage:
        chunks.append(_make_openai_sse_usage_chunk(prompt_tokens, total_tokens))

    chunks.append(_make_openai_done_chunk())
    return chunks


def _sse_chunks_to_bytes(sse_strings: list[str]) -> list[bytes]:
    """Convert SSE strings to byte chunks (simulating HTTP transport)."""
    return [s.encode() for s in sse_strings]


async def _parse_sse_stream(byte_chunks: list[bytes]) -> list[SSEMessage]:
    """Parse byte chunks through AsyncSSEStreamReader."""

    async def _iter():
        for chunk in byte_chunks:
            yield chunk

    reader = AsyncSSEStreamReader(_iter())
    return await reader.read_complete_stream()


def _snapshot_bytes(func):
    """Run *func* inside a tracemalloc snapshot and return bytes allocated."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    snapshot_before = tracemalloc.take_snapshot()
    result = func()
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snapshot_after.compare_to(snapshot_before, "filename")
    total = sum(s.size_diff for s in stats if s.size_diff > 0)
    return total, result


def _snapshot_bytes_async(coro_func):
    """Run an async function inside tracemalloc and return bytes allocated."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    snapshot_before = tracemalloc.take_snapshot()
    result = asyncio.run(coro_func())
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snapshot_after.compare_to(snapshot_before, "filename")
    total = sum(s.size_diff for s in stats if s.size_diff > 0)
    return total, result


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_sse_chunk_memory() -> None:
    """Measure memory of individual SSE chunks and messages at various sizes."""
    print("\n=== SSE Chunk Memory ===")

    # Single SSE message memory at various sizes
    cases = [
        (4, "1-token"),
        (16, "4-token"),
        (64, "16-token"),
        (256, "64-token"),
        (1024, "256-token"),
    ]
    for content_chars, label in cases:

        def build(cc=content_chars):
            raw = _make_openai_sse_chunk(_make_text(cc), 0)
            return SSEMessage.parse(raw, time.perf_counter_ns())

        bytes_used, msg = _snapshot_bytes(build)
        print(
            f"\n  SSEMessage ({label}, {content_chars} chars content): "
            f"{bytes_used:,} bytes ({bytes_used / 1024:.1f} KB)"
        )
        assert bytes_used > 0

    # SSEField NamedTuple overhead
    def build_fields():
        return [SSEField(name="data", value=_make_text(200)) for _ in range(1000)]

    bytes_used, fields = _snapshot_bytes(build_fields)
    per_field = bytes_used / len(fields)
    print(
        f"\n  1000 SSEField NamedTuples (200 chars each):"
        f"\n    Total: {bytes_used:,} bytes ({bytes_used / 1024:.1f} KB)"
        f"\n    Per field: {per_field:.0f} bytes"
    )
    assert bytes_used > 0


def bench_sse_stream_accumulation() -> None:
    """Measure memory of accumulated SSE responses in a RequestRecord."""
    print("\n=== SSE Stream Accumulation ===")

    cases = [
        (10, 1, 4, "10tok-1tpc"),
        (50, 1, 4, "50tok-1tpc"),
        (100, 1, 4, "100tok-1tpc"),
        (500, 1, 4, "500tok-1tpc"),
        (1000, 1, 4, "1000tok-1tpc"),
        (2000, 1, 4, "2000tok-1tpc"),
        (100, 5, 4, "100tok-5tpc"),
        (500, 5, 4, "500tok-5tpc"),
        (1000, 10, 4, "1000tok-10tpc"),
    ]

    for output_tokens, tokens_per_chunk, chars_per_token, _label in cases:

        async def run(ot=output_tokens, tpc=tokens_per_chunk, cpt=chars_per_token):
            sse_strings = _generate_streaming_response(ot, tpc, cpt)
            byte_chunks = _sse_chunks_to_bytes(sse_strings)
            messages = await _parse_sse_stream(byte_chunks)

            def build():
                return RequestRecord(
                    model_name="test-model",
                    timestamp_ns=time.time_ns(),
                    start_perf_ns=time.perf_counter_ns(),
                    end_perf_ns=time.perf_counter_ns(),
                    responses=messages,
                )

            bytes_used, record = _snapshot_bytes(build)
            num_responses = len(record.responses)
            per_response = bytes_used / num_responses if num_responses else 0
            total_content_chars = ot * cpt

            print(
                f"\n  RequestRecord SSE accumulation ({ot} tokens, {tpc} tok/chunk):"
                f"\n    SSE messages: {num_responses}"
                f"\n    Total content: ~{total_content_chars:,} chars"
                f"\n    Total memory: {bytes_used:,} bytes ({bytes_used / 1024:.1f} KB)"
                f"\n    Per SSE message: {per_response:,.0f} bytes"
            )
            assert bytes_used > 0

        asyncio.run(run())


def bench_endpoint_parsing() -> None:
    """Measure memory of endpoint parsing: SSEMessage -> ParsedResponse."""
    print("\n=== Endpoint Parsing ===")

    cases = [
        (10, 1, "10tok"),
        (50, 1, "50tok"),
        (100, 1, "100tok"),
        (500, 1, "500tok"),
        (1000, 1, "1000tok"),
        (2000, 1, "2000tok"),
        (500, 5, "500tok-5tpc"),
        (1000, 10, "1000tok-10tpc"),
    ]

    for output_tokens, tokens_per_chunk, _label in cases:

        async def run(ot=output_tokens, tpc=tokens_per_chunk):
            model_endpoint = _make_model_endpoint()
            endpoint = ChatEndpoint(model_endpoint=model_endpoint)

            sse_strings = _generate_streaming_response(
                ot, tpc, chars_per_token=4, include_usage=True
            )
            byte_chunks = _sse_chunks_to_bytes(sse_strings)
            messages = await _parse_sse_stream(byte_chunks)

            record = RequestRecord(
                model_name="test-model",
                timestamp_ns=time.time_ns(),
                start_perf_ns=time.perf_counter_ns(),
                end_perf_ns=time.perf_counter_ns(),
                responses=messages,
            )

            def build():
                return endpoint.extract_response_data(record)

            bytes_used, parsed = _snapshot_bytes(build)
            num_parsed = len(parsed)
            per_parsed = bytes_used / num_parsed if num_parsed else 0

            print(
                f"\n  ParsedResponse list ({ot} tokens, {tpc} tok/chunk):"
                f"\n    Parsed responses: {num_parsed}"
                f"\n    Total memory: {bytes_used:,} bytes ({bytes_used / 1024:.1f} KB)"
                f"\n    Per ParsedResponse: {per_parsed:,.0f} bytes"
            )
            assert bytes_used > 0

        asyncio.run(run())


def bench_full_record_pipeline() -> None:
    """Measure memory of the full record: raw SSE -> ParsedResponseRecord."""
    print("\n=== Full Record Pipeline ===")

    cases = [
        (50, 1, 512, "short-response"),
        (100, 1, 512, "medium-response"),
        (500, 1, 2048, "long-response"),
        (1000, 1, 2048, "very-long-response"),
        (2000, 1, 8192, "huge-response"),
        (500, 5, 2048, "batched-response"),
        (1000, 10, 8192, "batched-large"),
    ]

    for output_tokens, tokens_per_chunk, prompt_chars, label in cases:

        async def run(
            ot=output_tokens, tpc=tokens_per_chunk, pc=prompt_chars, lbl=label
        ):
            model_endpoint = _make_model_endpoint()
            endpoint = ChatEndpoint(model_endpoint=model_endpoint)

            sse_strings = _generate_streaming_response(
                ot,
                tpc,
                chars_per_token=4,
                include_usage=True,
                prompt_tokens=pc // 4,
            )
            byte_chunks = _sse_chunks_to_bytes(sse_strings)
            messages = await _parse_sse_stream(byte_chunks)

            def build():
                request_info = RequestInfo(
                    model_endpoint=model_endpoint,
                    credit_num=0,
                    credit_phase=CreditPhase.PROFILING,
                    x_request_id=str(uuid.uuid4()),
                    x_correlation_id=str(uuid.uuid4()),
                    conversation_id="conv-0",
                    turn_index=0,
                    turns=[Turn(role="user", texts=[Text(contents=[_make_text(pc)])])],
                    drop_perf_ns=time.perf_counter_ns(),
                )

                record = RequestRecord(
                    request_info=request_info,
                    model_name="test-model",
                    timestamp_ns=time.time_ns(),
                    start_perf_ns=time.perf_counter_ns(),
                    end_perf_ns=time.perf_counter_ns(),
                    responses=messages,
                    turns=request_info.turns,
                )

                parsed_responses = endpoint.extract_response_data(record)

                return ParsedResponseRecord(
                    request=record,
                    responses=parsed_responses,
                    token_counts=TokenCounts(
                        input=pc // 4,
                        output=ot,
                    ),
                )

            bytes_used, parsed_record = _snapshot_bytes(build)
            raw_count = len(parsed_record.request.responses)
            parsed_count = len(parsed_record.responses)

            print(
                f"\n  Full ParsedResponseRecord ({lbl}):"
                f"\n    Prompt: {pc} chars, Output: {ot} tokens"
                f"\n    Raw SSE messages: {raw_count}, Parsed responses: {parsed_count}"
                f"\n    Total memory: {bytes_used:,} bytes ({bytes_used / 1024:.1f} KB, {bytes_used / (1024 * 1024):.2f} MB)"
            )
            assert bytes_used > 0

        asyncio.run(run())


def bench_concurrent_streaming_load() -> None:
    """Simulate concurrent in-flight streaming requests and measure total memory."""
    print("\n=== Concurrent Streaming Load ===")

    cases = [
        (1, 100, 1, 512),
        (10, 100, 1, 512),
        (50, 100, 1, 512),
        (100, 100, 1, 512),
        (10, 500, 1, 2048),
        (50, 500, 1, 2048),
        (100, 500, 1, 2048),
        (10, 1000, 1, 2048),
        (50, 1000, 1, 2048),
        (100, 1000, 1, 2048),
        (50, 2000, 1, 8192),
        (100, 2000, 1, 8192),
    ]

    for concurrency, output_tokens, tokens_per_chunk, prompt_chars in cases:

        async def run(
            c=concurrency, ot=output_tokens, tpc=tokens_per_chunk, pc=prompt_chars
        ):
            model_endpoint = _make_model_endpoint()
            endpoint = ChatEndpoint(model_endpoint=model_endpoint)

            sse_strings = _generate_streaming_response(
                ot,
                tpc,
                chars_per_token=4,
                include_usage=True,
                prompt_tokens=pc // 4,
            )
            byte_chunks = _sse_chunks_to_bytes(sse_strings)
            messages_template = await _parse_sse_stream(byte_chunks)

            def build():
                records = []
                for i in range(c):
                    messages = [
                        SSEMessage.parse(msg.extract_data_content() or "", msg.perf_ns)
                        if msg.extract_data_content()
                        else msg
                        for msg in messages_template
                    ]

                    request_info = RequestInfo(
                        model_endpoint=model_endpoint,
                        credit_num=i,
                        credit_phase=CreditPhase.PROFILING,
                        x_request_id=str(uuid.uuid4()),
                        x_correlation_id=str(uuid.uuid4()),
                        conversation_id=f"conv-{i}",
                        turn_index=0,
                        turns=[
                            Turn(
                                role="user",
                                texts=[Text(contents=[_make_text(pc)])],
                            )
                        ],
                        drop_perf_ns=time.perf_counter_ns(),
                    )

                    record = RequestRecord(
                        request_info=request_info,
                        model_name="test-model",
                        timestamp_ns=time.time_ns(),
                        start_perf_ns=time.perf_counter_ns(),
                        end_perf_ns=time.perf_counter_ns(),
                        responses=messages,
                        turns=request_info.turns,
                    )

                    parsed_responses = endpoint.extract_response_data(record)

                    parsed_record = ParsedResponseRecord(
                        request=record,
                        responses=parsed_responses,
                        token_counts=TokenCounts(
                            input=pc // 4,
                            output=ot,
                        ),
                    )
                    records.append(parsed_record)

                return records

            bytes_used, records = _snapshot_bytes(build)
            per_slot = bytes_used / c if c else 0
            total_mb = bytes_used / (1024 * 1024)

            print(
                f"\n  Concurrent streaming load:"
                f"\n    Concurrency: {c}"
                f"\n    Output tokens: {ot}, Prompt: {pc} chars"
                f"\n    SSE messages per request: {len(messages_template)}"
                f"\n    ─────────────────────────────────────"
                f"\n    Total memory: {bytes_used:,} bytes ({total_mb:.2f} MB)"
                f"\n    Per concurrent slot: {per_slot:,.0f} bytes ({per_slot / 1024:.1f} KB)"
            )
            assert bytes_used > 0

        asyncio.run(run())


def bench_sse_parsing_overhead() -> None:
    """Measure the overhead of SSE parsing vs raw data size."""
    print("\n=== SSE Parsing Overhead ===")

    cases = [50, 100, 500, 1000, 2000]

    for output_tokens in cases:

        async def run(ot=output_tokens):
            model_endpoint = _make_model_endpoint()
            endpoint = ChatEndpoint(model_endpoint=model_endpoint)

            sse_strings = _generate_streaming_response(
                ot, tokens_per_chunk=1, chars_per_token=4, include_usage=True
            )
            byte_chunks = _sse_chunks_to_bytes(sse_strings)
            raw_bytes = sum(len(c) for c in byte_chunks)

            messages = await _parse_sse_stream(byte_chunks)

            sse_mem, _ = _snapshot_bytes(
                lambda: [
                    SSEMessage.parse(s, time.perf_counter_ns())
                    for s in sse_strings
                    if not s.startswith("data: [DONE]")
                ]
            )

            record = RequestRecord(
                model_name="test-model",
                timestamp_ns=time.time_ns(),
                start_perf_ns=time.perf_counter_ns(),
                end_perf_ns=time.perf_counter_ns(),
                responses=messages,
            )

            parsed_mem, parsed = _snapshot_bytes(
                lambda: endpoint.extract_response_data(record)
            )

            total_mem = sse_mem + parsed_mem
            overhead = total_mem / raw_bytes if raw_bytes else 0

            print(
                f"\n  Parsing overhead ({ot} tokens):"
                f"\n    Raw SSE bytes: {raw_bytes:,} bytes ({raw_bytes / 1024:.1f} KB)"
                f"\n    SSEMessage objects: {sse_mem:,} bytes ({sse_mem / 1024:.1f} KB)"
                f"\n    ParsedResponse objects: {parsed_mem:,} bytes ({parsed_mem / 1024:.1f} KB)"
                f"\n    Total in-memory: {total_mem:,} bytes ({total_mem / 1024:.1f} KB)"
                f"\n    Overhead ratio: {overhead:.2f}x raw bytes"
            )
            assert total_mem > 0

        asyncio.run(run())


if __name__ == "__main__":
    bench_sse_chunk_memory()
    bench_sse_stream_accumulation()
    bench_endpoint_parsing()
    bench_full_record_pipeline()
    bench_concurrent_streaming_load()
    bench_sse_parsing_overhead()
