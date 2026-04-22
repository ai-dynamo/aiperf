#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory profiling benchmarks for Worker data structures under various load conditions.

Measures memory consumed by key Worker data structures to help capacity planning:
- UserSession (cached per concurrent conversation)
- Conversation + Turn (dataset data held in session)
- RequestInfo (built per request, contains growing turn history)
- RequestRecord + SSE responses (inference results)
- Credit + CreditContext (per-credit tracking via msgspec Structs)

Run: uv run python dev/benchmarks/worker_memory_profile.py
"""

import time
import tracemalloc
import uuid

from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    Conversation,
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
from aiperf.credit.structs import Credit, CreditContext
from aiperf.workers.session_manager import UserSession, UserSessionManager


def _make_model_endpoint() -> ModelEndpointInfo:
    """Create a minimal ModelEndpointInfo for tests."""
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy="round_robin",
        ),
        endpoint=EndpointInfo(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text(char_count: int) -> str:
    """Generate a repeatable string of the given length."""
    base = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    repetitions = (char_count // len(base)) + 1
    return (base * repetitions)[:char_count]


def _make_turn(role: str, prompt_chars: int) -> Turn:
    """Create a Turn with a text payload of the given size."""
    return Turn(
        role=role,
        texts=[Text(contents=[_make_text(prompt_chars)])],
    )


def _make_conversation(
    num_turns: int, prompt_chars: int, session_id: str = "conv-0"
) -> Conversation:
    """Create a Conversation with *num_turns* user turns."""
    return Conversation(
        session_id=session_id,
        turns=[_make_turn("user", prompt_chars) for _ in range(num_turns)],
    )


def _make_credit(
    credit_id: int,
    turn_index: int,
    num_turns: int,
    conversation_id: str = "conv-0",
) -> Credit:
    return Credit(
        id=credit_id,
        phase=CreditPhase.PROFILING,
        conversation_id=conversation_id,
        x_correlation_id=str(uuid.uuid4()),
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=time.time_ns(),
    )


def _make_sse_message(response_chars: int) -> SSEMessage:
    """Create an SSE message simulating a streaming chunk."""
    return SSEMessage(
        perf_ns=time.perf_counter_ns(),
        packets=[("data", _make_text(response_chars))],
    )


def _snapshot_bytes(func):
    """Run *func* inside a tracemalloc snapshot and return bytes allocated."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    snapshot_before = tracemalloc.take_snapshot()
    result = func()
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Sum traced allocations that are new or grew between snapshots
    stats = snapshot_after.compare_to(snapshot_before, "filename")
    total = sum(s.size_diff for s in stats if s.size_diff > 0)
    return total, result


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


PROMPT_SIZES = {
    "tiny": 128,  # ~128 chars, very short prompt
    "small": 512,  # ~512 chars, typical short prompt
    "medium": 2_048,  # ~2 KB, typical chat prompt
    "large": 8_192,  # ~8 KB, long context prompt
    "xlarge": 32_768,  # ~32 KB, document-level prompt
}

RESPONSE_SIZES = {
    "tiny": 64,  # ~64 chars, short completion
    "small": 256,  # ~256 chars, typical completion
    "medium": 2_048,  # ~2 KB, paragraph response
    "large": 8_192,  # ~8 KB, long response
}


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_single_object_memory() -> None:
    """Measure memory of individual data structures at various payload sizes."""

    print("\n=== Single Turn Memory ===")
    single_turn_cases = [
        ("tiny", 128),
        ("small", 512),
        ("medium", 2_048),
        ("large", 8_192),
        ("xlarge", 32_768),
    ]
    for prompt_label, prompt_chars in single_turn_cases:
        bytes_used, _ = _snapshot_bytes(lambda pc=prompt_chars: _make_turn("user", pc))
        print(
            f"\n  Turn ({prompt_label}, {prompt_chars} chars): "
            f"{bytes_used:,} bytes ({bytes_used / 1024:.1f} KB)"
        )
        assert bytes_used >= prompt_chars

    print("\n=== Conversation Memory ===")
    conversation_cases = [
        (1, 512),
        (5, 512),
        (10, 512),
        (20, 512),
        (5, 8_192),
        (10, 8_192),
    ]
    for num_turns, prompt_chars in conversation_cases:
        bytes_used, _ = _snapshot_bytes(
            lambda nt=num_turns, pc=prompt_chars: _make_conversation(nt, pc)
        )
        print(
            f"\n  Conversation ({num_turns} turns x {prompt_chars} chars): "
            f"{bytes_used:,} bytes ({bytes_used / 1024:.1f} KB)"
        )
        assert bytes_used > 0

    print("\n=== RequestRecord Memory ===")
    request_record_cases = [
        (1, 64),
        (10, 64),
        (50, 64),
        (100, 64),
        (50, 256),
        (100, 256),
    ]
    for num_chunks, chunk_chars in request_record_cases:

        def build(nc=num_chunks, cc=chunk_chars):
            return RequestRecord(
                model_name="test-model",
                timestamp_ns=time.time_ns(),
                start_perf_ns=time.perf_counter_ns(),
                end_perf_ns=time.perf_counter_ns(),
                responses=[_make_sse_message(cc) for _ in range(nc)],
            )

        bytes_used, _ = _snapshot_bytes(build)
        print(
            f"\n  RequestRecord ({num_chunks} SSE chunks x {chunk_chars} chars): "
            f"{bytes_used:,} bytes ({bytes_used / 1024:.1f} KB)"
        )
        assert bytes_used > 0

    print("\n=== Credit + CreditContext Memory ===")
    bytes_used, _ = _snapshot_bytes(
        lambda: CreditContext(
            credit=_make_credit(0, 0, 1),
            drop_perf_ns=time.perf_counter_ns(),
        )
    )
    print(f"\n  Credit + CreditContext: {bytes_used:,} bytes")
    assert bytes_used < 4096


def bench_session_cache_memory() -> None:
    """Measure memory of the UserSessionManager cache under load."""

    print("\n=== Session Cache Memory ===")
    cases = [
        (1, 1, 512),
        (10, 1, 512),
        (50, 1, 512),
        (100, 1, 512),
        (500, 1, 512),
        (10, 5, 512),
        (50, 5, 512),
        (100, 5, 512),
        (10, 10, 2_048),
        (50, 10, 2_048),
        (100, 10, 2_048),
        (100, 5, 8_192),
        (500, 5, 8_192),
    ]
    for num_sessions, num_turns, prompt_chars in cases:

        def build(ns=num_sessions, nt=num_turns, pc=prompt_chars):
            mgr = UserSessionManager()
            for i in range(ns):
                conv = _make_conversation(nt, pc, session_id=f"conv-{i}")
                mgr.create_and_store(
                    x_correlation_id=f"corr-{i}",
                    conversation=conv,
                    num_turns=nt,
                )
            return mgr

        bytes_used, mgr = _snapshot_bytes(build)
        per_session = bytes_used / num_sessions if num_sessions else 0
        total_mb = bytes_used / (1024 * 1024)
        print(
            f"\n  SessionCache ({num_sessions} sessions x {num_turns} turns x {prompt_chars} chars):"
            f"\n    Total: {bytes_used:,} bytes ({total_mb:.2f} MB)"
            f"\n    Per session: {per_session:,.0f} bytes ({per_session / 1024:.1f} KB)"
        )
        assert bytes_used > 0


def bench_multi_turn_growth() -> None:
    """Measure how memory grows as a session accumulates assistant responses."""

    print("\n=== Multi-Turn Growth ===")
    cases = [
        (1, 512, 256),
        (5, 512, 256),
        (10, 512, 256),
        (20, 512, 256),
        (5, 2_048, 2_048),
        (10, 2_048, 2_048),
        (10, 8_192, 8_192),
    ]
    for num_turns, prompt_chars, response_chars in cases:

        def build(nt=num_turns, pc=prompt_chars, rc=response_chars):
            conv = _make_conversation(nt, pc)
            session = UserSession(
                x_correlation_id="test-corr",
                num_turns=nt,
                conversation=conv,
                turn_list=[],
            )
            for i in range(nt):
                session.advance_turn(i)
                session.store_response(_make_turn("assistant", rc))
            return session

        bytes_used, session = _snapshot_bytes(build)
        turn_list_len = len(session.turn_list)
        per_turn_pair = bytes_used / num_turns if num_turns else 0
        print(
            f"\n  Session after {num_turns} turns (prompt={prompt_chars}, response={response_chars}):"
            f"\n    Total: {bytes_used:,} bytes ({bytes_used / 1024:.1f} KB)"
            f"\n    turn_list length: {turn_list_len} (user+assistant pairs)"
            f"\n    Per turn pair: {per_turn_pair:,.0f} bytes ({per_turn_pair / 1024:.1f} KB)"
        )
        assert turn_list_len == num_turns * 2


def bench_concurrent_load_memory() -> None:
    """Simulate realistic concurrent load and measure total Worker memory footprint.

    Models the steady-state memory of a Worker with N concurrent requests,
    each with its own session, credit context, and in-flight request record.
    """

    print("\n=== Concurrent Load Memory ===")
    cases = [
        (1, 1, 512, 256, 10),
        (10, 1, 512, 256, 10),
        (50, 1, 512, 256, 10),
        (100, 1, 512, 256, 10),
        (500, 1, 512, 256, 10),
        (10, 5, 2_048, 2_048, 50),
        (50, 5, 2_048, 2_048, 50),
        (100, 5, 2_048, 2_048, 50),
        (10, 10, 8_192, 8_192, 100),
        (50, 10, 8_192, 8_192, 100),
        (100, 10, 8_192, 8_192, 100),
    ]
    for concurrency, num_turns, prompt_chars, response_chars, sse_chunks in cases:
        mid_turn = max(1, num_turns // 2)

        def build(
            c=concurrency,
            nt=num_turns,
            pc=prompt_chars,
            rc=response_chars,
            sc=sse_chunks,
            mt=mid_turn,
        ):
            sessions = []
            credits = []
            records = []

            for i in range(c):
                conv = _make_conversation(nt, pc, session_id=f"conv-{i}")
                session = UserSession(
                    x_correlation_id=f"corr-{i}",
                    num_turns=nt,
                    conversation=conv,
                    turn_list=[],
                )
                for t in range(mt):
                    session.advance_turn(t)
                    session.store_response(_make_turn("assistant", rc))
                sessions.append(session)

                credit = _make_credit(i, mt, nt, conversation_id=f"conv-{i}")
                ctx = CreditContext(credit=credit, drop_perf_ns=time.perf_counter_ns())
                credits.append(ctx)

                record = RequestRecord(
                    model_name="test-model",
                    timestamp_ns=time.time_ns(),
                    start_perf_ns=time.perf_counter_ns(),
                    end_perf_ns=time.perf_counter_ns(),
                    responses=[_make_sse_message(rc // sc or 1) for _ in range(sc)],
                )
                records.append(record)

            return sessions, credits, records

        bytes_used, (sessions, credits, records) = _snapshot_bytes(build)
        per_worker_slot = bytes_used / concurrency if concurrency else 0
        total_mb = bytes_used / (1024 * 1024)

        print(
            f"\n  Worker load simulation:"
            f"\n    Concurrency: {concurrency}"
            f"\n    Turns: {num_turns} ({mid_turn} completed)"
            f"\n    Prompt: {prompt_chars} chars, Response: {response_chars} chars"
            f"\n    SSE chunks per request: {sse_chunks}"
            f"\n    ─────────────────────────────────────"
            f"\n    Total memory: {bytes_used:,} bytes ({total_mb:.2f} MB)"
            f"\n    Per concurrent slot: {per_worker_slot:,.0f} bytes ({per_worker_slot / 1024:.1f} KB)"
        )
        assert bytes_used > 0


def bench_request_info_growth() -> None:
    """Measure how RequestInfo grows with conversation history.

    RequestInfo.turns contains the full conversation history sent to the
    inference server. This grows linearly with each turn in a multi-turn
    conversation and is the primary memory concern for long conversations.
    """

    print("\n=== RequestInfo Growth ===")
    cases = [
        (1, 512, 256),
        (5, 512, 256),
        (10, 512, 256),
        (20, 512, 256),
        (5, 2_048, 2_048),
        (10, 2_048, 2_048),
        (20, 2_048, 2_048),
        (10, 8_192, 8_192),
    ]
    for completed_turns, prompt_chars, response_chars in cases:

        def build(ct=completed_turns, pc=prompt_chars, rc=response_chars):
            turns = []
            for _ in range(ct):
                turns.append(_make_turn("user", pc))
                turns.append(_make_turn("assistant", rc))

            return RequestInfo(
                model_endpoint=_make_model_endpoint(),
                credit_num=0,
                credit_phase=CreditPhase.PROFILING,
                x_request_id=str(uuid.uuid4()),
                x_correlation_id=str(uuid.uuid4()),
                conversation_id="conv-0",
                turn_index=ct,
                turns=turns,
                drop_perf_ns=time.perf_counter_ns(),
            )

        bytes_used, req_info = _snapshot_bytes(build)
        num_turn_objects = len(req_info.turns)
        per_turn = bytes_used / num_turn_objects if num_turn_objects else 0
        print(
            f"\n  RequestInfo ({completed_turns} completed turns, prompt={prompt_chars}, response={response_chars}):"
            f"\n    Total: {bytes_used:,} bytes ({bytes_used / 1024:.1f} KB)"
            f"\n    Turn objects: {num_turn_objects}"
            f"\n    Per turn object: {per_turn:,.0f} bytes"
        )
        assert bytes_used > 0


if __name__ == "__main__":
    bench_single_object_memory()
    bench_session_cache_memory()
    bench_multi_turn_growth()
    bench_concurrent_load_memory()
    bench_request_info_growth()
