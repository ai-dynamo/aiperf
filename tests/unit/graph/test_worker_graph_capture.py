# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker-side graph response capture: what each reply shape pools, and how failures map to sentinels without escaping the credit task."""

import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from pytest import param

from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.credit.structs import Credit, CreditContext
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)
from aiperf.graph.dynamic_pool import (
    GraphCapturedReply,
    GraphDynamicPool,
    GraphPoolSentinel,
)
from aiperf.workers.worker import Worker

TRACE = "t-1"
# Instance id is ``{template}::{nonce}``; the worker strips the ``::{nonce}``
# back to the base template id for catalog / store lookups.
INSTANCE = "t-1::inst0"


def _credit_context() -> CreditContext:
    """A profiling credit context for node ordinal 0 of the single test trace."""
    return CreditContext(
        credit=Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id=TRACE,
            x_correlation_id="t-1::corr0",
            turn_index=0,
            num_turns=1,
            issued_at_ns=0,
            trace_id=INSTANCE,
            node_ordinal=0,
        ),
        drop_perf_ns=0,
    )


def _mock_worker(
    client: GraphSegmentUnifiedClient,
    *,
    record: MagicMock,
    assistant_turn: Any,
) -> MagicMock:
    """A mock Worker self with the real capture methods bound and every collaborator stubbed."""
    self = MagicMock()
    self._graph_store_reader = MagicMock(return_value=client)
    self._graph_dynamic_pool = GraphDynamicPool(max_bytes=1024 * 1024)
    self.model_endpoint.primary_model_name = "mock-model"
    self.model_endpoint.endpoint.cache_bust = CacheBustTarget.NONE
    self.model_endpoint.endpoint.use_legacy_max_tokens = False
    self._build_graph_request_info = MagicMock(return_value=MagicMock())
    self.inference_client.send_request = AsyncMock(return_value=record)
    self.inference_client.endpoint.build_assistant_turn = MagicMock(
        return_value=assistant_turn
    )
    self._send_inference_result_message = AsyncMock()
    self._set_graph_envelope_missing = MagicMock()
    self.task_stats = MagicMock()
    # Bind the real methods under test onto the mock self.
    self._process_graph_credit = types.MethodType(Worker._process_graph_credit, self)
    self._dispatch_graph_request = types.MethodType(
        Worker._dispatch_graph_request, self
    )
    self._graph_capture_value = types.MethodType(Worker._graph_capture_value, self)
    # Real, not a stub: it is what pairs the error-record emit with the
    # ``record_emitted`` flag, and the catch-all's use of it is under test.
    self._fail_graph_credit = types.MethodType(Worker._fail_graph_credit, self)
    return self


def _text_turn(text: str) -> types.SimpleNamespace:
    """Completions-endpoint shape: assistant text only, no raw_messages."""
    return types.SimpleNamespace(
        texts=[types.SimpleNamespace(contents=[text])],
        raw_messages=None,
    )


TOOL_CALLS = [
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
    }
]


def _tool_calls_turn(content: str | None) -> types.SimpleNamespace:
    """Chat-endpoint shape: ``build_assistant_turn`` returned ``raw_messages``."""
    return types.SimpleNamespace(
        texts=None,
        raw_messages=[
            {"role": "assistant", "content": content, "tool_calls": TOOL_CALLS}
        ],
    )


def _multi_entry_turn() -> types.SimpleNamespace:
    """Endpoint returned more than one assistant message (e.g. openai_responses)."""
    return types.SimpleNamespace(
        texts=None,
        raw_messages=[
            {"role": "assistant", "content": "first"},
            {"role": "assistant", "content": "second"},
        ],
    )


def _unserializable_turn() -> types.SimpleNamespace:
    """Single assistant message whose content cannot be orjson-serialized."""
    return types.SimpleNamespace(
        texts=None,
        raw_messages=[{"role": "assistant", "content": object()}],
    )


def _record(error: str | None = None) -> MagicMock:
    """An inference record that either succeeded or carries a transport-level error."""
    record = MagicMock()
    record.error = error
    return record


async def _seeded_client(tmp_path: Path, envelope: dict) -> GraphSegmentUnifiedClient:
    """A finalized unified store holding one user segment plus the given node envelope."""
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id="bench")
    handle = store.put_segment("seg0", "user", "hello")
    store.add_node_manifest(TRACE, 0, orjson.dumps({"handles": [handle], **envelope}))
    await store.finalize()
    return GraphSegmentUnifiedClient(tmp_path, "bench").open()


async def _run(tmp_path: Path, envelope: dict, **worker_kw: Any) -> MagicMock:
    """Process one graph credit against a freshly seeded store, returning the mock worker."""
    client = await _seeded_client(tmp_path, envelope)
    self = _mock_worker(client, **worker_kw)
    ctx = _credit_context()
    await self._process_graph_credit(ctx, "x-req-1", None)
    self._ctx = ctx
    return self


@pytest.mark.asyncio
async def test_captured_node_pools_response_text(tmp_path: Path) -> None:
    """A plain text reply on a captured node pools the joined text with no error."""
    self = await _run(
        tmp_path,
        {"capture": True},
        record=_record(),
        assistant_turn=_text_turn("the answer"),
    )
    assert self._graph_dynamic_pool.get(INSTANCE, 0) == GraphCapturedReply(
        text="the answer", message_json=None
    )
    assert self._ctx.error is None


@pytest.mark.asyncio
async def test_tool_calls_only_reply_pools_structured_capture(tmp_path: Path) -> None:
    """A tool_calls-only reply is a structured capture, NOT EMPTY."""
    self = await _run(
        tmp_path,
        {"capture": True},
        record=_record(),
        assistant_turn=_tool_calls_turn(None),
    )
    value = self._graph_dynamic_pool.get(INSTANCE, 0)
    assert isinstance(value, GraphCapturedReply)
    assert value.text == ""
    assert orjson.loads(value.message_json) == {
        "role": "assistant",
        "content": None,
        "tool_calls": TOOL_CALLS,
    }


@pytest.mark.asyncio
async def test_mixed_text_and_tool_calls_reply_captures_both(tmp_path: Path) -> None:
    """A reply with both text and tool_calls pools the text AND the whole assistant message."""
    turn = _tool_calls_turn("let me check")
    self = await _run(
        tmp_path,
        {"capture": True},
        record=_record(),
        assistant_turn=turn,
    )
    value = self._graph_dynamic_pool.get(INSTANCE, 0)
    assert isinstance(value, GraphCapturedReply)
    assert value.text == "let me check"
    assert value.message_json == orjson.dumps(turn.raw_messages[0]).decode()


@pytest.mark.parametrize(
    "record,assistant_turn,expected",
    [
        param(_record(), _text_turn(""), GraphPoolSentinel.EMPTY, id="text_joins_to_empty"),
        param(_record(), None, GraphPoolSentinel.EMPTY, id="no_assistant_turn"),
        param(_record(error="boom"), None, GraphPoolSentinel.FAILED, id="record_carries_error"),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_unusable_reply_pools_sentinel(
    tmp_path: Path,
    record: MagicMock,
    assistant_turn: Any,
    expected: GraphPoolSentinel,
) -> None:
    """A reply with no usable content pools EMPTY; a failed request pools FAILED."""
    self = await _run(
        tmp_path, {"capture": True}, record=record, assistant_turn=assistant_turn
    )
    assert self._graph_dynamic_pool.get(INSTANCE, 0) is expected


@pytest.mark.asyncio
async def test_uncaptured_node_never_touches_pool(tmp_path: Path) -> None:
    """A node without the capture flag writes nothing to the pool at all."""
    self = await _run(
        tmp_path,
        {},
        record=_record(),
        assistant_turn=_text_turn("ignored"),
    )
    assert self._graph_dynamic_pool.get(INSTANCE, 0) is None
    assert self._graph_dynamic_pool.total_bytes == 0


@pytest.mark.asyncio
async def test_send_exception_pools_failed_and_attributes_error(tmp_path: Path) -> None:
    """A raising send pools FAILED, attributes the error on the context, and emits a synthetic error record."""
    # The outer handler only logs, so an escaping exception would count the credit
    # as SUCCESS with no record and starve the RecordsManager barrier.
    client = await _seeded_client(tmp_path, {"capture": True})
    self = _mock_worker(client, record=_record(), assistant_turn=None)
    self.inference_client.send_request = AsyncMock(side_effect=TimeoutError("dead"))
    self._send_graph_error_record = AsyncMock()

    ctx = _credit_context()
    await self._process_graph_credit(ctx, "x-req-1", None)

    assert self._graph_dynamic_pool.get(INSTANCE, 0) is GraphPoolSentinel.FAILED
    assert ctx.error is not None and ctx.error.type == "TimeoutError"
    self._send_graph_error_record.assert_awaited_once()
    # The catch-all must also MARK the emission. Without this the task-level
    # lockstep guard in _on_credit_drop_message_task sees record_emitted False
    # and appends a second error record for the same credit, double-counting it
    # in total_records, error_records, and the JSONL export.
    assert ctx.record_emitted is True


@pytest.mark.asyncio
async def test_capture_extraction_failure_sets_error_and_pools_failed(
    tmp_path: Path,
) -> None:
    """A raising assistant-turn extractor pools FAILED with capture_failed attribution."""
    self = await _run(
        tmp_path,
        {"capture": True},
        record=_record(),
        assistant_turn=None,
    )
    # Re-run with a raising extractor against the same harness shape.
    self.inference_client.endpoint.build_assistant_turn = MagicMock(
        side_effect=RuntimeError("bad parse")
    )
    ctx = _credit_context()
    await self._process_graph_credit(ctx, "x-req-2", None)

    assert self._graph_dynamic_pool.get(INSTANCE, 0) is GraphPoolSentinel.FAILED
    assert ctx.error is not None and "aiperf.graph.capture_failed" in ctx.error


@pytest.mark.asyncio
async def test_multi_entry_raw_messages_pools_failed_with_pointed_error(
    tmp_path: Path,
) -> None:
    """A single-message capture cannot represent >1 assistant message."""
    self = await _run(
        tmp_path,
        {"capture": True},
        record=_record(),
        assistant_turn=_multi_entry_turn(),
    )
    assert self._graph_dynamic_pool.get(INSTANCE, 0) is GraphPoolSentinel.FAILED
    assert self._ctx.error == (
        "aiperf.graph.capture_failed: multi-entry raw_messages (2 entries) is "
        "not representable in a single-message capture; single-message "
        "endpoints only"
    )


@pytest.mark.asyncio
async def test_unserializable_reply_pools_failed_without_escape(
    tmp_path: Path,
) -> None:
    """An assistant message that orjson cannot serialize must map to FAILED with capture_failed attribution."""
    self = await _run(
        tmp_path,
        {"capture": True},
        record=_record(),
        assistant_turn=_unserializable_turn(),
    )
    assert self._graph_dynamic_pool.get(INSTANCE, 0) is GraphPoolSentinel.FAILED
    assert self._ctx.error is not None
    assert "aiperf.graph.capture_failed" in self._ctx.error


@pytest.mark.asyncio
async def test_trace_end_mid_flight_defers_eviction(tmp_path: Path) -> None:
    """GraphTraceEnd during processing evicts only after the bracket closes."""
    client = await _seeded_client(tmp_path, {"capture": True})
    self = _mock_worker(client, record=_record(), assistant_turn=_text_turn("late"))

    pool = self._graph_dynamic_pool

    async def _send(request_info, first_token_callback=None):
        # TraceEnd lands while the request is in flight.
        pool.trace_end(INSTANCE)
        return _record()

    self.inference_client.send_request = AsyncMock(side_effect=_send)
    await self._process_graph_credit(_credit_context(), "x-req-1", None)

    # The capture write landed in a live entry, then the deferred end evicted.
    assert pool.get(INSTANCE, 0) is None
    assert pool.total_bytes == 0
