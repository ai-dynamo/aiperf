# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic-content slots end-to-end through the real content plane.

The marquee assertions of the dynamic-content pool: a native
two-node graph is lowered (`parse_native`), drained into a REAL unified store
(`build_unified_trie_store_interned`), and driven through the REAL worker
credit path (mock-self harness) with a REAL `GraphDynamicPool` — proving
request 2's wire payload contains request 1's actual response text, omission
shapes for FAILED producers on both `{"s"}` and `{"sv"}` slots, and the loud
`pool_missing` error attribution when stickiness is broken.
"""

import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.credit.structs import Credit, CreditContext
from aiperf.dataset.graph.parser import parse_native
from aiperf.dataset.graph.segment_ir.store_builder import (
    build_unified_trie_store_interned,
)
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

PLANNER_REVIEWER = """
graph:
  nodes:
    plan:
      prompt: [{role: user, content: "Make a plan."}]
      output: plan_out
    review:
      prompt:
        - role: user
          content: ["Review this plan: ", "@plan_out"]
      output: review_out
  edges:
    - {source: START, target: plan}
    - {source: plan, target: review}
    - {source: review, target: END}
traces:
  - id: t1
"""

ACCUMULATE_CHAIN = """
graph:
  state:
    hist: {type: messages, reducer: add_messages}
  nodes:
    a:
      prompt: ["@hist", {role: user, content: "turn one"}]
      output: hist
    b:
      prompt: ["@hist", {role: user, content: "turn two"}]
      output: hist
    c:
      prompt: ["@hist", {role: user, content: "turn three"}]
      output: c_out
  edges:
    - {source: START, target: a}
    - {source: a, target: b}
    - {source: b, target: c}
    - {source: c, target: END}
traces:
  - id: t1
    initial_state:
      hist:
        - {role: system, content: "be brief"}
"""


MERGE_CHAIN = """
graph:
  state:
    hist: {type: messages, reducer: add_messages}
  nodes:
    a:
      prompt: ["@hist", {role: user, content: "warm up"}]
      output: hist
    b:
      prompt: ["@hist", {role: user, content: "second question"}]
      output: b_out
  edges:
    - {source: START, target: a}
    - {source: a, target: b}
    - {source: b, target: END}
traces:
  - id: t1
    initial_state:
      hist:
        - {role: user, content: "first question"}
"""


async def _built_client(
    tmp_path: Path, yaml_text: str
) -> tuple[GraphSegmentUnifiedClient, dict[str, int]]:
    p = tmp_path / "workload.yaml"
    p.write_text(yaml_text)
    parsed = parse_native(p)
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id="bench")
    catalog = await build_unified_trie_store_interned(parsed, store)
    return GraphSegmentUnifiedClient(tmp_path, "bench").open(), catalog["t1"]


def _worker(client: GraphSegmentUnifiedClient) -> MagicMock:
    self = MagicMock()
    self._graph_store_reader = MagicMock(return_value=client)
    self._graph_dynamic_pool = GraphDynamicPool(max_bytes=1024 * 1024)
    self.model_endpoint.primary_model_name = "mock-model"
    endpoint = self.model_endpoint.endpoint
    endpoint.cache_bust = CacheBustTarget.NONE
    endpoint.use_legacy_max_tokens = False
    endpoint.streaming = False
    endpoint.extra = None
    endpoint.use_server_token_count = False
    self._build_graph_request_info = MagicMock(return_value=MagicMock())
    self._send_inference_result_message = AsyncMock()
    self._send_graph_error_record = AsyncMock()
    self._set_graph_envelope_missing = MagicMock()
    self.task_stats = MagicMock()
    self.run.benchmark_id = "bench"
    self._process_graph_credit = types.MethodType(Worker._process_graph_credit, self)
    self._dispatch_graph_request = types.MethodType(
        Worker._dispatch_graph_request, self
    )
    self._graph_capture_value = types.MethodType(Worker._graph_capture_value, self)
    return self


def _respond_with(self: MagicMock, text: str) -> None:
    record = MagicMock()
    record.error = None
    self.inference_client.send_request = AsyncMock(return_value=record)
    self.inference_client.endpoint.build_assistant_turn = MagicMock(
        return_value=types.SimpleNamespace(
            texts=[types.SimpleNamespace(contents=[text])],
            raw_messages=None,
        )
    )


# Instance id is ``{template}::{nonce}``; the worker strips ``::{nonce}`` back
# to the base template id ("t1") for catalog / store / dynamic-pool keying.
INSTANCE = "t1::inst0"


def _ctx(ordinal: int) -> CreditContext:
    return CreditContext(
        credit=Credit(
            id=ordinal + 1,
            phase=CreditPhase.PROFILING,
            conversation_id="t1",
            x_correlation_id="t1::corr",
            turn_index=0,
            num_turns=1,
            issued_at_ns=0,
            trace_id=INSTANCE,
            node_ordinal=ordinal,
        ),
        drop_perf_ns=0,
    )


def _sent_payload(self: MagicMock) -> dict:
    return self._build_graph_request_info.call_args[0][1]


@pytest.mark.asyncio
async def test_request_two_contains_response_one(tmp_path: Path) -> None:
    client, ordinals = await _built_client(tmp_path, PLANNER_REVIEWER)
    self = _worker(client)

    _respond_with(self, "1. write tests 2. ship")
    await self._process_graph_credit(_ctx(ordinals["plan"]), "x-req-1", None)

    ctx = _ctx(ordinals["review"])
    await self._process_graph_credit(ctx, "x-req-2", None)

    payload = _sent_payload(self)
    assert payload["messages"] == [
        {"role": "user", "content": "Review this plan: 1. write tests 2. ship"}
    ]
    assert ctx.error is None


@pytest.mark.asyncio
async def test_failed_producer_composes_static_prefix_only(tmp_path: Path) -> None:
    client, ordinals = await _built_client(tmp_path, PLANNER_REVIEWER)
    self = _worker(client)
    self._graph_dynamic_pool.put(
        INSTANCE, "profiling", ordinals["plan"], GraphPoolSentinel.FAILED
    )
    _respond_with(self, "unused")

    await self._process_graph_credit(_ctx(ordinals["review"]), "x-req-2", None)

    assert _sent_payload(self)["messages"] == [
        {"role": "user", "content": "Review this plan: "}
    ]


@pytest.mark.asyncio
async def test_accumulated_history_chain_splices_in_order(tmp_path: Path) -> None:
    client, ordinals = await _built_client(tmp_path, ACCUMULATE_CHAIN)
    self = _worker(client)

    _respond_with(self, "answer one")
    await self._process_graph_credit(_ctx(ordinals["a"]), "x-req-1", None)

    _respond_with(self, "answer two")
    await self._process_graph_credit(_ctx(ordinals["b"]), "x-req-2", None)
    b_payload = _sent_payload(self)
    # Full alternation: b sees the seed, a's authored turn, a's live reply,
    # then its own turn.
    assert b_payload["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "turn two"},
    ]

    _respond_with(self, "unused")
    await self._process_graph_credit(_ctx(ordinals["c"]), "x-req-3", None)
    c_payload = _sent_payload(self)
    assert c_payload["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "turn two"},
        {"role": "assistant", "content": "answer two"},
        {"role": "user", "content": "turn three"},
    ]


@pytest.mark.asyncio
async def test_tool_calls_capture_splices_verbatim_assistant_message(
    tmp_path: Path,
) -> None:
    """A structured (tool_calls) capture's ``{"s"}`` splice is the verbatim
    recorded assistant message -- tool_calls survive into the successor's wire
    body, byte-matching the legacy child-seed rendering."""
    client, ordinals = await _built_client(tmp_path, ACCUMULATE_CHAIN)
    self = _worker(client)
    tool_calls_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    }
    pool = self._graph_dynamic_pool
    pool.put(
        INSTANCE,
        "profiling",
        ordinals["a"],
        GraphCapturedReply(text="", message_json=orjson.dumps(tool_calls_msg).decode()),
    )
    pool.put(
        INSTANCE, "profiling", ordinals["b"], GraphCapturedReply(text="answer two")
    )
    _respond_with(self, "unused")

    await self._process_graph_credit(_ctx(ordinals["c"]), "x-req-3", None)

    assert _sent_payload(self)["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "turn one"},
        tool_calls_msg,
        {"role": "user", "content": "turn two"},
        {"role": "assistant", "content": "answer two"},
        {"role": "user", "content": "turn three"},
    ]


@pytest.mark.asyncio
async def test_composed_slot_concatenates_text_of_structured_capture(
    tmp_path: Path,
) -> None:
    """A ``{"sv"}`` block part uses the structured capture's ``.text`` (the
    reply's replayable text), never its message JSON."""
    client, ordinals = await _built_client(tmp_path, PLANNER_REVIEWER)
    self = _worker(client)
    plan_msg = {
        "role": "assistant",
        "content": "1. write tests 2. ship",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    }
    self._graph_dynamic_pool.put(
        INSTANCE,
        "profiling",
        ordinals["plan"],
        GraphCapturedReply(
            text="1. write tests 2. ship",
            message_json=orjson.dumps(plan_msg).decode(),
        ),
    )
    _respond_with(self, "unused")

    await self._process_graph_credit(_ctx(ordinals["review"]), "x-req-2", None)

    assert _sent_payload(self)["messages"] == [
        {"role": "user", "content": "Review this plan: 1. write tests 2. ship"}
    ]


@pytest.mark.asyncio
async def test_failed_array_slot_omits_assistant_turn(tmp_path: Path) -> None:
    client, ordinals = await _built_client(tmp_path, ACCUMULATE_CHAIN)
    self = _worker(client)
    pool = self._graph_dynamic_pool
    pool.put(
        INSTANCE, "profiling", ordinals["a"], GraphCapturedReply(text="answer one")
    )
    pool.put(INSTANCE, "profiling", ordinals["b"], GraphPoolSentinel.FAILED)
    _respond_with(self, "unused")

    await self._process_graph_credit(_ctx(ordinals["c"]), "x-req-3", None)

    # b's reply FAILED -> its assistant turn is omitted, but its authored user
    # turn ("turn two") stays visible/unanswered, adjacent to "turn three".
    assert _sent_payload(self)["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "turn two"},
        {"role": "user", "content": "turn three"},
    ]


@pytest.mark.asyncio
async def test_merge_consecutive_user_off_by_default(tmp_path: Path) -> None:
    """A FAILED producer leaves its user turn unanswered, adjacent to the next
    user turn; default policy sends them as-is."""
    client, ordinals = await _built_client(tmp_path, MERGE_CHAIN)
    self = _worker(client)
    self._graph_dynamic_pool.put(
        INSTANCE, "profiling", ordinals["a"], GraphPoolSentinel.FAILED
    )
    _respond_with(self, "unused")

    await self._process_graph_credit(_ctx(ordinals["b"]), "x-req-2", None)

    # init "first question", a's delta "warm up" (a's reply omitted, FAILED),
    # b's delta "second question" -> three consecutive user turns, as-is.
    assert _sent_payload(self)["messages"] == [
        {"role": "user", "content": "first question"},
        {"role": "user", "content": "warm up"},
        {"role": "user", "content": "second question"},
    ]


@pytest.mark.asyncio
async def test_merge_consecutive_user_when_enabled(tmp_path: Path, monkeypatch) -> None:
    from aiperf.common.environment import Environment

    monkeypatch.setattr(
        Environment.GRAPH, "MERGE_CONSECUTIVE_USER", True, raising=False
    )
    client, ordinals = await _built_client(tmp_path, MERGE_CHAIN)
    self = _worker(client)
    self._graph_dynamic_pool.put(
        INSTANCE, "profiling", ordinals["a"], GraphPoolSentinel.FAILED
    )
    _respond_with(self, "unused")

    await self._process_graph_credit(_ctx(ordinals["b"]), "x-req-2", None)

    assert _sent_payload(self)["messages"] == [
        {"role": "user", "content": "first question\nwarm up\nsecond question"},
    ]


@pytest.mark.asyncio
async def test_merge_leaves_assistant_separated_users_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    """With the knob on, an assistant-separated user pair is NOT merged. Uses a
    system-seeded chain so the only user runs are genuine (no init/delta user
    boundary), and all replies succeed -> clean alternation stays intact."""
    from aiperf.common.environment import Environment

    monkeypatch.setattr(
        Environment.GRAPH, "MERGE_CONSECUTIVE_USER", True, raising=False
    )
    client, ordinals = await _built_client(tmp_path, ACCUMULATE_CHAIN)
    self = _worker(client)
    pool = self._graph_dynamic_pool
    pool.put(
        INSTANCE, "profiling", ordinals["a"], GraphCapturedReply(text="answer one")
    )
    pool.put(
        INSTANCE, "profiling", ordinals["b"], GraphCapturedReply(text="answer two")
    )
    _respond_with(self, "unused")

    await self._process_graph_credit(_ctx(ordinals["c"]), "x-req-3", None)

    # Every user turn is separated by an assistant reply -> nothing merges.
    assert _sent_payload(self)["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "turn two"},
        {"role": "assistant", "content": "answer two"},
        {"role": "user", "content": "turn three"},
    ]


@pytest.mark.asyncio
async def test_missing_pool_value_errors_loudly_without_dispatch(
    tmp_path: Path,
) -> None:
    client, ordinals = await _built_client(tmp_path, PLANNER_REVIEWER)
    self = _worker(client)
    _respond_with(self, "unused")

    ctx = _ctx(ordinals["review"])
    await self._process_graph_credit(ctx, "x-req-2", None)

    assert ctx.error is not None
    assert ctx.error.startswith(f"aiperf.graph.pool_missing: {INSTANCE}/")
    self.inference_client.send_request.assert_not_awaited()
    # WK2: the pre-dispatch failure still emits a synthetic error record so the
    # RecordsManager completion barrier stays in lockstep with the credit side.
    self._send_graph_error_record.assert_awaited_once()


@pytest.mark.asyncio
async def test_pool_missing_error_is_a_trace_stop() -> None:
    """The adapter sniffs the prefix; the executor never contains it (§6.3)."""
    import asyncio

    from aiperf.dataset.graph.models import (
        ChannelRequirement,
        ChannelSpec,
        GraphRecord,
        LlmNode,
        ParsedGraph,
        StaticEdge,
        TraceRecord,
    )
    from aiperf.dataset.graph.segment_ir.pool import SegmentPool
    from aiperf.graph.credit_dispatch_adapter import GraphStickinessError
    from aiperf.graph.executor import TraceExecutor

    class _PoolMissingIssuer:
        async def dispatch(self, node, request, ctx, first_token_cb=None):
            if request.node_id == "b":
                raise GraphStickinessError("aiperf.graph.pool_missing: t1#0/0")
            return ""

    graph = GraphRecord(
        state={"a_out": ChannelSpec(), "b_out": ChannelSpec()},
        nodes={
            "a": LlmNode(prompt=[{"role": "user", "content": "q"}], output="a_out"),
            "b": LlmNode(
                prompt=[{"role": "user", "content": "q"}],
                output="b_out",
                inputs=[ChannelRequirement(channel="a_out", count=1)],
            ),
        },
        edges=[
            StaticEdge(source="START", target="a"),
            StaticEdge(source="a", target="b"),
            StaticEdge(source="b", target="END"),
        ],
    )
    parsed = ParsedGraph(
        graph=graph, traces=[TraceRecord(id="t1")], segment_pool=SegmentPool()
    )
    executor = TraceExecutor(parsed, credit_issuer=_PoolMissingIssuer())
    with pytest.raises(ExceptionGroup) as excinfo:
        async with asyncio.TaskGroup():
            await executor.run(parsed.traces[0])
    # The trace-stop must carry the stickiness error itself -- proof the
    # executor propagated (never contained) the pool-missing failure.
    assert excinfo.group_contains(GraphStickinessError)
