# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GraphDynamicPool lifecycle: capture values, byte accounting, deferred eviction, LRU backstop, and post-trace_end tombstones."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.graph.dynamic_pool import (
    GraphCapturedReply,
    GraphDynamicPool,
    GraphPoolSentinel,
    PoolValue,
)

TRACE = "t-1#0"


def _reply(text: str) -> GraphCapturedReply:
    return GraphCapturedReply(text=text)


@pytest.fixture
def pool() -> GraphDynamicPool:
    """A pool whose cap is far above any single test payload (no LRU pressure)."""
    return GraphDynamicPool(max_bytes=1024)


def test_put_get_roundtrip_and_sentinels(pool: GraphDynamicPool) -> None:
    """Captures and both sentinels round-trip; unknown ordinal/trace reads as None."""
    pool.put(TRACE, 0, _reply("hello"))
    pool.put(TRACE, 1, GraphPoolSentinel.FAILED)
    pool.put(TRACE, 2, GraphPoolSentinel.EMPTY)

    assert pool.get(TRACE, 0) == _reply("hello")
    assert pool.get(TRACE, 1) is GraphPoolSentinel.FAILED
    assert pool.get(TRACE, 2) is GraphPoolSentinel.EMPTY
    assert pool.get(TRACE, 9) is None
    assert pool.get("t-9#0", 0) is None


def test_distinct_trace_instances_are_independent_entries(
    pool: GraphDynamicPool,
) -> None:
    """Two trace INSTANCES are independent entries.

    Instance ids are ``{template}::{uuid4}``, unique across lanes, recycles and
    phases, so the instance id alone separates entries -- there is no second key
    dimension to test.
    """
    other = f"{TRACE}-other"
    pool.put(TRACE, 0, _reply("a"))
    pool.put(other, 0, _reply("b"))
    pool.trace_end(TRACE)

    assert pool.get(TRACE, 0) is None
    assert pool.get(other, 0) == _reply("b")


def test_trace_end_evicts_when_idle(pool: GraphDynamicPool) -> None:
    """With no credit in flight, trace_end evicts the entry and frees its bytes."""
    pool.put(TRACE, 0, _reply("hello"))
    pool.trace_end(TRACE)

    assert pool.get(TRACE, 0) is None
    assert pool.total_bytes == 0


def test_trace_end_defers_while_inflight(pool: GraphDynamicPool) -> None:
    """trace_end with a credit in flight defers eviction until credit_finished (S5.5)."""
    pool.credit_started(TRACE)
    pool.put(TRACE, 0, _reply("first"))
    pool.trace_end(TRACE)

    # A cancelled credit's FAILED write must still land in a live entry.
    assert pool.get(TRACE, 0) == _reply("first")
    pool.put(TRACE, 1, GraphPoolSentinel.FAILED)
    pool.credit_finished(TRACE)

    assert pool.get(TRACE, 0) is None
    assert pool.get(TRACE, 1) is None
    assert pool.total_bytes == 0


def test_trace_end_idempotent_and_unknown_noop(pool: GraphDynamicPool) -> None:
    """Repeated and unknown-key trace_end calls are harmless no-ops."""
    pool.put(TRACE, 0, _reply("x"))
    pool.trace_end(TRACE)
    pool.trace_end(TRACE)
    pool.trace_end("never")
    assert pool.total_bytes == 0


def test_overwrite_same_ordinal_accounts_bytes_once(pool: GraphDynamicPool) -> None:
    """Re-putting an ordinal replaces its byte cost instead of double-counting it."""
    pool.put(TRACE, 0, _reply("aaaa"))
    pool.put(TRACE, 0, _reply("bb"))
    assert pool.total_bytes == 2


def test_lru_backstop_evicts_least_recently_used_trace() -> None:
    """Exceeding max_bytes evicts the least recently touched trace, not the writer."""
    pool = GraphDynamicPool(max_bytes=10)
    pool.put(TRACE, 0, _reply("aaaa"))  # 4 bytes
    pool.put("t-2#0", 0, _reply("bbbb"))  # 4 bytes
    pool.get(TRACE, 0)  # touch t-1: t-2 becomes LRU
    pool.put("t-3#0", 0, _reply("cccc"))  # 12 > 10: evict t-2

    assert pool.get("t-2#0", 0) is None
    assert pool.get(TRACE, 0) == _reply("aaaa")
    assert pool.get("t-3#0", 0) == _reply("cccc")
    assert pool.total_bytes == 8


def test_lru_evicts_writer_last_when_it_alone_exceeds_cap() -> None:
    """A single value larger than the whole cap is evicted rather than retained."""
    pool = GraphDynamicPool(max_bytes=4)
    pool.put(TRACE, 0, _reply("way more than four bytes"))
    assert pool.get(TRACE, 0) is None
    assert pool.total_bytes == 0


_MESSAGE_JSON = '{"role":"assistant","content":null,"tool_calls":[]}'


@pytest.mark.parametrize(
    "value,expected_bytes",
    [
        param(_reply("abcd"), 4, id="text_only_costs_text_bytes"),
        param(
            GraphCapturedReply(text="ab", message_json=_MESSAGE_JSON),
            2 + len(_MESSAGE_JSON.encode()),
            id="structured_costs_text_plus_json_bytes",
        ),
        param(GraphPoolSentinel.FAILED, 8, id="sentinel_costs_flat_cost_bytes"),
    ],
)  # fmt: skip
def test_value_byte_accounting(
    pool: GraphDynamicPool, value: PoolValue, expected_bytes: int
) -> None:
    """Each pool value kind charges its documented byte cost against the cap."""
    pool.put(TRACE, 0, value)
    assert pool.total_bytes == expected_bytes


def test_message_json_bytes_count_toward_lru_cap() -> None:
    """A structured capture's JSON bytes count toward the cap and can trigger LRU eviction."""
    pool = GraphDynamicPool(max_bytes=25)
    pool.put(TRACE, 0, _reply("aaaa"))  # 4 bytes
    # 0 text bytes + 22 json bytes: 26 > 25 evicts the LRU entry t-1.
    pool.put(
        "t-2#0",
        0,
        GraphCapturedReply(text="", message_json='{"tool_calls":[1,2,3]}'),
    )

    assert pool.get(TRACE, 0) is None
    assert pool.get("t-2#0", 0) is not None
    assert pool.total_bytes == 22


def test_credit_started_after_trace_end_leaves_no_entry(
    pool: GraphDynamicPool,
) -> None:
    """A straggler credit_started behind trace_end must not resurrect the entry (R5)."""
    pool.put(TRACE, 0, _reply("hello"))
    pool.trace_end(TRACE)
    assert len(pool._entries) == 0

    # Resurrected entries have no end message left to evict them and would
    # accumulate forever in recycle-heavy runs.
    pool.credit_started(TRACE)

    assert len(pool._entries) == 0
    assert pool.total_bytes == 0


def test_put_after_trace_end_is_dropped(pool: GraphDynamicPool) -> None:
    """A straggler capture behind trace_end is dropped, not resurrected."""
    pool.put(TRACE, 0, _reply("hello"))
    pool.trace_end(TRACE)

    pool.put(TRACE, 1, _reply("late capture"))

    assert len(pool._entries) == 0
    assert pool.total_bytes == 0
    assert pool.get(TRACE, 1) is None


def test_deferred_eviction_also_tombstones(pool: GraphDynamicPool) -> None:
    """The deferred (end_pending) eviction path tombstones the key too."""
    pool.credit_started(TRACE)
    pool.trace_end(TRACE)  # deferred: credit in flight
    pool.credit_finished(TRACE)  # drains -> evict + tombstone
    assert len(pool._entries) == 0

    pool.credit_started(TRACE)
    pool.put(TRACE, 0, _reply("zombie"))

    assert len(pool._entries) == 0
    assert pool.total_bytes == 0


def test_trace_end_for_unknown_key_blocks_out_of_order_start(
    pool: GraphDynamicPool,
) -> None:
    """trace_end arriving before any event for a key still blocks later resurrection."""
    pool.trace_end("t-9#0")

    pool.credit_started("t-9#0")

    assert len(pool._entries) == 0


def test_tombstone_fifo_is_bounded() -> None:
    """Tombstones age out FIFO at tombstone_capacity, so the set cannot grow unbounded."""
    pool = GraphDynamicPool(max_bytes=1024, tombstone_capacity=2)
    for i in range(4):
        pool.put(f"t-{i}", 0, _reply("x"))
        pool.trace_end(f"t-{i}")

    assert len(pool._tombstones) == 2
    # The oldest key aged out: a straggler for it CAN recreate an entry (the
    # accepted bounded-memory trade); recent keys stay blocked.
    pool.credit_started("t-0")
    assert len(pool._entries) == 1
    pool.credit_started("t-3")
    assert len(pool._entries) == 1


def test_one_instances_end_does_not_tombstone_another(
    pool: GraphDynamicPool,
) -> None:
    """Ending one instance must not tombstone a different one."""
    other = f"{TRACE}-other"
    pool.put(TRACE, 0, _reply("a"))
    pool.trace_end(TRACE)

    pool.credit_started(other)
    pool.put(other, 0, _reply("b"))

    assert pool.get(other, 0) == _reply("b")
    assert len(pool._entries) == 1
