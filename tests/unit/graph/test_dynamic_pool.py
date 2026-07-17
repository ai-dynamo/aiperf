# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GraphDynamicPool lifecycle: capture values, deferred eviction, LRU backstop."""

from aiperf.graph.dynamic_pool import (
    GraphCapturedReply,
    GraphDynamicPool,
    GraphPoolSentinel,
)


def _reply(text: str) -> GraphCapturedReply:
    return GraphCapturedReply(text=text)


def test_put_get_roundtrip_and_sentinels() -> None:
    pool = GraphDynamicPool(max_bytes=1024)
    pool.put("t-1#0", "profiling", 0, _reply("hello"))
    pool.put("t-1#0", "profiling", 1, GraphPoolSentinel.FAILED)
    pool.put("t-1#0", "profiling", 2, GraphPoolSentinel.EMPTY)

    assert pool.get("t-1#0", "profiling", 0) == _reply("hello")
    assert pool.get("t-1#0", "profiling", 1) is GraphPoolSentinel.FAILED
    assert pool.get("t-1#0", "profiling", 2) is GraphPoolSentinel.EMPTY
    assert pool.get("t-1#0", "profiling", 9) is None
    assert pool.get("t-9#0", "profiling", 0) is None


def test_phase_variants_are_distinct_entries() -> None:
    pool = GraphDynamicPool(max_bytes=1024)
    pool.put("t-1#0", "warmup", 0, _reply("w"))
    pool.put("t-1#0", "profiling", 0, _reply("p"))
    pool.trace_end("t-1#0", "warmup")

    assert pool.get("t-1#0", "warmup", 0) is None
    assert pool.get("t-1#0", "profiling", 0) == _reply("p")


def test_trace_end_evicts_when_idle() -> None:
    pool = GraphDynamicPool(max_bytes=1024)
    pool.put("t-1#0", "profiling", 0, _reply("hello"))
    pool.trace_end("t-1#0", "profiling")

    assert pool.get("t-1#0", "profiling", 0) is None
    assert pool.total_bytes == 0


def test_trace_end_defers_while_inflight() -> None:
    """A cancelled credit's FAILED write must land in a live entry (§5.5)."""
    pool = GraphDynamicPool(max_bytes=1024)
    pool.credit_started("t-1#0", "profiling")
    pool.put("t-1#0", "profiling", 0, _reply("first"))
    pool.trace_end("t-1#0", "profiling")

    assert pool.get("t-1#0", "profiling", 0) == _reply("first")
    pool.put("t-1#0", "profiling", 1, GraphPoolSentinel.FAILED)
    pool.credit_finished("t-1#0", "profiling")

    assert pool.get("t-1#0", "profiling", 0) is None
    assert pool.get("t-1#0", "profiling", 1) is None
    assert pool.total_bytes == 0


def test_trace_end_idempotent_and_unknown_noop() -> None:
    pool = GraphDynamicPool(max_bytes=1024)
    pool.put("t-1#0", "profiling", 0, _reply("x"))
    pool.trace_end("t-1#0", "profiling")
    pool.trace_end("t-1#0", "profiling")
    pool.trace_end("never", "profiling")
    assert pool.total_bytes == 0


def test_overwrite_same_ordinal_accounts_bytes_once() -> None:
    pool = GraphDynamicPool(max_bytes=1024)
    pool.put("t-1#0", "profiling", 0, _reply("aaaa"))
    pool.put("t-1#0", "profiling", 0, _reply("bb"))
    assert pool.total_bytes == 2


def test_lru_backstop_evicts_least_recently_used_trace() -> None:
    pool = GraphDynamicPool(max_bytes=10)
    pool.put("t-1#0", "profiling", 0, _reply("aaaa"))  # 4 bytes
    pool.put("t-2#0", "profiling", 0, _reply("bbbb"))  # 4 bytes
    pool.get("t-1#0", "profiling", 0)  # touch t-1: t-2 becomes LRU
    pool.put("t-3#0", "profiling", 0, _reply("cccc"))  # 12 > 10: evict t-2

    assert pool.get("t-2#0", "profiling", 0) is None
    assert pool.get("t-1#0", "profiling", 0) == _reply("aaaa")
    assert pool.get("t-3#0", "profiling", 0) == _reply("cccc")
    assert pool.total_bytes == 8


def test_lru_evicts_writer_last_when_it_alone_exceeds_cap() -> None:
    pool = GraphDynamicPool(max_bytes=4)
    pool.put("t-1#0", "profiling", 0, _reply("way more than four bytes"))
    assert pool.get("t-1#0", "profiling", 0) is None
    assert pool.total_bytes == 0


# ---------------------------------------------------------------------------
# Byte accounting for structured values
# ---------------------------------------------------------------------------


def test_text_only_value_costs_text_bytes() -> None:
    pool = GraphDynamicPool(max_bytes=1024)
    pool.put("t-1#0", "profiling", 0, _reply("abcd"))
    assert pool.total_bytes == 4


def test_message_json_value_costs_text_plus_json_bytes() -> None:
    pool = GraphDynamicPool(max_bytes=1024)
    message_json = '{"role":"assistant","content":null,"tool_calls":[]}'
    pool.put(
        "t-1#0",
        "profiling",
        0,
        GraphCapturedReply(text="ab", message_json=message_json),
    )
    assert pool.total_bytes == 2 + len(message_json.encode())


def test_sentinel_value_costs_flat_bytes() -> None:
    pool = GraphDynamicPool(max_bytes=1024)
    pool.put("t-1#0", "profiling", 0, GraphPoolSentinel.FAILED)
    assert pool.total_bytes == 8  # _SENTINEL_COST_BYTES


def test_message_json_bytes_count_toward_lru_cap() -> None:
    pool = GraphDynamicPool(max_bytes=25)
    pool.put("t-1#0", "profiling", 0, _reply("aaaa"))  # 4 bytes
    # 0 text bytes + 22 json bytes: 26 > 25 evicts the LRU entry t-1.
    pool.put(
        "t-2#0",
        "profiling",
        0,
        GraphCapturedReply(text="", message_json='{"tool_calls":[1,2,3]}'),
    )

    assert pool.get("t-1#0", "profiling", 0) is None
    assert pool.get("t-2#0", "profiling", 0) is not None
    assert pool.total_bytes == 22


# ---------------------------------------------------------------------------
# R5 -- no resurrection after trace_end (zero-byte immortal entries)
# ---------------------------------------------------------------------------


def test_credit_started_after_trace_end_leaves_no_entry() -> None:
    """A straggler credit_started behind GraphTraceEnd must not resurrect the
    (trace_id, phase_variant) entry -- resurrected entries have no end message
    left to evict them and accumulate forever in recycle-heavy runs."""
    pool = GraphDynamicPool(max_bytes=1024)
    pool.put("t-1#0", "profiling", 0, _reply("hello"))
    pool.trace_end("t-1#0", "profiling")
    assert pool.entry_count == 0

    pool.credit_started("t-1#0", "profiling")

    assert pool.entry_count == 0
    assert pool.total_bytes == 0


def test_put_after_trace_end_is_dropped() -> None:
    """A straggler capture behind GraphTraceEnd is dropped, not resurrected."""
    pool = GraphDynamicPool(max_bytes=1024)
    pool.put("t-1#0", "profiling", 0, _reply("hello"))
    pool.trace_end("t-1#0", "profiling")

    pool.put("t-1#0", "profiling", 1, _reply("late capture"))

    assert pool.entry_count == 0
    assert pool.total_bytes == 0
    assert pool.get("t-1#0", "profiling", 1) is None


def test_deferred_eviction_also_tombstones() -> None:
    """The deferred (end_pending) eviction path blocks resurrection too."""
    pool = GraphDynamicPool(max_bytes=1024)
    pool.credit_started("t-1#0", "profiling")
    pool.trace_end("t-1#0", "profiling")  # deferred: credit in flight
    pool.credit_finished("t-1#0", "profiling")  # drains -> evict + tombstone
    assert pool.entry_count == 0

    pool.credit_started("t-1#0", "profiling")
    pool.put("t-1#0", "profiling", 0, _reply("zombie"))

    assert pool.entry_count == 0
    assert pool.total_bytes == 0


def test_trace_end_for_unknown_key_blocks_out_of_order_start() -> None:
    """End arriving BEFORE any event for the key still blocks resurrection."""
    pool = GraphDynamicPool(max_bytes=1024)
    pool.trace_end("t-9#0", "profiling")

    pool.credit_started("t-9#0", "profiling")

    assert pool.entry_count == 0


def test_tombstone_fifo_is_bounded() -> None:
    """Tombstones age out FIFO at the capacity bound (no unbounded growth)."""
    pool = GraphDynamicPool(max_bytes=1024, tombstone_capacity=2)
    for i in range(4):
        pool.put(f"t-{i}", "profiling", 0, _reply("x"))
        pool.trace_end(f"t-{i}", "profiling")

    assert len(pool._tombstones) == 2
    # The oldest key aged out: a straggler for it CAN recreate an entry (the
    # accepted bounded-memory trade); recent keys stay blocked.
    pool.credit_started("t-0", "profiling")
    assert pool.entry_count == 1
    pool.credit_started("t-3", "profiling")
    assert pool.entry_count == 1


def test_distinct_phase_variant_not_blocked_by_other_variants_end() -> None:
    """Ending the warmup variant must not tombstone the profiling variant."""
    pool = GraphDynamicPool(max_bytes=1024)
    pool.put("t-1#0", "warmup", 0, _reply("w"))
    pool.trace_end("t-1#0", "warmup")

    pool.credit_started("t-1#0", "profiling")
    pool.put("t-1#0", "profiling", 0, _reply("p"))

    assert pool.get("t-1#0", "profiling", 0) == _reply("p")
    assert pool.entry_count == 1
