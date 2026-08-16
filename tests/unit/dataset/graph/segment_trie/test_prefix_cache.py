# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Theoretical prefix-cache accounting over a shared trie, and its stamping onto LlmNodes."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ToolNode,
    resolve_trace_graph,
)
from aiperf.dataset.graph.segment_trie.prefix_cache import (
    CausalRequest,
    compute_causal_prefix_hits,
    compute_shared_prefix_cache_counts,
    extract_prefix_cache_by_node,
    stamp_theoretical_prefix_cache,
)
from aiperf.dataset.graph.segment_trie.trie_content import TrieNode
from aiperf.dataset.graph.store_build import GraphStoreBuilder
from tests.unit.dataset.graph.segment_trie.conftest import (
    DYNAMO_NESTED_FIXTURE,
    trie_node,
)


def hash_node(
    node_id: str,
    hash_ids: list[int],
    t: float,
    order: int,
    api_time: float = 1.0,
) -> TrieNode:
    """A TrieNode whose input length follows from its block hashes (4 tokens per block).

    `api_time` defaults to 1.0, so a node's blocks become cache-available one
    time unit after its recorded start.
    """
    return trie_node(
        node_id,
        hash_ids=hash_ids,
        input_length=len(hash_ids) * 4,
        output_length=8,
        t=t,
        order=order,
        api_time=api_time,
    )


def llm(node_id: str) -> LlmNode:
    """A bare LlmNode ready to be stamped."""
    return LlmNode(prompt=[], output=f"{node_id}_out", metadata={})


class TestComputeSharedPrefixCacheCounts:
    """Counts are (hit_blocks, total_blocks) per node, scored against a cache
    seeded in recorded ARRIVAL order under causal availability: a block is a hit
    only once the request that produced it has finished."""

    @pytest.mark.parametrize(
        ("nodes", "expected"),
        [
            param(
                [("a", [1, 2, 3], 0.0, 0)],
                {"a": (0, 3)},
                id="cold_cache_first_request_zero_hits",
            ),
            param(
                [("a", [1, 2], 0.0, 0), ("b", [1, 2, 3], 1.0, 1)],
                {"a": (0, 2), "b": (2, 3)},
                id="full_prefix_reuse_hits_leading_run",
            ),
            param(
                # 9 is unseen: the leading run stops there even though 2 was cached.
                [("a", [1, 2], 0.0, 0), ("b", [1, 9, 2], 1.0, 1)],
                {"a": (0, 2), "b": (1, 3)},
                id="leading_run_stops_at_first_miss",
            ),
            param(
                # "late" is first in the list but has the LATER recorded t, so
                # "early" seeds the cache and "late" scores hits off it.
                [("late", [1, 2], 5.0, 0), ("early", [1, 2], 1.0, 1)],
                {"early": (0, 2), "late": (2, 2)},
                id="global_time_order_not_list_order",
            ),
            param(
                # Same arrival: "first" is still in flight (it ends at 2.0) when
                # "second" arrives at 1.0, so its block is not yet available and
                # the flatten-order tiebreak cannot manufacture a hit.
                [("second", [1], 1.0, 1), ("first", [1], 1.0, 0)],
                {"first": (0, 1), "second": (0, 1)},
                id="equal_t_concurrent_producer_is_not_yet_available",
            ),
            param(
                # Same nodes, but "first" completes instantly (api_time 0), so
                # its block IS available to the co-arriving "second".
                [("second", [1], 1.0, 1, 1.0), ("first", [1], 1.0, 0, 0.0)],
                {"first": (0, 1), "second": (1, 1)},
                id="equal_t_instant_producer_is_available",
            ),
            param(
                # "producer" arrives first but runs long (ends at 100.0);
                # "consumer" arrives at 2.0 while it is still in flight, so the
                # block it would have hit did not exist yet.
                [("producer", [1], 0.0, 0, 100.0), ("consumer", [1], 2.0, 1)],
                {"producer": (0, 1), "consumer": (0, 1)},
                id="in_flight_producer_denies_hit",
            ),
            param(
                # Two producers of block 1: the slow one arrives first, but
                # availability is the MINIMUM end over producers, so the fast
                # second producer makes the block available to "consumer".
                [
                    ("slow", [1], 0.0, 0, 100.0),
                    ("fast", [1], 1.0, 1, 0.5),
                    ("consumer", [1], 2.0, 2),
                ],
                {"slow": (0, 1), "fast": (0, 1), "consumer": (1, 1)},
                id="min_end_over_producers_wins",
            ),
            param(
                # b misses on 9, but its trailing 7 still enters the cache for c.
                [("a", [1], 0.0, 0), ("b", [9, 7], 1.0, 1), ("c", [9, 7], 2.0, 2)],
                {"a": (0, 1), "b": (0, 2), "c": (2, 2)},
                id="all_blocks_enter_cache_not_just_hits",
            ),
            param([("a", [], 0.0, 0)], {"a": (0, 0)}, id="empty_hash_ids_zero_total"),
        ],
    )  # fmt: skip
    def test_counts(
        self,
        nodes: list[
            tuple[str, list[int], float, int] | tuple[str, list[int], float, int, float]
        ],
        expected: dict[str, tuple[int, int]],
    ) -> None:
        counts = compute_shared_prefix_cache_counts(
            [hash_node(*n) for n in nodes], block_size=4
        )
        assert counts == expected


class TestComputeCausalPrefixHits:
    """The shared accounting every loader path routes through."""

    def test_result_is_aligned_to_input_order_not_arrival_order(self) -> None:
        """Hits come back positionally, so callers can zip against their own list."""
        hits = compute_causal_prefix_hits(
            [
                CausalRequest(hash_ids=[1, 2], start=10.0, end=11.0),
                CausalRequest(hash_ids=[1, 2], start=0.0, end=1.0),
            ]
        )
        # Index 1 arrived FIRST and seeds the cache; index 0 scores off it.
        assert hits == [2, 0]

    def test_zero_width_intervals_degrade_to_order_only_bound(self) -> None:
        """With end == start, a producer is available to anything strictly later."""
        hits = compute_causal_prefix_hits(
            [
                CausalRequest(hash_ids=[1, 2], start=0.0, end=0.0),
                CausalRequest(hash_ids=[1, 2, 3], start=1.0, end=1.0),
            ]
        )
        assert hits == [0, 2]

    def test_producer_finishing_exactly_at_consumer_start_is_available(self) -> None:
        """Availability is inclusive: end <= start counts as a hit."""
        hits = compute_causal_prefix_hits(
            [
                CausalRequest(hash_ids=[1], start=0.0, end=5.0),
                CausalRequest(hash_ids=[1], start=5.0, end=6.0),
            ]
        )
        assert hits == [0, 1]

    def test_hit_never_exceeds_the_causal_bound(self) -> None:
        """A long-running producer cannot be hit by anything that overlaps it."""
        hits = compute_causal_prefix_hits(
            [
                CausalRequest(hash_ids=[1], start=0.0, end=100.0),
                CausalRequest(hash_ids=[1], start=99.9, end=100.0),
            ]
        )
        assert hits == [0, 0]

    def test_empty_input_returns_empty(self) -> None:
        assert compute_causal_prefix_hits([]) == []


class TestStampTheoreticalPrefixCache:
    """Stamping writes the counts onto the native LlmNode fields without disturbing existing metadata."""

    def test_stamps_native_fields(self) -> None:
        llm_nodes = {"a": llm("a"), "b": llm("b")}
        stamp_theoretical_prefix_cache(
            llm_nodes,
            [hash_node("a", [1, 2], 0.0, 0), hash_node("b", [1, 2, 3], 1.0, 1)],
            block_size=4,
        )
        assert llm_nodes["a"].theoretical_prefix_cache_hit_blocks == 0
        assert llm_nodes["a"].theoretical_prefix_cache_total_blocks == 2
        assert llm_nodes["b"].theoretical_prefix_cache_hit_blocks == 2
        assert llm_nodes["b"].theoretical_prefix_cache_total_blocks == 3

    def test_zero_hash_node_left_unstamped(self) -> None:
        """A node with no block hashes has nothing to report, so both fields stay None rather than being stamped 0."""
        llm_nodes = {"a": llm("a")}
        stamp_theoretical_prefix_cache(
            llm_nodes, [hash_node("a", [], 0.0, 0)], block_size=4
        )
        assert llm_nodes["a"].theoretical_prefix_cache_hit_blocks is None
        assert llm_nodes["a"].theoretical_prefix_cache_total_blocks is None

    def test_preserves_existing_metadata_keys(self) -> None:
        """Stamping leaves the node's existing trie/dispatch metadata intact."""
        llm_node_ = LlmNode(
            prompt=[],
            output="a_out",
            metadata={"trie": {"prompt_segment_ids": ["s1"]}, "dispatch": {"k": 1}},
        )
        llm_nodes = {"a": llm_node_}
        stamp_theoretical_prefix_cache(
            llm_nodes, [hash_node("a", [1], 0.0, 0)], block_size=4
        )
        stamped = llm_nodes["a"]
        assert stamped.metadata["trie"] == {"prompt_segment_ids": ["s1"]}
        assert stamped.metadata["dispatch"]["k"] == 1
        assert stamped.theoretical_prefix_cache_total_blocks == 1

    def test_extract_round_trips_stamped_graph(self) -> None:
        """extract_prefix_cache_by_node reads back exactly what stamping wrote."""
        llm_nodes = {"a": llm("a"), "b": llm("b")}
        stamp_theoretical_prefix_cache(
            llm_nodes,
            [hash_node("a", [1], 0.0, 0), hash_node("b", [1, 2], 1.0, 1)],
            block_size=4,
        )
        graph = GraphRecord(nodes=dict(llm_nodes))
        assert extract_prefix_cache_by_node(graph) == {"a": [0, 1], "b": [1, 2]}


class TestAdapterIntegration:
    """The dynamo adapter stamps counts during parse, and the store builder exposes them per trace."""

    def test_dynamo_stamps_recorded_hash_counts(self, tmp_path: Path) -> None:
        """Records carrying replay metadata score against the RECORDED input_sequence_hashes: turn 2 extends turn 1, so its leading 2 blocks hit."""

        def rec(ts: int, hashes: list[int], input_tokens: int) -> dict:
            return {
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": ts,
                "event_source": "dynamo",
                "agent_context": {"session_id": "s1"},
                "request": {
                    "request_id": f"r{ts}",
                    "model": "m",
                    "input_tokens": input_tokens,
                    "output_tokens": 8,
                    "cached_tokens": 0,
                    "replay": {
                        "trace_block_size": 16,
                        "input_length": input_tokens,
                        "input_sequence_hashes": hashes,
                    },
                },
            }

        p = tmp_path / "dyn_kv.jsonl"
        p.write_bytes(
            b"\n".join(
                orjson.dumps(r)
                for r in [
                    rec(1000, [111, 222], 32),
                    rec(2000, [111, 222, 333, 444], 64),
                ]
            )
        )
        pg = from_dynamo_trace(p, content_root_seed=0, content_tokenizer="builtin")
        counts = extract_prefix_cache_by_node(resolve_trace_graph(pg, pg.traces[0]))
        by_turn = {node_id.rsplit(":", 1)[1]: v for node_id, v in counts.items()}
        assert by_turn == {"0": [0, 2], "1": [2, 4]}

    def test_dynamo_trie_build_stamps_hash_nodes(self) -> None:
        """Every trie node from the nested fixture carries counts satisfying 0 <= hits <= total, with a nonzero total."""
        pg = from_dynamo_trace(DYNAMO_NESTED_FIXTURE, content_root_seed=0)
        counts_by_trace = {
            t.id: extract_prefix_cache_by_node(resolve_trace_graph(pg, t))
            for t in pg.traces
        }
        stamped = [c for c in counts_by_trace.values() if c]
        assert stamped, "dynamo trie nodes must carry prefix-cache counts"
        for counts in stamped:
            assert all(t > 0 for _, t in counts.values())
            assert all(0 <= h <= t for h, t in counts.values())

    def test_store_builder_builds_per_trace_map(self) -> None:
        """The store builder's per-trace map is non-empty and keyed only by real trace ids."""
        pg = from_dynamo_trace(DYNAMO_NESTED_FIXTURE, content_root_seed=0)
        by_trace = GraphStoreBuilder._build_graph_prefix_cache_by_trace(pg)
        assert set(by_trace) <= {t.id for t in pg.traces}
        assert by_trace, "per-trace prefix-cache map must not be empty"
        for counts in by_trace.values():
            assert all(0 <= h <= t for h, t in counts.values())

    def test_tool_nodes_are_skipped_not_read(self) -> None:
        """A ToolNode has no prefix-cache fields at all; extraction must skip it rather than AttributeError."""
        graph = GraphRecord(
            nodes={
                "n0": LlmNode(
                    prompt=["hi"],
                    output="n0_out",
                    theoretical_prefix_cache_hit_blocks=1,
                    theoretical_prefix_cache_total_blocks=2,
                ),
                "t0": ToolNode(commands=["true"], output="t0_out"),
            }
        )

        assert extract_prefix_cache_by_node(graph) == {"n0": [1, 2]}
