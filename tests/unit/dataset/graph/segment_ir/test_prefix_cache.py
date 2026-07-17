# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared trie theoretical prefix-cache stamping.

Locks the shared-pre-pass semantics onto the segment-trie IR: one shared per-trace
seen-set consumed in recorded global time order, leading-run hit counting
(stop at first miss), full hash-id totals, and every request's blocks entering
the infinite cache. Plus the stamping contract every consumer relies on: the
native ``theoretical_prefix_cache_hit_blocks`` / ``_total_blocks`` fields on
each hash-carrying ``LlmNode``, surviving the graph_meta sidecar strip.
"""

from pathlib import Path

from aiperf.dataset.graph.models import LlmNode
from aiperf.dataset.graph.segment_ir.prefix_cache import (
    compute_shared_prefix_cache_counts,
    extract_prefix_cache_by_node,
    stamp_theoretical_prefix_cache,
)
from aiperf.dataset.graph.segment_ir.trie_content import TrieNode, TrieRequest

WEKA_FIXTURE = Path(__file__).parents[3] / "graph" / "fixtures" / "weka_min.json"


def _node(node_id: str, hash_ids: list[int], t: float, order: int) -> TrieNode:
    return TrieNode(
        node_id=node_id,
        request=TrieRequest(
            hash_ids=hash_ids,
            input_length=len(hash_ids) * 4,
            output_length=8,
            t=t,
            api_time=1.0,
        ),
        order=order,
    )


def _llm(node_id: str) -> LlmNode:
    return LlmNode(prompt=[], output=f"{node_id}_out", metadata={})


class TestComputeSharedPrefixCacheCounts:
    def test_cold_cache_first_request_zero_hits(self):
        counts = compute_shared_prefix_cache_counts([_node("a", [1, 2, 3], 0.0, 0)])
        assert counts == {"a": (0, 3)}

    def test_full_prefix_reuse_hits_leading_run(self):
        counts = compute_shared_prefix_cache_counts(
            [
                _node("a", [1, 2], 0.0, 0),
                _node("b", [1, 2, 3], 1.0, 1),
            ]
        )
        assert counts == {"a": (0, 2), "b": (2, 3)}

    def test_leading_run_stops_at_first_miss(self):
        # 9 is unseen: the run stops there even though 2 was cached.
        counts = compute_shared_prefix_cache_counts(
            [
                _node("a", [1, 2], 0.0, 0),
                _node("b", [1, 9, 2], 1.0, 1),
            ]
        )
        assert counts["b"] == (1, 3)

    def test_global_time_order_not_list_order(self):
        # "late" appears first in the list but has the LATER recorded t, so
        # "early" seeds the cache first and "late" scores hits off it.
        counts = compute_shared_prefix_cache_counts(
            [
                _node("late", [1, 2], 5.0, 0),
                _node("early", [1, 2], 1.0, 1),
            ]
        )
        assert counts == {"early": (0, 2), "late": (2, 2)}

    def test_equal_t_tiebreak_by_flatten_order(self):
        counts = compute_shared_prefix_cache_counts(
            [
                _node("second", [1], 1.0, 1),
                _node("first", [1], 1.0, 0),
            ]
        )
        assert counts == {"first": (0, 1), "second": (1, 1)}

    def test_all_blocks_enter_cache_not_just_hits(self):
        # b misses on 9, but its trailing 7 still enters the cache for c.
        counts = compute_shared_prefix_cache_counts(
            [
                _node("a", [1], 0.0, 0),
                _node("b", [9, 7], 1.0, 1),
                _node("c", [9, 7], 2.0, 2),
            ]
        )
        assert counts["c"] == (2, 2)

    def test_empty_hash_ids_yield_zero_total(self):
        counts = compute_shared_prefix_cache_counts([_node("a", [], 0.0, 0)])
        assert counts == {"a": (0, 0)}


class TestStampTheoreticalPrefixCache:
    def test_stamps_native_fields(self):
        llm_nodes = {"a": _llm("a"), "b": _llm("b")}
        stamp_theoretical_prefix_cache(
            llm_nodes,
            [_node("a", [1, 2], 0.0, 0), _node("b", [1, 2, 3], 1.0, 1)],
        )
        assert llm_nodes["a"].theoretical_prefix_cache_hit_blocks == 0
        assert llm_nodes["a"].theoretical_prefix_cache_total_blocks == 2
        assert llm_nodes["b"].theoretical_prefix_cache_hit_blocks == 2
        assert llm_nodes["b"].theoretical_prefix_cache_total_blocks == 3

    def test_zero_hash_node_left_unstamped(self):
        llm_nodes = {"a": _llm("a")}
        stamp_theoretical_prefix_cache(llm_nodes, [_node("a", [], 0.0, 0)])
        assert llm_nodes["a"].theoretical_prefix_cache_hit_blocks is None
        assert llm_nodes["a"].theoretical_prefix_cache_total_blocks is None

    def test_preserves_existing_metadata_keys(self):
        llm = LlmNode(
            prompt=[],
            output="a_out",
            metadata={"trie": {"prompt_segment_ids": ["s1"]}, "dispatch": {"k": 1}},
        )
        llm_nodes = {"a": llm}
        stamp_theoretical_prefix_cache(llm_nodes, [_node("a", [1], 0.0, 0)])
        stamped = llm_nodes["a"]
        assert stamped.metadata["trie"] == {"prompt_segment_ids": ["s1"]}
        assert stamped.metadata["dispatch"]["k"] == 1
        assert stamped.theoretical_prefix_cache_total_blocks == 1

    def test_extract_round_trips_stamped_graph(self):
        from aiperf.dataset.graph.models import GraphRecord

        llm_nodes = {"a": _llm("a"), "b": _llm("b")}
        stamp_theoretical_prefix_cache(
            llm_nodes,
            [_node("a", [1], 0.0, 0), _node("b", [1, 2], 1.0, 1)],
        )
        graph = GraphRecord(nodes=dict(llm_nodes))
        assert extract_prefix_cache_by_node(graph) == {"a": [0, 1], "b": [1, 2]}


class TestAdapterIntegration:
    def test_weka_trie_build_stamps_every_hash_node(self):
        from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
        from aiperf.dataset.graph.models import resolve_trace_graph

        pg = from_weka_trace(WEKA_FIXTURE, content_root_seed=0)
        trace = pg.traces[0]
        counts = extract_prefix_cache_by_node(resolve_trace_graph(pg, trace))
        # weka_min: three requests with growing hash-id prefixes; the exact
        # counts pin the shared-seen-set walk on the recorded timeline. Node ids
        # are ``{trace_id}:{k}`` (0-based turn) -- weka_min's trace is trace_03_n3.
        assert counts == {
            "trace_03_n3:0": [0, 2],
            "trace_03_n3:1": [2, 3],
            "trace_03_n3:2": [3, 4],
        }

    def test_weka_counts_survive_sidecar_strip(self):
        from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
        from aiperf.dataset.graph.graph_meta_sidecar import strip_replay_text
        from aiperf.dataset.graph.models import resolve_trace_graph

        pg = from_weka_trace(WEKA_FIXTURE, content_root_seed=0)
        stripped = strip_replay_text(pg)
        counts = extract_prefix_cache_by_node(
            resolve_trace_graph(stripped, stripped.traces[0])
        )
        assert counts == {
            "trace_03_n3:0": [0, 2],
            "trace_03_n3:1": [2, 3],
            "trace_03_n3:2": [3, 4],
        }

    def test_dynamo_stamps_recorded_hash_counts(self, tmp_path):
        # Records with replay metadata use the RECORDED input_sequence_hashes:
        # turn 2's hashes extend turn 1's, so its leading 2 blocks hit the
        # shared cache.
        import orjson

        from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
        from aiperf.dataset.graph.models import resolve_trace_graph

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
        pg = from_dynamo_trace(
            p,
            content_root_seed=0,
            content_tokenizer="builtin",
        )
        counts = extract_prefix_cache_by_node(resolve_trace_graph(pg, pg.traces[0]))
        by_turn = {node_id.rsplit(":", 1)[1]: v for node_id, v in counts.items()}
        assert by_turn == {"0": [0, 2], "1": [2, 4]}

    def test_dynamo_trie_build_stamps_hash_nodes(self):
        from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
        from aiperf.dataset.graph.models import resolve_trace_graph

        fixture = (
            Path(__file__).parents[1]
            / "adapters"
            / "fixtures"
            / "dynamo_nested"
            / "nested_2_level.jsonl.gz"
        )
        pg = from_dynamo_trace(fixture, content_root_seed=0)
        counts_by_trace = {
            t.id: extract_prefix_cache_by_node(resolve_trace_graph(pg, t))
            for t in pg.traces
        }
        stamped = [c for c in counts_by_trace.values() if c]
        assert stamped, "dynamo trie nodes must carry prefix-cache counts"
        for counts in stamped:
            assert all(t > 0 for _, t in counts.values())
            assert all(0 <= h <= t for h, t in counts.values())

    def test_store_builder_builds_per_trace_map(self):
        from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
        from aiperf.dataset.graph.store_build import GraphStoreBuilder

        pg = from_weka_trace(WEKA_FIXTURE, content_root_seed=0)
        by_trace = GraphStoreBuilder._build_graph_prefix_cache_by_trace(pg)
        assert by_trace == {
            pg.traces[0].id: {
                "trace_03_n3:0": [0, 2],
                "trace_03_n3:1": [2, 3],
                "trace_03_n3:2": [3, 4],
            }
        }
