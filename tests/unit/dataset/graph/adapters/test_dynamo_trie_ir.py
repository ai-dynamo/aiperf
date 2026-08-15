# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the dynamo -> shared-LCP-segment-trie content lowering: later turns must materialize at the block-aligned COVERED token count of their recorded hashes (64 at bs=16), never the old channel-replay inflation (144), plus shared-prefix identity and seed-pinned determinism."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from aiperf.dataset.graph.models import LlmNode
from aiperf.dataset.graph.segment_trie.store_builder import (
    _prompt_segment_ids,
    _trie_llm_nodes,
)
from tests.unit.dataset.graph.adapters.conftest import write_jsonl


def _rec(
    *,
    ts: int,
    sid: str,
    input_tokens: int,
    output_tokens: int,
    hashes: list[int] | None = None,
    block_size: int = 16,
    input_length: int | None = None,
) -> dict:
    """One current-schema ``dynamo.request.trace.v1`` ``request_end`` record."""
    req: dict = {
        "request_id": f"r{ts}",
        "model": "m",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": 0,
    }
    if hashes is not None:
        req["replay"] = {
            "trace_block_size": block_size,
            "input_length": input_length if input_length is not None else input_tokens,
            "input_sequence_hashes": hashes,
        }
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": {"session_id": sid},
        "request": req,
    }


def _dyn_fixture(tmp_path: Path) -> Path:
    """Linear 2-turn single-session trace whose turn-2 hashes EXTEND turn 1's, so turn 2 shares turn 1's leading messages."""
    return write_jsonl(
        tmp_path / "dyn_recorded.jsonl",
        [
            _rec(
                ts=1000,
                sid="s1",
                input_tokens=32,
                output_tokens=8,
                hashes=[111, 222],
                input_length=32,
            ),
            _rec(
                ts=2000,
                sid="s1",
                input_tokens=64,
                output_tokens=12,
                hashes=[111, 222, 333, 444],
                input_length=64,
            ),
        ],
    )


def _dyn_fixture_no_hashes(tmp_path: Path) -> Path:
    """Same 2 extending turns, but with NO replay metadata (virtual-hash path)."""
    return write_jsonl(
        tmp_path / "dyn_virtual.jsonl",
        [
            _rec(ts=1000, sid="s1", input_tokens=32, output_tokens=8),
            _rec(ts=2000, sid="s1", input_tokens=64, output_tokens=12),
        ],
    )


def _builtin_encode(root_seed: int) -> Any:
    """The builtin-tokenizer ``encode`` of the SAME cached synthesizer the parse used."""
    from aiperf.dataset.graph.adapters.shared.content import (
        get_or_build_synthesizer,
    )

    synth = get_or_build_synthesizer(
        "builtin", prompt_corpus="coding", root_seed=root_seed
    )
    return synth._pg.tokenizer.encode


def _materialized_token_count(pool: Any, path: list[str], encode: Any) -> int:
    """Token count of the messages ``pool`` materializes for ``path``."""
    return sum(len(encode(m["content"])) for m in pool.materialize(path))


def _arrival_ordered_llm_nodes(parsed: Any) -> list[tuple[str, LlmNode]]:
    """All ``LlmNode``s in the parsed graph, ordered by arrival offset then node id."""
    return sorted(
        ((nid, n) for nid, n in parsed.graph.nodes.items() if isinstance(n, LlmNode)),
        key=lambda kv: (kv[1].arrival_offset_us or 0, kv[0]),
    )


@pytest.mark.parametrize(
    "make_fixture",
    [
        param(_dyn_fixture, id="recorded_hashes"),
        param(_dyn_fixture_no_hashes, id="virtual_hashes"),
    ],
)  # fmt: skip
def test_turn2_isl_is_covered_count_not_inflated(
    make_fixture: Callable[[Path], Path], tmp_path: Path
) -> None:
    """THE regression: a 2-turn extending chain materializes turn 2 at 64 tokens (covered count == input_length, bs=16), not ~144, and turn 2's path keeps turn 1's prefix."""
    # Seed 1234 is pinned to a corpus window where the builtin decode->encode
    # round trip is block-exact (some windows merge a token across a message
    # boundary, e.g. seed 7 counts 31/63); the sizing itself is seed-independent.
    parsed = from_dynamo_trace(
        make_fixture(tmp_path), content_root_seed=1234, content_tokenizer="builtin"
    )
    pool = parsed.segment_pool
    assert pool is not None
    nodes = _arrival_ordered_llm_nodes(parsed)
    assert len(nodes) == 2

    encode = _builtin_encode(1234)
    p1 = _prompt_segment_ids(nodes[0][1])
    p2 = _prompt_segment_ids(nodes[1][1])
    assert p1 and p2
    assert _materialized_token_count(pool, p1, encode) == 32
    assert _materialized_token_count(pool, p2, encode) == 64
    assert p2[: len(p1)] == p1  # prefix identity survives


def test_trie_parse_is_deterministic_under_pinned_seed(tmp_path: Path) -> None:
    """Parsing twice with the same seed yields identical paths + pool bytes."""
    p = _dyn_fixture(tmp_path)
    parsed_a = from_dynamo_trace(p, content_root_seed=7, content_tokenizer="builtin")
    parsed_b = from_dynamo_trace(p, content_root_seed=7, content_tokenizer="builtin")
    node_a = list(_trie_llm_nodes(parsed_a, parsed_a.traces[0]).values())[-1]
    node_b = list(_trie_llm_nodes(parsed_b, parsed_b.traces[0]).values())[-1]
    path_a = _prompt_segment_ids(node_a)
    path_b = _prompt_segment_ids(node_b)
    assert path_a and path_a == path_b
    assert parsed_a.segment_pool.materialize(
        path_a
    ) == parsed_b.segment_pool.materialize(path_b)


def test_every_node_materializes_from_the_pool(tmp_path: Path) -> None:
    """Every trie node stamps a prompt_segment_ids path the pool can materialize into role/content messages."""
    p = _dyn_fixture(tmp_path)
    parsed = from_dynamo_trace(p, content_root_seed=1234)
    pool = parsed.segment_pool
    assert pool is not None
    for trace in parsed.traces:
        nodes = _trie_llm_nodes(parsed, trace)
        assert nodes
        for nid, node in nodes.items():
            path = _prompt_segment_ids(node)
            assert path, f"{nid} has no prompt_segment_ids"
            msgs = pool.materialize(path)
            assert msgs and all(set(m) == {"role", "content"} for m in msgs)


def test_post_strip_sidecar_carries_no_recorded_replay_hashes(tmp_path: Path) -> None:
    """The content-free graph_meta sidecar must not embed the recorded input_sequence_hashes, so the replay payload must never ride node metadata in the first place."""
    # strip_replay_text clears only prompt and metadata["trie"]; hashes are
    # consumed at lowering time via TrieRequest.hash_ids, never round-tripped.
    from aiperf.dataset.graph.codecs import encode_graph_meta_sidecar
    from aiperf.dataset.graph.graph_meta_sidecar import strip_replay_text

    p = _dyn_fixture(tmp_path)
    parsed = from_dynamo_trace(p, content_root_seed=1234, content_tokenizer="builtin")
    stripped = strip_replay_text(parsed)
    for nid, node in stripped.graph.nodes.items():
        assert set(node.metadata) == {"dynamo", "trie"}, nid
    blob = encode_graph_meta_sidecar(stripped, source_fingerprint={"kind": "test"})
    assert b"input_sequence_hashes" not in blob


def test_recorded_hashes_are_content_global_across_files(tmp_path: Path) -> None:
    """Dynamo hash scope is deliberately GLOBAL (unlike weka's ``local``): equal recorded hashes in two DIFFERENT trace files must synthesize identical bytes."""
    # Recorded input_sequence_hashes are chained hashes over the actual prompt
    # tokens, so equal hash means equal upstream content by construction.
    # Contrast test_weka_trie_hash_scope.py, where hash_id_scope is "local".
    pools = []
    for name in ("capture-a", "capture-b"):
        p = write_jsonl(
            tmp_path / name / "trace.jsonl",
            [
                _rec(
                    ts=1000,
                    sid=f"session-{name}",
                    input_tokens=32,
                    output_tokens=8,
                    hashes=[111, 222],
                    input_length=32,
                ),
            ],
        )
        parsed = from_dynamo_trace(
            p,
            content_root_seed=1234,
            content_tokenizer="builtin",
        )
        node = list(_trie_llm_nodes(parsed, parsed.traces[0]).values())[0]
        pools.append(parsed.segment_pool.materialize(_prompt_segment_ids(node)))

    assert pools[0] == [
        {"role": m["role"], "content": m["content"]} for m in pools[1]
    ], "equal recorded hashes must synthesize identical bytes across files"
