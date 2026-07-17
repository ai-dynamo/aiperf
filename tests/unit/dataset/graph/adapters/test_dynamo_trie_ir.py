# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the dynamo -> shared-LCP-segment-trie content lowering.

THE regression this file guards: a multi-turn session's later turns must
materialize at the block-aligned COVERED token count of their recorded hashes
(turn 2 of a 2-turn extending kv trace = 64 tokens at bs=16), never the old
channel-replay inflation (history + full per-turn reconstruction = 144).
Shared-prefix identity (same leading segment ids across extending turns) and
seed-pinned determinism are asserted alongside.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from aiperf.dataset.graph.models import LlmNode
from aiperf.dataset.graph.segment_ir.store_builder import (
    _prompt_segment_ids,
    _trie_llm_nodes,
)


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
    """Write a current-schema linear 2-turn single-session dynamo trace.

    Turn 2's replay hashes EXTEND turn 1's (``[111,222]`` -> ``[111,222,333,444]``)
    so the accumulated turn-2 prompt shares turn 1's leading messages.
    """
    p = tmp_path / "dyn_recorded.jsonl"
    records = [
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
    ]
    p.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
    return p


def _dyn_fixture_no_hashes(tmp_path: Path) -> Path:
    """Same 2 extending turns, but with NO replay metadata (virtual-hash path)."""
    p = tmp_path / "dyn_virtual.jsonl"
    records = [
        _rec(ts=1000, sid="s1", input_tokens=32, output_tokens=8),
        _rec(ts=2000, sid="s1", input_tokens=64, output_tokens=12),
    ]
    p.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
    return p


def _builtin_encode(root_seed: int):
    """The builtin-tokenizer ``encode`` of the SAME cached synthesizer the parse used."""
    from aiperf.dataset.graph.adapters.shared.content import (
        get_or_build_synthesizer,
    )

    synth = get_or_build_synthesizer(
        "builtin", prompt_corpus="coding", root_seed=root_seed
    )
    return synth._pg.tokenizer.encode


def _materialized_token_count(pool, path: list[str], encode) -> int:
    return sum(len(encode(m["content"])) for m in pool.materialize(path))


def _arrival_ordered_llm_nodes(parsed) -> list[tuple[str, LlmNode]]:
    return sorted(
        ((nid, n) for nid, n in parsed.graph.nodes.items() if isinstance(n, LlmNode)),
        key=lambda kv: (kv[1].arrival_offset_us or 0, kv[0]),
    )


def test_turn2_isl_is_covered_count_not_inflated(tmp_path):
    """THE regression: 2-turn extending chain must materialize turn 2 at 64
    tokens (covered count == input_length, bs=16), not ~144 (history + full
    reconstruction double-count)."""
    p = _dyn_fixture(tmp_path)
    parsed = from_dynamo_trace(p, content_root_seed=1234, content_tokenizer="builtin")
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


def test_virtual_mode_isl_is_covered_count(tmp_path):
    """Virtual (no recorded hashes) path: the same covered-count contract holds.

    Seed 1234 is pinned to a corpus window where the builtin decode->encode
    round trip is block-exact (a handful of windows merge one token across a
    message boundary, e.g. seed 7 counts 31/63); the covered-count sizing
    under test is seed-independent.
    """
    p = _dyn_fixture_no_hashes(tmp_path)
    parsed = from_dynamo_trace(
        p,
        content_root_seed=1234,
        content_tokenizer="builtin",
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
    assert p2[: len(p1)] == p1


def test_trie_parse_is_deterministic_under_pinned_seed(tmp_path):
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


def test_every_node_materializes_from_the_pool(tmp_path):
    """Every trie node stamps a prompt_segment_ids path the pool can materialize."""
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


def test_post_strip_sidecar_carries_no_recorded_replay_hashes(tmp_path):
    """The content-free graph_meta sidecar must not embed the recorded
    input_sequence_hashes: strip_replay_text clears only prompt and
    metadata["trie"], so the replay payload must never ride node metadata in
    the first place (hash consumption happens at lowering time via
    TrieRequest.hash_ids; no recorded-scalar round-trip is stamped)."""
    from aiperf.dataset.graph.codecs import encode_graph_meta_sidecar
    from aiperf.dataset.graph.graph_meta_sidecar import strip_replay_text

    p = _dyn_fixture(tmp_path)
    parsed = from_dynamo_trace(p, content_root_seed=1234, content_tokenizer="builtin")
    stripped = strip_replay_text(parsed)
    for nid, node in stripped.graph.nodes.items():
        assert set(node.metadata) == {"dynamo", "trie"}, nid
    blob = encode_graph_meta_sidecar(stripped, source_fingerprint={"kind": "test"})
    assert b"input_sequence_hashes" not in blob


def test_recorded_hashes_are_content_global_across_files(tmp_path):
    """Dynamo hash scope is deliberately GLOBAL, unlike weka's ``local`` scope.

    Recorded ``input_sequence_hashes`` are chained sequence hashes over the
    actual prompt tokens -- equal hash means equal upstream content by
    construction -- so the same hashes in two DIFFERENT trace files (e.g. two
    captures of the same session) must synthesize identical bytes. Contrast
    ``test_weka_trie_hash_scope.py``, where weka's ``hash_id_scope: "local"``
    requires the opposite (weka's ``"global"`` scope matches dynamo).
    """
    pools = []
    for name in ("capture-a", "capture-b"):
        d = tmp_path / name
        d.mkdir()
        p = d / "trace.jsonl"
        records = [
            _rec(
                ts=1000,
                sid=f"session-{name}",
                input_tokens=32,
                output_tokens=8,
                hashes=[111, 222],
                input_length=32,
            ),
        ]
        p.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
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
