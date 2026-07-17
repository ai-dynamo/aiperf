# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Driver-level tests for the shared trie-content core."""

from __future__ import annotations

import pytest

from aiperf.dataset.graph.segment_ir.pool import SegmentPool
from aiperf.dataset.graph.segment_ir.trie_content import (
    ReconCallbacks,
    TrieISLMismatchError,
    TrieNode,
    TrieRequest,
    build_trie_ir,
    compute_asst_caps,
    resolve_content_parents,
)


def _stub_callbacks() -> ReconCallbacks:
    # Collision-free deterministic stubs: block tokens = [hash_id] * 4.
    return ReconCallbacks(
        decode_block_tokens=lambda hids: [t for h in hids for t in [h] * 4],
        sample_partial_tail_tokens=lambda n, seed: [7] * n,
        decode_tokens_to_text=lambda toks: " ".join(str(t) for t in toks),
    )


def _node(
    nid: str, order: int, hashes: list[int], in_tok: int, out_tok: int, t: float
) -> TrieNode:
    return TrieNode(
        node_id=nid,
        request=TrieRequest(
            hash_ids=hashes,
            input_length=in_tok,
            output_length=out_tok,
            t=t,
            api_time=1.0,
        ),
        order=order,
    )


def test_extending_chain_prompt_paths_share_prefix_and_cover_isl() -> None:
    nodes = [
        _node("a", 0, [1, 2], 8, 4, 0.0),
        _node("b", 1, [1, 2, 3, 4], 16, 4, 5.0),
    ]
    pool = SegmentPool()
    result = build_trie_ir(
        nodes,
        block_size=4,
        callbacks=_stub_callbacks(),
        pool=pool,
        idle_gap_cap_seconds=None,
    )
    p1 = result.builds["a"].prompt_path
    p2 = result.builds["b"].prompt_path
    assert p2[: len(p1)] == p1
    tok2 = sum(len(m["content"].split()) for m in pool.materialize(p2))
    assert tok2 == 16  # covered count == input_length (block-aligned), NOT inflated


def test_recorded_edge_delays_always_replay() -> None:
    nodes = [
        _node("a", 0, [1], 4, 4, 0.0),
        _node("b", 1, [1, 2], 8, 4, 100.0),
    ]
    result = build_trie_ir(
        nodes,
        block_size=4,
        callbacks=_stub_callbacks(),
        pool=SegmentPool(),
        idle_gap_cap_seconds=None,
    )
    delays = [
        e.delay_after_predecessor_us or 0.0
        for edges in result.edges_by_node.values()
        for e in edges
    ]
    assert any(d > 0.0 for d in delays), (
        f"recorded end-to-start gaps must survive onto the edges; got {delays}"
    )


def test_isl_gate_rejects_decode_drift() -> None:
    """The ISL gate must see the ACTUAL assembled token count: a decode
    callback emitting 2x-block_size blocks (the W1 hazard, e.g. 64-token
    default decode against a smaller recorded block size) must hard-abort the
    build instead of shipping inflated prompts silently."""
    drifted = ReconCallbacks(
        decode_block_tokens=lambda hids: [t for h in hids for t in [h] * 8],  # 2x bs
        sample_partial_tail_tokens=lambda n, seed: [7] * n,
        decode_tokens_to_text=lambda toks: " ".join(str(t) for t in toks),
    )
    nodes = [_node("a", 0, [1, 2], 8, 4, 0.0)]
    with pytest.raises(TrieISLMismatchError):
        build_trie_ir(
            nodes,
            block_size=4,
            callbacks=drifted,
            pool=SegmentPool(),
            idle_gap_cap_seconds=None,
        )


def test_isl_gate_skipped_for_placeholder_callbacks() -> None:
    """``block_exact=False`` (deliberate placeholder content, scheduling-only
    timing-plane parse) opts out of the assembled-token gate; the same
    mis-sized decode that aborts a content-bearing build must succeed."""
    placeholder = ReconCallbacks(
        decode_block_tokens=lambda hids: [t for h in hids for t in [h] * 2],
        sample_partial_tail_tokens=lambda n, seed: [7] * n,
        decode_tokens_to_text=lambda toks: " ".join(str(t) for t in toks),
        block_exact=False,
    )
    nodes = [_node("a", 0, [1, 2], 8, 4, 0.0)]
    result = build_trie_ir(
        nodes,
        block_size=4,
        callbacks=placeholder,
        pool=SegmentPool(),
        idle_gap_cap_seconds=None,
    )
    assert result.builds["a"].prompt_path


def test_compute_asst_caps_over_share_row_caps_frozen_tag_owner() -> None:
    """On an over-share row (in // bs < lcp) the cap planner must clamp the
    inherited count with the SAME three-way min ``assign_block_tags`` freezes
    with, so the degenerate pull-back caps the owner of the re-exposed block
    the tags actually see -- not the owner at the unclamped lcp boundary."""
    nodes = [
        _node("a", 0, [1], 2, 0, 0.0),  # root; tiles [a]
        _node("b", 1, [1, 2], 4, 0, 1.0),  # tiles [a, b]
        _node("c", 2, [1, 2, 3], 6, 0, 2.0),  # tiles [a, b, c]
        _node("d", 3, [1, 2, 3], 4, 0, 3.0),  # over-share: in//2 = 2 < lcp 3
    ]
    resolve_content_parents(nodes)
    caps = compute_asst_caps(nodes, 2)
    # d's degenerate pull-back re-exposes block index 1 (clamped boundary),
    # owned by b -- the boundary the frozen tags land on. Unclamped, the
    # planner would cap c (block index 2's owner) instead.
    assert caps.get("b") == 0
    assert caps.get("c") is None


def test_small_prompt_fallback_emits_single_user_message() -> None:
    nodes = [_node("tiny", 0, [], 3, 2, 0.0)]
    pool = SegmentPool()
    result = build_trie_ir(
        nodes,
        block_size=4,
        callbacks=_stub_callbacks(),
        pool=pool,
        idle_gap_cap_seconds=None,
        small_prompt_fallback=True,
    )
    b = result.builds["tiny"]
    assert b.small_prompt
    msgs = pool.materialize(b.prompt_path)
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    assert len(msgs[0]["content"].split()) == 3


def test_fragment_boundary_reuses_leading_whole_messages_reemits_straddler() -> None:
    """A parent message straddling the child's (clamped) inherited boundary must
    be re-emitted FRESH; only WHOLE parent messages inside the boundary reuse.

    ``c`` over-shares its content-parent ``b`` (``in // bs`` = 3 < lcp = 6), so
    the three-way clamp lands the inherited boundary at block 3 -- strictly
    inside ``b``'s second (assistant) message [2, 4). The rewrite must reuse
    ``b``'s first whole message verbatim (identical content-addressed sid) and
    re-emit the truncated straddling message fresh (a distinct sid, fewer
    blocks). The property is byte-identical before and after the rewrite: shared
    prefixes are content-addressed either way.
    """
    nodes = [
        _node("a", 0, [1, 2], 8, 8, 0.0),  # root -> one user message [0,1]
        _node("b", 1, [1, 2, 3, 4, 5, 6], 24, 4, 1.0),  # 3 msgs, ends [2,4,6]
        _node("c", 2, [1, 2, 3, 4, 5, 6, 9, 10], 12, 4, 2.0),  # over-share: covered 3
    ]
    pool = SegmentPool()
    result = build_trie_ir(
        nodes,
        block_size=4,
        callbacks=_stub_callbacks(),
        pool=pool,
        idle_gap_cap_seconds=None,
    )
    pa = result.builds["a"].prompt_path
    pb = result.builds["b"].prompt_path
    pc = result.builds["c"].prompt_path
    # b reuses a's whole first message; c reuses that same first whole message.
    assert pb[0] == pa[0]
    assert pc[0] == pb[0]
    # c covers only 3 blocks: one reused whole message + one fresh straddler.
    assert len(pc) == 2
    # The straddling message is re-emitted fresh -- NOT b's full-width msg1.
    assert pc[1] != pb[1]
    # And the fresh straddler is exactly one block wide (block 2 only).
    c_frag = pool.materialize([pc[1]])[0]
    assert len(c_frag["content"].split()) == 4  # one block * block_size 4


def test_reuse_boundary_uses_geometry_not_tag_prefix() -> None:
    """The reuse boundary MUST be the geometric ``inherited``, never a tag-prefix
    comparison: tags can coincide beyond the lcp while hash ids differ.

    ``a`` = [1,2,3,4] and ``b`` = [1,2,8,9] share only a 2-block hash prefix but
    tag identically (all user), so a tag-prefix rule would wrongly splice ``a``'s
    block-2 message into ``b``. Geometry stops reuse at block 2, so ``b``'s
    block-2+ segment is a DISTINCT content-addressed sid. Holds before and after
    the rewrite; the rewrite must not regress to a tag-prefix boundary.
    """
    nodes = [
        _node("a", 0, [1, 2, 3, 4], 16, 4, 0.0),
        _node("b", 1, [1, 2, 8, 9], 16, 4, 1.0),
    ]
    pool = SegmentPool()
    result = build_trie_ir(
        nodes,
        block_size=4,
        callbacks=_stub_callbacks(),
        pool=pool,
        idle_gap_cap_seconds=None,
    )
    pa = result.builds["a"].prompt_path
    pb = result.builds["b"].prompt_path
    # No segment beyond the 2-block shared prefix may be shared: a and b are one
    # user message each (all-user roots), so the whole-message sids must differ
    # (blocks 3,4 vs 8,9 give distinct tokens -> distinct content-addressed ids).
    assert set(pa).isdisjoint(set(pb))


def test_deep_chain_decodes_each_shared_block_exactly_once() -> None:
    """A deep extending chain decodes each covered block ONCE (linear), pinning
    the death of the quadratic prefix re-emission.

    Counting stub: every ``decode_block_tokens`` call is recorded. With
    prefix-path reuse each block is materialized exactly once at first occurrence
    and spliced thereafter, so the total decode count equals the number of
    DISTINCT covered blocks (the deepest node's covered count), not the per-node
    re-decode sum ``n(n+1)/2``. This test FAILS on the pre-rewrite quadratic and
    PASSES on the reuse rewrite -- the regression guard for the frozen file.
    """
    n = 10
    nodes = [
        _node(f"x{i}", i, list(range(200, 200 + i + 1)), 4 * (i + 1), 4, float(i))
        for i in range(n)
    ]
    calls: list[int] = []

    def _counting_decode(hids: list[int]) -> list[int]:
        calls.append(len(hids))
        return [t for h in hids for t in [h] * 4]

    cb = ReconCallbacks(
        decode_block_tokens=_counting_decode,
        sample_partial_tail_tokens=lambda n, seed: [7] * n,
        decode_tokens_to_text=lambda toks: " ".join(str(t) for t in toks),
    )
    build_trie_ir(
        nodes,
        block_size=4,
        callbacks=cb,
        pool=SegmentPool(),
        idle_gap_cap_seconds=None,
    )
    assert sum(calls) == n, (
        f"decoded {sum(calls)} blocks; linear reuse decodes {n} "
        f"(pre-rewrite quadratic decoded {n * (n + 1) // 2})"
    )
