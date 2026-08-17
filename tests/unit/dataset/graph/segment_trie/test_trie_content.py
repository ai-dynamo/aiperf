# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Driver-level tests for the shared trie-content core: prefix reuse, the ISL gate, and cap planning."""

from __future__ import annotations

import pytest

from aiperf.dataset.graph.segment_trie.pool import SegmentPool
from aiperf.dataset.graph.segment_trie.trie_content import (
    ReconCallbacks,
    TrieAnalysis,
    TrieISLMismatchError,
    TrieNode,
    build_segment_trie,
    compute_asst_caps,
    resolve_content_parents,
)
from tests.unit.dataset.graph.segment_trie.conftest import (
    stub_recon_callbacks,
    trie_node,
)


def node(
    nid: str, order: int, hashes: list[int], in_tok: int, out_tok: int, t: float
) -> TrieNode:
    """A recorded request positioned in the flatten order, in this file's positional shorthand."""
    return trie_node(
        nid,
        hash_ids=hashes,
        input_length=in_tok,
        output_length=out_tok,
        t=t,
        order=order,
    )


def build(
    nodes: list[TrieNode],
    *,
    pool: SegmentPool | None = None,
    callbacks: ReconCallbacks | None = None,
    small_prompt_fallback: bool = False,
    analysis: TrieAnalysis | None = None,
):
    """Run the trie build over ``nodes`` with block_size 4 and no idle-gap cap."""
    return build_segment_trie(
        nodes,
        block_size=4,
        callbacks=callbacks or stub_recon_callbacks(),
        pool=pool or SegmentPool(),
        idle_gap_cap_seconds=None,
        small_prompt_fallback=small_prompt_fallback,
        analysis=analysis,
    )


class TestPrefixReuse:
    """An extending chain shares its prompt prefix and covers exactly the recorded input length."""

    def test_extending_chain_shares_prefix_and_covers_isl(self) -> None:
        """The child's prompt path extends the parent's, and its covered token count equals input_length."""
        pool = SegmentPool()
        result = build(
            [node("a", 0, [1, 2], 8, 4, 0.0), node("b", 1, [1, 2, 3, 4], 16, 4, 5.0)],
            pool=pool,
        )
        p1 = result.builds["a"].prompt_path
        p2 = result.builds["b"].prompt_path
        assert p2[: len(p1)] == p1
        tok2 = sum(len(m["content"].split()) for m in pool.materialize(p2))
        assert tok2 == 16  # block-aligned covered count, NOT inflated

    def test_optional_analysis_uses_existing_prefix_automaton(self) -> None:
        """Opt-in analysis reports trie state counts without changing the build."""
        analysis = TrieAnalysis()
        result = build(
            [
                node("a", 0, [1, 2], 8, 4, 0.0),
                node("b", 1, [1, 2, 3], 12, 6, 5.0),
            ],
            analysis=analysis,
        )

        assert result.analysis is analysis
        assert analysis.input_lengths == [8, 12]
        assert analysis.output_lengths == [4, 6]
        assert analysis.context_lengths == [0, 8]
        assert analysis.unique_prompt_lengths == [8, 4]

    def test_analysis_is_absent_by_default(self) -> None:
        """The default build keeps the analysis collector completely disabled."""
        assert build([node("a", 0, [1], 4, 2, 0.0)]).analysis is None

    def test_recorded_edge_delays_always_replay(self) -> None:
        """Recorded end-to-start gaps survive onto the generated edges rather than collapsing to zero."""
        result = build(
            [node("a", 0, [1], 4, 4, 0.0), node("b", 1, [1, 2], 8, 4, 100.0)]
        )
        delays = [
            e.delay_after_predecessor_us or 0.0
            for edges in result.edges_by_node.values()
            for e in edges
        ]
        assert any(d > 0.0 for d in delays), (
            f"recorded end-to-start gaps must survive onto the edges; got {delays}"
        )

    def test_reuse_boundary_uses_geometry_not_tag_prefix(self) -> None:
        """The reuse boundary is the geometric ``inherited`` count, never a tag-prefix comparison."""
        pool = SegmentPool()
        result = build(
            [
                node("a", 0, [1, 2, 3, 4], 16, 4, 0.0),
                node("b", 1, [1, 2, 8, 9], 16, 4, 1.0),
            ],
            pool=pool,
        )
        # a and b are one all-user message each, so nothing beyond the 2-block
        # shared prefix may alias: blocks 3,4 vs 8,9 decode to distinct tokens
        # and therefore to distinct content-addressed segment ids.
        assert set(result.builds["a"].prompt_path).isdisjoint(
            set(result.builds["b"].prompt_path)
        )

    def test_deep_chain_decodes_each_shared_block_exactly_once(self) -> None:
        """A deep extending chain decodes each covered block once, pinning the death of the quadratic prefix re-emission."""
        n = 10
        nodes = [
            node(f"x{i}", i, list(range(200, 200 + i + 1)), 4 * (i + 1), 4, float(i))
            for i in range(n)
        ]
        calls: list[int] = []

        def counting_decode(hids: list[int]) -> list[int]:
            calls.append(len(hids))
            return [t for h in hids for t in [h] * 4]

        build(
            nodes, callbacks=stub_recon_callbacks(decode_block_tokens=counting_decode)
        )
        assert sum(calls) == n, (
            f"decoded {sum(calls)} blocks; linear reuse decodes {n} "
            f"(pre-rewrite quadratic decoded {n * (n + 1) // 2})"
        )


class TestISLGate:
    """The assembled-token gate compares what was actually built against the recorded input length."""

    def test_rejects_decode_drift(self) -> None:
        """A decode emitting twice the recorded block size aborts the build (W1 hazard: a 64-token default decode against a smaller recorded block size)."""
        drifted = stub_recon_callbacks(tokens_per_block=8)
        with pytest.raises(TrieISLMismatchError):
            build([node("a", 0, [1, 2], 8, 4, 0.0)], callbacks=drifted)

    def test_skipped_for_placeholder_callbacks(self) -> None:
        """``block_exact=False`` opts a scheduling-only timing-plane parse out of the gate, so the same mis-sized decode succeeds."""
        placeholder = stub_recon_callbacks(tokens_per_block=2, block_exact=False)
        result = build([node("a", 0, [1, 2], 8, 4, 0.0)], callbacks=placeholder)
        assert result.builds["a"].prompt_path


class TestCapPlanning:
    """``compute_asst_caps`` must clamp inherited counts exactly as ``assign_block_tags`` freezes them."""

    def test_over_share_row_caps_frozen_tag_owner(self) -> None:
        """On an over-share row (in // bs < lcp) the cap lands on the owner of the re-exposed block the frozen tags actually see."""
        nodes = [
            node("a", 0, [1], 2, 0, 0.0),  # root; tiles [a]
            node("b", 1, [1, 2], 4, 0, 1.0),  # tiles [a, b]
            node("c", 2, [1, 2, 3], 6, 0, 2.0),  # tiles [a, b, c]
            node("d", 3, [1, 2, 3], 4, 0, 3.0),  # over-share: in//2 = 2 < lcp 3
        ]
        resolve_content_parents(nodes)
        caps = compute_asst_caps(nodes, 2)
        # d's degenerate pull-back re-exposes block index 1 (the clamped
        # boundary), owned by b. Unclamped, the planner would cap c instead
        # (block index 2's owner at the unclamped lcp boundary).
        assert caps.get("b") == 0
        assert caps.get("c") is None


class TestMessageFragmentation:
    """Message-level emission respects the clamped inherited boundary."""

    def test_small_prompt_fallback_emits_single_user_message(self) -> None:
        """A sub-block prompt falls back to one user message carrying exactly the recorded tokens."""
        pool = SegmentPool()
        result = build(
            [node("tiny", 0, [], 3, 2, 0.0)], pool=pool, small_prompt_fallback=True
        )
        b = result.builds["tiny"]
        assert b.small_prompt
        msgs = pool.materialize(b.prompt_path)
        assert len(msgs) == 1 and msgs[0]["role"] == "user"
        assert len(msgs[0]["content"].split()) == 3

    def test_boundary_reuses_whole_messages_and_reemits_straddler(self) -> None:
        """Only WHOLE parent messages inside the clamped boundary reuse; a parent message straddling it is re-emitted fresh."""
        pool = SegmentPool()
        result = build(
            [
                node("a", 0, [1, 2], 8, 8, 0.0),  # root -> one user message [0,1]
                node("b", 1, [1, 2, 3, 4, 5, 6], 24, 4, 1.0),  # 3 msgs, ends [2,4,6]
                # covered 3 (hashed partial tail block 4 is dropped), so the
                # boundary lands mid-way through b's second message.
                node("c", 2, [1, 2, 3, 4], 13, 4, 2.0),
            ],
            pool=pool,
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
        c_frag = pool.materialize([pc[1]])[0]
        assert len(c_frag["content"].split()) == 4  # one block * block_size 4
