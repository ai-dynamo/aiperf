# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block-exact ISL: hash-count validation, dropped partial tails, and exact token accounting.

The trie emits WHOLE blocks only. Two consequences are pinned here:

* the partial ``input_length % block_size`` tail is DROPPED (correct: a block
  boundary need not fall on a message/role boundary), counted, and reported once;
* a hash-id count that cannot span the recorded input length is MALFORMED and
  aborts the build instead of being silently absorbed by a ``min()``.
"""

from __future__ import annotations

import pytest

from aiperf.dataset.graph.segment_trie.pool import SegmentPool
from aiperf.dataset.graph.segment_trie.prefix_cache import (
    compute_shared_prefix_cache_counts,
)
from aiperf.dataset.graph.segment_trie.trie_content import (
    TrieAnalysis,
    TrieBlockCountError,
    TrieNode,
    build_segment_trie,
)
from tests.unit.dataset.graph.segment_trie.conftest import (
    stub_recon_callbacks,
    trie_node,
)

BS = 4


def node(nid: str, order: int, hashes: list[int], in_tok: int) -> TrieNode:
    """A recorded request with the block hashes and input length under test."""
    return trie_node(
        nid,
        hash_ids=hashes,
        input_length=in_tok,
        output_length=4,
        t=float(order),
        order=order,
    )


def build(
    nodes: list[TrieNode],
    *,
    pool: SegmentPool | None = None,
    small_prompt_fallback: bool = False,
    analysis: TrieAnalysis | None = None,
):
    """Run the trie build at block_size 4 with the deterministic stub callbacks."""
    return build_segment_trie(
        nodes,
        block_size=BS,
        callbacks=stub_recon_callbacks(),
        pool=pool or SegmentPool(),
        idle_gap_cap_seconds=None,
        small_prompt_fallback=small_prompt_fallback,
        analysis=analysis,
    )


class TestHashCountValidation:
    """``m_full <= len(hash_ids) <= m_full + 1`` or the build aborts."""

    def test_short_block_count_raises_naming_node_and_counts(self) -> None:
        """Fewer hashes than whole blocks is a broken recording, not a short prompt."""
        with pytest.raises(TrieBlockCountError) as excinfo:
            build([node("sess-a:0", 0, [1, 2], 16)])
        message = str(excinfo.value)
        assert "sess-a:0" in message
        assert "needs 4 block(s), got 2" in message

    def test_excess_hash_count_raises(self) -> None:
        """More hashes than the prompt can contain (past the one tail block) aborts."""
        with pytest.raises(TrieBlockCountError):
            build([node("sess-a:0", 0, [1, 2, 3, 4], 4)])

    def test_exact_full_block_count_builds(self) -> None:
        """``len(hash_ids) == m_full`` (block-aligned prompt) is the clean case."""
        pool = SegmentPool()
        result = build([node("a:0", 0, [1, 2], 8)], pool=pool)
        emitted = sum(
            len(m["content"].split())
            for m in pool.materialize(result.builds["a:0"].prompt_path)
        )
        assert emitted == 8

    def test_hashed_tail_builds_and_tail_is_dropped(self) -> None:
        """``len(hash_ids) == m_full + 1`` is legal; the tail block is NOT emitted."""
        pool = SegmentPool()
        result = build([node("a:0", 0, [1, 2, 3], 10)], pool=pool)
        emitted = sum(
            len(m["content"].split())
            for m in pool.materialize(result.builds["a:0"].prompt_path)
        )
        assert emitted == 8  # 2 whole blocks, tail block 3 dropped

    def test_sub_block_prompt_uses_small_prompt_fallback(self) -> None:
        """A prompt shorter than one block is legitimate and keeps its full length."""
        pool = SegmentPool()
        result = build([node("a:0", 0, [], 3)], pool=pool, small_prompt_fallback=True)
        assert result.builds["a:0"].small_prompt is True
        emitted = sum(
            len(m["content"].split())
            for m in pool.materialize(result.builds["a:0"].prompt_path)
        )
        assert emitted == 3


class TestExactRecordedIsl:
    """Analysis reports the length that is actually emitted, never the raw record."""

    def test_recorded_isl_equals_covered_blocks(self) -> None:
        """A hashed-tail node reports ``m_covered * block_size``, matching the wire."""
        analysis = TrieAnalysis()
        pool = SegmentPool()
        result = build([node("a:0", 0, [1, 2, 3], 10)], pool=pool, analysis=analysis)
        emitted = sum(
            len(m["content"].split())
            for m in pool.materialize(result.builds["a:0"].prompt_path)
        )
        assert analysis.input_lengths == [8]
        assert analysis.input_lengths[0] == emitted

    def test_small_prompt_isl_is_not_zeroed(self) -> None:
        """The sub-block carve-out reports its real length, not a floored zero."""
        analysis = TrieAnalysis()
        build([node("a:0", 0, [], 3)], small_prompt_fallback=True, analysis=analysis)
        assert analysis.input_lengths == [3]


class TestPrefixCacheTotals:
    """The theoretical-cache denominator is measured over emitted traffic only."""

    def test_total_blocks_excludes_dropped_tail(self) -> None:
        """``total_blocks`` is ``m_covered``, not the raw hash-id count."""
        counts = compute_shared_prefix_cache_counts(
            [node("a:0", 0, [1, 2, 3], 10)], block_size=BS
        )
        assert counts["a:0"][1] == 2


class TestDroppedTailReporting:
    """Dropped partial tails are counted once for the whole build."""

    def test_dropped_tail_count_over_mixed_corpus(self) -> None:
        """Only nodes with a partial remainder over covered blocks are counted."""
        result = build(
            [
                node("a:0", 0, [1, 2], 8),  # aligned, no tail
                node("b:0", 1, [3, 4, 5], 10),  # 2-token tail dropped
                node("c:0", 2, [6, 7, 8], 11),  # 3-token tail dropped
            ]
        )
        assert result.dropped_tail_nodes == 2
        assert result.dropped_tail_tokens == 5
