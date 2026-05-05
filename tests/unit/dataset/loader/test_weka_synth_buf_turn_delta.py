# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``ConversationReconstructor.turn_delta``.

Covers the four classification cases from the delta-encoding spec
(``docs/dev/proposal-weka-delta-encoding.md``):

* Case 0 (baseline): first call after ``init_turn_0`` emits ALL segments,
  ``reset_context=False``.
* Case 1 (strict append): monotonic LCP — emits only newly-appended
  segments, ``reset_context=False``.
* Case 2 (boundary cut on emitted segment): partial-tail strip on a
  previously-emitted segment forces a context reset.
* Case 3 (mid-segment cut on emitted segment): re-slice of a previously
  emitted segment forces a context reset.
"""

from __future__ import annotations

from aiperf.dataset.loader.weka_synth_buf import (
    ConversationReconstructor,
    RoleSegment,
    TurnDelta,
    truncate_synth_buf_at_block,
)

BLOCK_SIZE = 16


def _stub_decode_block_tokens(hash_ids: list[int]) -> list[int]:
    """Each block is BLOCK_SIZE distinct token IDs keyed on the hash id."""
    out: list[int] = []
    for h in hash_ids:
        out.extend(range(h * 1000, h * 1000 + BLOCK_SIZE))
    return out


def _stub_partial_tail_tokens(n_tokens: int, seed: str) -> list[int]:
    base = (sum(ord(c) for c in seed) % 97) * 100_000 + 50_000
    return list(range(base, base + n_tokens))


def _stub_decode_tokens_to_text(tokens: list[int]) -> str:
    return "|".join(str(t) for t in tokens)


def _make_recon() -> ConversationReconstructor:
    return ConversationReconstructor(
        block_size=BLOCK_SIZE,
        decode_block_tokens=_stub_decode_block_tokens,
        sample_partial_tail_tokens=_stub_partial_tail_tokens,
        decode_tokens_to_text=_stub_decode_tokens_to_text,
    )


# ---------------------------------------------------------------------------
# Case 0: baseline (first call after init_turn_0)
# ---------------------------------------------------------------------------


def test_turn_delta_case_0_baseline_emits_all_segments_no_reset():
    r = _make_recon()
    # Block-aligned: 2 blocks * 16 = 32 tokens, no partial tail.
    r.init_turn_0(
        hash_ids=[1, 2],
        in_tokens=2 * BLOCK_SIZE,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )
    delta = r.turn_delta()
    assert isinstance(delta, TurnDelta)
    assert delta.reset_context is False
    # All current segments emitted.
    assert len(delta.delta_messages) == len(r._segments)
    for msg, seg in zip(delta.delta_messages, r._segments, strict=True):
        assert msg == {"role": seg.role, "content": seg.content}
    # _emitted_segment_count now reflects the full segment list.
    assert r._emitted_segment_count == len(r._segments)
    assert r._last_disturbance_at is None


def test_turn_delta_case_0_with_system_prefix():
    """Baseline with tool+system prefix yields system + user messages."""
    r = _make_recon()
    # in=4*16=64, tool=16, sys=0 -> system block_count=1, user block_count=3.
    r.init_turn_0(
        hash_ids=[1, 2, 3, 4],
        in_tokens=4 * BLOCK_SIZE,
        tool_tokens=BLOCK_SIZE,
        system_tokens=0,
        seed="t:0",
    )
    delta = r.turn_delta()
    roles = [m["role"] for m in delta.delta_messages]
    assert roles == ["system", "user"]
    assert delta.reset_context is False


# ---------------------------------------------------------------------------
# Case 1: strict append (monotonic LCP, no disturbance to emitted segments)
# ---------------------------------------------------------------------------


def test_turn_delta_case_1_strict_append_emits_only_new_segments():
    """Pattern A: full LCP + block-aligned prev_in -> no truncate disturbance."""
    r = _make_recon()
    # Turn 0: 2 blocks, block-aligned (32 tokens, no partial tail).
    r.init_turn_0(
        hash_ids=[1, 2],
        in_tokens=2 * BLOCK_SIZE,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )
    d0 = r.turn_delta()
    assert d0.reset_context is False
    assert len(d0.delta_messages) == len(r._segments)
    n_after_t0 = len(r._segments)

    # Turn 1: extend with 3 new blocks (curr_hash_ids prev is full prefix).
    # prev_in=32, prev_partial_tail=0 -> boundary cut at LCP=2 strips nothing.
    # advance appends asst + user_k.
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=2 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,  # ceil(16/16)=1 asst block
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="t:1",
    )
    d1 = r.turn_delta()
    assert d1.reset_context is False
    # Newly-appended segments only.
    expected_new = len(r._segments) - n_after_t0
    assert len(d1.delta_messages) == expected_new
    # The emitted messages match the segments at index >= n_after_t0.
    for msg, seg in zip(d1.delta_messages, r._segments[n_after_t0:], strict=True):
        assert msg == {"role": seg.role, "content": seg.content}
    # State updated.
    assert r._emitted_segment_count == len(r._segments)
    assert r._last_disturbance_at is None


def test_turn_delta_case_1_strict_append_three_turns_chain():
    """Three sequential strict-append advances: each delta is incremental."""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2],
        in_tokens=2 * BLOCK_SIZE,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )
    d0 = r.turn_delta()
    n0 = len(r._segments)
    assert d0.reset_context is False

    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=2 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[1, 2, 3, 4],
        curr_in_tokens=4 * BLOCK_SIZE,
        seed="t:1",
    )
    d1 = r.turn_delta()
    n1 = len(r._segments)
    assert d1.reset_context is False
    assert len(d1.delta_messages) == n1 - n0

    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4],
        prev_in_tokens=4 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[1, 2, 3, 4, 5, 6],
        curr_in_tokens=6 * BLOCK_SIZE,
        seed="t:2",
    )
    d2 = r.turn_delta()
    n2 = len(r._segments)
    assert d2.reset_context is False
    assert len(d2.delta_messages) == n2 - n1

    # Concatenating the deltas reproduces the full snapshot.
    full = d0.delta_messages + d1.delta_messages + d2.delta_messages
    assert full == r.snapshot_messages()


# ---------------------------------------------------------------------------
# Case 2: boundary cut on a previously-emitted segment
# ---------------------------------------------------------------------------


def test_turn_delta_case_2_boundary_cut_resets_context():
    """Boundary cut strips partial-tail of a previously-emitted segment."""
    r = _make_recon()
    # Turn 0: 2 full blocks + partial tail of 5 -> 37 tokens.
    # block_count=2, len(tokens)=37. We pass exactly 2 hash_ids so total
    # block_count == LCP boundary at advance time.
    r.init_turn_0(
        hash_ids=[1, 2],
        in_tokens=2 * BLOCK_SIZE + 5,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )
    d0 = r.turn_delta()
    assert d0.reset_context is False
    n_after_t0 = len(r._segments)
    assert n_after_t0 >= 1

    # Turn 1: prev_hash_ids=[1, 2], curr extends. LCP=2, prev_partial_tail=5.
    # Boundary cut on segment 0 strips the 5 tail tokens (segment block_count=2,
    # cumulative cursor=0, cursor+block_count==2==target_blocks). Disturbance
    # recorded at index 0 -> reset.
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=2 * BLOCK_SIZE + 5,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="t:1",
    )
    # Verify disturbance was recorded.
    assert r._last_disturbance_at == 0
    assert r._last_disturbance_at < n_after_t0

    d1 = r.turn_delta()
    assert d1.reset_context is True
    # Emits ALL current segments.
    assert len(d1.delta_messages) == len(r._segments)
    for msg, seg in zip(d1.delta_messages, r._segments, strict=True):
        assert msg == {"role": seg.role, "content": seg.content}
    assert r._emitted_segment_count == len(r._segments)
    assert r._last_disturbance_at is None


# ---------------------------------------------------------------------------
# Case 3: mid-segment cut on a previously-emitted segment
# ---------------------------------------------------------------------------


def test_turn_delta_case_3_mid_segment_cut_resets_context():
    """LCP lands inside a previously-emitted segment -> reset_context."""
    r = _make_recon()
    # Turn 0: 5 blocks, block-aligned (80 tokens, no partial tail).
    # The user segment has block_count=5.
    r.init_turn_0(
        hash_ids=[1, 2, 3, 4, 5],
        in_tokens=5 * BLOCK_SIZE,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )
    d0 = r.turn_delta()
    assert d0.reset_context is False
    n_after_t0 = len(r._segments)
    assert n_after_t0 == 1  # single user segment for turn 0.

    # Turn 1: LCP=2 (mid-segment cut at block 2 of segment 0).
    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4, 5],
        prev_in_tokens=5 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[1, 2, 99, 100, 101],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="t:1",
    )
    # Mid-segment cut on segment 0.
    assert r._last_disturbance_at == 0
    assert r._last_disturbance_at < n_after_t0

    d1 = r.turn_delta()
    assert d1.reset_context is True
    assert len(d1.delta_messages) == len(r._segments)
    for msg, seg in zip(d1.delta_messages, r._segments, strict=True):
        assert msg == {"role": seg.role, "content": seg.content}


# ---------------------------------------------------------------------------
# truncate_synth_buf_at_block return-value contract
# ---------------------------------------------------------------------------


def test_truncate_returns_none_on_clean_boundary_no_partial_tail():
    """Boundary cut with prev_partial_tail=0 is a no-op on tokens -> None."""
    segs = [
        RoleSegment(
            role="user",
            block_start=0,
            block_count=2,
            tokens=list(range(2 * BLOCK_SIZE)),
            content="usr",
        ),
        RoleSegment(
            role="assistant",
            block_start=2,
            block_count=1,
            tokens=list(range(BLOCK_SIZE)),
            content="ast",
        ),
    ]
    result = truncate_synth_buf_at_block(
        segs,
        target_blocks=2,
        block_size=BLOCK_SIZE,
        decode_tokens_to_text=_stub_decode_tokens_to_text,
        prev_partial_tail=0,
    )
    assert result is None
    assert len(segs) == 1


def test_truncate_returns_segment_index_on_boundary_strip():
    """Boundary cut with prev_partial_tail>0 returns the stripped seg index."""
    segs = [
        RoleSegment(
            role="system",
            block_start=0,
            block_count=1,
            tokens=list(range(BLOCK_SIZE)),
            content="sys",
        ),
        RoleSegment(
            role="user",
            block_start=1,
            block_count=2,
            tokens=list(range(2 * BLOCK_SIZE + 5)),  # tail of 5
            content="usr",
        ),
    ]
    result = truncate_synth_buf_at_block(
        segs,
        target_blocks=3,
        block_size=BLOCK_SIZE,
        decode_tokens_to_text=_stub_decode_tokens_to_text,
        prev_partial_tail=5,
    )
    assert result == 1


def test_truncate_returns_segment_index_on_mid_segment_cut():
    """Mid-segment cut returns the re-sliced seg index."""
    segs = [
        RoleSegment(
            role="system",
            block_start=0,
            block_count=2,
            tokens=list(range(2 * BLOCK_SIZE)),
            content="sys",
        ),
        RoleSegment(
            role="user",
            block_start=2,
            block_count=4,
            tokens=list(range(4 * BLOCK_SIZE)),
            content="usr",
        ),
    ]
    result = truncate_synth_buf_at_block(
        segs,
        target_blocks=4,  # cuts inside the user segment at kept_blocks=2
        block_size=BLOCK_SIZE,
        decode_tokens_to_text=_stub_decode_tokens_to_text,
    )
    assert result == 1


def test_truncate_returns_none_when_zeroes_segments():
    segs = [
        RoleSegment(
            role="user",
            block_start=0,
            block_count=1,
            tokens=list(range(BLOCK_SIZE)),
            content="x",
        ),
    ]
    result = truncate_synth_buf_at_block(segs, target_blocks=0, block_size=BLOCK_SIZE)
    assert result is None
    assert segs == []


# ---------------------------------------------------------------------------
# emit_assistant_segments=False (live-assistant mode):
# delta_messages drops role=='assistant' segments while _segments retains them
# for LCP / truncation accounting on subsequent turns.
# ---------------------------------------------------------------------------


def _make_recon_user_only() -> ConversationReconstructor:
    return ConversationReconstructor(
        block_size=BLOCK_SIZE,
        decode_block_tokens=_stub_decode_block_tokens,
        sample_partial_tail_tokens=_stub_partial_tail_tokens,
        decode_tokens_to_text=_stub_decode_tokens_to_text,
        emit_assistant_segments=False,
    )


def test_turn_delta_user_only_baseline_keeps_system_and_user():
    """Turn 0 has no assistant segment; user-only mode emits both segments unchanged."""
    r = _make_recon_user_only()
    r.init_turn_0(
        hash_ids=[1, 2, 3, 4],
        in_tokens=4 * BLOCK_SIZE,
        tool_tokens=BLOCK_SIZE,
        system_tokens=0,
        seed="t:0",
    )
    delta = r.turn_delta()
    assert [m["role"] for m in delta.delta_messages] == ["system", "user"]
    assert delta.reset_context is False


def test_turn_delta_user_only_strict_append_drops_assistant_segment():
    """Strict-append turn produces (asst, user) internally; emission is user-only."""
    r = _make_recon_user_only()
    r.init_turn_0(
        hash_ids=[1, 2],
        in_tokens=2 * BLOCK_SIZE,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )
    _ = r.turn_delta()
    n_after_t0 = len(r._segments)

    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=2 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,  # 1 asst block
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="t:1",
    )
    new_segs = r._segments[n_after_t0:]
    new_roles = [s.role for s in new_segs]
    assert new_roles == ["assistant", "user"], (
        "internal segments should still carry the assistant entry"
    )
    delta = r.turn_delta()
    assert [m["role"] for m in delta.delta_messages] == ["user"]
    assert delta.reset_context is False


def test_turn_delta_user_only_default_includes_assistant_segment():
    """Sanity: default mode (emit_assistant_segments=True) does emit the asst delta."""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2],
        in_tokens=2 * BLOCK_SIZE,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )
    _ = r.turn_delta()
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=2 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="t:1",
    )
    delta = r.turn_delta()
    assert [m["role"] for m in delta.delta_messages] == ["assistant", "user"]


def test_turn_delta_user_only_lcp_invariant_preserved_across_turns():
    """LCP/truncation accounting depends on _segments, not delta_messages.

    Run two strict-append turns in user-only mode and confirm the next turn's
    LCP truncation still fires correctly (no IndexError, segments shrink as
    expected) by triggering a pull-back on turn 2.
    """
    r = _make_recon_user_only()
    r.init_turn_0(
        hash_ids=[1, 2, 3, 4],
        in_tokens=4 * BLOCK_SIZE,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )
    _ = r.turn_delta()
    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4],
        prev_in_tokens=4 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[1, 2, 3, 4, 5, 6],
        curr_in_tokens=6 * BLOCK_SIZE,
        seed="t:1",
    )
    _ = r.turn_delta()
    blocks_before = sum(s.block_count for s in r._segments)
    assert blocks_before == 6

    # Pull-back: shrink to 3 blocks of shared prefix; LCP=3 strips trailing.
    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4, 5, 6],
        prev_in_tokens=6 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[1, 2, 3, 7],
        curr_in_tokens=4 * BLOCK_SIZE,
        seed="t:2",
    )
    blocks_after = sum(s.block_count for s in r._segments)
    assert blocks_after == 4, "LCP truncation should have shrunk segments to 4 blocks"
