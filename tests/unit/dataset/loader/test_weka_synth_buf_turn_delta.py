# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``ConversationReconstructor.turn_delta``."""

from __future__ import annotations

from aiperf.dataset.loader.weka_synth_buf import (
    ConversationReconstructor,
    RoleSegment,
    TurnDelta,
    truncate_segments_at,
)
from tests.unit.dataset.loader.conftest import make_segments

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
    # in=4*16=64, tool=16, sys=0 -> system holds 1 block, user 3 blocks.
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
    # cut = min(2*16, 32, 80) = 32 == prev_in, so nothing is cut.
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


def test_turn_delta_equal_context_retry_reemits_full_context():
    """An unchanged retry resets to the full prompt instead of emitting ``[]``."""
    r = _make_recon()
    hash_ids = [1, 2]
    in_tokens = 2 * BLOCK_SIZE
    r.init_turn_0(
        hash_ids=hash_ids,
        in_tokens=in_tokens,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )
    first = r.turn_delta()

    r.advance_turn(
        prev_hash_ids=hash_ids,
        prev_in_tokens=in_tokens,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=hash_ids,
        curr_in_tokens=in_tokens,
        seed="t:1",
    )
    retry = r.turn_delta()

    assert retry.reset_context is True
    assert retry.delta_messages == first.delta_messages
    assert retry.delta_messages
    assert retry.delta_messages[-1]["role"] == "user"


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


def test_turn_delta_case_2_boundary_cut_resets_context():
    """A cut that drops filler tokens of a previously-emitted segment resets the context."""
    r = _make_recon()
    # Turn 0: 2 full blocks + an unhashed partial tail of 5 -> 37 tokens.
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

    # Turn 1: prev_hash_ids=[1, 2], curr extends. LCP=2, so the cut at token
    # 32 drops the 5 tail tokens of segment 0. Disturbance recorded at
    # index 0 -> reset.
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


def test_turn_delta_reused_partial_block_appends_without_reset():
    """A later turn that reuses the previous request's partial last block cuts at the previous ``in``: nothing is removed and the delta is a strict append."""
    r = _make_recon()
    prev_in = 2 * BLOCK_SIZE + 5
    r.init_turn_0(
        hash_ids=[1, 2, 3],
        in_tokens=prev_in,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )
    d0 = r.turn_delta()
    before = list(r._segments[0].tokens)

    # cut = min(3*16, 37, 80) = 37: the new region continues hash 3.
    r.advance_turn(
        prev_hash_ids=[1, 2, 3],
        prev_in_tokens=prev_in,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="t:1",
    )
    assert r._last_disturbance_at is None
    assert r._segments[0].tokens == before
    assert r._segments[1].start == prev_in
    assert [t for s in r._segments for t in s.tokens] == _stub_decode_block_tokens(
        [1, 2, 3, 4, 5]
    )

    d1 = r.turn_delta()
    assert d1.reset_context is False
    assert d0.delta_messages + d1.delta_messages == r.snapshot_messages()


def test_turn_delta_case_3_mid_segment_cut_resets_context():
    """LCP lands inside a previously-emitted segment -> reset_context."""
    r = _make_recon()
    # Turn 0: 5 blocks, block-aligned (80 tokens, no partial tail).
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


def _cut(segs: list[RoleSegment], cut: int) -> int | None:
    return truncate_segments_at(
        segs, cut, decode_tokens_to_text=_stub_decode_tokens_to_text
    )


def test_truncate_returns_index_when_cut_at_segment_end_drops_segments_past_it():
    """A cut at a segment end that deletes a following segment must report the first deleted segment index so turn_delta can reset the context."""
    segs = make_segments(("user", 2 * BLOCK_SIZE), ("assistant", BLOCK_SIZE))
    assert _cut(segs, 2 * BLOCK_SIZE) == 1
    assert len(segs) == 1


def test_truncate_returns_none_at_the_end_of_the_buffer():
    """A cut at the end of the trailing segment is a true no-op and returns None."""
    segs = make_segments(("user", 2 * BLOCK_SIZE))
    assert _cut(segs, 2 * BLOCK_SIZE) is None
    assert [len(s.tokens) for s in segs] == [2 * BLOCK_SIZE]


def test_truncate_returns_segment_index_when_cut_lands_at_segment_start():
    """Truncation landing exactly at the start of segment i deletes segments[i:] and reports i as the earliest disturbed index."""
    segs = make_segments(
        ("user", 2 * BLOCK_SIZE), ("assistant", BLOCK_SIZE), ("user", BLOCK_SIZE)
    )
    assert _cut(segs, 3 * BLOCK_SIZE) == 2
    assert [s.role for s in segs] == ["user", "assistant"]


def test_truncate_returns_segment_index_on_mid_segment_cut():
    """Mid-segment cut returns the re-sliced seg index."""
    segs = make_segments(("system", 2 * BLOCK_SIZE), ("user", 4 * BLOCK_SIZE))
    assert _cut(segs, 4 * BLOCK_SIZE) == 1


def test_truncate_returns_zero_when_clearing_non_empty_buffer():
    """A cut at 0 with a non-empty buffer clears every segment and reports the disturbance at index 0."""
    segs = make_segments(("user", BLOCK_SIZE))
    assert _cut(segs, 0) == 0
    assert segs == []


def test_truncate_returns_none_when_zeroes_empty_buffer():
    """A cut at 0 with an already-empty buffer has nothing to disturb."""
    segs: list[RoleSegment] = []
    assert _cut(segs, 0) is None
    assert segs == []


# Regression coverage: truncation deletes previously-emitted segments without
# modifying a surviving segment in place (context must reset).


def test_turn_delta_resets_when_lcp_zero_after_emitted_turn():
    """LCP==0 after an emitted turn clears the synth buffer, so turn_delta reports reset_context=True with a non-empty rebuilt message list."""
    r = _make_recon()
    # Turn 0: 2 blocks, block-aligned (no partial tail) so the only
    # disturbance on turn 1 will be the LCP=0 clear, not a tail strip.
    r.init_turn_0(
        hash_ids=[1, 2],
        in_tokens=2 * BLOCK_SIZE,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )
    d0 = r.turn_delta()
    assert d0.reset_context is False
    assert r._emitted_segment_count == len(r._segments) >= 1

    # Turn 1: disjoint hash_ids -> LCP=0 -> truncate clears the buffer.
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=2 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[97, 98, 99],
        curr_in_tokens=3 * BLOCK_SIZE,
        seed="t:1",
    )
    # The clear-on-LCP=0 path must record a disturbance at index 0, which is
    # strictly less than the prior _emitted_segment_count.
    assert r._last_disturbance_at == 0

    d1 = r.turn_delta()
    assert d1.reset_context is True
    # Emits ALL current segments rebuilt for the new turn.
    assert len(d1.delta_messages) > 0
    assert len(d1.delta_messages) == len(r._segments)
    for msg, seg in zip(d1.delta_messages, r._segments, strict=True):
        assert msg == {"role": seg.role, "content": seg.content}


def test_turn_delta_resets_when_boundary_cut_deletes_emitted_segments():
    """A boundary cut deleting previously-emitted segments (without slicing the boundary segment) resets context, since the earliest deleted segment was emitted."""
    r = _make_recon()
    # Turn 0: 3 blocks, block-aligned (48 tokens).
    r.init_turn_0(
        hash_ids=[1, 2, 3],
        in_tokens=3 * BLOCK_SIZE,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )
    d0 = r.turn_delta()
    assert d0.reset_context is False
    assert len(r._segments) == 1  # single user segment

    # Turn 1: extend by 2 blocks. LCP=3 -> boundary cut at end of seg 0 with
    # nothing to delete, then append asst + user. After turn_delta we have
    # 3 emitted segments: [user(3), asst, user].
    r.advance_turn(
        prev_hash_ids=[1, 2, 3],
        prev_in_tokens=3 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="t:1",
    )
    d1 = r.turn_delta()
    assert d1.reset_context is False
    assert len(r._segments) >= 3
    n_emitted = r._emitted_segment_count
    assert n_emitted == len(r._segments)

    # Turn 2: LCP=3 again. The cut at token 48 lands at the end of
    # segment 0, so nothing is sliced. The cut then deletes segments[1:] (the asst
    # and user from turn 1) -- both already emitted. The fix must report the
    # earliest deleted index (1) so turn_delta resets context.
    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4, 5],
        prev_in_tokens=5 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[1, 2, 3, 88, 99],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="t:2",
    )
    assert r._last_disturbance_at is not None
    assert r._last_disturbance_at < n_emitted

    d2 = r.turn_delta()
    assert d2.reset_context is True
    # All current segments emitted on reset.
    assert len(d2.delta_messages) == len(r._segments)
    for msg, seg in zip(d2.delta_messages, r._segments, strict=True):
        assert msg == {"role": seg.role, "content": seg.content}


def test_turn_delta_strict_append_when_truncation_only_deletes_unemitted_segments():
    """When truncation only deletes segments past _emitted_segment_count, no emitted content is invalidated so turn_delta strict-appends (reset_context=False)."""
    r = _make_recon()
    # Turn 0: 2 blocks, block-aligned (no partial tail).
    r.init_turn_0(
        hash_ids=[1, 2],
        in_tokens=2 * BLOCK_SIZE,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )
    d0 = r.turn_delta()
    assert d0.reset_context is False
    assert r._emitted_segment_count == 1

    # Turn 1: extend by 3 blocks. After advance_turn, _segments grows but we
    # deliberately do NOT call turn_delta(), so _emitted_segment_count stays
    # at 1 -- the new asst+user segments are unemitted.
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=2 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="t:1",
    )
    assert len(r._segments) >= 3
    assert r._emitted_segment_count == 1

    # Turn 2: LCP=2 boundary cut deletes the (unemitted) asst+user appended
    # in turn 1. The fix correctly reports disturbance at index 1, but since
    # 1 >= _emitted_segment_count (1), turn_delta must NOT reset.
    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4, 5],
        prev_in_tokens=5 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[1, 2, 77, 88, 99],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="t:2",
    )
    assert r._last_disturbance_at is not None
    # Disturbance reported but it lies outside the emitted region.
    assert r._last_disturbance_at >= r._emitted_segment_count

    d2 = r.turn_delta()
    assert d2.reset_context is False
    # Strict append emits only segments[_emitted_segment_count:].
    assert len(d2.delta_messages) == len(r._segments) - 1


def test_turn_delta_emits_assistant_segment():
    """A strict-append turn emits its (assistant, user) delta on the wire."""
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


# Context-loss rule: a conversation resumes at a USER turn. When truncation
# removes every user segment (or turn 0 was system-only), the new region must
# not open with an assistant segment — the wire cannot present assistant output
# before any user input.


def test_context_loss_to_system_boundary_resumes_with_user_turn():
    r = _make_recon()
    # Turn 0: 1 system block + 3 user blocks.
    r.init_turn_0(
        hash_ids=[1, 2, 3, 4],
        in_tokens=4 * BLOCK_SIZE,
        tool_tokens=BLOCK_SIZE,
        system_tokens=0,
        seed="s0",
    )
    r.turn_delta()
    # Compaction: only the system block survives; prev_out would normally
    # attribute the head of the new region as assistant.
    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4],
        prev_in_tokens=4 * BLOCK_SIZE,
        prev_out_tokens=20,
        curr_hash_ids=[1, 90, 91],
        curr_in_tokens=3 * BLOCK_SIZE,
        seed="s1",
    )
    delta = r.turn_delta()
    assert delta.reset_context is True
    roles = [m["role"] for m in delta.delta_messages]
    assert roles == ["system", "user"], roles


def test_system_only_turn0_next_turn_resumes_with_user_turn():
    r = _make_recon()
    # Turn 0 fully covered by the system prefix (exact-prefix worker shape).
    r.init_turn_0(
        hash_ids=[1, 2],
        in_tokens=2 * BLOCK_SIZE,
        tool_tokens=0,
        system_tokens=2 * BLOCK_SIZE,
        seed="s0",
    )
    d0 = r.turn_delta()
    assert [m["role"] for m in d0.delta_messages] == ["system"]
    # Pure growth: no user segment exists yet, so the new region is user.
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=2 * BLOCK_SIZE,
        prev_out_tokens=20,
        curr_hash_ids=[1, 2, 9],
        curr_in_tokens=3 * BLOCK_SIZE,
        seed="s1",
    )
    delta = r.turn_delta()
    assert delta.reset_context is False
    assert [m["role"] for m in delta.delta_messages] == ["user"]


def test_pure_growth_after_tail_only_segment_keeps_block_alignment():
    """A boundary cut landing on a non-trailing segment must not strip the previous turn's partial tail from it, preserving the hash-content invariant."""
    r = _make_recon()
    # Turn 0: [user 3b].
    r.init_turn_0(
        hash_ids=[1, 2, 3],
        in_tokens=3 * BLOCK_SIZE,
        tool_tokens=0,
        system_tokens=0,
        seed="s0",
    )
    r.turn_delta()
    # Turn 1: grow by 2 tail-free blocks with a large prev_out. The assistant
    # target would take both, but the final block is reserved for the user so
    # the turn ends with a user segment: [user 3b, assistant(hash4), user(hash5)].
    r.advance_turn(
        prev_hash_ids=[1, 2, 3],
        prev_in_tokens=3 * BLOCK_SIZE,
        prev_out_tokens=200,
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="s1",
    )
    r.turn_delta()
    assert [s.role for s in r._segments] == ["user", "assistant", "user"]
    # Turn 2: tail-only tool result (+12 tokens, no new hash block).
    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4, 5],
        prev_in_tokens=5 * BLOCK_SIZE,
        prev_out_tokens=10,
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=5 * BLOCK_SIZE + 12,
        seed="s2",
        is_tool_result=True,
    )
    r.turn_delta()
    # Turn 3: pure growth ([1,2,3,4,5] -> [1,2,3,4,99]); LCP=4 cut lands exactly
    # on the assistant segment's boundary, deleting the trailing user(hash5)
    # and the tail-only segment past it.
    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4, 5],
        prev_in_tokens=5 * BLOCK_SIZE + 12,
        prev_out_tokens=8,
        curr_hash_ids=[1, 2, 3, 4, 99],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="s3",
    )
    delta = r.turn_delta()
    # Replacing the already-sent tail-only segment is a context reset.
    assert delta.reset_context is True
    # Byte accounting must hold exactly.
    assert sum(len(s.tokens) for s in r._segments) == 5 * BLOCK_SIZE
    # The boundary assistant segment keeps its full hash-block content.
    assert r._segments[1].role == "assistant"
    assert r._segments[1].tokens == _stub_decode_block_tokens([4])
    # The turn still ends with a user segment (the lone new block went to it).
    assert r._segments[-1].role == "user"
    # Re-emitted messages mirror the (uncorrupted) segment contents 1:1.
    for msg, seg in zip(delta.delta_messages, r._segments, strict=True):
        assert msg["content"] == seg.content


def test_context_loss_with_surviving_user_keeps_assistant_attribution():
    r = _make_recon()
    # Turn 0: 1 system block + 3 user blocks.
    r.init_turn_0(
        hash_ids=[1, 2, 3, 4],
        in_tokens=4 * BLOCK_SIZE,
        tool_tokens=BLOCK_SIZE,
        system_tokens=0,
        seed="s0",
    )
    r.turn_delta()
    # Truncation keeps system + part of the user segment: a user turn still
    # precedes the new region, so normal symmetric attribution applies.
    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4],
        prev_in_tokens=4 * BLOCK_SIZE,
        prev_out_tokens=20,
        curr_hash_ids=[1, 2, 90, 91, 92],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="s1",
    )
    delta = r.turn_delta()
    assert delta.reset_context is True
    roles = [m["role"] for m in delta.delta_messages]
    assert roles == ["system", "user", "assistant", "user"], roles
