# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``ConversationReconstructor.turn_delta``."""

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


def test_turn_delta_case_0_baseline_emits_all_segments_no_reset():
    r = _make_recon()
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
    assert len(delta.delta_messages) == len(r._segments)
    for msg, seg in zip(delta.delta_messages, r._segments, strict=True):
        assert msg == {"role": seg.role, "content": seg.content}
    assert r._emitted_segment_count == len(r._segments)
    assert r._last_disturbance_at is None


def test_turn_delta_case_0_with_system_prefix():
    """Baseline with tool+system prefix yields system + user messages."""
    r = _make_recon()
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

    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=2 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="t:1",
    )
    d1 = r.turn_delta()
    assert d1.reset_context is False
    expected_new = len(r._segments) - n_after_t0
    assert len(d1.delta_messages) == expected_new
    for msg, seg in zip(d1.delta_messages, r._segments[n_after_t0:], strict=True):
        assert msg == {"role": seg.role, "content": seg.content}
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

    full = d0.delta_messages + d1.delta_messages + d2.delta_messages
    assert full == r.snapshot_messages()


def test_turn_delta_case_2_boundary_cut_resets_context():
    """Boundary cut strips partial-tail of a previously-emitted segment."""
    r = _make_recon()
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

    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=2 * BLOCK_SIZE + 5,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="t:1",
    )
    assert r._last_disturbance_at == 0
    assert r._last_disturbance_at < n_after_t0

    d1 = r.turn_delta()
    assert d1.reset_context is True
    assert len(d1.delta_messages) == len(r._segments)
    for msg, seg in zip(d1.delta_messages, r._segments, strict=True):
        assert msg == {"role": seg.role, "content": seg.content}
    assert r._emitted_segment_count == len(r._segments)
    assert r._last_disturbance_at is None


def test_turn_delta_case_3_mid_segment_cut_resets_context():
    """LCP lands inside a previously-emitted segment -> reset_context."""
    r = _make_recon()
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
    assert n_after_t0 == 1

    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4, 5],
        prev_in_tokens=5 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[1, 2, 99, 100, 101],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="t:1",
    )
    assert r._last_disturbance_at == 0
    assert r._last_disturbance_at < n_after_t0

    d1 = r.turn_delta()
    assert d1.reset_context is True
    assert len(d1.delta_messages) == len(r._segments)
    for msg, seg in zip(d1.delta_messages, r._segments, strict=True):
        assert msg == {"role": seg.role, "content": seg.content}


def test_truncate_returns_index_when_boundary_cut_drops_segments_past_boundary():
    """Boundary cut with prev_partial_tail=0 that deletes a following segment"""
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
    )
    assert result == 1
    assert len(segs) == 1


def test_truncate_returns_none_on_clean_boundary_with_no_segments_past():
    """Boundary cut at the end of the trailing segment, no partial tail and"""
    segs = [
        RoleSegment(
            role="user",
            block_start=0,
            block_count=2,
            tokens=list(range(2 * BLOCK_SIZE)),
            content="usr",
        ),
    ]
    result = truncate_synth_buf_at_block(
        segs,
        target_blocks=2,
        block_size=BLOCK_SIZE,
        decode_tokens_to_text=_stub_decode_tokens_to_text,
    )
    assert result is None
    assert len(segs) == 1


def test_truncate_returns_segment_index_when_cut_lands_at_segment_start():
    """Truncation lands exactly at the start of segment i and deletes"""
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
            content="ast0",
        ),
        RoleSegment(
            role="user",
            block_start=3,
            block_count=1,
            tokens=list(range(BLOCK_SIZE)),
            content="usr1",
        ),
    ]
    result = truncate_synth_buf_at_block(
        segs,
        target_blocks=3,
        block_size=BLOCK_SIZE,
        decode_tokens_to_text=_stub_decode_tokens_to_text,
    )
    assert result == 2
    assert [s.role for s in segs] == ["user", "assistant"]


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
            tokens=list(range(2 * BLOCK_SIZE + 5)),
            content="usr",
        ),
    ]
    result = truncate_synth_buf_at_block(
        segs,
        target_blocks=3,
        block_size=BLOCK_SIZE,
        decode_tokens_to_text=_stub_decode_tokens_to_text,
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
        target_blocks=4,
        block_size=BLOCK_SIZE,
        decode_tokens_to_text=_stub_decode_tokens_to_text,
    )
    assert result == 1


def test_truncate_returns_zero_when_clearing_non_empty_buffer():
    """target_blocks<=0 with a non-empty buffer clears every segment, which is"""
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
    assert result == 0
    assert segs == []


def test_truncate_returns_none_when_zeroes_empty_buffer():
    """target_blocks<=0 with an already-empty buffer has nothing to disturb."""
    segs: list[RoleSegment] = []
    result = truncate_synth_buf_at_block(segs, target_blocks=0, block_size=BLOCK_SIZE)
    assert result is None
    assert segs == []


def test_turn_delta_resets_when_lcp_zero_after_emitted_turn():
    """LCP==0 after at least one emitted turn forces target_blocks=0, which"""
    r = _make_recon()
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

    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=2 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[97, 98, 99],
        curr_in_tokens=3 * BLOCK_SIZE,
        seed="t:1",
    )
    assert r._last_disturbance_at == 0

    d1 = r.turn_delta()
    assert d1.reset_context is True
    assert len(d1.delta_messages) > 0
    assert len(d1.delta_messages) == len(r._segments)
    for msg, seg in zip(d1.delta_messages, r._segments, strict=True):
        assert msg == {"role": seg.role, "content": seg.content}


def test_turn_delta_resets_when_boundary_cut_deletes_emitted_segments():
    """Boundary cut deletes one or more previously-emitted segments without"""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2, 3],
        in_tokens=3 * BLOCK_SIZE,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )
    d0 = r.turn_delta()
    assert d0.reset_context is False
    assert len(r._segments) == 1

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
    assert len(d2.delta_messages) == len(r._segments)
    for msg, seg in zip(d2.delta_messages, r._segments, strict=True):
        assert msg == {"role": seg.role, "content": seg.content}


def test_turn_delta_strict_append_when_truncation_only_deletes_unemitted_segments():
    """When truncation only deletes segments past _emitted_segment_count, the"""
    r = _make_recon()
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

    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4, 5],
        prev_in_tokens=5 * BLOCK_SIZE,
        prev_out_tokens=BLOCK_SIZE,
        curr_hash_ids=[1, 2, 77, 88, 99],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="t:2",
    )
    assert r._last_disturbance_at is not None
    assert r._last_disturbance_at >= r._emitted_segment_count

    d2 = r.turn_delta()
    assert d2.reset_context is False
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


def test_context_loss_to_system_boundary_resumes_with_user_turn():
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2, 3, 4],
        in_tokens=4 * BLOCK_SIZE,
        tool_tokens=BLOCK_SIZE,
        system_tokens=0,
        seed="s0",
    )
    r.turn_delta()
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
    r.init_turn_0(
        hash_ids=[1, 2],
        in_tokens=2 * BLOCK_SIZE,
        tool_tokens=0,
        system_tokens=2 * BLOCK_SIZE,
        seed="s0",
    )
    d0 = r.turn_delta()
    assert [m["role"] for m in d0.delta_messages] == ["system"]
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
    """A boundary cut landing on a NON-trailing segment must not strip the"""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2, 3],
        in_tokens=3 * BLOCK_SIZE,
        tool_tokens=0,
        system_tokens=0,
        seed="s0",
    )
    r.turn_delta()
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
    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4, 5],
        prev_in_tokens=5 * BLOCK_SIZE + 12,
        prev_out_tokens=8,
        curr_hash_ids=[1, 2, 3, 4, 99],
        curr_in_tokens=5 * BLOCK_SIZE,
        seed="s3",
    )
    delta = r.turn_delta()
    assert delta.reset_context is True
    assert sum(len(s.tokens) for s in r._segments) == 5 * BLOCK_SIZE
    assert r._segments[1].role == "assistant"
    assert r._segments[1].tokens == _stub_decode_block_tokens([4])
    assert r._segments[-1].role == "user"
    for msg, seg in zip(delta.delta_messages, r._segments, strict=True):
        assert msg["content"] == seg.content


def test_context_loss_with_surviving_user_keeps_assistant_attribution():
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2, 3, 4],
        in_tokens=4 * BLOCK_SIZE,
        tool_tokens=BLOCK_SIZE,
        system_tokens=0,
        seed="s0",
    )
    r.turn_delta()
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
