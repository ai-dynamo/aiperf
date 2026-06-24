# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the byte-exact weka conversation reconstructor.

These tests stub out the real prompt synthesis so they don't need a
tokenizer; they verify segment shapes, LCP-driven truncation, and the
symmetric asst|user attribution rule.

Invariants tested:
- ``sum(len(seg.tokens)) == in_tokens`` exactly after init_turn_0 and
  advance_turn (block-aligned segment sizes).
- Every segment except the trailing user holds ``block_count * bs`` tokens.
- The hash-content invariant: a given ``hash_id`` decodes to the identical
  token sequence in every segment of every turn (no terminator stamp on
  the trailing tokens).
"""

import math

import pytest

from aiperf.dataset.loader.weka_synth_buf import (
    ConversationReconstructor,
    RoleSegment,
    longest_common_prefix,
    truncate_synth_buf_at_block,
)


def _stub_decode_block_tokens(hash_ids):
    """Each block is 64 distinct token IDs keyed on the hash id."""
    out: list[int] = []
    for h in hash_ids:
        out.extend(range(h * 100, h * 100 + 64))
    return out


def _stub_partial_tail_tokens(n_tokens, seed):
    """Deterministic n token IDs keyed on seed."""
    base = sum(ord(c) for c in seed) * 1000
    return list(range(base, base + n_tokens))


def _stub_decode_tokens_to_text(tokens):
    return "|".join(str(t) for t in tokens)


def _make_recon(bs=64, terminator_tokens=None):
    return ConversationReconstructor(
        block_size=bs,
        decode_block_tokens=_stub_decode_block_tokens,
        sample_partial_tail_tokens=_stub_partial_tail_tokens,
        decode_tokens_to_text=_stub_decode_tokens_to_text,
        bpe_stable_terminator_tokens=terminator_tokens or [],
    )


def test_init_creates_empty_synth_buf():
    r = _make_recon()
    assert r.snapshot_messages() == []


def test_init_turn_0_no_prefix_emits_one_user_segment():
    r = _make_recon()
    # in=200, hash_ids covers floor(200/64) = 3 blocks, partial_tail = 8 tokens
    r.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=200, tool_tokens=0, system_tokens=0, seed="t:0"
    )
    segs = r._segments
    assert len(segs) == 1
    assert segs[0].role == "user"
    assert segs[0].block_start == 0
    assert segs[0].block_count == 3
    assert segs[0].content_token_count == 200
    assert len(segs[0].tokens) == 200


def test_init_turn_0_with_tool_and_system_prefix_split():
    r = _make_recon()
    # in=500, tool=100, system=50, user=remainder (block_size=64).
    # tool+system merged into ONE system segment.
    # prefix_tokens = 150 -> prefix_blocks = ceil(150/64) = 3 -> 3*64 = 192 tokens.
    # M_full = floor(500/64) = 7 -> user_blocks = 7 - 3 = 4 -> 256 tokens.
    # partial_tail = 500 % 64 = 52 -> user_total = 256 + 52 = 308.
    # sum = 192 + 308 = 500 == in_tokens (exact).
    r.init_turn_0(
        hash_ids=list(range(1, 8)),
        in_tokens=500,
        tool_tokens=100,
        system_tokens=50,
        seed="t:0",
    )
    roles = [s.role for s in r._segments]
    assert roles == ["system", "user"]  # tool+system merged per spec §4.3
    assert r._segments[0].content_token_count == 192
    assert r._segments[1].content_token_count == 308
    # Block-aligned merged prefix: holds full block content for blocks 1,2,3.
    assert r._segments[0].tokens == _stub_decode_block_tokens([1, 2, 3])
    # Token-level invariant: tokens list size == content_token_count.
    for seg in r._segments:
        assert len(seg.tokens) == seg.content_token_count
    # Byte-exact sum: total tokens == recorded in_tokens.
    assert sum(len(s.tokens) for s in r._segments) == 500


def test_init_turn_0_prefix_block_rounding_overshoot_clamps_to_budget():
    """Regression: a declared prefix whose BLOCK count exceeds the prompt's own
    covered-block count must clamp the system segment, not emit a negative
    block_count / over-budget tokens.

    in=170 (m_full=2), tool=130 -> prefix_blocks=ceil(130/64)=3 > m_full=2, with
    3 recorded hash blocks so the hash-availability guard does not fire. Without
    the clamp user_blocks = covered(2) - cursor(3) = -1 and the system segment
    holds 3*64=192 tokens (> in_tokens), breaking sum == in_tokens.
    """
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=170, tool_tokens=130, system_tokens=0, seed="t:0"
    )
    segs = r._segments
    assert all(s.block_count >= 0 for s in segs), [s.block_count for s in segs]
    sys_seg = next(s for s in segs if s.role == "system")
    assert sys_seg.block_count == 2  # clamped from 3 to floor(170/64)
    assert sum(len(s.tokens) for s in segs) == 170


def test_init_turn_0_prefix_exceeding_input_tokens_clamps_to_budget():
    """Regression: a prefix that outright exceeds the whole turn-0 input must
    clamp rather than produce a negative-block_count segment.

    in=100 (m_full=1), tool=130 -> prefix_blocks=ceil(130/64)=3, hash_ids has 3
    blocks (guard passes). covered_blocks=min(1,3)=1, so the system segment
    clamps to 1 block and the user tail carries the partial remainder.
    """
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=100, tool_tokens=130, system_tokens=0, seed="t:0"
    )
    segs = r._segments
    assert all(s.block_count >= 0 for s in segs), [s.block_count for s in segs]
    sys_seg = next(s for s in segs if s.role == "system")
    assert sys_seg.block_count == 1
    assert sum(len(s.tokens) for s in segs) == 100


def test_init_turn_0_partial_tail_appended_to_user_content():
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=200, tool_tokens=0, system_tokens=0, seed="t:0"
    )
    # Partial-tail tokens come from _stub_partial_tail_tokens(8, "t:0").
    expected_tail = _stub_partial_tail_tokens(8, "t:0")
    user_tokens = r._segments[0].tokens
    # Last 8 tokens of the user segment must be the partial-tail tokens.
    assert user_tokens[-8:] == expected_tail


def test_init_turn_0_zero_partial_tail_no_tail_marker():
    r = _make_recon()
    # in=192 = 3*64 exactly, no partial tail
    r.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=192, tool_tokens=0, system_tokens=0, seed="t:0"
    )
    # User tokens should be exactly the concatenated block tokens — no tail.
    expected = _stub_decode_block_tokens([1, 2, 3])
    assert r._segments[0].tokens == expected


def test_init_turn_0_combines_tool_and_system_into_single_system():
    """tool+system must emit exactly ONE role="system" segment.

    Some serving stacks reject multiple adjacent system messages, so the
    reconstructor merges trace-level tool_tokens and system_tokens into a
    single system segment whose hash-block range covers what two separate
    segments would otherwise cover.
    """
    bs = 64
    in_tokens = 1000
    tool_tokens = 200
    system_tokens = 300
    m_full = in_tokens // bs  # 15
    hash_ids = list(range(1, m_full + 1))
    r = _make_recon()
    r.init_turn_0(
        hash_ids=hash_ids,
        in_tokens=in_tokens,
        tool_tokens=tool_tokens,
        system_tokens=system_tokens,
        seed="t:0:p19",
    )
    roles = [s.role for s in r._segments]
    # Exactly ONE system segment, immediately followed by user.
    assert roles.count("system") == 1
    assert roles == ["system", "user"]
    sys_seg = r._segments[0]
    expected_prefix_blocks = math.ceil((tool_tokens + system_tokens) / bs)
    assert sys_seg.block_count == expected_prefix_blocks
    assert len(sys_seg.tokens) == expected_prefix_blocks * bs
    # The merged system segment consumes the prefix block range [0..N).
    assert sys_seg.block_start == 0
    # Byte-exact: all segments together total in_tokens.
    assert sum(len(s.tokens) for s in r._segments) == in_tokens


def test_role_segment_invariants():
    seg = RoleSegment(
        role="user",
        block_start=0,
        block_count=3,
        tokens=list(range(180)),
        content="abc",
    )
    # content_token_count is a property derived from tokens.
    assert seg.content_token_count == 180
    # content_token_count <= block_count * bs (with bs=64)
    assert seg.content_token_count <= seg.block_count * 64


def test_snapshot_messages_round_trips_segments():
    r = _make_recon()
    r._segments = [
        RoleSegment(
            role="system",
            block_start=0,
            block_count=1,
            tokens=list(range(50)),
            content="sys",
        ),
        RoleSegment(
            role="user",
            block_start=1,
            block_count=2,
            tokens=list(range(120)),
            content="usr",
        ),
    ]
    msgs = r.snapshot_messages()
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_lcp_identical_lists():
    assert longest_common_prefix([1, 2, 3], [1, 2, 3]) == 3


def test_lcp_empty():
    assert longest_common_prefix([], []) == 0
    assert longest_common_prefix([], [1]) == 0
    assert longest_common_prefix([1], []) == 0


def test_lcp_prefix_extension():
    assert longest_common_prefix([1, 2, 3], [1, 2, 3, 4, 5]) == 3
    assert longest_common_prefix([1, 2, 3, 4, 5], [1, 2, 3]) == 3


def test_lcp_divergence_at_first_position():
    assert longest_common_prefix([1, 2, 3], [4, 5, 6]) == 0


def test_lcp_mid_sequence_replacement():
    # Pattern B: trailing-block churn
    assert longest_common_prefix([1, 2, 3, 4], [1, 2, 3, 5, 6]) == 3


def test_truncate_at_segment_boundary():
    segs = [
        RoleSegment(
            role="system",
            block_start=0,
            block_count=2,
            tokens=list(range(120)),
            content="sys",
        ),
        RoleSegment(
            role="user",
            block_start=2,
            block_count=3,
            tokens=list(range(180)),
            content="usr",
        ),
        RoleSegment(
            role="assistant",
            block_start=5,
            block_count=2,
            tokens=list(range(120)),
            content="ast",
        ),
    ]
    truncate_synth_buf_at_block(segs, target_blocks=5, block_size=64)
    assert [s.role for s in segs] == ["system", "user"]


def test_truncate_at_zero_drops_all():
    segs = [
        RoleSegment(
            role="system",
            block_start=0,
            block_count=2,
            tokens=list(range(120)),
            content="sys",
        ),
        RoleSegment(
            role="user",
            block_start=2,
            block_count=3,
            tokens=list(range(180)),
            content="usr",
        ),
    ]
    truncate_synth_buf_at_block(segs, target_blocks=0, block_size=64)
    assert segs == []


def test_truncate_mid_segment_preserves_partial_content():
    segs = [
        RoleSegment(
            role="system",
            block_start=0,
            block_count=2,
            tokens=list(range(120)),
            content="sys",
        ),
        RoleSegment(
            role="user",
            block_start=2,
            block_count=4,
            tokens=list(range(240)),
            content="x" * 240,
        ),
    ]
    # truncate at block 4 — drops last 2 blocks of user segment
    truncate_synth_buf_at_block(
        segs,
        target_blocks=4,
        block_size=64,
        decode_tokens_to_text=_stub_decode_tokens_to_text,
    )
    assert [s.role for s in segs] == ["system", "user"]
    user = segs[1]
    assert user.block_count == 2
    assert user.content_token_count == 128  # 2 * 64
    assert len(user.tokens) == 128
    # content should have been re-derived from the sliced tokens.
    assert user.content == _stub_decode_tokens_to_text(list(range(128)))


def test_truncate_beyond_total_blocks_no_op():
    segs = [
        RoleSegment(
            role="system",
            block_start=0,
            block_count=2,
            tokens=list(range(120)),
            content="sys",
        ),
    ]
    truncate_synth_buf_at_block(segs, target_blocks=999, block_size=64)
    assert len(segs) == 1


def test_truncate_at_boundary_strips_partial_tail():
    """At a boundary cut, the trailing ``prev_partial_tail`` tokens are
    stripped. The only trailing tokens past ``block_count * bs`` are the
    partial tail (block-aligned segments eliminate asst-block-rounding
    overhead at segment boundaries)."""
    bs = 64
    block_count = 1
    partial_tail = 36  # superseded by next turn's tiling
    total_tokens = block_count * bs + partial_tail
    segs = [
        RoleSegment(
            role="user",
            block_start=4,
            block_count=block_count,
            tokens=list(range(total_tokens)),
            content="usr",
        ),
    ]
    truncate_synth_buf_at_block(
        segs,
        target_blocks=block_count,
        block_size=bs,
        decode_tokens_to_text=_stub_decode_tokens_to_text,
    )
    assert len(segs) == 1
    seg = segs[0]
    assert len(seg.tokens) == block_count * bs
    assert seg.tokens == list(range(block_count * bs))
    # Content re-derived from the surviving tokens.
    assert seg.content == _stub_decode_tokens_to_text(list(range(block_count * bs)))


def test_truncate_at_boundary_no_partial_tail_keeps_all_tokens():
    """With ``prev_partial_tail=0``, no trailing tokens are stripped."""
    bs = 64
    block_count = 2
    total_tokens = block_count * bs
    segs = [
        RoleSegment(
            role="user",
            block_start=2,
            block_count=block_count,
            tokens=list(range(total_tokens)),
            content="usr",
        ),
    ]
    truncate_synth_buf_at_block(
        segs,
        target_blocks=block_count,
        block_size=bs,
        decode_tokens_to_text=_stub_decode_tokens_to_text,
    )
    assert len(segs) == 1
    seg = segs[0]
    assert len(seg.tokens) == total_tokens
    assert seg.tokens == list(range(total_tokens))


def test_advance_pattern_a_clean_append():
    """LCP == M_prev: add asst sized to ceil(out[k-1]/bs)*bs, rest as user."""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2], in_tokens=128, tool_tokens=0, system_tokens=0, seed="s0"
    )
    # turn k: hash_ids extends by 3 blocks. in=320, partial_tail=0.
    # new_region = 3*64 = 192 tokens. out[k-1] = 100 ->
    # asst_blocks = ceil(100/64) = 2 -> asst_tokens = 128.
    # user_blocks = 3 - 2 = 1 -> user_tokens = 64.
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=128,
        prev_out_tokens=100,
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=320,
        seed="s1",
    )
    roles = [s.role for s in r._segments]
    assert roles == ["user", "assistant", "user"]
    asst = r._segments[1]
    assert asst.content_token_count == 128
    assert asst.block_count == 2
    user_k = r._segments[2]
    assert user_k.content_token_count == 64
    assert user_k.block_count == 1
    # Byte-exact sum: 128 (turn-0 user, untouched) + 128 (asst) + 64 (user_k) == 320.
    assert sum(len(s.tokens) for s in r._segments) == 320


def test_advance_pattern_b_trailing_block_churn():
    """LCP == M_prev - 1 (trailing-block recomposition).

    ``curr_hash_ids`` has 5 entries while ``curr_in_tokens=300`` covers only
    ``300 // 64 = 4`` full blocks -- the 5th hash is a partial last block
    (300 % 64 = 44 tokens). ``_advance_to_turn`` clamps the new region to the
    covered-block budget (``min(m_curr, m_curr_full)``), exactly mirroring
    ``init_turn_0``'s ``covered_blocks = min(m_full, len(hash_ids))``, so the
    byte-exact invariant ``sum(seg.tokens) == curr_in_tokens`` holds rather
    than overshooting by ~bs tokens.
    """
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=180, tool_tokens=0, system_tokens=0, seed="s0"
    )
    # turn-0 user: in=180, m_full=2, partial_tail=52 -> block_count=2, 180 tokens.
    #
    # turn k: LCP=2, m_curr=5, m_curr_full=300//64=4 -> m_curr_covered=4.
    # truncate at LCP=2 strips turn-0 user's 52-token partial tail -> 128 tokens.
    # new_region = (m_curr_covered - lcp)=2 covered blocks * 64 + (300 % 64)=44
    #            = 128 + 44 = 172 tokens (the partial 5th hash is the tail).
    # out=50 -> asst_blocks = ceil(50/64) = 1 -> asst_tokens = 64.
    # user_k = 172 - 64 = 108 tokens, block_count = (m_curr_covered-lcp)-asst = 1.
    # sum = 128 + 64 + 108 = 300 == curr_in_tokens (exact).
    r.advance_turn(
        prev_hash_ids=[1, 2, 3],
        prev_in_tokens=180,
        prev_out_tokens=50,
        curr_hash_ids=[1, 2, 99, 100, 101],
        curr_in_tokens=300,
        seed="s1",
    )
    roles = [s.role for s in r._segments]
    assert roles == ["user", "assistant", "user"]
    # turn-0 user truncated to LCP=2 with prev_partial_tail=52 stripped -> 128.
    assert r._segments[0].block_count == 2
    assert r._segments[0].content_token_count == 128
    # asst: ceil(50/64)*64 = 64.
    assert r._segments[1].content_token_count == 64
    assert r._segments[1].block_count == 1
    # user_k: 1 remaining covered block * 64 + 44 partial_tail = 108.
    assert r._segments[2].content_token_count == 108
    assert r._segments[2].block_count == 1
    # Byte-exact: total equals the recorded input length.
    assert sum(len(s.tokens) for s in r._segments) == 300


def test_advance_pattern_c_pull_back():
    """M_curr < M_prev: significant compaction. Asst still attributed up to recorded size."""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=list(range(1, 11)),
        in_tokens=620,
        tool_tokens=0,
        system_tokens=0,
        seed="s0",
    )
    # turn-0: m_full = 620 // 64 = 9, partial_tail = 620 % 64 = 44.
    # User block_count = 9, len(tokens) = 9*64 + 44 = 620.
    #
    # turn k: LCP=3. prev_partial_tail = 620 % 64 = 44.
    # truncate at LCP=3: mid-segment cut on turn-0 user (kept_blocks=3) ->
    # block_count=3, len(tokens)=192. Trailing partial_tail/asst-overflow gone.
    # new_region = 2*64 + (320 mod 64) = 128 + 0 = 128 tokens.
    # out=80 -> asst_blocks = ceil(80/64) = 2 -> asst_tokens = 128.
    # user_blocks = 2 - 2 = 0 -> no user_k.
    r.advance_turn(
        prev_hash_ids=list(range(1, 11)),
        prev_in_tokens=620,
        prev_out_tokens=80,
        curr_hash_ids=[1, 2, 3, 99, 100],
        curr_in_tokens=320,
        seed="s1",
    )
    roles = [s.role for s in r._segments]
    assert roles == ["user", "assistant"]
    assert r._segments[0].block_count == 3
    assert r._segments[0].content_token_count == 192
    assert r._segments[1].content_token_count == 128
    assert r._segments[1].block_count == 2
    # Sum = 192 + 128 = 320 == curr_in_tokens.
    assert sum(len(s.tokens) for s in r._segments) == 320


def test_advance_asst_overflow_pattern_a_template_drift():
    """new_region < ceil(out[k-1]/bs)*bs: asst clamped to fit, user empty."""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2], in_tokens=128, tool_tokens=0, system_tokens=0, seed="s0"
    )
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=128,
        prev_out_tokens=200,
        curr_hash_ids=[1, 2, 3, 4],
        curr_in_tokens=256,
        seed="s1",
    )
    # new_region = 2*64 = 128 tokens. asst_blocks_target = ceil(200/64) = 4,
    # clamped to new_blocks_count = 2. asst_tokens = 128. user empty.
    roles = [s.role for s in r._segments]
    assert roles == ["user", "assistant"]
    assert r._segments[1].content_token_count == 128
    assert r._segments[1].block_count == 2


def test_advance_asst_overflow_pattern_c_deep_compaction():
    """Pattern C with new_region < ceil(out[k-1]/bs)*bs: asst clamped, no user_k."""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=list(range(1, 11)),
        in_tokens=620,
        tool_tokens=0,
        system_tokens=0,
        seed="s0",
    )
    r.advance_turn(
        prev_hash_ids=list(range(1, 11)),
        prev_in_tokens=620,
        prev_out_tokens=200,
        curr_hash_ids=[1, 99],
        curr_in_tokens=128,
        seed="s1",
    )
    # LCP=1, kept=1 block (64 tokens). new_region = 1*64 = 64 tokens.
    # asst_blocks_target = ceil(200/64) = 4, clamped to 1. asst_tokens=64.
    # user empty.
    roles = [s.role for s in r._segments]
    assert roles == ["user", "assistant"]
    assert r._segments[1].content_token_count == 64
    assert r._segments[1].block_count == 1


def test_advance_zero_out_skips_assistant_segment():
    """When out[k-1] is 0, no asst segment is emitted — only user_k."""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2], in_tokens=128, tool_tokens=0, system_tokens=0, seed="s0"
    )
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=128,
        prev_out_tokens=0,
        curr_hash_ids=[1, 2, 3],
        curr_in_tokens=192,
        seed="s1",
    )
    roles = [s.role for s in r._segments]
    assert roles == ["user", "user"]


def test_advance_zero_user_skips_user_segment():
    """When asst exactly fills new_region, no user_k segment emitted."""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2], in_tokens=128, tool_tokens=0, system_tokens=0, seed="s0"
    )
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=128,
        prev_out_tokens=64,
        curr_hash_ids=[1, 2, 3],
        curr_in_tokens=192,
        seed="s1",
    )
    # new_region = 1 block + 0 partial_tail = 64 tokens.
    # asst_blocks = ceil(64/64) = 1 -> asst_tokens = 64. user empty.
    roles = [s.role for s in r._segments]
    assert roles == ["user", "assistant"]


def test_advance_boundary_cut_strips_missing_block_overhang():
    """A boundary cut on the trailing segment strips its ENTIRE overhang past
    ``block_count * bs`` — missing-block synth tokens AND the partial tail.
    Stripping only the partial tail leaves stale synth tokens behind, so the
    rebuilt context exceeds ``curr_in_tokens`` and reset re-emissions drift.

    Shape: a truncated hash recording (hash_ids shorter than
    ``in_tokens // bs``) puts missing-block synth tokens on the trailing
    user segment; the next turn's pure-growth LCP cut lands exactly on its
    covered-block boundary."""
    r = _make_recon()
    # in=242, hash covers 2 of floor(242/64)=3 blocks -> user seg holds
    # 2*64 block tokens + (64 missing + 50 tail) = 242 tokens.
    r.init_turn_0(
        hash_ids=[1, 2], in_tokens=242, tool_tokens=0, system_tokens=0, seed="s0"
    )
    assert sum(len(s.tokens) for s in r._segments) == 242
    # Pure growth: LCP=2 is a boundary cut at the trailing user segment.
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=242,
        prev_out_tokens=0,
        curr_hash_ids=[1, 2, 3],
        curr_in_tokens=300,
        seed="s1",
    )
    assert sum(len(s.tokens) for s in r._segments) == 300
    # The surviving turn-0 segment holds exactly its covered block content.
    assert r._segments[0].tokens == _stub_decode_block_tokens([1, 2])


def test_advance_token_level_slicing_asst_user_split():
    """Block-aligned slicing puts the first asst_blocks*bs tokens in the
    assistant segment and the remaining new_region tokens in the user segment."""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2], in_tokens=128, tool_tokens=0, system_tokens=0, seed="s0"
    )
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=128,
        prev_out_tokens=100,
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=320,
        seed="s1",
    )
    # New-region tokens are decode_block_tokens([3, 4, 5]) (no partial tail
    # since 320 % 64 == 0). asst_blocks = ceil(100/64) = 2 -> 128 tokens.
    new_region = _stub_decode_block_tokens([3, 4, 5])
    assert r._segments[1].tokens == new_region[:128]
    assert r._segments[2].tokens == new_region[128:192]


# ---------------------------------------------------------------------------
# Byte-exact sum + hash-content stability
# ---------------------------------------------------------------------------


def test_byte_exact_sum_matches_recorded_init_turn_0():
    """sum(len(seg.tokens)) == in_tokens after init_turn_0 across various
    tool/sys/in combinations including edge cases that previously had
    block-rounding shortfall."""
    cases = [
        # (in, tool, sys, expected_sum)
        (200, 0, 0, 200),
        (192, 0, 0, 192),  # block-aligned
        (500, 100, 50, 500),  # multi-prefix from existing test
        (1000, 200, 200, 1000),
        (64, 0, 0, 64),
        (127, 0, 0, 127),
        (300, 0, 100, 300),
        (300, 100, 0, 300),
    ]
    for in_tokens, tool, sys_n, expected_sum in cases:
        bs = 64
        m_full = in_tokens // bs
        # Need enough hash_ids for the full block tile.
        hash_ids = list(range(1, m_full + 1)) if m_full > 0 else []
        r = _make_recon()
        r.init_turn_0(
            hash_ids=hash_ids,
            in_tokens=in_tokens,
            tool_tokens=tool,
            system_tokens=sys_n,
            seed=f"t:0:{in_tokens}",
        )
        actual_sum = sum(len(s.tokens) for s in r._segments)
        assert actual_sum == expected_sum, (
            f"in={in_tokens} tool={tool} sys={sys_n}: "
            f"sum={actual_sum} expected={expected_sum}"
        )


def test_byte_exact_sum_matches_recorded_advance_turn():
    """sum(len(seg.tokens)) == curr_in_tokens after advance_turn under all
    three structural patterns (clean append, mid-seq replace, pull-back)."""
    # Pattern A: clean append, in[k] = lcp*bs + new_region exactly.
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2], in_tokens=128, tool_tokens=0, system_tokens=0, seed="s0"
    )
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=128,
        prev_out_tokens=100,
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=320,
        seed="s1",
    )
    # turn-0 user kept (lcp=2 boundary cut, prev_partial_tail=0 -> no strip).
    assert sum(len(s.tokens) for s in r._segments) == 320

    # Pattern C: pull-back via mid-segment cut.
    r2 = _make_recon()
    r2.init_turn_0(
        hash_ids=list(range(1, 11)),
        in_tokens=640,  # 10 blocks * 64, no partial_tail
        tool_tokens=0,
        system_tokens=0,
        seed="s0",
    )
    r2.advance_turn(
        prev_hash_ids=list(range(1, 11)),
        prev_in_tokens=640,
        prev_out_tokens=80,
        curr_hash_ids=[1, 2, 3, 99, 100],
        curr_in_tokens=320,
        seed="s1",
    )
    # lcp=3, kept=3 blocks=192. new_region=2*64+0=128. asst=ceil(80/64)*64=128. user=0.
    # sum = 192 + 128 + 0 = 320.
    assert sum(len(s.tokens) for s in r2._segments) == 320

    # Pattern A with non-zero partial tail in the new turn.
    r3 = _make_recon()
    r3.init_turn_0(
        hash_ids=[1, 2], in_tokens=128, tool_tokens=0, system_tokens=0, seed="s0"
    )
    r3.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=128,
        prev_out_tokens=50,
        curr_hash_ids=[1, 2, 3, 4],
        curr_in_tokens=200,  # 3 full blocks + 8 partial_tail; hash 4 is partial
        seed="s1",
    )
    # lcp=2 boundary, prev_partial_tail=0 (128 % 64 = 0) -> turn-0 user kept (128).
    # m_curr=4, m_curr_full=200//64=3 -> m_curr_covered=3: the 4th hash is the
    # partial last block, so the new region is clamped to the covered budget.
    # new_region = (3-2) covered block * 64 + (200 % 64)=8 = 72 tokens.
    # asst = ceil(50/64)*64 = 64. user = 72 - 64 = 8.
    # sum = 128 + 64 + 8 = 200 == curr_in_tokens (exact byte-exact contract).
    assert sum(len(s.tokens) for s in r3._segments) == 200


def test_hash_content_stability_across_segments():
    """A given ``hash_id`` decodes to identical tokens across every segment
    it appears in. There is no BPE-stable terminator stamp on the trailing
    tokens — each cached block's tokens are emitted unmodified."""
    r = _make_recon()
    # turn 0: hash_ids = [1, 2, 3], block-aligned to 192 tokens (no partial_tail).
    r.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=192, tool_tokens=0, system_tokens=0, seed="t:0"
    )
    turn0_tokens = list(r._segments[0].tokens)
    # turn 1: hash_ids = [1, 2, 3, 4, 5], LCP=3 -> turn-0 user (3 blocks)
    # is preserved as-is (boundary cut, no partial_tail to strip).
    r.advance_turn(
        prev_hash_ids=[1, 2, 3],
        prev_in_tokens=192,
        prev_out_tokens=64,
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=320,
        seed="t:1",
    )
    # The first segment's tokens should be byte-identical to turn 0's user,
    # because LCP=3 means hashes [1,2,3] survive verbatim.
    assert r._segments[0].tokens == turn0_tokens
    # Independently, the underlying decode of [1, 2, 3] is what's stored —
    # no terminator overwrote any trailing tokens.
    assert r._segments[0].tokens == _stub_decode_block_tokens([1, 2, 3])


def test_hash_content_stability_terminator_field_unused():
    """Setting ``bpe_stable_terminator_tokens`` has no effect on emitted
    segment tokens — the reconstructor algorithm does not consume the field
    (no terminator stamp is applied; hash-content stability is preserved)."""
    r_no_term = _make_recon(terminator_tokens=[])
    r_no_term.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=192, tool_tokens=0, system_tokens=0, seed="t:0"
    )
    r_with_term = _make_recon(terminator_tokens=[99999])
    r_with_term.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=192, tool_tokens=0, system_tokens=0, seed="t:0"
    )
    # Same emitted tokens regardless of terminator field — the algorithm
    # ignores it.
    for s_no, s_yes in zip(r_no_term._segments, r_with_term._segments, strict=True):
        assert s_no.tokens == s_yes.tokens
        # Last token is the underlying block's last token, not 99999.
        assert s_yes.tokens[-1] != 99999


# ---------------------------------------------------------------------------
# Prefix-stability invariant: surviving segments are strict prefixes
# ---------------------------------------------------------------------------


def _snapshot_segments(recon):
    """Snapshot (role, block_start, tokens copy) for each segment. Identity
    by (role, block_start) lets us tell a surviving segment apart from a
    freshly appended one that happens to land at the same list index after
    upstream segments were dropped."""
    return [(seg.role, seg.block_start, list(seg.tokens)) for seg in recon._segments]


def _assert_prefix_stable(snapshot, recon):
    """For every old segment that still exists at the same list index with
    the same (role, block_start), its tokens must be a strict prefix of the
    old tokens. Old segments dropped entirely (replaced by freshly appended
    segments) are skipped — replacement is not prefix mutation, the index
    just rebinds. The invariant under test: nothing surviving from a prior
    turn ever has its prefix rewritten."""
    new_segs = recon._segments
    for i, (old_role, old_start, old_tokens) in enumerate(snapshot):
        if i >= len(new_segs):
            break
        new = new_segs[i]
        if new.role != old_role or new.block_start != old_start:
            # Different segment occupies this index now — old one was dropped.
            # All remaining indices are post-drop appends; stop checking.
            break
        new_tokens = new.tokens
        assert len(new_tokens) <= len(old_tokens), (
            f"segment {i} ({old_role}@{old_start}) grew from {len(old_tokens)} "
            f"to {len(new_tokens)} — prefix mutation"
        )
        assert new_tokens == old_tokens[: len(new_tokens)], (
            f"segment {i} ({old_role}@{old_start}) prefix mutated: "
            f"old[:{len(new_tokens)}] != new"
        )


def test_prefix_stability_pattern_a_clean_append():
    """Pattern A (LCP == M_prev): turn-0 segment must be byte-identical."""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=192, tool_tokens=0, system_tokens=0, seed="t:0"
    )
    snapshot = _snapshot_segments(r)

    r.advance_turn(
        prev_hash_ids=[1, 2, 3],
        prev_in_tokens=192,
        prev_out_tokens=64,
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=320,
        seed="t:1",
    )
    _assert_prefix_stable(snapshot, r)
    # Pattern A: append-only, turn-0 segment retained at full length.
    old_user_tokens = snapshot[0][2]
    assert r._segments[0].tokens == old_user_tokens
    assert len(r._segments[0].tokens) == len(old_user_tokens)
    # Two new segments appended (asst + user_k).
    assert len(r._segments) == 3


def test_prefix_stability_pattern_b_trailing_block_churn():
    """Pattern B (LCP == M_prev - 1): boundary segment shrinks to drop
    partial_tail; earlier segments byte-identical; later segments dropped."""
    r = _make_recon()
    # in=180 -> m_full=2, partial_tail=52. turn-0 user holds 180 tokens,
    # block_count=2.
    r.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=180, tool_tokens=0, system_tokens=0, seed="t:0"
    )
    snapshot = _snapshot_segments(r)
    assert len(snapshot[0][2]) == 180

    r.advance_turn(
        prev_hash_ids=[1, 2, 3],
        prev_in_tokens=180,
        prev_out_tokens=50,
        curr_hash_ids=[1, 2, 99, 100, 101],
        curr_in_tokens=300,
        seed="t:1",
    )
    _assert_prefix_stable(snapshot, r)
    # Boundary cut at LCP=2 with prev_partial_tail=52: turn-0 user shrinks
    # from 180 to 128 tokens (2 blocks * 64), strict prefix of original.
    old_user_tokens = snapshot[0][2]
    assert len(r._segments[0].tokens) == 128
    assert r._segments[0].tokens == old_user_tokens[:128]


def test_prefix_stability_pattern_c_deep_pull_back():
    """Pattern C (LCP < M_prev - 1, mid-segment cut): boundary segment
    suffix-truncated; earlier byte-identical; later dropped."""
    r = _make_recon()
    # turn-0: 10 blocks + 44 partial_tail = 620 tokens, all in one user segment.
    r.init_turn_0(
        hash_ids=list(range(1, 11)),
        in_tokens=620,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )
    snapshot = _snapshot_segments(r)
    assert len(snapshot[0][2]) == 620

    r.advance_turn(
        prev_hash_ids=list(range(1, 11)),
        prev_in_tokens=620,
        prev_out_tokens=80,
        curr_hash_ids=[1, 2, 3, 99, 100],
        curr_in_tokens=320,
        seed="t:1",
    )
    _assert_prefix_stable(snapshot, r)
    # Mid-segment cut at LCP=3 lands inside the single turn-0 user segment
    # (block_count=10). kept_blocks=3 -> 192 tokens, strict prefix.
    old_user_tokens = snapshot[0][2]
    assert len(r._segments[0].tokens) == 192
    assert r._segments[0].tokens == old_user_tokens[:192]


def test_prefix_stability_sweep_multi_turn():
    """Chain advances exercising A -> B -> C -> A -> C and assert
    prefix-stability on every step. Distinct hash_ids per block ensure any
    prefix mutation surfaces immediately via the hash-keyed token IDs in
    ``_stub_decode_block_tokens``."""
    r = _make_recon()

    # Turn 0: seed with 5 blocks + 32 partial_tail = 352 tokens.
    r.init_turn_0(
        hash_ids=[10, 11, 12, 13, 14],
        in_tokens=352,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )

    # Turn 1: Pattern A. LCP=5, append 3 new blocks. prev_partial_tail=32,
    # boundary cut at end of turn-0 user strips the 32 tail tokens.
    snapshot = _snapshot_segments(r)
    r.advance_turn(
        prev_hash_ids=[10, 11, 12, 13, 14],
        prev_in_tokens=352,
        prev_out_tokens=64,
        curr_hash_ids=[10, 11, 12, 13, 14, 20, 21, 22],
        curr_in_tokens=512,
        seed="t:1",
    )
    _assert_prefix_stable(snapshot, r)

    # Turn 2: Pattern B. LCP=7, last block of prev (22) churned to 30.
    snapshot = _snapshot_segments(r)
    r.advance_turn(
        prev_hash_ids=[10, 11, 12, 13, 14, 20, 21, 22],
        prev_in_tokens=512,
        prev_out_tokens=64,
        curr_hash_ids=[10, 11, 12, 13, 14, 20, 21, 30, 31],
        curr_in_tokens=576,
        seed="t:2",
    )
    _assert_prefix_stable(snapshot, r)

    # Turn 3: Pattern C. LCP=3, deep pull-back into turn-0 user.
    snapshot = _snapshot_segments(r)
    r.advance_turn(
        prev_hash_ids=[10, 11, 12, 13, 14, 20, 21, 30, 31],
        prev_in_tokens=576,
        prev_out_tokens=80,
        curr_hash_ids=[10, 11, 12, 40, 41, 42],
        curr_in_tokens=384,
        seed="t:3",
    )
    _assert_prefix_stable(snapshot, r)

    # Turn 4: Pattern A again. LCP=6 (full M_prev), append 2 blocks.
    # prev_in=384 % 64 = 0, no partial_tail to strip.
    snapshot = _snapshot_segments(r)
    r.advance_turn(
        prev_hash_ids=[10, 11, 12, 40, 41, 42],
        prev_in_tokens=384,
        prev_out_tokens=100,
        curr_hash_ids=[10, 11, 12, 40, 41, 42, 50, 51],
        curr_in_tokens=512,
        seed="t:4",
    )
    _assert_prefix_stable(snapshot, r)

    # Turn 5: Pattern C again. LCP=2, hits the very first turn-0 hash block
    # group. Confirms repeat pull-back stays prefix-stable.
    snapshot = _snapshot_segments(r)
    r.advance_turn(
        prev_hash_ids=[10, 11, 12, 40, 41, 42, 50, 51],
        prev_in_tokens=512,
        prev_out_tokens=64,
        curr_hash_ids=[10, 11, 60, 61],
        curr_in_tokens=256,
        seed="t:5",
    )
    _assert_prefix_stable(snapshot, r)
    # First segment must still hold hash-block [10] decode (block 0 of
    # original turn-0 user) byte-identically — confirms hash content
    # for hash_id=10 was never mutated across 5 advances.
    block_10_tokens = _stub_decode_block_tokens([10])
    assert r._segments[0].tokens[:64] == block_10_tokens


def sentinel_count(tokens):
    return sum(1 for t in tokens if t == -1)


def test_init_turn_0_with_truncated_hash_ids_synthesizes_tail():
    """When len(hash_ids) < floor(in_tokens/bs), the missing region is
    synthesized as additional partial-tail tokens on the trailing user
    segment. The reconstructor must NOT raise.
    Total tokens emitted must equal in_tokens.
    """
    bs = 64
    in_tokens = 1000  # floor(1000/64) = 15 blocks needed, partial tail = 40
    # Provide only 10 hash_ids — short by 5 blocks (320 tokens) of the block tile.
    hash_ids = list(range(100, 110))

    decoded_block_calls: list[list[int]] = []

    def decode_block_tokens(hids):
        decoded_block_calls.append(list(hids))
        return [hids[0] if hids else 0] * (len(hids) * bs)

    def sample_partial_tail_tokens(n, seed):
        return [-1] * n  # sentinel for synth-tail tokens

    recon = ConversationReconstructor(
        block_size=bs,
        decode_block_tokens=decode_block_tokens,
        sample_partial_tail_tokens=sample_partial_tail_tokens,
        decode_tokens_to_text=lambda toks: f"t{len(toks)}",
        bpe_stable_terminator_tokens=[],
    )

    # MUST NOT raise.
    recon.init_turn_0(
        hash_ids=hash_ids,
        in_tokens=in_tokens,
        tool_tokens=0,
        system_tokens=0,
        seed="seed",
    )

    # Total tokens across all segments must equal in_tokens.
    total = sum(len(seg.tokens) for seg in recon._segments)
    assert total == in_tokens, (
        f"reconstructed total {total} != in_tokens {in_tokens}; "
        f"the relaxed validator must fill the gap with synth-tail tokens"
    )

    # The user segment carries the synth-tail tokens (sentinel value -1)
    # AS WELL AS the decoded block tokens.
    user_seg = next(s for s in recon._segments if s.role == "user")
    sentinel_n = sum(1 for t in user_seg.tokens if t == -1)
    expected_synth_tokens = (15 - 10) * bs + 40  # 5 missing blocks + partial tail = 360
    assert sentinel_n == expected_synth_tokens, (
        f"user segment should carry {expected_synth_tokens} synth-tail "
        f"sentinel tokens, got {sentinel_n}"
    )


def test_init_turn_0_with_truncated_hash_ids_and_system_prefix_synthesizes_user_tail():
    """When tool_tokens + system_tokens consume the first N blocks AND hash_ids
    is still long enough to cover those, the user segment's synth tail handles
    only the post-system gap.
    """
    bs = 64
    tool_tokens = 64  # 1 block of system prefix
    system_tokens = 64  # 1 more block of system prefix
    # in_tokens=1000, bs=64 -> 15 blocks needed (+ 40 partial). System consumes 2.
    in_tokens = 1000
    # Provide 5 hash_ids: 2 for system, 3 for user. Short by 10 blocks (640 tokens).
    hash_ids = list(range(100, 105))

    def decode_block_tokens(hids):
        return [0] * (len(hids) * bs)

    def sample_partial_tail_tokens(n, seed):
        return [-1] * n

    recon = ConversationReconstructor(
        block_size=bs,
        decode_block_tokens=decode_block_tokens,
        sample_partial_tail_tokens=sample_partial_tail_tokens,
        decode_tokens_to_text=lambda toks: f"t{len(toks)}",
        bpe_stable_terminator_tokens=[],
    )

    recon.init_turn_0(
        hash_ids=hash_ids,
        in_tokens=in_tokens,
        tool_tokens=tool_tokens,
        system_tokens=system_tokens,
        seed="seed",
    )

    # Total tokens == in_tokens.
    total = sum(len(seg.tokens) for seg in recon._segments)
    assert total == in_tokens

    # System segment carries 2 blocks of decoded tokens (no synth).
    sys_seg = next((s for s in recon._segments if s.role == "system"), None)
    assert sys_seg is not None
    assert len(sys_seg.tokens) == 2 * bs
    assert sentinel_count(sys_seg.tokens) == 0, (
        "system segment must not contain synth tokens"
    )

    # User segment carries the rest.
    user_seg = next(s for s in recon._segments if s.role == "user")
    expected_user_tokens = in_tokens - 2 * bs  # 872
    assert len(user_seg.tokens) == expected_user_tokens


def test_init_turn_0_system_prefix_exceeding_hash_ids_still_raises():
    """If even the system+tool prefix can't be filled from hash_ids,
    the loader should still error — synthesizing the SYSTEM segment from
    random tokens would silently corrupt the prefix cache.
    """
    bs = 64
    tool_tokens = 128
    system_tokens = 128  # 4 blocks of system prefix
    # Only 2 hash_ids — can't even fill the system prefix.
    hash_ids = [100, 200]
    in_tokens = 1000

    recon = ConversationReconstructor(
        block_size=bs,
        decode_block_tokens=lambda hids: [0] * (len(hids) * bs),
        sample_partial_tail_tokens=lambda n, seed: [-1] * n,
        decode_tokens_to_text=lambda toks: "",
        bpe_stable_terminator_tokens=[],
    )

    with pytest.raises(ValueError, match="system prefix"):
        recon.init_turn_0(
            hash_ids=hash_ids,
            in_tokens=in_tokens,
            tool_tokens=tool_tokens,
            system_tokens=system_tokens,
            seed="seed",
        )


def test_advance_turn_with_truncated_curr_hash_ids_synthesizes_tail():
    """When ``len(curr_hash_ids) * bs < curr_in_tokens``, advance_turn must
    synthesize the missing-block region as additional partial-tail tokens
    so the final synth_buf state has exactly curr_in_tokens tokens (less
    the prev_out_tokens that went to the assistant segment).
    """
    bs = 64
    # Turn-0 baseline: 5 hash_ids fully covering in_tokens=320 (5*64=320, no partial tail).
    turn0_hash_ids = list(range(1, 6))
    turn0_in_tokens = 320

    # Turn-1 has prev_out=128 (2 blocks of assistant) and curr_in_tokens=960
    # (15 full blocks). curr_hash_ids is TRUNCATED — only 10 hash_ids
    # (covering 640 tokens) instead of the expected 15 (960 tokens).
    # The first 5 hash_ids equal turn0_hash_ids (LCP=5 — the prior user
    # turn's blocks are preserved). The next 5 are new.
    curr_hash_ids = turn0_hash_ids + list(range(6, 11))
    curr_in_tokens = 960
    prev_out_tokens = 128

    def decode_block_tokens(hids):
        return [hids[0] if hids else 0] * (len(hids) * bs)

    def sample_partial_tail_tokens(n, seed):
        return [-1] * n

    recon = ConversationReconstructor(
        block_size=bs,
        decode_block_tokens=decode_block_tokens,
        sample_partial_tail_tokens=sample_partial_tail_tokens,
        decode_tokens_to_text=lambda toks: f"t{len(toks)}",
        bpe_stable_terminator_tokens=[],
    )

    recon.init_turn_0(
        hash_ids=turn0_hash_ids,
        in_tokens=turn0_in_tokens,
        tool_tokens=0,
        system_tokens=0,
        seed="s0",
    )
    # Sanity: 320 tokens, 5 blocks.
    assert sum(len(s.tokens) for s in recon._segments) == turn0_in_tokens

    recon.advance_turn(
        prev_hash_ids=turn0_hash_ids,
        prev_in_tokens=turn0_in_tokens,
        prev_out_tokens=prev_out_tokens,
        curr_hash_ids=curr_hash_ids,
        curr_in_tokens=curr_in_tokens,
        seed="s1",
    )

    # Expected total tokens after advance: curr_in_tokens (960).
    total = sum(len(s.tokens) for s in recon._segments)
    assert total == curr_in_tokens, (
        f"after advance_turn with truncated curr_hash_ids, total tokens "
        f"= {total}; expected {curr_in_tokens}. The missing-block region "
        f"must be synthesized as additional tail tokens."
    )

    # Sentinel count: 5 truncated blocks * 64 = 320 sentinel tokens
    # synthesized on the trailing user segment. (No partial tail beyond
    # block alignment: 960 % 64 == 0.)
    all_tokens = [t for s in recon._segments for t in s.tokens]
    sentinel_n = sum(1 for t in all_tokens if t == -1)
    expected_sentinel = (15 - 10) * bs
    assert sentinel_n == expected_sentinel, (
        f"expected {expected_sentinel} synth-tail sentinels, got {sentinel_n}"
    )


def test_advance_turn_with_full_curr_hash_ids_unchanged():
    """Regression guard: when curr_hash_ids fully covers curr_in_tokens (no
    truncation), advance_turn behavior is byte-identical to today's logic —
    no synth-tail tokens are appended for the missing-block region (because
    there is none)."""
    bs = 64
    turn0_hash_ids = list(range(1, 6))
    turn0_in_tokens = 320
    # Fully covered: 15 hash_ids * 64 = 960 tokens.
    curr_hash_ids = turn0_hash_ids + list(range(6, 16))
    curr_in_tokens = 960
    prev_out_tokens = 128

    recon = ConversationReconstructor(
        block_size=bs,
        decode_block_tokens=lambda hids: [hids[0] if hids else 0] * (len(hids) * bs),
        sample_partial_tail_tokens=lambda n, seed: [-1] * n,
        decode_tokens_to_text=lambda toks: f"t{len(toks)}",
        bpe_stable_terminator_tokens=[],
    )

    recon.init_turn_0(
        hash_ids=turn0_hash_ids,
        in_tokens=turn0_in_tokens,
        tool_tokens=0,
        system_tokens=0,
        seed="s0",
    )
    recon.advance_turn(
        prev_hash_ids=turn0_hash_ids,
        prev_in_tokens=turn0_in_tokens,
        prev_out_tokens=prev_out_tokens,
        curr_hash_ids=curr_hash_ids,
        curr_in_tokens=curr_in_tokens,
        seed="s1",
    )

    total = sum(len(s.tokens) for s in recon._segments)
    assert total == curr_in_tokens
    all_tokens = [t for s in recon._segments for t in s.tokens]
    sentinel_n = sum(1 for t in all_tokens if t == -1)
    assert sentinel_n == 0, (
        f"non-truncated curr_hash_ids must NOT produce sentinel tokens; got {sentinel_n}"
    )


def test_advance_turn_partial_last_hashed_block_clamps_to_budget():
    """Regression: a hashed-but-partial last block (len(curr_hash_ids) >
    curr_in_tokens // bs) must clamp to the covered-block budget instead of
    decoding the partial block as full AND appending the partial tail.

    This mirrors ``init_turn_0``'s ``covered_blocks = min(m_full,
    len(hash_ids))`` clamp. Before the fix, ``_advance_to_turn`` used
    ``m_curr = len(curr_hash_ids)`` unclamped, so a turn like in=250 with
    hash_ids=[..,4] at bs=64 emitted block 4 as a full 64-token block AND an
    extra ``250 % 64 = 58`` synth tail -- overshooting curr_in_tokens by ~bs
    and breaking the byte-exact ``sum(seg.tokens) == in_tokens`` contract
    (this is exactly the shape in tests/fixtures/weka_traces/simple.json
    turn 1, which the byte-exact ISL drift contract enforces).
    """
    r = _make_recon()
    # turn 0: in=200, hash_ids=[1,2,3] (3 full blocks + 8 partial tail).
    r.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=200, tool_tokens=0, system_tokens=0, seed="t:0"
    )
    assert sum(len(s.tokens) for s in r._segments) == 200
    # turn 1: in=250, hash_ids=[1,2,3,4]; m_curr_full = 250 // 64 = 3 < 4, so
    # hash 4 is a partial last block contributing only 250 % 64 = 58 tokens.
    r.advance_turn(
        prev_hash_ids=[1, 2, 3],
        prev_in_tokens=200,
        prev_out_tokens=30,
        curr_hash_ids=[1, 2, 3, 4],
        curr_in_tokens=250,
        seed="t:1",
    )
    # Byte-exact: the reconstructed prefix is exactly the recorded input length,
    # not 250 + bs.
    assert sum(len(s.tokens) for s in r._segments) == 250
    assert all(s.block_count >= 0 for s in r._segments)
