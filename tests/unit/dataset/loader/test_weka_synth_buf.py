# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the byte-exact weka conversation reconstructor."""

import math

import pytest
from pytest import param

from aiperf.dataset.loader.weka_synth_buf import (
    ConversationReconstructor,
    RoleSegment,
    compute_asst_block_caps,
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
    r.init_turn_0(
        hash_ids=list(range(1, 8)),
        in_tokens=500,
        tool_tokens=100,
        system_tokens=50,
        seed="t:0",
    )
    roles = [s.role for s in r._segments]
    assert roles == ["system", "user"]
    assert r._segments[0].content_token_count == 192
    assert r._segments[1].content_token_count == 308
    assert r._segments[0].tokens == _stub_decode_block_tokens([1, 2, 3])
    for seg in r._segments:
        assert len(seg.tokens) == seg.content_token_count
    assert sum(len(s.tokens) for s in r._segments) == 500


def test_init_turn_0_prefix_block_rounding_overshoot_clamps_to_budget():
    """Regression: a declared prefix whose BLOCK count exceeds the prompt's own"""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=170, tool_tokens=130, system_tokens=0, seed="t:0"
    )
    segs = r._segments
    assert all(s.block_count >= 0 for s in segs), [s.block_count for s in segs]
    sys_seg = next(s for s in segs if s.role == "system")
    assert sys_seg.block_count == 2
    assert sum(len(s.tokens) for s in segs) == 170


def test_init_turn_0_prefix_exceeding_input_tokens_clamps_to_budget():
    """Regression: a prefix that outright exceeds the whole turn-0 input must"""
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
    expected_tail = _stub_partial_tail_tokens(8, "t:0")
    user_tokens = r._segments[0].tokens
    assert user_tokens[-8:] == expected_tail


def test_init_turn_0_zero_partial_tail_no_tail_marker():
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=192, tool_tokens=0, system_tokens=0, seed="t:0"
    )
    expected = _stub_decode_block_tokens([1, 2, 3])
    assert r._segments[0].tokens == expected


def test_init_turn_0_combines_tool_and_system_into_single_system():
    """tool+system must emit exactly ONE role="system" segment."""
    bs = 64
    in_tokens = 1000
    tool_tokens = 200
    system_tokens = 300
    m_full = in_tokens // bs
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
    assert roles.count("system") == 1
    assert roles == ["system", "user"]
    sys_seg = r._segments[0]
    expected_prefix_blocks = math.ceil((tool_tokens + system_tokens) / bs)
    assert sys_seg.block_count == expected_prefix_blocks
    assert len(sys_seg.tokens) == expected_prefix_blocks * bs
    assert sys_seg.block_start == 0
    assert sum(len(s.tokens) for s in r._segments) == in_tokens


def test_role_segment_invariants():
    seg = RoleSegment(
        role="user",
        block_start=0,
        block_count=3,
        tokens=list(range(180)),
        content="abc",
    )
    assert seg.content_token_count == 180
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


@pytest.mark.parametrize(
    "cases",
    [
        param([([1, 2, 3], [1, 2, 3], 3)], id="identical_lists"),
        param([([], [], 0), ([], [1], 0), ([1], [], 0)], id="empty"),
        param(
            [([1, 2, 3], [1, 2, 3, 4, 5], 3), ([1, 2, 3, 4, 5], [1, 2, 3], 3)],
            id="prefix_extension",
        ),
        param([([1, 2, 3], [4, 5, 6], 0)], id="divergence_at_first_position"),
        param([([1, 2, 3, 4], [1, 2, 3, 5, 6], 3)], id="mid_sequence_replacement"),
    ],
)  # fmt: skip
def test_lcp(cases):
    """longest_common_prefix over identical, empty, extension, and churn shapes."""
    for a, b, expected in cases:
        assert longest_common_prefix(a, b) == expected


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
    truncate_synth_buf_at_block(
        segs,
        target_blocks=4,
        block_size=64,
        decode_tokens_to_text=_stub_decode_tokens_to_text,
    )
    assert [s.role for s in segs] == ["system", "user"]
    user = segs[1]
    assert user.block_count == 2
    assert user.content_token_count == 128
    assert len(user.tokens) == 128
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
    """At a boundary cut, the trailing ``prev_partial_tail`` tokens are"""
    bs = 64
    block_count = 1
    partial_tail = 36
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
    assert sum(len(s.tokens) for s in r._segments) == 320


def test_advance_pattern_b_trailing_block_churn():
    """LCP == M_prev - 1 (trailing-block recomposition)."""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=180, tool_tokens=0, system_tokens=0, seed="s0"
    )
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
    assert r._segments[0].block_count == 2
    assert r._segments[0].content_token_count == 128
    assert r._segments[1].content_token_count == 64
    assert r._segments[1].block_count == 1
    assert r._segments[2].content_token_count == 108
    assert r._segments[2].block_count == 1
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
    r.advance_turn(
        prev_hash_ids=list(range(1, 11)),
        prev_in_tokens=620,
        prev_out_tokens=80,
        curr_hash_ids=[1, 2, 3, 99, 100],
        curr_in_tokens=320,
        seed="s1",
    )
    roles = [s.role for s in r._segments]
    assert roles == ["user", "assistant", "user"]
    assert r._segments[0].block_count == 3
    assert r._segments[0].content_token_count == 192
    assert r._segments[1].content_token_count == 64
    assert r._segments[1].block_count == 1
    assert r._segments[2].content_token_count == 64
    assert r._segments[2].block_count == 1
    assert sum(len(s.tokens) for s in r._segments) == 320


def test_advance_asst_overflow_pattern_a_template_drift():
    """new_region < ceil(out[k-1]/bs)*bs: asst clamps to the region, but the"""
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
    roles = [s.role for s in r._segments]
    assert roles == ["user", "assistant", "user"]
    assert r._segments[1].content_token_count == 64
    assert r._segments[1].block_count == 1
    assert r._segments[2].content_token_count == 64
    assert r._segments[2].block_count == 1


def test_advance_asst_overflow_pattern_c_deep_compaction():
    """Pattern C, single-block tail-free region: the lone new block must seed"""
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
    roles = [s.role for s in r._segments]
    assert roles == ["user", "user"]
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


def test_advance_asst_exactly_fills_region_yields_trailing_user():
    """When the assistant target exactly equals a tail-free new region, the"""
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
    roles = [s.role for s in r._segments]
    assert roles == ["user", "user"]


def test_advance_boundary_cut_strips_missing_block_overhang():
    """A boundary cut on the trailing segment strips its ENTIRE overhang past"""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2], in_tokens=242, tool_tokens=0, system_tokens=0, seed="s0"
    )
    assert sum(len(s.tokens) for s in r._segments) == 242
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=242,
        prev_out_tokens=0,
        curr_hash_ids=[1, 2, 3],
        curr_in_tokens=300,
        seed="s1",
    )
    assert sum(len(s.tokens) for s in r._segments) == 300
    assert r._segments[0].tokens == _stub_decode_block_tokens([1, 2])


def test_advance_token_level_slicing_asst_user_split():
    """Block-aligned slicing puts the first asst_blocks*bs tokens in the"""
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
    new_region = _stub_decode_block_tokens([3, 4, 5])
    assert r._segments[1].tokens == new_region[:128]
    assert r._segments[2].tokens == new_region[128:192]


def test_byte_exact_sum_matches_recorded_init_turn_0():
    """sum(len(seg.tokens)) == in_tokens after init_turn_0 across various"""
    cases = [
        (200, 0, 0, 200),
        (192, 0, 0, 192),
        (500, 100, 50, 500),
        (1000, 200, 200, 1000),
        (64, 0, 0, 64),
        (127, 0, 0, 127),
        (300, 0, 100, 300),
        (300, 100, 0, 300),
    ]
    for in_tokens, tool, sys_n, expected_sum in cases:
        bs = 64
        m_full = in_tokens // bs
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
    """sum(len(seg.tokens)) == curr_in_tokens after advance_turn under all"""
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
    assert sum(len(s.tokens) for s in r._segments) == 320

    r2 = _make_recon()
    r2.init_turn_0(
        hash_ids=list(range(1, 11)),
        in_tokens=640,
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
    assert sum(len(s.tokens) for s in r2._segments) == 320

    r3 = _make_recon()
    r3.init_turn_0(
        hash_ids=[1, 2], in_tokens=128, tool_tokens=0, system_tokens=0, seed="s0"
    )
    r3.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=128,
        prev_out_tokens=50,
        curr_hash_ids=[1, 2, 3, 4],
        curr_in_tokens=200,
        seed="s1",
    )
    assert sum(len(s.tokens) for s in r3._segments) == 200


def test_hash_content_stability_across_segments():
    """A given ``hash_id`` decodes to identical tokens across every segment"""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=192, tool_tokens=0, system_tokens=0, seed="t:0"
    )
    turn0_tokens = list(r._segments[0].tokens)
    r.advance_turn(
        prev_hash_ids=[1, 2, 3],
        prev_in_tokens=192,
        prev_out_tokens=64,
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=320,
        seed="t:1",
    )
    assert r._segments[0].tokens == turn0_tokens
    assert r._segments[0].tokens == _stub_decode_block_tokens([1, 2, 3])


def test_hash_content_stability_terminator_field_unused():
    """Setting ``bpe_stable_terminator_tokens`` has no effect on emitted"""
    r_no_term = _make_recon(terminator_tokens=[])
    r_no_term.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=192, tool_tokens=0, system_tokens=0, seed="t:0"
    )
    r_with_term = _make_recon(terminator_tokens=[99999])
    r_with_term.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=192, tool_tokens=0, system_tokens=0, seed="t:0"
    )
    for s_no, s_yes in zip(r_no_term._segments, r_with_term._segments, strict=True):
        assert s_no.tokens == s_yes.tokens
        assert s_yes.tokens[-1] != 99999


def _snapshot_segments(recon):
    """Snapshot (role, block_start, tokens copy) for each segment. Identity"""
    return [(seg.role, seg.block_start, list(seg.tokens)) for seg in recon._segments]


def _assert_prefix_stable(snapshot, recon):
    """For every old segment that still exists at the same list index with"""
    new_segs = recon._segments
    for i, (old_role, old_start, old_tokens) in enumerate(snapshot):
        if i >= len(new_segs):
            break
        new = new_segs[i]
        if new.role != old_role or new.block_start != old_start:
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
    old_user_tokens = snapshot[0][2]
    assert r._segments[0].tokens == old_user_tokens
    assert len(r._segments[0].tokens) == len(old_user_tokens)
    assert len(r._segments) == 3


def test_prefix_stability_pattern_b_trailing_block_churn():
    """Pattern B (LCP == M_prev - 1): boundary segment shrinks to drop"""
    r = _make_recon()
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
    old_user_tokens = snapshot[0][2]
    assert len(r._segments[0].tokens) == 128
    assert r._segments[0].tokens == old_user_tokens[:128]


def test_prefix_stability_pattern_c_deep_pull_back():
    """Pattern C (LCP < M_prev - 1, mid-segment cut): boundary segment"""
    r = _make_recon()
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
    old_user_tokens = snapshot[0][2]
    assert len(r._segments[0].tokens) == 192
    assert r._segments[0].tokens == old_user_tokens[:192]


def test_prefix_stability_sweep_multi_turn():
    """Chain advances exercising A -> B -> C -> A -> C and assert"""
    r = _make_recon()

    r.init_turn_0(
        hash_ids=[10, 11, 12, 13, 14],
        in_tokens=352,
        tool_tokens=0,
        system_tokens=0,
        seed="t:0",
    )

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
    block_10_tokens = _stub_decode_block_tokens([10])
    assert r._segments[0].tokens[:64] == block_10_tokens


def sentinel_count(tokens):
    return sum(1 for t in tokens if t == -1)


def test_init_turn_0_with_truncated_hash_ids_synthesizes_tail():
    """When len(hash_ids) < floor(in_tokens/bs), the missing region is"""
    bs = 64
    in_tokens = 1000
    hash_ids = list(range(100, 110))

    decoded_block_calls: list[list[int]] = []

    def decode_block_tokens(hids):
        decoded_block_calls.append(list(hids))
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
        hash_ids=hash_ids,
        in_tokens=in_tokens,
        tool_tokens=0,
        system_tokens=0,
        seed="seed",
    )

    total = sum(len(seg.tokens) for seg in recon._segments)
    assert total == in_tokens, (
        f"reconstructed total {total} != in_tokens {in_tokens}; "
        f"the relaxed validator must fill the gap with synth-tail tokens"
    )

    user_seg = next(s for s in recon._segments if s.role == "user")
    sentinel_n = sum(1 for t in user_seg.tokens if t == -1)
    expected_synth_tokens = (15 - 10) * bs + 40
    assert sentinel_n == expected_synth_tokens, (
        f"user segment should carry {expected_synth_tokens} synth-tail "
        f"sentinel tokens, got {sentinel_n}"
    )


def test_init_turn_0_with_truncated_hash_ids_and_system_prefix_synthesizes_user_tail():
    """When tool_tokens + system_tokens consume the first N blocks AND hash_ids"""
    bs = 64
    tool_tokens = 64
    system_tokens = 64
    in_tokens = 1000
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

    total = sum(len(seg.tokens) for seg in recon._segments)
    assert total == in_tokens

    sys_seg = next((s for s in recon._segments if s.role == "system"), None)
    assert sys_seg is not None
    assert len(sys_seg.tokens) == 2 * bs
    assert sentinel_count(sys_seg.tokens) == 0, (
        "system segment must not contain synth tokens"
    )

    user_seg = next(s for s in recon._segments if s.role == "user")
    expected_user_tokens = in_tokens - 2 * bs
    assert len(user_seg.tokens) == expected_user_tokens


def test_init_turn_0_system_prefix_exceeding_hash_ids_still_raises():
    """If even the system+tool prefix can't be filled from hash_ids,"""
    bs = 64
    tool_tokens = 128
    system_tokens = 128
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
    """When ``len(curr_hash_ids) * bs < curr_in_tokens``, advance_turn must"""
    bs = 64
    turn0_hash_ids = list(range(1, 6))
    turn0_in_tokens = 320

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
    assert sum(len(s.tokens) for s in recon._segments) == turn0_in_tokens

    recon.advance_turn(
        prev_hash_ids=turn0_hash_ids,
        prev_in_tokens=turn0_in_tokens,
        prev_out_tokens=prev_out_tokens,
        curr_hash_ids=curr_hash_ids,
        curr_in_tokens=curr_in_tokens,
        seed="s1",
    )

    total = sum(len(s.tokens) for s in recon._segments)
    assert total == curr_in_tokens, (
        f"after advance_turn with truncated curr_hash_ids, total tokens "
        f"= {total}; expected {curr_in_tokens}. The missing-block region "
        f"must be synthesized as additional tail tokens."
    )

    all_tokens = [t for s in recon._segments for t in s.tokens]
    sentinel_n = sum(1 for t in all_tokens if t == -1)
    expected_sentinel = (15 - 10) * bs
    assert sentinel_n == expected_sentinel, (
        f"expected {expected_sentinel} synth-tail sentinels, got {sentinel_n}"
    )


def test_advance_turn_with_full_curr_hash_ids_unchanged():
    """Regression guard: when curr_hash_ids fully covers curr_in_tokens (no"""
    bs = 64
    turn0_hash_ids = list(range(1, 6))
    turn0_in_tokens = 320
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
    """Regression: a hashed-but-partial last block (len(curr_hash_ids) >"""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2, 3], in_tokens=200, tool_tokens=0, system_tokens=0, seed="t:0"
    )
    assert sum(len(s.tokens) for s in r._segments) == 200
    r.advance_turn(
        prev_hash_ids=[1, 2, 3],
        prev_in_tokens=200,
        prev_out_tokens=30,
        curr_hash_ids=[1, 2, 3, 4],
        curr_in_tokens=250,
        seed="t:1",
    )
    assert sum(len(s.tokens) for s in r._segments) == 250
    assert all(s.block_count >= 0 for s in r._segments)


@pytest.mark.parametrize(
    ("prev_out_tokens", "curr_hash_ids", "curr_in_tokens"),
    [
        (128, [1, 2, 3, 4], 256),
        (500, [1, 2, 3, 4], 256),
        (200, [1, 2, 3], 192),
        (300, [1, 2, 3, 4, 5], 320),
    ],
)
def test_advance_always_ends_with_user_segment(
    prev_out_tokens, curr_hash_ids, curr_in_tokens
):
    """Wire invariant: every turn that adds new content ends with a user"""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2], in_tokens=128, tool_tokens=0, system_tokens=0, seed="s0"
    )
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=128,
        prev_out_tokens=prev_out_tokens,
        curr_hash_ids=curr_hash_ids,
        curr_in_tokens=curr_in_tokens,
        seed="s1",
    )
    assert r._segments[-1].role == "user"
    assert not r._trailing_non_user_turns
    assert sum(len(s.tokens) for s in r._segments) == curr_in_tokens


def test_advance_zero_new_region_records_trailing_non_user_caveat():
    """The one shape that cannot end with a user: a fully block-aligned"""
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
    assert [s.role for s in r._segments] == ["user", "assistant", "user"]
    assert not r._trailing_non_user_turns
    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4],
        prev_in_tokens=256,
        prev_out_tokens=64,
        curr_hash_ids=[1, 2, 3],
        curr_in_tokens=192,
        seed="s2",
    )
    assert [s.role for s in r._segments] == ["user", "assistant"]
    assert r._trailing_non_user_turns == [2]
    assert sum(len(s.tokens) for s in r._segments) == 192


def test_init_turn_0_system_only_prompt_records_caveat():
    """A turn-0 prompt entirely consumed by the cached tool/system prefix has"""
    r = _make_recon()
    r.init_turn_0(
        hash_ids=[1, 2],
        in_tokens=128,
        tool_tokens=128,
        system_tokens=0,
        seed="s0",
    )
    assert [s.role for s in r._segments] == ["system"]
    assert r._trailing_non_user_turns == [0]


def test_compute_caps_canonical_degenerate():
    """The canonical pull-back: turn 2 truncates onto the assistant block that"""
    caps = compute_asst_block_caps(
        [([1, 2], 128), ([1, 2, 3, 4], 256), ([1, 2, 3], 192)],
        64,
    )
    assert caps == [None, 0, None]


def test_compute_caps_clean_append_no_constraints():
    """A pure-growth conversation has no degenerate pull-backs -> no caps."""
    caps = compute_asst_block_caps(
        [([1, 2], 128), ([1, 2, 3, 4, 5], 320)],
        64,
    )
    assert caps == [None, None]


def test_compute_caps_target_owned_by_turn_0_no_cap():
    """A pull-back landing on a block created by turn 0 needs no cap (turn 0"""
    caps = compute_asst_block_caps(
        [([1, 2], 128), ([1, 2, 3, 4], 256), ([1, 2], 128)],
        64,
    )
    assert caps == [None, None, None]


def test_compute_caps_two_targets_same_owner_takes_min():
    """Two later degenerate pull-backs landing inside the same turn's assistant"""
    caps = compute_asst_block_caps(
        [
            ([1, 2], 128),
            ([1, 2, 3, 4, 5, 6], 384),
            ([1, 2, 3, 4, 5], 320),
            ([1, 2, 3, 4], 256),
        ],
        64,
    )
    assert caps[1] == 1
    assert caps[0] is None


def test_compute_caps_overcovered_prefix_clamps_no_indexerror():
    """lcp can exceed the current turn's covered-block count when the recorder"""
    caps = compute_asst_block_caps(
        [([1, 2, 3, 4], 256), ([1, 2, 3, 4], 128), ([1, 2], 128)],
        64,
    )
    assert len(caps) == 3
    assert caps[2] is None


def test_compute_caps_partial_last_hashed_block_uses_covered_budget():
    """end_k must use min(len(hash_ids), in_tokens // bs): a partial last hashed"""
    caps = compute_asst_block_caps(
        [([1, 2, 3], 192), ([1, 2, 3, 4], 250)],
        64,
    )
    assert caps == [None, None]


def _run_canonical_three_turns(caps):
    """Drive the canonical degenerate 3-turn sequence, applying per-turn caps."""
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
        max_asst_blocks=caps[1],
    )
    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4],
        prev_in_tokens=256,
        prev_out_tokens=64,
        curr_hash_ids=[1, 2, 3],
        curr_in_tokens=192,
        seed="s2",
        max_asst_blocks=caps[2],
    )
    return r


def test_advance_with_cap_eliminates_trailing_assistant():
    """Applying the planner cap to turn 1 makes the turn-2 pull-back land on a"""
    caps = compute_asst_block_caps(
        [([1, 2], 128), ([1, 2, 3, 4], 256), ([1, 2, 3], 192)], 64
    )
    r = _run_canonical_three_turns(caps)
    assert r._segments[-1].role == "user"
    assert r._trailing_non_user_turns == []
    assert sum(len(s.tokens) for s in r._segments) == 192


def test_advance_without_cap_reproduces_trailing_assistant():
    """Regression guard: max_asst_blocks=None reproduces the pre-fix degenerate"""
    r = _run_canonical_three_turns([None, None, None])
    assert [s.role for s in r._segments] == ["user", "assistant"]
    assert r._trailing_non_user_turns == [2]


def test_advance_cap_larger_than_region_is_noop():
    """A cap >= new_blocks_count does not shrink the assistant below what the"""
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
        max_asst_blocks=99,
    )
    assert [s.role for s in r._segments] == ["user", "assistant", "user"]
    assert r._segments[1].block_count == 2


def _make_tool_shaped_recon(bs=64):
    return ConversationReconstructor(
        block_size=bs,
        decode_block_tokens=_stub_decode_block_tokens,
        sample_partial_tail_tokens=_stub_partial_tail_tokens,
        decode_tokens_to_text=_stub_decode_tokens_to_text,
        tool_shaped_messages=True,
    )


def test_cap_demotes_unpaired_tool_result_to_plain_user():
    """When a planner cap removes the assistant a tool-result turn would have"""
    r_uncapped = _make_tool_shaped_recon()
    r_uncapped.init_turn_0(
        hash_ids=[1, 2], in_tokens=128, tool_tokens=0, system_tokens=0, seed="s0"
    )
    r_uncapped.turn_delta()
    r_uncapped.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=128,
        prev_out_tokens=200,
        curr_hash_ids=[1, 2, 3, 4],
        curr_in_tokens=256,
        seed="s1",
        is_tool_result=True,
    )
    d_uncapped = r_uncapped.turn_delta()
    assert d_uncapped.delta_messages[-1]["role"] == "tool"

    r_capped = _make_tool_shaped_recon()
    r_capped.init_turn_0(
        hash_ids=[1, 2], in_tokens=128, tool_tokens=0, system_tokens=0, seed="s0"
    )
    r_capped.turn_delta()
    r_capped.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=128,
        prev_out_tokens=200,
        curr_hash_ids=[1, 2, 3, 4],
        curr_in_tokens=256,
        seed="s1",
        is_tool_result=True,
        max_asst_blocks=0,
    )
    d_capped = r_capped.turn_delta()
    assert [m["role"] for m in d_capped.delta_messages] == ["user"]
    assert all("tool_calls" not in m for m in d_capped.delta_messages)
    r_capped._emitted_segment_count = 0
    r_capped._last_disturbance_at = None
    d_reset = r_capped.turn_delta()
    assert d_reset.delta_messages[-1]["role"] == "user"
    assert all(m["role"] != "tool" for m in d_reset.delta_messages)
