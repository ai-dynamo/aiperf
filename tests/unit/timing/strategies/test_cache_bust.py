# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.timing.strategies.cache_bust import (
    build_cache_bust_marker,
    estimate_marker_token_cost,
)

_RID_PATTERN = re.compile(r"\[rid:[0-9a-f]{12}\]")


def test_marker_is_deterministic():
    a = build_cache_bust_marker(
        "bench-1", 0, 0, "trace_a", target=CacheBustTarget.SYSTEM_PREFIX
    )
    b = build_cache_bust_marker(
        "bench-1", 0, 0, "trace_a", target=CacheBustTarget.SYSTEM_PREFIX
    )
    assert a == b


@pytest.mark.parametrize(
    "target",
    [
        CacheBustTarget.SYSTEM_PREFIX,
        CacheBustTarget.SYSTEM_SUFFIX,
        CacheBustTarget.FIRST_TURN_PREFIX,
        CacheBustTarget.FIRST_TURN_SUFFIX,
    ],
)
def test_marker_contains_rid_token(target):
    marker = build_cache_bust_marker("bench", 0, 0, "trace_a", target=target)
    assert _RID_PATTERN.search(marker) is not None


def test_prefix_variants_have_trailing_newlines():
    for target in (CacheBustTarget.SYSTEM_PREFIX, CacheBustTarget.FIRST_TURN_PREFIX):
        marker = build_cache_bust_marker("bench", 0, 0, "trace_a", target=target)
        assert marker.endswith("\n\n")
        assert not marker.startswith("\n\n")


def test_suffix_variants_have_leading_newlines():
    for target in (CacheBustTarget.SYSTEM_SUFFIX, CacheBustTarget.FIRST_TURN_SUFFIX):
        marker = build_cache_bust_marker("bench", 0, 0, "trace_a", target=target)
        assert marker.startswith("\n\n")
        assert not marker.endswith("\n\n")


def test_marker_changes_per_input_dimension():
    base = build_cache_bust_marker(
        "bench", 0, 0, "trace_a", target=CacheBustTarget.SYSTEM_PREFIX
    )
    assert (
        build_cache_bust_marker(
            "other", 0, 0, "trace_a", target=CacheBustTarget.SYSTEM_PREFIX
        )
        != base
    )
    assert (
        build_cache_bust_marker(
            "bench", 1, 0, "trace_a", target=CacheBustTarget.SYSTEM_PREFIX
        )
        != base
    )
    assert (
        build_cache_bust_marker(
            "bench", 0, 1, "trace_a", target=CacheBustTarget.SYSTEM_PREFIX
        )
        != base
    )
    # trace_id is part of the digest tuple — changing it must change the digest.
    assert (
        build_cache_bust_marker(
            "bench", 0, 0, "trace_b", target=CacheBustTarget.SYSTEM_PREFIX
        )
        != base
    )


def test_marker_position_does_not_change_digest():
    pre = build_cache_bust_marker(
        "bench", 0, 0, "trace_a", target=CacheBustTarget.SYSTEM_PREFIX
    )
    suf = build_cache_bust_marker(
        "bench", 0, 0, "trace_a", target=CacheBustTarget.SYSTEM_SUFFIX
    )
    digest_pre = _RID_PATTERN.search(pre).group()
    digest_suf = _RID_PATTERN.search(suf).group()
    assert digest_pre == digest_suf


def test_marker_position_does_not_change_digest_with_trace_id():
    """Same (bid, pass, lane, trace_id), different position -> same rid embedded."""
    pre = build_cache_bust_marker(
        "bench", 3, 7, "trace_xyz", target=CacheBustTarget.SYSTEM_PREFIX
    )
    suf = build_cache_bust_marker(
        "bench", 3, 7, "trace_xyz", target=CacheBustTarget.SYSTEM_SUFFIX
    )
    first_pre = build_cache_bust_marker(
        "bench", 3, 7, "trace_xyz", target=CacheBustTarget.FIRST_TURN_PREFIX
    )
    first_suf = build_cache_bust_marker(
        "bench", 3, 7, "trace_xyz", target=CacheBustTarget.FIRST_TURN_SUFFIX
    )
    digests = {
        _RID_PATTERN.search(pre).group(),
        _RID_PATTERN.search(suf).group(),
        _RID_PATTERN.search(first_pre).group(),
        _RID_PATTERN.search(first_suf).group(),
    }
    assert len(digests) == 1


def test_marker_differs_when_only_trace_id_differs():
    """Same (bid, pass, lane), different trace_id -> different rids.

    This is the entire point of the collision-free fix: two different traces
    landing on the same (recycle_pass, lane) tuple must produce distinct
    markers so submission compliance can rely on per-session uniqueness.
    """
    a = build_cache_bust_marker(
        "bench", 0, 0, "trace_a", target=CacheBustTarget.SYSTEM_PREFIX
    )
    b = build_cache_bust_marker(
        "bench", 0, 0, "trace_b", target=CacheBustTarget.SYSTEM_PREFIX
    )
    assert a != b
    assert _RID_PATTERN.search(a).group() != _RID_PATTERN.search(b).group()


def test_target_none_returns_none():
    assert (
        build_cache_bust_marker("bench", 0, 0, "trace_a", target=CacheBustTarget.NONE)
        is None
    )


class _FakeTokenizer:
    """Minimal tokenizer stub: 1 token per 4 chars (rounded up)."""

    def encode(self, text: str, **_kwargs):
        return [0] * ((len(text) + 3) // 4)


def test_estimate_marker_token_cost_none_returns_zero():
    assert estimate_marker_token_cost(CacheBustTarget.NONE, _FakeTokenizer()) == 0


@pytest.mark.parametrize(
    "target",
    [
        CacheBustTarget.SYSTEM_PREFIX,
        CacheBustTarget.SYSTEM_SUFFIX,
        CacheBustTarget.FIRST_TURN_PREFIX,
        CacheBustTarget.FIRST_TURN_SUFFIX,
    ],
)
def test_estimate_marker_token_cost_positive_for_active_targets(target):
    cost = estimate_marker_token_cost(target, _FakeTokenizer())
    # Marker is 20 chars; fake tokenizer gives ceil(20/4) = 5 tokens.
    assert cost == 5


def test_estimate_marker_token_cost_averages_across_samples():
    """Tokenizer is called once per sample so the result is a real average."""

    class CountingTokenizer:
        def __init__(self):
            self.calls = 0

        def encode(self, text: str, **_kwargs):
            self.calls += 1
            return [0] * len(text)

    tok = CountingTokenizer()
    estimate_marker_token_cost(CacheBustTarget.SYSTEM_PREFIX, tok, samples=4)
    assert tok.calls == 4


def test_estimate_marker_token_cost_rounds_to_int():
    """Variable token counts across samples round to a clean int."""

    class JitterTokenizer:
        def __init__(self):
            self.n = 0

        def encode(self, text: str, **_kwargs):
            self.n += 1
            # Returns 5,6,5,6,5,6,5,6 -> mean 5.5 -> rounds to 6 (banker's rounding).
            return [0] * (5 if self.n % 2 else 6)

    cost = estimate_marker_token_cost(
        CacheBustTarget.SYSTEM_PREFIX, JitterTokenizer(), samples=8
    )
    assert cost == 6
