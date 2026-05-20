# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the deterministic cache-bust marker builder."""

from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.timing.strategies.cache_bust import (
    build_cache_bust_marker,
    estimate_marker_token_cost,
)


class TestBuildCacheBustMarker:
    """``build_cache_bust_marker`` is the source of per-conversation cache-bust
    markers consumed by composer/orchestrator. Determinism and
    position-correctness are load-bearing for warmup-to-profiling coherence
    (a trajectory's warmup turn k_i and first profiling turn k_i+1 must share
    the same marker so warmup KV-cache work transfers to profiling)."""

    def test_returns_none_for_target_none(self) -> None:
        """NONE target short-circuits — composer relies on this to skip
        injection without re-checking the target."""
        assert (
            build_cache_bust_marker(
                benchmark_id="b",
                recycle_pass=0,
                trajectory_index=0,
                trace_id="t",
                target=CacheBustTarget.NONE,
            )
            is None
        )

    @pytest.mark.parametrize(
        "target",
        [
            CacheBustTarget.SYSTEM_PREFIX,
            CacheBustTarget.SYSTEM_SUFFIX,
            CacheBustTarget.FIRST_TURN_PREFIX,
            CacheBustTarget.FIRST_TURN_SUFFIX,
        ],
    )
    def test_same_inputs_same_marker(self, target: CacheBustTarget) -> None:
        """Determinism: same (benchmark_id, recycle_pass, trajectory_index,
        trace_id) -> identical marker across calls. Required so reruns
        produce byte-equivalent prompts."""
        a = build_cache_bust_marker(
            benchmark_id="bench-1",
            recycle_pass=3,
            trajectory_index=7,
            trace_id="trace-xyz",
            target=target,
        )
        b = build_cache_bust_marker(
            benchmark_id="bench-1",
            recycle_pass=3,
            trajectory_index=7,
            trace_id="trace-xyz",
            target=target,
        )
        assert a == b
        assert a is not None

    def test_different_trace_id_yields_different_marker(self) -> None:
        """Trace id is in the digest input specifically to prevent the
        empirical 33%-collision rate seen at MVP scale when two traces
        shared a (recycle_pass, lane) tuple."""
        a = build_cache_bust_marker(
            benchmark_id="b",
            recycle_pass=0,
            trajectory_index=0,
            trace_id="trace-A",
            target=CacheBustTarget.SYSTEM_PREFIX,
        )
        b = build_cache_bust_marker(
            benchmark_id="b",
            recycle_pass=0,
            trajectory_index=0,
            trace_id="trace-B",
            target=CacheBustTarget.SYSTEM_PREFIX,
        )
        assert a != b

    def test_different_recycle_pass_yields_different_marker(self) -> None:
        a = build_cache_bust_marker(
            benchmark_id="b",
            recycle_pass=0,
            trajectory_index=0,
            trace_id="t",
            target=CacheBustTarget.SYSTEM_PREFIX,
        )
        b = build_cache_bust_marker(
            benchmark_id="b",
            recycle_pass=1,
            trajectory_index=0,
            trace_id="t",
            target=CacheBustTarget.SYSTEM_PREFIX,
        )
        assert a != b

    def test_different_trajectory_index_yields_different_marker(self) -> None:
        a = build_cache_bust_marker(
            benchmark_id="b",
            recycle_pass=0,
            trajectory_index=0,
            trace_id="t",
            target=CacheBustTarget.SYSTEM_PREFIX,
        )
        b = build_cache_bust_marker(
            benchmark_id="b",
            recycle_pass=0,
            trajectory_index=1,
            trace_id="t",
            target=CacheBustTarget.SYSTEM_PREFIX,
        )
        assert a != b

    def test_prefix_targets_have_marker_first(self) -> None:
        """SYSTEM_PREFIX and FIRST_TURN_PREFIX inject before content — the
        marker must come before any whitespace separator so it lands at
        token 0 of the rendered prompt."""
        for target in (
            CacheBustTarget.SYSTEM_PREFIX,
            CacheBustTarget.FIRST_TURN_PREFIX,
        ):
            m = build_cache_bust_marker(
                benchmark_id="b",
                recycle_pass=0,
                trajectory_index=0,
                trace_id="t",
                target=target,
            )
            assert m is not None
            assert m.startswith("[rid:")
            assert m.endswith("\n\n")

    def test_suffix_targets_have_marker_last(self) -> None:
        """SYSTEM_SUFFIX and FIRST_TURN_SUFFIX append after content — the
        leading separator preserves leading-prefix KV-cache locality."""
        for target in (
            CacheBustTarget.SYSTEM_SUFFIX,
            CacheBustTarget.FIRST_TURN_SUFFIX,
        ):
            m = build_cache_bust_marker(
                benchmark_id="b",
                recycle_pass=0,
                trajectory_index=0,
                trace_id="t",
                target=target,
            )
            assert m is not None
            assert m.startswith("\n\n")
            assert m.endswith("]")

    def test_marker_format_is_rid_with_12_hex(self) -> None:
        """The rid:<digest> shape is what downstream tooling greps for in
        request logs; the 12-hex width is part of that contract."""
        marker = build_cache_bust_marker(
            benchmark_id="b",
            recycle_pass=0,
            trajectory_index=0,
            trace_id="t",
            target=CacheBustTarget.SYSTEM_PREFIX,
        )
        assert marker is not None
        # Strip whitespace separator; verify "[rid:XXXXXXXXXXXX]" shape.
        core = marker.strip()
        assert core.startswith("[rid:")
        assert core.endswith("]")
        digest = core[len("[rid:") : -1]
        assert len(digest) == 12
        assert all(c in "0123456789abcdef" for c in digest)


class TestEstimateMarkerTokenCost:
    def test_returns_zero_for_target_none(self) -> None:
        """NONE has no marker so the cost is structurally zero, not just
        empirically — callers branch on this to skip token-budget accounting."""
        tok = MagicMock()
        assert estimate_marker_token_cost(CacheBustTarget.NONE, tok) == 0
        tok.encode.assert_not_called()

    def test_averages_tokenizer_samples(self) -> None:
        """The estimator tokenizes a handful of distinct markers and rounds
        the mean. Hard-coding a deterministic tokenizer lets us assert the
        rounding behavior."""
        tok = MagicMock()
        # Pretend the tokenizer always returns 5 tokens.
        tok.encode.return_value = [0, 1, 2, 3, 4]
        cost = estimate_marker_token_cost(CacheBustTarget.SYSTEM_PREFIX, tok, samples=4)
        assert cost == 5
        assert tok.encode.call_count == 4

    def test_default_sample_count(self) -> None:
        """The default sample count is fixed at 8 — keep it that way so
        callers don't need to know about it."""
        tok = MagicMock()
        tok.encode.return_value = [0, 1, 2]
        estimate_marker_token_cost(CacheBustTarget.SYSTEM_PREFIX, tok)
        assert tok.encode.call_count == 8
