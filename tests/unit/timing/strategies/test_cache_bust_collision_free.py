# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""High-volume robustness coverage for ``build_cache_bust_marker``.

The basic determinism / position / per-dimension digest assertions live in
``test_cache_bust.py``. This file is the regression bar for the
collision-free fix (commit ``9261865fc``): the marker tuple now embeds
``trace_id`` so cross-trace collisions on the same ``(recycle_pass, lane)``
slot are eliminated by construction.

Tests here scale to 10k+ inputs to make any digest truncation, hashing
mistake, or input-string concatenation regression visible. Each test runs
in well under a second on a modern laptop; tune the loop counts down if
xdist contention surfaces flakes.
"""

from __future__ import annotations

import re

import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.timing.strategies.cache_bust import build_cache_bust_marker

_RID_PATTERN = re.compile(r"\[rid:[0-9a-f]{12}\]")
_TARGET = CacheBustTarget.SYSTEM_PREFIX
_BENCHMARK_ID = "bench-stress"


def _rid(marker: str) -> str:
    """Extract the ``[rid:HEX]`` token from a rendered marker."""
    m = _RID_PATTERN.search(marker)
    assert m is not None, f"no rid token in marker {marker!r}"
    return m.group(0)


def test_no_collisions_across_10k_distinct_inputs():
    """Cartesian product of 10 trace_ids x 10 lanes x 100 recycle_passes.

    All 10,000 inputs are distinct under the (recycle_pass, lane, trace_id)
    tuple, so all 10,000 markers must be distinct.
    """
    markers: set[str] = set()
    expected = 10 * 10 * 100
    for trace_idx in range(10):
        trace_id = f"trace_{trace_idx}"
        for lane in range(10):
            for recycle_pass in range(100):
                marker = build_cache_bust_marker(
                    _BENCHMARK_ID, recycle_pass, lane, trace_id, target=_TARGET
                )
                markers.add(marker)
    assert len(markers) == expected, (
        f"expected {expected} distinct markers; got {len(markers)} "
        f"({expected - len(markers)} collisions)"
    )


def test_collision_free_at_same_pass_lane_different_traces():
    """Pin (pass=0, lane=0); pivot only trace_id across 100 distinct values.

    Regression bar for the fix: pre-fix this collapsed to a single digest
    because the tuple did not include trace_id. Post-fix, every trace_id
    must produce its own digest at the same (pass, lane) slot.
    """
    markers: set[str] = set()
    for i in range(100):
        marker = build_cache_bust_marker(
            _BENCHMARK_ID, 0, 0, f"trace_collision_{i}", target=_TARGET
        )
        markers.add(_rid(marker))
    assert len(markers) == 100, (
        "Two distinct trace_ids at (recycle_pass=0, lane=0) must produce "
        f"distinct rids; got {len(markers)} distinct from 100 inputs"
    )


def test_same_input_yields_same_marker_across_calls():
    """Determinism: same args -> same digest, every call."""
    args = (_BENCHMARK_ID, 7, 3, "trace_determ")
    first = build_cache_bust_marker(*args, target=_TARGET)
    for _ in range(100):
        assert build_cache_bust_marker(*args, target=_TARGET) == first


def test_input_dimensions_each_independently_change_digest():
    """Holding 3 of 4 inputs constant, flipping the 4th changes the digest.

    Mirrors ``test_marker_changes_per_input_dimension`` but as 4 independent
    micro-checks so a regression in any one dimension surfaces clearly.
    """
    base_args = ("bench", 5, 2, "trace_dim")
    base = build_cache_bust_marker(*base_args, target=_TARGET)

    # benchmark_id
    assert (
        build_cache_bust_marker("other_bench", 5, 2, "trace_dim", target=_TARGET)
        != base
    )

    # recycle_pass
    assert build_cache_bust_marker("bench", 6, 2, "trace_dim", target=_TARGET) != base

    # trajectory_index
    assert build_cache_bust_marker("bench", 5, 99, "trace_dim", target=_TARGET) != base

    # trace_id
    assert build_cache_bust_marker("bench", 5, 2, "trace_other", target=_TARGET) != base


def test_trace_id_collision_within_pass_zero_lane_zero():
    """Locks in the trace_id contribution at the worst-case slot.

    Two traces, same (pass=0, lane=0): the ONLY differentiator is trace_id,
    so a regression that drops trace_id from the digest input would collapse
    these two markers. Distinct rids required.
    """
    a = build_cache_bust_marker(_BENCHMARK_ID, 0, 0, "trace_a", target=_TARGET)
    b = build_cache_bust_marker(_BENCHMARK_ID, 0, 0, "trace_b", target=_TARGET)
    assert _rid(a) != _rid(b)


@pytest.mark.parametrize("count", [50_000])
def test_marker_is_collision_free_under_birthday_paradox_stress(count):
    """Smoke check that the input is actually being hashed (not truncated).

    Generate a large grid of structured inputs spread across (pass<10000,
    lane<100, trace_id of 10 chars). 12 hex chars = 48 bits, so for 50k
    inputs the expected birthday-paradox collision count is
    ``50000^2 / (2 * 2^48) ~= 0.0044`` -- effectively zero. We allow up to
    9 collisions before the test fails, which would still indicate a
    malformed digest input (e.g. truncation, wrong field order).
    """
    markers: set[str] = set()
    duplicates = 0
    # Deterministic structured space: 100 lanes x 100 traces x 5 passes = 50k
    for lane in range(100):
        for trace_idx in range(100):
            trace_id = f"t_{trace_idx:04d}_x"
            for pass_offset in range(5):
                # Spread recycle_pass widely so we sample the input domain.
                recycle_pass = pass_offset * 1900 + lane * 7 + trace_idx
                marker = build_cache_bust_marker(
                    _BENCHMARK_ID, recycle_pass, lane, trace_id, target=_TARGET
                )
                rid = _rid(marker)
                if rid in markers:
                    duplicates += 1
                markers.add(rid)
    assert len(markers) + duplicates == count, (
        f"sanity: generated {len(markers) + duplicates} != expected {count}"
    )
    assert duplicates < 10, (
        f"sha256[:12] should be effectively collision-free at {count} inputs; "
        f"saw {duplicates} duplicates -- hint at digest truncation or input "
        "string regression"
    )
