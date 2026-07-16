# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive unit tests for the pure Baseten replay time-model transforms."""

from __future__ import annotations

from itertools import pairwise

import pytest

from aiperf.dataset.loader._baseten_replay_timemodel import reflow_idle_gaps


class TestReflowIdleGaps:
    def test_identity_when_cap_disabled(self):
        ts = [0, 100, 5000, 5050]
        assert reflow_idle_gaps(ts, None) == ts

    @pytest.mark.parametrize("cap", [0, -5, 0.0])
    def test_non_positive_cap_rejected(self, cap):
        # A user passing 0 wants back-to-back replay, not a silent no-op;
        # the config layer enforces > 0, so a non-positive cap here is a bug.
        with pytest.raises(ValueError, match="cap_ms"):
            reflow_idle_gaps([0, 100, 5000], cap)

    def test_sub_millisecond_cap_rounds_up_to_one_ms(self):
        # int-ms timestamps cannot express sub-ms gaps; round the cap up
        # instead of truncating it to a 0ms collapse-everything cap.
        assert reflow_idle_gaps([0, 100_000], 0.5) == [0, 1]

    def test_empty(self):
        assert reflow_idle_gaps([], 1000) == []

    def test_single(self):
        assert reflow_idle_gaps([4242], 1000) == [4242]

    def test_all_gaps_within_cap_unchanged(self):
        ts = [0, 500, 1500, 3000]  # gaps 500, 1000, 1500 all <= 2000
        assert reflow_idle_gaps(ts, 2000) == ts

    def test_oversized_gap_collapsed_to_cap(self):
        # gaps: 500 (ok), 11000 (-> 5000), 200 (ok)
        ts = [0, 500, 11500, 11700]
        assert reflow_idle_gaps(ts, 5000) == [0, 500, 5500, 5700]

    def test_first_value_preserved(self):
        # leading offset is NOT trimmed by the gap-cap (origin policy's job)
        assert reflow_idle_gaps([3000, 20000], 5000) == [3000, 8000]

    def test_unsorted_input_maps_back_positionally(self):
        # input order: [late, early, mid]; time order: early(0), mid(100), late(20000->capped)
        out = reflow_idle_gaps([20000, 0, 100], 5000)
        # early stays 0, mid stays 100 (gap 100), late capped to 100+5000
        assert out == [5100, 0, 100]
        # monotonic in time order
        assert sorted(out) == [0, 100, 5100]

    def test_ties_are_zero_gap_and_stable(self):
        out = reflow_idle_gaps([0, 0, 10000, 10000], 1000)
        # both zeros stay 0; jump to the pair (gap 10000 -> 1000); the tie stays 0-gap
        assert out == [0, 0, 1000, 1000]

    def test_monotonic_nondecreasing_in_time_order(self):
        ts = [0, 100, 99999, 100050, 500000]
        out = reflow_idle_gaps(ts, 2000)
        in_time_order = [out[i] for i in sorted(range(len(ts)), key=lambda i: ts[i])]
        assert in_time_order == sorted(in_time_order)
        # every consecutive gap <= cap
        gaps = [b - a for a, b in pairwise(in_time_order)]
        assert all(g <= 2000 for g in gaps)

    def test_total_span_shrinks_by_excess(self):
        # one 11000 gap capped to 5000 -> span shrinks by exactly 6000
        ts = [0, 1000, 12000]
        out = reflow_idle_gaps(ts, 5000)
        assert max(out) == max(ts) - (11000 - 5000)

    def test_float_inputs_coerced_to_int(self):
        assert reflow_idle_gaps([0.0, 999.0, 50000.0], 5000) == [0, 999, 5999]
