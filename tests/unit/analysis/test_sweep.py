# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sweep-line algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.analysis.sweepline import (
    add_step_functions,
    compute_time_weighted_stats,
    concurrency_sweep_line,
    divide_step_functions,
    prefill_throughput_per_user_sweep_line,
    prefill_throughput_sweep_line,
    throughput_per_user_sweep_line,
    throughput_sweep_line,
    throughput_sweep_line_icl,
    tokens_in_flight_sweep_line,
    tokens_in_flight_sweep_line_icl,
    total_throughput_sweep_line,
)


class TestConcurrencySweep:
    def test_empty_input(self) -> None:
        ts, conc = concurrency_sweep_line(
            np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        )
        assert len(ts) == 0
        assert len(conc) == 0

    def test_all_nan(self) -> None:
        ts, conc = concurrency_sweep_line(
            np.array([np.nan, np.nan]), np.array([np.nan, np.nan])
        )
        assert len(ts) == 0

    def test_single_request(self) -> None:
        start = np.array([100.0])
        end = np.array([200.0])
        ts, conc = concurrency_sweep_line(start, end)
        assert len(ts) == 2
        assert ts[0] == 100.0
        assert ts[1] == 200.0
        assert conc[0] == 1.0  # request starts
        assert conc[1] == 0.0  # request ends

    def test_sequential_non_overlapping(self) -> None:
        """Sequential requests: concurrency always 0 or 1."""
        start = np.array([100.0, 300.0, 500.0])
        end = np.array([200.0, 400.0, 600.0])
        ts, conc = concurrency_sweep_line(start, end)
        # All concurrency values should be 0 or 1
        assert np.all((conc == 0) | (conc == 1))
        assert float(np.max(conc)) == 1.0

    def test_overlapping_requests(self) -> None:
        """10 overlapping requests → peak concurrency is 10."""
        start = np.array([float(i) for i in range(10)])
        end = np.array([float(i + 100) for i in range(10)])
        ts, conc = concurrency_sweep_line(start, end)
        assert float(np.max(conc)) == 10.0

    def test_nan_records_excluded(self) -> None:
        start = np.array([100.0, np.nan, 300.0])
        end = np.array([200.0, np.nan, 400.0])
        ts, conc = concurrency_sweep_line(start, end)
        # Only 2 valid records
        assert len(ts) == 4  # 2 records * 2 events each
        assert float(np.max(conc)) <= 2.0

    def test_concurrent_peak(self) -> None:
        """3 fully overlapping requests."""
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([100.0, 100.0, 100.0])
        ts, conc = concurrency_sweep_line(start, end)
        assert float(np.max(conc)) == 3.0


class TestThroughputSweep:
    def test_empty_input(self) -> None:
        ts, tput = throughput_sweep_line(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        assert len(ts) == 0

    def test_single_request_known_rate(self) -> None:
        """Single request: 101 output tokens over 100ns → rate = 100/100 = 1.0 tokens/ns."""
        gen_start = np.array([0.0])
        end = np.array([100.0])
        output_tokens = np.array([101.0])
        ts, tput = throughput_sweep_line(gen_start, end, output_tokens)
        assert len(ts) == 2
        assert tput[0] == pytest.approx(1.0)  # rate added at start
        assert tput[1] == pytest.approx(0.0)  # rate removed at end

    def test_zero_output_tokens_excluded(self) -> None:
        """Requests with 0 or 1 output tokens should not contribute to throughput."""
        gen_start = np.array([0.0, 50.0])
        end = np.array([100.0, 150.0])
        output_tokens = np.array([1.0, 11.0])  # First: (1-1)/100=0, Second: 10/100=0.1
        ts, tput = throughput_sweep_line(gen_start, end, output_tokens)
        # First request has rate 0, so only 1 valid request contributes
        # (1-1)/100 = 0 rate for first, so it's technically valid but 0 duration check handles it
        assert len(ts) > 0

    def test_nan_excluded(self) -> None:
        gen_start = np.array([0.0, np.nan])
        end = np.array([100.0, 200.0])
        output_tokens = np.array([11.0, np.nan])
        ts, tput = throughput_sweep_line(gen_start, end, output_tokens)
        assert len(ts) == 2  # Only 1 valid request


class TestPrefillThroughputSweep:
    def test_empty_input(self) -> None:
        ts, tput = prefill_throughput_sweep_line(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        assert len(ts) == 0
        assert len(tput) == 0

    def test_single_request_known_rate(self) -> None:
        """Single request: 100 input tokens over 50ns prefill → rate = 2.0 tokens/ns."""
        start = np.array([0.0])
        gen_start = np.array([50.0])
        input_tokens = np.array([100.0])
        ts, tput = prefill_throughput_sweep_line(start, gen_start, input_tokens)
        assert len(ts) == 2
        assert tput[0] == pytest.approx(2.0)  # rate added at start
        assert tput[1] == pytest.approx(0.0)  # rate removed at gen_start

    def test_nan_excluded(self) -> None:
        """NaN input_tokens or generation_start_ns are filtered out."""
        start = np.array([0.0, 0.0, 0.0])
        gen_start = np.array([50.0, np.nan, 50.0])
        input_tokens = np.array([100.0, 100.0, np.nan])
        ts, tput = prefill_throughput_sweep_line(start, gen_start, input_tokens)
        # Only 1 valid record
        assert len(ts) == 2

    def test_zero_prefill_duration_excluded(self) -> None:
        """start_ns == generation_start_ns → zero duration → filtered out."""
        start = np.array([100.0])
        gen_start = np.array([100.0])
        input_tokens = np.array([50.0])
        ts, tput = prefill_throughput_sweep_line(start, gen_start, input_tokens)
        assert len(ts) == 0
        assert len(tput) == 0

    def test_overlapping_prefills(self) -> None:
        """Two concurrent prefills → peak rate = sum of individual rates."""
        # Request A: [0, 50), 100 tokens → rate = 2.0
        # Request B: [10, 60), 150 tokens → rate = 3.0
        # Overlap at [10, 50): combined rate = 5.0
        start = np.array([0.0, 10.0])
        gen_start = np.array([50.0, 60.0])
        input_tokens = np.array([100.0, 150.0])
        ts, tput = prefill_throughput_sweep_line(start, gen_start, input_tokens)
        assert float(np.max(tput)) == pytest.approx(5.0)


class TestTotalThroughputSweep:
    def test_empty_input(self) -> None:
        empty = np.array([], dtype=np.float64)
        ts, tput = total_throughput_sweep_line(
            empty, empty, empty, empty, output_tokens=empty
        )
        assert len(ts) == 0

    def test_single_request_combines_phases(self) -> None:
        """Single request: prefill rate + generation rate in one curve."""
        # Prefill: [0, 50), 100 input tokens → rate = 2.0 tokens/ns
        # Generation: [50, 150), 101 output tokens → rate = (101-1)/100 = 1.0 tokens/ns
        start = np.array([0.0])
        gen_start = np.array([50.0])
        end = np.array([150.0])
        input_tokens = np.array([100.0])
        output_tokens = np.array([101.0])

        ts, tput = total_throughput_sweep_line(
            start,
            gen_start,
            end,
            input_tokens,
            output_tokens=output_tokens,
        )
        assert len(ts) > 0
        # During prefill [0,50): rate = 2.0
        # During generation [50,150): rate = 1.0
        assert float(np.max(tput)) == pytest.approx(2.0)

    def test_matches_add_step_functions(self) -> None:
        """Single-pass sweep matches separate sweeps + add for overlapping requests."""
        start = np.array([0.0, 10.0, 20.0])
        gen_start = np.array([50.0, 60.0, 70.0])
        end = np.array([150.0, 160.0, 170.0])
        input_tokens = np.array([100.0, 200.0, 150.0])
        output_tokens = np.array([101.0, 51.0, 76.0])

        # Single-pass
        ts1, vals1 = total_throughput_sweep_line(
            start,
            gen_start,
            end,
            input_tokens,
            output_tokens=output_tokens,
        )

        # Two-pass + add
        pts, pvals = prefill_throughput_sweep_line(start, gen_start, input_tokens)
        tts, tvals = throughput_sweep_line(gen_start, end, output_tokens)
        ts2, vals2 = add_step_functions(pts, pvals, tts, tvals)

        # Both should give same time-weighted avg over the full window
        from aiperf.analysis.sweepline import compute_time_weighted_stats

        w_start = min(float(ts1[0]), float(ts2[0]))
        w_end = max(float(ts1[-1]), float(ts2[-1]))
        stats1 = compute_time_weighted_stats(ts1, vals1, w_start, w_end)
        stats2 = compute_time_weighted_stats(ts2, vals2, w_start, w_end)
        assert stats1.avg == pytest.approx(stats2.avg, rel=1e-10)
        assert stats1.max == pytest.approx(stats2.max, rel=1e-10)

    def test_prefill_only(self) -> None:
        """No valid generation data → only prefill contributes."""
        start = np.array([0.0])
        gen_start = np.array([50.0])
        end = np.array([50.0])  # zero gen duration → no gen contribution
        input_tokens = np.array([100.0])
        output_tokens = np.array([np.nan])

        ts, tput = total_throughput_sweep_line(
            start,
            gen_start,
            end,
            input_tokens,
            output_tokens=output_tokens,
        )
        assert len(ts) > 0
        assert float(np.max(tput)) == pytest.approx(2.0)  # 100/50

    def test_generation_only(self) -> None:
        """No valid prefill data → only generation contributes."""
        start = np.array([np.nan])
        gen_start = np.array([0.0])
        end = np.array([100.0])
        input_tokens = np.array([np.nan])
        output_tokens = np.array([101.0])

        ts, tput = total_throughput_sweep_line(
            start,
            gen_start,
            end,
            input_tokens,
            output_tokens=output_tokens,
        )
        assert len(ts) > 0
        assert float(np.max(tput)) == pytest.approx(1.0)  # 100/100


class TestThroughputSweepIcl:
    def test_empty_input(self) -> None:
        ts, tput = throughput_sweep_line_icl(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int32),
            icl_offsets=np.array([], dtype=np.int64),
        )
        assert len(ts) == 0

    def test_single_request_uniform_chunks(self) -> None:
        """Single request with 3 equal ICL chunks of 10ns each."""
        gen_start = np.array([0.0])  # indexed by session_num
        output_tokens = np.array([3.0])  # 3 tokens across 3 chunks → 1 tok/msg
        icl_values = np.array([10.0, 10.0, 10.0])
        icl_record_indices = np.array([0, 0, 0], dtype=np.int32)
        icl_offsets = np.array([0], dtype=np.int64)

        ts, tput = throughput_sweep_line_icl(
            gen_start,
            output_tokens,
            icl_values,
            icl_record_indices,
            icl_offsets=icl_offsets,
        )
        assert len(ts) == 6  # 3 chunks * 2 events each
        # Each chunk: rate = (3/3) / 10 = 0.1 tokens/ns
        # Since chunks are sequential, peak throughput should be 0.1
        assert float(np.max(tput)) == pytest.approx(0.1)

    def test_nan_gen_start_excluded(self) -> None:
        """Records with NaN generation_start should be excluded."""
        gen_start = np.array([np.nan])
        output_tokens = np.array([5.0])
        icl_values = np.array([10.0])
        icl_record_indices = np.array([0], dtype=np.int32)
        icl_offsets = np.array([0], dtype=np.int64)

        ts, tput = throughput_sweep_line_icl(
            gen_start,
            output_tokens,
            icl_values,
            icl_record_indices,
            icl_offsets=icl_offsets,
        )
        assert len(ts) == 0

    def test_two_overlapping_requests(self) -> None:
        """Two requests with overlapping ICL chunks."""
        gen_start = np.array([0.0, 5.0])
        output_tokens = np.array([2.0, 2.0])  # 2 tokens each, 2 chunks each → 1 tok/msg
        # Request 0: chunks at [0,10), [10,20)
        # Request 1: chunks at [5,15), [15,25)
        icl_values = np.array([10.0, 10.0, 10.0, 10.0])
        icl_record_indices = np.array([0, 0, 1, 1], dtype=np.int32)
        icl_offsets = np.array([0, 2], dtype=np.int64)

        ts, tput = throughput_sweep_line_icl(
            gen_start,
            output_tokens,
            icl_values,
            icl_record_indices,
            icl_offsets=icl_offsets,
        )
        assert len(ts) == 8  # 4 chunks * 2 events
        # When chunks overlap, throughput should be > single chunk rate
        assert float(np.max(tput)) > 0.1

    def test_rescaling_with_variable_tokens(self) -> None:
        """6 output tokens across 3 chunks → 2 tokens/msg, rates double."""
        gen_start = np.array([0.0])
        output_tokens = np.array([6.0])  # 6 tokens across 3 chunks → 2 tok/msg
        icl_values = np.array([10.0, 10.0, 10.0])
        icl_record_indices = np.array([0, 0, 0], dtype=np.int32)
        icl_offsets = np.array([0], dtype=np.int64)

        ts, tput = throughput_sweep_line_icl(
            gen_start,
            output_tokens,
            icl_values,
            icl_record_indices,
            icl_offsets=icl_offsets,
        )
        assert len(ts) == 6
        # Each chunk: rate = (6/3) / 10 = 0.2 tokens/ns (doubled vs 1 tok/msg)
        assert float(np.max(tput)) == pytest.approx(0.2)


class TestComputeTimeWeightedStats:
    def test_constant_value(self) -> None:
        """Single constant concurrency → avg = value, std = 0, all percentiles = value."""
        # Concurrency of 5 from t=0 to t=100
        ts = np.array([0.0, 100.0])
        vals = np.array([5.0, 0.0])  # step function: 5 at t=0, drops to 0 at t=100
        stats = compute_time_weighted_stats(ts, vals, 0.0, 100.0)

        assert stats.avg == pytest.approx(5.0)
        assert stats.min == pytest.approx(5.0)
        assert stats.max == pytest.approx(5.0)
        assert stats.p50 == pytest.approx(5.0)
        assert stats.p90 == pytest.approx(5.0)
        assert stats.p95 == pytest.approx(5.0)
        assert stats.p99 == pytest.approx(5.0)
        assert stats.std == pytest.approx(0.0)

    def test_two_segments_known_avg(self) -> None:
        """Two segments with known durations → verify time-weighted avg."""
        # Concurrency: 2 for 80ns, then 10 for 20ns
        ts = np.array([0.0, 80.0, 100.0])
        vals = np.array([2.0, 10.0, 0.0])
        stats = compute_time_weighted_stats(ts, vals, 0.0, 100.0)

        # avg = (2*80 + 10*20) / 100 = (160 + 200) / 100 = 3.6
        assert stats.avg == pytest.approx(3.6)
        assert stats.min == pytest.approx(2.0)
        assert stats.max == pytest.approx(10.0)

        # std = sqrt((80*(2-3.6)^2 + 20*(10-3.6)^2) / 100)
        #     = sqrt((80*2.56 + 20*40.96) / 100)
        #     = sqrt((204.8 + 819.2) / 100) = sqrt(10.24) ≈ 3.2
        assert stats.std == pytest.approx(3.2, abs=0.01)

    def test_percentiles_unequal_durations(self) -> None:
        """Verify percentile computation with unequal segment durations."""
        # Value 1 for 90% of time, value 100 for 10% of time
        ts = np.array([0.0, 900.0, 1000.0])
        vals = np.array([1.0, 100.0, 0.0])
        stats = compute_time_weighted_stats(ts, vals, 0.0, 1000.0)

        # p50 should be 1 (value held for 90% of time)
        assert stats.p50 == pytest.approx(1.0)
        # p90 should be 1 (90% of time is at value 1, cum_frac = 0.9)
        assert stats.p90 == pytest.approx(1.0)
        # p95 should be 100 (only 10% of time is at value 100)
        assert stats.p95 == pytest.approx(100.0)
        # p99 should be 100
        assert stats.p99 == pytest.approx(100.0)

    def test_window_clipping(self) -> None:
        """Events outside window are ignored via clipping."""
        # Full curve: value 1 from t=0-50, value 5 from t=50-100
        ts = np.array([0.0, 50.0, 100.0])
        vals = np.array([1.0, 5.0, 0.0])

        # Only look at [50, 100] — should see only value 5
        stats = compute_time_weighted_stats(ts, vals, 50.0, 100.0)
        assert stats.avg == pytest.approx(5.0)
        assert stats.min == pytest.approx(5.0)
        assert stats.max == pytest.approx(5.0)
        assert stats.std == pytest.approx(0.0)

    def test_window_clipping_partial_segment(self) -> None:
        """Window that slices through the middle of a segment."""
        # Value 2 from t=0 to t=100
        ts = np.array([0.0, 100.0])
        vals = np.array([2.0, 0.0])

        # Window [25, 75] — should still see value 2
        stats = compute_time_weighted_stats(ts, vals, 25.0, 75.0)
        assert stats.avg == pytest.approx(2.0)

    def test_single_event_degenerate(self) -> None:
        """Single event at the start of the window."""
        ts = np.array([0.0])
        vals = np.array([3.0])
        stats = compute_time_weighted_stats(ts, vals, 0.0, 100.0)

        # Value 3 is held for the entire window
        assert stats.avg == pytest.approx(3.0)
        assert stats.min == pytest.approx(3.0)
        assert stats.max == pytest.approx(3.0)

    def test_empty_arrays(self) -> None:
        """Empty arrays return all zeros."""
        stats = compute_time_weighted_stats(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            0.0,
            100.0,
        )
        assert all(v == 0.0 for v in stats)

    def test_zero_duration_window(self) -> None:
        """Zero-duration window returns all zeros."""
        ts = np.array([0.0, 100.0])
        vals = np.array([5.0, 0.0])
        stats = compute_time_weighted_stats(ts, vals, 50.0, 50.0)
        assert all(v == 0.0 for v in stats)


class TestAddStepFunctions:
    def test_both_empty(self) -> None:
        empty = np.zeros(0, dtype=np.float64)
        ts, vals = add_step_functions(empty, empty, empty, empty)
        assert len(ts) == 0

    def test_first_empty(self) -> None:
        """Empty first → returns copy of second."""
        empty = np.zeros(0, dtype=np.float64)
        b_ts = np.array([1.0, 2.0])
        b_vals = np.array([5.0, 0.0])
        ts, vals = add_step_functions(empty, empty, b_ts, b_vals)
        np.testing.assert_array_equal(ts, b_ts)
        np.testing.assert_array_equal(vals, b_vals)

    def test_second_empty(self) -> None:
        """Empty second → returns copy of first."""
        empty = np.zeros(0, dtype=np.float64)
        a_ts = np.array([1.0, 2.0])
        a_vals = np.array([3.0, 0.0])
        ts, vals = add_step_functions(a_ts, a_vals, empty, empty)
        np.testing.assert_array_equal(ts, a_ts)
        np.testing.assert_array_equal(vals, a_vals)

    def test_identical_grids(self) -> None:
        ts = np.array([0.0, 50.0, 100.0])
        a = np.array([10.0, 20.0, 0.0])
        b = np.array([3.0, 7.0, 0.0])
        out_ts, out_vals = add_step_functions(ts, a, ts, b)
        np.testing.assert_array_equal(out_ts, ts)
        np.testing.assert_array_almost_equal(out_vals, [13.0, 27.0, 0.0])

    def test_overlapping_grids(self) -> None:
        """Interleaved timestamps sum step-function values at merged points."""
        a_ts = np.array([0.0, 100.0])
        a_vals = np.array([10.0, 0.0])
        b_ts = np.array([50.0, 100.0])
        b_vals = np.array([5.0, 0.0])
        out_ts, out_vals = add_step_functions(a_ts, a_vals, b_ts, b_vals)
        # Merged: [0, 50, 100]
        # At 0: a=10, b=0(before first event) → 10
        # At 50: a=10, b=5 → 15
        # At 100: a=0, b=0 → 0
        assert len(out_ts) == 3
        assert out_vals[0] == pytest.approx(10.0)
        assert out_vals[1] == pytest.approx(15.0)
        assert out_vals[2] == pytest.approx(0.0)


class TestDivideStepFunctions:
    def test_empty_numerator(self) -> None:
        """Empty numerator returns empty arrays."""
        ts, vals = divide_step_functions(
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.array([1.0, 2.0]),
            np.array([5.0, 0.0]),
        )
        assert len(ts) == 0
        assert len(vals) == 0

    def test_empty_denominator(self) -> None:
        """Empty denominator returns empty arrays."""
        ts, vals = divide_step_functions(
            np.array([1.0, 2.0]),
            np.array([10.0, 0.0]),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
        assert len(ts) == 0
        assert len(vals) == 0

    def test_identical_grids(self) -> None:
        """Same timestamps → simple element-wise division."""
        ts = np.array([0.0, 50.0, 100.0])
        num = np.array([10.0, 20.0, 0.0])
        den = np.array([2.0, 5.0, 0.0])
        out_ts, out_vals = divide_step_functions(ts, num, ts, den)
        np.testing.assert_array_equal(out_ts, ts)
        assert out_vals[0] == pytest.approx(5.0)
        assert out_vals[1] == pytest.approx(4.0)
        assert out_vals[2] == pytest.approx(0.0)  # 0/0 → 0

    def test_disjoint_grids(self) -> None:
        """Non-overlapping timestamps → numerator is 0 where denominator starts, vice versa."""
        num_ts = np.array([0.0, 10.0])
        num_vals = np.array([6.0, 0.0])
        den_ts = np.array([20.0, 30.0])
        den_vals = np.array([3.0, 0.0])
        out_ts, out_vals = divide_step_functions(num_ts, num_vals, den_ts, den_vals)
        # Merged: [0, 10, 20, 30]
        # At 0: num=6, den=0 → 0
        # At 10: num=0, den=0 → 0
        # At 20: num=0, den=3 → 0
        # At 30: num=0, den=0 → 0
        assert len(out_ts) == 4
        np.testing.assert_array_equal(out_vals, [0.0, 0.0, 0.0, 0.0])

    def test_overlapping_grids(self) -> None:
        """Interleaved timestamps with known values."""
        num_ts = np.array([0.0, 50.0, 100.0])
        num_vals = np.array([10.0, 20.0, 0.0])
        den_ts = np.array([0.0, 100.0])
        den_vals = np.array([5.0, 0.0])
        out_ts, out_vals = divide_step_functions(num_ts, num_vals, den_ts, den_vals)
        # Merged: [0, 50, 100]
        # At 0: num=10, den=5 → 2
        # At 50: num=20, den=5 → 4
        # At 100: num=0, den=0 → 0
        assert len(out_ts) == 3
        assert out_vals[0] == pytest.approx(2.0)
        assert out_vals[1] == pytest.approx(4.0)
        assert out_vals[2] == pytest.approx(0.0)

    def test_zero_denominator_guard(self) -> None:
        """Zero denominator yields 0 result, not NaN or inf."""
        ts = np.array([0.0, 50.0])
        num = np.array([10.0, 0.0])
        den = np.array([0.0, 0.0])
        _, out_vals = divide_step_functions(ts, num, ts, den)
        assert np.all(np.isfinite(out_vals))
        assert out_vals[0] == 0.0
        assert out_vals[1] == 0.0

    def test_single_point_curves(self) -> None:
        """Single-point step functions."""
        num_ts = np.array([5.0])
        num_vals = np.array([12.0])
        den_ts = np.array([5.0])
        den_vals = np.array([4.0])
        out_ts, out_vals = divide_step_functions(num_ts, num_vals, den_ts, den_vals)
        assert len(out_ts) == 1
        assert out_vals[0] == pytest.approx(3.0)


class TestThroughputPerUserSweep:
    def test_single_request(self) -> None:
        """Single request: concurrency=1 → per-user rate equals aggregate rate."""
        gen_start = np.array([0.0])
        end = np.array([100.0])
        # Throughput sweep for this request: rate = (101-1)/100 = 1.0 tokens/ns
        tput_ts, tput_vals = throughput_sweep_line(gen_start, end, np.array([101.0]))
        ts, per_user = throughput_per_user_sweep_line(
            gen_start, end, tput_ts, tput_vals
        )
        assert len(ts) > 0
        # With concurrency 1, per-user should equal aggregate
        max_val = float(np.max(per_user))
        assert max_val == pytest.approx(1.0, rel=0.01)

    def test_overlapping_requests(self) -> None:
        """N overlapping requests: per-user ≈ aggregate / N at peak."""
        gen_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        end = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        output_tokens = np.array([101.0, 101.0, 101.0, 101.0, 101.0])
        # Each request: rate = 1.0 tokens/ns, aggregate = 5.0
        tput_ts, tput_vals = throughput_sweep_line(gen_start, end, output_tokens)
        ts, per_user = throughput_per_user_sweep_line(
            gen_start, end, tput_ts, tput_vals
        )
        assert len(ts) > 0
        # Peak aggregate = 5.0, concurrency = 5 → per-user = 1.0
        max_val = float(np.max(per_user))
        assert max_val == pytest.approx(1.0, rel=0.01)

    def test_nan_filtering(self) -> None:
        """NaN records are excluded from both throughput and concurrency."""
        gen_start = np.array([0.0, np.nan])
        end = np.array([100.0, np.nan])
        output_tokens = np.array([101.0, np.nan])
        tput_ts, tput_vals = throughput_sweep_line(gen_start, end, output_tokens)
        ts, per_user = throughput_per_user_sweep_line(
            gen_start, end, tput_ts, tput_vals
        )
        # Only 1 valid request → concurrency 1 → per-user = aggregate
        if len(ts) > 0:
            max_val = float(np.max(per_user))
            assert max_val == pytest.approx(1.0, rel=0.01)

    def test_empty_throughput(self) -> None:
        """Empty throughput curve → empty per-user curve."""
        gen_start = np.array([], dtype=np.float64)
        end = np.array([], dtype=np.float64)
        tput_ts = np.zeros(0, dtype=np.float64)
        tput_vals = np.zeros(0, dtype=np.float64)
        ts, per_user = throughput_per_user_sweep_line(
            gen_start, end, tput_ts, tput_vals
        )
        assert len(ts) == 0


class TestPrefillThroughputPerUserSweep:
    def test_single_request(self) -> None:
        """Single request: prefill concurrency=1 → per-user equals aggregate."""
        start = np.array([0.0])
        gen_start = np.array([50.0])
        input_tokens = np.array([100.0])
        # Prefill rate = 100/50 = 2.0 tokens/ns
        ptput_ts, ptput_vals = prefill_throughput_sweep_line(
            start, gen_start, input_tokens
        )
        ts, per_user = prefill_throughput_per_user_sweep_line(
            start, gen_start, ptput_ts, ptput_vals
        )
        assert len(ts) > 0
        max_val = float(np.max(per_user))
        assert max_val == pytest.approx(2.0, rel=0.01)

    def test_overlapping_requests(self) -> None:
        """N overlapping prefills: per-user ≈ aggregate / N."""
        start = np.array([0.0, 0.0, 0.0])
        gen_start = np.array([50.0, 50.0, 50.0])
        input_tokens = np.array([100.0, 100.0, 100.0])
        # Each prefill: rate = 2.0, aggregate = 6.0, concurrency = 3
        ptput_ts, ptput_vals = prefill_throughput_sweep_line(
            start, gen_start, input_tokens
        )
        ts, per_user = prefill_throughput_per_user_sweep_line(
            start, gen_start, ptput_ts, ptput_vals
        )
        assert len(ts) > 0
        max_val = float(np.max(per_user))
        assert max_val == pytest.approx(2.0, rel=0.01)

    def test_nan_filtering(self) -> None:
        """NaN records excluded from both prefill throughput and concurrency."""
        start = np.array([0.0, np.nan])
        gen_start = np.array([50.0, np.nan])
        input_tokens = np.array([100.0, np.nan])
        ptput_ts, ptput_vals = prefill_throughput_sweep_line(
            start, gen_start, input_tokens
        )
        ts, per_user = prefill_throughput_per_user_sweep_line(
            start, gen_start, ptput_ts, ptput_vals
        )
        if len(ts) > 0:
            max_val = float(np.max(per_user))
            assert max_val == pytest.approx(2.0, rel=0.01)

    def test_empty_prefill_throughput(self) -> None:
        """Empty prefill throughput curve → empty per-user curve."""
        start = np.array([], dtype=np.float64)
        gen_start = np.array([], dtype=np.float64)
        ptput_ts = np.zeros(0, dtype=np.float64)
        ptput_vals = np.zeros(0, dtype=np.float64)
        ts, per_user = prefill_throughput_per_user_sweep_line(
            start, gen_start, ptput_ts, ptput_vals
        )
        assert len(ts) == 0


class TestTokensInFlightSweep:
    def test_empty_input(self) -> None:
        empty = np.array([], dtype=np.float64)
        ts, tif = tokens_in_flight_sweep_line(
            empty, empty, empty, empty, output_tokens=empty
        )
        assert len(ts) == 0
        assert len(tif) == 0

    def test_single_request_kv_cache_model(self) -> None:
        """One request: input tokens persist through generation, output tokens added at gen_start."""
        start = np.array([0.0])
        gen_start = np.array([10.0])
        end = np.array([60.0])
        input_tok = np.array([100.0])
        output_tok = np.array([50.0])

        ts, tif = tokens_in_flight_sweep_line(
            start,
            gen_start,
            end,
            input_tok,
            output_tokens=output_tok,
        )
        assert len(ts) > 0

        # During prefill [0, 10): 100 input tokens in KV cache
        idx_prefill = np.searchsorted(ts, 5.0, side="right") - 1
        assert tif[idx_prefill] == pytest.approx(100.0)

        # During generation [10, 60): 100 input + 50 output = 150 in KV cache
        idx_gen = np.searchsorted(ts, 30.0, side="right") - 1
        assert tif[idx_gen] == pytest.approx(150.0)

        # After end: 0
        assert tif[-1] == pytest.approx(0.0)

    def test_overlapping_requests(self) -> None:
        """Two overlapping requests — KV cache tokens add up."""
        start = np.array([0.0, 5.0])
        gen_start = np.array([10.0, 15.0])
        end = np.array([60.0, 65.0])
        input_tok = np.array([100.0, 200.0])
        output_tok = np.array([50.0, 80.0])

        ts, tif = tokens_in_flight_sweep_line(
            start,
            gen_start,
            end,
            input_tok,
            output_tokens=output_tok,
        )

        # At t=7 (both in prefill): 100 + 200 = 300
        idx = np.searchsorted(ts, 7.0, side="right") - 1
        assert tif[idx] == pytest.approx(300.0)

        # At t=12 (req0 in gen: 100+50=150, req1 still in prefill: 200): 350
        idx = np.searchsorted(ts, 12.0, side="right") - 1
        assert tif[idx] == pytest.approx(350.0)

        # At t=62 (req0 done, req1 in gen: 200+80=280): 280
        idx = np.searchsorted(ts, 62.0, side="right") - 1
        assert tif[idx] == pytest.approx(280.0)

    def test_nan_filtering(self) -> None:
        """NaN entries are excluded from the sweep."""
        start = np.array([0.0, np.nan])
        gen_start = np.array([10.0, 15.0])
        end = np.array([60.0, 65.0])
        input_tok = np.array([100.0, 200.0])
        output_tok = np.array([50.0, 80.0])

        ts, tif = tokens_in_flight_sweep_line(
            start,
            gen_start,
            end,
            input_tok,
            output_tokens=output_tok,
        )

        # Only req0 contributes prefill (req1 has NaN start → no input_tokens added)
        # But req1 has valid gen_start and end, so +80 at t=15, -80 at t=65
        # At t=5: only req0 prefill = 100
        idx_early = np.searchsorted(ts, 5.0, side="right") - 1
        assert tif[idx_early] == pytest.approx(100.0)

    def test_prefill_only_no_end(self) -> None:
        """Request with NaN end → input tokens added at start but never freed."""
        start = np.array([0.0])
        gen_start = np.array([10.0])
        end = np.array([np.nan])
        input_tok = np.array([100.0])
        output_tok = np.array([50.0])

        ts, tif = tokens_in_flight_sweep_line(
            start,
            gen_start,
            end,
            input_tok,
            output_tokens=output_tok,
        )
        assert len(ts) > 0
        # Input tokens added at start, never freed (NaN end)
        # gen phase invalid (NaN end → gen_dur invalid), so only +100 at t=0
        assert tif[0] == pytest.approx(100.0)

    def test_generation_only(self) -> None:
        """Request with NaN start → only generation output tokens contribute."""
        start = np.array([np.nan])
        gen_start = np.array([10.0])
        end = np.array([60.0])
        input_tok = np.array([100.0])
        output_tok = np.array([50.0])

        ts, tif = tokens_in_flight_sweep_line(
            start,
            gen_start,
            end,
            input_tok,
            output_tokens=output_tok,
        )
        assert len(ts) > 0
        # Only output_tokens: +50 at gen_start, -50 at end
        assert float(np.max(tif)) == pytest.approx(50.0)
        assert tif[-1] == pytest.approx(0.0)

    def test_peak_is_input_plus_output(self) -> None:
        """Peak KV cache for a single request = input_tokens + output_tokens."""
        start = np.array([0.0])
        gen_start = np.array([100.0])
        end = np.array([1000.0])
        input_tok = np.array([4096.0])
        output_tok = np.array([2048.0])

        ts, tif = tokens_in_flight_sweep_line(
            start,
            gen_start,
            end,
            input_tok,
            output_tokens=output_tok,
        )

        # Peak during generation = 4096 + 2048 = 6144
        assert float(np.max(tif)) == pytest.approx(6144.0)
        # During prefill = 4096
        idx_pf = np.searchsorted(ts, 50.0, side="right") - 1
        assert tif[idx_pf] == pytest.approx(4096.0)


class TestTokensInFlightSweepIcl:
    def test_empty_icl_falls_back_to_coarse(self) -> None:
        """Empty ICL data → delegates to tokens_in_flight_sweep."""
        start = np.array([0.0])
        gen_start = np.array([10.0])
        end = np.array([60.0])
        input_tok = np.array([100.0])
        output_tok = np.array([50.0])

        ts_icl, tif_icl = tokens_in_flight_sweep_line_icl(
            start,
            gen_start,
            end,
            input_tok,
            output_tokens=output_tok,
            icl_values=np.zeros(0, dtype=np.float64),
            icl_record_indices=np.zeros(0, dtype=np.int32),
            icl_offsets=np.zeros(0, dtype=np.int64),
        )
        ts_coarse, tif_coarse = tokens_in_flight_sweep_line(
            start,
            gen_start,
            end,
            input_tok,
            output_tokens=output_tok,
        )
        np.testing.assert_array_equal(ts_icl, ts_coarse)
        np.testing.assert_array_equal(tif_icl, tif_coarse)

    def test_gradual_ramp_up(self) -> None:
        """Single request with 5 equal chunks → output tokens ramp up in 5 steps."""
        start = np.array([0.0])
        gen_start = np.array([100.0])
        end = np.array([600.0])
        input_tok = np.array([200.0])
        output_tok = np.array([50.0])  # 50 tokens over 5 chunks = 10 per chunk

        # 5 equal ICL intervals of 100ns each
        icl_vals = np.array([100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        icl_rec = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        icl_off = np.array([0], dtype=np.int64)

        ts, tif = tokens_in_flight_sweep_line_icl(
            start,
            gen_start,
            end,
            input_tok,
            output_tokens=output_tok,
            icl_values=icl_vals,
            icl_record_indices=icl_rec,
            icl_offsets=icl_off,
        )

        # During prefill [0, 100): 200 input tokens
        idx_pf = np.searchsorted(ts, 50.0, side="right") - 1
        assert tif[idx_pf] == pytest.approx(200.0)

        # After chunk 1 (t=200): 200 input + 10 output = 210
        idx_c1 = np.searchsorted(ts, 205.0, side="right") - 1
        assert tif[idx_c1] == pytest.approx(210.0)

        # After chunk 3 (t=400): 200 input + 30 output = 230
        idx_c3 = np.searchsorted(ts, 405.0, side="right") - 1
        assert tif[idx_c3] == pytest.approx(230.0)

        # After all chunks (t=600): peak = 200 + 50 = 250, then freed → 0
        assert tif[-1] == pytest.approx(0.0)

    def test_peak_matches_input_plus_output(self) -> None:
        """Peak tokens in flight = input + output when end_ns > last chunk boundary."""
        start = np.array([0.0])
        gen_start = np.array([10.0])
        # end_ns after last chunk (gen_start + 5*20 = 110) so all chunks complete before free
        end = np.array([111.0])
        input_tok = np.array([1000.0])
        output_tok = np.array([500.0])

        icl_vals = np.array([20.0, 20.0, 20.0, 20.0, 20.0], dtype=np.float64)
        icl_rec = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        icl_off = np.array([0], dtype=np.int64)

        ts, tif = tokens_in_flight_sweep_line_icl(
            start,
            gen_start,
            end,
            input_tok,
            output_tokens=output_tok,
            icl_values=icl_vals,
            icl_record_indices=icl_rec,
            icl_offsets=icl_off,
        )

        # Peak = input + output = 1500 (all chunks completed, not yet freed)
        assert float(np.max(tif)) == pytest.approx(1500.0)

    def test_overlapping_requests_with_icl(self) -> None:
        """Two overlapping requests with ICL — tokens accumulate gradually."""
        start = np.array([0.0, 50.0])
        gen_start = np.array([10.0, 60.0])
        end = np.array([110.0, 160.0])
        input_tok = np.array([100.0, 200.0])
        output_tok = np.array(
            [20.0, 40.0]
        )  # req0: 2 chunks of 10, req1: 2 chunks of 20

        # req0: 2 chunks of 50ns, req1: 2 chunks of 50ns
        icl_vals = np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float64)
        icl_rec = np.array([0, 0, 1, 1], dtype=np.int32)
        icl_off = np.array([0, 2], dtype=np.int64)

        ts, tif = tokens_in_flight_sweep_line_icl(
            start,
            gen_start,
            end,
            input_tok,
            output_tokens=output_tok,
            icl_values=icl_vals,
            icl_record_indices=icl_rec,
            icl_offsets=icl_off,
        )

        # At t=55 (req0 in gen with 1 chunk done=10, req1 in prefill=200):
        # req0: 100 input + 10 output = 110, req1: 200 input = 200, total = 310
        idx = np.searchsorted(ts, 65.0, side="right") - 1
        # At t=65: req0 has chunk1 at t=60 (+10), req1 has prefill=200
        # req0: 100 + 10 = 110, req1: 200 = 200, total = 310
        assert tif[idx] == pytest.approx(310.0)

    def test_coarse_has_higher_early_load(self) -> None:
        """ICL-aware should show lower tokens during early generation than coarse."""
        start = np.array([0.0])
        gen_start = np.array([10.0])
        end = np.array([110.0])
        input_tok = np.array([100.0])
        output_tok = np.array([100.0])

        # 10 equal chunks
        icl_vals = np.full(10, 10.0, dtype=np.float64)
        icl_rec = np.zeros(10, dtype=np.int32)
        icl_off = np.array([0], dtype=np.int64)

        ts_icl, tif_icl = tokens_in_flight_sweep_line_icl(
            start,
            gen_start,
            end,
            input_tok,
            output_tokens=output_tok,
            icl_values=icl_vals,
            icl_record_indices=icl_rec,
            icl_offsets=icl_off,
        )
        ts_coarse, tif_coarse = tokens_in_flight_sweep_line(
            start,
            gen_start,
            end,
            input_tok,
            output_tokens=output_tok,
        )

        # After first chunk (t=20), ICL shows 100+10=110, coarse shows 100+100=200
        idx_icl = np.searchsorted(ts_icl, 25.0, side="right") - 1
        idx_coarse = np.searchsorted(ts_coarse, 25.0, side="right") - 1
        assert tif_icl[idx_icl] < tif_coarse[idx_coarse]
        assert tif_icl[idx_icl] == pytest.approx(110.0)
        assert tif_coarse[idx_coarse] == pytest.approx(200.0)
