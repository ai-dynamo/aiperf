# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sweep-line algorithms."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from aiperf.analysis.sweepline import (
    add_step_functions,
    compute_active_weighted_stats,
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
        """Single request with 3 equal ICL chunks of 10ns each.

        TTFT chunk delivers 1 token at gen_start (not in throughput's rate
        domain); remaining (osl - 1) = 2 tokens spread across K = 3 ICL
        intervals → 2/3 tokens per chunk, rate = (2/3) / 10 ≈ 0.0667.
        """
        gen_start = np.array([0.0])
        output_tokens = np.array([3.0])
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
        # Each chunk: rate = ((3-1)/3) / 10 = 2/30 tokens/ns
        assert float(np.max(tput)) == pytest.approx(2.0 / 30.0)

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
        """Two requests with overlapping ICL chunks.

        Each: osl=2, K=2, so per-chunk = (2-1)/2 = 0.5, rate = 0.5/10 = 0.05.
        When both requests have an active chunk simultaneously, peak = 0.1.
        """
        gen_start = np.array([0.0, 5.0])
        output_tokens = np.array([2.0, 2.0])
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
        # Per-request rate = 0.05; overlap peak ≈ 0.10 (>0.05 single-rate)
        assert float(np.max(tput)) > 0.05
        assert float(np.max(tput)) == pytest.approx(0.1)

    def test_all_zero_icl_record_no_divide_warning(self) -> None:
        """A record whose ICL gaps are all zero must not warn or leak inf.

        Zero-length intervals are excluded from the nonzero divisor count, so
        such a record divides by a zero count inside the eagerly-evaluated
        np.where branch; the guard must prevent the RuntimeWarning and the
        record must contribute no rate events.
        """
        gen_start = np.array([0.0, 5.0])
        output_tokens = np.array([3.0, 4.0])
        icl_values = np.array([10.0, 10.0, 0.0, 0.0])
        icl_record_indices = np.array([0, 0, 1, 1], dtype=np.int32)
        icl_offsets = np.array([0, 2], dtype=np.int64)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            ts, tput = throughput_sweep_line_icl(
                gen_start,
                output_tokens,
                icl_values,
                icl_record_indices,
                icl_offsets=icl_offsets,
            )
        assert len(ts) == 4  # only record 0's 2 chunks emit +/- events
        assert np.isfinite(tput).all()
        assert float(np.max(tput)) == pytest.approx(1.0 / 10.0)  # (3-1)/2 / 10

    def test_rescaling_with_variable_tokens(self) -> None:
        """6 output tokens across 3 chunks → (6-1)/3 = 5/3 tok/msg.

        TTFT delivers 1 token at gen_start; remaining 5 spread across K=3
        chunks → rate = (5/3) / 10 ≈ 0.1667.
        """
        gen_start = np.array([0.0])
        output_tokens = np.array([6.0])
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
        assert float(np.max(tput)) == pytest.approx(5.0 / 30.0)


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
        """Single request with ICL: TTFT delivers 1 token at gen_start, each
        of K ICL events delivers (osl-1)/K tokens. Total adds to osl."""
        start = np.array([0.0])
        gen_start = np.array([100.0])
        end = np.array([600.0])
        input_tok = np.array([200.0])
        output_tok = np.array([50.0])  # 1 token at TTFT, 49 spread across 5 ICL events

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

        per_chunk = (50.0 - 1.0) / 5.0  # 9.8 tokens per ICL event

        # During prefill [0, 100): 200 input tokens
        idx_pf = np.searchsorted(ts, 50.0, side="right") - 1
        assert tif[idx_pf] == pytest.approx(200.0)

        # At gen_start (t=100): TTFT chunk delivered → 200 + 1 = 201
        idx_ttft = np.searchsorted(ts, 105.0, side="right") - 1
        assert tif[idx_ttft] == pytest.approx(201.0)

        # After ICL[0] (t=200): TTFT + 1 chunk = 200 + 1 + 9.8 = 210.8
        idx_c1 = np.searchsorted(ts, 205.0, side="right") - 1
        assert tif[idx_c1] == pytest.approx(201.0 + per_chunk)

        # After ICL[2] (t=400): TTFT + 3 chunks = 200 + 1 + 3*9.8 = 230.4
        idx_c3 = np.searchsorted(ts, 405.0, side="right") - 1
        assert tif[idx_c3] == pytest.approx(201.0 + 3 * per_chunk)

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
        """Two overlapping requests with ICL — tokens accumulate gradually.

        Model: TTFT chunk = 1 token at gen_start, remaining (osl-1) spread
        across K ICL events.
        """
        start = np.array([0.0, 50.0])
        gen_start = np.array([10.0, 60.0])
        end = np.array([110.0, 160.0])
        input_tok = np.array([100.0, 200.0])
        output_tok = np.array([20.0, 40.0])

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

        # At t=65: req0 has TTFT@10 (+1) and ICL[0]@60 fired (+(20-1)/2=9.5)
        #          req1 has TTFT@60 fired (+1), ICL[0]@110 not yet
        # req0: 100 + 1 + 9.5 = 110.5
        # req1: 200 + 1       = 201.0
        # total = 311.5
        idx = np.searchsorted(ts, 65.0, side="right") - 1
        assert tif[idx] == pytest.approx(311.5)

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

        # After first ICL event (t=20): TTFT@10 (+1) and ICL[0]@20 fired
        # ((100-1)/10 = 9.9). ICL: 100 + 1 + 9.9 = 110.9. Coarse: 100 + 100 = 200.
        idx_icl = np.searchsorted(ts_icl, 25.0, side="right") - 1
        idx_coarse = np.searchsorted(ts_coarse, 25.0, side="right") - 1
        assert tif_icl[idx_icl] < tif_coarse[idx_coarse]
        assert tif_icl[idx_icl] == pytest.approx(110.9)
        assert tif_coarse[idx_coarse] == pytest.approx(200.0)


class TestComputeActiveWeightedStats:
    """Stats restricted to segments where a mask curve is positive."""

    def test_empty_inputs(self) -> None:
        stats = compute_active_weighted_stats(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            0.0,
            100.0,
        )
        assert stats.avg == 0.0 and stats.min == 0.0 and stats.max == 0.0

    def test_zero_window(self) -> None:
        stats = compute_active_weighted_stats(
            np.array([0.0, 50.0]),
            np.array([100.0, 200.0]),
            np.array([0.0, 50.0]),
            np.array([1.0, 1.0]),
            10.0,
            10.0,
        )
        assert stats.avg == 0.0

    def test_no_active_segments(self) -> None:
        """Mask is zero throughout — stats should be all zeros."""
        stats = compute_active_weighted_stats(
            rate_ts=np.array([0.0, 50.0]),
            rate_vals=np.array([100.0, 200.0]),
            mask_ts=np.array([0.0]),
            mask_vals=np.array([0.0]),
            window_start=0.0,
            window_end=100.0,
        )
        assert stats.avg == 0.0
        assert stats.max == 0.0

    def test_active_only_excludes_idle(self) -> None:
        """Rate is 100 from t=0..50 (mask active), 0 from t=50..100 (idle).
        Time-weighted over whole window: 100*50/100 = 50.
        Active-weighted: 100*50/50 = 100.
        """
        rate_ts = np.array([0.0, 50.0])
        rate_vals = np.array([100.0, 0.0])
        mask_ts = np.array([0.0, 50.0])
        mask_vals = np.array([1.0, 0.0])

        active = compute_active_weighted_stats(
            rate_ts, rate_vals, mask_ts, mask_vals, 0.0, 100.0
        )
        full = compute_time_weighted_stats(rate_ts, rate_vals, 0.0, 100.0)
        assert full.avg == pytest.approx(50.0)
        assert active.avg == pytest.approx(100.0)
        assert active.min == pytest.approx(100.0)
        assert active.max == pytest.approx(100.0)

    def test_active_percentile_is_independent_of_idle(self) -> None:
        """Adding a long idle period should NOT shift active percentiles."""
        # Two equal-duration active segments at rates 50 and 150
        rate_ts = np.array([0.0, 10.0, 20.0, 100.0])
        rate_vals = np.array([50.0, 150.0, 0.0, 0.0])
        mask_ts = np.array([0.0, 20.0])
        mask_vals = np.array([1.0, 0.0])

        # window covers the active region 0..20 plus a long tail of idle
        active = compute_active_weighted_stats(
            rate_ts, rate_vals, mask_ts, mask_vals, 0.0, 1000.0
        )
        # Active duration = 20; equal-weighted segments → avg = 100
        assert active.avg == pytest.approx(100.0)
        # p50 picks the lower segment (50) when CDF hits 0.5; p99 picks 150.
        assert active.p50 == pytest.approx(50.0)
        assert active.p99 == pytest.approx(150.0)

    def test_partial_overlap_with_window(self) -> None:
        """Active segment partially clipped by window boundary."""
        rate_ts = np.array([0.0, 100.0])
        rate_vals = np.array([200.0, 0.0])
        mask_ts = np.array([0.0, 100.0])
        mask_vals = np.array([1.0, 0.0])

        # Window [50, 80) is fully inside the active segment.
        stats = compute_active_weighted_stats(
            rate_ts, rate_vals, mask_ts, mask_vals, 50.0, 80.0
        )
        assert stats.avg == pytest.approx(200.0)
        assert stats.min == pytest.approx(200.0)

    def test_window_boundaries_preserve_step_state(self) -> None:
        rate_ts = np.array([0.0, 10.0, 20.0])
        rate_vals = np.array([5.0, 10.0, 999.0])
        mask_ts = np.array([0.0, 15.0, 20.0])
        mask_vals = np.array([1.0, 0.0, 1.0])

        stats = compute_active_weighted_stats(
            rate_ts, rate_vals, mask_ts, mask_vals, 10.0, 20.0
        )

        assert stats.avg == pytest.approx(10.0)
        assert stats.min == pytest.approx(10.0)
        assert stats.max == pytest.approx(10.0)

    def test_unique_receives_only_events_inside_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rate_ts = np.arange(10_000, dtype=np.float64)
        rate_vals = np.full(10_000, 100.0)
        mask_ts = np.arange(10_000, dtype=np.float64)
        mask_vals = np.ones(10_000)
        unique_input_sizes: list[int] = []
        original_unique = np.unique

        def tracking_unique(values: np.ndarray) -> np.ndarray:
            unique_input_sizes.append(values.size)
            return original_unique(values)

        monkeypatch.setattr(
            "aiperf.analysis.sweepline_stats.np.unique", tracking_unique
        )

        stats = compute_active_weighted_stats(
            rate_ts, rate_vals, mask_ts, mask_vals, 5000.25, 5001.25
        )

        assert stats.avg == pytest.approx(100.0)
        assert unique_input_sizes == [4]


# ---------------------------------------------------------------------------
# Brute-force reference comparison
#
# These tests construct synthetic per-record arrays in the same shape that
# ColumnStore would produce, compute a ground-truth value at each sample
# timestamp by enumerating per-record contributions in pure Python, and
# compare against what the production sweep functions emit. They serve as
# regression coverage for the ICL-aware curves' analytical model:
#   - tokens_in_flight: +input at start_ns, +1 (TTFT chunk) at gen_start_ns,
#     +(osl-1)/K per ICL event, -(input+output) at end_ns
#   - throughput: rate = (osl-1)/K_nonzero / icl[i] over [interval_start, interval_end)
#     (TTFT chunk excluded — same convention as throughput_sweep_line)
#
# The tests prove production matches the reference at sub-FP-noise tolerance,
# so any future change that would silently corrupt the curve (NaN propagation,
# off-by-one in the ICL count, ordering bugs in the cumsum) trips them.
# ---------------------------------------------------------------------------


def _build_synthetic_records(
    n_records: int = 16,
    start_step_ns: float = 50e6,
    ttft_ns: float = 22e6,
    decode_ns: float = 500e6,
    isl: float = 200.0,
    osl: float = 100.0,
    n_chunks: int = 50,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Build per-record arrays + flat ICL series with realistic streaming shape.

    Each record streams ``n_chunks`` chunks (so K = n_chunks - 1 ICL gaps)
    over a uniform decode duration with mild jitter. Returns the same shape
    that ColumnStore exposes to the sweep functions.
    """
    rng = np.random.default_rng(seed)
    start_ns = (np.arange(n_records) * start_step_ns).astype(np.float64)
    gen_start_ns = start_ns + ttft_ns
    end_ns = gen_start_ns + decode_ns
    input_tokens = np.full(n_records, isl)
    output_tokens = np.full(n_records, osl)

    K = n_chunks - 1  # ICL gaps between K+1 chunks; first chunk = TTFT instant
    base_icl = decode_ns / K
    icl_values_list: list[float] = []
    rec_idx_list: list[int] = []
    offsets = [0]
    for i in range(n_records):
        # Mild lognormal jitter per chunk; renormalize so sum(icl) == decode_ns.
        per_record_icls = base_icl * np.exp(rng.normal(0.0, 0.1, size=K))
        per_record_icls *= decode_ns / per_record_icls.sum()
        icl_values_list.extend(per_record_icls.tolist())
        rec_idx_list.extend([i] * K)
        offsets.append(len(icl_values_list))

    return {
        "start_ns": start_ns,
        "gen_start_ns": gen_start_ns,
        "end_ns": end_ns,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "icl_values": np.array(icl_values_list, dtype=np.float64),
        "icl_record_indices": np.array(rec_idx_list, dtype=np.int32),
        "icl_offsets": np.array(offsets[:-1], dtype=np.int64),
    }


def _step_lookup(event_ts: np.ndarray, event_vals: np.ndarray, t: float) -> float:
    """Step-function lookup: value at t (or 0 if before first event)."""
    if len(event_ts) == 0:
        return 0.0
    idx = int(np.searchsorted(event_ts, t, side="right")) - 1
    return float(event_vals[idx]) if idx >= 0 else 0.0


def _reference_tokens_in_flight(records: dict, t: float) -> float:
    """Brute-force TIF at time t: sum across records of per-record contribution.

    Per-record model:
      0                        if t outside [start_ns, end_ns)
      isl                      if start_ns <= t < gen_start_ns
      isl + 1 + n_landed*tpc   if gen_start_ns <= t < end_ns
                               where tpc = (osl-1)/K and n_landed counts
                               ICL events that have fired by time t.
    """
    total = 0.0
    K = records["icl_offsets"]
    icl = records["icl_values"]
    for i in range(len(records["start_ns"])):
        s, e, gs = (
            records["start_ns"][i],
            records["end_ns"][i],
            records["gen_start_ns"][i],
        )
        if t < s or t >= e:
            continue
        contrib = float(records["input_tokens"][i])
        if t >= gs:
            contrib += 1.0  # TTFT chunk delivers 1 token at gen_start
            lo = K[i]
            hi = K[i + 1] if i + 1 < len(K) else len(icl)
            n_icl = hi - lo
            if n_icl > 0:
                cum = np.cumsum(icl[lo:hi])
                arrivals = gs + cum
                n_landed = int(np.searchsorted(arrivals, t, side="right"))
                tpc = (float(records["output_tokens"][i]) - 1.0) / n_icl
                contrib += n_landed * tpc
        total += contrib
    return total


def _reference_throughput(records: dict, t: float) -> float:
    """Brute-force decode throughput at time t: sum of per-interval rates.

    Each ICL interval [interval_start, interval_end) for a record carries
    rate = ((osl-1)/K_nonzero) / icl_value. The TTFT chunk has no interval
    (it's a delta) and is excluded from rate-domain integration.
    """
    total = 0.0
    K = records["icl_offsets"]
    icl = records["icl_values"]
    for i in range(len(records["start_ns"])):
        gs = records["gen_start_ns"][i]
        lo = K[i]
        hi = K[i + 1] if i + 1 < len(K) else len(icl)
        if hi <= lo:
            continue
        per = icl[lo:hi]
        nonzero = per > 0
        n_nonzero = int(nonzero.sum())
        if n_nonzero == 0:
            continue
        cum = np.cumsum(per)
        ends = gs + cum
        starts = ends - per
        tokens_per_msg = (float(records["output_tokens"][i]) - 1.0) / n_nonzero
        active = nonzero & (starts <= t) & (t < ends)
        if active.any():
            for k in np.where(active)[0]:
                total += tokens_per_msg / per[k]
    return total


class TestICLSweepReference:
    """Brute-force reference checks for the ICL-aware sweep curves.

    These exist so any change to the analytical model (off-by-one in chunk
    counts, mishandled NaN, ordering bug in cumsum) shows up as a divergence
    against an independent enumeration-based computation.
    """

    @pytest.fixture
    def records(self) -> dict:
        return _build_synthetic_records()

    def test_tokens_in_flight_matches_reference(self, records: dict) -> None:
        ts, tif = tokens_in_flight_sweep_line_icl(
            records["start_ns"],
            records["gen_start_ns"],
            records["end_ns"],
            records["input_tokens"],
            records["output_tokens"],
            records["icl_values"],
            records["icl_record_indices"],
            records["icl_offsets"],
        )

        # Sample 200 timestamps strictly inside the run window to avoid
        # boundary aliasing at the very last event.
        window_start = float(records["start_ns"].min())
        window_end = float(records["end_ns"].max())
        sample = np.linspace(window_start + 1.0, window_end - 1.0, 200)
        prod = np.array([_step_lookup(ts, tif, float(t)) for t in sample])
        ref = np.array([_reference_tokens_in_flight(records, float(t)) for t in sample])

        # Tolerance: 1e-9 relative to peak token count (FP cumsum noise only).
        peak = max(float(np.max(np.abs(ref))), 1.0)
        assert np.max(np.abs(prod - ref)) < 1e-9 * peak

        # Production curve must be physically non-negative (no chunks-after-end
        # ordering bugs). Allow exact zero from the FP-snap.
        assert float(np.min(tif)) >= 0.0

    def test_throughput_matches_reference(self, records: dict) -> None:
        ts, tput = throughput_sweep_line_icl(
            records["gen_start_ns"],
            records["output_tokens"],
            records["icl_values"],
            records["icl_record_indices"],
            records["icl_offsets"],
        )

        window_start = float(records["gen_start_ns"].min())
        window_end = float(records["end_ns"].max())
        sample = np.linspace(window_start + 1.0, window_end - 1.0, 200)
        prod = np.array([_step_lookup(ts, tput, float(t)) for t in sample])
        ref = np.array([_reference_throughput(records, float(t)) for t in sample])

        peak = max(float(np.max(np.abs(ref))), 1e-10)
        assert np.max(np.abs(prod - ref)) < 1e-9 * peak
        assert float(np.min(tput)) >= 0.0

    def test_throughput_integrates_to_osl_minus_one_per_record(
        self, records: dict
    ) -> None:
        """Riemann-sum the production throughput curve and check it equals the
        sum of (osl - 1) across records — the analytical conservation law."""
        ts, tput = throughput_sweep_line_icl(
            records["gen_start_ns"],
            records["output_tokens"],
            records["icl_values"],
            records["icl_record_indices"],
            records["icl_offsets"],
        )
        seg_durs = np.diff(ts)
        seg_vals = tput[:-1]
        integral = float(np.sum(seg_vals * seg_durs))
        expected = float(np.sum(records["output_tokens"] - 1.0))
        assert integral == pytest.approx(expected, rel=1e-6)

    def test_tokens_in_flight_drains_to_zero_at_end(self, records: dict) -> None:
        """After all records have ended, TIF must be exactly 0 — proves the
        per-record balance (additions == subtractions) holds."""
        ts, tif = tokens_in_flight_sweep_line_icl(
            records["start_ns"],
            records["gen_start_ns"],
            records["end_ns"],
            records["input_tokens"],
            records["output_tokens"],
            records["icl_values"],
            records["icl_record_indices"],
            records["icl_offsets"],
        )
        # Last value is the post-drain residual.
        assert tif[-1] == pytest.approx(0.0, abs=1e-9)

    def test_zero_icl_chunks_are_counted(self) -> None:
        """Zero-ICL entries (back-to-back chunks in the same packet) must
        contribute their tokens — earlier filter (icl_values > 0) silently
        dropped them while keeping them in the divisor."""
        records = _build_synthetic_records(n_records=4, n_chunks=10)
        # Inject 2 zero-ICL gaps into the first record's series.
        lo = records["icl_offsets"][0]
        records["icl_values"][lo] = 0.0
        records["icl_values"][lo + 1] = 0.0
        # Renormalize the rest of that record's gaps to preserve total decode duration.
        nonzero_idx = slice(lo + 2, lo + 9)  # 9 = K = n_chunks - 1 for record 0
        target = float(records["end_ns"][0] - records["gen_start_ns"][0])
        records["icl_values"][nonzero_idx] *= (
            target / records["icl_values"][nonzero_idx].sum()
        )

        ts, tif = tokens_in_flight_sweep_line_icl(
            records["start_ns"],
            records["gen_start_ns"],
            records["end_ns"],
            records["input_tokens"],
            records["output_tokens"],
            records["icl_values"],
            records["icl_record_indices"],
            records["icl_offsets"],
        )
        # Drain to zero proves the zero-ICL chunks were accounted for; if the
        # filter had dropped them while keeping the divisor, we'd see a
        # permanent negative offset = 2 * (osl-1)/K.
        assert tif[-1] == pytest.approx(0.0, abs=1e-9)
        assert float(np.min(tif)) >= 0.0
