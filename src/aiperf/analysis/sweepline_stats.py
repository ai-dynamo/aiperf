# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time-weighted statistics over sweep-line step functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aiperf.analysis.sweepline import (
    ZERO_SWEEP_LINE_STATS,
    FloatArray,
    SweepLineStats,
    _merge_sorted_unique,
    _step_lookup,
)
from aiperf.common.enums import MetricConsoleGroup
from aiperf.common.finite import scrub_non_finite
from aiperf.common.models import MetricResult

BoolArray = NDArray[np.bool_]


def _window_durations(
    sorted_ts: FloatArray, lo: int, hi: int, window_start: float, window_end: float
) -> FloatArray:
    """Durations of the segments [window_start, ts[lo], ..., ts[hi-1], window_end].

    Equals ``np.diff`` of that grid; the end segments are computed against the
    window edges directly so no clamp pass over the window is needed.
    """
    durations = np.empty(hi - lo + 1, dtype=np.float64)
    if hi == lo:
        durations[0] = window_end - window_start
        return durations
    durations[0] = sorted_ts[lo] - window_start
    np.subtract(sorted_ts[lo + 1 : hi], sorted_ts[lo : hi - 1], out=durations[1:-1])
    durations[-1] = window_end - sorted_ts[hi - 1]
    return durations


def _build_clipped_segments(
    sorted_ts: FloatArray,
    values: FloatArray,
    window_start: float,
    window_end: float,
) -> tuple[FloatArray, FloatArray]:
    """Slice the step function to [window_start, window_end] and return (durations, values).

    The value held at ``window_start`` is that of the last event at or before
    it (0 before the first event); zero-length segments from duplicate
    timestamps are dropped. Requires ``window_start < window_end``.
    """
    # Events strictly inside the window are ts[lo:hi]; event lo - 1 (or the
    # implicit zero before the first event) holds from window_start.
    lo = int(np.searchsorted(sorted_ts, window_start, side="right"))
    hi = int(np.searchsorted(sorted_ts, window_end, side="left"))
    durations = _window_durations(sorted_ts, lo, hi, window_start, window_end)
    seg_values = values[lo - 1 : hi] if lo > 0 else np.concatenate(([0.0], values[:hi]))
    mask = durations > 0
    return durations[mask], seg_values[mask]


def _segment_stats(
    val: FloatArray, dur: FloatArray, total_dur: float
) -> SweepLineStats:
    """Duration-weighted stats of ``val`` over segments of length ``dur``."""
    avg = float(np.sum(val * dur) / total_dur)
    mn = float(np.min(val))
    mx = float(np.max(val))
    # The run-level window spans every event, so square and weight the
    # deviations in place, drop the permutation once applied and turn the
    # cumulative duration into the CDF in place rather than holding a chain of
    # full-length temporaries.
    dev = val - avg
    np.square(dev, out=dev)
    np.multiply(dev, dur, out=dev)
    std = float(np.sqrt(np.sum(dev) / total_dur))
    del dev

    order = np.argsort(val)
    sorted_val = val[order]
    cum_dur = np.cumsum(dur[order])
    del order
    cum_dur /= cum_dur[-1]

    indices = np.searchsorted(cum_dur, [0.50, 0.90, 0.95, 0.99])
    np.minimum(indices, len(sorted_val) - 1, out=indices)
    p50, p90, p95, p99 = sorted_val[indices].tolist()

    return SweepLineStats(
        avg=avg, min=mn, max=mx, p50=p50, p90=p90, p95=p95, p99=p99, std=std
    )


def compute_time_weighted_stats(
    sorted_ts: FloatArray,
    values: FloatArray,
    window_start: float,
    window_end: float,
) -> SweepLineStats:
    """Compute time-weighted statistics over a step-function within a window.

    The sweep-line output defines a step function: value[i] is held from
    sorted_ts[i] to sorted_ts[i+1]. This function clips the step function
    to [window_start, window_end] and computes time-weighted stats.
    """
    total_dur = window_end - window_start
    if len(sorted_ts) == 0 or total_dur <= 0:
        return ZERO_SWEEP_LINE_STATS

    dur, val = _build_clipped_segments(sorted_ts, values, window_start, window_end)
    if dur.size == 0:
        return ZERO_SWEEP_LINE_STATS
    return _segment_stats(val, dur, total_dur)


@dataclass(frozen=True, slots=True)
class ActiveCurve:
    """A rate step function joined with an activity mask on their merged event grid.

    Built once per (rate, mask) pair so every window evaluation is a slice of
    the precomputed grid instead of a merge and two step lookups over the
    window's events.
    """

    grid: FloatArray
    """Sorted unique timestamps of both curves' events."""
    rate: FloatArray
    """Rate value held from each grid point."""
    active: BoolArray
    """Whether the mask is strictly positive from each grid point."""

    @classmethod
    def from_step_functions(
        cls,
        rate_ts: FloatArray,
        rate_vals: FloatArray,
        mask_ts: FloatArray,
        mask_vals: FloatArray,
    ) -> ActiveCurve:
        """Join a rate curve onto a mask curve.

        Either curve being empty yields an empty grid, which every window
        evaluates to ``ZERO_SWEEP_LINE_STATS``.
        """
        if len(rate_ts) == 0 or len(mask_ts) == 0:
            empty = np.zeros(0, dtype=np.float64)
            return cls(empty, empty, np.zeros(0, dtype=np.bool_))
        grid = _merge_sorted_unique(rate_ts, mask_ts)
        # A curve already defined on the merged grid (e.g. a per-user quotient
        # of these same events) needs no lookup, and no second copy.
        if len(grid) == len(rate_ts) and np.array_equal(grid, rate_ts):
            grid, rate = rate_ts, rate_vals
        else:
            rate = _step_lookup(rate_ts, rate_vals, grid)
        active = _step_lookup(mask_ts, mask_vals, grid) > 0
        return cls(grid, rate, active)

    def window_stats(self, window_start: float, window_end: float) -> SweepLineStats:
        """Time-weighted stats of the rate over the window's active segments."""
        total_dur = window_end - window_start
        if total_dur <= 0 or len(self.grid) == 0:
            return ZERO_SWEEP_LINE_STATS

        grid = self.grid
        lo = int(np.searchsorted(grid, window_start, side="right"))
        hi = int(np.searchsorted(grid, window_end, side="left"))
        durations = _window_durations(grid, lo, hi, window_start, window_end)

        # The segment starting at window_start holds the value of the last
        # grid point at or before it; the rest start at grid points.
        rate_at = np.empty(len(durations), dtype=np.float64)
        rate_at[0] = self.rate[lo - 1] if lo > 0 else 0.0
        rate_at[1:] = self.rate[lo:hi]
        active = np.empty(len(durations), dtype=np.bool_)
        active[0] = self.active[lo - 1] if lo > 0 else False
        active[1:] = self.active[lo:hi]
        active &= durations > 0
        if not active.any():
            return ZERO_SWEEP_LINE_STATS

        val = rate_at[active]
        dur = durations[active]
        active_dur = float(dur.sum())
        if active_dur <= 0:
            return ZERO_SWEEP_LINE_STATS
        return _segment_stats(val, dur, active_dur)


def compute_active_weighted_stats(
    rate_ts: FloatArray,
    rate_vals: FloatArray,
    mask_ts: FloatArray,
    mask_vals: FloatArray,
    window_start: float,
    window_end: float,
) -> SweepLineStats:
    """Time-weighted stats over a rate curve, restricted to segments where a
    mask curve is strictly positive.

    Useful for "phase-aware" throughput metrics: e.g. average decode
    throughput restricted to time periods when at least one record is in
    decode. Inactive segments (mask <= 0) are excluded from the weighted
    average and from the duration-weighted percentile CDF, so the result
    reflects intensity *while the phase is happening* rather than averaged
    over the whole run window.

    One-shot form of :meth:`ActiveCurve.window_stats`; callers evaluating
    many windows should build the :class:`ActiveCurve` once.

    Args:
        rate_ts: Sorted event timestamps of the rate step function.
        rate_vals: Rate values at each rate_ts (held until next event).
        mask_ts: Sorted event timestamps of the mask step function.
        mask_vals: Mask values at each mask_ts (held until next event).
        window_start: Left boundary of the analysis window.
        window_end: Right boundary of the analysis window.

    Returns:
        SweepLineStats over the active-only segments. Returns
        ZERO_SWEEP_LINE_STATS if no active segments overlap the window.
    """
    if window_end - window_start <= 0:
        return ZERO_SWEEP_LINE_STATS
    return ActiveCurve.from_step_functions(
        rate_ts, rate_vals, mask_ts, mask_vals
    ).window_stats(window_start, window_end)


def metric_result_from_sweep_line_stats(
    tag: str,
    header: str,
    unit: str,
    stats: SweepLineStats,
    *,
    scale: float = 1.0,
    console_group: MetricConsoleGroup = MetricConsoleGroup.EFFECTIVE,
) -> MetricResult:
    """Build a MetricResult from compute_time_weighted_stats output.

    Stats are scrubbed to finite-or-None: MetricResult feeds JSON/console
    exports, and NaN/inf leaking into them would silently serialize as null.
    """
    return MetricResult(
        tag=tag,
        header=header,
        unit=unit,
        avg=scrub_non_finite(stats.avg * scale),
        min=scrub_non_finite(stats.min * scale),
        max=scrub_non_finite(stats.max * scale),
        p50=scrub_non_finite(stats.p50 * scale),
        p90=scrub_non_finite(stats.p90 * scale),
        p95=scrub_non_finite(stats.p95 * scale),
        p99=scrub_non_finite(stats.p99 * scale),
        std=scrub_non_finite(stats.std * scale),
        console_group=console_group,
    )
