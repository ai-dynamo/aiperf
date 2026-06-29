# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time-weighted statistics over sweep-line step functions."""

from __future__ import annotations

import numpy as np

from aiperf.analysis.sweepline import (
    ZERO_SWEEP_LINE_STATS,
    FloatArray,
    SweepLineStats,
    _step_lookup,
)
from aiperf.common.enums import MetricConsoleGroup
from aiperf.common.models import MetricResult


def _build_clipped_segments(
    sorted_ts: FloatArray,
    values: FloatArray,
    window_start: float,
    window_end: float,
) -> tuple[FloatArray, FloatArray]:
    """Slice the step function to [window_start, window_end] and return (durations, values)."""
    lo = max(0, int(np.searchsorted(sorted_ts, window_start, side="right")) - 1)
    hi = min(
        len(sorted_ts), int(np.searchsorted(sorted_ts, window_end, side="left")) + 1
    )
    ts_slice = sorted_ts[lo:hi]
    val_slice = values[lo:hi]

    n_s = len(ts_slice)
    seg_starts = np.empty(n_s + 1, dtype=np.float64)
    seg_values = np.empty(n_s + 1, dtype=np.float64)

    seg_starts[0] = window_start
    seg_values[0] = float(values[lo - 1]) if lo > 0 else 0.0
    seg_starts[1:] = ts_slice
    seg_values[1:] = val_slice

    seg_ends = np.empty(n_s + 1, dtype=np.float64)
    seg_ends[:-1] = seg_starts[1:]
    seg_ends[-1] = window_end

    seg_starts = np.maximum(seg_starts, window_start)
    seg_ends = np.minimum(seg_ends, window_end)
    durations = np.maximum(seg_ends - seg_starts, 0.0)

    mask = durations > 0
    return durations[mask], seg_values[mask]


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

    avg = float(np.sum(val * dur) / total_dur)
    mn = float(np.min(val))
    mx = float(np.max(val))
    std = float(np.sqrt(np.sum(dur * (val - avg) ** 2) / total_dur))

    order = np.argsort(val)
    sorted_val = val[order]
    sorted_dur = dur[order]
    cum_dur = np.cumsum(sorted_dur)
    cum_frac = cum_dur / cum_dur[-1]

    indices = np.searchsorted(cum_frac, [0.50, 0.90, 0.95, 0.99])
    np.minimum(indices, len(sorted_val) - 1, out=indices)
    p50, p90, p95, p99 = sorted_val[indices].tolist()

    return SweepLineStats(
        avg=avg, min=mn, max=mx, p50=p50, p90=p90, p95=p95, p99=p99, std=std
    )


def _build_active_window_grid(
    rate_ts: FloatArray,
    mask_ts: FloatArray,
    window_start: float,
    window_end: float,
) -> FloatArray:
    rate_lo = int(np.searchsorted(rate_ts, window_start, side="right"))
    rate_hi = int(np.searchsorted(rate_ts, window_end, side="left"))
    mask_lo = int(np.searchsorted(mask_ts, window_start, side="right"))
    mask_hi = int(np.searchsorted(mask_ts, window_end, side="left"))

    return np.unique(
        np.concatenate(
            [
                np.array([window_start, window_end], dtype=np.float64),
                rate_ts[rate_lo:rate_hi],
                mask_ts[mask_lo:mask_hi],
            ]
        )
    )


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
    total_dur = window_end - window_start
    if total_dur <= 0 or len(rate_ts) == 0:
        return ZERO_SWEEP_LINE_STATS

    # Window edges cover exact-boundary events; _step_lookup resolves the value
    # held from any predecessor outside the window.
    grid = _build_active_window_grid(rate_ts, mask_ts, window_start, window_end)
    if len(grid) < 2:
        return ZERO_SWEEP_LINE_STATS

    seg_starts = grid[:-1]
    seg_durations = np.diff(grid)
    rate_at = _step_lookup(rate_ts, rate_vals, seg_starts)
    mask_at = _step_lookup(mask_ts, mask_vals, seg_starts)

    active = (mask_at > 0) & (seg_durations > 0)
    if not active.any():
        return ZERO_SWEEP_LINE_STATS

    val = rate_at[active]
    dur = seg_durations[active]
    active_dur = float(dur.sum())
    if active_dur <= 0:
        return ZERO_SWEEP_LINE_STATS

    avg = float(np.sum(val * dur) / active_dur)
    mn = float(np.min(val))
    mx = float(np.max(val))
    std = float(np.sqrt(np.sum(dur * (val - avg) ** 2) / active_dur))

    order = np.argsort(val)
    sorted_val = val[order]
    sorted_dur = dur[order]
    cum_dur = np.cumsum(sorted_dur)
    cum_frac = cum_dur / cum_dur[-1]
    indices = np.searchsorted(cum_frac, [0.50, 0.90, 0.95, 0.99])
    np.minimum(indices, len(sorted_val) - 1, out=indices)
    p50, p90, p95, p99 = sorted_val[indices].tolist()

    return SweepLineStats(
        avg=avg, min=mn, max=mx, p50=p50, p90=p90, p95=p95, p99=p99, std=std
    )


def metric_result_from_sweep_line_stats(
    tag: str,
    header: str,
    unit: str,
    stats: SweepLineStats,
    *,
    scale: float = 1.0,
    console_group: MetricConsoleGroup = MetricConsoleGroup.EFFECTIVE,
) -> MetricResult:
    """Build a MetricResult from compute_time_weighted_stats output."""
    return MetricResult(
        tag=tag,
        header=header,
        unit=unit,
        avg=stats.avg * scale,
        min=stats.min * scale,
        max=stats.max * scale,
        p50=stats.p50 * scale,
        p90=stats.p90 * scale,
        p95=stats.p95 * scale,
        p99=stats.p99 * scale,
        std=stats.std * scale,
        console_group=console_group,
    )
