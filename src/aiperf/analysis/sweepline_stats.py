# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time-weighted statistics over sweep-line step functions."""

from __future__ import annotations

import numpy as np

from aiperf.analysis.sweepline import (
    ZERO_SWEEP_LINE_STATS,
    FloatArray,
    SweepLineStats,
)
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


def metric_result_from_sweep_line_stats(
    tag: str,
    header: str,
    unit: str,
    stats: SweepLineStats,
    *,
    scale: float = 1.0,
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
    )
