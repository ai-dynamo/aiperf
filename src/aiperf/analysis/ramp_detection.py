# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ramp boundary detection from concurrency/throughput curves."""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DetectionResult(NamedTuple):
    """Result from steady-state window detection."""

    window_start: float
    window_end: float
    detection_method: str


def cusum_steady_state_window(
    sorted_ts: NDArray[np.float64],
    concurrency: NDArray[np.float64],
    min_window_pct: float = 10.0,
) -> tuple[float, float]:
    """Detect ramp boundaries using time-weighted retrospective CUSUM.

    The concurrency sweep output is a step function: concurrency[i] is held
    from sorted_ts[i] to sorted_ts[i+1]. Each deviation from the target is
    weighted by its segment duration so that concurrency levels held for
    longer periods contribute proportionally more to the CUSUM statistic.

    The target is the time-weighted p95 of concurrency (robust to brief
    overshoots). The ramp-up end is where the time-weighted cumulative
    deficit is maximized (argmin of CUSUM). The ramp-down start is the
    mirror from the right.

    Args:
        sorted_ts: Sorted event timestamps from concurrency_sweep().
        concurrency: Concurrency values at each event boundary.
        min_window_pct: Minimum window size as % of total; below this, fall back to full.

    Returns:
        (window_start, window_end) — wall-clock timestamps.
    """
    if len(sorted_ts) == 0:
        return 0.0, 0.0

    min_ts = float(sorted_ts[0])
    max_ts = float(sorted_ts[-1])
    total_duration = max_ts - min_ts
    if total_duration <= 0:
        return min_ts, max_ts

    # Segment durations: concurrency[i] is held from sorted_ts[i] to sorted_ts[i+1]
    durations = np.empty(len(sorted_ts), dtype=np.float64)
    durations[:-1] = sorted_ts[1:] - sorted_ts[:-1]
    durations[-1] = 0.0  # terminal event has no forward duration

    # Time-weighted p95 target (concurrency below which 95% of time is spent)
    positive_mask = durations > 0
    if not positive_mask.any():
        return min_ts, max_ts

    tw_order = np.argsort(concurrency[positive_mask])
    tw_conc = concurrency[positive_mask][tw_order]
    tw_dur = durations[positive_mask][tw_order]
    cum_frac = np.cumsum(tw_dur) / tw_dur.sum()
    p95_idx = int(np.searchsorted(cum_frac, 0.95))
    target = float(tw_conc[min(p95_idx, len(tw_conc) - 1)])

    # Time-weighted CUSUM: deviation × segment duration
    # argmin of forward cumsum = point of maximum cumulative time-deficit = ramp-up end
    deviations = (concurrency - target) * durations
    forward_cusum = np.cumsum(deviations)
    ramp_up_idx = int(np.argmin(forward_cusum))

    # Backward CUSUM: reverse the time-weighted deviations, find ramp-down start
    backward_cusum = np.cumsum(deviations[::-1])
    ramp_down_offset = int(np.argmin(backward_cusum))
    ramp_down_idx = len(concurrency) - 1 - ramp_down_offset

    # Ensure valid ordering
    if ramp_up_idx >= ramp_down_idx:
        return min_ts, max_ts

    window_start = float(sorted_ts[ramp_up_idx])
    window_end = float(sorted_ts[ramp_down_idx])

    # min_window_pct safety guard
    if (window_end - window_start) < total_duration * min_window_pct / 100.0:
        logger.warning(
            "CUSUM steady-state window (%.1f%%) below minimum (%.1f%%), "
            "falling back to full range",
            (window_end - window_start) / total_duration * 100,
            min_window_pct,
        )
        return min_ts, max_ts

    return window_start, window_end


def mser5_truncation_point(
    values: NDArray[np.float64],
) -> int:
    """Find the optimal truncation point using MSER-5.

    Batches observations into groups of 5, computes the MSER statistic
    for each candidate truncation point, and returns the point that
    minimizes it. MSER = MSE of the truncated mean.

    Args:
        values: Time-ordered metric observations (NaN-free).

    Returns:
        Truncation index into the original (unbatched) array.
        0 means no truncation needed (already stationary).
    """
    if len(values) < 10:
        return 0

    # Batch into groups of 5
    batch_size = 5
    n_full = (len(values) // batch_size) * batch_size
    batched = values[:n_full].reshape(-1, batch_size).mean(axis=1)
    m = len(batched)

    if m < 4:
        return 0

    # Reverse cumulative sums for O(m) total
    # MSER(d) = variance(batched[d:]) / (m - d)
    max_d = m // 2  # constraint: cannot delete more than half

    sum_x = np.cumsum(batched[::-1])[::-1]  # sum_x[d] = sum(batched[d:])
    sum_x2 = np.cumsum((batched**2)[::-1])[::-1]

    counts = np.arange(m, 0, -1, dtype=np.float64)  # counts[d] = m - d
    means = sum_x / counts
    variances = np.maximum(sum_x2 / counts - means**2, 0.0)
    mser = variances / counts  # MSER(d) = var / count

    d_star = int(np.argmin(mser[: max_d + 1]))
    return d_star * batch_size


def mser5_boundary_ns(
    metric_values: NDArray[np.float64],
    start_ns: NDArray[np.float64],
    end_ns: NDArray[np.float64],
    filled: NDArray[np.bool_],
) -> tuple[float | None, float | None]:
    """Run forward and backward MSER-5 on a metric, returning wall-clock boundaries.

    Args:
        metric_values: Per-record metric values (session-indexed, may contain NaN).
        start_ns: Per-record start timestamps (session-indexed).
        end_ns: Per-record end timestamps (session-indexed).
        filled: Boolean mask of records with valid start/end timestamps.

    Returns:
        (ramp_up_end_ns, ramp_down_start_ns) or (None, None) if insufficient data.
    """
    metric_valid = filled & ~np.isnan(metric_values)
    if metric_valid.sum() < 20:
        return None, None

    valid_indices = np.where(metric_valid)[0]
    sort_order = np.argsort(start_ns[valid_indices])
    sorted_indices = valid_indices[sort_order]

    sorted_metric = metric_values[sorted_indices]
    sorted_start = start_ns[sorted_indices]
    sorted_end = end_ns[sorted_indices]

    # Forward: ramp-up truncation
    fwd_trunc = mser5_truncation_point(sorted_metric)
    ramp_up_ns = (
        float(sorted_start[fwd_trunc]) if fwd_trunc < len(sorted_start) else None
    )

    # Backward: ramp-down truncation
    bwd_trunc = mser5_truncation_point(sorted_metric[::-1])
    if bwd_trunc > 0:
        ramp_down_idx = len(sorted_end) - bwd_trunc - 1
        ramp_down_ns = float(sorted_end[ramp_down_idx])
    else:
        ramp_down_ns = None

    return ramp_up_ns, ramp_down_ns


def _collect_detection_signals(
    *,
    sorted_ts: NDArray[np.float64],
    concurrency: NDArray[np.float64],
    start_ns: NDArray[np.float64],
    end_ns: NDArray[np.float64],
    latency: NDArray[np.float64],
    ttft: NDArray[np.float64],
    sorted_tput_ts: NDArray[np.float64] | None,
    throughput: NDArray[np.float64] | None,
) -> tuple[list[float], list[float], list[str]]:
    """Collect candidate ramp_up/ramp_down boundaries from each detection signal."""
    filled = ~np.isnan(start_ns) & ~np.isnan(end_ns)

    cusum_start, cusum_end = cusum_steady_state_window(
        sorted_ts,
        concurrency,
        min_window_pct=0.0,
    )
    starts: list[float] = [cusum_start]
    ends: list[float] = [cusum_end]
    signals_used: list[str] = ["cusum"]

    lat_start, lat_end = mser5_boundary_ns(latency, start_ns, end_ns, filled)
    if lat_start is not None:
        starts.append(lat_start)
        signals_used.append("mser5_latency")
    if lat_end is not None:
        ends.append(lat_end)

    ttft_start, ttft_end = mser5_boundary_ns(ttft, start_ns, end_ns, filled)
    if ttft_start is not None:
        starts.append(ttft_start)
        signals_used.append("mser5_ttft")
    if ttft_end is not None:
        ends.append(ttft_end)

    if (
        sorted_tput_ts is not None
        and throughput is not None
        and len(sorted_tput_ts) > 0
    ):
        tput_start, tput_end = cusum_steady_state_window(
            sorted_tput_ts, throughput, min_window_pct=0.0
        )
        starts.append(tput_start)
        ends.append(tput_end)
        signals_used.append("cusum_throughput")

    return starts, ends, signals_used


def detect_steady_state_window(
    sorted_ts: NDArray[np.float64],
    concurrency: NDArray[np.float64],
    start_ns: NDArray[np.float64],
    end_ns: NDArray[np.float64],
    *,
    latency: NDArray[np.float64],
    ttft: NDArray[np.float64],
    min_window_pct: float = 10.0,
    sorted_tput_ts: NDArray[np.float64] | None = None,
    throughput: NDArray[np.float64] | None = None,
) -> DetectionResult:
    """Detect steady-state window using CUSUM + MSER-5 + optional CUSUM(throughput).

    Combines concurrency-based CUSUM (load), metric-based MSER-5 on both
    latency and TTFT (performance), and optionally CUSUM on the aggregate
    throughput curve (output). The effective boundary is the most conservative
    across all signals.
    """
    if len(sorted_ts) == 0:
        return DetectionResult(0.0, 0.0, "empty")

    min_ts = float(sorted_ts[0])
    max_ts = float(sorted_ts[-1])
    total_duration = max_ts - min_ts
    if total_duration <= 0:
        return DetectionResult(min_ts, max_ts, "zero_duration")

    starts, ends, signals_used = _collect_detection_signals(
        sorted_ts=sorted_ts,
        concurrency=concurrency,
        start_ns=start_ns,
        end_ns=end_ns,
        latency=latency,
        ttft=ttft,
        sorted_tput_ts=sorted_tput_ts,
        throughput=throughput,
    )

    window_start = max(starts)
    window_end = min(ends)
    method = "_".join(signals_used)

    if window_start >= window_end:
        logger.warning(
            "Detection signals do not agree on a valid window (%s), "
            "falling back to full range",
            method,
        )
        return DetectionResult(min_ts, max_ts, "fallback_no_overlap")

    if (window_end - window_start) < total_duration * min_window_pct / 100.0:
        logger.warning(
            "Steady-state window (%.1f%%) below minimum (%.1f%%), "
            "falling back to full range",
            (window_end - window_start) / total_duration * 100,
            min_window_pct,
        )
        return DetectionResult(min_ts, max_ts, "fallback_min_window")

    return DetectionResult(window_start, window_end, method)


def manual_steady_state_window(
    min_ts: float,
    max_ts: float,
    start_pct: float,
    end_pct: float,
) -> tuple[float, float]:
    """Compute steady-state window from user-specified percentages.

    Args:
        min_ts: Minimum timestamp of the benchmark.
        max_ts: Maximum timestamp of the benchmark.
        start_pct: Start of window as % of total duration.
        end_pct: End of window as % of total duration.

    Returns:
        (window_start, window_end) — wall-clock timestamps.
    """
    duration = max_ts - min_ts
    return (min_ts + duration * start_pct / 100.0, min_ts + duration * end_pct / 100.0)
