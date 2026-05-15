# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vectorized sweep-line algorithms for concurrency and throughput curves.

All functions operate on numpy arrays — no record objects, no Python loops.
Input arrays are expected to be session_num-indexed (from ColumnStore).
"""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

from aiperf.common.constants import NANOS_PER_SECOND

FloatArray: TypeAlias = NDArray[np.float64]
Int64Array: TypeAlias = NDArray[np.int64]
Int32Array: TypeAlias = NDArray[np.int32]


class SweepLineStats(NamedTuple):
    """Time-weighted statistics from a sweep-line step function."""

    avg: float
    min: float
    max: float
    p50: float
    p90: float
    p95: float
    p99: float
    std: float


ZERO_SWEEP_LINE_STATS = SweepLineStats(
    avg=0.0, min=0.0, max=0.0, p50=0.0, p90=0.0, p95=0.0, p99=0.0, std=0.0
)


class SweepLineMetricSpec(NamedTuple):
    """Specification for a sweep-line metric (tag, header, unit, scale)."""

    tag: str
    header: str
    unit: str
    scale: float


SWEEP_LINE_METRIC_SPECS: tuple[SweepLineMetricSpec, ...] = (
    SweepLineMetricSpec(
        "effective_concurrency", "Effective Concurrency", "requests", 1.0
    ),
    SweepLineMetricSpec(
        "effective_throughput", "Effective Throughput", "tokens/sec", NANOS_PER_SECOND
    ),
    SweepLineMetricSpec(
        "effective_prefill_throughput",
        "Effective Prefill Throughput",
        "tokens/sec",
        NANOS_PER_SECOND,
    ),
    SweepLineMetricSpec(
        "effective_generation_concurrency",
        "Effective Generation Concurrency",
        "requests",
        1.0,
    ),
    SweepLineMetricSpec(
        "effective_prefill_concurrency",
        "Effective Prefill Concurrency",
        "requests",
        1.0,
    ),
    SweepLineMetricSpec(
        "effective_total_throughput",
        "Effective Total Throughput",
        "tokens/sec",
        NANOS_PER_SECOND,
    ),
    SweepLineMetricSpec(
        "effective_throughput_per_user",
        "Effective Throughput Per User",
        "tokens/sec/user",
        NANOS_PER_SECOND,
    ),
    SweepLineMetricSpec(
        "effective_prefill_throughput_per_user",
        "Effective Prefill Throughput Per User",
        "tokens/sec/user",
        NANOS_PER_SECOND,
    ),
    SweepLineMetricSpec(
        "tokens_in_flight",
        "Tokens In Flight",
        "tokens",
        1.0,
    ),
)


class ActiveVariantSpec(NamedTuple):
    """Spec for an active-only throughput metric: (tag, header, unit, rate_attr, mask_attr)."""

    tag: str
    header: str
    unit: str
    rate_attr: str
    mask_attr: str


ACTIVE_VARIANT_SPECS: tuple[ActiveVariantSpec, ...] = (
    ActiveVariantSpec(
        "effective_decode_throughput_active",
        "Effective Decode Throughput (Active)",
        "tokens/sec",
        "throughput",
        "generation_concurrency",
    ),
    ActiveVariantSpec(
        "effective_prefill_throughput_active",
        "Effective Prefill Throughput (Active)",
        "tokens/sec",
        "prefill_throughput",
        "prefill_concurrency",
    ),
    ActiveVariantSpec(
        "effective_decode_throughput_per_user_active",
        "Effective Decode Throughput Per User (Active)",
        "tokens/sec/user",
        "throughput_per_user",
        "generation_concurrency",
    ),
    ActiveVariantSpec(
        "effective_prefill_throughput_per_user_active",
        "Effective Prefill Throughput Per User (Active)",
        "tokens/sec/user",
        "prefill_throughput_per_user",
        "prefill_concurrency",
    ),
    ActiveVariantSpec(
        "effective_total_throughput_active",
        "Effective Total Throughput (Active)",
        "tokens/sec",
        "total_throughput",
        "concurrency",
    ),
)


def _sweep_line_cumsum(
    timestamps: FloatArray,
    deltas: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Sort events by timestamp (ends before starts at ties) and cumsum deltas."""
    # lexsort: primary key = timestamps, secondary key = event_type (0=end, 1=start).
    # Ends sort before starts at the same timestamp.
    event_type = (deltas > 0).astype(np.int8)
    order = np.lexsort((event_type, timestamps))
    vals = np.cumsum(deltas[order])
    # Snap FP roundoff to zero. All sweep curves represent physically
    # non-negative quantities (concurrency, tokens, throughput); imperfect
    # cancellation of large +/- pairs leaves residuals of relative size ~1e-12
    # at peak magnitudes that render as "-0.00" in formatted output. A real
    # ordering bug would produce a magnitude orders larger than this threshold
    # and remain visible.
    if len(vals) > 0:
        max_abs = float(np.max(np.abs(vals)))
        if max_abs > 0.0:
            vals = np.where(np.abs(vals) < 1e-9 * max_abs, 0.0, vals)
    return timestamps[order], vals


def _step_lookup(
    event_ts: FloatArray,
    event_vals: FloatArray,
    query_ts: FloatArray,
) -> FloatArray:
    """Look up step-function values at query timestamps (0 before first event)."""
    idx = np.searchsorted(event_ts, query_ts, side="right").astype(np.intp) - 1
    return np.where(idx >= 0, event_vals[np.clip(idx, 0, len(event_vals) - 1)], 0.0)


def add_step_functions(
    a_ts: FloatArray,
    a_vals: FloatArray,
    b_ts: FloatArray,
    b_vals: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Add two step functions, returning a new step function on merged timestamps.

    Args:
        a_ts: Sorted timestamps of the first step function.
        a_vals: Values of the first step function.
        b_ts: Sorted timestamps of the second step function.
        b_vals: Values of the second step function.

    Returns:
        Tuple of (merged_timestamps, sum_values).
    """
    if len(a_ts) == 0:
        return b_ts.copy(), b_vals.copy()
    if len(b_ts) == 0:
        return a_ts.copy(), a_vals.copy()

    merged_ts = np.unique(np.concatenate([a_ts, b_ts]))
    return merged_ts, _step_lookup(a_ts, a_vals, merged_ts) + _step_lookup(
        b_ts, b_vals, merged_ts
    )


def divide_step_functions(
    num_ts: FloatArray,
    num_vals: FloatArray,
    den_ts: FloatArray,
    den_vals: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Divide two step functions, returning a new step function on merged timestamps.

    Where denominator is zero the result is zero (safe division).

    Args:
        num_ts: Sorted timestamps of the numerator step function.
        num_vals: Values of the numerator step function.
        den_ts: Sorted timestamps of the denominator step function.
        den_vals: Values of the denominator step function.

    Returns:
        Tuple of (merged_timestamps, quotient_values).
    """
    if len(num_ts) == 0 or len(den_ts) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    merged_ts = np.unique(np.concatenate([num_ts, den_ts]))
    num_at = _step_lookup(num_ts, num_vals, merged_ts)
    den_at = _step_lookup(den_ts, den_vals, merged_ts)

    result = np.zeros_like(num_at)
    np.divide(num_at, den_at, out=result, where=den_at > 0)
    return merged_ts, result


def throughput_per_user_sweep_line(
    generation_start_ns: FloatArray,
    end_ns: FloatArray,
    tput_ts: FloatArray,
    tput_vals: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute per-user throughput by dividing aggregate throughput by generation-phase concurrency.

    Args:
        generation_start_ns: First-token wall-clock timestamps. NaN for missing.
        end_ns: Request end timestamps. NaN for missing.
        tput_ts: Sorted timestamps from throughput_sweep (or ICL variant).
        tput_vals: Throughput values (tokens/ns) at each timestamp.

    Returns:
        Tuple of (timestamps, per_user_throughput) in tokens/ns/user.
    """
    conc_ts, conc_vals = concurrency_sweep_line(generation_start_ns, end_ns)
    return divide_step_functions(tput_ts, tput_vals, conc_ts, conc_vals)


def prefill_throughput_per_user_sweep_line(
    start_ns: FloatArray,
    generation_start_ns: FloatArray,
    ptput_ts: FloatArray,
    ptput_vals: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute per-user prefill throughput by dividing aggregate prefill throughput by prefill-phase concurrency.

    Args:
        start_ns: Request start timestamps. NaN for missing.
        generation_start_ns: First-token wall-clock timestamps. NaN for missing.
        ptput_ts: Sorted timestamps from prefill_throughput_sweep.
        ptput_vals: Prefill throughput values (tokens/ns) at each timestamp.

    Returns:
        Tuple of (timestamps, per_user_prefill_throughput) in tokens/ns/user.
    """
    conc_ts, conc_vals = concurrency_sweep_line(start_ns, generation_start_ns)
    return divide_step_functions(ptput_ts, ptput_vals, conc_ts, conc_vals)


def concurrency_sweep_line(
    start_ns: FloatArray,
    end_ns: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute exact instantaneous concurrency at every event boundary.

    Args:
        start_ns: Request start timestamps (wall-clock). NaN for missing records.
        end_ns: Request end timestamps (wall-clock). NaN for missing records.

    Returns:
        Tuple of (sorted_timestamps, concurrency_values).
        sorted_timestamps has shape (2K,), concurrency_values has shape (2K,),
        where K is the number of valid (non-NaN) records.
    """
    valid = ~np.isnan(start_ns) & ~np.isnan(end_ns)
    k = int(valid.sum())
    if k == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    timestamps = np.concatenate([start_ns[valid], end_ns[valid]])
    deltas = np.concatenate(
        [np.ones(k, dtype=np.float64), -np.ones(k, dtype=np.float64)]
    )

    sorted_ts, concurrency = _sweep_line_cumsum(timestamps, deltas)
    return sorted_ts, concurrency


def throughput_sweep_line(
    generation_start_ns: FloatArray,
    end_ns: FloatArray,
    output_tokens: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute exact instantaneous throughput (tokens/ns) at every event boundary.

    Uses uniform per-request rate: (output_tokens - 1) / generation_duration.

    Args:
        generation_start_ns: First-token wall-clock timestamps. NaN for missing.
        end_ns: Request end timestamps. NaN for missing.
        output_tokens: Output token counts. NaN for missing.

    Returns:
        Tuple of (sorted_timestamps, throughput_values) in tokens/ns.
    """
    gen_dur = end_ns - generation_start_ns
    valid = ~np.isnan(generation_start_ns) & ~np.isnan(output_tokens) & (gen_dur > 0)
    k = int(valid.sum())
    if k == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    rates = (output_tokens[valid] - 1.0) / gen_dur[valid]

    timestamps = np.concatenate([generation_start_ns[valid], end_ns[valid]])
    deltas = np.concatenate([rates, -rates])

    sorted_ts, throughput = _sweep_line_cumsum(timestamps, deltas)
    return sorted_ts, throughput


def prefill_throughput_sweep_line(
    start_ns: FloatArray,
    generation_start_ns: FloatArray,
    input_tokens: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute exact instantaneous prefill throughput (tokens/ns) at every event boundary.

    During prefill [start_ns, generation_start_ns), the model processes
    input_tokens tokens. The per-request prefill rate is
    input_tokens / prefill_duration.

    Args:
        start_ns: Request start timestamps (wall-clock). NaN for missing.
        generation_start_ns: First-token wall-clock timestamps. NaN for missing.
        input_tokens: Input token counts. NaN for missing.

    Returns:
        Tuple of (sorted_timestamps, prefill_throughput_values) in tokens/ns.
    """
    prefill_dur = generation_start_ns - start_ns
    valid = (
        ~np.isnan(start_ns)
        & ~np.isnan(generation_start_ns)
        & ~np.isnan(input_tokens)
        & (prefill_dur > 0)
    )
    k = int(valid.sum())
    if k == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    rates = input_tokens[valid] / prefill_dur[valid]

    timestamps = np.concatenate([start_ns[valid], generation_start_ns[valid]])
    deltas = np.concatenate([rates, -rates])

    sorted_ts, prefill_tput = _sweep_line_cumsum(timestamps, deltas)
    return sorted_ts, prefill_tput


def total_throughput_sweep_line(
    start_ns: FloatArray,
    generation_start_ns: FloatArray,
    end_ns: FloatArray,
    input_tokens: FloatArray,
    *,
    output_tokens: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute total throughput (prefill + generation) in a single sweep pass.

    Combines prefill rate events [start_ns, generation_start_ns) and generation
    rate events [generation_start_ns, end_ns) into one sweep, avoiding the
    overhead of two separate sweeps + grid merge + searchsorted lookups.

    Args:
        start_ns: Request start timestamps. NaN for missing.
        generation_start_ns: First-token wall-clock timestamps. NaN for missing.
        end_ns: Request end timestamps. NaN for missing.
        input_tokens: Input token counts. NaN for missing.
        output_tokens: Output token counts. NaN for missing.

    Returns:
        Tuple of (sorted_timestamps, total_throughput_values) in tokens/ns.
    """
    # Prefill: input_tokens / prefill_duration during [start, gen_start)
    prefill_dur = generation_start_ns - start_ns
    pf_valid = (
        ~np.isnan(start_ns)
        & ~np.isnan(generation_start_ns)
        & ~np.isnan(input_tokens)
        & (prefill_dur > 0)
    )
    pf_k = int(pf_valid.sum())

    # Generation: (output_tokens - 1) / gen_duration during [gen_start, end)
    gen_dur = end_ns - generation_start_ns
    gn_valid = ~np.isnan(generation_start_ns) & ~np.isnan(output_tokens) & (gen_dur > 0)
    gn_k = int(gn_valid.sum())

    if pf_k == 0 and gn_k == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    parts_ts: list[FloatArray] = []
    parts_delta: list[FloatArray] = []

    if pf_k > 0:
        pf_rates = input_tokens[pf_valid] / prefill_dur[pf_valid]
        parts_ts.extend([start_ns[pf_valid], generation_start_ns[pf_valid]])
        parts_delta.extend([pf_rates, -pf_rates])

    if gn_k > 0:
        gn_rates = (output_tokens[gn_valid] - 1.0) / gen_dur[gn_valid]
        parts_ts.extend([generation_start_ns[gn_valid], end_ns[gn_valid]])
        parts_delta.extend([gn_rates, -gn_rates])

    return _sweep_line_cumsum(np.concatenate(parts_ts), np.concatenate(parts_delta))


# Re-export submodule symbols for backwards compatibility with existing imports.
from aiperf.analysis.sweepline_curves import (  # noqa: E402
    SweepLineCurves as SweepLineCurves,
)
from aiperf.analysis.sweepline_kv_cache import (  # noqa: E402
    _icl_chunk_events as _icl_chunk_events,
)
from aiperf.analysis.sweepline_kv_cache import (  # noqa: E402
    _kv_cache_events as _kv_cache_events,
)
from aiperf.analysis.sweepline_kv_cache import (  # noqa: E402
    throughput_sweep_line_icl as throughput_sweep_line_icl,
)
from aiperf.analysis.sweepline_kv_cache import (  # noqa: E402
    tokens_in_flight_sweep_line as tokens_in_flight_sweep_line,
)
from aiperf.analysis.sweepline_kv_cache import (  # noqa: E402
    tokens_in_flight_sweep_line_icl as tokens_in_flight_sweep_line_icl,
)
from aiperf.analysis.sweepline_stats import (  # noqa: E402
    _build_clipped_segments as _build_clipped_segments,
)
from aiperf.analysis.sweepline_stats import (  # noqa: E402
    compute_active_weighted_stats as compute_active_weighted_stats,
)
from aiperf.analysis.sweepline_stats import (  # noqa: E402
    compute_time_weighted_stats as compute_time_weighted_stats,
)
from aiperf.analysis.sweepline_stats import (  # noqa: E402
    metric_result_from_sweep_line_stats as metric_result_from_sweep_line_stats,
)
