# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV cache (tokens-in-flight) sweep-line algorithms, including ICL-aware variants."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aiperf.analysis.sweepline import (
    FloatArray,
    Int32Array,
    Int64Array,
    _sweep_line_cumsum,
)


def _kv_cache_events(
    start_ns: FloatArray,
    generation_start_ns: FloatArray,
    end_ns: FloatArray,
    input_tokens: FloatArray,
    *,
    output_tokens: FloatArray,
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Collect (timestamp, token-delta) events for input + output tokens in KV cache."""
    has_start = ~np.isnan(start_ns) & ~np.isnan(input_tokens)
    gen_dur = end_ns - generation_start_ns
    has_gen = ~np.isnan(generation_start_ns) & ~np.isnan(output_tokens) & (gen_dur > 0)
    has_end = ~np.isnan(end_ns)

    parts_ts: list[FloatArray] = []
    parts_delta: list[FloatArray] = []

    # Event 1: +input_tokens at start_ns (prefill begins)
    pf_valid = (
        has_start & ~np.isnan(generation_start_ns) & (generation_start_ns > start_ns)
    )
    if pf_valid.any():
        parts_ts.append(start_ns[pf_valid])
        parts_delta.append(input_tokens[pf_valid])

    # Event 2: +output_tokens at generation_start_ns
    if has_gen.any():
        parts_ts.append(generation_start_ns[has_gen])
        parts_delta.append(output_tokens[has_gen])

    # Event 3: free tokens at end_ns
    end_with_input = pf_valid & has_end
    end_with_gen = has_gen & has_end
    both = end_with_input & end_with_gen
    input_only = end_with_input & ~end_with_gen
    gen_only = end_with_gen & ~end_with_input

    if both.any():
        parts_ts.append(end_ns[both])
        parts_delta.append(-(input_tokens[both] + output_tokens[both]))
    if input_only.any():
        parts_ts.append(end_ns[input_only])
        parts_delta.append(-input_tokens[input_only])
    if gen_only.any():
        parts_ts.append(end_ns[gen_only])
        parts_delta.append(-output_tokens[gen_only])

    return parts_ts, parts_delta


def tokens_in_flight_sweep_line(
    start_ns: FloatArray,
    generation_start_ns: FloatArray,
    end_ns: FloatArray,
    input_tokens: FloatArray,
    *,
    output_tokens: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute instantaneous KV cache token load at every event boundary.

    Models the total tokens held in server memory (KV cache) per request:
    - During prefill [start_ns, generation_start_ns): input_tokens
    - During generation [generation_start_ns, end_ns): input_tokens + output_tokens

    Input tokens stay in the KV cache throughout the request lifetime, and
    output tokens accumulate on top during generation. This reveals GPU
    memory pressure — two concurrent 4K-token requests look identical to two
    128-token requests in concurrency but wildly different here.
    """
    parts_ts, parts_delta = _kv_cache_events(
        start_ns,
        generation_start_ns,
        end_ns,
        input_tokens,
        output_tokens=output_tokens,
    )
    if len(parts_ts) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    return _sweep_line_cumsum(np.concatenate(parts_ts), np.concatenate(parts_delta))


def tokens_in_flight_sweep_line_icl(
    start_ns: FloatArray,
    generation_start_ns: FloatArray,
    end_ns: FloatArray,
    input_tokens: FloatArray,
    output_tokens: FloatArray,
    icl_values: FloatArray,
    icl_record_indices: Int32Array,
    icl_offsets: Int64Array,
) -> tuple[FloatArray, FloatArray]:
    """ICL-aware tokens in flight: output tokens ramp up at chunk boundaries.

    Instead of adding all output_tokens at generation_start_ns, this function
    adds tokens_per_chunk at each SSE chunk boundary during generation,
    modeling the gradual KV cache growth as tokens are generated.
    """
    if len(icl_values) == 0:
        return tokens_in_flight_sweep_line(
            start_ns,
            generation_start_ns,
            end_ns,
            input_tokens,
            output_tokens=output_tokens,
        )

    chunk_ts, chunk_delta, has_icl = _icl_chunk_events(
        generation_start_ns=generation_start_ns,
        end_ns=end_ns,
        output_tokens=output_tokens,
        icl_values=icl_values,
        icl_record_indices=icl_record_indices,
        icl_offsets=icl_offsets,
    )

    parts_ts: list[FloatArray] = []
    parts_delta: list[FloatArray] = []

    # TTFT chunk events: +1 token at gen_start_ns for each record with ICL data
    # and at least 1 output token. The first chunk arrives at the TTFT instant
    # and is not represented in the ICL series (ICL[0] is the gap between
    # chunks 1 and 2, not between gen_start and chunk 1).
    ttft_valid = (
        ~np.isnan(generation_start_ns)
        & has_icl
        & ~np.isnan(output_tokens)
        & (output_tokens >= 1)
    )
    if ttft_valid.any():
        parts_ts.append(generation_start_ns[ttft_valid])
        parts_delta.append(np.ones(int(ttft_valid.sum()), dtype=np.float64))

    if chunk_ts is not None:
        parts_ts.append(chunk_ts)
        parts_delta.append(chunk_delta)

    has_start = ~np.isnan(start_ns) & ~np.isnan(input_tokens)
    pf_valid = (
        has_start & ~np.isnan(generation_start_ns) & (generation_start_ns > start_ns)
    )
    if pf_valid.any():
        parts_ts.append(start_ns[pf_valid])
        parts_delta.append(input_tokens[pf_valid])

    has_end = ~np.isnan(end_ns)
    end_with_input_and_icl = pf_valid & has_end & has_icl
    end_with_input_only = pf_valid & has_end & ~has_icl
    end_with_icl_only = ~pf_valid & has_end & has_icl

    if end_with_input_and_icl.any():
        parts_ts.append(end_ns[end_with_input_and_icl])
        parts_delta.append(
            -(
                input_tokens[end_with_input_and_icl]
                + output_tokens[end_with_input_and_icl]
            )
        )
    if end_with_input_only.any():
        parts_ts.append(end_ns[end_with_input_only])
        parts_delta.append(-input_tokens[end_with_input_only])
    if end_with_icl_only.any():
        parts_ts.append(end_ns[end_with_icl_only])
        parts_delta.append(-output_tokens[end_with_icl_only])

    if len(parts_ts) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    return _sweep_line_cumsum(np.concatenate(parts_ts), np.concatenate(parts_delta))


def _icl_chunk_events(
    *,
    generation_start_ns: FloatArray,
    end_ns: FloatArray,
    output_tokens: FloatArray,
    icl_values: FloatArray,
    icl_record_indices: Int32Array,
    icl_offsets: Int64Array,
) -> tuple[FloatArray | None, FloatArray, NDArray[np.bool_]]:
    """Build per-chunk +tokens delta events; also return a per-record mask of
    records that emitted at least one valid chunk event.

    ICL gives K = icl_count timestamps for K+1 actual chunks (the first chunk
    arrives at gen_start_ns; ICL[k] is the gap between chunk k+1 and k+2).
    The TTFT chunk delivers exactly 1 token at gen_start_ns and is emitted
    by the caller; this function distributes the remaining (osl - 1) tokens
    across the K ICL events. Total per record = osl.
    """
    rec_idx = icl_record_indices

    global_cs = np.cumsum(icl_values)
    request_offsets = icl_offsets[rec_idx]
    start_cs = np.where(request_offsets > 0, global_cs[request_offsets - 1], 0.0)
    relative_cs = global_cs - start_cs

    gen_start = generation_start_ns[rec_idx]
    interval_end = gen_start + relative_cs

    icl_counts = np.bincount(rec_idx, minlength=len(output_tokens)).astype(np.float64)
    per_req_tokens = output_tokens[rec_idx]
    per_req_icl_count = icl_counts[rec_idx]
    tokens_per_chunk = np.where(
        per_req_icl_count > 0,
        (per_req_tokens - 1.0) / per_req_icl_count,
        0.0,
    )

    # Valid chunks: non-NaN gen_start, non-NaN ICL, non-NaN output_tokens.
    # Zero ICL is allowed: back-to-back chunks in the same network packet are
    # legitimate (common for the first 1-2 tokens of a streaming response).
    # Strictly negative ICL is dropped — the recorder should never produce it,
    # but if it does, NaN-via-comparison would silently corrupt downstream math.
    # Records with osl < 1 produce a negative tokens_per_chunk; clamp by also
    # requiring per_req_tokens >= 1 (mirrors throughput_sweep_line_icl).
    chunk_valid = (
        ~np.isnan(gen_start)
        & (icl_values >= 0)
        & ~np.isnan(per_req_tokens)
        & (per_req_tokens >= 1)
    )
    # has_icl must mirror chunk_valid at record level: a record whose chunks
    # were ALL filtered out (NaN gen_start, NaN/zero output_tokens) emits no
    # +chunk/+TTFT additions, so the caller's end-of-request subtraction
    # branches must not fire for it either — otherwise the cumsum picks up a
    # permanent negative offset (or NaN, when output_tokens is NaN).
    has_icl = np.bincount(rec_idx[chunk_valid], minlength=len(output_tokens)) > 0

    # Clamp chunk arrival to strictly before the record's end_ns. Recorder
    # jitter (chunks streamed slightly after end_ns is wall-clocked, sum of
    # ICL gaps drifting by 100s of ns) places some chunks past end_ns; the
    # lexsort tie-breaker (ends before starts at equal timestamps) then
    # orders -end before +chunk, so the cumsum would subtract (input+output)
    # before all chunks have been added, leaving a permanent negative offset
    # from that record's contribution. We use np.nextafter rather than
    # subtracting a constant: at ns-epoch timestamps (~1.7e18) float64
    # precision is ~256 ns, so subtracting 1 round-trips to the same value.
    rec_end_ns = end_ns[rec_idx]
    needs_clamp = ~np.isnan(rec_end_ns) & (interval_end >= rec_end_ns)
    interval_end = np.where(
        needs_clamp,
        np.nextafter(rec_end_ns, -np.inf),
        interval_end,
    )

    if not chunk_valid.any():
        return None, np.zeros(0, dtype=np.float64), has_icl
    return interval_end[chunk_valid], tokens_per_chunk[chunk_valid], has_icl


def throughput_sweep_line_icl(
    generation_start_ns: FloatArray,
    output_tokens: FloatArray,
    icl_values: FloatArray,
    icl_record_indices: Int32Array,
    icl_offsets: Int64Array,
) -> tuple[FloatArray, FloatArray]:
    """Compute ICL-aware instantaneous throughput at every chunk boundary.

    Uses per-request rescaled rates: each ICL interval carries
    ``output_tokens / n_icl_intervals`` tokens instead of exactly 1.
    This preserves the accurate temporal shape from SSE message boundaries
    while matching the known total token count per request.

    Args:
        generation_start_ns: Per-record first-token wall-clock (indexed by session_num).
        output_tokens: Per-record output token count (indexed by session_num).
        icl_values: Flat array of all ICL durations (M values).
        icl_record_indices: Session_num per ICL value (M values).
        icl_offsets: Per-session_num start offset into icl_values.

    Returns:
        Tuple of (sorted_timestamps, throughput_values) in tokens/ns.
        Has 2M events (one +rate and one -rate per chunk interval).
    """
    if len(icl_values) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    rec_idx = icl_record_indices

    # Per-request cumulative ICL — vectorized grouped cumsum
    global_cs = np.cumsum(icl_values)
    request_offsets = icl_offsets[rec_idx]
    start_cs = np.where(request_offsets > 0, global_cs[request_offsets - 1], 0.0)
    relative_cs = global_cs - start_cs

    # Wall-clock chunk boundaries
    gen_start = generation_start_ns[rec_idx]
    interval_end = gen_start + relative_cs
    interval_start = interval_end - icl_values

    # Per-request count of NON-ZERO ICL intervals. Zero-ICL entries (back-to-back
    # chunks at the same instant) can't carry a meaningful rate — division by
    # zero would produce inf — so they're excluded both as events and from the
    # divisor. Using icl_counts[total] would under-divide here and leak tokens.
    nonzero_mask = icl_values > 0
    icl_counts = np.bincount(
        rec_idx[nonzero_mask], minlength=len(output_tokens)
    ).astype(np.float64)
    per_req_tokens = output_tokens[rec_idx]
    per_req_icl_count = icl_counts[rec_idx]
    # Subtract 1 from osl: the TTFT chunk delivers 1 token instantaneously at
    # gen_start_ns and can't be modeled as a continuous rate over an interval.
    # Matches the non-ICL throughput_sweep_line which uses (osl - 1) / gen_dur.
    # Integrates to (osl - 1) tokens per record over the K nonzero intervals.
    # The inner where guards the divisor: np.where evaluates both branches
    # eagerly, and records whose ICL gaps are all zero have a nonzero-count
    # of 0, which would warn on divide-by-zero even though the result is
    # discarded by the outer mask.
    tokens_per_msg = np.where(
        per_req_icl_count > 0,
        (per_req_tokens - 1.0)
        / np.where(per_req_icl_count > 0, per_req_icl_count, 1.0),
        0.0,
    )
    rates = np.where(
        icl_values > 0, tokens_per_msg / np.where(icl_values > 0, icl_values, 1.0), 0.0
    )

    # Filter out invalid (NaN gen_start, zero/negative ICL, NaN output_tokens).
    # Records with osl < 1 produce a negative tokens_per_msg; clamp by also
    # requiring per_req_tokens >= 1.
    valid = (
        ~np.isnan(gen_start)
        & (icl_values > 0)
        & ~np.isnan(per_req_tokens)
        & (per_req_tokens >= 1)
    )
    if not valid.any():
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    timestamps = np.concatenate([interval_start[valid], interval_end[valid]])
    deltas = np.concatenate([rates[valid], -rates[valid]])

    sorted_ts, throughput = _sweep_line_cumsum(timestamps, deltas)
    return sorted_ts, throughput
