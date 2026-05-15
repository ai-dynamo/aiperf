# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-function helpers for ICL-aware throughput and tokens-in-flight sweeps."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

from aiperf.analysis.sweepline import (
    SweepLineCurves,
    concurrency_sweep_line,
    divide_step_functions,
    prefill_throughput_sweep_line,
    throughput_sweep_line,
    throughput_sweep_line_icl,
    total_throughput_sweep_line,
)
from aiperf.analysis.sweepline_kv_cache import (
    tokens_in_flight_sweep_line,
    tokens_in_flight_sweep_line_icl,
)

if TYPE_CHECKING:
    from aiperf.metrics.column_store import ColumnStore
    from aiperf.metrics.ragged_series import RaggedSeries

FloatArray: TypeAlias = NDArray[np.float64]


def _get_icl_data(store: ColumnStore) -> RaggedSeries | None:
    """Return inter-chunk-latency ragged series if available for replay, else None.

    Returns ``None`` both when ICL was never recorded and when the configured
    list backend (``Environment.METRICS.LIST_BACKEND=tdigest``) does not retain
    per-record structure. In both cases, callers fall through to the
    request-level (non-ICL) sweep helpers.
    """
    if "inter_chunk_latency" not in store.ragged_tags():
        return None
    icl = store.ragged("inter_chunk_latency")
    if not getattr(icl, "SUPPORTS_PER_RECORD_REPLAY", False):
        return None
    if len(icl.values) == 0:
        return None
    return icl


def icl_aware_throughput(
    store: ColumnStore,
    generation_start_ns: FloatArray,
    end_ns: FloatArray,
    output_tokens: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute throughput sweep, preferring ICL-aware when available."""
    icl = _get_icl_data(store)
    if icl is not None:
        return throughput_sweep_line_icl(
            generation_start_ns,
            output_tokens,
            icl.values,
            icl.record_indices,
            icl_offsets=icl.offsets,
        )
    return throughput_sweep_line(generation_start_ns, end_ns, output_tokens)


def icl_aware_tokens_in_flight(
    store: ColumnStore,
    start_ns: FloatArray,
    generation_start_ns: FloatArray,
    end_ns: FloatArray,
    *,
    input_tokens: FloatArray,
    output_tokens: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute tokens in flight, preferring ICL-aware when available."""
    icl = _get_icl_data(store)
    if icl is not None:
        return tokens_in_flight_sweep_line_icl(
            start_ns,
            generation_start_ns,
            end_ns,
            input_tokens,
            output_tokens=output_tokens,
            icl_values=icl.values,
            icl_record_indices=icl.record_indices,
            icl_offsets=icl.offsets,
        )
    return tokens_in_flight_sweep_line(
        start_ns,
        generation_start_ns,
        end_ns,
        input_tokens,
        output_tokens=output_tokens,
    )


def compute_sweep_curves(store: ColumnStore) -> SweepLineCurves:
    """Compute the full SweepLineCurves bundle for the records in ``store``.

    ICL-aware variants are used when the configured list backend exposes
    per-record replay (i.e. ``RaggedSeries``); otherwise the request-level
    fallbacks fire — see ``_get_icl_data``.
    """
    n = store.count
    start_ns = store.start_ns[:n]
    end_ns = store.end_ns[:n]
    generation_start_ns = store.generation_start_ns[:n]
    output_tokens = store.numeric("output_sequence_length")
    input_tokens = store.numeric("input_sequence_length")

    concurrency_ts, concurrency_vals = concurrency_sweep_line(start_ns, end_ns)
    throughput_ts, throughput_vals = icl_aware_throughput(
        store, generation_start_ns, end_ns, output_tokens
    )
    prefill_throughput_ts, prefill_throughput_vals = prefill_throughput_sweep_line(
        start_ns, generation_start_ns, input_tokens
    )

    gen_conc_ts, gen_conc_vals = concurrency_sweep_line(generation_start_ns, end_ns)
    prefill_conc_ts, prefill_conc_vals = concurrency_sweep_line(
        start_ns, generation_start_ns
    )

    total_throughput_ts, total_throughput_vals = total_throughput_sweep_line(
        start_ns,
        generation_start_ns,
        end_ns,
        input_tokens,
        output_tokens=output_tokens,
    )
    tput_per_user_ts, tput_per_user_vals = divide_step_functions(
        throughput_ts, throughput_vals, gen_conc_ts, gen_conc_vals
    )
    prefill_tput_per_user_ts, prefill_tput_per_user_vals = divide_step_functions(
        prefill_throughput_ts,
        prefill_throughput_vals,
        prefill_conc_ts,
        prefill_conc_vals,
    )
    tokens_in_flight_ts, tokens_in_flight_vals = icl_aware_tokens_in_flight(
        store,
        start_ns,
        generation_start_ns,
        end_ns,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return SweepLineCurves(
        concurrency_ts=concurrency_ts,
        concurrency=concurrency_vals,
        throughput_ts=throughput_ts,
        throughput=throughput_vals,
        prefill_throughput_ts=prefill_throughput_ts,
        prefill_throughput=prefill_throughput_vals,
        generation_concurrency_ts=gen_conc_ts,
        generation_concurrency=gen_conc_vals,
        prefill_concurrency_ts=prefill_conc_ts,
        prefill_concurrency=prefill_conc_vals,
        total_throughput_ts=total_throughput_ts,
        total_throughput=total_throughput_vals,
        throughput_per_user_ts=tput_per_user_ts,
        throughput_per_user=tput_per_user_vals,
        prefill_throughput_per_user_ts=prefill_tput_per_user_ts,
        prefill_throughput_per_user=prefill_tput_per_user_vals,
        tokens_in_flight_ts=tokens_in_flight_ts,
        tokens_in_flight=tokens_in_flight_vals,
    )
