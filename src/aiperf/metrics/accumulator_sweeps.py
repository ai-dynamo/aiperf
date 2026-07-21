# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-function helpers for ICL-aware throughput and tokens-in-flight sweeps.

Sweep curves live in ``aiperf.analysis.sweepline*``; this module wraps them
with ICL-aware variants that use per-chunk decode timing when the configured
list backend retains it (i.e. ``RaggedSeries``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from aiperf.analysis import sweepline as _sweepline
from aiperf.analysis import sweepline_kv_cache as _kv_cache

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
    return icl  # type: ignore[return-value]


def icl_aware_throughput(
    store: ColumnStore,
    generation_start_ns: FloatArray,
    end_ns: FloatArray,
    output_tokens: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute throughput sweep, preferring ICL-aware when available."""
    icl = _get_icl_data(store)
    if icl is not None:
        return _sweepline.throughput_sweep_line_icl(
            generation_start_ns,
            output_tokens,
            icl.values,
            icl.record_indices,
            icl_offsets=icl.offsets,
        )
    return _sweepline.throughput_sweep_line(generation_start_ns, end_ns, output_tokens)


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
        return _kv_cache.tokens_in_flight_sweep_line_icl(
            start_ns,
            generation_start_ns,
            end_ns,
            input_tokens,
            output_tokens=output_tokens,
            icl_values=icl.values,
            icl_record_indices=icl.record_indices,
            icl_offsets=icl.offsets,
        )
    return _kv_cache.tokens_in_flight_sweep_line(
        start_ns,
        generation_start_ns,
        end_ns,
        input_tokens,
        output_tokens=output_tokens,
    )


def _build_concurrency_curves(
    sweepline: Any,
    start_ns: Any,
    end_ns: Any,
    generation_start_ns: Any,
) -> dict[str, Any]:
    """Return the three concurrency step functions (overall, generation, prefill)."""
    concurrency_ts, concurrency_vals = sweepline.concurrency_sweep_line(
        start_ns, end_ns
    )
    gen_conc_ts, gen_conc_vals = sweepline.concurrency_sweep_line(
        generation_start_ns, end_ns
    )
    prefill_conc_ts, prefill_conc_vals = sweepline.concurrency_sweep_line(
        start_ns, generation_start_ns
    )
    return {
        "concurrency_ts": concurrency_ts,
        "concurrency": concurrency_vals,
        "gen_conc_ts": gen_conc_ts,
        "gen_conc_vals": gen_conc_vals,
        "prefill_conc_ts": prefill_conc_ts,
        "prefill_conc_vals": prefill_conc_vals,
    }


def _build_throughput_curves(
    sweepline: Any,
    *,
    store: ColumnStore,
    start_ns: Any,
    end_ns: Any,
    generation_start_ns: Any,
    input_tokens: Any,
    output_tokens: Any,
    conc: dict[str, Any],
) -> dict[str, Any]:
    """Return the throughput, prefill-throughput, total-throughput, and per-user curves."""
    throughput_ts, throughput_vals = icl_aware_throughput(
        store, generation_start_ns, end_ns, output_tokens
    )
    prefill_throughput_ts, prefill_throughput_vals = (
        sweepline.prefill_throughput_sweep_line(
            start_ns, generation_start_ns, input_tokens
        )
    )
    total_throughput_ts, total_throughput_vals = sweepline.total_throughput_sweep_line(
        start_ns,
        generation_start_ns,
        end_ns,
        input_tokens,
        output_tokens=output_tokens,
    )
    tput_per_user_ts, tput_per_user_vals = sweepline.divide_step_functions(
        throughput_ts, throughput_vals, conc["gen_conc_ts"], conc["gen_conc_vals"]
    )
    prefill_tput_per_user_ts, prefill_tput_per_user_vals = (
        sweepline.divide_step_functions(
            prefill_throughput_ts,
            prefill_throughput_vals,
            conc["prefill_conc_ts"],
            conc["prefill_conc_vals"],
        )
    )
    return {
        "throughput_ts": throughput_ts,
        "throughput": throughput_vals,
        "prefill_throughput_ts": prefill_throughput_ts,
        "prefill_throughput": prefill_throughput_vals,
        "total_throughput_ts": total_throughput_ts,
        "total_throughput": total_throughput_vals,
        "tput_per_user_ts": tput_per_user_ts,
        "tput_per_user": tput_per_user_vals,
        "prefill_tput_per_user_ts": prefill_tput_per_user_ts,
        "prefill_tput_per_user": prefill_tput_per_user_vals,
    }


def _apply_record_mask(
    values: FloatArray, mask: NDArray[np.bool_] | None
) -> FloatArray:
    """Return ``values`` with non-selected records replaced by NaN.

    Keeping the original record-indexed shape lets ICL ragged arrays keep their
    existing record indices while every excluded record naturally drops out of
    the downstream sweep algorithms through NaN validity checks.
    """
    if mask is None:
        return values
    masked = values.copy()
    masked[~mask] = np.nan
    return masked


def compute_sweep_curves(
    store: ColumnStore, mask: NDArray[np.bool_] | None = None
) -> _sweepline.SweepLineCurves:
    """Compute the full SweepLineCurves bundle for the records in ``store``.

    ICL-aware variants are used when the configured list backend exposes
    per-record replay (i.e. ``RaggedSeries``); otherwise the request-level
    fallbacks fire — see ``_get_icl_data``.
    """
    n = store.count
    start_ns = _apply_record_mask(store.start_ns[:n], mask)
    end_ns = _apply_record_mask(store.end_ns[:n], mask)
    generation_start_ns = _apply_record_mask(store.generation_start_ns[:n], mask)
    output_tokens = _apply_record_mask(store.numeric("output_sequence_length"), mask)
    input_tokens = _apply_record_mask(store.numeric("input_sequence_length"), mask)

    conc = _build_concurrency_curves(_sweepline, start_ns, end_ns, generation_start_ns)
    tput = _build_throughput_curves(
        _sweepline,
        store=store,
        start_ns=start_ns,
        end_ns=end_ns,
        generation_start_ns=generation_start_ns,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        conc=conc,
    )
    tokens_in_flight_ts, tokens_in_flight_vals = icl_aware_tokens_in_flight(
        store,
        start_ns,
        generation_start_ns,
        end_ns,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return _sweepline.SweepLineCurves(
        concurrency_ts=conc["concurrency_ts"],
        concurrency=conc["concurrency"],
        throughput_ts=tput["throughput_ts"],
        throughput=tput["throughput"],
        prefill_throughput_ts=tput["prefill_throughput_ts"],
        prefill_throughput=tput["prefill_throughput"],
        generation_concurrency_ts=conc["gen_conc_ts"],
        generation_concurrency=conc["gen_conc_vals"],
        prefill_concurrency_ts=conc["prefill_conc_ts"],
        prefill_concurrency=conc["prefill_conc_vals"],
        total_throughput_ts=tput["total_throughput_ts"],
        total_throughput=tput["total_throughput"],
        throughput_per_user_ts=tput["tput_per_user_ts"],
        throughput_per_user=tput["tput_per_user"],
        prefill_throughput_per_user_ts=tput["prefill_tput_per_user_ts"],
        prefill_throughput_per_user=tput["prefill_tput_per_user"],
        tokens_in_flight_ts=tokens_in_flight_ts,
        tokens_in_flight=tokens_in_flight_vals,
    )
