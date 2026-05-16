#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-pipeline memory + throughput probe for the metrics-accumulator comparison.

Auto-detects which pipeline is importable in the current worktree:
- new: ``aiperf.metrics.accumulator.MetricsAccumulator`` (k8s-metrics branch)
- old: ``aiperf.post_processors.metric_results_processor.MetricResultsProcessor``
       (new-config-kube branch)

Generates a deterministic synthetic stream of ``MetricRecordsData`` records,
ingests them, then summarizes. Reports peak/post-stage RSS and tracemalloc
peaks, plus a per-structure ``asizeof`` breakdown attributed to the container
the pipeline owns.

The companion driver ``metrics_pipeline_compare.py`` runs this script once
per pipeline (per worktree) and aggregates JSON output.

Usage:
    uv run python dev/benchmarks/metrics_pipeline_worker.py \\
        --n-records 100000 --avg-icl-chunks 100 --json
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import resource
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from pympler import asizeof

from aiperf.common.metric_records_wire import (
    MetricRecordMetadata,
    MetricRecordsData,
)
from aiperf.config import BenchmarkConfig

PIPELINE_NEW = "new"
PIPELINE_OLD = "old"


def _detect_pipeline() -> tuple[str, type, str]:
    """Return (label, processor_class, ingest_method_name) for the importable pipeline."""
    try:
        from aiperf.metrics.accumulator import MetricsAccumulator

        return PIPELINE_NEW, MetricsAccumulator, "process_record"
    except ImportError:
        from aiperf.post_processors.metric_results_processor import (
            MetricResultsProcessor,
        )

        return PIPELINE_OLD, MetricResultsProcessor, "process_result"


_MINIMAL_CONFIG_KWARGS: dict[str, Any] = {
    "models": ["test-model"],
    "endpoint": {
        "type": "chat",
        "urls": ["http://localhost:8000/v1/test"],
        "streaming": True,
    },
    "datasets": [
        {
            "name": "main",
            "type": "synthetic",
            "entries": 1,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {
            "name": "profiling",
            "type": "concurrency",
            "concurrency": 1,
            "requests": 10,
            "dataset": "main",
        }
    ],
}


def _make_run(slice_duration: float | None) -> SimpleNamespace:
    cfg = BenchmarkConfig(**_MINIMAL_CONFIG_KWARGS)
    if slice_duration is not None:
        # slice_duration lives on cfg.output and defaults to None on both branches;
        # set it via assignment so the new-pipeline picks it up in __init__.
        cfg.output.slice_duration = slice_duration
    return SimpleNamespace(cfg=cfg, resolved=SimpleNamespace(tokenizer_names={}))


# Numeric metric tags found in real ``profile_export.jsonl`` artifacts. Keeping
# this list aligned with the real export shape is critical: the per-record
# dispatch cost in ``ColumnStore.ingest`` scales with the number of tags, so a
# 5-tag synthetic understates the per-record overhead by ~5x. UUID-shaped
# string fields below match real metadata cardinality for the same reason.
_REAL_EXPORT_NUMERIC_TAGS: tuple[str, ...] = (
    "request_latency",
    "http_req_blocked",
    "http_req_connecting",
    "http_req_sending",
    "http_req_waiting",
    "http_req_dns_lookup",
    "http_req_receiving",
    "output_token_count",
    "http_req_chunks_sent",
    "output_sequence_length",
    "time_to_first_token",
    "http_req_duration",
    "input_sequence_length",
    "http_req_connection_reused",
    "http_req_data_received",
    "time_to_second_token",
    "time_to_first_output_token",
    "http_req_data_sent",
    "http_req_chunks_received",
    "http_req_connection_overhead",
    "http_req_total",
    "inter_token_latency",
    "prefill_throughput_per_user",
    "output_token_throughput_per_user",
)


def _uuid4_hex_dashed(np_rng: np.random.Generator) -> str:
    """Return a 36-char string in canonical UUID-4 layout — same wire shape and
    cardinality as real ``x_request_id`` values, without the ``uuid`` module
    overhead in the hot path."""
    raw = np_rng.bytes(16)
    h = raw.hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _generate_records(
    n_records: int,
    avg_icl_chunks: int,
    *,
    seed: int = 42,
    avg_turns_per_session: int = 5,
    n_conversation_templates: int = 1000,
) -> list[MetricRecordsData]:
    """Build n_records synthetic ``MetricRecordsData`` matching the real
    ``profile_export.jsonl`` schema as closely as possible:

    - 24 numeric metric tags (the full real-export set), each with a
      plausible distribution shape (log-normal for latencies, integer
      counts for chunks_sent/received, KB scalars for byte counts).
    - 1 list-valued ``inter_chunk_latency`` (Poisson chunk count around
      ``avg_icl_chunks``, log-normal values around 30 ms).
    - **``x_request_id`` cardinality = n_records** (one fresh UUID per
      request, just like real production).
    - **``x_correlation_id`` cardinality ≈ n_records / avg_turns_per_session**
      (sticky-routed conversation instances; defaults to ~5-turn sessions
      to model a typical multi-turn chat workload).
    - **``conversation_id`` cardinality = n_conversation_templates**
      (dataset template ID, replayed across many sessions; defaults to
      1000 templates).
    - Short labels for ``worker_id`` (~32 unique), ``record_processor_id``
      (~8 unique), ``benchmark_phase`` (single literal).
    """
    np_rng = np.random.default_rng(seed)
    base_ns = 1_700_000_000_000_000_000

    # Pre-roll bulk arrays for tags that have a well-defined distribution shape.
    # Generic "scalar" tags use a wide log-normal so the sketch + percentiles
    # path actually exercises non-trivial spread.
    n = n_records
    ttfts = np_rng.lognormal(mean=np.log(50.0), sigma=0.4, size=n)  # ms
    latencies = np_rng.lognormal(mean=np.log(500.0), sigma=0.5, size=n)  # ms
    chunk_counts = np_rng.poisson(lam=avg_icl_chunks, size=n).clip(min=1)
    isls = np_rng.integers(low=200, high=2000, size=n)
    osl_offsets = np_rng.integers(low=-5, high=5, size=n)
    http_blocked = np_rng.lognormal(mean=np.log(0.05), sigma=1.0, size=n)
    http_dns = np_rng.lognormal(mean=np.log(0.5), sigma=1.5, size=n)
    http_connect = np_rng.lognormal(mean=np.log(2.0), sigma=0.8, size=n)
    http_sending = np_rng.lognormal(mean=np.log(2.0), sigma=0.6, size=n)
    http_receiving = np_rng.lognormal(mean=np.log(7.0), sigma=0.5, size=n)
    data_received_kb = np_rng.lognormal(mean=np.log(200.0), sigma=0.8, size=n)
    data_sent_kb = np_rng.lognormal(mean=np.log(1.5), sigma=0.5, size=n)
    chunks_received = chunk_counts + 1
    connection_reused = np_rng.integers(low=0, high=2, size=n)
    ttst = np_rng.lognormal(mean=np.log(30.0), sigma=0.5, size=n)
    iter_token_latency = np_rng.lognormal(mean=np.log(30.0), sigma=0.4, size=n)
    prefill_tput_per_user = np_rng.lognormal(mean=np.log(500.0), sigma=0.4, size=n)
    output_tput_per_user = np_rng.lognormal(mean=np.log(35.0), sigma=0.4, size=n)
    connection_overhead = http_connect + http_dns

    # Pre-allocate small-cardinality string pools so multi-turn / template-replay
    # workloads collapse into a finite set of unique strings (matching real
    # categorical-encoding savings rather than synthetic full-cardinality pessimism).
    n_correlation_ids = max(1, n // max(1, avg_turns_per_session))
    correlation_pool = [_uuid4_hex_dashed(np_rng) for _ in range(n_correlation_ids)]
    conversation_pool = [
        _uuid4_hex_dashed(np_rng) for _ in range(n_conversation_templates)
    ]
    correlation_assignments = np_rng.integers(0, n_correlation_ids, size=n)
    conversation_assignments = np_rng.integers(0, n_conversation_templates, size=n)
    # Per-record credit-queue wait — log-normal so the derived
    # credit_to_start_latency / effective_latency percentiles have a real
    # tail rather than a degenerate fixed value.
    credit_queue_ns = np_rng.lognormal(
        mean=np.log(500_000.0), sigma=0.6, size=n
    ).astype(np.int64)
    # Per-record turn_index (0..avg_turns_per_session-1, uniform within a session).
    turn_indices = np_rng.integers(0, max(1, avg_turns_per_session), size=n)

    records: list[MetricRecordsData] = []
    cur_start = base_ns
    for i in range(n):
        n_chunks = int(chunk_counts[i])
        # ICL values: log-normal median 30 ms, sigma_log 0.5 — tolist() to match
        # the real wire (Python list, not ndarray) since msgpack decode produces lists.
        icl_arr = np_rng.lognormal(mean=np.log(30.0), sigma=0.5, size=n_chunks).tolist()
        ttft = float(ttfts[i])
        lat = float(latencies[i])
        request_start = cur_start
        request_end = request_start + int(lat * 1_000_000)  # ms -> ns wall offset
        cur_start += int(np_rng.integers(1_000_000, 10_000_000))

        meta = MetricRecordMetadata(
            session_num=i,
            request_num=i,
            request_start_ns=request_start,
            request_end_ns=request_end,
            credit_issued_ns=request_start - int(credit_queue_ns[i]),
            request_ack_ns=request_start + 100_000,
            worker_id=f"worker_{i % 32:08x}",
            record_processor_id=f"record_processor_{i % 8:08x}",
            x_request_id=_uuid4_hex_dashed(np_rng),
            x_correlation_id=correlation_pool[correlation_assignments[i]],
            conversation_id=conversation_pool[conversation_assignments[i]],
            turn_index=int(turn_indices[i]),
            benchmark_phase="profiling",
        )
        # Full real-export tag set. Values are float for ms/KB/throughput tags,
        # int for counts — same Python-type distribution the wire delivers.
        metrics: dict[str, Any] = {
            "request_latency": lat,
            "http_req_blocked": float(http_blocked[i]),
            "http_req_connecting": float(http_connect[i]),
            "http_req_sending": float(http_sending[i]),
            "http_req_waiting": lat - 10.0,
            "http_req_dns_lookup": float(http_dns[i]),
            "http_req_receiving": float(http_receiving[i]),
            "output_token_count": int(n_chunks + 1),
            "http_req_chunks_sent": 1,
            "output_sequence_length": int(n_chunks + 1 + osl_offsets[i]),
            "time_to_first_token": ttft,
            "http_req_duration": lat,
            "input_sequence_length": int(isls[i]),
            "http_req_connection_reused": int(connection_reused[i]),
            "http_req_data_received": float(data_received_kb[i]),
            "time_to_second_token": float(ttst[i]),
            "inter_chunk_latency": icl_arr,
            "time_to_first_output_token": ttft + float(ttst[i]),
            "http_req_data_sent": float(data_sent_kb[i]),
            "http_req_chunks_received": int(chunks_received[i]),
            "http_req_connection_overhead": float(connection_overhead[i]),
            "http_req_total": lat,
            "inter_token_latency": float(iter_token_latency[i]),
            "prefill_throughput_per_user": float(prefill_tput_per_user[i]),
            "output_token_throughput_per_user": float(output_tput_per_user[i]),
        }
        records.append(MetricRecordsData(metadata=meta, metrics=metrics))
    return records


def _unwrap_metric(v: Any) -> Any:
    """Extract the numeric/list payload from an exported ``{"value": X, "unit": Y}`` dict.

    Real ``profile_export.jsonl`` files wrap every metric value in a unit envelope.
    The pipeline ingests bare scalars/lists, so we strip the envelope here. Magnitudes
    differ from native ingest (display ms vs native ns), but for memory/throughput
    measurement only the *shape* of the values matters.
    """
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v


def _load_records_from_jsonl(
    path: Path, *, repeat: int = 1
) -> tuple[list[MetricRecordsData], dict[str, Any]]:
    """Load records from a ``profile_export.jsonl`` file, looped ``repeat`` times.

    Each repeat re-issues monotonically-increasing ``session_num`` so the
    ColumnStore does not overwrite previous slots. ``metadata`` defaults that the
    export omits (``record_processor_id``, ``request_num``) get filled in from
    the original record or synthesized.

    Returns (records, summary) where ``summary`` reports tag coverage and
    chunk-count distribution drawn from the source file (one cycle).
    """
    raw_lines: list[dict[str, Any]] = []
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_lines.append(json.loads(line))

    if not raw_lines:
        raise RuntimeError(f"no records in {path}")

    tag_counts: dict[str, int] = {}
    list_tag_counts: dict[str, int] = {}
    chunk_count_total = 0
    for d in raw_lines:
        for tag, v in d.get("metrics", {}).items():
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            payload = _unwrap_metric(v)
            if isinstance(payload, list):
                list_tag_counts[tag] = list_tag_counts.get(tag, 0) + 1
                chunk_count_total += len(payload)

    records: list[MetricRecordsData] = []
    base_session = 0
    for _cycle in range(repeat):
        for j, d in enumerate(raw_lines):
            md = d.get("metadata", {})
            session_num = base_session + j
            meta = MetricRecordMetadata(
                session_num=session_num,
                request_num=md.get("request_num", session_num),
                x_request_id=md.get("x_request_id"),
                x_correlation_id=md.get("x_correlation_id"),
                conversation_id=md.get("conversation_id"),
                turn_index=md.get("turn_index"),
                credit_issued_ns=md.get("credit_issued_ns"),
                credit_received_ns=md.get("credit_received_ns"),
                request_start_ns=md.get("request_start_ns") or 0,
                request_ack_ns=md.get("request_ack_ns"),
                request_end_ns=md.get("request_end_ns") or 0,
                worker_id=md.get("worker_id") or "worker_unknown",
                record_processor_id=md.get("record_processor_id") or "rp_unknown",
                benchmark_phase=md.get("benchmark_phase") or "profiling",
                was_cancelled=md.get("was_cancelled", False),
                cancellation_time_ns=md.get("cancellation_time_ns"),
                clock_offset_ns=md.get("clock_offset_ns"),
            )
            metrics = {
                tag: _unwrap_metric(v) for tag, v in d.get("metrics", {}).items()
            }
            records.append(MetricRecordsData(metadata=meta, metrics=metrics))
        base_session += len(raw_lines)

    summary = {
        "source_path": str(path),
        "source_lines": len(raw_lines),
        "repeat": repeat,
        "total_records": len(records),
        "n_unique_tags": len(tag_counts),
        "n_list_tags": len(list_tag_counts),
        "list_tags": sorted(list_tag_counts),
        "tags_top": sorted(tag_counts.items(), key=lambda kv: -kv[1])[:10],
        "avg_chunks_per_record_in_source": (chunk_count_total / max(1, len(raw_lines))),
    }
    return records, summary


@dataclass
class StageResult:
    wall_time_s: float = 0.0
    tracemalloc_peak_bytes: int = 0
    rss_after_bytes: int = 0


@dataclass
class StructureBreakdown:
    """Pipeline-specific size attribution via pympler.asizeof.

    ``parts`` may contain plain ints (top-level totals) and nested dicts
    (per-tag breakdowns), so the value type is ``Any``.
    """

    total_bytes: int = 0
    parts: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerResult:
    pipeline: str
    n_records: int
    avg_icl_chunks: int
    slice_duration: float | None
    record_count_actual: int
    total_icl_chunks_generated: int
    ingest: StageResult
    summarize: StageResult
    breakdown: StructureBreakdown
    final_rss_bytes: int
    notes: dict[str, Any]


def _rss_bytes() -> int:
    """Current resident set size in bytes (Linux /proc fast path; rusage fallback)."""
    try:
        with open("/proc/self/status", "r") as f:  # noqa: UP015
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except OSError:
        pass
    # Linux ru_maxrss is in KiB; macOS is in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _measure_breakdown_new(processor: Any) -> StructureBreakdown:
    """Per-substructure asizeof for MetricsAccumulator (column_store + ragged).

    ``per_tag_numeric``: scalar RECORD/AGGREGATE metric storage broken down by
    metric tag. Comparable to the old pipeline's ``per_tag_metric_array``.
    ``per_tag_ragged``: list-valued RECORD storage by tag.
    ``per_field_metadata_numeric`` / ``per_field_metadata_string`` /
    ``per_field_metadata_bool`` / ``per_field_metadata_categorical``: per-field
    metadata column sizes. The old pipeline does not store metadata at all, so
    these have no old-side counterpart.
    ``per_field_timestamps``: per-field sizes of start_ns / end_ns /
    generation_start_ns. Same — new-only.
    """
    store = processor._column_store
    per_tag_numeric: dict[str, int] = {
        tag: asizeof.asizeof(arr) for tag, arr in store._numeric.items()
    }
    per_tag_ragged: dict[str, int] = {
        tag: asizeof.asizeof(rag) for tag, rag in store._ragged.items()
    }
    per_field_metadata_numeric: dict[str, int] = {
        tag: asizeof.asizeof(arr) for tag, arr in store._metadata_numeric.items()
    }
    per_field_metadata_string: dict[str, int] = {
        tag: asizeof.asizeof(col) for tag, col in store._metadata_string.items()
    }
    per_field_metadata_bool: dict[str, int] = {
        tag: asizeof.asizeof(arr)
        for tag, arr in getattr(store, "_metadata_bool", {}).items()
    }
    per_field_metadata_categorical: dict[str, int] = {
        # Categorical = int16 codes column + the per-tag string-pool dict.
        tag: asizeof.asizeof(arr)
        + asizeof.asizeof(getattr(store, "_metadata_categories", {}).get(tag, {}))
        for tag, arr in getattr(store, "_metadata_categorical", {}).items()
    }
    per_field_timestamps: dict[str, int] = {
        "start_ns": asizeof.asizeof(store.start_ns),
        "end_ns": asizeof.asizeof(store.end_ns),
        "generation_start_ns": asizeof.asizeof(store.generation_start_ns),
    }
    parts: dict[str, Any] = {
        "column_store_numeric_arrays": sum(per_tag_numeric.values()),
        "column_store_string_arrays": asizeof.asizeof(list(store._string.values())),
        "column_store_ragged": sum(per_tag_ragged.values()),
        "column_store_metadata_numeric": sum(per_field_metadata_numeric.values()),
        "column_store_metadata_string": sum(per_field_metadata_string.values()),
        "column_store_metadata_bool": sum(per_field_metadata_bool.values()),
        "column_store_metadata_categorical": sum(
            per_field_metadata_categorical.values()
        ),
        "column_store_running_sums": asizeof.asizeof(store._sums)
        + asizeof.asizeof(store._counts),
        "column_store_timestamps": sum(per_field_timestamps.values()),
        "per_tag_numeric": per_tag_numeric,
        "per_tag_ragged": per_tag_ragged,
        "per_field_metadata_numeric": per_field_metadata_numeric,
        "per_field_metadata_string": per_field_metadata_string,
        "per_field_metadata_bool": per_field_metadata_bool,
        "per_field_metadata_categorical": per_field_metadata_categorical,
        "per_field_timestamps": per_field_timestamps,
        "numeric_tag_count": len(per_tag_numeric),
        "ragged_tag_count": len(per_tag_ragged),
    }
    total = asizeof.asizeof(store)
    parts["__total_column_store"] = total
    return StructureBreakdown(total_bytes=total, parts=parts)


def _measure_breakdown_old(processor: Any) -> StructureBreakdown:
    """Per-substructure asizeof for MetricResultsProcessor._results.

    ``per_tag_metric_array``: scalar RECORD-metric MetricArrays by tag.
    ``per_tag_tdigest``: list-valued RECORD t-digest sketches by tag.
    """
    from aiperf.metrics.list_metric_aggregation import TDigestListMetricAggregator
    from aiperf.metrics.metric_dicts import MetricArray

    per_tag_metric_array: dict[str, int] = {}
    per_tag_tdigest: dict[str, int] = {}
    other_total = 0
    for tag, val in processor._results.items():
        sz = asizeof.asizeof(val)
        if isinstance(val, TDigestListMetricAggregator):
            per_tag_tdigest[tag] = sz
        elif isinstance(val, MetricArray):
            per_tag_metric_array[tag] = sz
        else:
            other_total += sz
    total = asizeof.asizeof(processor._results)
    parts: dict[str, Any] = {
        "metric_array_total": sum(per_tag_metric_array.values()),
        "metric_array_count": len(per_tag_metric_array),
        "tdigest_total": sum(per_tag_tdigest.values()),
        "tdigest_count": len(per_tag_tdigest),
        "other_scalar_total": other_total,
        "instances_map": asizeof.asizeof(processor._instances_map),
        "per_tag_metric_array": per_tag_metric_array,
        "per_tag_tdigest": per_tag_tdigest,
        "__total_results_dict": total,
    }
    return StructureBreakdown(total_bytes=total, parts=parts)


def _measure_breakdown(pipeline: str, processor: Any) -> StructureBreakdown:
    if pipeline == PIPELINE_NEW:
        return _measure_breakdown_new(processor)
    return _measure_breakdown_old(processor)


async def _run_pipeline(
    pipeline: str,
    processor_cls: type,
    ingest_method: str,
    records: list[MetricRecordsData],
    slice_duration: float | None,
    *,
    skip_breakdown: bool,
) -> WorkerResult:
    """Drive the pipeline through ingest + summarize and collect timings/sizes."""
    run = _make_run(slice_duration)

    # Construct processor outside any tracemalloc window — registry init etc.
    # is one-time and not what we're measuring.
    processor = processor_cls(run=run)
    rss_before = _rss_bytes()

    ingest = StageResult()
    summarize_stage = StageResult()
    ingest_method_callable = getattr(processor, ingest_method)

    # --- Ingest stage ---
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    for rec in records:
        await ingest_method_callable(rec)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ingest.wall_time_s = t1 - t0
    ingest.tracemalloc_peak_bytes = peak
    ingest.rss_after_bytes = _rss_bytes()

    # --- Summarize stage ---
    gc.collect()
    tracemalloc.start()
    t2 = time.perf_counter()
    summary = await processor.summarize()
    t3 = time.perf_counter()
    _, peak2 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    summarize_stage.wall_time_s = t3 - t2
    summarize_stage.tracemalloc_peak_bytes = peak2
    summarize_stage.rss_after_bytes = _rss_bytes()

    # --- Structure breakdown (post-summarize, separate from timing) ---
    breakdown = (
        StructureBreakdown(total_bytes=0, parts={})
        if skip_breakdown
        else _measure_breakdown(pipeline, processor)
    )

    notes: dict[str, Any] = {
        "rss_before_ingest_bytes": rss_before,
        "summary_type": type(summary).__name__,
    }
    if pipeline == PIPELINE_NEW:
        notes["new_results_count"] = len(summary.results)
        notes["new_timeslice_count"] = (
            0 if summary.timeslices is None else len(summary.timeslices)
        )
    else:
        notes["old_results_count"] = len(summary)

    icl_lens = [
        len(r.metrics["inter_chunk_latency"])
        for r in records
        if isinstance(r.metrics.get("inter_chunk_latency"), list)
    ]
    return WorkerResult(
        pipeline=pipeline,
        n_records=len(records),
        avg_icl_chunks=int(sum(icl_lens[:64]) / max(1, min(64, len(icl_lens))))
        if icl_lens
        else 0,
        slice_duration=slice_duration,
        record_count_actual=len(records),
        total_icl_chunks_generated=sum(icl_lens),
        ingest=ingest,
        summarize=summarize_stage,
        breakdown=breakdown,
        final_rss_bytes=_rss_bytes(),
        notes=notes,
    )


def _result_to_dict(r: WorkerResult) -> dict[str, Any]:
    return {
        "pipeline": r.pipeline,
        "n_records": r.n_records,
        "avg_icl_chunks": r.avg_icl_chunks,
        "slice_duration": r.slice_duration,
        "record_count_actual": r.record_count_actual,
        "total_icl_chunks_generated": r.total_icl_chunks_generated,
        "ingest": {
            "wall_time_s": r.ingest.wall_time_s,
            "tracemalloc_peak_bytes": r.ingest.tracemalloc_peak_bytes,
            "rss_after_bytes": r.ingest.rss_after_bytes,
            "records_per_second": (
                r.record_count_actual / r.ingest.wall_time_s
                if r.ingest.wall_time_s > 0
                else 0
            ),
        },
        "summarize": {
            "wall_time_s": r.summarize.wall_time_s,
            "tracemalloc_peak_bytes": r.summarize.tracemalloc_peak_bytes,
            "rss_after_bytes": r.summarize.rss_after_bytes,
        },
        "breakdown": {
            "total_bytes": r.breakdown.total_bytes,
            "parts": r.breakdown.parts,
        },
        "final_rss_bytes": r.final_rss_bytes,
        "notes": r.notes,
    }


async def _amain(args: argparse.Namespace) -> None:
    pipeline, processor_cls, ingest_method = _detect_pipeline()
    source_summary: dict[str, Any] | None = None
    if args.records_file is not None:
        records, source_summary = _load_records_from_jsonl(
            Path(args.records_file), repeat=args.repeat
        )
    else:
        records = _generate_records(
            args.n_records,
            args.avg_icl_chunks,
            seed=args.seed,
            avg_turns_per_session=args.avg_turns_per_session,
            n_conversation_templates=args.n_conversation_templates,
        )
    result = await _run_pipeline(
        pipeline,
        processor_cls,
        ingest_method,
        records,
        args.slice_duration,
        skip_breakdown=args.no_asizeof,
    )
    payload = _result_to_dict(result)
    if args.label is not None:
        payload["label"] = args.label
    if source_summary is not None:
        payload["source"] = source_summary
    else:
        payload["source"] = {
            "kind": "synthetic",
            "n_records": args.n_records,
            "avg_icl_chunks": args.avg_icl_chunks,
            "seed": args.seed,
        }
    print(json.dumps(payload))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_argument_group("record source (one required)")
    src.add_argument(
        "--records-file",
        type=str,
        default=None,
        help="Path to a profile_export.jsonl file. Replays it through the pipeline. "
        "Mutually exclusive with --n-records / --avg-icl-chunks.",
    )
    src.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="When --records-file is set, replay the file this many times with "
        "monotonically-increasing session_num (default 1).",
    )
    src.add_argument(
        "--n-records",
        type=int,
        default=None,
        help="Synthetic mode: number of records to generate.",
    )
    src.add_argument(
        "--avg-icl-chunks",
        type=int,
        default=None,
        help="Synthetic mode: target avg chunks per inter_chunk_latency list.",
    )
    p.add_argument(
        "--avg-turns-per-session",
        type=int,
        default=5,
        help="Synthetic mode: avg turns per conversation instance — drives "
        "x_correlation_id cardinality (= n_records / avg_turns_per_session). "
        "Default 5 models a typical multi-turn chat workload; pass 1 to model "
        "single-turn streaming where every record gets a fresh correlation_id.",
    )
    p.add_argument(
        "--n-conversation-templates",
        type=int,
        default=1000,
        help="Synthetic mode: pool size for conversation_id (template IDs from "
        "the dataset). Default 1000 models a typical template-replay workload; "
        "pass n_records to model the worst case where every record uses a unique "
        "template.",
    )
    p.add_argument(
        "--slice-duration",
        type=float,
        default=None,
        help="Timeslice duration in seconds; None disables timeslicing.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", type=str, default=None)
    p.add_argument(
        "--no-asizeof",
        action="store_true",
        help="Skip pympler.asizeof breakdown (faster on huge runs).",
    )
    p.add_argument(
        "--json", action="store_true", help="Reserved; output is always JSON."
    )
    args = p.parse_args()
    if args.records_file is None and (
        args.n_records is None or args.avg_icl_chunks is None
    ):
        p.error(
            "either --records-file or both --n-records and --avg-icl-chunks must be set"
        )
    if args.records_file is not None and (
        args.n_records is not None or args.avg_icl_chunks is not None
    ):
        p.error(
            "--records-file is mutually exclusive with --n-records / --avg-icl-chunks"
        )
    # Avoid tee'd stdout going through the logger
    os.environ.setdefault("AIPERF_LOG_LEVEL", "WARNING")
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
