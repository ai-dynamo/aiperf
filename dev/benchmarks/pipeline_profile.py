#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cProfile probe for the metrics-accumulator ingest + summarize path.

Generates a synthetic batch of ``MetricRecordsData`` matching the real-export
schema (24 numeric tags + ``inter_chunk_latency`` list), then runs cProfile
across the ingest loop and the summarize() call separately so we can see where
each stage actually spends its time.

Usage:
    uv run python dev/benchmarks/pipeline_profile.py
    AIPERF_METRICS_LIST_BACKEND=tdigest uv run python dev/benchmarks/pipeline_profile.py
"""

from __future__ import annotations

import asyncio
import cProfile
import io
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from aiperf.common.metric_records_wire import MetricRecordMetadata, MetricRecordsData

from aiperf.common.environment import Environment
from aiperf.config import BenchmarkConfig
from aiperf.metrics.accumulator import MetricsAccumulator

N_RECORDS = 50_000
SEED = 42

# 24 metric tags from a real profile_export.jsonl + inter_chunk_latency
SCALAR_TAGS = [
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
]


def _make_run() -> object:
    cfg = BenchmarkConfig(
        models=["test-model"],
        endpoint={
            "type": "chat",
            "urls": ["http://localhost:8000/v1/test"],
            "streaming": True,
        },
        datasets=[
            {
                "name": "main",
                "type": "synthetic",
                "entries": 1,
                "prompts": {"isl": 128, "osl": 64},
            }
        ],
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 1,
                "requests": 10,
                "dataset": "main",
            }
        ],
    )
    from types import SimpleNamespace

    return SimpleNamespace(cfg=cfg, resolved=SimpleNamespace(tokenizer_names={}))


def gen_records(n: int) -> list[MetricRecordsData]:
    rng = np.random.default_rng(SEED)
    base = 1_700_000_000_000_000_000
    chunk_counts = rng.poisson(lam=100, size=n).clip(min=1)
    out: list[MetricRecordsData] = []
    cur = base
    for i in range(n):
        ttft = int(rng.lognormal(np.log(50_000_000), 0.4))
        lat = int(rng.lognormal(np.log(500_000_000), 0.5))
        n_chunks = int(chunk_counts[i])
        icl = rng.lognormal(np.log(30.0), 0.5, n_chunks).tolist()
        meta = MetricRecordMetadata(
            session_num=i,
            request_num=i,
            request_start_ns=cur,
            request_end_ns=cur + lat,
            credit_issued_ns=cur - 1_000_000,
            request_ack_ns=cur + 100_000,
            worker_id=f"worker-{i % 32}",
            record_processor_id=f"rp-{i % 8}",
            x_request_id=f"req-{i:09d}",
            x_correlation_id=f"corr-{i % 1000:06d}",
            conversation_id=f"conv-{i % 1000:06d}",
            turn_index=0,
            benchmark_phase="profiling",
        )
        # Realistic: 24 numeric scalars + 1 list metric
        metrics: dict[str, float | int | list[float]] = {
            tag: float(rng.normal(100.0, 10.0)) for tag in SCALAR_TAGS
        }
        metrics["time_to_first_token"] = ttft
        metrics["request_latency"] = lat
        metrics["output_token_count"] = n_chunks + 1
        metrics["output_sequence_length"] = n_chunks + 1
        metrics["input_sequence_length"] = int(rng.integers(200, 2000))
        metrics["inter_chunk_latency"] = icl
        out.append(MetricRecordsData(metadata=meta, metrics=metrics))
        cur += int(rng.integers(1_000_000, 10_000_000))
    return out


async def _ingest(
    processor: MetricsAccumulator, records: list[MetricRecordsData]
) -> None:
    for r in records:
        await processor.process_record(r)


def _print_stats(prof: cProfile.Profile, label: str, top: int = 25) -> None:
    s = io.StringIO()
    pstats.Stats(prof, stream=s).strip_dirs().sort_stats("cumulative").print_stats(top)
    print(f"\n=== {label} (top {top} by cumulative) ===")
    print(s.getvalue())
    s2 = io.StringIO()
    pstats.Stats(prof, stream=s2).strip_dirs().sort_stats("tottime").print_stats(top)
    print(f"\n=== {label} (top {top} by tottime — own time only) ===")
    print(s2.getvalue())


def main() -> None:
    backend = Environment.METRICS.LIST_BACKEND
    print(f"# Pipeline profile — backend={backend}, n_records={N_RECORDS:,}")
    records = gen_records(N_RECORDS)
    processor = MetricsAccumulator(run=_make_run())

    prof_ingest = cProfile.Profile()
    prof_ingest.enable()
    asyncio.run(_ingest(processor, records))
    prof_ingest.disable()

    prof_summarize = cProfile.Profile()
    prof_summarize.enable()
    asyncio.run(processor.summarize())
    prof_summarize.disable()

    _print_stats(prof_ingest, f"INGEST {N_RECORDS:,} records (backend={backend})")
    _print_stats(prof_summarize, f"SUMMARIZE (backend={backend})")


if __name__ == "__main__":
    main()
