#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-worker percentiles of credit-to-request-start latency.

Demonstrates how to combine the column-store filter primitives + the stored
timestamp + numeric-metadata columns to derive a custom per-X metric without
adding new API surface. Specifically: for each ``worker_id``, computes
``request_start_ns - credit_issued_ns`` percentiles — the time a credit spent
between issuance and the worker actually starting the request.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from aiperf.common.metric_records_wire import MetricRecordMetadata, MetricRecordsData

from aiperf.config import BenchmarkConfig
from aiperf.metrics.accumulator import MetricsAccumulator


def _make_run() -> SimpleNamespace:
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
    return SimpleNamespace(cfg=cfg, resolved=SimpleNamespace(tokenizer_names={}))


async def main() -> None:
    proc = MetricsAccumulator(run=_make_run())
    rng = np.random.default_rng(42)

    # 4 workers, 5000 records, varied per-worker credit→start latency profiles
    n_per = 5000
    base = 1_700_000_000_000_000_000
    # Each worker has a different per-worker queue-pickup latency mean,
    # so percentiles differ — this is what production looks like.
    worker_means_us = {
        "worker_a": 50,  # fastest
        "worker_b": 200,
        "worker_c": 800,  # slowest, contention
        "worker_d": 120,
    }
    workers = list(worker_means_us.keys())

    for w_idx, worker in enumerate(workers):
        mean_ns = worker_means_us[worker] * 1_000  # us → ns
        for j in range(n_per):
            i = w_idx * n_per + j
            credit_issued_ns = base + i * 100_000
            # Per-worker queue latency, log-normal so we get a real tail
            queue_lat = int(rng.lognormal(np.log(mean_ns), 0.5))
            request_start_ns = credit_issued_ns + queue_lat
            meta = MetricRecordMetadata(
                session_num=i,
                request_num=i,
                request_start_ns=request_start_ns,
                request_end_ns=request_start_ns + 50_000_000,
                credit_issued_ns=credit_issued_ns,
                request_ack_ns=request_start_ns + 100_000,
                worker_id=worker,
                record_processor_id="rp1",
                x_request_id=f"req-{i}",
                x_correlation_id=f"corr-{i}",
                conversation_id="conv-x",
                turn_index=0,
                benchmark_phase="profiling",
            )
            await proc.process_record(
                MetricRecordsData(
                    metadata=meta,
                    metrics={"request_latency": 50.0},
                    error=None,
                )
            )

    # ---- The actual analysis (this is the part you'd write in real code) ----
    store = proc._column_store
    n = store.count

    # Pull both columns at once as float64 arrays
    start_ns = store.start_ns[:n]
    issued_ns = store.metadata_numeric("credit_issued_ns")  # NaN where missing

    # Derived per-record: nanoseconds from credit issuance to request start
    credit_to_start_ns = start_ns - issued_ns  # NaN propagates correctly

    print(
        f"{'worker':<10} {'n':>6} {'p50 (us)':>10} {'p90 (us)':>10} {'p95 (us)':>10} {'p99 (us)':>10}  {'max (us)':>10}"
    )
    for worker in store.unique_categorical_values("worker_id"):
        mask = store.mask_for_categorical("worker_id", worker)
        values = credit_to_start_ns[mask]
        values = values[~np.isnan(values)]
        if len(values) == 0:
            continue
        p50, p90, p95, p99 = np.percentile(values, [50, 90, 95, 99])
        # Convert ns -> us for display
        print(
            f"{worker:<10} {len(values):>6} "
            f"{p50 / 1000:>10.1f} {p90 / 1000:>10.1f} {p95 / 1000:>10.1f} {p99 / 1000:>10.1f}  "
            f"{values.max() / 1000:>10.1f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
