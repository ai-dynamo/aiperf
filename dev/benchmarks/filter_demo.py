#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Demo: filter + summarize by metadata fields against MetricsAccumulator.

Exercises the column_store filter primitives and shows what composes today
without further code changes:

- mask_for_categorical(tag, value) for worker_id / record_processor_id /
  benchmark_phase / x_correlation_id / conversation_id
- metadata_bool(tag) == 1 for was_cancelled / has_error
- query_time_range(start_ns, end_ns) for time-window slicing
- raw numpy comparison on metadata_numeric for ad-hoc range filters
- numpy boolean ops (& | ~) for multi-field combinations
- accumulator.compute_results_for_mask(mask) as the summarize-subset entry point
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

    # Synthesize 8 records across 2 workers, 2 conversations, 2 phases —
    # plus one cancelled and one errored.
    rng = np.random.default_rng(42)
    base = 1_700_000_000_000_000_000
    workers = ["worker_a", "worker_b"]
    convs = ["conv_x", "conv_y"]
    phases = ["warmup", "profiling"]
    for i in range(8):
        meta = MetricRecordMetadata(
            session_num=i,
            request_num=i,
            request_start_ns=base + i * 100_000_000,
            request_end_ns=base + i * 100_000_000 + 50_000_000,
            credit_issued_ns=base + i * 100_000_000 - 1_000_000,
            request_ack_ns=base + i * 100_000_000 + 100_000,
            worker_id=workers[i % 2],
            record_processor_id="rp1",
            x_request_id=f"req-{i}",
            x_correlation_id=f"corr-{i % 2}",
            conversation_id=convs[i % 2],
            turn_index=0,
            benchmark_phase=phases[
                1 if i >= 2 else 0
            ],  # 0,1 = warmup; 2..7 = profiling
            was_cancelled=(i == 3),
        )
        msg = MetricRecordsData(
            metadata=meta,
            metrics={
                "request_latency": float(rng.uniform(100, 1000)),
                "time_to_first_token": float(rng.uniform(20, 80)),
                "output_token_count": int(rng.integers(10, 50)),
            },
            error=None,  # set on i==5 below
        )
        await proc.process_record(msg)

    # ------------------------------------------------------------------
    # Filter primitives — every one of these is supported today
    # ------------------------------------------------------------------
    store = proc._column_store

    # 1. Filter by categorical metadata field
    workers_unique = store.unique_categorical_values("worker_id")
    print(f"unique workers: {workers_unique}")
    mask_worker_a = store.mask_for_categorical("worker_id", "worker_a")
    print(f"worker_a mask:        {mask_worker_a.astype(int).tolist()}")

    # 2. Filter by bool metadata field — numpy comparison
    cancelled_mask = store.metadata_bool("was_cancelled") == 1
    print(f"cancelled mask:       {cancelled_mask.astype(int).tolist()}")

    # 3. Filter by time window (using request_start_ns timestamps)
    win_mask = proc.query_time_range(base + 200_000_000, base + 600_000_000)
    print(f"time-window mask:     {win_mask.astype(int).tolist()}")

    # 4. Filter by numeric metadata range (ad-hoc, no helper today but composable)
    issued = store.metadata_numeric("credit_issued_ns")
    issued_mask = (issued >= base) & (issued < base + 400_000_000)
    print(f"credit_issued range:  {issued_mask.astype(int).tolist()}")

    # 5. Compose: profiling phase ∩ worker_b ∩ NOT cancelled
    phase_mask = store.mask_for_categorical("benchmark_phase", "profiling")
    worker_b_mask = store.mask_for_categorical("worker_id", "worker_b")
    composed = phase_mask & worker_b_mask & ~cancelled_mask
    print(f"composed (phase∩wb∩~cancelled): {composed.astype(int).tolist()}")

    # 6. Summarize that filtered subset
    results = proc.compute_results_for_mask(composed)
    print()
    print(f"summarize subset (n={composed.sum()}):")
    for tag in ("request_latency", "time_to_first_token", "output_token_count"):
        if tag in results:
            r = results[tag]
            print(f"  {tag:<25} avg={r.avg:.2f} count={r.count}")

    # 7. Per-X grouping: enumerate groups and summarize each
    print()
    print("per-conversation grouping:")
    for value in store.unique_categorical_values("conversation_id"):
        mask = store.mask_for_categorical("conversation_id", value)
        sub = proc.compute_results_for_mask(mask)
        lat = sub.get("request_latency")
        tput = sub.get("output_token_count")
        avg_lat = f"{lat.avg:.1f}" if lat else "n/a"
        avg_tok = f"{tput.avg:.1f}" if tput else "n/a"
        print(
            f"  {value:<10} n={int(mask.sum())} avg_latency={avg_lat} avg_tokens={avg_tok}"
        )


if __name__ == "__main__":
    asyncio.run(main())
