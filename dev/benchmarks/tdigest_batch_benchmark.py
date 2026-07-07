#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: per-record vs cross-record batching for the t-digest wrapper.

Question: how much would batching ``inter_chunk_latency`` lists across multiple
records (instead of one ``td.update(arr)`` call per record) speed up
``TDigestListMetricAggregator``?

Raw ``crick.TDigest.update(arr)`` over large batched ndarrays peaks at
~12 M samples/s, but AIPerf's wrapper hits only ~2.2 M samples/s end-to-end.
This microbenchmark measures where the gap goes by varying the cross-record
batch size K:

- K=1:   current behavior (one td.update + Welford combine per record)
- K=10, 100, 1000: buffer K records, concat into one array, one td.update +
  one Welford combine
- K=all: theoretical maximum (one update over the entire dataset)

Welford parallel combine is included in every variant so we measure the same
exact-stats-preserving shape, not just raw td.update.

Usage:
    uv run python dev/benchmarks/tdigest_batch_benchmark.py
"""

from __future__ import annotations

import math
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from crick import TDigest

COMPRESSION = 500
N_RECORDS = 100_000
CHUNKS_PER_RECORD = 100
SEED = 42


@dataclass
class WelfordState:
    count: int = 0
    sum: float = 0.0
    mean: float = 0.0
    m2: float = 0.0
    min: float | None = None
    max: float | None = None


def welford_combine_batch(state: WelfordState, arr: np.ndarray) -> None:
    """Same parallel combine as ``TDigestListMetricAggregator.extend`` —
    keeps exact sum/mean/std/min/max alongside the sketch."""
    n_b = int(arr.size)
    if n_b == 0:
        return
    sum_b = float(arr.sum())
    mean_b = sum_b / n_b
    m2_b = float(((arr - mean_b) ** 2).sum())
    n_a = state.count
    if n_a == 0:
        state.mean = mean_b
        state.m2 = m2_b
    else:
        new_count = n_a + n_b
        delta = mean_b - state.mean
        state.mean += delta * n_b / new_count
        state.m2 += m2_b + delta * delta * n_a * n_b / new_count
    state.count += n_b
    state.sum += sum_b
    batch_min = float(arr.min())
    batch_max = float(arr.max())
    state.min = batch_min if state.min is None else min(state.min, batch_min)
    state.max = batch_max if state.max is None else max(state.max, batch_max)


def gen_records(n_records: int, chunks: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = np.clip(
        rng.lognormal(mean=math.log(5.0), sigma=0.4, size=n_records * chunks),
        0.5,
        50.0,
    ).astype(np.float64)
    return [raw[i * chunks : (i + 1) * chunks] for i in range(n_records)]


def run_per_record(records: list[np.ndarray]) -> tuple[float, int, WelfordState]:
    """Status-quo path: one td.update + one Welford combine per record."""
    td = TDigest(compression=COMPRESSION)
    state = WelfordState()
    tracemalloc.start()
    t0 = time.perf_counter()
    for arr in records:
        td.update(arr)
        welford_combine_batch(state, arr)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak, state


def run_buffered(
    records: list[np.ndarray], batch_size: int
) -> tuple[float, int, WelfordState]:
    """Buffer ``batch_size`` records' lists into a Python list, concat to a
    single ndarray, then one td.update + one Welford combine per buffer flush.

    Models the cross-record batching change: each record still hands its
    ICL list to the accumulator, but the accumulator defers td.update until
    a buffer hits the threshold.
    """
    td = TDigest(compression=COMPRESSION)
    state = WelfordState()
    buffer: list[np.ndarray] = []
    buffered_size = 0
    tracemalloc.start()
    t0 = time.perf_counter()
    for arr in records:
        buffer.append(arr)
        buffered_size += len(arr)
        if len(buffer) >= batch_size:
            big = np.concatenate(buffer)
            td.update(big)
            welford_combine_batch(state, big)
            buffer.clear()
            buffered_size = 0
    if buffer:
        big = np.concatenate(buffer)
        td.update(big)
        welford_combine_batch(state, big)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak, state


def run_all_at_once(records: list[np.ndarray]) -> tuple[float, int, WelfordState]:
    """Theoretical maximum: concat the entire dataset, one update + one combine."""
    td = TDigest(compression=COMPRESSION)
    state = WelfordState()
    tracemalloc.start()
    t0 = time.perf_counter()
    big = np.concatenate(records)
    td.update(big)
    welford_combine_batch(state, big)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak, state


def _percentiles(td: TDigest) -> dict[str, float]:
    return {f"p{p}": float(td.quantile(p / 100)) for p in (50, 90, 95, 99)}


def main() -> None:
    print(
        "# Cross-record batching benchmark — TDigestListMetricAggregator wrapper",
        flush=True,
    )
    print(
        f"# n_records={N_RECORDS:,}, chunks/record={CHUNKS_PER_RECORD}, "
        f"compression={COMPRESSION}, seed={SEED}",
        flush=True,
    )
    print(file=sys.stderr)
    print("Generating records...", file=sys.stderr, flush=True)
    records = gen_records(N_RECORDS, CHUNKS_PER_RECORD, SEED)
    total_samples = sum(len(r) for r in records)

    runs: list[tuple[str, float, int, WelfordState]] = []

    print("Running per_record (K=1)...", file=sys.stderr, flush=True)
    elapsed, peak, st = run_per_record(records)
    runs.append(("per_record (K=1)", elapsed, peak, st))

    for k in (10, 100, 1000, 10_000):
        print(f"Running buffered K={k}...", file=sys.stderr, flush=True)
        elapsed, peak, st = run_buffered(records, k)
        runs.append((f"buffered K={k:,}", elapsed, peak, st))

    print("Running all_at_once (K=all)...", file=sys.stderr, flush=True)
    elapsed, peak, st = run_all_at_once(records)
    runs.append(("all_at_once (K=all)", elapsed, peak, st))

    base_elapsed = runs[0][1]
    print(file=sys.stderr)
    print(
        f"| {'pattern':<22} | {'wall (s)':>9} | {'rec/s':>10} | {'samples/s':>12} | "
        f"{'speedup':>8} | {'tracemalloc peak (MB)':>22} |"
    )
    print(f"|{'-' * 24}|{'-' * 11}|{'-' * 12}|{'-' * 14}|{'-' * 10}|{'-' * 24}|")
    for name, elapsed, peak, _state in runs:
        rps = N_RECORDS / elapsed
        sps = total_samples / elapsed
        speedup = base_elapsed / elapsed
        print(
            f"| {name:<22} | {elapsed:>9.3f} | {rps:>10,.0f} | {sps:>12,.0f} | "
            f"{speedup:>7.2f}x | {peak / 1024 / 1024:>22.2f} |"
        )

    # Sanity check: percentiles agree across patterns (within sketch error)
    print(file=sys.stderr)
    print(
        "# Sanity: per-record vs all-at-once percentiles should match within sketch error",
        file=sys.stderr,
    )
    base_state = runs[0][3]
    last_state = runs[-1][3]
    print(
        f"# count: per-record={base_state.count:,} all-at-once={last_state.count:,}",
        file=sys.stderr,
    )
    print(
        f"# sum:   per-record={base_state.sum:,.4f} all-at-once={last_state.sum:,.4f}",
        file=sys.stderr,
    )
    print(
        f"# min:   per-record={base_state.min:.6f} all-at-once={last_state.min:.6f}",
        file=sys.stderr,
    )
    print(
        f"# max:   per-record={base_state.max:.6f} all-at-once={last_state.max:.6f}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
