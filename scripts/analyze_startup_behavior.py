#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep investigation of credit startup behavior from a profile_export.jsonl.

Analyzes the first N credits (default 250,000) to characterize:
  1. Credit issuance rate over time (1s buckets)
  2. Credit pipeline latency (ZMQ transit, dispatch, total)
  3. Pipeline latency evolution over time (200ms buckets)
  4. Worker activation order and spread
  5. Credit distribution balance across workers
  6. Credit issuance gaps / stalls
  7. Request ack latency (start -> first server ack)
  8. In-flight concurrency build-up (100ms resolution)

Usage:
    python scripts/analyze_startup_behavior.py <profile_export.jsonl> [--n 250000]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import orjson


def load_records(path: Path, n: int) -> list[dict]:
    """Load and return the first n records sorted by credit_issued_ns."""
    records: list[dict] = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))
    records.sort(key=lambda r: r["metadata"]["credit_issued_ns"])
    return records[:n]


def ns_to_ms(ns: int, t0_ns: int) -> float:
    return (ns - t0_ns) / 1e6


def dur_ms(a_ns: int, b_ns: int) -> float:
    return (b_ns - a_ns) / 1e6


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def percentile_row(arr: np.ndarray, label: str) -> None:
    print(
        f"  {label}\n"
        f"    Mean: {np.mean(arr):.3f}ms  Median: {np.median(arr):.3f}ms  Std: {np.std(arr):.3f}ms\n"
        f"    P50: {np.percentile(arr, 50):.3f}ms  P90: {np.percentile(arr, 90):.3f}ms  "
        f"P99: {np.percentile(arr, 99):.3f}ms  P99.9: {np.percentile(arr, 99.9):.3f}ms  "
        f"Max: {np.max(arr):.3f}ms"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Path to profile_export.jsonl")
    parser.add_argument(
        "--n",
        type=int,
        default=250_000,
        metavar="N",
        help="Number of startup credits to analyze (default: 250000)",
    )
    parser.add_argument(
        "--stall-threshold-ms",
        type=float,
        default=1.0,
        metavar="MS",
        help="Minimum gap to report as a stall (default: 1.0ms)",
    )
    args = parser.parse_args(argv)

    path = Path(args.input)
    n = args.n

    print(f"Loading {path} ...")
    startup = load_records(path, n)
    actual_n = len(startup)
    print(f"Loaded {actual_n:,} records (requested {n:,})")

    t0_ns: int = startup[0]["metadata"]["credit_issued_ns"]

    issue_times_ms = np.array(
        [ns_to_ms(r["metadata"]["credit_issued_ns"], t0_ns) for r in startup]
    )
    total_duration_ms = issue_times_ms[-1] - issue_times_ms[0]

    # ── 1. Credit issuance rate ───────────────────────────────────────────────
    section("1. Credit Issuance Rate During Startup")
    print("  First credit:       t=0.000ms")
    print(
        f"  Credit #{actual_n:,}:    t={total_duration_ms:,.1f}ms  ({total_duration_ms / 1000:.2f}s)"
    )
    print(
        f"  Overall issue rate: {actual_n / (total_duration_ms / 1000):,.0f} credits/s"
    )

    buckets: dict[int, int] = defaultdict(int)
    for t in issue_times_ms:
        buckets[int(t / 1000)] += 1

    print("\n  Credits issued per second (1s buckets):")
    for sec in sorted(buckets):
        bar = "█" * (buckets[sec] // 1000)
        print(f"    t={sec:3d}s: {buckets[sec]:>7,}  {bar}")

    # ── 2. Credit pipeline latency ────────────────────────────────────────────
    section("2. Credit Pipeline Latency During Startup (all records)")

    issued_to_recv = np.array(
        [
            dur_ms(
                r["metadata"]["credit_issued_ns"], r["metadata"]["credit_received_ns"]
            )
            for r in startup
        ]
    )
    recv_to_start = np.array(
        [
            dur_ms(
                r["metadata"]["credit_received_ns"], r["metadata"]["request_start_ns"]
            )
            for r in startup
        ]
    )
    total_pipeline = issued_to_recv + recv_to_start

    for label, arr in [
        ("Issued -> Received (ZMQ transit)", issued_to_recv),
        ("Received -> Request Start", recv_to_start),
        ("Total pipeline (Issued -> Start)", total_pipeline),
    ]:
        percentile_row(arr, f"{label} (n={len(arr):,})")

    # ── 3. Pipeline latency over time (200ms buckets) ─────────────────────────
    section("3. Credit Pipeline Latency Over Time During Startup (200ms buckets)")
    print(
        f"  {'Bucket':>10}  {'Count':>7}  {'Med S->R':>9}  {'P99 S->R':>9}"
        f"  {'Med R->Start':>12}  {'In-flight':>10}"
    )

    bucket_ms = 200
    for b_start in range(0, int(total_duration_ms) + bucket_ms, bucket_ms):
        b_end = b_start + bucket_ms
        mask = (issue_times_ms >= b_start) & (issue_times_ms < b_end)
        if not mask.any():
            continue
        n_b = int(mask.sum())
        s2r = issued_to_recv[mask]
        r2s = recv_to_start[mask]
        in_flight = int((issue_times_ms < b_end).sum())
        print(
            f"  {b_start:>7}ms-{b_end:<5}ms  {n_b:>7,}"
            f"  {np.median(s2r):>8.3f}ms  {np.percentile(s2r, 99):>8.3f}ms"
            f"  {np.median(r2s):>11.3f}ms  {in_flight:>10,}"
        )

    # ── 4. Worker activation order ────────────────────────────────────────────
    section("4. Worker Activation Order (first credit per worker)")

    worker_first: dict[str, float] = {}
    worker_counts: dict[str, int] = defaultdict(int)
    for r in startup:
        wid = r["metadata"]["worker_id"]
        t = ns_to_ms(r["metadata"]["credit_issued_ns"], t0_ns)
        if wid not in worker_first:
            worker_first[wid] = t
        worker_counts[wid] += 1

    sorted_workers = sorted(worker_first.items(), key=lambda x: x[1])
    total_workers = len(sorted_workers)
    print(f"  Total workers active: {total_workers}")

    print("\n  First 20 workers to activate:")
    for i, (wid, t) in enumerate(sorted_workers[:20]):
        print(
            f"    [{i + 1:3d}] {wid:<22}  first at t={t:8.1f}ms  ({worker_counts[wid]:,} credits)"
        )

    if total_workers > 20:
        print("\n  Last 10 workers to activate:")
        for i, (wid, t) in enumerate(sorted_workers[-10:]):
            rank = total_workers - 9 + i
            print(
                f"    [{rank:3d}] {wid:<22}  first at t={t:8.1f}ms  ({worker_counts[wid]:,} credits)"
            )

    activation_times = np.array([t for _, t in sorted_workers])
    print("\n  Worker activation spread:")
    print(f"    First worker:     t={activation_times[0]:.1f}ms")
    print(f"    50% workers by:   t={np.percentile(activation_times, 50):.1f}ms")
    print(f"    90% workers by:   t={np.percentile(activation_times, 90):.1f}ms")
    print(f"    Last worker:      t={activation_times[-1]:.1f}ms")
    print(
        f"    Spread (p90-p10): {np.percentile(activation_times, 90) - np.percentile(activation_times, 10):.1f}ms"
    )

    # ── 5. Credit distribution balance ───────────────────────────────────────
    section("5. Credit Distribution Balance Across Workers")
    counts = np.array(list(worker_counts.values()))
    cv = np.std(counts) / np.mean(counts) * 100 if np.mean(counts) > 0 else 0.0
    print(f"  Workers active:  {len(counts)}")
    print(f"  Mean credits:    {np.mean(counts):.1f}")
    print(f"  Std credits:     {np.std(counts):.1f}  (CV={cv:.1f}%)")
    print(f"  Min/Max:         {np.min(counts)} / {np.max(counts)}")
    print(
        f"  P10/P90:         {np.percentile(counts, 10):.0f} / {np.percentile(counts, 90):.0f}"
    )

    top10 = sorted(worker_counts.items(), key=lambda x: -x[1])[:10]
    print("\n  Top 10 most loaded workers:")
    for wid, cnt in top10:
        bar = "█" * (cnt // 10)
        print(f"    {wid:<22}  {cnt:>5} credits  {bar}")

    # ── 6. Stalls / gaps in credit issuance ──────────────────────────────────
    section(f"6. Credit Issuance Gaps (stalls > {args.stall_threshold_ms}ms)")
    diffs = np.diff(issue_times_ms)
    stall_mask = diffs > args.stall_threshold_ms
    stalls = diffs[stall_mask]
    stall_indices = np.where(stall_mask)[0]

    print(f"  Total gaps > {args.stall_threshold_ms}ms: {len(stalls)}")
    if len(stalls):
        print(
            f"  Max gap:  {np.max(stalls):.3f}ms  at credit #{stall_indices[np.argmax(stalls)]:,}"
        )
        print(f"  Mean gap: {np.mean(stalls):.3f}ms")
        print("\n  Top 15 largest gaps:")
        top_idx = np.argsort(stalls)[-15:][::-1]
        for idx in top_idx:
            ci = stall_indices[idx]
            print(
                f"    credit #{ci:>7,}  at t={issue_times_ms[ci]:>8.1f}ms"
                f"  gap={diffs[ci]:>8.3f}ms"
                f"  worker={startup[ci]['metadata']['worker_id']}"
            )

    # ── 7. Request ack latency ────────────────────────────────────────────────
    section("7. Request Ack Latency (request_start -> request_ack)")
    ack_latencies = np.array(
        [
            dur_ms(r["metadata"]["request_start_ns"], r["metadata"]["request_ack_ns"])
            for r in startup
        ]
    )
    percentile_row(ack_latencies, f"Start -> Ack (n={len(ack_latencies):,})")

    # ── 8. In-flight concurrency build-up ────────────────────────────────────
    section("8. In-Flight Concurrency Build-Up (100ms resolution)")
    start_times_ms = np.array(
        [ns_to_ms(r["metadata"]["request_start_ns"], t0_ns) for r in startup]
    )

    print(f"  {'t(ms)':>8}  {'In-flight':>10}  {'Δ starts':>9}  {'Rate (req/s)':>12}")
    prev = 0
    for t in range(0, int(total_duration_ms) + 100, 100):
        count = int((start_times_ms <= t).sum())
        delta = count - prev
        rate = delta / 0.1 if t > 0 else 0.0
        bar = "▓" * min(50, delta // 200)
        print(f"  {t:>8}  {count:>10,}  {delta:>9,}  {rate:>11,.0f}  {bar}")
        prev = count

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
