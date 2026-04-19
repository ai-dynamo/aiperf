#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credit pipeline latency per request from profile_export.jsonl."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import orjson


def get_metric_ms(rec: dict, key: str) -> float | None:
    metric = rec.get("metrics", {}).get(key)
    if not metric:
        return None
    value = metric.get("value")
    if value is None:
        return None
    if metric.get("unit", "") == "s":
        return value * 1000.0
    return float(value)


def get_lifecycle_start_ns(rec: dict) -> int | None:
    m = rec["metadata"]
    req_end = m.get("request_end_ns")
    http_total_ms = get_metric_ms(rec, "http_req_total")
    if req_end and http_total_ms is not None:
        return req_end - int(http_total_ms * 1e6)
    return m.get("request_start_ns")


def stats(arr: np.ndarray, label: str) -> None:
    print(f"\n  {label} (n={len(arr):,})")
    print(
        f"    mean={np.mean(arr):.3f}ms  median={np.median(arr):.3f}ms  std={np.std(arr):.3f}ms"
    )
    print(f"    min={np.min(arr):.3f}ms  max={np.max(arr):.3f}ms")
    for p in [50, 90, 95, 99, 99.9]:
        print(f"    p{p:<5} {np.percentile(arr, p):.4f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()

    issued_to_start: list[float] = []
    issued_to_recv: list[float] = []
    recv_to_start: list[float] = []
    end_to_issued: list[float] = []
    request_latencies: list[float] = []

    # Group by worker_id, sort by session_num (== credit_num) for sequential pairs
    by_worker: dict[str, list[dict]] = {}

    with args.input.open("rb") as f:
        for line in f:
            rec = orjson.loads(line)
            m = rec["metadata"]
            issued = m.get("credit_issued_ns")
            received = m.get("credit_received_ns")
            start = get_lifecycle_start_ns(rec)
            if not issued or not start:
                continue
            issued_to_start.append((start - issued) / 1e6)
            if received:
                issued_to_recv.append((received - issued) / 1e6)
                recv_to_start.append((start - received) / 1e6)

            req_latency_ms = get_metric_ms(rec, "request_latency")
            if req_latency_ms is not None:
                request_latencies.append(req_latency_ms)

            worker = m.get("worker_id")
            session = m.get("session_num")
            if worker is not None and session is not None:
                by_worker.setdefault(worker, []).append(rec)

    inter_request_gap: list[float] = []
    gap_latency_pairs: list[tuple[float, float]] = []

    for recs in by_worker.values():
        recs.sort(key=lambda r: r["metadata"]["session_num"])
        for i in range(1, len(recs)):
            prev_end = recs[i - 1]["metadata"].get("request_end_ns")
            cur_issued = recs[i]["metadata"].get("credit_issued_ns")
            cur_start = get_lifecycle_start_ns(recs[i])
            if prev_end and cur_issued:
                end_to_issued.append((cur_issued - prev_end) / 1e6)
            if prev_end and cur_start:
                gap_ms = (cur_start - prev_end) / 1e6
                inter_request_gap.append(gap_ms)
                cur_latency = get_metric_ms(recs[i], "request_latency")
                if cur_latency is not None:
                    gap_latency_pairs.append((gap_ms, cur_latency))

    if not issued_to_start:
        raise SystemExit("No records with credit_issued_ns found.")

    print(f"Records: {len(issued_to_start):,}")
    print(f"Workers: {len(by_worker):,}")
    stats(
        np.array(issued_to_start), "credit_issued -> request_start (pipeline overhead)"
    )
    if issued_to_recv:
        stats(
            np.array(issued_to_recv), "credit_issued -> credit_received (ZMQ transit)"
        )
        stats(
            np.array(recv_to_start), "credit_received -> request_start (worker queue)"
        )
    if end_to_issued:
        arr = np.array(end_to_issued)
        positive = arr[arr >= 0]
        print(
            f"\n  request_end -> next credit_issued: {len(arr):,} pairs, {len(positive):,} non-negative ({len(positive) / len(arr) * 100:.1f}%)"
        )
        if len(positive):
            stats(
                positive, "request_end -> next credit_issued (non-pipelined turnaround)"
            )


if __name__ == "__main__":
    raise SystemExit(main())
