#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep line analysis of AIPerf profile_export.jsonl files.

Produces:
  1. Summary statistics (credits, concurrency, TTFT, ITL, E2E, errors)
  2. Sweep line time-series (in-flight, pre/post-TTFT, throughput, errors)
  3. Credit pipeline latency over time
  4. Multi-panel PNG plot

Usage:
    python scripts/analyze_profile_export.py <profile_export.jsonl> [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import orjson

matplotlib.use("Agg")


class EventType(IntEnum):
    CREDIT_ISSUED = 0
    CREDIT_RECEIVED = 1
    REQUEST_START = 2
    FIRST_RESPONSE = 3
    REQUEST_END = 4


METADATA_EVENT_FIELDS: list[tuple[str, EventType]] = [
    ("credit_issued_ns", EventType.CREDIT_ISSUED),
    ("credit_received_ns", EventType.CREDIT_RECEIVED),
]


@dataclass(slots=True)
class SweepState:
    in_flight: int = 0
    pre_ttft: int = 0
    post_ttft: int = 0
    credits_pending: int = 0
    credits_issued: int = 0
    credits_received: int = 0
    requests_started: int = 0
    requests_acked: int = 0
    requests_completed: int = 0
    errors_total: int = 0
    peak_in_flight: int = 0
    peak_in_flight_t: int = 0
    peak_credits_pending: int = 0
    peak_pre_ttft: int = 0


@dataclass(slots=True)
class TimeSeries:
    n: int
    in_flight: np.ndarray = field(init=False)
    pre_ttft: np.ndarray = field(init=False)
    post_ttft: np.ndarray = field(init=False)
    credits_pending: np.ndarray = field(init=False)
    starts: np.ndarray = field(init=False)
    ends: np.ndarray = field(init=False)
    errors: np.ndarray = field(init=False)
    credits_issued: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "in_flight",
            "pre_ttft",
            "post_ttft",
            "credits_pending",
            "starts",
            "ends",
            "errors",
            "credits_issued",
        ):
            setattr(self, name, np.zeros(self.n))


@dataclass(slots=True)
class ExpectedWindow:
    start_s: int
    end_s: int
    expected_start_s: float
    expected_end_s: float


@dataclass(slots=True)
class WaveCompletionWindow:
    start_s: int
    end_s: int
    first_end_s: int | None


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, "rb") as f:
        for line in f:
            records.append(orjson.loads(line))
    return records


def get_metric_ms(rec: dict, key: str) -> float | None:
    metric = rec.get("metrics", {}).get(key)
    if not metric:
        return None
    value = metric.get("value")
    if value is None:
        return None
    unit = metric.get("unit", "")
    if unit == "s":
        return value * 1000.0
    return float(value)


def get_request_lifecycle_start_ns(rec: dict) -> int | None:
    """Reconstruct earliest HTTP lifecycle start from computed HTTP metrics.

    Prefer the true trace-derived lifecycle start when `http_req_total` is present.
    Fall back to the app-level request start timestamp otherwise.
    """
    metadata = rec["metadata"]
    req_end = metadata.get("request_end_ns")
    http_total_ms = get_metric_ms(rec, "http_req_total")
    if req_end and http_total_ms is not None:
        return req_end - int(http_total_ms * 1e6)
    return metadata.get("request_start_ns")


def get_request_send_start_ns(rec: dict) -> int | None:
    """Reconstruct HTTP send start from `http_req_duration` when available."""
    metadata = rec["metadata"]
    req_end = metadata.get("request_end_ns")
    http_duration_ms = get_metric_ms(rec, "http_req_duration")
    if req_end and http_duration_ms is not None:
        return req_end - int(http_duration_ms * 1e6)
    return metadata.get("request_start_ns")


def get_first_response_ns(rec: dict) -> int | None:
    """Return the first response/ACK timestamp.

    Prefer metadata.request_ack_ns, which is derived from recv_start_perf_ns.
    If missing, reconstruct from request_end - http_req_receiving. As a final
    fallback, derive a TTFT-aligned timestamp from request_start_ns.
    """
    metadata = rec["metadata"]
    req_ack = metadata.get("request_ack_ns")
    if req_ack:
        return req_ack

    req_end = metadata.get("request_end_ns")
    http_receiving_ms = get_metric_ms(rec, "http_req_receiving")
    if req_end and http_receiving_ms is not None:
        return req_end - int(http_receiving_ms * 1e6)

    ttft_ms = get_metric_ms(rec, "time_to_first_token")
    req_start = metadata.get("request_start_ns")
    if req_start and ttft_ms is not None:
        return req_start + int(ttft_ms * 1e6)

    return None


def build_events(
    records: list[dict],
) -> list[tuple[int, EventType, int, bool, str, str | None]]:
    events = []
    for i, rec in enumerate(records):
        m = rec["metadata"]
        is_error = rec.get("error") is not None
        worker = m.get("worker_id", "unknown")
        error_type = rec["error"]["type"] if is_error else None
        for ts_key, etype in METADATA_EVENT_FIELDS:
            ts = m.get(ts_key)
            if ts:
                events.append((ts, etype, i, is_error, worker, error_type))
        lifecycle_start = get_request_lifecycle_start_ns(rec)
        if lifecycle_start:
            events.append(
                (
                    lifecycle_start,
                    EventType.REQUEST_START,
                    i,
                    is_error,
                    worker,
                    error_type,
                )
            )

        first_response = get_first_response_ns(rec)
        if first_response:
            events.append(
                (
                    first_response,
                    EventType.FIRST_RESPONSE,
                    i,
                    is_error,
                    worker,
                    error_type,
                )
            )

        req_end = m.get("request_end_ns")
        if req_end:
            events.append(
                (req_end, EventType.REQUEST_END, i, is_error, worker, error_type)
            )
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def sweep(
    events: list[tuple], bucket_ns: int = 1_000_000_000
) -> tuple[SweepState, TimeSeries, int]:
    t0 = events[0][0]
    n_buckets = int((events[-1][0] - t0) / bucket_ns) + 2
    state = SweepState()
    ts = TimeSeries(n_buckets)
    last_bucket = -1
    # Track which requests have received a first response/ACK to correctly
    # decrement pre_ttft vs post_ttft on REQUEST_END.
    acked_requests: set[int] = set()

    for t, etype, idx, is_error, _worker, _etype in events:
        bucket = int((t - t0) / bucket_ns)

        if bucket != last_bucket and last_bucket >= 0:
            for b in range(last_bucket, min(bucket, n_buckets)):
                ts.in_flight[b] = state.in_flight
                ts.pre_ttft[b] = state.pre_ttft
                ts.post_ttft[b] = state.post_ttft
                ts.credits_pending[b] = state.credits_pending
        if last_bucket < 0:
            last_bucket = bucket
        last_bucket = bucket

        if etype == EventType.CREDIT_ISSUED:
            state.credits_issued += 1
            state.credits_pending += 1
            if bucket < n_buckets:
                ts.credits_issued[bucket] += 1
            if state.credits_pending > state.peak_credits_pending:
                state.peak_credits_pending = state.credits_pending

        elif etype == EventType.CREDIT_RECEIVED:
            state.credits_received += 1
            state.credits_pending -= 1

        elif etype == EventType.REQUEST_START:
            state.requests_started += 1
            state.in_flight += 1
            state.pre_ttft += 1
            if bucket < n_buckets:
                ts.starts[bucket] += 1
            if state.in_flight > state.peak_in_flight:
                state.peak_in_flight = state.in_flight
                state.peak_in_flight_t = t
            if state.pre_ttft > state.peak_pre_ttft:
                state.peak_pre_ttft = state.pre_ttft

        elif etype == EventType.FIRST_RESPONSE:
            state.requests_acked += 1
            state.pre_ttft -= 1
            state.post_ttft += 1
            acked_requests.add(idx)

        elif etype == EventType.REQUEST_END:
            state.requests_completed += 1
            state.in_flight -= 1
            if idx in acked_requests:
                state.post_ttft -= 1
                acked_requests.discard(idx)
            else:
                state.pre_ttft -= 1
            if bucket < n_buckets:
                ts.ends[bucket] += 1
            if is_error:
                state.errors_total += 1
                if bucket < n_buckets:
                    ts.errors[bucket] += 1

    for b in range(last_bucket, n_buckets):
        ts.in_flight[b] = state.in_flight
        ts.pre_ttft[b] = state.pre_ttft
        ts.post_ttft[b] = state.post_ttft
        ts.credits_pending[b] = state.credits_pending

    return state, ts, t0


def print_stats(arr: np.ndarray, label: str, unit: str = "ms") -> None:
    print(f"\n  {label} (n={len(arr):,})")
    print(f"    Mean:   {np.mean(arr):.2f} {unit}")
    print(f"    Median: {np.median(arr):.2f} {unit}")
    print(f"    Std:    {np.std(arr):.2f} {unit}")
    print(f"    Min:    {np.min(arr):.2f} {unit}")
    print(f"    Max:    {np.max(arr):.2f} {unit}")
    for p in [1, 5, 25, 50, 75, 90, 95, 99, 99.9]:
        print(f"    P{p:<5}:  {np.percentile(arr, p):.4f} {unit}")


def analyze_metrics(records: list[dict]) -> None:
    errors = 0
    successes = 0
    error_types: dict[str, int] = defaultdict(int)
    credit_send_recv: list[float] = []
    credit_recv_lifecycle_start: list[float] = []
    credit_recv_send_start: list[float] = []
    clock_offsets: list[float] = []
    ttft_vals: list[float] = []
    itl_vals: list[float] = []
    osl_vals: list[float] = []
    isl_vals: list[float] = []
    e2e_success: list[float] = []
    e2e_error: list[float] = []
    http_blocked_vals: list[float] = []
    http_dns_vals: list[float] = []
    http_connecting_vals: list[float] = []
    http_conn_overhead_vals: list[float] = []
    http_sending_vals: list[float] = []
    http_waiting_vals: list[float] = []
    http_receiving_vals: list[float] = []
    http_duration_vals: list[float] = []
    http_total_vals: list[float] = []
    http_reused_vals: list[float] = []
    per_worker_send_recv: dict[str, list[float]] = defaultdict(list)

    for rec in records:
        m = rec["metadata"]
        metrics = rec.get("metrics", {})
        error = rec.get("error")
        worker = m.get("worker_id", "unknown")
        issued = m.get("credit_issued_ns")
        received = m.get("credit_received_ns")
        req_end = m.get("request_end_ns")
        lifecycle_start = get_request_lifecycle_start_ns(rec)
        send_start = get_request_send_start_ns(rec)
        offset = m.get("clock_offset_ns")

        if error:
            errors += 1
            error_types[error.get("type", "unknown")] += 1
        else:
            successes += 1

        if issued and received and lifecycle_start:
            d1 = (received - issued) / 1e6
            d2 = (lifecycle_start - received) / 1e6
            credit_send_recv.append(d1)
            credit_recv_lifecycle_start.append(d2)
            per_worker_send_recv[worker].append(d1)

        if issued and received and send_start:
            credit_recv_send_start.append((send_start - received) / 1e6)

        if offset is not None:
            clock_offsets.append(offset / 1e6)

        if lifecycle_start and req_end:
            e2e = (req_end - lifecycle_start) / 1e6
            if error:
                e2e_error.append(e2e)
            else:
                e2e_success.append(e2e)

        for key, target in [
            ("http_req_blocked", http_blocked_vals),
            ("http_req_dns_lookup", http_dns_vals),
            ("http_req_connecting", http_connecting_vals),
            ("http_req_connection_overhead", http_conn_overhead_vals),
            ("http_req_sending", http_sending_vals),
            ("http_req_waiting", http_waiting_vals),
            ("http_req_receiving", http_receiving_vals),
            ("http_req_duration", http_duration_vals),
            ("http_req_total", http_total_vals),
            ("http_req_connection_reused", http_reused_vals),
        ]:
            v = get_metric_ms(rec, key)
            if v is not None:
                target.append(v)

        if not error:
            for key, target in [
                ("time_to_first_token", ttft_vals),
                ("inter_token_latency", itl_vals),
                ("output_sequence_length", osl_vals),
                ("input_sequence_length", isl_vals),
            ]:
                if key in metrics:
                    v = metrics[key]["value"]
                    u = metrics[key].get("unit", "")
                    target.append(v * 1000 if u == "s" else v)

    total = errors + successes
    print(f"\n{'=' * 70}")
    print("  Record Summary")
    print(f"{'=' * 70}")
    print(f"  Total:     {total:,}")
    print(f"  Errors:    {errors:,} ({errors / total * 100:.1f}%)")
    print(f"  Success:   {successes:,} ({successes / total * 100:.1f}%)")
    print("\n  Error types:")
    for et, cnt in sorted(error_types.items(), key=lambda x: -x[1]):
        print(f"    {et}: {cnt:,}")

    if credit_send_recv:
        s2r = np.array(credit_send_recv)
        r2l = np.array(credit_recv_lifecycle_start)
        print_stats(s2r, "Credit Issued -> Credit Received (ZMQ transit)")
        print_stats(r2l, "Credit Received -> HTTP Lifecycle Start")
        total_pipeline = s2r + r2l
        print_stats(
            total_pipeline, "Total Credit Pipeline: Issued -> HTTP Lifecycle Start"
        )
        if credit_recv_send_start:
            print_stats(
                np.array(credit_recv_send_start), "Credit Received -> HTTP Send Start"
            )
        neg = np.sum(s2r < 0)
        if neg > 0:
            print(f"\n  WARNING: {neg:,} negative send->recv deltas (clock skew)")

    if clock_offsets:
        print_stats(np.array(clock_offsets), "Clock Offset", "ms")

    if http_total_vals:
        print(f"\n{'=' * 70}")
        print("  HTTP Lifecycle Metrics")
        print(f"{'=' * 70}")
        print_stats(np.array(http_total_vals), "HTTP Total")
        if http_conn_overhead_vals:
            print_stats(np.array(http_conn_overhead_vals), "HTTP Connection Overhead")
        if http_blocked_vals:
            print_stats(np.array(http_blocked_vals), "HTTP Blocked")
        if http_dns_vals:
            print_stats(np.array(http_dns_vals), "HTTP DNS Lookup")
        if http_connecting_vals:
            print_stats(np.array(http_connecting_vals), "HTTP Connecting")
        if http_sending_vals:
            print_stats(np.array(http_sending_vals), "HTTP Sending")
        if http_waiting_vals:
            print_stats(np.array(http_waiting_vals), "HTTP Waiting (TTFB)")
        if http_receiving_vals:
            print_stats(np.array(http_receiving_vals), "HTTP Receiving")
        if http_duration_vals:
            print_stats(np.array(http_duration_vals), "HTTP Duration")
        if http_reused_vals:
            reused = np.array(http_reused_vals)
            print(f"\n  Connection reused: {np.mean(reused) * 100:.2f}%")

    if ttft_vals:
        print_stats(np.array(ttft_vals), "Time to First Token (TTFT)")
    if itl_vals:
        print_stats(np.array(itl_vals), "Inter-Token Latency (ITL)")
    if osl_vals:
        print_stats(np.array(osl_vals), "Output Sequence Length", "tokens")
    if isl_vals:
        print_stats(np.array(isl_vals), "Input Sequence Length", "tokens")
    if e2e_success:
        print_stats(
            np.array(e2e_success), "E2E Latency (success, lifecycle start -> end)"
        )
    if e2e_error:
        print_stats(np.array(e2e_error), "E2E Latency (error, lifecycle start -> end)")

    # Mock vs real indicators
    print(f"\n{'=' * 70}")
    print("  Mock vs Real Indicators")
    print(f"{'=' * 70}")
    if osl_vals:
        unique_osl = len(set(int(v) for v in osl_vals))
        print(
            f"  Unique OSL values: {unique_osl}  {'-> REAL' if unique_osl > 5 else '-> MOCK'}"
        )
    if ttft_vals:
        cv = np.std(ttft_vals) / np.mean(ttft_vals)
        print(f"  TTFT CV: {cv:.4f}  {'-> REAL' if cv > 0.1 else '-> MOCK'}")

    # Per-worker credit latency (top/bottom 10)
    print(f"\n{'=' * 70}")
    print("  Per-Worker Credit Latency (Send->Recv, top 10 slowest)")
    print(f"{'=' * 70}")
    worker_means = [
        (w, np.mean(v), np.percentile(v, 99), np.max(v), len(v))
        for w, v in per_worker_send_recv.items()
    ]
    worker_means.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Worker':<25} {'Count':>8} {'Mean':>10} {'P99':>10} {'Max':>10}")
    for w, mean, p99, mx, cnt in worker_means[:10]:
        print(f"  {w:<25} {cnt:>8,} {mean:>9.4f}ms {p99:>9.4f}ms {mx:>9.4f}ms")


def print_sweep_summary(
    state: SweepState,
    ts: TimeSeries,
    t0: int,
    events: list[tuple],
    records: list[dict],
) -> None:
    duration_s = (events[-1][0] - t0) / 1e9

    print(f"\n{'=' * 70}")
    print("  Sweep Line Summary")
    print(f"{'=' * 70}")
    print(f"  Duration:             {duration_s:.1f}s ({duration_s / 60:.1f} min)")
    print(f"  Credits issued:       {state.credits_issued:,}")
    print(f"  Credits received:     {state.credits_received:,}")
    print(f"  Credits still pending:{state.credits_pending:,}")
    print(f"  Requests started:     {state.requests_started:,}")
    print(f"  Requests w/ 1st resp: {state.requests_acked:,}")
    print(f"  Requests completed:   {state.requests_completed:,}")
    print(f"  Requests in-flight:   {state.in_flight:,}")
    print(f"  Errors:               {state.errors_total:,}")
    print(
        f"\n  Peak in-flight:       {state.peak_in_flight:,}  at t={((state.peak_in_flight_t - t0) / 1e9):.1f}s"
    )
    print(f"  Peak credits pending: {state.peak_credits_pending:,}")
    print(f"  Peak pre-1st-response:{state.peak_pre_ttft:,}")

    # Phase transitions
    print(f"\n{'=' * 70}")
    print("  Phase Transitions")
    print(f"{'=' * 70}")

    # Error storms (>100 err/s)
    storm_buckets = np.where(ts.errors > 100)[0]
    if len(storm_buckets) > 0:
        storms = []
        s_start = s_end = storm_buckets[0]
        for b in storm_buckets[1:]:
            if b == s_end + 1:
                s_end = b
            else:
                storms.append((s_start, s_end))
                s_start = s_end = b
        storms.append((s_start, s_end))
        for ss, se in storms:
            total_err = int(np.sum(ts.errors[ss : se + 1]))
            peak_err = int(np.max(ts.errors[ss : se + 1]))
            print(
                f"  Error storm: t={ss}s-{se}s ({se - ss + 1}s), {total_err:,} errors, peak {peak_err:,}/s"
            )

    # Last credit issued
    send_complete = 0
    for b in range(n_buckets - 1, -1, -1):
        if ts.credits_issued[b] > 0:
            send_complete = b
            break
    print(f"  Last credit issued:  t={send_complete}s")

    # Concurrency thresholds
    print(f"\n{'=' * 70}")
    print("  Concurrency Curve")
    print(f"{'=' * 70}")
    for threshold in [1000, 10000, 50000, 90000, 100000]:
        first = last = None
        for b in range(n_buckets):
            if ts.in_flight[b] >= threshold:
                if first is None:
                    first = b
                last = b
        if first is not None:
            print(f"  >= {threshold:>7,}:  t={first}s to t={last}s  ({last - first}s)")

    # Time series table (active portion only)
    active_end = n_buckets
    for b in range(n_buckets - 1, -1, -1):
        if ts.in_flight[b] > 0 or ts.ends[b] > 0:
            active_end = b + 10
            break
    active_end = min(active_end, n_buckets)

    print(f"\n{'=' * 70}")
    print("  Time Series (10s snapshots, active portion)")
    print(f"{'=' * 70}")
    print(
        f"  {'t(s)':>6} {'InFlight':>10} {'PreResp':>9} {'PostResp':>10} "
        f"{'Starts/s':>10} {'Ends/s':>10} {'Errs/s':>8}"
    )
    interval = 10
    for b in range(0, active_end, interval):
        eb = min(b + interval, n_buckets)
        print(
            f"  {b:>6} {np.mean(ts.in_flight[b:eb]):>10.0f} "
            f"{np.mean(ts.pre_ttft[b:eb]):>9.0f} {np.mean(ts.post_ttft[b:eb]):>10.0f} "
            f"{np.sum(ts.starts[b:eb]) / interval:>10.1f} "
            f"{np.sum(ts.ends[b:eb]) / interval:>10.1f} "
            f"{np.sum(ts.errors[b:eb]) / interval:>8.1f}"
        )

    # Error rate by 30s window
    print(f"\n{'=' * 70}")
    print("  Error Rate by 30s Window")
    print(f"{'=' * 70}")
    print(f"  {'Window':>12} {'Requests':>10} {'Errors':>8} {'ErrRate':>8}")
    window = 30
    for b in range(0, active_end, window):
        eb = min(b + window, n_buckets)
        reqs = int(np.sum(ts.ends[b:eb]))
        errs = int(np.sum(ts.errors[b:eb]))
        if reqs > 0:
            print(
                f"  {b:>4}s-{b + window:>4}s {reqs:>10,} {errs:>8,} "
                f"{errs / reqs * 100:>7.1f}%"
            )

    # Credit pipeline latency over time
    print(f"\n{'=' * 70}")
    print("  Credit Pipeline Latency Over Time (30s windows)")
    print(f"{'=' * 70}")
    cr_lat_by_win: dict[int, list[float]] = defaultdict(list)
    cr_r2l_by_win: dict[int, list[float]] = defaultdict(list)
    cr_r2send_by_win: dict[int, list[float]] = defaultdict(list)
    for rec in records:
        m = rec["metadata"]
        issued = m.get("credit_issued_ns")
        received = m.get("credit_received_ns")
        lifecycle_start = get_request_lifecycle_start_ns(rec)
        send_start = get_request_send_start_ns(rec)
        if issued and received and lifecycle_start:
            w = int((issued - t0) / 1_000_000_000 / window)
            cr_lat_by_win[w].append((received - issued) / 1e6)
            cr_r2l_by_win[w].append((lifecycle_start - received) / 1e6)
            if send_start:
                cr_r2send_by_win[w].append((send_start - received) / 1e6)

    print(
        f"  {'Window':>12} {'Send->Recv':>12} {'Recv->Life':>13} "
        f"{'Recv->Send':>13} "
        f"{'Total':>10} {'p99 S->R':>10} {'Count':>8}"
    )
    for w in sorted(cr_lat_by_win.keys()):
        s2r = np.array(cr_lat_by_win[w])
        r2l = np.array(cr_r2l_by_win[w])
        r2send = np.array(cr_r2send_by_win[w]) if cr_r2send_by_win[w] else np.array([])
        total = s2r + r2l
        t_start = w * window
        print(
            f"  {t_start:>4}s-{t_start + window:>4}s "
            f"{np.median(s2r):>11.3f}ms {np.median(r2l):>12.3f}ms "
            f"{(np.median(r2send) if len(r2send) else float('nan')):>12.3f}ms "
            f"{np.median(total):>9.3f}ms {np.percentile(s2r, 99):>9.3f}ms "
            f"{len(s2r):>8,}"
        )


def detect_start_blocks(
    ts: TimeSeries, min_rate: float = 1.0, gap_s: int = 10
) -> list[tuple[int, int]]:
    """Group contiguous request-start activity into coarse load blocks.

    A block starts when request-start activity exceeds ``min_rate`` req/s.
    Small gaps are bridged to avoid over-fragmenting bursts.
    """
    active = np.where(ts.starts >= min_rate)[0]
    if len(active) == 0:
        return []

    blocks: list[tuple[int, int]] = []
    start = prev = int(active[0])
    for idx in active[1:]:
        idx = int(idx)
        if idx <= prev + gap_s:
            prev = idx
            continue
        blocks.append((start, prev))
        start = prev = idx
    blocks.append((start, prev))
    return blocks


def build_expected_windows(
    ts: TimeSeries,
    expected_latency_s: float | None,
) -> list[ExpectedWindow]:
    if expected_latency_s is None:
        return []
    windows: list[ExpectedWindow] = []
    for start_s, end_s in detect_start_blocks(ts):
        windows.append(
            ExpectedWindow(
                start_s=start_s,
                end_s=end_s,
                expected_start_s=start_s + expected_latency_s,
                expected_end_s=end_s + expected_latency_s,
            )
        )
    return windows


def build_wave_completion_windows(
    ts: TimeSeries,
    records: list[dict],
    t0: int,
) -> list[WaveCompletionWindow]:
    windows: list[WaveCompletionWindow] = []
    blocks = detect_start_blocks(ts)
    for start_s, end_s in blocks:
        first_end_s: int | None = None
        matching_end_buckets: list[int] = []
        for rec in records:
            lifecycle_start = get_request_lifecycle_start_ns(rec)
            req_end = rec["metadata"].get("request_end_ns")
            if not lifecycle_start or not req_end:
                continue
            start_bucket = int((lifecycle_start - t0) / 1_000_000_000)
            if start_s <= start_bucket <= end_s:
                end_bucket = int((req_end - t0) / 1_000_000_000)
                matching_end_buckets.append(end_bucket)
        if matching_end_buckets:
            first_end_s = min(matching_end_buckets)
        windows.append(
            WaveCompletionWindow(
                start_s=start_s,
                end_s=end_s,
                first_end_s=first_end_s,
            )
        )
    return windows


def plot(
    ts: TimeSeries,
    state: SweepState,
    t0: int,
    events: list[tuple],
    output_path: Path,
    expected_windows: list[ExpectedWindow] | None = None,
    wave_completion_windows: list[WaveCompletionWindow] | None = None,
) -> None:
    n_buckets = ts.n
    t_sec = np.arange(n_buckets)
    expected_windows = expected_windows or []
    wave_completion_windows = wave_completion_windows or []

    # Find active end for zoom
    active_end = 300
    for b in range(n_buckets - 1, -1, -1):
        if ts.starts[b] > 0:
            active_end = b + 180
            break
    active_end = min(active_end, n_buckets)

    send_complete = 0
    for b in range(n_buckets - 1, -1, -1):
        if ts.credits_issued[b] > 0:
            send_complete = b
            break

    # Detect stall
    stall_start = None
    for b in range(send_complete, n_buckets - 10):
        if ts.in_flight[b] > 0 and np.sum(ts.ends[b : b + 10]) == 0:
            stall_start = b
            break

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(20, 16),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1]},
    )
    title = output_path.stem.replace("sweepline_", "").replace("_", "-")
    fig.suptitle(
        f"Sweep Line Analysis: {title}", fontsize=16, fontweight="bold", y=0.98
    )

    # Panel 1: Full timeline
    ax1 = axes[0]
    ax1.fill_between(
        t_sec, ts.in_flight, alpha=0.3, color="#2196F3", label="In-Flight (total)"
    )
    ax1.plot(t_sec, ts.in_flight, color="#1565C0", linewidth=0.8)
    ax1.fill_between(
        t_sec, ts.pre_ttft, alpha=0.3, color="#FF9800", label="Pre-1st-Response"
    )
    ax1.plot(t_sec, ts.pre_ttft, color="#E65100", linewidth=0.8)
    ax1.fill_between(
        t_sec, ts.post_ttft, alpha=0.3, color="#4CAF50", label="Post-1st-Response"
    )
    ax1.plot(t_sec, ts.post_ttft, color="#2E7D32", linewidth=0.8)
    ax1.axhline(
        y=state.peak_in_flight,
        color="red",
        linestyle="--",
        alpha=0.5,
        linewidth=0.8,
        label=f"Peak ({state.peak_in_flight:,})",
    )
    ax1.axvline(
        x=send_complete,
        color="green",
        linestyle="--",
        alpha=0.5,
        linewidth=0.8,
        label=f"Last credit (t={send_complete}s)",
    )
    if stall_start:
        ax1.axvline(
            x=stall_start,
            color="purple",
            linestyle="--",
            alpha=0.5,
            linewidth=0.8,
            label=f"Stall (t={stall_start}s)",
        )
    for i, win in enumerate(expected_windows):
        ax1.axvline(
            x=win.expected_start_s,
            color="#8E24AA",
            linestyle=":",
            alpha=0.55,
            linewidth=1.0,
            label="Expected mock completion window" if i == 0 else None,
        )
        ax1.axvline(
            x=win.expected_end_s,
            color="#8E24AA",
            linestyle=":",
            alpha=0.55,
            linewidth=1.0,
        )
    for i, win in enumerate(wave_completion_windows):
        if win.first_end_s is None:
            continue
        ax1.axvline(
            x=win.first_end_s,
            color="#D81B60",
            linestyle="-.",
            alpha=0.7,
            linewidth=1.0,
            label="First observed completion after wave start" if i == 0 else None,
        )
    ax1.set_ylabel("Concurrent Requests", fontsize=11)
    ax1.set_xlabel("Time (seconds)", fontsize=10)
    ax1.set_ylim(-max(1000, state.peak_in_flight * 0.01), state.peak_in_flight * 1.15)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax1.set_title("Concurrency Over Time (Full Run)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Zoomed active window
    ax2 = axes[1]
    z = slice(0, active_end)
    ax2.fill_between(t_sec[z], ts.in_flight[z], alpha=0.15, color="#2196F3")
    ax2.plot(
        t_sec[z],
        ts.in_flight[z],
        color="#1565C0",
        linewidth=1.2,
        label="In-Flight (total)",
    )
    ax2.fill_between(t_sec[z], ts.pre_ttft[z], alpha=0.35, color="#FF9800")
    ax2.plot(
        t_sec[z], ts.pre_ttft[z], color="#E65100", linewidth=1, label="Pre-1st-Response"
    )
    ax2.fill_between(t_sec[z], ts.post_ttft[z], alpha=0.35, color="#4CAF50")
    ax2.plot(
        t_sec[z],
        ts.post_ttft[z],
        color="#2E7D32",
        linewidth=1,
        label="Post-1st-Response",
    )
    ax2.axvline(
        x=send_complete, color="green", linestyle="--", alpha=0.5, linewidth=0.8
    )
    for win in expected_windows:
        ax2.axvline(
            x=win.expected_start_s,
            color="#8E24AA",
            linestyle=":",
            alpha=0.55,
            linewidth=1.0,
        )
        ax2.axvline(
            x=win.expected_end_s,
            color="#8E24AA",
            linestyle=":",
            alpha=0.55,
            linewidth=1.0,
        )
    for win in wave_completion_windows:
        if win.first_end_s is not None:
            ax2.axvline(
                x=win.first_end_s,
                color="#D81B60",
                linestyle="-.",
                alpha=0.7,
                linewidth=1.0,
            )
    ax2.set_ylabel("Concurrent Requests", fontsize=11)
    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.legend(loc="right", fontsize=8, framealpha=0.9)
    ax2.set_title(
        f"Active Window Detail (t=0-{active_end}s): Pre/Post First Response",
        fontsize=12,
    )
    ax2.grid(True, alpha=0.3)

    # Panel 3: Throughput
    ax3 = axes[2]
    window = 5
    starts_s = np.convolve(ts.starts[z], np.ones(window) / window, mode="same")
    ends_s = np.convolve(ts.ends[z], np.ones(window) / window, mode="same")
    errors_s = np.convolve(ts.errors[z], np.ones(window) / window, mode="same")
    ax3.plot(
        t_sec[z], starts_s, color="#2196F3", linewidth=1.2, label="Requests Started/s"
    )
    ax3.plot(
        t_sec[z], ends_s, color="#4CAF50", linewidth=1.2, label="Requests Completed/s"
    )
    ax3.fill_between(t_sec[z], errors_s, alpha=0.4, color="#F44336", label="Errors/s")
    ax3.plot(t_sec[z], errors_s, color="#B71C1C", linewidth=0.8)
    ax3.axvline(
        x=send_complete, color="green", linestyle="--", alpha=0.5, linewidth=0.8
    )
    for win in expected_windows:
        ax3.axvline(
            x=win.expected_start_s,
            color="#8E24AA",
            linestyle=":",
            alpha=0.55,
            linewidth=1.0,
        )
        ax3.axvline(
            x=win.expected_end_s,
            color="#8E24AA",
            linestyle=":",
            alpha=0.55,
            linewidth=1.0,
        )
    for win in wave_completion_windows:
        if win.first_end_s is not None:
            ax3.axvline(
                x=win.first_end_s,
                color="#D81B60",
                linestyle="-.",
                alpha=0.7,
                linewidth=1.0,
            )
    ax3.set_ylabel("Rate (req/s)", fontsize=11)
    ax3.set_xlabel("Time (seconds)", fontsize=10)
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax3.set_title(
        f"Throughput & Error Rate (5s rolling avg, t=0-{active_end}s)", fontsize=12
    )
    ax3.grid(True, alpha=0.3)

    # Panel 4: Credit pipeline
    ax4 = axes[3]
    ax4.plot(
        t_sec[z],
        ts.credits_pending[z],
        color="#9C27B0",
        linewidth=1.2,
        label="Credits Pending (issued - received)",
    )
    for win in expected_windows:
        ax4.axvline(
            x=win.expected_start_s,
            color="#8E24AA",
            linestyle=":",
            alpha=0.55,
            linewidth=1.0,
        )
        ax4.axvline(
            x=win.expected_end_s,
            color="#8E24AA",
            linestyle=":",
            alpha=0.55,
            linewidth=1.0,
        )
    for win in wave_completion_windows:
        if win.first_end_s is not None:
            ax4.axvline(
                x=win.first_end_s,
                color="#D81B60",
                linestyle="-.",
                alpha=0.7,
                linewidth=1.0,
            )
    ax4.set_ylabel("Credits Pending", fontsize=11)
    ax4.set_xlabel("Time (seconds)", fontsize=10)
    ax4.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax4.set_title(f"Credit Pipeline Backpressure (t=0-{active_end}s)", fontsize=12)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nPlot saved to {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to profile_export.jsonl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plot (default: same as input)",
    )
    parser.add_argument(
        "--expected-latency-s",
        type=float,
        default=None,
        help="Draw expected completion windows by shifting request-start blocks by this many seconds.",
    )
    parser.add_argument(
        "--mock-ttft-ms",
        type=float,
        default=None,
        help="Mock TTFT in ms. Combined with --mock-itl-ms and --mock-osl to compute expected latency.",
    )
    parser.add_argument(
        "--mock-itl-ms", type=float, default=None, help="Mock ITL in ms."
    )
    parser.add_argument(
        "--mock-osl",
        type=float,
        default=None,
        help="Expected output sequence length in tokens for mock latency model.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem.replace(" ", "_").replace("(", "").replace(")", "")
    plot_path = output_dir / f"sweepline_{stem}.png"

    print(f"Loading {args.input} ...")
    records = load_records(args.input)
    print(f"Loaded {len(records):,} records")

    print("\nBuilding events ...")
    events = build_events(records)
    print(f"Total events: {len(events):,}")

    print("\n--- Metrics Analysis ---")
    analyze_metrics(records)

    print("\n--- Sweep Line Analysis ---")
    state, ts, t0 = sweep(events)
    print_sweep_summary(state, ts, t0, events, records)

    expected_latency_s = args.expected_latency_s
    if expected_latency_s is None and None not in (
        args.mock_ttft_ms,
        args.mock_itl_ms,
        args.mock_osl,
    ):
        expected_latency_s = (
            args.mock_ttft_ms + args.mock_itl_ms * args.mock_osl
        ) / 1000.0
        print(
            f"\nExpected mock latency: {expected_latency_s:.3f}s "
            f"(ttft={args.mock_ttft_ms}ms + itl={args.mock_itl_ms}ms * osl={args.mock_osl})"
        )
    expected_windows = build_expected_windows(ts, expected_latency_s)
    if expected_windows:
        print(
            "\nExpected completion windows (from request-start blocks + mock latency):"
        )
        for win in expected_windows:
            print(
                f"  block {win.start_s}s-{win.end_s}s -> expected completion {win.expected_start_s:.1f}s-{win.expected_end_s:.1f}s"
            )
    wave_completion_windows = build_wave_completion_windows(ts, records, t0)
    if wave_completion_windows:
        print("\nFirst observed completion after each request-start wave:")
        for win in wave_completion_windows:
            if win.first_end_s is None:
                print(f"  block {win.start_s}s-{win.end_s}s -> no completion observed")
            else:
                print(
                    f"  block {win.start_s}s-{win.end_s}s -> first completion at {win.first_end_s}s "
                    f"(delta {win.first_end_s - win.start_s}s)"
                )

    print("\n--- Generating Plot ---")
    plot(ts, state, t0, events, plot_path, expected_windows, wave_completion_windows)


if __name__ == "__main__":
    main()
