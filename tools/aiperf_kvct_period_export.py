# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-assessment-period CSV export from aiperf profile_export.jsonl.

Reads aiperf's per-record JSONL output and re-aggregates it using the same
fixed-window assessment-period methodology that ``trace_replay_tester.py``
uses (default 30 s). Lets you compare aiperf and kv-cache-tester runs on
matched per-period throughput / latency metrics rather than aiperf's whole-
benchmark averages.

Aggregation rules (matching kvct ``summary_trace_replay.csv``):
- A request is counted in the period whose `[start, end)` window contains
  ``request_end_ns`` (kvct's "completion bucket" rule). Period 0 starts at
  the earliest ``request_start_ns`` across the file.
- ``requests_per_second`` = ``requests_completed_new / period_duration_s``.
- ``{input,output}_tokens_per_second`` = sum of `{ISL,OSL}.value` for
  requests completed in the period, divided by period_duration_s.
- ``ttft_*`` aggregates over `time_to_first_token` (ms) of the same set.
- ``avg_decode_tps_per_user`` is the mean of `output_token_throughput_per_user`
  for requests in the period (aiperf already exports it per-record).

Columns dropped from kvct's schema because aiperf doesn't track them:
admission/in-flight, working_set_blocks, otpm/itpm buckets, cache_hit_rate,
SLO goodput counters, fairness, dispatch delay.

Usage:
  python tools/aiperf_kvct_period_export.py \\
      --input  /path/to/profile_export.jsonl \\
      --output /path/to/aiperf_kvct_period_summary.csv \\
      [--assessment-period 30]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PERIOD_COLUMNS = (
    "period_number",
    "start_time",
    "end_time",
    "requests_completed",
    "requests_launched",
    "requests_completed_new",
    "requests_in_progress",
    "requests_per_second",
    "input_tokens_per_second",
    "output_tokens_per_second",
    "ttft_avg",
    "ttft_p50",
    "ttft_p95",
    "ttft_p99",
    "request_latency_avg_ms",
    "avg_decode_tps_per_user",
    "users_completed",
    "requests_per_user_per_min",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to aiperf profile_export.jsonl",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write per-period CSV",
    )
    p.add_argument(
        "--assessment-period",
        type=float,
        default=30.0,
        help="Assessment period length in seconds (default: 30, matches kvct)",
    )
    p.add_argument(
        "--users",
        type=int,
        default=0,
        help="Concurrent-users count for requests_per_user_per_min "
        "(0 = derive from distinct conversation_ids per period)",
    )
    return p.parse_args()


def _metric_value(rec: dict[str, Any], key: str) -> float | None:
    m = rec.get("metrics", {}).get(key)
    if not isinstance(m, dict):
        return None
    v = m.get("value")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        print("ERROR: no records in input", file=sys.stderr)
        return 2

    t0_ns = min(r["metadata"]["request_start_ns"] for r in records)
    period_ns = int(args.assessment_period * 1e9)

    by_period: dict[int, list[dict[str, Any]]] = {}
    launched_by_period: dict[int, int] = {}
    max_period = 0
    for r in records:
        end_ns = r["metadata"]["request_end_ns"]
        start_ns = r["metadata"]["request_start_ns"]
        end_period = (end_ns - t0_ns) // period_ns
        start_period = (start_ns - t0_ns) // period_ns
        by_period.setdefault(end_period, []).append(r)
        launched_by_period[start_period] = launched_by_period.get(start_period, 0) + 1
        max_period = max(max_period, end_period, start_period)

    cumulative_completed = 0
    cumulative_launched = 0
    rows: list[dict[str, Any]] = []
    for p in range(int(max_period) + 1):
        period_recs = by_period.get(p, [])
        launched = launched_by_period.get(p, 0)
        cumulative_completed += len(period_recs)
        cumulative_launched += launched

        ttfts = sorted(
            v
            for r in period_recs
            if (v := _metric_value(r, "time_to_first_token")) is not None
        )
        latencies = [
            v
            for r in period_recs
            if (v := _metric_value(r, "request_latency")) is not None
        ]
        decode_tps = [
            v
            for r in period_recs
            if (v := _metric_value(r, "output_token_throughput_per_user")) is not None
        ]
        isl_sum = sum(
            v
            for r in period_recs
            if (v := _metric_value(r, "input_sequence_length")) is not None
        )
        osl_sum = sum(
            v
            for r in period_recs
            if (v := _metric_value(r, "output_sequence_length")) is not None
        )

        if args.users > 0:
            users_in_period = args.users
        else:
            users_in_period = (
                len({r["metadata"].get("conversation_id") for r in period_recs}) or 1
            )

        rps = len(period_recs) / args.assessment_period
        in_progress = cumulative_launched - cumulative_completed
        rows.append(
            {
                "period_number": p + 1,
                "start_time": round(p * args.assessment_period, 3),
                "end_time": round((p + 1) * args.assessment_period, 3),
                "requests_completed": cumulative_completed,
                "requests_launched": cumulative_launched,
                "requests_completed_new": len(period_recs),
                "requests_in_progress": in_progress,
                "requests_per_second": round(rps, 6),
                "input_tokens_per_second": round(isl_sum / args.assessment_period, 3),
                "output_tokens_per_second": round(osl_sum / args.assessment_period, 3),
                "ttft_avg": round(statistics.mean(ttfts) / 1000, 6) if ttfts else 0.0,
                "ttft_p50": round(_percentile(ttfts, 50) / 1000, 6) if ttfts else 0.0,
                "ttft_p95": round(_percentile(ttfts, 95) / 1000, 6) if ttfts else 0.0,
                "ttft_p99": round(_percentile(ttfts, 99) / 1000, 6) if ttfts else 0.0,
                "request_latency_avg_ms": round(statistics.mean(latencies), 3)
                if latencies
                else 0.0,
                "avg_decode_tps_per_user": round(statistics.mean(decode_tps), 3)
                if decode_tps
                else 0.0,
                "users_completed": 0,
                "requests_per_user_per_min": round(
                    len(period_recs)
                    / max(users_in_period, 1)
                    * (60.0 / args.assessment_period),
                    3,
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PERIOD_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    total_dur = (records[-1]["metadata"]["request_end_ns"] - t0_ns) / 1e9
    print(f"Wrote {len(rows)} period(s) to {args.output}")
    print(f"  total records:   {len(records)}")
    print(
        f"  total duration:  {total_dur:.2f} s "
        f"({len(rows)} x {args.assessment_period:.0f} s windows)"
    )
    for row in rows:
        print(
            f"  period {row['period_number']:>2d}: "
            f"completed_new={row['requests_completed_new']:>3d}  "
            f"req/s={row['requests_per_second']:.3f}  "
            f"in_tps={row['input_tokens_per_second']:>8.0f}  "
            f"out_tps={row['output_tokens_per_second']:>5.0f}  "
            f"ttft_avg={row['ttft_avg'] * 1000:.1f} ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
