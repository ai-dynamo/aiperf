#!/usr/bin/env python3
"""Post-process sweep_isl_osl_mem CR snapshots into a clean analysis CSV.

Reads:
  - dev/results/sweep-isl-osl-mem.csv (the live sampler output — has memory peaks
    but the rps/output_tokens_per_second columns came back None because the
    operator image's status.summary uses different keys than the script expected).
  - dev/results/cr-snapshots/sweep-iom-*.json (full CR JSON per cell, captured
    while the TTL was still alive).

Writes:
  - dev/results/sweep-isl-osl-mem-analysis.csv with:
      * concurrency, isl, osl, requests, completed
      * rps (derived from completed / benchmark_duration.avg)
      * output_token_throughput (sum of OSL × completed / benchmark_duration)
      * error_rate
      * controller / worker / operator memory peaks (mock excluded from total)
      * total_no_mock_mib

This is a pure post-processor — does not touch the cluster.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
CSV_IN = REPO / "dev/results/sweep-isl-osl-mem.csv"
SNAP_DIR = REPO / "dev/results/cr-snapshots"
CSV_OUT = REPO / "dev/results/sweep-isl-osl-mem-analysis.csv"

NAME_RE = re.compile(r"^sweep-iom-c(\d+)-i(\d+)-o(\d+)$")

OUT_FIELDS = [
    "concurrency",
    "isl",
    "osl",
    "requests",
    "completed",
    "errors",
    "error_rate",
    "duration_s",
    "benchmark_duration_s",
    "rps",
    "output_token_throughput",
    # Memory (mock excluded from total)
    "ctrl_records_mib",
    "ctrl_total_mib",
    "worker_pod_count",
    "workers_sum_mib",
    "workers_avg_per_pod_mib",
    "workers_max_per_pod_mib",
    "operator_mib",
    "total_no_mock_mib",
    # Kept for reference
    "mock_mib",
]


def load_live_csv() -> dict[tuple[int, int, int], dict]:
    rows = {}
    with CSV_IN.open() as f:
        for r in csv.DictReader(f):
            key = (int(r["concurrency"]), int(r["isl"]), int(r["osl"]))
            rows[key] = r
    return rows


def parse_cr(path: Path) -> dict:
    d = json.loads(path.read_text())
    name = d["metadata"]["name"]
    m = NAME_RE.match(name)
    if not m:
        return {}
    conc, isl, osl = (int(g) for g in m.groups())
    summary = (d.get("status") or {}).get("summary") or {}
    bench_dur = (summary.get("benchmark_duration") or {}).get("avg") or 0.0
    err_count = (summary.get("error_request_count") or {}).get("avg") or 0
    err_rate = summary.get("error_rate") or 0.0
    osl_avg = (summary.get("output_sequence_length") or {}).get("avg") or 0
    phases = (d.get("status") or {}).get("phases") or {}
    completed = (phases.get("profiling") or {}).get("requestsCompleted") or 0
    return {
        "name": name,
        "key": (conc, isl, osl),
        "benchmark_duration_s": round(float(bench_dur), 2),
        "errors": int(err_count or 0),
        "error_rate": round(float(err_rate or 0), 4),
        "completed": int(completed),
        "osl_avg": float(osl_avg or 0),
    }


def main() -> int:
    live = load_live_csv()
    cr_rows: dict[tuple[int, int, int], dict] = {}
    for f in sorted(SNAP_DIR.glob("sweep-iom-*.json")):
        info = parse_cr(f)
        if info:
            cr_rows[info["key"]] = info

    out = []
    for key, lr in live.items():
        cr = cr_rows.get(key, {})
        completed = cr.get("completed") or int(lr.get("completed") or 0)
        bench_dur = cr.get("benchmark_duration_s") or 0.0
        rps = round(completed / bench_dur, 2) if bench_dur > 0 else None
        osl_avg = cr.get("osl_avg") or float(key[2])
        otps = round(completed * osl_avg / bench_dur, 2) if bench_dur > 0 else None

        ctrl_total = int(lr["ctrl_peak_total_mib"])
        wkr_sum = int(lr["workers_peak_total_mib"])
        op = int(lr["operator_peak_mib"])
        mock = int(lr["mock_peak_mib"])

        out.append(
            {
                "concurrency": key[0],
                "isl": key[1],
                "osl": key[2],
                "requests": lr["requests"],
                "completed": completed,
                "errors": cr.get("errors", ""),
                "error_rate": cr.get("error_rate", ""),
                "duration_s": lr["duration_s"],
                "benchmark_duration_s": bench_dur,
                "rps": rps,
                "output_token_throughput": otps,
                "ctrl_records_mib": int(lr["ctrl_peak_records_mib"]),
                "ctrl_total_mib": ctrl_total,
                "worker_pod_count": int(lr["worker_pod_count"]),
                "workers_sum_mib": wkr_sum,
                "workers_avg_per_pod_mib": int(lr["workers_peak_avg_per_pod_mib"]),
                "workers_max_per_pod_mib": int(lr["workers_peak_max_per_pod_mib"]),
                "operator_mib": op,
                "total_no_mock_mib": ctrl_total + wkr_sum + op,
                "mock_mib": mock,
            }
        )

    out.sort(key=lambda r: (r["concurrency"], r["isl"], r["osl"]))
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {CSV_OUT.relative_to(REPO)}")
    print()
    hdr = (
        f"{'conc':>5} {'isl':>5} {'osl':>5}  {'rps':>7} {'tok/s':>7}  "
        f"{'rm':>4} {'ctrl':>5} {'wkr-sum':>7} {'mx-pod':>6} {'op':>4}  "
        f"{'tot':>5}  {'err%':>4}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in out:
        print(
            f"{r['concurrency']:>5} {r['isl']:>5} {r['osl']:>5}  "
            f"{r['rps']:>7.1f} {r['output_token_throughput'] or 0:>7.0f}  "
            f"{r['ctrl_records_mib']:>4} {r['ctrl_total_mib']:>5} "
            f"{r['workers_sum_mib']:>7} {r['workers_max_per_pod_mib']:>6} "
            f"{r['operator_mib']:>4}  "
            f"{r['total_no_mock_mib']:>5}  "
            f"{(r['error_rate'] or 0) * 100:>4.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
