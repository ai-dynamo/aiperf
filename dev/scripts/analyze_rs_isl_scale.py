#!/usr/bin/env python3
"""Post-process sweep_rs_isl_scale CR snapshots into a clean analysis CSV.

Same shape as analyze_isl_osl_mem.py — pulls real RPS / output_token_throughput
from each CR's status.summary.benchmark_duration (since the live sampler's
status.summary.throughput_rps key returned None for several cells).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
CSV_IN = REPO / "dev/results/sweep-rs-isl-scale.csv"
SNAP_DIR = REPO / "dev/results/cr-snapshots-rs"
CSV_OUT = REPO / "dev/results/sweep-rs-isl-scale-analysis.csv"

NAME_RE = re.compile(r"^sweep-rs-c(\d+)-i(\d+)-o(\d+)$")

OUT_FIELDS = [
    "concurrency", "isl", "osl",
    "entries", "requests", "completed",
    "records_success", "records_error",
    "errors", "error_rate",
    "duration_s", "benchmark_duration_s",
    "rps", "output_token_throughput",
    "ctrl_records_mib", "ctrl_dataset_mib", "ctrl_total_mib",
    "worker_pod_count", "workers_sum_mib", "workers_avg_per_pod_mib",
    "workers_max_per_pod_mib",
    "operator_mib",
    "total_no_mock_mib",
    "mock_mib",
]


def load_live_csv() -> dict[tuple[int, int, int], dict]:
    rows = {}
    with CSV_IN.open() as f:
        for r in csv.DictReader(f):
            if r.get("phase") != "Completed":
                continue
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
    profiling = phases.get("profiling") or {}
    completed = profiling.get("requestsCompleted") or 0
    # Fallback: at very large ISL, records-side processing fails (ISL >= 512K)
    # so summary.benchmark_duration is unset. Use phases.profiling.elapsedTimeSeconds
    # + requestsPerSecond as the canonical live measurement.
    if not bench_dur:
        bench_dur = profiling.get("elapsedTimeSeconds") or 0.0
    live_rps = profiling.get("requestsPerSecond") or 0.0
    records_success = profiling.get("recordsSuccess") or 0
    records_error = profiling.get("recordsError") or 0
    return {
        "name": name,
        "key": (conc, isl, osl),
        "benchmark_duration_s": round(float(bench_dur), 2),
        "errors": int(err_count or 0),
        "error_rate": round(float(err_rate or 0), 4),
        "completed": int(completed),
        "osl_avg": float(osl_avg or 0),
        "live_rps": float(live_rps),
        "records_success": int(records_success),
        "records_error": int(records_error),
    }


def main() -> int:
    live = load_live_csv()
    cr_rows: dict[tuple[int, int, int], dict] = {}
    for f in sorted(SNAP_DIR.glob("sweep-rs-*.json")):
        info = parse_cr(f)
        if info:
            cr_rows[info["key"]] = info

    out = []
    for key, lr in live.items():
        cr = cr_rows.get(key, {})
        completed = cr.get("completed") or int(lr.get("completed") or 0)
        bench_dur = cr.get("benchmark_duration_s") or 0.0
        # Prefer the live requests-per-second value when records-side processing
        # didn't run to completion (ISL >= 512K cells).
        rps = cr.get("live_rps") or 0.0
        if not rps and bench_dur > 0:
            rps = round(completed / bench_dur, 2)
        osl_avg = cr.get("osl_avg") or float(key[2])
        otps = round(rps * osl_avg, 2) if rps else None

        ctrl_total = int(lr["ctrl_peak_total_mib"])
        wkr_sum = int(lr["workers_peak_total_mib"])
        op = int(lr["operator_peak_mib"])
        mock = int(lr["mock_peak_mib"])

        out.append({
            "concurrency": key[0],
            "isl": key[1],
            "osl": key[2],
            "entries": lr["entries"],
            "requests": lr["requests"],
            "completed": completed,
            "records_success": cr.get("records_success", ""),
            "records_error": cr.get("records_error", ""),
            "errors": cr.get("errors", ""),
            "error_rate": cr.get("error_rate", ""),
            "duration_s": lr["duration_s"],
            "benchmark_duration_s": bench_dur,
            "rps": round(rps, 2) if rps else None,
            "output_token_throughput": otps,
            "ctrl_records_mib": int(lr["ctrl_peak_records_mib"]),
            "ctrl_dataset_mib": int(lr["ctrl_peak_dataset_mib"]),
            "ctrl_total_mib": ctrl_total,
            "worker_pod_count": int(lr["worker_pod_count"]),
            "workers_sum_mib": wkr_sum,
            "workers_avg_per_pod_mib": int(lr["workers_peak_avg_per_pod_mib"]),
            "workers_max_per_pod_mib": int(lr["workers_peak_max_per_pod_mib"]),
            "operator_mib": op,
            "total_no_mock_mib": ctrl_total + wkr_sum + op,
            "mock_mib": mock,
        })

    out.sort(key=lambda r: (r["isl"], r["concurrency"], r["osl"]))
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {CSV_OUT.relative_to(REPO)}")
    print()
    hdr = (f"{'conc':>5} {'isl':>8} {'osl':>5}  {'rps':>7} {'tok/s':>8}  "
           f"{'rec-ok':>6} {'rec-er':>6}  {'ds':>4} {'rm':>4} {'ctrl':>5} "
           f"{'wkr-mx':>6} {'mock':>4}  {'tot':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in out:
        print(
            f"{r['concurrency']:>5} {r['isl']:>8} {r['osl']:>5}  "
            f"{(r['rps'] or 0):>7.1f} {(r['output_token_throughput'] or 0):>8.0f}  "
            f"{r['records_success']:>6} {r['records_error']:>6}  "
            f"{r['ctrl_dataset_mib']:>4} {r['ctrl_records_mib']:>4} "
            f"{r['ctrl_total_mib']:>5} {r['workers_max_per_pod_mib']:>6} "
            f"{r['mock_mib']:>4}  {r['total_no_mock_mib']:>5}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
