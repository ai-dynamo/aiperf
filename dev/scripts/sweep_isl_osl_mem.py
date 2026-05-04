#!/usr/bin/env python3
"""ISL/OSL × concurrency sweep with system-wide memory tracking.

Goal: measure how input/output sequence length affects per-component
peak memory and request throughput on the K8s mock-server stack —
specifically whether the RecordProcessor queue-depth amplification (cited
in `docs/kubernetes/memory-estimator.md` as ~10x at ISL+OSL=173K) shows up
in worker-pod RSS.

Per-cell, tracks peak memory across one sample cycle per ~3s for:
  - Controller pod (per-container, with records-manager called out)
  - All worker pods (aggregated: sum, avg, max per pod)
  - Operator pod (control-plane)
  - Mock-server pod

Adapted from `/tmp/aiperf_sweep_allmem.py` (the prior concurrency-axis sweep
on 2026-04-26). Differences:
  - Sweep grid is now ISL/OSL × concurrency rather than concurrency only.
  - Manifest emits `prompts: {isl: {mean: <isl>}, osl: {mean: <osl>}}`.
  - Per-cell `requests` scales with concurrency, capped to keep wall time
    bounded at large OSL where mock-server slows down per-request.
  - Output CSV adds isl/osl columns and an output_tokens_per_second column.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import textwrap
import time

CTX = "nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-01"
NS_BENCH = "acasagrande-aiperf-bench"
NS_OP = "acasagrande-aiperf"
CONNECTIONS_PER_WORKER = 250
WORKERS_PER_POD = 10
SLEEP_S = 3
RUN_TIMEOUT_S = 1800

# Cells: (concurrency, isl, osl, requests).
# Requests trimmed at large OSL — mock emits one SSE chunk per ~ms per OSL
# token, so wall time grows linearly in OSL. We aim for ~30-90s of steady
# state per cell, which is enough to hit peak RSS.
CELLS: list[tuple[int, int, int, int]] = [
    # Main ISL/OSL axis at concurrency=5000.
    (5000, 128, 128, 50_000),
    (5000, 1024, 128, 50_000),
    (5000, 128, 1024, 30_000),
    (5000, 1024, 1024, 30_000),
    (5000, 4096, 1024, 30_000),
    (5000, 1024, 4096, 15_000),
    (5000, 4096, 4096, 15_000),
    # Concurrency=10000 sanity cells (3 symmetric shapes).
    # Note: prior plan had concurrency=20000 here, but 8 worker pods × ~3 CPU
    # don't fit on 3× 8-CPU customer-cpu nodes that this cluster actually has
    # (system-cpu pool is tainted off-limits). 10000 needs 4 worker pods → fits.
    (10000, 128, 128, 100_000),
    (10000, 1024, 1024, 60_000),
    (10000, 4096, 4096, 20_000),
]

CTRL_CONTAINERS = [
    "records-manager",
    "event-bus-proxy",
    "api",
    "dataset-manager",
    "timing-manager",
    "control-plane",
    "results-sidecar",
    "server-metrics-manager",
    "gpu-telemetry-manager",
]

FIELDNAMES = [
    "concurrency",
    "isl",
    "osl",
    "requests",
    "workers",
    "phase",
    "duration_s",
    "rps",
    "output_tokens_per_second",
    "completed",
    # Controller pod
    "ctrl_peak_records_mib",
    "ctrl_peak_total_mib",
    # Worker pods (aggregated)
    "worker_pod_count",
    "workers_peak_total_mib",
    "workers_peak_avg_per_pod_mib",
    "workers_peak_max_per_pod_mib",
    # Other system pods
    "operator_peak_mib",
    "mock_peak_mib",
    # Grand total
    "grand_total_peak_mib",
]


def kubectl(*args: str, check: bool = True) -> str:
    cmd = ["kubectl", "--context", CTX, *args]
    res = subprocess.run(cmd, capture_output=True, text=True, check=check)
    return res.stdout


def manifest(
    name: str,
    image: str,
    concurrency: int,
    isl: int,
    osl: int,
    requests: int,
    workers: int,
) -> str:
    return textwrap.dedent(f"""
        apiVersion: aiperf.nvidia.com/v1alpha1
        kind: AIPerfJob
        metadata:
          name: {name}
          namespace: {NS_BENCH}
        spec:
          image: {image}
          connectionsPerWorker: {CONNECTIONS_PER_WORKER}
          resourceMode: burstable
          ttlSecondsAfterFinished: 600
          benchmark:
            models: {{ items: [{{name: mock}}] }}
            tokenizer: {{ name: builtin }}
            endpoint:
              streaming: true
              urls:
              - http://aiperf-mock-server.{NS_BENCH}.svc.cluster.local:8000/v1/chat/completions
            datasets:
            - name: main
              type: synthetic
              prompts:
                isl: {{ mean: {isl} }}
                osl: {{ mean: {osl} }}
            phases:
            - name: profiling
              type: concurrency
              concurrency: {concurrency}
              requests: {requests}
            runtime: {{ ui: none, workers: {workers}, workersPerPod: {WORKERS_PER_POD}, recordProcessorsPerPod: 1 }}
          podTemplate:
            imagePullSecrets: [nvcr-imagepullsecret]
            nodeSelector: {{ kubernetes.io/arch: amd64, nodeGroup: customer-cpu }}
            tolerations:
            - {{ effect: NoSchedule, key: dedicated, operator: Equal, value: user-workload }}
            - {{ effect: NoExecute,  key: dedicated, operator: Equal, value: user-workload }}
            - {{ effect: NoSchedule, key: team, operator: Equal, value: nemo-ci }}
    """).lstrip()


def parse_mem_mib(token: str) -> float:
    m = re.match(r"(\d+(?:\.\d+)?)(Mi|Gi|Ki)$", token)
    if not m:
        return 0.0
    v = float(m.group(1))
    unit = m.group(2)
    if unit == "Gi":
        v *= 1024
    elif unit == "Ki":
        v /= 1024
    return v


def top_pods(ns: str) -> list[tuple[str, str, float]]:
    out = kubectl("-n", ns, "top", "pod", "--containers", "--no-headers", check=False)
    rows: list[tuple[str, str, float]] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        rows.append((parts[0], parts[1], parse_mem_mib(parts[3])))
    return rows


def cr_status(name: str) -> dict:
    out = kubectl(
        "-n",
        NS_BENCH,
        "get",
        "aiperfjob",
        name,
        "-o",
        "json",
        check=False,
    )
    if not out:
        return {}
    try:
        return json.loads(out).get("status", {})
    except json.JSONDecodeError:
        return {}


class PeakSampler:
    def __init__(self, job_name: str) -> None:
        self.job_name = job_name
        self.bench_peaks: dict[tuple[str, str], float] = {}
        self.operator_peaks: dict[tuple[str, str], float] = {}

    def sample(self) -> None:
        bench = top_pods(NS_BENCH)
        for pod, container, mib in bench:
            if not pod.startswith(f"aiperf-{self.job_name}-") and not pod.startswith(
                "aiperf-mock-server-"
            ):
                continue
            key = (pod, container)
            self.bench_peaks[key] = max(self.bench_peaks.get(key, 0.0), mib)
        op = top_pods(NS_OP)
        for pod, container, mib in op:
            key = (pod, container)
            self.operator_peaks[key] = max(self.operator_peaks.get(key, 0.0), mib)

    def summarize(self) -> dict:
        ctrl_pod = next(
            (
                p
                for (p, _c) in self.bench_peaks
                if p.startswith(f"aiperf-{self.job_name}-controller-")
            ),
            None,
        )
        ctrl_peaks = {c: 0.0 for c in CTRL_CONTAINERS}
        ctrl_total = 0.0
        if ctrl_pod:
            for (p, c), mib in self.bench_peaks.items():
                if p == ctrl_pod:
                    ctrl_total += mib
                    if c in ctrl_peaks:
                        ctrl_peaks[c] = mib

        worker_pods: dict[str, float] = {}
        for (p, _c), mib in self.bench_peaks.items():
            if p.startswith(f"aiperf-{self.job_name}-workers-"):
                worker_pods[p] = worker_pods.get(p, 0.0) + mib

        mock_pod_total: dict[str, float] = {}
        for (p, _c), mib in self.bench_peaks.items():
            if p.startswith("aiperf-mock-server-"):
                mock_pod_total[p] = mock_pod_total.get(p, 0.0) + mib
        mock_peak = max(mock_pod_total.values(), default=0.0)

        op_pod_total: dict[str, float] = {}
        for (p, _c), mib in self.operator_peaks.items():
            if p.startswith("aiperf-operator-"):
                op_pod_total[p] = op_pod_total.get(p, 0.0) + mib
        operator_peak = max(op_pod_total.values(), default=0.0)

        worker_total = sum(worker_pods.values())
        worker_count = len(worker_pods)
        worker_avg = worker_total / worker_count if worker_count else 0.0
        worker_max = max(worker_pods.values(), default=0.0)

        return {
            "ctrl_peak_records_mib": int(ctrl_peaks["records-manager"]),
            "ctrl_peak_total_mib": int(ctrl_total),
            "worker_pod_count": worker_count,
            "workers_peak_total_mib": int(worker_total),
            "workers_peak_avg_per_pod_mib": int(worker_avg),
            "workers_peak_max_per_pod_mib": int(worker_max),
            "operator_peak_mib": int(operator_peak),
            "mock_peak_mib": int(mock_peak),
            "grand_total_peak_mib": int(
                ctrl_total + worker_total + operator_peak + mock_peak
            ),
        }


def make_row(
    concurrency: int, isl: int, osl: int, requests: int, workers: int, **extra
) -> dict:
    row = {k: 0 for k in FIELDNAMES}
    row["concurrency"] = concurrency
    row["isl"] = isl
    row["osl"] = osl
    row["requests"] = requests
    row["workers"] = workers
    row["phase"] = ""
    row["rps"] = None
    row["output_tokens_per_second"] = None
    row.update(extra)
    return row


def run_one(
    name: str,
    image: str,
    concurrency: int,
    isl: int,
    osl: int,
    requests: int,
    workers: int,
) -> dict:
    print(
        f"\n=== {name} (concurrency={concurrency}, isl={isl}, osl={osl}, "
        f"requests={requests}, workers={workers}) ==="
    )

    yaml_text = manifest(name, image, concurrency, isl, osl, requests, workers)
    p = subprocess.run(
        ["kubectl", "--context", CTX, "apply", "-f", "-"],
        input=yaml_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode != 0:
        print(f"  apply FAILED: {p.stderr.strip()}", file=sys.stderr)
        return make_row(
            concurrency, isl, osl, requests, workers, phase="ApplyError", duration_s=-1
        )
    print(p.stdout.strip())

    start = time.monotonic()
    sampler = PeakSampler(name)
    last_phase = ""

    while time.monotonic() - start < RUN_TIMEOUT_S:
        s = cr_status(name)
        phase = s.get("phase", "")
        if phase != last_phase:
            elapsed = int(time.monotonic() - start)
            print(f"  [{elapsed:4}s] phase={phase}")
            last_phase = phase

        try:
            sampler.sample()
        except Exception as e:  # noqa: BLE001
            print(f"  sample error: {e}", file=sys.stderr)

        if phase in ("Completed", "Failed", "Error"):
            try:
                sampler.sample()
            except Exception:
                pass
            break

        time.sleep(SLEEP_S)

    final = cr_status(name)
    duration = time.monotonic() - start
    summary = final.get("summary") or {}
    rps = summary.get("throughput_rps")
    out_tps = summary.get("output_tokens_per_second")
    progress = (
        (final.get("phases") or {})
        .get("profiling", {})
        .get(
            "requestsCompleted",
            0,
        )
    )

    summary_row = sampler.summarize()
    row = make_row(
        concurrency,
        isl,
        osl,
        requests,
        workers,
        phase=final.get("phase", ""),
        duration_s=round(duration, 1),
        rps=rps,
        output_tokens_per_second=out_tps,
        completed=progress,
        **summary_row,
    )
    print(
        f"  done: phase={row['phase']} duration={row['duration_s']}s "
        f"rps={rps} out_tps={out_tps} progress={progress}\n"
        f"        ctrl_records={row['ctrl_peak_records_mib']}MiB "
        f"ctrl_total={row['ctrl_peak_total_mib']}MiB\n"
        f"        workers: {row['worker_pod_count']} pods, "
        f"sum={row['workers_peak_total_mib']}MiB "
        f"avg={row['workers_peak_avg_per_pod_mib']}MiB "
        f"max-pod={row['workers_peak_max_per_pod_mib']}MiB\n"
        f"        operator={row['operator_peak_mib']}MiB "
        f"mock={row['mock_peak_mib']}MiB\n"
        f"        grand-total={row['grand_total_peak_mib']}MiB"
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image",
        required=True,
        help="Operator-side image, e.g. nvcr.io/.../aiperf:k8s-multi-...",
    )
    ap.add_argument("--out", default="dev/results/sweep-isl-osl-mem.csv")
    args = ap.parse_args()

    rows: list[dict] = []
    for c, isl, osl, req in CELLS:
        workers = -(-c // CONNECTIONS_PER_WORKER)
        name = f"sweep-iom-c{c:05d}-i{isl:04d}-o{osl:04d}"
        kubectl(
            "-n",
            NS_BENCH,
            "delete",
            "aiperfjob",
            name,
            "--ignore-not-found=true",
            check=False,
        )
        try:
            row = run_one(name, args.image, c, isl, osl, req, workers)
        except Exception as e:  # noqa: BLE001
            print(f"  unexpected error on {name}: {e}", file=sys.stderr)
            row = make_row(
                c, isl, osl, req, workers, phase="UnexpectedError", duration_s=-1
            )
        rows.append(row)

        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {args.out}")

    print(f"\nSweep complete. Results: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
