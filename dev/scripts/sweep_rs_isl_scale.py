#!/usr/bin/env python3
"""Scale-up + agentic-ISL sweep against the Rust mock-server.

Goal: extend dev/scripts/sweep_isl_osl_mem.py to cover (a) higher
concurrency and (b) **huge agentic ISLs up to 1M tokens**, against the
Rust mock at `aiperf-mock-server-rs.acasagrande-aiperf-bench.svc:8000`.

Why the Rust mock:
  - 1.55 MB binary (vs ~1 GB Python mock) — startup is instant.
  - Native axum SSE — no GIL, no asyncio overhead. Per-token emission
    cost dominated by socket write.
  - Pinned tokenizer + Shakespeare corpus is deterministic, so each
    cell is reproducible.

Cell design notes:
  - At ISL ≥ 64K, prompts dominate request size (1M tokens ≈ 4 MB JSON
    body). DatasetManager pre-generates `entries` conversations of full
    prompt length, so memory = entries × ISL_bytes × 1.3 overhead.
    Setting `entries` proportional to expected concurrency keeps that
    bounded.
  - At very low concurrency we still need enough requests to reach a
    steady-state worker peak — so cells set `requests = max(1000, conc*5)`
    or similar.
  - workersPerPod=10 with 3× 8-CPU customer-cpu nodes means realistic
    concurrency ceiling ~15K (5 worker pods × 3 CPU each + the rest of
    the cluster overhead). At ISL=1M, low concurrency anyway.

Captures the same memory peaks as sweep_isl_osl_mem.py via kubectl top
sampling; mock-server is the rust deploy this time.
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
MOCK_HOST = f"aiperf-mock-server-rs.{NS_BENCH}.svc.cluster.local:8000"
CONNECTIONS_PER_WORKER = 100
WORKERS_PER_POD = 10
SLEEP_S = 3
RUN_TIMEOUT_S = 1800

# (concurrency, isl, osl, entries, requests, workersPerPod_override)
# entries = num_conversations (DatasetManager pool size).
# At ISL=1M, prompt cache memory = entries × ~4 MB; cap at 20 entries.
CELLS: list[tuple[int, int, int, int, int, int | None]] = [
    # Cells 1-2 already ran (5K, 128/128) and (10K, 128/128) at conn/worker=250.
    # The 15K cells were skipped — at 150 worker registrations the controller
    # registration storm (52s) + completion path broke under Failed status
    # despite the benchmark actually running. AIPerf controller has a
    # registration scaling limit somewhere between 60 and 150 services.
    # Real focus is agentic ISL anyway — proceeding straight to that ladder.
    # Agentic shapes — long input, short completion (tool/RAG context, agent reply).
    (5000,   8192,  256,  100,  20_000, None),
    (1000,  65536,  256,   80,   3_000, None),
    (500,  262144,  256,   50,   1_000, None),
    (200,  524288,  256,   30,     500, None),
    # The big one: 1M ISL, low concurrency.
    (50,  1048576,  256,   20,     200, None),
    (20,  1048576, 1024,   20,     100, None),
]

CTRL_CONTAINERS = [
    "records-manager", "event-bus-proxy", "api", "dataset-manager",
    "timing-manager", "control-plane", "results-sidecar",
    "server-metrics-manager", "gpu-telemetry-manager",
]

FIELDNAMES = [
    "concurrency", "isl", "osl", "entries", "requests", "workers",
    "phase", "duration_s", "rps", "output_tokens_per_second", "completed",
    "ctrl_peak_records_mib", "ctrl_peak_dataset_mib", "ctrl_peak_total_mib",
    "worker_pod_count",
    "workers_peak_total_mib", "workers_peak_avg_per_pod_mib",
    "workers_peak_max_per_pod_mib",
    "operator_peak_mib",
    "mock_peak_mib",
    "total_no_mock_mib",
]


def kubectl(*args: str, check: bool = True) -> str:
    cmd = ["kubectl", "--context", CTX, *args]
    res = subprocess.run(cmd, capture_output=True, text=True, check=check)
    return res.stdout


def manifest(name: str, image: str, concurrency: int, isl: int, osl: int,
             entries: int, requests: int, workers: int,
             workers_per_pod: int) -> str:
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
              - http://{MOCK_HOST}/v1/chat/completions
            datasets:
            - name: main
              type: synthetic
              entries: {entries}
              prompts:
                isl: {{ mean: {isl} }}
                osl: {{ mean: {osl} }}
            phases:
            - name: profiling
              type: concurrency
              concurrency: {concurrency}
              requests: {requests}
            runtime: {{ ui: none, workers: {workers}, workersPerPod: {workers_per_pod}, recordProcessorsPerPod: 1 }}
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
        "-n", NS_BENCH, "get", "aiperfjob", name, "-o", "json", check=False,
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
        for pod, container, mib in top_pods(NS_BENCH):
            if not pod.startswith(f"aiperf-{self.job_name}-") and \
               not pod.startswith("aiperf-mock-server-rs-") and \
               not pod.startswith("aiperf-mock-server-"):
                continue
            key = (pod, container)
            self.bench_peaks[key] = max(self.bench_peaks.get(key, 0.0), mib)
        for pod, container, mib in top_pods(NS_OP):
            key = (pod, container)
            self.operator_peaks[key] = max(self.operator_peaks.get(key, 0.0), mib)

    def summarize(self) -> dict:
        ctrl_pod = next(
            (p for (p, _c) in self.bench_peaks
             if p.startswith(f"aiperf-{self.job_name}-controller-")),
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

        # Identify the rust mock pod for memory tracking.
        mock_pod_total: dict[str, float] = {}
        for (p, _c), mib in self.bench_peaks.items():
            if p.startswith("aiperf-mock-server-rs-"):
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
            "ctrl_peak_dataset_mib": int(ctrl_peaks["dataset-manager"]),
            "ctrl_peak_total_mib": int(ctrl_total),
            "worker_pod_count": worker_count,
            "workers_peak_total_mib": int(worker_total),
            "workers_peak_avg_per_pod_mib": int(worker_avg),
            "workers_peak_max_per_pod_mib": int(worker_max),
            "operator_peak_mib": int(operator_peak),
            "mock_peak_mib": int(mock_peak),
            "total_no_mock_mib": int(ctrl_total + worker_total + operator_peak),
        }


def make_row(concurrency: int, isl: int, osl: int, entries: int, requests: int,
             workers: int, **extra) -> dict:
    row = {k: 0 for k in FIELDNAMES}
    row.update({
        "concurrency": concurrency, "isl": isl, "osl": osl,
        "entries": entries, "requests": requests, "workers": workers,
        "phase": "", "rps": None, "output_tokens_per_second": None,
    })
    row.update(extra)
    return row


def run_one(name: str, image: str, concurrency: int, isl: int, osl: int,
            entries: int, requests: int, workers: int,
            workers_per_pod: int) -> dict:
    print(f"\n=== {name} (concurrency={concurrency}, isl={isl}, osl={osl}, "
          f"entries={entries}, requests={requests}, workers={workers}) ===")

    yaml_text = manifest(name, image, concurrency, isl, osl, entries,
                         requests, workers, workers_per_pod)
    p = subprocess.run(
        ["kubectl", "--context", CTX, "apply", "-f", "-"],
        input=yaml_text, capture_output=True, text=True, check=False,
    )
    if p.returncode != 0:
        print(f"  apply FAILED: {p.stderr.strip()}", file=sys.stderr)
        return make_row(concurrency, isl, osl, entries, requests, workers,
                        phase="ApplyError", duration_s=-1)
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
    bench_dur = (summary.get("benchmark_duration") or {}).get("avg") or 0.0
    progress = (final.get("phases") or {}).get("profiling", {}).get(
        "requestsCompleted", 0,
    )
    rps = round(progress / bench_dur, 2) if bench_dur > 0 else None
    osl_avg = (summary.get("output_sequence_length") or {}).get("avg") or osl
    out_tps = round(progress * osl_avg / bench_dur, 2) if bench_dur > 0 else None

    summary_row = sampler.summarize()
    row = make_row(
        concurrency, isl, osl, entries, requests, workers,
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
        f"dataset={row['ctrl_peak_dataset_mib']}MiB "
        f"ctrl_total={row['ctrl_peak_total_mib']}MiB\n"
        f"        workers: {row['worker_pod_count']} pods, "
        f"sum={row['workers_peak_total_mib']}MiB "
        f"avg={row['workers_peak_avg_per_pod_mib']}MiB "
        f"max-pod={row['workers_peak_max_per_pod_mib']}MiB\n"
        f"        operator={row['operator_peak_mib']}MiB "
        f"mock={row['mock_peak_mib']}MiB\n"
        f"        total-no-mock={row['total_no_mock_mib']}MiB"
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True,
                    help="Operator-side image (controller pod uses this).")
    ap.add_argument("--out", default="dev/results/sweep-rs-isl-scale.csv")
    ap.add_argument("--append", action="store_true",
                    help="Append to existing CSV (no header) — for resuming.")
    args = ap.parse_args()

    rows: list[dict] = []
    for c, isl, osl, entries, req, wpp_override in CELLS:
        workers = -(-c // CONNECTIONS_PER_WORKER)
        wpp = wpp_override or WORKERS_PER_POD
        name = f"sweep-rs-c{c:05d}-i{isl}-o{osl}"
        kubectl("-n", NS_BENCH, "delete", "aiperfjob", name,
                "--ignore-not-found=true", check=False)
        try:
            row = run_one(name, args.image, c, isl, osl, entries, req,
                          workers, wpp)
        except Exception as e:  # noqa: BLE001
            print(f"  unexpected error on {name}: {e}", file=sys.stderr)
            row = make_row(c, isl, osl, entries, req, workers,
                           phase="UnexpectedError", duration_s=-1)
        rows.append(row)

        if args.append:
            with open(args.out, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writerow(row)
        else:
            with open(args.out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
        print(f"  wrote {args.out}")

    print(f"\nSweep complete. Results: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
