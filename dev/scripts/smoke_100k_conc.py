#!/usr/bin/env python3
"""One-shot AIPerfJob submission + memory sampling for the 100K-concurrency smoke.

Single cell: conc=100K, ISL=OSL=128, conn/worker=2500 → 40 workers / 4 pods.
Targets the rust mock-server at aiperf-mock-server-rs.acasagrande-aiperf-bench.svc.

Reuses the same kubectl-top sampling approach as sweep_rs_isl_scale.py, but
trimmed to one cell. CR JSON saved on completion.
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
SLEEP_S = 3
RUN_TIMEOUT_S = 1800

# Single cell.
CONCURRENCY = 500_000
ISL = 128
OSL = 128
CONNECTIONS_PER_WORKER = 2500
WORKERS_PER_POD = 20  # 200 workers / 20 = 10 pods (avoid >150 service registration cliff)
ENTRIES = 1000
REQUESTS = 2_000_000

JOB_NAME = "smoke-rs-c500k"

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


def kubectl(*args: str, check: bool = True) -> str:
    cmd = ["kubectl", "--context", CTX, *args]
    res = subprocess.run(cmd, capture_output=True, text=True, check=check)
    return res.stdout


def manifest(image: str, workers: int) -> str:
    return textwrap.dedent(f"""
        apiVersion: aiperf.nvidia.com/v1alpha1
        kind: AIPerfJob
        metadata:
          name: {JOB_NAME}
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
              entries: {ENTRIES}
              prompts:
                isl: {{ mean: {ISL} }}
                osl: {{ mean: {OSL} }}
            phases:
            - name: profiling
              type: concurrency
              concurrency: {CONCURRENCY}
              requests: {REQUESTS}
            runtime: {{ ui: none, workers: {workers}, workersPerPod: {WORKERS_PER_POD}, recordProcessorsPerPod: 4 }}
          podTemplate:
            imagePullSecrets: [nvcr-imagepullsecret]
            nodeSelector: {{ kubernetes.io/arch: amd64 }}
            tolerations:
            - {{ effect: NoSchedule, key: dedicated, operator: Equal, value: user-workload }}
            - {{ effect: NoExecute,  key: dedicated, operator: Equal, value: user-workload }}
            - {{ effect: NoSchedule, key: team, operator: Equal, value: nemo-ci }}
            - {{ effect: NoExecute, key: components.gke.io/gke-managed-components, operator: Equal, value: "true" }}
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
        for pod, container, mib in top_pods(NS_BENCH):
            if not pod.startswith(f"aiperf-{self.job_name}-") and not pod.startswith(
                "aiperf-mock-server-rs-"
            ):
                continue
            key = (pod, container)
            self.bench_peaks[key] = max(self.bench_peaks.get(key, 0.0), mib)
        for pod, container, mib in top_pods(NS_OP):
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
        worker_max = max(worker_pods.values(), default=0.0)

        return {
            "ctrl_records_mib": int(ctrl_peaks["records-manager"]),
            "ctrl_dataset_mib": int(ctrl_peaks["dataset-manager"]),
            "ctrl_total_mib": int(ctrl_total),
            "worker_pod_count": worker_count,
            "workers_sum_mib": int(worker_total),
            "workers_max_per_pod_mib": int(worker_max),
            "operator_mib": int(operator_peak),
            "mock_mib": int(mock_peak),
            "total_no_mock_mib": int(ctrl_total + worker_total + operator_peak),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument(
        "--snap-out", default=f"dev/results/cr-snapshots-rs/{JOB_NAME}.json"
    )
    args = ap.parse_args()

    workers = -(-CONCURRENCY // CONNECTIONS_PER_WORKER)  # 40
    print(
        f"=== {JOB_NAME}: conc={CONCURRENCY}, isl={ISL}, osl={OSL}, "
        f"conn/worker={CONNECTIONS_PER_WORKER}, workers={workers}, "
        f"workersPerPod={WORKERS_PER_POD} -> {-(-workers // WORKERS_PER_POD)} pods ==="
    )

    yaml_text = manifest(args.image, workers)
    p = subprocess.run(
        ["kubectl", "--context", CTX, "apply", "-f", "-"],
        input=yaml_text,
        capture_output=True,
        text=True,
        check=False,
    )
    print(p.stdout.strip() or p.stderr.strip())
    if p.returncode != 0:
        return 1

    start = time.monotonic()
    sampler = PeakSampler(JOB_NAME)
    last_phase = ""
    last_progress = 0

    while time.monotonic() - start < RUN_TIMEOUT_S:
        s = cr_status(JOB_NAME)
        phase = s.get("phase", "")
        prof = (s.get("phases") or {}).get("profiling", {}) or {}
        progress = prof.get("requestsCompleted") or 0
        live_rps = prof.get("requestsPerSecond") or 0.0
        in_flight = prof.get("requestsInFlight") or 0
        rec_ok = prof.get("recordsSuccess") or 0
        rec_er = prof.get("recordsError") or 0
        elapsed = int(time.monotonic() - start)

        if phase != last_phase or progress - last_progress > 50_000:
            print(
                f"  [{elapsed:4}s] phase={phase} progress={progress} "
                f"in-flight={in_flight} live_rps={live_rps:.1f} "
                f"rec_ok={rec_ok} rec_er={rec_er}"
            )
            last_phase = phase
            last_progress = progress

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

    duration = time.monotonic() - start
    final = cr_status(JOB_NAME)
    summary = final.get("summary") or {}
    bench_dur = (summary.get("benchmark_duration") or {}).get("avg") or 0.0
    prof = (final.get("phases") or {}).get("profiling", {}) or {}
    progress = prof.get("requestsCompleted") or 0
    live_rps = prof.get("requestsPerSecond") or 0.0
    rec_ok = prof.get("recordsSuccess") or 0
    rec_er = prof.get("recordsError") or 0

    summary_row = sampler.summarize()
    print()
    print(f"=== {JOB_NAME} done ===")
    print(
        f"  phase={final.get('phase')} duration={duration:.1f}s "
        f"benchmark_duration={bench_dur:.1f}s"
    )
    print(f"  progress={progress}/{REQUESTS} live_rps={live_rps:.1f}")
    print(f"  records: success={rec_ok} error={rec_er}")
    print(
        f"  ctrl_records={summary_row['ctrl_records_mib']}MiB "
        f"dataset={summary_row['ctrl_dataset_mib']}MiB "
        f"ctrl_total={summary_row['ctrl_total_mib']}MiB"
    )
    print(
        f"  workers: {summary_row['worker_pod_count']} pods, "
        f"sum={summary_row['workers_sum_mib']}MiB "
        f"max-pod={summary_row['workers_max_per_pod_mib']}MiB"
    )
    print(
        f"  operator={summary_row['operator_mib']}MiB "
        f"mock={summary_row['mock_mib']}MiB "
        f"total-no-mock={summary_row['total_no_mock_mib']}MiB"
    )

    # Save CR snapshot.
    cr_json = kubectl(
        "-n", NS_BENCH, "get", "aiperfjob", JOB_NAME, "-o", "json", check=False
    )
    if cr_json:
        with open(args.snap_out, "w") as f:
            f.write(cr_json)
        print(f"  saved CR snapshot: {args.snap_out}")

    return 0 if final.get("phase") == "Completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
