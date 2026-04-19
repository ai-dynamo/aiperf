#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reliability test suite for AIPerf on Kubernetes.

Runs 100k concurrency at increasing pod counts with three test scenarios:
  1. Sustained load — hold concurrency for 5 minutes, verify no crashes or data loss.
  2. Worker pod chaos — kill random worker pods mid-benchmark, verify recovery.
  3. Mock server restart — rolling-restart the inference server, verify error handling.

Usage:
    python dev/deploy/reliability_tests.py [--scenario sustained|chaos|server-restart|all]
    python dev/deploy/reliability_tests.py --pods 100 --scenario chaos
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path
from string import Template

NAMESPACE = "acasagrande-aiperf-bench"
CONCURRENCY = 100_000
WORKERS_PER_POD = 5
RECORD_PROCESSORS_PER_POD = 5
SUSTAINED_DURATION_S = 300  # 5 minutes
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Enough requests to sustain load for the full duration + margin.
# At ~2000 QPS for 100k concurrency: 2000 * 300s * 2 = 1.2M
REQUESTS = 1_200_000

POD_COUNTS = [20, 50, 100, 150, 200]

TEMPLATE = Template(
    """\
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: reliability-${scenario}-${pods}p
  namespace: ${namespace}
spec:
  image: ${image}
  resourceMode: burstable
  connectionsPerWorker: ${connections_per_worker}
  timeoutSeconds: 0
  ttlSecondsAfterFinished: 600
  benchmark:
    models:
      items:
        - name: mock
    endpoint:
      type: chat
      streaming: true
      urls:
        - http://aiperf-mock-server.${namespace}.svc.cluster.local:8000/v1/chat/completions
    tokenizer:
      name: builtin
    datasets:
      main:
        type: synthetic
        prompts:
          isl:
            mean: 550
          osl: 50000
    phases:
      profiling:
        type: concurrency
        concurrency: ${concurrency}
        requests: ${requests}
    runtime:
      ui: none
      workers: ${workers}
      workers_per_pod: ${workers_per_pod}
      record_processors_per_pod: ${record_processors_per_pod}
    artifacts:
      records:
        - jsonl
    gpu_telemetry:
      enabled: false
    server_metrics:
      enabled: false
    logging:
      level: debug
  keepFailedPods: true
  podTemplate:
    imagePullSecrets:
      - nvcr-imagepullsecret
    nodeSelector:
      kubernetes.io/arch: arm64
      nodeGroup: customer-gpu
    tolerations:
      - key: dedicated
        operator: Equal
        value: user-workload
        effect: NoSchedule
      - key: dedicated
        operator: Equal
        value: user-workload
        effect: NoExecute
      - key: nvidia.com/gpu
        operator: Equal
        value: present
        effect: NoSchedule
      - key: team
        operator: Equal
        value: nemo-ci
        effect: NoSchedule
      - key: kubernetes.io/arch
        operator: Equal
        value: arm64
        effect: NoSchedule
    env:
      - name: HF_HOME
        value: /app/.cache/huggingface
      - name: HF_HUB_DISABLE_SSL_VERIFICATION
        value: "1"
      - name: AIPERF_K8S_SYSTEM_CONTROLLER_MEMORY
        value: 8Gi
      - name: AIPERF_K8S_API_MEMORY
        value: 32Gi
      - name: AIPERF_K8S_API_CPU
        value: 1000m
      - name: AIPERF_K8S_RECORD_PROCESSOR_CPU_REQUEST
        value: "1"
      - name: AIPERF_SERVICE_EVENT_LOOP_HEALTH_STACKTRACE
        value: "true"
"""
)


def kubectl(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", *args], capture_output=True, text=True, check=check
    )


def get_image() -> str:
    yaml_path = Path(__file__).parent / "mock-250k-benchmark.yaml"
    for line in yaml_path.read_text().splitlines():
        if "image:" in line and "nvcr.io" in line:
            return line.split("image:")[1].strip()
    raise RuntimeError("Could not find image")


def job_name(scenario: str, pods: int) -> str:
    return f"reliability-{scenario}-{pods}p"


def make_manifest(scenario: str, pods: int, image: str) -> str:
    workers = pods * WORKERS_PER_POD
    conn_per_worker = max(CONCURRENCY // workers, 1)
    return TEMPLATE.substitute(
        namespace=NAMESPACE,
        image=image,
        scenario=scenario,
        pods=pods,
        concurrency=CONCURRENCY,
        requests=REQUESTS,
        workers=workers,
        workers_per_pod=WORKERS_PER_POD,
        record_processors_per_pod=RECORD_PROCESSORS_PER_POD,
        connections_per_worker=conn_per_worker,
    )


def submit_job(scenario: str, pods: int, image: str) -> str:
    name = job_name(scenario, pods)
    manifest = make_manifest(scenario, pods, image)
    r = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=manifest,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr}")
        return ""
    print(f"  Submitted {name}")
    return name


def wait_for_running(name: str, timeout: int = 600) -> bool:
    """Wait until the job is Running with progress > 0."""
    deadline = time.time() + timeout
    not_found = 0
    while time.time() < deadline:
        r = kubectl("get", "aiperfjob", name, "-n", NAMESPACE, "--no-headers")
        if r.returncode != 0 or not r.stdout.strip():
            not_found += 1
            if not_found >= 3:
                print("  Job disappeared")
                return False
            time.sleep(10)
            continue
        not_found = 0
        parts = r.stdout.split()
        phase = parts[1] if len(parts) > 1 else "?"
        try:
            progress = int(parts[3]) if len(parts) > 3 else 0
        except ValueError:
            progress = 0
        print(f"  [{name}] {phase} progress={progress:,}")
        if phase == "Running" and progress > 0:
            return True
        if phase in ("Failed", "Cancelled"):
            return False
        time.sleep(10)
    return False


def get_job_progress(name: str) -> tuple[str, int]:
    r = kubectl("get", "aiperfjob", name, "-n", NAMESPACE, "--no-headers")
    if r.returncode != 0:
        return "Gone", 0
    parts = r.stdout.split()
    phase = parts[1] if len(parts) > 1 else "?"
    try:
        progress = int(parts[3]) if len(parts) > 3 else 0
    except ValueError:
        progress = 0
    return phase, progress


def cleanup(name: str) -> None:
    kubectl("delete", "aiperfjob", name, "-n", NAMESPACE, "--wait=false")


# =============================================================================
# Scenario: Sustained Load
# =============================================================================


def test_sustained(pods: int, image: str) -> bool:
    """Hold 100k concurrency for 5 minutes. Verify no crashes, progress keeps advancing."""
    name = submit_job("sustained", pods, image)
    if not name:
        return False

    if not wait_for_running(name):
        print("  Failed to reach Running")
        cleanup(name)
        return False

    print(f"  Running sustained load for {SUSTAINED_DURATION_S}s...")
    start = time.time()
    last_progress = 0
    stall_start = None

    while time.time() - start < SUSTAINED_DURATION_S:
        phase, progress = get_job_progress(name)
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:3d}s] {phase} progress={progress:,}")

        if phase in ("Failed", "Cancelled", "Gone"):
            print(f"  FAIL: Job entered {phase} during sustained load")
            cleanup(name)
            return False

        if progress > last_progress:
            last_progress = progress
            stall_start = None
        elif stall_start is None:
            stall_start = time.time()
        elif time.time() - stall_start > 120:
            print(f"  FAIL: Progress stalled for 120s at {progress:,}")
            cleanup(name)
            return False

        time.sleep(15)

    phase, final_progress = get_job_progress(name)
    print(
        f"  Sustained test complete: {final_progress:,} requests in {SUSTAINED_DURATION_S}s"
    )
    cleanup(name)
    return phase not in ("Failed", "Cancelled", "Gone")


# =============================================================================
# Scenario: Worker Pod Chaos
# =============================================================================


def test_chaos(pods: int, image: str) -> bool:
    """Kill random worker pods mid-benchmark. Verify the benchmark continues."""
    name = submit_job("chaos", pods, image)
    if not name:
        return False

    if not wait_for_running(name):
        print("  Failed to reach Running")
        cleanup(name)
        return False

    # Let it stabilize for 30s
    print("  Stabilizing for 30s...")
    time.sleep(30)

    _, progress_before = get_job_progress(name)
    print(f"  Progress before chaos: {progress_before:,}")

    # Kill 10% of worker pods (random selection)
    r = kubectl("get", "pods", "-n", NAMESPACE, "--no-headers")
    worker_pods = [
        line.split()[0]
        for line in r.stdout.splitlines()
        if f"{name.replace('reliability-', 'aiperf-reliability-')}-workers" in line
        and "Running" in line
    ]
    kill_count = max(1, len(worker_pods) // 10)
    targets = random.sample(worker_pods, min(kill_count, len(worker_pods)))
    print(f"  Killing {len(targets)} of {len(worker_pods)} worker pods...")
    for pod in targets:
        kubectl("delete", "pod", pod, "-n", NAMESPACE, "--grace-period=0", "--force")
        print(f"    Killed {pod}")

    # Monitor for 2 minutes — progress should continue
    print("  Monitoring recovery for 120s...")
    start = time.time()
    while time.time() - start < 120:
        phase, progress = get_job_progress(name)
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:3d}s] {phase} progress={progress:,}")
        if phase in ("Failed", "Cancelled", "Gone"):
            print(f"  FAIL: Job died after chaos — {phase}")
            cleanup(name)
            return False
        time.sleep(15)

    _, progress_after = get_job_progress(name)
    delta = progress_after - progress_before
    print(
        f"  Chaos test complete: progress went from {progress_before:,} to "
        f"{progress_after:,} (+{delta:,}) despite killing {len(targets)} pods"
    )
    cleanup(name)
    return delta > 0


# =============================================================================
# Scenario: Mock Server Restart
# =============================================================================


def test_server_restart(pods: int, image: str) -> bool:
    """Rolling-restart the mock server mid-benchmark. Verify error handling."""
    name = submit_job("restart", pods, image)
    if not name:
        return False

    if not wait_for_running(name):
        print("  Failed to reach Running")
        cleanup(name)
        return False

    # Let it stabilize
    print("  Stabilizing for 30s...")
    time.sleep(30)

    _, progress_before = get_job_progress(name)
    print(f"  Progress before server restart: {progress_before:,}")

    # Rolling restart of mock server
    print("  Triggering rolling restart of aiperf-mock-server...")
    kubectl(
        "rollout",
        "restart",
        "deployment/aiperf-mock-server",
        "-n",
        NAMESPACE,
    )

    # Monitor for 3 minutes — progress should continue (with possible errors)
    print("  Monitoring through restart for 180s...")
    start = time.time()
    while time.time() - start < 180:
        phase, progress = get_job_progress(name)
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:3d}s] {phase} progress={progress:,}")
        if phase in ("Failed", "Cancelled", "Gone"):
            print(f"  FAIL: Job died during server restart — {phase}")
            cleanup(name)
            return False
        time.sleep(15)

    _, progress_after = get_job_progress(name)
    delta = progress_after - progress_before
    print(
        f"  Server restart test complete: progress went from {progress_before:,} to "
        f"{progress_after:,} (+{delta:,}) through a rolling restart"
    )
    cleanup(name)
    return delta > 0


# =============================================================================
# Main
# =============================================================================

SCENARIOS = {
    "sustained": test_sustained,
    "chaos": test_chaos,
    "server-restart": test_server_restart,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scenario",
        choices=["sustained", "chaos", "server-restart", "all"],
        default="all",
    )
    parser.add_argument(
        "--pods",
        nargs="+",
        type=int,
        default=POD_COUNTS,
        help=f"Pod counts to test (default: {POD_COUNTS})",
    )
    parser.add_argument("--image", help="Override image")
    args = parser.parse_args(argv)

    image = args.image or get_image()
    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

    print(f"Image: {image}")
    print(f"Concurrency: {CONCURRENCY:,}")
    print(f"Pod counts: {args.pods}")
    print(f"Scenarios: {scenarios}")
    print(f"Sustained duration: {SUSTAINED_DURATION_S}s")
    print()

    results: list[tuple[str, int, str, bool]] = []

    for scenario in scenarios:
        test_fn = SCENARIOS[scenario]
        for pods in args.pods:
            workers = pods * WORKERS_PER_POD
            conn = max(CONCURRENCY // workers, 1)
            print(
                f"\n{'=' * 65}\n"
                f"  TEST: {scenario} @ {pods} pods ({workers} workers × {conn} conn)\n"
                f"{'=' * 65}"
            )
            ok = test_fn(pods, image)
            status = "PASS" if ok else "FAIL"
            results.append((scenario, pods, status, ok))
            print(f"\n  RESULT: {status}")

    # Summary
    print(f"\n{'=' * 65}")
    print("  RELIABILITY TEST SUMMARY")
    print(f"{'=' * 65}")
    for scenario, pods, status, _ in results:
        icon = "+" if status == "PASS" else "X"
        print(f"  [{icon}] {scenario:>15} @ {pods:>3} pods: {status}")

    passed = sum(1 for *_, ok in results if ok)
    total = len(results)
    print(f"\n  {passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
