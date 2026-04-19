#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durability ramp: run benchmarks at distinct concurrency levels.

For each level the script:
  1. Submits the benchmark job.
  2. Waits until `progress >= concurrency` (pipeline fully loaded).
  3. Snapshots profile_export.jsonl from the live controller pod.
  4. Runs scripts/analyze_profile_export.py and scripts/analyze_startup_behavior.py.
  5. Deletes the job and moves to the next level.

Workers scale with concurrency (concurrency / 500, rounded to nearest 5).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from string import Template

NAMESPACE = "acasagrande-aiperf-bench"
FIXED_WORKERS = 500  # Keep pod count fixed at 100 (500 workers / 5 per pod)
WORKERS_PER_POD = 5
RECORD_PROCESSORS_PER_POD = 5
# Keep 3× requests so the job stays live during the snapshot window
REQUESTS_MULTIPLIER = 3

DEFAULT_LEVELS = [300_000, 400_000, 500_000, 600_000, 750_000, 1_000_000]

MOCK_TTFT_MS = 10000
MOCK_ITL_MS = 50
MOCK_OSL = 50000

TEMPLATE = Template(
    """\
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: mock-ramp-${concurrency_label}
  namespace: ${namespace}
spec:
  image: ${image}
  resourceMode: burstable
  connectionsPerWorker: ${connections_per_worker}
  # Fixed 100 pods (500 workers); connectionsPerWorker scales with concurrency
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_LOGS_SCRIPT = Path(__file__).parent / "capture_pod_logs.py"


def kubectl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", *args], capture_output=True, text=True)


def get_image() -> str:
    yaml_path = Path(__file__).parent / "mock-250k-benchmark.yaml"
    for line in yaml_path.read_text().splitlines():
        if "image:" in line and "nvcr.io" in line:
            return line.split("image:")[1].strip()
    raise RuntimeError("Could not find image in mock-250k-benchmark.yaml")


def calc_connections_per_worker(concurrency: int) -> int:
    """Scale connectionsPerWorker to reach target concurrency with fixed worker count."""
    return max(concurrency // FIXED_WORKERS, 1)


def make_manifest(concurrency: int, image: str) -> str:
    """Render the job manifest for a given concurrency level."""
    conn_per_worker = calc_connections_per_worker(concurrency)
    return TEMPLATE.substitute(
        namespace=NAMESPACE,
        image=image,
        concurrency_label=f"{concurrency // 1000}k",
        concurrency=concurrency,
        requests=concurrency * REQUESTS_MULTIPLIER,
        workers=FIXED_WORKERS,
        workers_per_pod=WORKERS_PER_POD,
        record_processors_per_pod=RECORD_PROCESSORS_PER_POD,
        connections_per_worker=conn_per_worker,
    )


def job_name(concurrency: int) -> str:
    return f"mock-ramp-{concurrency // 1000}k"


def parse_job_status(line: str) -> tuple[str, int, float, str]:
    """Parse: NAME PHASE STAGE PROGRESS QPS AGE"""
    parts = line.split()
    phase = parts[1] if len(parts) > 1 else "Unknown"
    try:
        progress = int(parts[3]) if len(parts) > 3 else 0
    except ValueError:
        progress = 0
    try:
        qps = float(parts[4]) if len(parts) > 4 else 0.0
    except ValueError:
        qps = 0.0
    age = parts[5] if len(parts) > 5 else "?"
    return phase, progress, qps, age


def _find_controller_pod(name: str) -> str | None:
    """Return the controller pod name if it exists and is Running, else None."""
    r = kubectl("get", "pods", "-n", NAMESPACE, "--no-headers")
    for line in r.stdout.splitlines():
        if f"{name}-controller" in line and "Running" in line:
            return line.split()[0]
    return None


def _controller_pod_alive(name: str) -> bool:
    return _find_controller_pod(name) is not None


def _jsonl_line_count(ctrl_pod: str) -> int:
    """Get the line count of profile_export.jsonl on the live controller pod."""
    r = subprocess.run(
        [
            "kubectl",
            "exec",
            ctrl_pod,
            "-n",
            NAMESPACE,
            "-c",
            "results-sidecar",
            "--",
            "wc",
            "-l",
            "/results/profile_export.jsonl",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return 0
    try:
        return int(r.stdout.strip().split()[0])
    except (ValueError, IndexError):
        return 0


def wait_for_records(
    name: str,
    target: int,
    poll_interval: int = 15,
    timeout: int = 1800,
) -> bool:
    """Wait until the jsonl file on the controller has >= target completed records."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        ctrl_pod = _find_controller_pod(name)
        if not ctrl_pod:
            print("  Controller pod gone while waiting for records.")
            return False
        count = _jsonl_line_count(ctrl_pod)
        pct = count / target * 100 if target > 0 else 0
        print(f"  [{name}] records: {count:,}/{target:,} ({pct:.0f}%)")
        if count >= target:
            return True
        time.sleep(poll_interval)
    print("  Timed out waiting for records")
    return False


def wait_until_progress(
    name: str,
    target: int,
    poll_interval: int = 15,
    timeout: int = 7200,
    stuck_timeout: int = 300,
) -> bool:
    """Wait until progress >= target. Returns False on failure/timeout.

    Also detects the "stuck initializing" state: if the phase is Initializing
    or Pending with 0 progress and the controller pod is gone, treat as failure
    immediately rather than waiting for the full timeout.
    """
    deadline = time.time() + timeout
    last_progress_time = time.time()
    last_progress = -1
    not_found_count = 0
    while time.time() < deadline:
        r = kubectl("get", "aiperfjob", name, "-n", NAMESPACE, "--no-headers")
        if r.returncode != 0 or not r.stdout.strip():
            not_found_count += 1
            if not_found_count >= 3:
                print("  Job resource disappeared — treating as dead.")
                return False
            time.sleep(poll_interval)
            continue
        not_found_count = 0
        if True:
            phase, progress, qps, age = parse_job_status(r.stdout.strip())
            pct = progress / target * 100 if target > 0 else 0
            print(
                f"  [{name}] {phase}  {progress:,}/{target:,} ({pct:.0f}%)"
                f"  {qps:.0f} QPS  age={age}"
            )
            if progress >= target:
                return True
            if phase in ("Failed", "Cancelled"):
                print(f"  Terminal phase: {phase}")
                return False

            # Track progress; reset timer when it advances
            if progress > last_progress:
                last_progress = progress
                last_progress_time = time.time()

            # Stuck detection: no progress for stuck_timeout and controller is gone
            if (
                time.time() - last_progress_time > stuck_timeout
                and not _controller_pod_alive(name)
            ):
                print(
                    f"  No progress for {stuck_timeout}s and controller pod "
                    f"is gone — treating as dead."
                )
                return False
        time.sleep(poll_interval)
    print("  Timed out")
    return False


def capture_logs(name: str, out_dir: Path, label: str = "snapshot") -> None:
    """Download all container logs for the job in parallel."""
    log_dir = out_dir / f"logs-{label}"
    print(f"  Capturing all container logs → {log_dir.relative_to(PROJECT_ROOT)}")
    result = subprocess.run(
        [
            "uv",
            "run",
            str(CAPTURE_LOGS_SCRIPT),
            name,
            "--out-dir",
            str(log_dir),
            "--previous",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        print("  Log capture encountered errors (partial capture may exist)")


def snapshot_jsonl(name: str, out_dir: Path) -> Path | None:
    """Atomically freeze and download profile_export.jsonl from the live controller pod.

    Uses an in-container cp to create a point-in-time snapshot before downloading,
    avoiding torn reads from concurrent writes to the live file.
    """
    r = kubectl("get", "pods", "-n", NAMESPACE, "--no-headers")
    if r.returncode != 0:
        return None
    ctrl_pod = None
    for line in r.stdout.splitlines():
        if f"{name}-controller" in line:
            ctrl_pod = line.split()[0]
            break
    if not ctrl_pod:
        print("  Could not find controller pod")
        return None

    # Freeze: cp inside the container to a temp path so we get a consistent snapshot.
    freeze_path = "/tmp/profile_export_snapshot.jsonl"
    freeze = subprocess.run(
        [
            "kubectl",
            "exec",
            ctrl_pod,
            "-n",
            NAMESPACE,
            "-c",
            "results-sidecar",
            "--",
            "cp",
            "/results/profile_export.jsonl",
            freeze_path,
        ],
        capture_output=True,
        text=True,
    )
    if freeze.returncode != 0:
        print(
            f"  In-container freeze failed: {freeze.stderr} — falling back to live copy"
        )
        freeze_path = "/results/profile_export.jsonl"

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "profile_export.jsonl"
    print(f"  Downloading frozen snapshot from {ctrl_pod}:{freeze_path} → {dest}")
    cp = subprocess.run(
        [
            "kubectl",
            "cp",
            f"{NAMESPACE}/{ctrl_pod}:{freeze_path}",
            str(dest),
            "-c",
            "results-sidecar",
        ],
        capture_output=True,
        text=True,
    )
    if cp.returncode != 0:
        print(f"  kubectl cp failed: {cp.stderr}")
        return None
    # Truncate any partial trailing record from a torn read during active writes.
    content = dest.read_bytes()
    last_newline = content.rfind(b"\n")
    if last_newline >= 0 and last_newline < len(content) - 1:
        dest.write_bytes(content[: last_newline + 1])
        content = content[: last_newline + 1]

    lines = content.count(b"\n")
    print(f"  Downloaded {lines:,} records ({len(content) / 1e6:.1f} MB)")
    return dest


def run_scripts(jsonl: Path, out_dir: Path, concurrency: int) -> None:
    """Run the analysis scripts from scripts/."""
    scripts_dir = PROJECT_ROOT / "scripts"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    tasks = [
        (
            "analyze_profile_export",
            [
                "uv",
                "run",
                str(scripts_dir / "analyze_profile_export.py"),
                str(jsonl),
                "--output-dir",
                str(plots_dir),
                "--mock-ttft-ms",
                str(MOCK_TTFT_MS),
                "--mock-itl-ms",
                str(MOCK_ITL_MS),
                "--mock-osl",
                str(MOCK_OSL),
            ],
        ),
        (
            "analyze_startup_behavior",
            [
                "uv",
                "run",
                str(scripts_dir / "analyze_startup_behavior.py"),
                str(jsonl),
                "--n",
                str(concurrency),
            ],
        ),
        (
            "plot_credit_pipeline_over_time",
            [
                "uv",
                "run",
                str(scripts_dir / "plot_credit_pipeline_over_time.py"),
                str(jsonl),
                "--output",
                str(plots_dir / "credit_pipeline.png"),
            ],
        ),
        (
            "render_sweepline_throughput_html",
            [
                "uv",
                "run",
                str(scripts_dir / "render_sweepline_throughput_html.py"),
                str(jsonl),
                "--output",
                str(plots_dir / "sweepline_throughput.html"),
            ],
        ),
        (
            "render_throughput_concurrency_html",
            [
                "uv",
                "run",
                str(scripts_dir / "render_throughput_concurrency_html.py"),
                str(jsonl),
                "--output",
                str(plots_dir / "throughput_concurrency.html"),
            ],
        ),
    ]

    for script_name, cmd in tasks:
        print(f"  Running {script_name}...")
        log_file = plots_dir / f"{script_name}.log"
        with log_file.open("w") as lf:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=lf,
                stderr=subprocess.STDOUT,
                env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
            )
        status = (
            "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        )
        print(f"    {status} — log: {log_file.relative_to(PROJECT_ROOT)}")


def run_level(concurrency: int, image: str) -> bool:
    name = job_name(concurrency)
    requests = concurrency * REQUESTS_MULTIPLIER
    out_dir = PROJECT_ROOT / "artifacts" / f"ramp-{concurrency // 1000}k"

    conn_per_worker = calc_connections_per_worker(concurrency)
    print(
        f"\n{'=' * 65}\n"
        f"  LEVEL: {concurrency:,} concurrency\n"
        f"  workers={FIXED_WORKERS}  pods={FIXED_WORKERS // WORKERS_PER_POD}"
        f"  conn/worker={conn_per_worker}"
        f"  target_requests={concurrency:,}  job_requests={requests:,}\n"
        f"{'=' * 65}"
    )

    manifest = make_manifest(concurrency, image)
    apply = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=manifest,
        capture_output=True,
        text=True,
    )
    if apply.returncode != 0:
        print(f"  ERROR: {apply.stderr}")
        return False
    print(f"  Submitted {name}")

    ok = wait_until_progress(name, concurrency)
    if not ok:
        print(
            "\n  First attempt failed — retrying once (first-attempt cluster race)..."
        )
        capture_logs(name, out_dir, label="failure-attempt1")
        kubectl("delete", "aiperfjob", name, "-n", NAMESPACE, "--wait=true")
        time.sleep(10)
        apply2 = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=make_manifest(concurrency, image),
            capture_output=True,
            text=True,
        )
        if apply2.returncode != 0:
            print(f"  ERROR on retry: {apply2.stderr}")
            return False
        print(f"  Resubmitted {name} (attempt 2)")
        ok = wait_until_progress(name, concurrency)
        if not ok:
            print("\n  FAIL on retry — capturing logs and stopping.")
            capture_logs(name, out_dir, label="failure-attempt2")
            kubectl("delete", "aiperfjob", name, "-n", NAMESPACE, "--wait=false")
            return False

    print(f"\n  {concurrency:,} credits issued — waiting for records to complete...")
    if not wait_for_records(name, concurrency):
        print("  Records did not reach target — snapshotting what we have.")
    capture_logs(name, out_dir, label="snapshot")
    jsonl = snapshot_jsonl(name, out_dir)
    if jsonl:
        print("  Running analysis scripts...")
        run_scripts(jsonl, out_dir, concurrency)

    kubectl("delete", "aiperfjob", name, "-n", NAMESPACE, "--wait=false")
    print(f"  Deleted {name}. Artifacts in artifacts/ramp-{concurrency // 1000}k/")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=DEFAULT_LEVELS,
        metavar="N",
        help=f"Concurrency levels (default: {DEFAULT_LEVELS})",
    )
    parser.add_argument("--image", help="Override image")
    args = parser.parse_args(argv)

    image = args.image or get_image()
    print(f"Image: {image}")
    print(f"Levels: {[f'{c // 1000}k' for c in args.levels]}")

    for concurrency in args.levels:
        if not run_level(concurrency, image):
            print(f"\nFAIL at {concurrency:,} — stopping ramp.")
            return 1

    print(f"\n{'=' * 65}\nAll {len(args.levels)} levels passed.\n{'=' * 65}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
