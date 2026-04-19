#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture logs from all containers in all benchmark pods simultaneously.

Downloads logs in parallel using kubectl, one file per container.
Output layout:
    <out_dir>/
        <pod_name>/
            <container_name>.log
            <container_name>.previous.log  (if --previous)

Usage:
    python dev/deploy/capture_pod_logs.py [job_name] [--out-dir DIR] [--previous]
    python dev/deploy/capture_pod_logs.py mock-ramp-300k --out-dir /tmp/logs-300k
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

NAMESPACE = "acasagrande-aiperf-bench"


def kubectl(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", *args], capture_output=True, text=True, check=check
    )


def get_pods(job_name: str | None) -> list[tuple[str, list[str]]]:
    """Return list of (pod_name, [container_names]) for the job."""
    r = kubectl("get", "pods", "-n", NAMESPACE, "--no-headers", "-o", "wide")
    pods = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        pod_name = parts[0]
        if job_name and job_name not in pod_name:
            continue
        if "mock-server" in pod_name:
            continue
        # Get container names for this pod
        cr = kubectl(
            "get",
            "pod",
            pod_name,
            "-n",
            NAMESPACE,
            "-o",
            "jsonpath={.spec.containers[*].name}",
        )
        containers = cr.stdout.strip().split() if cr.returncode == 0 else []
        if containers:
            pods.append((pod_name, containers))
    return pods


def fetch_log(
    pod: str, container: str, out_path: Path, previous: bool = False
) -> tuple[str, bool, int]:
    """Fetch one container log. Returns (label, success, lines)."""
    args = ["logs", pod, "-n", NAMESPACE, "-c", container]
    if previous:
        args.append("--previous")
    r = kubectl(*args)
    if r.returncode != 0:
        return f"{pod}/{container}", False, 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(r.stdout)
    lines = r.stdout.count("\n")
    return f"{pod}/{container}", True, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "job_name", nargs="?", help="Job name filter (e.g. mock-ramp-300k)"
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp/pod-logs",
        help="Output directory (default: /tmp/pod-logs)",
    )
    parser.add_argument(
        "--previous", action="store_true", help="Also capture previous container logs"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Parallel download workers (default: 32)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Discovering pods{f' matching {args.job_name!r}' if args.job_name else ''}..."
    )
    pods = get_pods(args.job_name)
    if not pods:
        print("No pods found.")
        return 1

    total_containers = sum(len(cs) for _, cs in pods)
    print(
        f"Found {len(pods)} pods, {total_containers} containers. Downloading to {out_dir}/"
    )

    tasks: list[tuple[str, str, Path, bool]] = []
    for pod_name, containers in pods:
        pod_dir = out_dir / pod_name
        for container in containers:
            tasks.append((pod_name, container, pod_dir / f"{container}.log", False))
            if args.previous:
                tasks.append(
                    (pod_name, container, pod_dir / f"{container}.previous.log", True)
                )

    results: list[tuple[str, bool, int]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_log, pod, c, path, prev): (pod, c)
            for pod, c, path, prev in tasks
        }
        for i, fut in enumerate(as_completed(futures), 1):
            label, ok, lines = fut.result()
            suffix = f"({lines:,} lines)" if ok else "(failed)"
            print(f"  [{i:3d}/{len(tasks)}] {'OK' if ok else 'FAIL'} {label} {suffix}")
            results.append((label, ok, lines))

    ok_count = sum(1 for _, ok, _ in results if ok)
    total_lines = sum(lines for _, ok, lines in results if ok)
    print(
        f"\nDone: {ok_count}/{len(results)} containers, {total_lines:,} total log lines → {out_dir}/"
    )
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
