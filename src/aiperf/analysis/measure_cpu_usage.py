#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure per-process CPU usage during an AIPerf benchmark run.

Starts an aiperf profile command, then polls /proc/[pid]/stat every
second for all aiperf child processes. After the run completes,
computes actual CPU cores consumed per process and per service type.

Usage:
    # Against a real or mock server:
    uv run python scripts/measure_cpu_usage.py -- \
        aiperf profile -m Qwen/Qwen3-0.6B --concurrency 50 \
        --url localhost:8765 --benchmark-duration 30 --streaming \
        --osl 128 --isl 512 --workers-max 16 --record-processors 4 \
        --no-gpu-telemetry --no-server-metrics --num-dataset-entries 500 \
        --ui simple

    # Or with env var to suppress event loop health spam:
    AIPERF_SERVICE_EVENT_LOOP_HEALTH_ENABLED=false uv run python scripts/measure_cpu_usage.py -- ...
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CpuSample:
    """Single CPU sample from /proc/[pid]/stat."""

    timestamp: float  # wall clock
    utime: int  # user ticks
    stime: int  # system ticks
    rss_pages: int  # resident set size in pages


@dataclass
class ProcessInfo:
    """Tracked process with CPU samples."""

    pid: int
    name: str  # e.g. "worker_0", "record_processor_2"
    service_type: str  # e.g. "worker", "record_processor", "records_manager"
    samples: list[CpuSample] = field(default_factory=list)

    @property
    def cpu_seconds(self) -> float:
        """Total CPU seconds (user + system) consumed over sampling period."""
        if len(self.samples) < 2:
            return 0.0
        first, last = self.samples[0], self.samples[-1]
        ticks = (last.utime + last.stime) - (first.utime + first.stime)
        return ticks / os.sysconf("SC_CLK_TCK")

    @property
    def wall_seconds(self) -> float:
        """Wall clock duration of sampling."""
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].timestamp - self.samples[0].timestamp

    @property
    def avg_cores(self) -> float:
        """Average CPU cores used (cpu_seconds / wall_seconds)."""
        wall = self.wall_seconds
        return self.cpu_seconds / wall if wall > 0 else 0.0

    @property
    def peak_rss_mib(self) -> float:
        """Peak RSS in MiB."""
        if not self.samples:
            return 0.0
        page_size = os.sysconf("SC_PAGE_SIZE")
        return max(s.rss_pages for s in self.samples) * page_size / (1024 * 1024)


def _parse_proc_stat(raw: str) -> list[str]:
    """Parse /proc/[pid]/stat handling comm fields with spaces/parens.

    The comm field (field 2) is wrapped in parens and can contain anything,
    including spaces and parens. Everything after the LAST ')' is safe to split.
    """
    close_paren = raw.rfind(")")
    if close_paren < 0:
        return raw.split()
    pid_str = raw[: raw.index("(")].strip()
    after_comm = raw[close_paren + 1 :].split()
    return [pid_str, "(comm)"] + after_comm


def _read_proc_stat(pid: int) -> CpuSample | None:
    """Read CPU and RSS from /proc/[pid]/stat."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = _parse_proc_stat(f.read())
        return CpuSample(
            timestamp=time.monotonic(),
            utime=int(parts[13]),
            stime=int(parts[14]),
            rss_pages=int(parts[23]),
        )
    except (
        FileNotFoundError,
        ProcessLookupError,
        IndexError,
        PermissionError,
        ValueError,
    ):
        return None


_SERVICE_NAMES = {
    "system_controller",
    "dataset_manager",
    "timing_manager",
    "worker_manager",
    "records_manager",
    "record_processor",
    "worker",
    "gpu_telemetry_manager",
    "server_metrics_manager",
    "worker_group_manager",
    "api",
}


def _classify_process(cmdline: str) -> tuple[str, str] | None:
    """Extract process name and service type from /proc/[pid]/cmdline.

    AIPerf subprocesses use setproctitle so cmdline looks like:
        "aiperf worker_0" or "aiperf record_processor_3"
    The null-byte-separated cmdline from /proc has "aiperf\x00worker_0\x00..."

    Returns (name, service_type) or None if not an aiperf service process.
    """
    # Split on null bytes and whitespace
    parts = [p for p in re.split(r"[\x00\s]+", cmdline) if p]

    if len(parts) < 2 or "aiperf" not in parts[0]:
        return None

    name = parts[1]

    # Derive service type: "worker_0" -> "worker", "record_processor_3" -> "record_processor"
    service_type = re.sub(r"_[0-9a-f]+$", "", name)

    if service_type in _SERVICE_NAMES:
        return name, service_type

    return None


def _discover_aiperf_processes(
    known_pids: set[int],
) -> dict[int, ProcessInfo]:
    """Find all aiperf service processes by scanning /proc.

    Scans all processes for aiperf service cmdlines (worker_N,
    record_processor_N, etc.). Skips PIDs already in known_pids.
    """
    procs: dict[int, ProcessInfo] = {}
    try:
        all_pids = [int(d) for d in os.listdir("/proc") if d.isdigit()]
    except OSError:
        return procs

    for pid in all_pids:
        if pid in known_pids:
            continue
        try:
            with open(f"/proc/{pid}/cmdline") as f:
                cmdline = f.read()
            info = _classify_process(cmdline)
            if info:
                name, stype = info
                procs[pid] = ProcessInfo(pid=pid, name=name, service_type=stype)
        except (
            FileNotFoundError,
            ProcessLookupError,
            PermissionError,
        ):
            continue

    return procs


def _sample_all(procs: dict[int, ProcessInfo]) -> int:
    """Sample CPU for all tracked processes. Returns count of live processes."""
    alive = 0
    for pid, info in procs.items():
        sample = _read_proc_stat(pid)
        if sample:
            info.samples.append(sample)
            alive += 1
    return alive


def run_and_measure(
    aiperf_args: list[str], poll_interval: float = 1.0
) -> list[ProcessInfo]:
    """Run an aiperf command and measure CPU usage of all child processes."""
    print(f"Starting: {' '.join(aiperf_args)}")
    print(f"Polling CPU every {poll_interval}s...")
    print()

    # Let aiperf output flow to terminal directly (no pipe capture — avoids
    # deadlock when aiperf generates more output than the 64KB pipe buffer).
    proc = subprocess.Popen(aiperf_args)

    # Wait for subprocesses to spawn
    time.sleep(3)

    all_procs: dict[int, ProcessInfo] = {}
    start_time = time.monotonic()

    try:
        while proc.poll() is None:
            new_procs = _discover_aiperf_processes(set(all_procs.keys()))
            all_procs.update(new_procs)
            _sample_all(all_procs)
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()

    # Final sample
    _sample_all(all_procs)
    elapsed = time.monotonic() - start_time

    print(
        f"\nBenchmark completed in {elapsed:.1f}s, {len(all_procs)} processes tracked."
    )
    return list(all_procs.values())


def print_cpu_report(procs: list[ProcessInfo]) -> None:
    """Print CPU usage report grouped by service type."""
    if not procs:
        print("No processes tracked.")
        return

    print()
    print("=" * 90)
    print("  CPU Usage Per Process")
    print("=" * 90)
    print()
    print(
        f"  {'Process':<30} {'CPU sec':>8} {'Wall sec':>9} {'Avg cores':>10} "
        f"{'Peak RSS':>10} {'Samples':>8}"
    )
    print("  " + "-" * 78)

    # Sort by service type then name
    procs_sorted = sorted(procs, key=lambda p: (p.service_type, p.name))

    for p in procs_sorted:
        if p.wall_seconds < 1:
            continue
        print(
            f"  {p.name:<30} {p.cpu_seconds:>7.1f}s {p.wall_seconds:>8.1f}s "
            f"{p.avg_cores:>9.3f} {p.peak_rss_mib:>9.1f}M {len(p.samples):>7}"
        )

    # Aggregate by service type
    print()
    print("=" * 90)
    print("  CPU Usage By Service Type (aggregated)")
    print("=" * 90)
    print()
    print(
        f"  {'Service Type':<25} {'Count':>6} {'Total CPU':>10} {'Avg cores':>10} "
        f"{'Per-proc':>10} {'Peak RSS/proc':>13}"
    )
    print("  " + "-" * 78)

    by_type: dict[str, list[ProcessInfo]] = defaultdict(list)
    for p in procs:
        if p.wall_seconds >= 1:
            by_type[p.service_type].append(p)

    total_cpu = 0.0
    total_cores = 0.0
    for stype in sorted(by_type.keys()):
        group = by_type[stype]
        count = len(group)
        group_cpu = sum(p.cpu_seconds for p in group)
        group_cores = sum(p.avg_cores for p in group)
        per_proc_cores = group_cores / count if count else 0
        avg_peak_rss = sum(p.peak_rss_mib for p in group) / count if count else 0
        total_cpu += group_cpu
        total_cores += group_cores
        print(
            f"  {stype:<25} {count:>5} {group_cpu:>9.1f}s {group_cores:>9.3f} "
            f"{per_proc_cores:>9.3f} {avg_peak_rss:>12.1f}M"
        )

    print("  " + "-" * 78)
    print(
        f"  {'TOTAL':<25} {sum(len(g) for g in by_type.values()):>5} "
        f"{total_cpu:>9.1f}s {total_cores:>9.3f}"
    )
    print()


def main() -> None:
    # Split args on "--"
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        aiperf_args = sys.argv[idx + 1 :]
    else:
        # Default demo command
        aiperf_args = [
            "aiperf",
            "profile",
            "-m",
            "Qwen/Qwen3-0.6B",
            "--concurrency",
            "50",
            "--url",
            "localhost:8765",
            "--benchmark-duration",
            "30",
            "--streaming",
            "--osl",
            "128",
            "--isl",
            "512",
            "--workers-max",
            "16",
            "--record-processors",
            "4",
            "--no-gpu-telemetry",
            "--no-server-metrics",
            "--num-dataset-entries",
            "500",
            "--ui",
            "simple",
        ]

    procs = run_and_measure(aiperf_args)
    print_cpu_report(procs)


if __name__ == "__main__":
    main()
