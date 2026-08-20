# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Human-readable formatter for ``ClusterMemoryEstimate``."""

from __future__ import annotations

from aiperf.kubernetes._memory_estimator.estimates import (
    ClusterMemoryEstimate,
    PodEstimate,
)


def format_estimate(est: ClusterMemoryEstimate) -> str:
    """Format a ClusterMemoryEstimate as a human-readable string."""
    p = est.params
    lines: list[str] = []

    lines.append("Memory Estimation for AIPerf Kubernetes Deployment")
    lines.append("=" * 68)
    lines.append("")
    lines.append(
        f"Topology: 1 controller + {p.num_worker_pods} worker pod(s) "
        f"({p.workers_per_pod} workers/pod, {p.record_processors_per_pod} RP/pod)"
    )
    lines.append(
        f"Total requests: ~{p.total_requests:,} | "
        f"Max concurrency: {p.max_concurrency:,} | "
        f"Duration: {p.total_benchmark_duration_s:.0f}s"
    )
    lines.append(
        f"Dataset: {p.dataset_count:,} conversations | "
        f"ISL: {p.avg_isl_tokens} | OSL: {p.avg_osl_tokens} | "
        f"Turns: {p.max_turns}"
    )
    lines.append("")

    _format_pod(lines, "Controller Pod", est.controller)
    _format_pod(lines, f"Worker Pod (x{p.num_worker_pods})", est.worker_pod)

    lines.append("Cluster Total")
    lines.append("-" * 68)
    lines.append(
        f"  {'Controller':<42} {est.controller.total_steady_state_mib:>7.0f} MiB"
    )
    worker_total = est.worker_pod.total_steady_state_mib * p.num_worker_pods
    lines.append(
        f"  {'Workers (' + str(p.num_worker_pods) + ' pods)':<42} {worker_total:>7.0f} MiB"
    )
    lines.append(f"  {'Operator':<42} {est.operator.total_steady_state_mib:>7.0f} MiB")
    lines.append("-" * 68)
    lines.append(f"  {'TOTAL':<42} {est.total_cluster_mib:>7.0f} MiB")
    lines.append("")

    if est.warnings:
        lines.append("Warnings:")
        for w in est.warnings:
            lines.append(f"  [!] {w}")
        lines.append("")

    if est.recommendations:
        lines.append("Recommendations:")
        for r in est.recommendations:
            lines.append(f"  - {r}")
        lines.append("")

    return "\n".join(lines)


def _format_pod(lines: list[str], title: str, pod: PodEstimate) -> None:
    """Format a single pod estimate table."""
    risk = " [!]" if pod.at_risk else ""
    lines.append(f"{title}{risk}")
    lines.append(f"  {'Component':<42} {'Steady':>7}  {'Peak':>7}")
    lines.append("-" * 68)
    for c in pod.components:
        warn = " [!]" if c.warning else ""
        lines.append(
            f"  {c.name:<42} {c.steady_state_mib:>6.0f}  {c.peak_mib:>6.0f}{warn}"
        )
    lines.append("-" * 68)
    lines.append(
        f"  {'TOTAL':<42} {pod.total_steady_state_mib:>6.0f}  {pod.total_peak_mib:>6.0f}  "
        f"(limit: {pod.current_limit_mib:.0f})"
    )
    headroom_str = f"{pod.headroom_pct:.1f}%"
    lines.append(f"  {'Headroom':<42} {headroom_str:>14}")
    lines.append("")
