#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Estimate memory and CPU for AIPerf Kubernetes deployments at scale.

Runs the memory estimator across common deployment scenarios and prints
a comparison table showing per-pod and cluster-wide resource usage.

Usage:
    uv run python scripts/estimate_deployment_resources.py
    uv run python scripts/estimate_deployment_resources.py --concurrency 100000
"""

from __future__ import annotations

import argparse

from aiperf.kubernetes.memory_estimator import (
    MemoryEstimationParams,
    MemoryEstimator,
    format_estimate,
)


def _make_params(
    concurrency: int,
    isl: int,
    osl: int,
    streaming: bool,
    workers: int,
    num_models: int = 1,
    req_multiplier: int = 4,
    duration_s: float = 7200.0,
    num_gpus: int = 0,
    num_server_metrics_endpoints: int = 0,
) -> MemoryEstimationParams:
    wpp = 10
    pods = max(1, workers // wpp)
    rp_per_pod = 2
    return MemoryEstimationParams(
        max_concurrency=concurrency,
        total_requests=concurrency * req_multiplier,
        total_benchmark_duration_s=duration_s,
        total_workers=workers,
        workers_per_pod=wpp,
        num_worker_pods=pods,
        record_processors_per_pod=rp_per_pod,
        dataset_count=10000,
        avg_isl_tokens=isl,
        avg_osl_tokens=osl,
        max_turns=1,
        streaming=streaming,
        num_endpoints=max(1, pods // 25),
        connections_per_worker=500,
        num_gpus=num_gpus,
        gpu_sample_interval_s=1.0,
        num_gpu_metrics=12,
        num_server_metrics_endpoints=num_server_metrics_endpoints,
        server_metrics_scrape_interval_s=5.0,
        est_unique_metric_series=500,
        est_histogram_metrics=50,
        est_histogram_buckets=15,
        num_models=num_models,
        num_standard_metrics=25,
        export_http_trace=False,
    )


def run_comparison_table(target_concurrency: int) -> None:
    workers = max(10, target_concurrency // 100)
    pods = max(1, workers // 10)

    scenarios = [
        ("SSE ISL=128  OSL=32", 128, 32, True),
        ("SSE ISL=512  OSL=128", 512, 128, True),
        ("SSE ISL=2048 OSL=512", 2048, 512, True),
        ("SSE ISL=4096 OSL=2048", 4096, 2048, True),
        ("txt ISL=512  OSL=128", 512, 128, False),
        ("txt ISL=2048 OSL=512", 2048, 512, False),
    ]

    reqs = target_concurrency * 4
    dur = 7200

    print(
        f"Resource Estimates: concurrency={target_concurrency:,}  workers={workers:,}  pods={pods}"
    )
    print(
        f"Requests={reqs:,} (4x concurrency)  duration={dur}s ({dur // 3600}h{(dur % 3600) // 60}m)"
    )
    print("=" * 105)
    print(
        f"{'Scenario':<25} {'W/pod':>7} {'RP/pod':>7} {'Ctrl':>7} "
        f"{'Cluster':>10} {'RM':>7} {'Headroom':>9} {'Warnings':>10}"
    )
    print("-" * 105)

    for label, isl, osl, streaming in scenarios:
        params = _make_params(target_concurrency, isl, osl, streaming, workers)
        est = MemoryEstimator(params).estimate()
        rm = next(c for c in est.controller.components if c.name == "RecordsManager")
        rp = next(c for c in est.worker_pod.components if "RecordProcessor" in c.name)
        headroom = min(est.controller.headroom_pct, est.worker_pod.headroom_pct)
        n_warn = len(est.warnings)
        print(
            f"{label:<25} "
            f"{est.worker_pod.total_steady_state_mib:>6.0f}M "
            f"{rp.steady_state_mib:>6.0f}M "
            f"{est.controller.total_steady_state_mib:>6.0f}M "
            f"{est.total_cluster_mib:>9.0f}M "
            f"{rm.steady_state_mib:>6.0f}M "
            f"{headroom:>8.0f}% "
            f"{n_warn:>9}"
        )

    print()


def run_detailed(target_concurrency: int, isl: int, osl: int, streaming: bool) -> None:
    workers = max(10, target_concurrency // 100)
    params = _make_params(target_concurrency, isl, osl, streaming, workers)
    est = MemoryEstimator(params).estimate()
    print(format_estimate(est))


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate AIPerf deployment resources")
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=None,
        help="Target concurrency (default: run standard tiers)",
    )
    parser.add_argument("--isl", type=int, default=512, help="Input sequence length")
    parser.add_argument("--osl", type=int, default=128, help="Output sequence length")
    parser.add_argument(
        "--no-streaming", action="store_true", help="Non-streaming mode"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed breakdown"
    )
    args = parser.parse_args()

    if args.concurrency:
        if args.detailed:
            run_detailed(args.concurrency, args.isl, args.osl, not args.no_streaming)
        else:
            run_comparison_table(args.concurrency)
    else:
        for conc in [1_000, 10_000, 100_000, 500_000, 1_000_000]:
            run_comparison_table(conc)
            print()


if __name__ == "__main__":
    main()
