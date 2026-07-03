#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end accuracy validation of the K8s memory estimator against real
``kubectl top`` measurements from the 2026-04-30 ISL/OSL memory sweep
(embedded below as ``_FINDINGS``).

For each (concurrency, ISL, OSL) cell the sweep recorded, run
``MemoryEstimator`` with the same topology the sweep used (250
connections-per-worker, 10 workers-per-pod, 1 record-processor per pod)
and compare ``worker_pod.total_peak_mib`` and ``controller.total_peak_mib``
to the measured per-pod and controller-pod RSS.

Use this whenever:
- You change a constant in ``_memory_estimator/constants.py``.
- You add a new component to the controller or worker pod model.
- You re-run the sweep on a new image and want to regression-check the
  estimator's predictions before trusting them.

Caveat from the sweep itself: high-OSL cells (≥1024) saw 25-55% errors
during the sweep, so effective concurrency was below nominal — the
estimator's over-prediction at those cells is appropriate. We treat them
as "expected loose" rather than failures.

Usage:
    uv run python -m aiperf.analysis.validate_memory_estimator_against_findings
    uv run python -m aiperf.analysis.validate_memory_estimator_against_findings --max-error-ratio 1.5
"""

from __future__ import annotations

import argparse
import math
import statistics

from aiperf.kubernetes.memory_estimator import (
    MemoryEstimationParams,
    MemoryEstimator,
)

# Topology the 2026-04-30 sweep ran with.
_SWEEP_CONNECTIONS_PER_WORKER = 250
_SWEEP_WORKERS_PER_POD = 10
_SWEEP_RP_PER_POD = 1
_SWEEP_DURATION_S = 300.0

# (concurrency, isl, osl, requests, measured_max_pod_mib, measured_ctrl_mib).
# Source: 2026-04-30 ISL/OSL memory sweep (real-cluster kubectl top data).
# Update this table when re-running the sweep on a different image.
_FINDINGS = [
    (5000, 128, 128, 50_000, 2569, 1092),
    (5000, 128, 1024, 30_000, 3089, 1093),
    (5000, 1024, 128, 50_000, 2811, 1080),
    (5000, 1024, 1024, 30_000, 4098, 1100),
    (5000, 1024, 4096, 15_000, 2746, 1084),
    (5000, 4096, 1024, 30_000, 4770, 1114),
    (5000, 4096, 4096, 15_000, 4233, 1113),
    (10000, 128, 128, 100_000, 1871, 1141),
    (10000, 1024, 1024, 60_000, 2992, 1161),
    (10000, 4096, 4096, 20_000, 2196, 1130),
]

# Cells with >25% measured error rate per the findings caveats — the
# nominal concurrency wasn't actually held, so retries inflate or deflate
# RSS in ways the estimator can't model from nominal inputs. The findings
# doc lists cell 5 (54%), cell 6 (50%), cell 7 (184% retries-over-completed)
# at 5K concurrency, plus the 10K/4096/4096 cell.
_HIGH_ERROR_CELLS = {
    (5000, 1024, 4096),
    (5000, 4096, 1024),
    (5000, 4096, 4096),
    (10000, 4096, 4096),
}


def _build_params(conc: int, isl: int, osl: int, reqs: int) -> MemoryEstimationParams:
    workers = math.ceil(conc / _SWEEP_CONNECTIONS_PER_WORKER)
    pods = math.ceil(workers / _SWEEP_WORKERS_PER_POD)
    actual_wpp = min(workers, _SWEEP_WORKERS_PER_POD)
    return MemoryEstimationParams(
        total_workers=workers,
        workers_per_pod=actual_wpp,
        num_worker_pods=pods,
        record_processors_per_pod=_SWEEP_RP_PER_POD,
        max_concurrency=conc,
        total_requests=reqs,
        total_benchmark_duration_s=_SWEEP_DURATION_S,
        dataset_count=1000,
        avg_isl_tokens=isl,
        avg_osl_tokens=osl,
        max_turns=1,
        streaming=True,
        num_endpoints=1,
        connections_per_worker=_SWEEP_CONNECTIONS_PER_WORKER,
        num_gpus=0,
        gpu_sample_interval_s=1.0,
        num_gpu_metrics=12,
        num_server_metrics_endpoints=0,
        server_metrics_scrape_interval_s=5.0,
        est_unique_metric_series=200,
        est_histogram_metrics=20,
        est_histogram_buckets=10,
        num_models=1,
        num_standard_metrics=25,
        export_http_trace=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate K8s memory estimator predictions against measured findings"
    )
    parser.add_argument(
        "--max-error-ratio",
        type=float,
        default=1.5,
        help=(
            "Pass criterion: predicted/measured outside [1/X, X] for a low-error "
            "cell triggers a non-zero exit (default: 1.5)"
        ),
    )
    args = parser.parse_args()

    print("=" * 88)
    print("  K8s memory estimator vs measured findings")
    print(
        f"  Tolerance: predicted/measured ∈ [{1 / args.max_error_ratio:.2f}, {args.max_error_ratio:.2f}]"
    )
    print(
        "  Cells in HIGH_ERROR set are excluded from pass/fail (>25% sweep error rate)."
    )
    print("=" * 88)
    print()
    print(
        f"  {'conc':>6} {'isl':>5} {'osl':>5} | "
        f"{'WP pred':>8} {'WP meas':>8} {'WP r':>5} | "
        f"{'CT pred':>8} {'CT meas':>8} {'CT r':>5} | flags"
    )
    print("  " + "-" * 84)

    wp_ratios_low_err: list[float] = []
    ct_ratios: list[float] = []
    failed_cells: list[str] = []

    for conc, isl, osl, reqs, wp_meas, ct_meas in _FINDINGS:
        est = MemoryEstimator(_build_params(conc, isl, osl, reqs)).estimate()
        wp_pred = est.worker_pod.total_peak_mib
        ct_pred = est.controller.total_peak_mib
        wp_r = wp_pred / wp_meas
        ct_r = ct_pred / ct_meas
        ct_ratios.append(ct_r)

        flags = []
        is_high_err = (conc, isl, osl) in _HIGH_ERROR_CELLS
        if is_high_err:
            flags.append("high-err")
        else:
            wp_ratios_low_err.append(wp_r)
            if not (1 / args.max_error_ratio <= wp_r <= args.max_error_ratio):
                flags.append("WP-FAIL")
                failed_cells.append(f"{conc}c/ISL={isl}/OSL={osl} WP {wp_r:.2f}x")
        if not (1 / args.max_error_ratio <= ct_r <= args.max_error_ratio):
            flags.append("CT-FAIL")
            failed_cells.append(f"{conc}c/ISL={isl}/OSL={osl} CT {ct_r:.2f}x")

        print(
            f"  {conc:>6} {isl:>5} {osl:>5} | "
            f"{wp_pred:>7.0f}M {wp_meas:>7}M {wp_r:>4.2f}x | "
            f"{ct_pred:>7.0f}M {ct_meas:>7}M {ct_r:>4.2f}x | {','.join(flags) or 'ok'}"
        )

    def _summarize(name: str, ratios: list[float]) -> None:
        if not ratios:
            return
        geo = math.exp(statistics.mean(math.log(r) for r in ratios))
        log_err = statistics.mean(abs(math.log(r)) for r in ratios)
        print(
            f"  {name:<22} geo-mean={geo:>5.2f}x   |log| error={log_err:>5.3f}   n={len(ratios)}"
        )

    print()
    _summarize("WORKER POD (low-err)", wp_ratios_low_err)
    _summarize("CONTROLLER (all)", ct_ratios)

    if failed_cells:
        print()
        print(
            f"!! {len(failed_cells)} cell(s) outside tolerance — investigate "
            "constants in ``_memory_estimator/constants.py`` or model changes:"
        )
        for c in failed_cells:
            print(f"  - {c}")
        return 1

    print()
    print("All low-error cells within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
