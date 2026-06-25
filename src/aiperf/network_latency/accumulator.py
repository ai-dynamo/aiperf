# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict

import numpy as np

from aiperf.common.models import (
    ErrorDetails,
    ErrorDetailsCount,
    NetworkLatencyResults,
    NetworkLatencySample,
    NetworkLatencyTargetSummary,
)

__all__ = ["NetworkLatencyAccumulator"]


def _percentile_stats(rtts_ns: list[int]) -> dict[str, float | None]:
    """Compute min/mean/median/p90/p99/stddev (ns) over successful RTT samples."""
    if not rtts_ns:
        return {
            "min_ns": None,
            "mean_ns": None,
            "median_ns": None,
            "p90_ns": None,
            "p99_ns": None,
            "stddev_ns": None,
        }
    arr = np.asarray(rtts_ns, dtype=np.float64)
    return {
        "min_ns": float(np.min(arr)),
        "mean_ns": float(np.mean(arr)),
        "median_ns": float(np.median(arr)),
        "p90_ns": float(np.percentile(arr, 90)),
        "p99_ns": float(np.percentile(arr, 99)),
        "stddev_ns": float(np.std(arr)),
    }


class NetworkLatencyAccumulator:
    """Accumulates RTT probe samples and computes per-target + aggregate stats.

    Instantiated directly by the RecordsManager (not a plugin results processor)
    because the aggregate ``mean_ns`` must be delivered to every
    MetricResultsProcessor via ``set_network_rtt_ns`` before ``summarize()`` is
    called, rather than flowing through the standard summarize pipeline.
    """

    def __init__(self, benchmark_id: str | None = None) -> None:
        self._benchmark_id = benchmark_id
        self._success_rtts_by_target: dict[str, list[int]] = defaultdict(list)
        self._counts_by_target: dict[str, int] = defaultdict(int)
        self._failures_by_target: dict[str, int] = defaultdict(int)
        self._target_meta: dict[str, tuple[str, str, int]] = {}
        self._error_counts: dict[ErrorDetails, int] = defaultdict(int)

    def add_sample(self, sample: NetworkLatencySample) -> None:
        """Accumulate one probe sample (success or failure)."""
        key = f"{sample.target_host}:{sample.target_port}"
        self._target_meta[key] = (
            sample.target_url,
            sample.target_host,
            sample.target_port,
        )
        self._counts_by_target[key] += 1
        if sample.success and sample.rtt_ns is not None:
            self._success_rtts_by_target[key].append(sample.rtt_ns)
        else:
            self._failures_by_target[key] += 1
            if sample.error is not None:
                self._error_counts[sample.error] += 1

    @property
    def mean_rtt_ns(self) -> float | None:
        """Mean RTT (ns) over all successful samples across all targets, or None."""
        all_rtts = [
            rtt for rtts in self._success_rtts_by_target.values() for rtt in rtts
        ]
        if not all_rtts:
            return None
        return float(np.mean(np.asarray(all_rtts, dtype=np.float64)))

    @property
    def successful_sample_count(self) -> int:
        """Number of successful RTT samples across all targets."""
        return sum(len(rtts) for rtts in self._success_rtts_by_target.values())

    def export_results(self) -> NetworkLatencyResults:
        """Compute the final per-target and aggregate RTT results."""
        target_summaries: dict[str, NetworkLatencyTargetSummary] = {}
        for key, (target_url, host, port) in self._target_meta.items():
            success_rtts = self._success_rtts_by_target.get(key, [])
            stats = _percentile_stats(success_rtts)
            target_summaries[key] = NetworkLatencyTargetSummary(
                target_url=target_url,
                target_host=host,
                target_port=port,
                count=self._counts_by_target.get(key, 0),
                success_count=len(success_rtts),
                failure_count=self._failures_by_target.get(key, 0),
                **stats,
            )

        all_rtts = [
            rtt for rtts in self._success_rtts_by_target.values() for rtt in rtts
        ]
        aggregate_stats = _percentile_stats(all_rtts)
        total_count = sum(self._counts_by_target.values())
        total_failures = sum(self._failures_by_target.values())

        error_summary = [
            ErrorDetailsCount(error_details=error_details, count=count)
            for error_details, count in self._error_counts.items()
        ]

        return NetworkLatencyResults(
            benchmark_id=self._benchmark_id,
            target_summaries=target_summaries,
            count=total_count,
            success_count=len(all_rtts),
            failure_count=total_failures,
            error_summary=error_summary,
            **aggregate_stats,
        )
