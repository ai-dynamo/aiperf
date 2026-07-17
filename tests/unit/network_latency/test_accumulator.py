# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NetworkLatencyAccumulator stat aggregation.

Successful samples feed the percentile stats; failed samples are excluded from
the RTT distribution but counted as failures. With zero successful samples the
mean is None (the metric processor then applies no adjustment).
"""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.common.models import ErrorDetails, NetworkLatencySample
from aiperf.network_latency.accumulator import NetworkLatencyAccumulator

_TARGET_URL = "http://localhost:8000/v1/chat/completions"
_HOST = "localhost"
_PORT = 8000


def _success_sample(rtt_ns: int) -> NetworkLatencySample:
    return NetworkLatencySample(
        timestamp_ns=1_000,
        target_url=_TARGET_URL,
        target_host=_HOST,
        target_port=_PORT,
        rtt_ns=rtt_ns,
        success=True,
    )


def _failure_sample() -> NetworkLatencySample:
    return NetworkLatencySample(
        timestamp_ns=2_000,
        target_url=_TARGET_URL,
        target_host=_HOST,
        target_port=_PORT,
        rtt_ns=None,
        success=False,
        error=ErrorDetails(type="ConnectionRefusedError", message="refused"),
    )


# Known RTT sequence (ns): mean=550, median=550, etc.
_RTTS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


class TestMeanRttNs:
    def test_mean_over_successful_samples(self) -> None:
        acc = NetworkLatencyAccumulator()
        for rtt in _RTTS:
            acc.add_sample(_success_sample(rtt))

        assert acc.mean_rtt_ns == pytest.approx(float(np.mean(_RTTS)))

    def test_failed_samples_excluded_from_mean(self) -> None:
        acc = NetworkLatencyAccumulator()
        for rtt in _RTTS:
            acc.add_sample(_success_sample(rtt))
        for _ in range(5):
            acc.add_sample(_failure_sample())

        assert acc.mean_rtt_ns == pytest.approx(float(np.mean(_RTTS)))

    def test_zero_successful_samples_mean_is_none(self) -> None:
        acc = NetworkLatencyAccumulator()
        acc.add_sample(_failure_sample())
        acc.add_sample(_failure_sample())

        assert acc.mean_rtt_ns is None

    def test_empty_accumulator_mean_is_none(self) -> None:
        assert NetworkLatencyAccumulator().mean_rtt_ns is None


class TestExportResults:
    def test_aggregate_stats_match_numpy(self) -> None:
        acc = NetworkLatencyAccumulator()
        for rtt in _RTTS:
            acc.add_sample(_success_sample(rtt))

        results = acc.export_results()
        arr = np.asarray(_RTTS, dtype=np.float64)

        assert results.min_ns == pytest.approx(float(np.min(arr)))
        assert results.mean_ns == pytest.approx(float(np.mean(arr)))
        assert results.median_ns == pytest.approx(float(np.median(arr)))
        assert results.p90_ns == pytest.approx(float(np.percentile(arr, 90)))
        assert results.p99_ns == pytest.approx(float(np.percentile(arr, 99)))
        assert results.stddev_ns == pytest.approx(float(np.std(arr)))

    def test_counts_split_success_and_failure(self) -> None:
        acc = NetworkLatencyAccumulator()
        for rtt in _RTTS:
            acc.add_sample(_success_sample(rtt))
        for _ in range(3):
            acc.add_sample(_failure_sample())

        results = acc.export_results()

        assert results.count == len(_RTTS) + 3
        assert results.success_count == len(_RTTS)
        assert results.failure_count == 3

    def test_failed_samples_excluded_from_distribution(self) -> None:
        acc = NetworkLatencyAccumulator()
        for rtt in _RTTS:
            acc.add_sample(_success_sample(rtt))
        for _ in range(3):
            acc.add_sample(_failure_sample())

        results = acc.export_results()
        arr = np.asarray(_RTTS, dtype=np.float64)

        assert results.mean_ns == pytest.approx(float(np.mean(arr)))
        assert results.min_ns == pytest.approx(float(np.min(arr)))

    def test_zero_successful_samples_stats_are_none(self) -> None:
        acc = NetworkLatencyAccumulator()
        acc.add_sample(_failure_sample())

        results = acc.export_results()

        assert results.mean_ns is None
        assert results.min_ns is None
        assert results.median_ns is None
        assert results.p90_ns is None
        assert results.p99_ns is None
        assert results.stddev_ns is None
        assert results.success_count == 0
        assert results.failure_count == 1

    def test_error_summary_counts_unique_errors(self) -> None:
        acc = NetworkLatencyAccumulator()
        for _ in range(4):
            acc.add_sample(_failure_sample())

        results = acc.export_results()

        assert len(results.error_summary) == 1
        assert results.error_summary[0].count == 4

    def test_benchmark_id_propagated(self) -> None:
        acc = NetworkLatencyAccumulator(benchmark_id="bench-123")
        acc.add_sample(_success_sample(100))

        assert acc.export_results().benchmark_id == "bench-123"

    def test_per_target_summary_keyed_by_host_port(self) -> None:
        acc = NetworkLatencyAccumulator()
        for rtt in _RTTS:
            acc.add_sample(_success_sample(rtt))

        results = acc.export_results()

        key = f"{_HOST}:{_PORT}"
        assert key in results.target_summaries
        summary = results.target_summaries[key]
        assert summary.success_count == len(_RTTS)
        assert summary.mean_ns == pytest.approx(float(np.mean(_RTTS)))
