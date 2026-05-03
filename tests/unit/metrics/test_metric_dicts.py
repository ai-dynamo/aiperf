# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MetricResultsDict."""

from __future__ import annotations

import pytest

from aiperf.common.enums import MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric


class TestObservationDuration:
    """Tests for MetricResultsDict.observation_duration()."""

    def test_uses_window_bounds_when_set(self) -> None:
        """Window bounds take priority over BenchmarkDurationMetric."""
        d = MetricResultsDict()
        d.window_start_ns = 0
        d.window_end_ns = 5_000_000_000  # 5 seconds
        d[BenchmarkDurationMetric.tag] = 2_000_000_000  # 2 seconds (ignored)

        result = d.observation_duration(MetricTimeUnit.SECONDS)
        assert result == pytest.approx(5.0)

    def test_falls_back_to_benchmark_duration(self) -> None:
        """Without window bounds, falls back to BenchmarkDurationMetric."""
        d = MetricResultsDict()
        d[BenchmarkDurationMetric.tag] = 3_000_000_000  # 3 seconds

        result = d.observation_duration(MetricTimeUnit.SECONDS)
        assert result == pytest.approx(3.0)

    def test_converts_to_milliseconds(self) -> None:
        """Window duration correctly converts to target unit."""
        d = MetricResultsDict()
        d.window_start_ns = 0
        d.window_end_ns = 2_000_000_000  # 2 seconds

        result = d.observation_duration(MetricTimeUnit.MILLISECONDS)
        assert result == pytest.approx(2000.0)

    def test_zero_window_duration_raises(self) -> None:
        """Zero-length window raises NoMetricValue."""
        d = MetricResultsDict()
        d.window_start_ns = 1_000_000_000
        d.window_end_ns = 1_000_000_000

        with pytest.raises(NoMetricValue):
            d.observation_duration(MetricTimeUnit.SECONDS)

    def test_zero_benchmark_duration_raises(self) -> None:
        """Zero BenchmarkDurationMetric raises NoMetricValue."""
        d = MetricResultsDict()
        d[BenchmarkDurationMetric.tag] = 0

        with pytest.raises(NoMetricValue):
            d.observation_duration(MetricTimeUnit.SECONDS)

    def test_missing_benchmark_duration_raises(self) -> None:
        """Missing BenchmarkDurationMetric without window bounds raises NoMetricValue."""
        d = MetricResultsDict()

        with pytest.raises(NoMetricValue):
            d.observation_duration(MetricTimeUnit.SECONDS)

    def test_partial_window_bounds_ignored(self) -> None:
        """Only start or only end set falls back to BenchmarkDurationMetric."""
        d = MetricResultsDict()
        d.window_start_ns = 0
        # window_end_ns not set
        d[BenchmarkDurationMetric.tag] = 4_000_000_000

        result = d.observation_duration(MetricTimeUnit.SECONDS)
        assert result == pytest.approx(4.0)

    def test_window_bounds_default_to_none(self) -> None:
        """New MetricResultsDict has None window bounds."""
        d = MetricResultsDict()
        assert d.window_start_ns is None
        assert d.window_end_ns is None
