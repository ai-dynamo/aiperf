# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``TDigestListMetricAggregator``.

The aggregator is the run-level storage for list-valued record metrics
(today only ``inter_chunk_latency``). It backs percentile reads with a
t-digest sketch but keeps ``count`` / ``sum`` / ``min`` / ``max`` /
``avg`` / ``std`` bit-exact via running side-channel scalars (``std``
via Welford's online algorithm for numerical stability).
"""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.common.environment import Environment
from aiperf.metrics.list_metric_aggregation import TDigestListMetricAggregator
from aiperf.metrics.metric_dicts import MetricArray, MetricSeriesProtocol


class TestTDigestListMetricAggregator:
    """Behavioral contract for the aggregator."""

    def test_empty_aggregator_returns_count_zero_result(self) -> None:
        agg = TDigestListMetricAggregator()
        result = agg.to_result(tag="test", header="Test", unit="ms")
        assert result.tag == "test"
        assert result.header == "Test"
        assert result.unit == "ms"
        assert result.count == 0
        assert result.sum is None
        assert result.min is None
        assert result.max is None
        assert result.avg is None
        assert result.std is None
        assert result.p50 is None
        assert result.p99 is None

    def test_append_single_value_count_one(self) -> None:
        agg = TDigestListMetricAggregator()
        agg.append(7.0)
        result = agg.to_result(tag="t", header="T", unit="ms")
        assert result.count == 1
        assert result.sum == pytest.approx(7.0)
        assert result.min == pytest.approx(7.0)
        assert result.max == pytest.approx(7.0)
        assert result.avg == pytest.approx(7.0)
        assert result.std == pytest.approx(0.0)
        assert result.p50 == pytest.approx(7.0)

    def test_extend_with_list_count_matches_len(self) -> None:
        agg = TDigestListMetricAggregator()
        agg.extend([1.0, 2.0, 3.0, 4.0, 5.0])
        result = agg.to_result(tag="t", header="T", unit="ms")
        assert result.count == 5
        assert result.sum == pytest.approx(15.0)

    def test_repeated_extend_accumulates(self) -> None:
        agg = TDigestListMetricAggregator()
        agg.extend([1.0, 2.0, 3.0])
        agg.extend([4.0, 5.0])
        result = agg.to_result(tag="t", header="T", unit="ms")
        assert result.count == 5
        assert result.sum == pytest.approx(15.0)
        assert result.min == pytest.approx(1.0)
        assert result.max == pytest.approx(5.0)

    def test_mixed_int_and_float_values(self) -> None:
        agg = TDigestListMetricAggregator()
        agg.append(1)
        agg.append(2.5)
        agg.extend([3, 4.5])
        result = agg.to_result(tag="t", header="T", unit="ms")
        assert result.count == 4
        assert result.sum == pytest.approx(11.0)

    def test_min_max_exact_across_random_inputs(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.uniform(low=-1000.0, high=1000.0, size=10_000)
        agg = TDigestListMetricAggregator()
        agg.extend(values.tolist())
        result = agg.to_result(tag="t", header="T", unit="ms")
        assert result.min == float(values.min())
        assert result.max == float(values.max())

    def test_count_sum_exact_across_random_inputs(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.uniform(low=0.0, high=1000.0, size=10_000)
        agg = TDigestListMetricAggregator()
        agg.extend(values.tolist())
        result = agg.to_result(tag="t", header="T", unit="ms")
        assert result.count == 10_000
        assert result.sum == pytest.approx(float(values.sum()), rel=1e-12)

    def test_avg_std_exact_against_numpy(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(loc=100.0, scale=15.0, size=10_000)
        agg = TDigestListMetricAggregator()
        agg.extend(values.tolist())
        result = agg.to_result(tag="t", header="T", unit="ms")
        assert result.avg == pytest.approx(float(np.mean(values)), rel=1e-9)
        assert result.std == pytest.approx(float(np.std(values)), rel=1e-9)

    def test_percentiles_within_tolerance(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.uniform(low=0.0, high=1000.0, size=100_000)
        agg = TDigestListMetricAggregator()
        agg.extend(values.tolist())
        result = agg.to_result(tag="t", header="T", unit="ms")
        assert result.p50 == pytest.approx(float(np.percentile(values, 50)), rel=0.005)
        assert result.p90 == pytest.approx(float(np.percentile(values, 90)), rel=0.005)
        assert result.p99 == pytest.approx(float(np.percentile(values, 99)), rel=0.005)

    def test_to_result_schema_matches_metric_array(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.uniform(low=0.0, high=1000.0, size=100_000)

        digest_agg = TDigestListMetricAggregator()
        digest_agg.extend(values.tolist())
        digest_result = digest_agg.to_result(tag="t", header="T", unit="ms")

        array_agg = MetricArray()
        array_agg.extend(values.tolist())
        array_result = array_agg.to_result(tag="t", header="T", unit="ms")

        # Both paths populate the same stat fields (dataclass schema parity).
        populated_fields = (
            "tag",
            "header",
            "unit",
            "count",
            "sum",
            "min",
            "max",
            "avg",
            "std",
            "p1",
            "p5",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
            "p99",
        )
        for f in populated_fields:
            assert getattr(digest_result, f) is not None, f
            assert getattr(array_result, f) is not None, f
        assert digest_result.count == array_result.count
        assert digest_result.min == pytest.approx(array_result.min)
        assert digest_result.max == pytest.approx(array_result.max)
        assert digest_result.sum == pytest.approx(array_result.sum, rel=1e-12)
        assert digest_result.avg == pytest.approx(array_result.avg, rel=1e-9)
        assert digest_result.std == pytest.approx(array_result.std, rel=1e-9)
        for pct_field in ("p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"):
            assert getattr(digest_result, pct_field) == pytest.approx(
                getattr(array_result, pct_field), rel=0.005
            )

    def test_extend_batched_matches_per_element_appends(self) -> None:
        """Single ``extend(list)`` (numpy batched C-level update) must give
        the same exact stats as N successive ``append(v)`` calls. Regression
        boundary for the batched code path.
        """
        rng = np.random.default_rng(42)
        values = rng.normal(loc=100.0, scale=15.0, size=10_000)

        agg_batched = TDigestListMetricAggregator()
        agg_batched.extend(values.tolist())
        r_batched = agg_batched.to_result(tag="t", header="T", unit="ms")

        agg_streamed = TDigestListMetricAggregator()
        for v in values:
            agg_streamed.append(float(v))
        r_streamed = agg_streamed.to_result(tag="t", header="T", unit="ms")

        assert r_batched.count == r_streamed.count
        assert r_batched.min == pytest.approx(r_streamed.min)
        assert r_batched.max == pytest.approx(r_streamed.max)
        assert r_batched.sum == pytest.approx(r_streamed.sum, rel=1e-12)
        assert r_batched.avg == pytest.approx(r_streamed.avg, rel=1e-12)
        assert r_batched.std == pytest.approx(r_streamed.std, rel=1e-9)

    def test_welford_std_is_stable_on_large_offset_distribution(self) -> None:
        """The textbook ``sum_sq/count - avg^2`` formula collapses to zero
        for large-offset, low-spread data because of catastrophic
        cancellation. Welford's algorithm preserves precision.
        """
        rng = np.random.default_rng(42)
        values = 1.0e9 + rng.normal(loc=0.0, scale=1.0, size=10_000)
        agg = TDigestListMetricAggregator()
        agg.extend(values.tolist())
        result = agg.to_result(tag="t", header="T", unit="ns")
        assert result.std == pytest.approx(float(np.std(values)), rel=1e-3)

    def test_protocol_runtime_isinstance(self) -> None:
        """Aggregator should satisfy ``MetricSeriesProtocol`` so
        ``isinstance`` dispatch in ``MetricResultsProcessor`` and
        ``DerivedSumMetric`` accepts both this and ``MetricArray``."""
        digest_agg = TDigestListMetricAggregator()
        array_agg = MetricArray()
        assert isinstance(digest_agg, MetricSeriesProtocol)
        assert isinstance(array_agg, MetricSeriesProtocol)

    def test_compression_env_var_flows_to_underlying_sketch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``AIPERF_METRICS_TDIGEST_COMPRESSION`` must be wired through to
        ``crick.TDigest`` so operators can tune accuracy/memory without code
        changes."""
        monkeypatch.setenv("AIPERF_METRICS_TDIGEST_COMPRESSION", "200")
        from aiperf.common.environment import _MetricsSettings

        monkeypatch.setattr(Environment, "METRICS", _MetricsSettings(), raising=True)
        agg = TDigestListMetricAggregator()
        assert agg._td.compression == 200

    def test_sum_property_for_derived_metric_protocol(self) -> None:
        """``MetricSeriesProtocol`` requires a ``sum`` property so
        :class:`DerivedSumMetric` can compute uniformly across this and
        :class:`MetricArray`."""
        agg = TDigestListMetricAggregator()
        agg.extend([1.0, 2.0, 3.0, 4.0, 5.0])
        assert agg.sum == pytest.approx(15.0)
