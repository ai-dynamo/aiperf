# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``TDigestListMetricAggregator``.

The aggregator is the run-level storage for list-valued record metrics
(today only ``inter_chunk_latency``). It backs percentile reads with a
t-digest sketch but keeps ``count`` / ``sum`` / ``min`` / ``max`` /
``avg`` / ``std`` bit-exact via running side-channel scalars.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.metrics.list_metric_aggregation import TDigestListMetricAggregator
from aiperf.metrics.metric_dicts import MetricArray


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
        # Percentiles of a single point all collapse to that point.
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
        agg.append(1)  # int
        agg.append(2.5)  # float
        agg.extend([3, 4.5])  # mixed list
        result = agg.to_result(tag="t", header="T", unit="ms")
        assert result.count == 4
        assert result.sum == pytest.approx(11.0)

    def test_min_max_exact_across_random_inputs(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.uniform(low=-1000.0, high=1000.0, size=10_000)
        agg = TDigestListMetricAggregator()
        agg.extend(values.tolist())
        result = agg.to_result(tag="t", header="T", unit="ms")
        # Bit-exact min/max via running side-channel.
        assert result.min == float(values.min())
        assert result.max == float(values.max())

    def test_count_sum_exact_across_random_inputs(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.uniform(low=0.0, high=1000.0, size=10_000)
        agg = TDigestListMetricAggregator()
        agg.extend(values.tolist())
        result = agg.to_result(tag="t", header="T", unit="ms")
        assert result.count == 10_000
        # Exact within float64 round-off (sum order matters slightly).
        assert result.sum == pytest.approx(float(values.sum()), rel=1e-12)

    def test_avg_std_exact_against_numpy(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(loc=100.0, scale=15.0, size=10_000)
        agg = TDigestListMetricAggregator()
        agg.extend(values.tolist())
        result = agg.to_result(tag="t", header="T", unit="ms")
        # avg = sum / count; std = sqrt(max(0, sum_sq/count - avg^2))
        # both within float64 round-off of numpy's reference.
        assert result.avg == pytest.approx(float(np.mean(values)), rel=1e-9)
        assert result.std == pytest.approx(float(np.std(values)), rel=1e-9)

    def test_percentiles_within_tolerance(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.uniform(low=0.0, high=1000.0, size=100_000)
        agg = TDigestListMetricAggregator()
        agg.extend(values.tolist())
        result = agg.to_result(tag="t", header="T", unit="ms")
        # T-digest's documented relative error band on percentiles is
        # well under 1% at this sample size; we hold ourselves to 0.5%.
        assert result.p50 == pytest.approx(float(np.percentile(values, 50)), rel=0.005)
        assert result.p90 == pytest.approx(float(np.percentile(values, 90)), rel=0.005)
        assert result.p99 == pytest.approx(float(np.percentile(values, 99)), rel=0.005)

    def test_to_result_schema_matches_metric_array(self) -> None:
        # 100k samples — t-digest's relative error on percentile *values*
        # at extreme tails (e.g. p1 on uniform[0, 1000] where the true
        # value is ~9.66, small relative to the data range) only stays
        # within 0.5% at this sample size. Smaller N exhibits 1% relative
        # error at p1 even though rank accuracy is well within the
        # documented t-digest bound.
        rng = np.random.default_rng(42)
        values = rng.uniform(low=0.0, high=1000.0, size=100_000)

        digest_agg = TDigestListMetricAggregator()
        digest_agg.extend(values.tolist())
        digest_result = digest_agg.to_result(tag="t", header="T", unit="ms")

        array_agg = MetricArray()
        array_agg.extend(values.tolist())
        array_result = array_agg.to_result(tag="t", header="T", unit="ms")

        # Same field set on the Pydantic model.
        assert set(digest_result.model_fields_set) == set(array_result.model_fields_set)
        # Bit-exact stats.
        assert digest_result.count == array_result.count
        assert digest_result.min == pytest.approx(array_result.min)
        assert digest_result.max == pytest.approx(array_result.max)
        assert digest_result.sum == pytest.approx(array_result.sum, rel=1e-12)
        assert digest_result.avg == pytest.approx(array_result.avg, rel=1e-9)
        assert digest_result.std == pytest.approx(array_result.std, rel=1e-9)
        # Approximate percentiles within t-digest tolerance.
        for pct_field in ("p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"):
            assert getattr(digest_result, pct_field) == pytest.approx(
                getattr(array_result, pct_field), rel=0.005
            )
