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
from aiperf.metrics.metric_dicts import MetricAggregator, MetricArray


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
        # avg = sum / count; std = sqrt(M2 / count) via Welford.
        # Both within float64 round-off of numpy's reference.
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

        # Same set of non-None fields on the dataclass.
        import dataclasses as _dc

        digest_set = {
            f.name
            for f in _dc.fields(digest_result)
            if getattr(digest_result, f.name) is not None
        }
        array_set = {
            f.name
            for f in _dc.fields(array_result)
            if getattr(array_result, f.name) is not None
        }
        assert digest_set == array_set
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

    def test_extend_batched_matches_per_element_appends(self) -> None:
        """Single ``extend(list)`` (numpy batched C-level update) must give
        the same exact stats as N successive ``append(v)`` calls. This is the
        regression boundary for the new batched code path.
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

        # Exact stats (count, sum, min, max, avg, std) must agree to
        # float64 round-off across the two paths.
        assert r_batched.count == r_streamed.count
        assert r_batched.min == pytest.approx(r_streamed.min)
        assert r_batched.max == pytest.approx(r_streamed.max)
        assert r_batched.sum == pytest.approx(r_streamed.sum, rel=1e-12)
        assert r_batched.avg == pytest.approx(r_streamed.avg, rel=1e-12)
        assert r_batched.std == pytest.approx(r_streamed.std, rel=1e-9)

    def test_batched_std_welford_parallel_combine_matches_numpy(self) -> None:
        """Multi-batch ``extend`` must combine per-batch M2 via Welford's
        parallel-combine term (list_metric_aggregation.py:117:
        ``self._m2 += m2_b + delta * delta * n_a * n_b / new_count``).

        The existing single-``extend`` std tests only hit the ``n_a == 0``
        first-batch branch (lines 110-112); ``test_repeated_extend_accumulates``
        crosses the combine branch but only asserts count/sum/min/max -- never
        std. So the mean-difference term on line 117 was untested.

        Two batches with DIFFERING means AND sizes make the cross term
        (``delta**2 * n_a * n_b / new_count``) dominant. Dropping it (the
        mutation) collapses std from the true ~149.0 to ~84.5 -- caught by the
        1e-9 tolerance against numpy's population std.
        """
        batch_a = [1.0, 2.0, 3.0]  # mean 2, n 3
        batch_b = [100.0, 200.0, 300.0, 400.0]  # mean 250, n 4 -- large offset
        agg = TDigestListMetricAggregator()
        agg.extend(batch_a)
        agg.extend(batch_b)
        result = agg.to_result(tag="t", header="T", unit="ms")

        all_values = np.array(batch_a + batch_b)
        assert result.count == 7
        assert result.std == pytest.approx(float(np.std(all_values, ddof=0)), rel=1e-9)

    def test_multibatch_std_matches_single_extend_and_numpy(self) -> None:
        """Three batches of differing means/sizes must give the same std as a
        single ``extend`` of the concatenation and as numpy's population std.

        A second, independent pin on the line-117 parallel-combine term: any
        wrong batch merge (dropped cross term, wrong ``n_a``/``n_b`` weighting)
        makes the incremental path diverge from the whole-array reference.
        """
        rng = np.random.default_rng(7)
        b1 = rng.normal(loc=0.0, scale=1.0, size=500)
        b2 = rng.normal(loc=1000.0, scale=5.0, size=1500)
        b3 = rng.normal(loc=-50.0, scale=20.0, size=800)

        incremental = TDigestListMetricAggregator()
        incremental.extend(b1.tolist())
        incremental.extend(b2.tolist())
        incremental.extend(b3.tolist())
        r_incremental = incremental.to_result(tag="t", header="T", unit="ms")

        all_values = np.concatenate([b1, b2, b3])
        single = TDigestListMetricAggregator()
        single.extend(all_values.tolist())
        r_single = single.to_result(tag="t", header="T", unit="ms")

        assert r_incremental.count == all_values.size
        assert r_incremental.std == pytest.approx(float(np.std(all_values)), rel=1e-9)
        assert r_incremental.std == pytest.approx(r_single.std, rel=1e-9)

    def test_welford_std_is_stable_on_large_offset_distribution(self) -> None:
        """The textbook ``sum_sq/count - avg^2`` formula collapses to zero
        for large-offset, low-spread data because of catastrophic
        cancellation. Welford's algorithm preserves precision.
        """
        # Mean ~1e9 (e.g. wall-clock ns timestamps), spread ~1.0 — exactly
        # the regime where the textbook formula loses ~9 of float64's 16
        # decimal digits.
        rng = np.random.default_rng(42)
        values = 1.0e9 + rng.normal(loc=0.0, scale=1.0, size=10_000)
        agg = TDigestListMetricAggregator()
        agg.extend(values.tolist())
        result = agg.to_result(tag="t", header="T", unit="ns")
        # Welford std should agree with numpy to better than 0.1% even
        # at this offset/spread ratio. The textbook formula would round
        # to ~0 here.
        assert result.std == pytest.approx(float(np.std(values)), rel=1e-3)

    def test_protocol_runtime_isinstance(self) -> None:
        """Aggregator should satisfy the ``MetricAggregator`` protocol so
        ``isinstance`` dispatch in ``MetricResultsProcessor`` and
        ``DerivedSumMetric`` accepts both this and ``MetricArray``."""
        digest_agg = TDigestListMetricAggregator()
        array_agg = MetricArray()
        assert isinstance(digest_agg, MetricAggregator)
        assert isinstance(array_agg, MetricAggregator)

    def test_compression_env_var_flows_to_underlying_sketch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``AIPERF_METRICS_TDIGEST_COMPRESSION`` must be wired through to
        ``crick.TDigest`` so operators can tune accuracy/memory without code
        changes."""
        monkeypatch.setenv("AIPERF_METRICS_TDIGEST_COMPRESSION", "200")
        # The compression knob is read in __init__; reset the cached settings
        # so the env var takes effect for this test.
        from aiperf.common.environment import _MetricsSettings

        monkeypatch.setattr(Environment, "METRICS", _MetricsSettings(), raising=True)
        agg = TDigestListMetricAggregator()
        assert agg._td.compression == 200

    def test_sum_property_for_derived_metric_protocol(self) -> None:
        """The ``MetricAggregator`` protocol requires a ``sum`` property so
        :class:`DerivedSumMetric` can compute uniformly across this and
        :class:`MetricArray`."""
        agg = TDigestListMetricAggregator()
        agg.extend([1.0, 2.0, 3.0, 4.0, 5.0])
        # Property is exposed and returns the running side-channel sum.
        assert agg.sum == pytest.approx(15.0)

    def test_add_for_record_dedups_redelivery(self) -> None:
        """``add_for_record`` is first-wins on re-delivery to the same ``idx``.

        The t-digest has no value-removal op, so re-delivery cannot be last-
        wins; first-wins keeps it idempotent, matching :class:`RaggedSeries`
        and the numeric column-store handler. A 3-chunk record delivered twice
        contributes its samples EXACTLY once (count 3 / sum 60, not 6 / 120).
        """
        agg = TDigestListMetricAggregator()
        agg.add_for_record(0, [10.0, 20.0, 30.0])
        agg.add_for_record(0, [10.0, 20.0, 30.0])  # re-delivery of the same record
        result = agg.to_result(tag="inter_chunk_latency", header="ICL", unit="ms")
        assert result.count == 3
        assert result.sum == pytest.approx(60.0)

    def test_add_for_record_distinct_idx_accumulate_and_grow(self) -> None:
        """Distinct records accumulate; the dedup bitmap grows past its cap.

        ``idx=1000`` exceeds the initial 256-slot ``_seen`` bitmap, exercising
        ``_grow_seen`` before the re-delivery skip fires.
        """
        agg = TDigestListMetricAggregator()
        agg.add_for_record(0, [1.0])
        agg.add_for_record(1000, [2.0])  # idx beyond initial _seen cap -> grow
        agg.add_for_record(1000, [2.0])  # re-delivery after grow -> skipped
        assert len(agg) == 2
        assert agg.sum == pytest.approx(3.0)
