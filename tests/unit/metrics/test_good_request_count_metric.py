# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import MetricTypeError
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.good_request_count_metric import GoodRequestCountMetric
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from tests.unit.metrics.conftest import create_record, run_simple_metrics_pipeline


class TestGoodRequestCountMetric:
    def setup_method(self):
        GoodRequestCountMetric.set_slos({})

    def test_unknown_tag_raises(self, monkeypatch):
        def mock_get_class(tag):
            raise MetricTypeError(f"Metric class with tag '{tag}' not found")

        monkeypatch.setattr(MetricRegistry, "get_class", mock_get_class)

        with pytest.raises(ValueError, match="Unknown metric tag"):
            GoodRequestCountMetric.set_slos({"does_not_exist": 123})

    def test_set_slos_populates_required_metrics(self):
        GoodRequestCountMetric.set_slos(
            {
                RequestLatencyMetric.tag: 250.0,
            }
        )
        assert GoodRequestCountMetric.required_metrics == {RequestLatencyMetric.tag}

    def test_set_slos_converts_display_to_native_units(self, monkeypatch):
        class MockLatencyMetric:
            tag = "mock_latency"
            unit = MetricTimeUnit.SECONDS  # native unit (s)
            display_unit = MetricTimeUnit.MILLISECONDS
            flags = MetricFlags.NONE

        monkeypatch.setattr(MetricRegistry, "get_class", lambda tag: MockLatencyMetric)
        GoodRequestCountMetric.set_slos({"mock_latency": 250})  # 250 ms

        # 250 ms -> 0.25 s stored in thresholds
        assert (
            pytest.approx(GoodRequestCountMetric._thresholds["mock_latency"], rel=1e-6)
            == 0.25
        )

    def test_counts_good_requests(self):
        GoodRequestCountMetric.set_slos({RequestLatencyMetric.tag: 250.0})

        records = [
            create_record(start_ns=0, responses=[100_000_000]),  # 100ms -> good
            create_record(
                start_ns=100_000_000, responses=[400_000_000]
            ),  # 300ms -> bad
            create_record(
                start_ns=200_000_000, responses=[450_000_000]
            ),  # 250ms -> good
        ]

        metrics = run_simple_metrics_pipeline(
            records,
            RequestLatencyMetric.tag,
            GoodRequestCountMetric.tag,
        )

        assert metrics[GoodRequestCountMetric.tag] == 2.0

    def test_no_slos_configured_returns_zero(self):
        records = [create_record(start_ns=0, responses=[100_000_000])]
        metrics = run_simple_metrics_pipeline(
            records,
            RequestLatencyMetric.tag,
            GoodRequestCountMetric.tag,
        )
        assert metrics[GoodRequestCountMetric.tag] == 0.0


class TestGoodRequestCountDirectionality:
    """Pin the SLO comparison directionality in ``GoodRequestCountMetric._passes``
    (good_request_count_metric.py lines 78-82).

    Mutation testing found that inverting ``_passes`` (or flipping ``>=`` to ``>``
    on the LARGER_IS_BETTER branch, line 81) survives the whole suite: no test
    used a LARGER_IS_BETTER metric as an SLO, so line 81 never executed, and no
    test pinned boundary equality. A corrupted comparison silently poisons
    good_request_count / goodput / good_request_fraction while every test stays
    green. These tests exercise BOTH directions and the boundary.
    """

    def setup_method(self):
        GoodRequestCountMetric.set_slos({})

    def teardown_method(self):
        # ``_thresholds`` is a ClassVar; reset so state cannot leak to other files.
        GoodRequestCountMetric.set_slos({})

    def test_smaller_is_better_all_below_threshold_all_good(self):
        """request_latency is NOT LARGER_IS_BETTER, so line 82 (``<=``) governs.
        With every latency STRICTLY BELOW the SLO, all N records are good.

        Catches an inverted ``_passes`` (``<=`` -> ``>=``): the inverted mutant
        would score every below-threshold record as BAD and return 0.
        """
        GoodRequestCountMetric.set_slos({RequestLatencyMetric.tag: 250.0})  # 250 ms
        records = [
            create_record(start_ns=0, responses=[t])
            for t in (100_000_000, 150_000_000, 200_000_000)  # 100/150/200 ms
        ]
        metrics = run_simple_metrics_pipeline(
            records, RequestLatencyMetric.tag, GoodRequestCountMetric.tag
        )
        assert metrics[GoodRequestCountMetric.tag] == 3.0

    def test_smaller_is_better_all_above_threshold_none_good(self):
        """request_latency above the SLO must score 0 good under line 82 (``<=``).

        Catches an inverted ``_passes`` (``<=`` -> ``>=``): the inverted mutant
        would score every above-threshold record as GOOD and return 3.
        """
        GoodRequestCountMetric.set_slos({RequestLatencyMetric.tag: 250.0})  # 250 ms
        records = [
            create_record(start_ns=0, responses=[t])
            for t in (300_000_000, 400_000_000, 500_000_000)  # 300/400/500 ms
        ]
        metrics = run_simple_metrics_pipeline(
            records, RequestLatencyMetric.tag, GoodRequestCountMetric.tag
        )
        assert metrics[GoodRequestCountMetric.tag] == 0.0

    def test_larger_is_better_straddling_set_exact_count(self):
        """output_sequence_length IS LARGER_IS_BETTER, so line 81 (``>=``) governs
        -- the branch no other test exercised. With OSL values straddling the
        SLO (one below, three above), exactly the three above are good.

        Catches an inverted ``_passes`` (line 81 ``>=`` -> ``<=``): the inverted
        mutant would count the one below-threshold record instead and return 1.
        The asymmetric 1-below/3-above split makes the two answers differ.
        """
        GoodRequestCountMetric.set_slos({OutputSequenceLengthMetric.tag: 10.0})
        # create_record OSL == len(responses) * output_tokens_per_response.
        records = [
            create_record(start_ns=0, responses=[100], output_tokens_per_response=osl)
            for osl in (5, 12, 15, 20)  # OSL 5 below; 12/15/20 above threshold 10
        ]
        metrics = run_simple_metrics_pipeline(
            records, OutputSequenceLengthMetric.tag, GoodRequestCountMetric.tag
        )
        assert metrics[GoodRequestCountMetric.tag] == 3.0

    def test_larger_is_better_boundary_equal_is_good(self):
        """A LARGER_IS_BETTER value EXACTLY EQUAL to the SLO is good (line 81
        is ``>=``, not ``>``). Pins finding [12] on the LARGER branch.

        Catches ``>=`` -> ``>`` on line 81: the ``>`` mutant would score the
        equal-to-threshold record as BAD and return 0.
        """
        GoodRequestCountMetric.set_slos({OutputSequenceLengthMetric.tag: 10.0})
        records = [
            create_record(start_ns=0, responses=[100], output_tokens_per_response=10)
        ]  # OSL 10 == threshold 10
        metrics = run_simple_metrics_pipeline(
            records, OutputSequenceLengthMetric.tag, GoodRequestCountMetric.tag
        )
        assert metrics[GoodRequestCountMetric.tag] == 1.0

    def test_smaller_is_better_boundary_equal_is_good(self):
        """A smaller-is-better latency EXACTLY EQUAL to the SLO is good (line 82
        is ``<=``, not ``<``). Pins finding [12] on the latency branch.

        Catches ``<=`` -> ``<`` on line 82: the ``<`` mutant would score the
        equal-to-threshold record as BAD and return 0.
        """
        GoodRequestCountMetric.set_slos({RequestLatencyMetric.tag: 250.0})  # 250 ms
        records = [
            create_record(start_ns=0, responses=[250_000_000])
        ]  # latency 250 ms == threshold 250 ms
        metrics = run_simple_metrics_pipeline(
            records, RequestLatencyMetric.tag, GoodRequestCountMetric.tag
        )
        assert metrics[GoodRequestCountMetric.tag] == 1.0
