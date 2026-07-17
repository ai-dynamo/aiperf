# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import approx, param

from aiperf.common.constants import STREAMED_REQUEST_COUNT_TAG, STREAMED_REQUEST_TAG
from aiperf.common.enums import MetricConsoleGroup, MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.accumulator import MetricsAccumulator
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.inter_chunk_latency_metric import InterChunkLatencyMetric
from aiperf.metrics.types.inter_token_latency_metric import InterTokenLatencyMetric
from aiperf.metrics.types.stream_latency_metrics import StreamSetupLatencyMetric
from aiperf.metrics.types.streamed_request_count_metric import (
    StreamedRequestCountMetric,
)
from aiperf.metrics.types.streamed_request_metric import StreamedRequestMetric
from aiperf.metrics.types.time_to_first_output_token_metric import (
    TimeToFirstOutputTokenMetric,
)
from aiperf.metrics.types.ttft_metric import TTFTMetric
from aiperf.metrics.types.ttst_metric import TTSTMetric
from tests.unit.conftest import make_benchmark_run
from tests.unit.metrics.conftest import create_record, run_simple_metrics_pipeline
from tests.unit.post_processors.conftest import create_metric_records_data

# Streaming metrics whose _parse_record reads response timing directly, so they
# carry the explicit membership guard + a required_metrics dependency on the
# per-record streaming predicate.
GUARDED_STREAMING_METRICS = [
    TTFTMetric,
    TTSTMetric,
    InterChunkLatencyMetric,
    TimeToFirstOutputTokenMetric,
    StreamSetupLatencyMetric,
]


class TestStreamedRequestPredicate:
    """The hidden per-record predicate that gates streaming metrics."""

    def test_predicate_parses_streamed_and_skips_non_streamed(self):
        """A streamed record parses to 1; a non-streamed record raises NoMetricValue."""
        metric = StreamedRequestMetric()

        streamed = create_record(start_ns=100, responses=[110], streamed=True)
        assert metric.parse_record(streamed, MetricRecordDict()) == 1

        non_streamed = create_record(start_ns=100, responses=[110], streamed=False)
        with pytest.raises(NoMetricValue):
            metric.parse_record(non_streamed, MetricRecordDict())

    def test_predicate_only_present_for_streamed_records(self):
        """Only streamed records carry a per-record predicate value."""
        records = [
            create_record(start_ns=100, responses=[110], streamed=True),
            create_record(start_ns=200, responses=[210], streamed=False),
            create_record(start_ns=300, responses=[310], streamed=True),
        ]
        results = run_simple_metrics_pipeline(records, STREAMED_REQUEST_TAG)
        assert results[STREAMED_REQUEST_TAG] == [1, 1]

    def test_predicate_is_console_hidden(self):
        """The predicate is excluded from the console (constant-1 gate is not useful)."""
        assert StreamedRequestMetric.console_group == MetricConsoleGroup.NONE

    @pytest.mark.asyncio
    async def test_predicate_absent_from_summary_export(self):
        """The predicate is INTERNAL, so summarize() drops it from exports.

        The accumulator keeps the hidden predicate available for dependent
        metrics, then filters the meaningless constant-1 stat row from the
        final summary while retaining the visible streamed_request_count.
        """
        assert StreamedRequestMetric.has_flags(MetricFlags.INTERNAL)

        accumulator = MetricsAccumulator(make_benchmark_run())
        accumulator._derive_funcs = {}
        await accumulator.process_record(
            create_metric_records_data(
                x_request_id="stream-1",
                results=[
                    {
                        StreamedRequestMetric.tag: 1,
                        StreamedRequestCountMetric.tag: 2,
                    }
                ],
            )
        )

        summary_tags = set((await accumulator.summarize()).results)
        assert StreamedRequestMetric.tag not in summary_tags
        assert StreamedRequestCountMetric.tag in summary_tags


class TestStreamedRequestCountMetric:
    """The visible aggregate denominator displayed beside Request Count."""

    def test_count_no_records(self):
        """No records means no aggregate value (not a 0 default)."""
        results = run_simple_metrics_pipeline([], STREAMED_REQUEST_COUNT_TAG)
        assert STREAMED_REQUEST_COUNT_TAG not in results

    def test_count_aggregates_streamed_and_skips_non_streamed(self):
        """The aggregate counts only records that streamed on the wire."""
        records = [
            create_record(start_ns=100, responses=[110], streamed=True),
            create_record(start_ns=200, responses=[210], streamed=False),
            create_record(start_ns=300, responses=[310], streamed=True),
        ]
        results = run_simple_metrics_pipeline(records, STREAMED_REQUEST_COUNT_TAG)
        assert results[STREAMED_REQUEST_COUNT_TAG] == approx(2)

    def test_count_parse_record_skips_non_streamed(self):
        """The aggregate's per-record parse mirrors the streamed predicate."""
        metric = StreamedRequestCountMetric()

        streamed = create_record(start_ns=100, responses=[110], streamed=True)
        assert metric.parse_record(streamed, MetricRecordDict()) == 1

        non_streamed = create_record(start_ns=100, responses=[110], streamed=False)
        with pytest.raises(NoMetricValue):
            metric.parse_record(non_streamed, MetricRecordDict())

    @pytest.mark.parametrize("num_streamed", [1, 3, 10, 100])
    def test_count_matches_streamed_record_count(self, num_streamed: int):
        """The aggregate equals the number of streamed records."""
        records = [
            create_record(start_ns=100 * i, streamed=True) for i in range(num_streamed)
        ]
        results = run_simple_metrics_pipeline(records, STREAMED_REQUEST_COUNT_TAG)
        assert results[STREAMED_REQUEST_COUNT_TAG] == approx(num_streamed)

    def test_count_is_not_console_hidden(self):
        """The visible aggregate uses the default console group (not hidden)."""
        assert StreamedRequestCountMetric.console_group == MetricConsoleGroup.DEFAULT


class TestStreamingMetricGating:
    def test_ttft_skips_non_streamed_record(self):
        """A non-streamed record must NOT produce a TTFT value (the pollution bug).

        Pre-fix, TTFT computed ``first content response - start`` even when the sole
        TextResponse timestamp was the completion time, reporting full latency as TTFT.
        """
        record = create_record(start_ns=100, responses=[150], streamed=False)

        # Full pipeline: TTFT is absent because the predicate never fires.
        results = run_simple_metrics_pipeline([record], TTFTMetric.tag)
        assert TTFTMetric.tag not in results

        # Inline guard: _parse_record raises when the predicate tag is absent.
        with pytest.raises(NoMetricValue):
            TTFTMetric()._parse_record(record, MetricRecordDict())

    def test_ttft_computes_for_streamed_record(self):
        """A streamed record yields the unchanged TTFT value."""
        record = create_record(start_ns=100, responses=[110, 120], streamed=True)

        # Full pipeline computes the predicate first, then TTFT (110 - 100 = 10).
        results = run_simple_metrics_pipeline([record], TTFTMetric.tag)
        assert results[TTFTMetric.tag] == [10]

        # Direct parse with the predicate primed (dependency order).
        primed = MetricRecordDict()
        primed[STREAMED_REQUEST_TAG] = 1
        assert TTFTMetric().parse_record(record, primed) == 10

    def test_inter_token_latency_inherits_skip_for_non_streamed(self):
        """ITL consumes only guarded metrics, so it inherits the skip transitively."""
        record = create_record(start_ns=100, responses=[110, 120, 130], streamed=False)
        results = run_simple_metrics_pipeline([record], InterTokenLatencyMetric.tag)
        assert InterTokenLatencyMetric.tag not in results

    @pytest.mark.parametrize(
        "metric_cls",
        [param(m, id=m.tag) for m in GUARDED_STREAMING_METRICS],
    )  # fmt: skip
    def test_guarded_metrics_declare_dependency(self, metric_cls):
        """Every directly-guarded metric declares the predicate in required_metrics."""
        assert metric_cls.required_metrics is not None
        assert STREAMED_REQUEST_TAG in metric_cls.required_metrics

    @pytest.mark.parametrize(
        "metric_cls",
        [param(m, id=m.tag) for m in GUARDED_STREAMING_METRICS],
    )  # fmt: skip
    def test_guarded_metric_parse_raises_without_predicate(self, metric_cls):
        """Each guarded metric's parse path raises NoMetricValue when the predicate is
        absent from record_metrics (the non-streamed record case)."""
        record = create_record(start_ns=100, responses=[110, 120], streamed=False)
        with pytest.raises(NoMetricValue):
            metric_cls()._parse_record(record, MetricRecordDict())

    def test_dependency_order_places_predicate_before_dependents(self):
        """Topological sort computes the predicate before its dependents."""
        order = MetricRegistry.create_dependency_order_for([TTFTMetric.tag])
        assert STREAMED_REQUEST_TAG in order
        assert order.index(STREAMED_REQUEST_TAG) < order.index(TTFTMetric.tag)
