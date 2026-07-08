# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import (
    ParsedResponse,
    ParsedResponseRecord,
    RequestInfo,
    RequestRecord,
)
from aiperf.common.models.record_models import TextResponseData, TokenCounts
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.inter_chunk_latency_metric import InterChunkLatencyMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from tests.unit.metrics.conftest import create_record, run_simple_metrics_pipeline


class TestRequestLatencyMetric:
    def test_request_latency_basic(self):
        """Test basic request latency calculation"""
        # Start at 100ns, response at 150ns = 50ns latency
        record = create_record(start_ns=100, responses=[150])

        metric_results = run_simple_metrics_pipeline(
            [record],
            RequestLatencyMetric.tag,
        )
        assert metric_results[RequestLatencyMetric.tag] == [50]

    def test_request_latency_multiple_responses(self):
        """Test latency with multiple responses uses final response timestamp"""
        # Start at 10ns, responses at 15ns, 25ns, 35ns = 25ns latency (final - start)
        record = create_record(start_ns=10, responses=[15, 25, 35])

        metric_results = run_simple_metrics_pipeline(
            [record],
            RequestLatencyMetric.tag,
        )
        assert metric_results[RequestLatencyMetric.tag] == [25]

    def test_request_latency_multiple_records(self):
        """Test processing multiple records"""
        records = [
            create_record(start_ns=10, responses=[25]),  # 15ns latency
            create_record(start_ns=20, responses=[35]),  # 15ns latency
            create_record(start_ns=30, responses=[50]),  # 20ns latency
        ]

        metric_results = run_simple_metrics_pipeline(
            records,
            RequestLatencyMetric.tag,
        )

        assert metric_results[RequestLatencyMetric.tag] == [15, 15, 20]

    def test_request_latency_invalid_timestamp(self):
        """Test error when response timestamp is before request start"""
        # Response at 90ns before request start at 100ns - should raise error
        record = create_record(start_ns=100, responses=[90])

        metric = RequestLatencyMetric()
        with pytest.raises(NoMetricValue, match="missing or marked invalid"):
            metric.parse_record(record, MetricRecordDict())


def _request_info() -> RequestInfo:
    return RequestInfo(
        turns=[],
        turn_index=0,
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="test-request-id",
        x_correlation_id="test-correlation-id",
        conversation_id="test-conversation",
    )


def _record_with_chunks(
    start_ns: int,
    chunks: list[tuple[int, TextResponseData | None, dict | None]],
) -> ParsedResponseRecord:
    """Build a ParsedResponseRecord from ``(perf_ns, data, usage)`` chunks.

    A chunk with ``data=None`` and ``usage`` set models the trailing/interior
    usage-only streaming chunk that arrives after (or between) content tokens.
    Such chunks are excluded from ``record.content_responses``.
    """
    responses = [
        ParsedResponse(perf_ns=perf_ns, data=data, usage=usage)
        for perf_ns, data, usage in chunks
    ]
    n_content = sum(1 for _, data, _ in chunks if data is not None)
    request = RequestRecord(
        request_info=_request_info(),
        model_name="test-model",
        start_perf_ns=start_ns,
        timestamp_ns=start_ns,
        end_perf_ns=chunks[-1][0],
        error=None,
    )
    return ParsedResponseRecord(
        request=request,
        responses=responses,
        token_counts=TokenCounts(input=None, output=n_content, reasoning=None),
    )


class TestRequestLatencyUsageChunkExclusion:
    """Pin the exclusion of usage-only streaming chunks (``data=None``) from
    latency timing (request_latency_metric.py:45 and the shared
    ``record.content_responses`` filter).

    Mutation testing found this critical path untested: swapping
    ``content_responses[-1]`` for ``responses[-1]`` (or dropping the
    content-only filter in the streaming-gap metric) silently measures latency
    to a usage/[DONE] chunk that arrives after the last real content token,
    inflating every latency stat while the suite stays green.
    """

    def test_request_latency_excludes_trailing_usage_only_chunk(self):
        """The LAST response is a usage-only chunk (data=None, usage set)
        arriving AFTER the last content token. request_latency must be measured
        to the last CONTENT token, not the usage chunk.

        Catches ``content_responses[-1]`` -> ``responses[-1]`` on line 45: the
        mutant would use the usage chunk at 400ns and report 300ns instead of
        the correct 150ns (last content token at 250ns minus start at 100ns).
        """
        content = TextResponseData(text="tok")
        record = _record_with_chunks(
            start_ns=100,
            chunks=[
                (150, content, None),  # content token
                (250, content, None),  # last content token
                (400, None, {"completion_tokens": 5}),  # trailing usage-only chunk
            ],
        )
        # Sanity: the usage-only chunk is not counted as content.
        assert [r.perf_ns for r in record.content_responses] == [150, 250]

        value = RequestLatencyMetric().parse_record(record, MetricRecordDict())
        assert value == 150  # 250 (last content) - 100 (start), NOT 400 - 100

    def test_inter_chunk_latency_excludes_interior_usage_only_chunk(self):
        """An interior usage-only chunk between two content tokens must NOT
        create a spurious inter-chunk gap -- the token gaps span content tokens
        only (finding [2], the "token gaps" half).

        Catches dropping the ``content_responses`` filter in
        inter_chunk_latency: with content at 150ns and 500ns and a usage-only
        chunk at 300ns, the correct result is a single gap [350]. The mutant
        that walks all responses would split it into [150, 200].
        """
        content = TextResponseData(text="tok")
        record = _record_with_chunks(
            start_ns=100,
            chunks=[
                (150, content, None),  # first content token
                (300, None, {"completion_tokens": 1}),  # interior usage-only chunk
                (500, content, None),  # second content token
            ],
        )
        gaps = InterChunkLatencyMetric().parse_record(record, MetricRecordDict())
        assert gaps == [350]  # 500 - 150, the usage chunk at 300 excluded
