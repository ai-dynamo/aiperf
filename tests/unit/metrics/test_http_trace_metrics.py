# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for HTTP trace-based metrics following k6 naming conventions."""

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import (
    AioHttpTraceData,
    BaseTraceData,
    ParsedResponse,
    ParsedResponseRecord,
    RequestRecord,
)
from aiperf.common.models.record_models import TextResponseData, TokenCounts
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.http_trace_metrics import (
    HttpBlockedMetric,
    HttpChunksReceivedMetric,
    HttpChunksSentMetric,
    HttpConnectingMetric,
    HttpConnectionOverheadMetric,
    HttpConnectionReusedMetric,
    HttpDataReceivedMetric,
    HttpDataSentMetric,
    HttpDnsLookupMetric,
    HttpDurationMetric,
    HttpReceivingMetric,
    HttpSendingMetric,
    HttpWaitingMetric,
)
from tests.unit.metrics.conftest import run_simple_metrics_pipeline


def create_record_with_trace(
    start_ns: int = 100,
    responses: list[int] | None = None,
    trace_data: BaseTraceData | AioHttpTraceData | None = None,
) -> ParsedResponseRecord:
    """Create a test record with optional trace data."""
    responses = responses or [start_ns + 50]

    request = RequestRecord(
        conversation_id="test-conversation",
        turn_index=0,
        model_name="test-model",
        start_perf_ns=start_ns,
        timestamp_ns=start_ns,
        end_perf_ns=responses[-1] if responses else start_ns,
        trace_data=trace_data,
    )

    response_data = []
    for perf_ns in responses:
        response_data.append(
            ParsedResponse(
                perf_ns=perf_ns,
                data=TextResponseData(text="test"),
            )
        )

    return ParsedResponseRecord(
        request=request,
        responses=response_data,
        token_counts=TokenCounts(
            input=10,
            output=len(responses),
            reasoning=None,
        ),
    )


def create_aiohttp_trace_data(
    # Reference times
    reference_perf_ns: int = 1000,
    reference_time_ns: int = 1_000_000_000_000,
    # Connection pool
    pool_wait_start: int | None = None,
    pool_wait_end: int | None = None,
    # Connection reuse
    connection_reused: int | None = None,
    # DNS
    dns_start: int | None = None,
    dns_end: int | None = None,
    dns_cache_hit: int | None = None,
    # TCP connect
    tcp_start: int | None = None,
    tcp_end: int | None = None,
    # Request send
    request_send_start: int | None = None,
    request_headers_sent: int | None = None,
    request_send_end: int | None = None,
    request_chunks: list[tuple[int, int]] | None = None,
    # Response receive
    response_receive_start: int | None = None,
    response_headers_received: int | None = None,
    response_receive_end: int | None = None,
    response_chunks: list[tuple[int, int]] | None = None,
) -> AioHttpTraceData:
    """Create AioHttpTraceData with specified timestamps.

    Note: request_send_end_perf_ns is computed from request_chunks[-1][0].
    If request_send_end is provided but request_chunks is not, a synthetic chunk is created.
    """
    # request_send_end_perf_ns is computed from request_chunks[-1][0]
    # Create synthetic chunk if request_send_end provided but no chunks
    if request_chunks is None and request_send_end is not None:
        request_chunks = [(request_send_end, 0)]
    return AioHttpTraceData(
        trace_type="aiohttp",
        reference_perf_ns=reference_perf_ns,
        reference_time_ns=reference_time_ns,
        # Connection pool
        connection_pool_wait_start_perf_ns=pool_wait_start,
        connection_pool_wait_end_perf_ns=pool_wait_end,
        # Connection reuse
        connection_reused_perf_ns=connection_reused,
        # DNS
        dns_lookup_start_perf_ns=dns_start,
        dns_lookup_end_perf_ns=dns_end,
        dns_cache_hit_perf_ns=dns_cache_hit,
        # TCP
        tcp_connect_start_perf_ns=tcp_start,
        tcp_connect_end_perf_ns=tcp_end,
        # Request
        request_send_start_perf_ns=request_send_start,
        request_headers_sent_perf_ns=request_headers_sent,
        request_chunks=request_chunks or [],
        # Response
        response_receive_start_perf_ns=response_receive_start,
        response_headers_received_perf_ns=response_headers_received,
        response_receive_end_perf_ns=response_receive_end,
        response_chunks=response_chunks or [],
    )


class TestHttpBlockedMetric:
    """Tests for http_blocked metric (connection pool wait time)."""

    def test_blocked_with_pool_wait(self):
        """Test blocked time when connection pool wait occurred."""
        trace = create_aiohttp_trace_data(
            pool_wait_start=1000,
            pool_wait_end=1050,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpBlockedMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 50

    def test_blocked_no_pool_wait(self):
        """Test blocked time when no pool wait occurred (connection immediately available)."""
        trace = create_aiohttp_trace_data(
            pool_wait_start=None,
            pool_wait_end=None,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpBlockedMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_blocked_incomplete_pool_wait(self):
        """Test error when pool wait started but never completed."""
        trace = create_aiohttp_trace_data(
            pool_wait_start=1000,
            pool_wait_end=None,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpBlockedMetric()
        with pytest.raises(
            NoMetricValue, match="Pool wait started but never completed"
        ):
            metric.parse_record(record, MetricRecordDict())

    def test_blocked_no_trace_data(self):
        """Test error when no trace data available."""
        record = create_record_with_trace(trace_data=None)

        metric = HttpBlockedMetric()
        with pytest.raises(NoMetricValue, match="No trace data available"):
            metric.parse_record(record, MetricRecordDict())


class TestHttpConnectionReusedMetric:
    """Tests for http_connection_reused metric."""

    def test_connection_reused(self):
        """Test when connection was reused."""
        trace = create_aiohttp_trace_data(connection_reused=1500)
        record = create_record_with_trace(trace_data=trace)

        metric = HttpConnectionReusedMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 1

    def test_connection_not_reused(self):
        """Test when new connection was established."""
        trace = create_aiohttp_trace_data(connection_reused=None)
        record = create_record_with_trace(trace_data=trace)

        metric = HttpConnectionReusedMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0


class TestHttpConnectingMetric:
    """Tests for http_connecting metric (TCP/TLS connection time)."""

    def test_connecting_new_connection(self):
        """Test TCP connection time for new connection."""
        trace = create_aiohttp_trace_data(
            tcp_start=1100,
            tcp_end=1200,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpConnectingMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 100

    def test_connecting_reused_connection(self):
        """Test connecting time is 0 when connection reused."""
        trace = create_aiohttp_trace_data(
            connection_reused=1050,
            tcp_start=None,
            tcp_end=None,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpConnectingMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_connecting_no_start_timestamp(self):
        """Test error when no TCP connect start timestamp."""
        trace = create_aiohttp_trace_data(
            tcp_start=None,
            tcp_end=1200,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpConnectingMetric()
        with pytest.raises(NoMetricValue, match="No TCP connect start timestamp"):
            metric.parse_record(record, MetricRecordDict())

    def test_connecting_incomplete(self):
        """Test error when TCP connect started but never completed."""
        trace = create_aiohttp_trace_data(
            tcp_start=1100,
            tcp_end=None,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpConnectingMetric()
        with pytest.raises(
            NoMetricValue, match="TCP connect started but never completed"
        ):
            metric.parse_record(record, MetricRecordDict())


class TestHttpDnsLookupMetric:
    """Tests for http_dns_lookup metric."""

    def test_dns_lookup_performed(self):
        """Test DNS lookup time when resolution occurred."""
        trace = create_aiohttp_trace_data(
            dns_start=1000,
            dns_end=1025,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpDnsLookupMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 25

    def test_dns_cache_hit(self):
        """Test DNS lookup time is 0 when cache hit."""
        trace = create_aiohttp_trace_data(
            dns_cache_hit=1010,
            dns_start=None,
            dns_end=None,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpDnsLookupMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_dns_connection_reused(self):
        """Test DNS lookup time is 0 when connection reused."""
        trace = create_aiohttp_trace_data(
            connection_reused=1050,
            dns_start=None,
            dns_end=None,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpDnsLookupMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_dns_no_lookup(self):
        """Test DNS lookup time is 0 when no lookup occurred."""
        trace = create_aiohttp_trace_data(
            dns_start=None,
            dns_end=None,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpDnsLookupMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_dns_incomplete(self):
        """Test error when DNS lookup started but never completed."""
        trace = create_aiohttp_trace_data(
            dns_start=1000,
            dns_end=None,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpDnsLookupMetric()
        with pytest.raises(
            NoMetricValue, match="DNS lookup started but never completed"
        ):
            metric.parse_record(record, MetricRecordDict())


class TestHttpSendingMetric:
    """Tests for http_sending metric."""

    def test_sending_basic(self):
        """Test request sending time calculation."""
        trace = create_aiohttp_trace_data(
            request_send_start=2000,
            request_send_end=2100,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpSendingMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 100

    def test_sending_no_start(self):
        """Test error when no request send start timestamp."""
        trace = create_aiohttp_trace_data(
            request_send_start=None,
            request_send_end=2100,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpSendingMetric()
        with pytest.raises(NoMetricValue, match="No request send start timestamp"):
            metric.parse_record(record, MetricRecordDict())

    def test_sending_incomplete(self):
        """Test error when request send started but never completed."""
        trace = create_aiohttp_trace_data(
            request_send_start=2000,
            request_send_end=None,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpSendingMetric()
        with pytest.raises(
            NoMetricValue, match="Request send started but never completed"
        ):
            metric.parse_record(record, MetricRecordDict())


class TestHttpWaitingMetric:
    """Tests for http_waiting metric (TTFB)."""

    def test_waiting_basic(self):
        """Test TTFB calculation."""
        trace = create_aiohttp_trace_data(
            request_send_end=2100,
            response_chunks=[(2500, 100), (2600, 200), (2700, 150)],
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpWaitingMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 400  # 2500 - 2100

    def test_waiting_no_send_end(self):
        """Test error when no request send end timestamp."""
        trace = create_aiohttp_trace_data(
            request_send_end=None,
            response_chunks=[(2500, 100)],
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpWaitingMetric()
        with pytest.raises(NoMetricValue, match="No request send end timestamp"):
            metric.parse_record(record, MetricRecordDict())

    def test_waiting_no_response(self):
        """Test error when no response timestamps."""
        trace = create_aiohttp_trace_data(
            request_send_end=2100,
            response_chunks=[],
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpWaitingMetric()
        with pytest.raises(NoMetricValue, match="No response chunks"):
            metric.parse_record(record, MetricRecordDict())


class TestHttpReceivingMetric:
    """Tests for http_receiving metric."""

    def test_receiving_multiple_chunks(self):
        """Test receiving time with multiple response chunks."""
        trace = create_aiohttp_trace_data(
            response_chunks=[(2500, 100), (2600, 200), (2800, 150)],
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpReceivingMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 300  # 2800 - 2500

    def test_receiving_single_chunk(self):
        """Test receiving time is 0 for single chunk response."""
        trace = create_aiohttp_trace_data(
            response_chunks=[(2500, 100)],
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpReceivingMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0

    def test_receiving_no_timestamps(self):
        """Test error when no response timestamps."""
        trace = create_aiohttp_trace_data(
            response_chunks=[],
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpReceivingMetric()
        with pytest.raises(NoMetricValue, match="No response chunks"):
            metric.parse_record(record, MetricRecordDict())


class TestHttpDurationMetric:
    """Tests for http_duration metric."""

    def test_duration_basic(self):
        """Test total request duration calculation."""
        trace = create_aiohttp_trace_data(
            request_send_start=2000,
            response_receive_end=3500,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpDurationMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 1500

    def test_duration_no_start(self):
        """Test error when no request send start."""
        trace = create_aiohttp_trace_data(
            request_send_start=None,
            response_receive_end=3500,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpDurationMetric()
        with pytest.raises(NoMetricValue, match="No request send start timestamp"):
            metric.parse_record(record, MetricRecordDict())

    def test_duration_no_end(self):
        """Test error when no response receive end."""
        trace = create_aiohttp_trace_data(
            request_send_start=2000,
            response_receive_end=None,
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpDurationMetric()
        with pytest.raises(NoMetricValue, match="No response receive end timestamp"):
            metric.parse_record(record, MetricRecordDict())


class TestHttpDataSentMetric:
    """Tests for http_data_sent metric."""

    def test_data_sent_multiple_chunks(self):
        """Test total bytes sent with multiple chunks."""
        trace = create_aiohttp_trace_data(
            request_chunks=[(1000, 100), (1050, 200), (1100, 150)],
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpDataSentMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 450

    def test_data_sent_no_chunks(self):
        """Test bytes sent is 0 when no write chunks."""
        trace = create_aiohttp_trace_data(
            request_chunks=[],
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpDataSentMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0


class TestHttpDataReceivedMetric:
    """Tests for http_data_received metric."""

    def test_data_received_multiple_chunks(self):
        """Test total bytes received with multiple chunks."""
        trace = create_aiohttp_trace_data(
            response_chunks=[(2000, 500), (2100, 1000), (2200, 250)],
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpDataReceivedMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 1750

    def test_data_received_no_chunks(self):
        """Test bytes received is 0 when no receive chunks."""
        trace = create_aiohttp_trace_data(
            response_chunks=[],
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpDataReceivedMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0


class TestHttpChunksSentMetric:
    """Tests for http_chunks_sent metric."""

    def test_chunks_sent(self):
        """Test number of chunks sent."""
        trace = create_aiohttp_trace_data(
            request_chunks=[(1000, 50), (1050, 60), (1100, 70), (1150, 80)],
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpChunksSentMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 4

    def test_chunks_sent_zero(self):
        """Test zero chunks when no writes."""
        trace = create_aiohttp_trace_data(
            request_chunks=[],
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpChunksSentMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0


class TestHttpChunksReceivedMetric:
    """Tests for http_chunks_received metric."""

    def test_chunks_received(self):
        """Test number of chunks received."""
        trace = create_aiohttp_trace_data(
            response_chunks=[(2000, 100), (2100, 200), (2200, 150)],
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpChunksReceivedMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 3

    def test_chunks_received_zero(self):
        """Test zero chunks when no receives."""
        trace = create_aiohttp_trace_data(
            response_chunks=[],
        )
        record = create_record_with_trace(trace_data=trace)

        metric = HttpChunksReceivedMetric()
        result = metric.parse_record(record, MetricRecordDict())
        assert result == 0


class TestHttpConnectionOverheadMetric:
    """Tests for http_connection_overhead metric."""

    def test_connection_overhead_all_components(self):
        """Test total connection overhead with all components."""
        trace = create_aiohttp_trace_data(
            pool_wait_start=1000,
            pool_wait_end=1050,  # blocked = 50
            dns_start=1060,
            dns_end=1090,  # dns = 30
            tcp_start=1100,
            tcp_end=1200,  # connecting = 100
        )
        record = create_record_with_trace(trace_data=trace)

        metric_results = run_simple_metrics_pipeline(
            [record],
            HttpConnectionOverheadMetric.tag,
        )
        assert metric_results[HttpConnectionOverheadMetric.tag] == [
            180
        ]  # 50 + 30 + 100

    def test_connection_overhead_reused_connection(self):
        """Test connection overhead is 0 when connection reused."""
        trace = create_aiohttp_trace_data(
            connection_reused=1050,
        )
        record = create_record_with_trace(trace_data=trace)

        metric_results = run_simple_metrics_pipeline(
            [record],
            HttpConnectionOverheadMetric.tag,
        )
        assert metric_results[HttpConnectionOverheadMetric.tag] == [0]


class TestHttpTraceMetricsPipeline:
    """Integration tests for running HTTP trace metrics through the pipeline."""

    def test_full_request_lifecycle_metrics(self):
        """Test all timing metrics for a complete request lifecycle."""
        trace = create_aiohttp_trace_data(
            # Connection pool: wait 50ns
            pool_wait_start=1000,
            pool_wait_end=1050,
            # DNS: 25ns
            dns_start=1060,
            dns_end=1085,
            # TCP connect: 100ns
            tcp_start=1100,
            tcp_end=1200,
            # Request send: 80ns
            request_send_start=1250,
            request_send_end=1330,
            # Response: TTFB 200ns, receiving 300ns
            response_chunks=[(1530, 100), (1700, 200), (1830, 150)],
            response_receive_end=1830,
        )
        record = create_record_with_trace(
            start_ns=1250,
            responses=[1530, 1700, 1830],
            trace_data=trace,
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            HttpBlockedMetric.tag,
            HttpDnsLookupMetric.tag,
            HttpConnectingMetric.tag,
            HttpSendingMetric.tag,
            HttpWaitingMetric.tag,
            HttpReceivingMetric.tag,
            HttpDurationMetric.tag,
        )

        assert metric_results[HttpBlockedMetric.tag] == [50]
        assert metric_results[HttpDnsLookupMetric.tag] == [25]
        assert metric_results[HttpConnectingMetric.tag] == [100]
        assert metric_results[HttpSendingMetric.tag] == [80]
        assert metric_results[HttpWaitingMetric.tag] == [200]  # 1530 - 1330
        assert metric_results[HttpReceivingMetric.tag] == [300]  # 1830 - 1530
        assert metric_results[HttpDurationMetric.tag] == [580]  # 1830 - 1250

    def test_reused_connection_lifecycle(self):
        """Test metrics when connection is reused (no pool wait, DNS, or TCP)."""
        trace = create_aiohttp_trace_data(
            connection_reused=1010,
            request_send_start=1050,
            request_send_end=1100,
            response_chunks=[(1300, 100), (1500, 200)],
            response_receive_end=1500,
        )
        record = create_record_with_trace(
            start_ns=1050,
            responses=[1300, 1500],
            trace_data=trace,
        )

        metric_results = run_simple_metrics_pipeline(
            [record],
            HttpBlockedMetric.tag,
            HttpDnsLookupMetric.tag,
            HttpConnectingMetric.tag,
            HttpConnectionReusedMetric.tag,
            HttpSendingMetric.tag,
            HttpWaitingMetric.tag,
            HttpReceivingMetric.tag,
            HttpDurationMetric.tag,
        )

        assert metric_results[HttpBlockedMetric.tag] == [0]
        assert metric_results[HttpDnsLookupMetric.tag] == [0]
        assert metric_results[HttpConnectingMetric.tag] == [0]
        assert metric_results[HttpConnectionReusedMetric.tag] == [1]
        assert metric_results[HttpSendingMetric.tag] == [50]
        assert metric_results[HttpWaitingMetric.tag] == [200]  # 1300 - 1100
        assert metric_results[HttpReceivingMetric.tag] == [200]  # 1500 - 1300
        assert metric_results[HttpDurationMetric.tag] == [450]  # 1500 - 1050

    def test_multiple_records(self):
        """Test processing multiple records with different trace data."""
        # First request: new connection
        trace1 = create_aiohttp_trace_data(
            tcp_start=1000,
            tcp_end=1100,
            request_send_start=1100,
            request_send_end=1150,
            response_chunks=[(1250, 100)],
            response_receive_end=1250,
        )
        record1 = create_record_with_trace(
            start_ns=1100,
            responses=[1250],
            trace_data=trace1,
        )

        # Second request: reused connection
        trace2 = create_aiohttp_trace_data(
            connection_reused=2010,
            request_send_start=2050,
            request_send_end=2080,
            response_chunks=[(2180, 100)],
            response_receive_end=2180,
        )
        record2 = create_record_with_trace(
            start_ns=2050,
            responses=[2180],
            trace_data=trace2,
        )

        metric_results = run_simple_metrics_pipeline(
            [record1, record2],
            HttpConnectingMetric.tag,
            HttpConnectionReusedMetric.tag,
            HttpDurationMetric.tag,
        )

        assert metric_results[HttpConnectingMetric.tag] == [100, 0]
        assert metric_results[HttpConnectionReusedMetric.tag] == [0, 1]
        assert metric_results[HttpDurationMetric.tag] == [150, 130]

    def test_data_size_metrics(self):
        """Test data size metrics with request and response chunks."""
        trace = create_aiohttp_trace_data(
            request_chunks=[(1000, 100), (1010, 200), (1020, 50)],
            response_chunks=[(1100, 500), (1200, 1500)],
        )
        record = create_record_with_trace(trace_data=trace)

        metric_results = run_simple_metrics_pipeline(
            [record],
            HttpDataSentMetric.tag,
            HttpDataReceivedMetric.tag,
            HttpChunksSentMetric.tag,
            HttpChunksReceivedMetric.tag,
        )

        assert metric_results[HttpDataSentMetric.tag] == [350]
        assert metric_results[HttpDataReceivedMetric.tag] == [2000]
        assert metric_results[HttpChunksSentMetric.tag] == [3]
        assert metric_results[HttpChunksReceivedMetric.tag] == [2]


class TestMetricAttributes:
    """Tests for metric class attributes (tags, headers, units, etc.)."""

    @pytest.mark.parametrize(
        "metric_class,expected_tag",
        [
            (HttpBlockedMetric, "http_req_blocked"),
            (HttpConnectionReusedMetric, "http_req_connection_reused"),
            (HttpConnectingMetric, "http_req_connecting"),
            (HttpDnsLookupMetric, "http_req_dns_lookup"),
            (HttpSendingMetric, "http_req_sending"),
            (HttpWaitingMetric, "http_req_waiting"),
            (HttpReceivingMetric, "http_req_receiving"),
            (HttpDurationMetric, "http_req_duration"),
            (HttpDataSentMetric, "http_req_data_sent"),
            (HttpDataReceivedMetric, "http_req_data_received"),
            (HttpChunksSentMetric, "http_req_chunks_sent"),
            (HttpChunksReceivedMetric, "http_req_chunks_received"),
            (HttpConnectionOverheadMetric, "http_req_connection_overhead"),
        ],
    )  # fmt: skip
    def test_metric_tags_follow_k6_naming_convention(self, metric_class, expected_tag):
        """Test that metric tags follow k6 naming convention (http_req_)."""
        assert metric_class.tag == expected_tag
        assert metric_class.tag.startswith("http_req_")

    @pytest.mark.parametrize(
        "metric_class",
        [
            HttpBlockedMetric,
            HttpConnectingMetric,
            HttpDnsLookupMetric,
            HttpSendingMetric,
            HttpWaitingMetric,
            HttpReceivingMetric,
            HttpDurationMetric,
            HttpConnectionOverheadMetric,
        ],
    )  # fmt: skip
    def test_timing_metrics_use_nanoseconds_internally(self, metric_class):
        """Test that timing metrics use nanoseconds as internal unit."""
        from aiperf.common.enums import MetricTimeUnit

        assert metric_class.unit == MetricTimeUnit.NANOSECONDS

    @pytest.mark.parametrize(
        "metric_class",
        [
            HttpBlockedMetric,
            HttpConnectingMetric,
            HttpDnsLookupMetric,
            HttpSendingMetric,
            HttpWaitingMetric,
            HttpReceivingMetric,
            HttpDurationMetric,
            HttpConnectionOverheadMetric,
        ],
    )  # fmt: skip
    def test_timing_metrics_display_in_milliseconds(self, metric_class):
        """Test that timing metrics display in milliseconds (like k6)."""
        from aiperf.common.enums import MetricTimeUnit

        assert metric_class.display_unit == MetricTimeUnit.MILLISECONDS

    @pytest.mark.parametrize(
        "metric_class",
        [
            HttpDataSentMetric,
            HttpDataReceivedMetric,
        ],
    )  # fmt: skip
    def test_size_metrics_use_bytes_internally(self, metric_class):
        """Test that size metrics use bytes as internal unit."""
        from aiperf.common.enums import MetricSizeUnit

        assert metric_class.unit == MetricSizeUnit.BYTES


class TestComputedPropertyAlignment:
    """Tests to verify metrics compute the same values as trace data export computed properties."""

    def test_sending_matches_export_sending_ns(self):
        """Verify http_sending computes the same value as TraceDataExport.sending_ns."""
        trace = create_aiohttp_trace_data(
            request_send_start=1000,
            request_send_end=1150,
        )
        record = create_record_with_trace(trace_data=trace)

        # Metric value
        metric = HttpSendingMetric()
        metric_value = metric.parse_record(record, MetricRecordDict())

        # Export computed property value
        export = trace.to_export()
        export_value = export.sending_ns

        assert metric_value == export_value

    def test_waiting_matches_export_waiting_ns(self):
        """Verify http_waiting computes the same value as TraceDataExport.waiting_ns."""
        trace = create_aiohttp_trace_data(
            request_send_end=1000,
            response_chunks=[(1500, 100), (1600, 200), (1700, 150)],
        )
        record = create_record_with_trace(trace_data=trace)

        # Metric value
        metric = HttpWaitingMetric()
        metric_value = metric.parse_record(record, MetricRecordDict())

        # Export computed property value
        export = trace.to_export()
        export_value = export.waiting_ns

        assert metric_value == export_value

    def test_receiving_matches_export_receiving_ns(self):
        """Verify http_receiving computes the same value as TraceDataExport.receiving_ns."""
        trace = create_aiohttp_trace_data(
            response_chunks=[(1000, 100), (1200, 200), (1500, 150)],
        )
        record = create_record_with_trace(trace_data=trace)

        # Metric value
        metric = HttpReceivingMetric()
        metric_value = metric.parse_record(record, MetricRecordDict())

        # Export computed property value
        export = trace.to_export()
        export_value = export.receiving_ns

        assert metric_value == export_value

    def test_duration_matches_export_duration_ns(self):
        """Verify http_duration computes the same value as TraceDataExport.duration_ns."""
        trace = create_aiohttp_trace_data(
            request_send_start=1000,
            response_receive_end=2500,
        )
        record = create_record_with_trace(trace_data=trace)

        # Metric value
        metric = HttpDurationMetric()
        metric_value = metric.parse_record(record, MetricRecordDict())

        # Export computed property value
        export = trace.to_export()
        export_value = export.duration_ns

        assert metric_value == export_value

    def test_blocked_matches_export_blocked_ns(self):
        """Verify http_blocked computes the same value as AioHttpTraceDataExport.blocked_ns."""
        trace = create_aiohttp_trace_data(
            pool_wait_start=1000,
            pool_wait_end=1250,
        )
        record = create_record_with_trace(trace_data=trace)

        # Metric value
        metric = HttpBlockedMetric()
        metric_value = metric.parse_record(record, MetricRecordDict())

        # Export computed property value
        export = trace.to_export()
        export_value = export.blocked_ns

        assert metric_value == export_value

    def test_dns_lookup_matches_export_dns_lookup_ns(self):
        """Verify http_dns_lookup computes the same value as AioHttpTraceDataExport.dns_lookup_ns."""
        trace = create_aiohttp_trace_data(
            dns_start=1000,
            dns_end=1100,
        )
        record = create_record_with_trace(trace_data=trace)

        # Metric value
        metric = HttpDnsLookupMetric()
        metric_value = metric.parse_record(record, MetricRecordDict())

        # Export computed property value
        export = trace.to_export()
        export_value = export.dns_lookup_ns

        assert metric_value == export_value

    def test_connecting_matches_export_connecting_ns(self):
        """Verify http_connecting computes the same value as AioHttpTraceDataExport.connecting_ns."""
        trace = create_aiohttp_trace_data(
            tcp_start=1000,
            tcp_end=1200,
        )
        record = create_record_with_trace(trace_data=trace)

        # Metric value
        metric = HttpConnectingMetric()
        metric_value = metric.parse_record(record, MetricRecordDict())

        # Export computed property value
        export = trace.to_export()
        export_value = export.connecting_ns

        assert metric_value == export_value
