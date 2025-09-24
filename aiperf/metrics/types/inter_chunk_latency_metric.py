# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class InterChunkLatencyMetric(BaseRecordMetric[list[int]]):
    """
    Calculates Inter Chunk Latency (ICL) for streaming responses.

    Inter Chunk Latency measures the time gaps between consecutive response chunks
    in a streaming response. For each request with multiple response chunks, this
    metric computes the time difference between each adjacent pair of chunks.

    Example:
        For a streaming response with chunks arriving at times [100ms, 150ms, 210ms]:
        ICL = [50ms, 60ms] (differences between consecutive timestamps)

    All ICL values from all requests are collected into a single list for
    statistical analysis across the entire benchmark run.

    Formula:
        ICL = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]

    Note: Only applicable to streaming responses with at least 2 chunks.
    """

    tag = "inter_chunk_latency"
    header = "Inter Chunk Latency"
    short_header = "ICL"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    flags = MetricFlags.STREAMING_TOKENS_ONLY | MetricFlags.EXPERIMENTAL
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> list[int]:
        """
        This method extracts the timestamps from the responses in the given
        ParsedResponseRecord object, computes the differences between consecutive responses (ICL),
        and returns the results.

        Raises:
            NoMetricValue: If the record does not have at least two responses
            ValueError: If any of the inter chunk latencies are not positive.
        """
        responses = record.responses

        if len(responses) < 2:
            raise NoMetricValue(
                "Record must have at least two responses to calculate Inter Chunk Latency."
            )

        inter_chunk_latencies = []
        for i in range(1, len(responses)):
            chunk_latency_ns = responses[i].perf_ns - responses[i - 1].perf_ns
            if chunk_latency_ns <= 0:
                raise ValueError("Each inter chunk latency must be positive.")
            inter_chunk_latencies.append(chunk_latency_ns)

        return inter_chunk_latencies
