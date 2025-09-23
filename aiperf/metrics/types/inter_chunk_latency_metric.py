# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class InterChunkLatencyMetric(BaseRecordMetric[float]):
    """
    Post-processor for calculating Inter Chunk Latency (ICL) metrics from records. This is the average
    of the differences between consecutive responses. This is only applicable to streaming responses.

    Formula:
        Inter Chunk Latency = Average time between consecutive responses
    """

    tag = "inter_chunk_latency"
    header = "Inter Chunk Latency"
    short_header = "ICL"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    flags = MetricFlags.STREAMING_TOKENS_ONLY | MetricFlags.HIDDEN
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """
        This method extracts the timestamps from the responses in the given
        RequestRecord object, computes the average of the differences between consecutive responses (ICL),
        and returns the result.

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
                raise ValueError("Inter chunk latencies must be positive.")
            inter_chunk_latencies.append(chunk_latency_ns)

        return sum(inter_chunk_latencies) / len(inter_chunk_latencies)
