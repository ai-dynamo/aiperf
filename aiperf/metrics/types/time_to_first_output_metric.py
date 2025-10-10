# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.models.record_models import ReasoningResponseData, TextResponseData
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class TimeToFirstOutputMetric(BaseRecordMetric[int]):
    """
    Post-processor for calculating Time to First Output metrics from records.

    NOTE: This metric is distinct from the Time to First Token (TTFT) metric, as it is the time to
    the first non-reasoning token, rather than the time to the first token.

    Formula:
        Time to First Output = First Non-Reasoning Token Timestamp - Request Start Timestamp
    """

    tag = "time_to_first_output"
    header = "Time to First Output"
    short_header = "TTFO"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    flags = (
        MetricFlags.STREAMING_TOKENS_ONLY
        | MetricFlags.SUPPORTS_REASONING
        | MetricFlags.EXPERIMENTAL
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        This method extracts the timestamps from the request start and the first non-reasoning token in the given
        RequestRecord object, computes the difference (Time to First Output), and returns the result.

        Raises:
            NoMetricValue: If the record does not have at least one non-reasoning token
            ValueError: If the first non-reasoning token is before the request start timestamp.
        """

        if len(record.responses) < 1:
            raise NoMetricValue(
                "Record must have at least one non-reasoning token to calculate Time to First Output."
            )

        try:
            # Try and find the first non-reasoning token output and extract the timestamp.
            # This is done by checking for the first response that is either a TextResponseData or a ReasoningResponseData
            # and has a non-empty text or content field. Note that ReasoningResponseData can have both reasoning and content.
            first_non_reasoning_token_ts: int = next(
                response.perf_ns
                for response in record.responses
                if (isinstance(response.data, TextResponseData) and response.data.text)
                or (
                    isinstance(response.data, ReasoningResponseData)
                    and response.data.content
                )
            )
        except StopIteration:
            raise NoMetricValue(
                "Record must have at least one non-reasoning token to calculate Time to First Output."
            ) from None

        request_ts: int = record.request.start_perf_ns
        if first_non_reasoning_token_ts < request_ts:
            raise ValueError(
                "First non-reasoning token timestamp is before request start timestamp, cannot compute Time to First Output."
            )

        return first_non_reasoning_token_ts - request_ts
