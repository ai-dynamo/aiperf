# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.models.record_models import ReasoningResponseData, TextResponseData
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class TimeToFirstOutputTokenMetric(BaseRecordMetric[int]):
    """
    Calculates the time elapsed from request start to the first non-reasoning output token.

    This metric measures the latency from when a request is initiated to when the first
    actual output token (non-reasoning content) is received. It is particularly relevant
    for models that perform extended reasoning before generating output.

    Key Distinctions:
        - TTFO vs TTFT: Time to First Output (TTFO) measures time to the first non-reasoning
          token, while Time to First Token (TTFT) measures time to any first token including
          reasoning tokens. For models without reasoning, TTFO and TTFT are equivalent.
        - Non-reasoning tokens: Includes TextResponseData with non-empty text, or
          ReasoningResponseData with non-empty content field (regardless of reasoning field).

    Formula:
        Time to First Output = First Non-Reasoning Token Timestamp - Request Start Timestamp
    """

    tag = "time_to_first_output_token"
    header = "Time to First Output Token"
    short_header = "TTFO"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    flags = (
        MetricFlags.STREAMING_TOKENS_ONLY
        | MetricFlags.SUPPORTS_REASONING
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extracts and calculates the time to first non-reasoning output token for a single record.

        This method iterates through the response stream to find the first token containing
        actual output content (not reasoning), then computes the elapsed time from the request
        start timestamp.

        Args:
            record: The parsed response record containing request and response timing data
            record_metrics: Dictionary of previously computed metrics for this record (unused)

        Returns:
            The time to first output token in nanoseconds
        """
        try:
            # Try and find the first non-reasoning token output and extract the timestamp.
            # This is done by checking for the first response that is either a TextResponseData or a ReasoningResponseData
            # and has a non-empty text or content field. Note that ReasoningResponseData can have both reasoning and content.
            first_non_reasoning_token_perf_ns: int = next(
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
                "Record must have at least one non-reasoning token to calculate Time to First Output Token."
            ) from None

        request_perf_ns: int = record.request.start_perf_ns
        return first_non_reasoning_token_perf_ns - request_perf_ns
