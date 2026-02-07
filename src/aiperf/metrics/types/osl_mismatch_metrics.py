# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output Sequence Length (OSL) mismatch metrics.

These metrics calculate the percentage difference between requested OSL (max_tokens)
and actual output token count. They help identify when the inference server is not
honoring the requested output sequence length, which typically happens when
ignore_eos is not set or unsupported by the server.
"""

from typing import ClassVar

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.environment import Environment
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.logging import AIPerfLogger
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.base_aggregate_counter_metric import BaseAggregateCounterMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)

_logger = AIPerfLogger(__name__)


class RequestedOSLMetric(BaseRecordMetric[int]):
    """
    Metric for extracting the requested output sequence length (max_tokens) from a record.

    This captures the max_tokens value that was sent to the server, which represents
    the desired output sequence length for the request.
    """

    tag = "requested_osl"
    header = "Requested OSL"
    short_header = "Req OSL"
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY | MetricFlags.NO_CONSOLE | MetricFlags.INTERNAL
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the requested max_tokens from the request.

        Raises:
            NoMetricValue: If max_tokens is not set in the request.
        """
        request_info = record.request.request_info
        if request_info is None or not request_info.turns:
            raise NoMetricValue("Request info or turns not available in record.")

        # Get max_tokens from the last turn (the one that was sent)
        max_tokens = request_info.turns[-1].max_tokens
        if max_tokens is None:
            raise NoMetricValue("max_tokens not set in request (--osl not used).")

        return max_tokens


class OSLMismatchDiffMetric(BaseRecordMetric[float]):
    """
    Percentage difference between actual output sequence length and requested OSL.

    This metric compares the actual number of output tokens generated with the
    max_tokens requested via --osl. Large discrepancies indicate the server is not
    honoring the requested output length, often because ignore_eos is not set.

    Formula:
        Diff % = ((Actual OSL - Requested OSL) / Requested OSL) * 100

    Note: This is a signed percentage - negative means actual < requested (stopped early),
    positive means actual > requested (generated more than requested).

    Example:
        If requested 100 tokens and got 50: Diff = ((50 - 100) / 100) * 100 = -50%
        If requested 100 tokens and got 100: Diff = ((100 - 100) / 100) * 100 = 0%
        If requested 100 tokens and got 120: Diff = ((120 - 100) / 100) * 100 = 20%
    """

    tag = "osl_mismatch_diff_pct"
    header = "OSL Mismatch Diff %"
    short_header = "OSL Diff"
    short_header_hide_unit = True
    unit = GenericMetricUnit.PERCENT
    flags = MetricFlags.PRODUCES_TOKENS_ONLY | MetricFlags.NO_CONSOLE
    required_metrics: ClassVar[set[str]] = {
        RequestedOSLMetric.tag,
        OutputSequenceLengthMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """
        Calculate the percentage difference between requested and actual OSL.

        Raises:
            NoMetricValue: If either metric is not available or requested OSL is zero.
        """
        requested_osl = record_metrics.get_or_raise(RequestedOSLMetric)
        actual_osl = record_metrics.get_or_raise(OutputSequenceLengthMetric)

        if requested_osl == 0:
            raise NoMetricValue(
                "Cannot calculate OSL mismatch with zero requested tokens."
            )

        # Negative = stopped early, positive = generated more than requested
        diff_pct = ((actual_osl - requested_osl) / requested_osl) * 100.0

        return diff_pct


class OSLMismatchCountMetric(BaseAggregateCounterMetric[int]):
    """
    Count of records where OSL mismatch exceeds the configured threshold.

    This aggregate counter metric increments by 1 for each record where the
    absolute token difference exceeds the effective threshold. The effective
    threshold is the minimum of:
    - requested_osl * (OSL_MISMATCH_PCT_THRESHOLD / 100)
    - OSL_MISMATCH_MAX_TOKEN_THRESHOLD (default: 50 tokens)

    This makes the threshold tighter for large OSL values - e.g., a 2000 token
    request is capped at 50 token diff instead of 100 tokens (5%).

    Formula:
        threshold_tokens = min(requested_osl * pct_threshold / 100, max_token_threshold)
        diff_tokens = abs(actual_osl - requested_osl)
        increment = 1 if diff_tokens > threshold_tokens else 0

        Total Count = Sum of all increments

    Example:
        With pct_threshold=5% and max_token_threshold=50:
        - Request 100 tokens: threshold = min(5, 50) = 5 tokens
        - Request 1000 tokens: threshold = min(50, 50) = 50 tokens
        - Request 2000 tokens: threshold = min(100, 50) = 50 tokens (capped)
    """

    tag = "osl_mismatch_count"
    header = "OSL Mismatch Count"
    short_header = "OSL Mismatches"
    short_header_hide_unit = True
    unit = GenericMetricUnit.REQUESTS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.NO_CONSOLE
        | MetricFlags.NO_INDIVIDUAL_RECORDS
    )
    required_metrics: ClassVar[set[str]] = {
        OSLMismatchDiffMetric.tag,
        RequestedOSLMetric.tag,
        OutputSequenceLengthMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Return 1 if OSL mismatch exceeds threshold, 0 otherwise.

        Returns:
            1 if token diff > min(requested * pct_threshold, max_token_threshold), 0 otherwise
        """
        pct_threshold = Environment.METRICS.OSL_MISMATCH_PCT_THRESHOLD
        max_token_threshold = Environment.METRICS.OSL_MISMATCH_MAX_TOKEN_THRESHOLD

        requested = record_metrics.get_or_raise(RequestedOSLMetric)
        actual = record_metrics.get_or_raise(OutputSequenceLengthMetric)
        osl_diff_pct = record_metrics.get_or_raise(OSLMismatchDiffMetric)

        # Effective threshold is min of percentage-based and absolute token limit
        threshold_tokens = min(requested * (pct_threshold / 100), max_token_threshold)
        diff_tokens = abs(actual - requested)

        if diff_tokens > threshold_tokens:
            _logger.warning(
                f"OSL mismatch: got {actual} tokens (requested {requested}, diff {osl_diff_pct:.1f}%)"
            )
            return 1

        return 0
