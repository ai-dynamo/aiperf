# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_record_metric import BaseRecordMetric
from aiperf.metrics.derived_sum_metric import DerivedSumMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class InputSequenceLengthMetric(BaseRecordMetric[int]):
    """
    Post-processor for calculating Input Sequence Length (ISL) metrics from valid records.

    Formula:
        Input Sequence Length = Sum of Input Token Counts
    """

    tag = "input_sequence_length"
    header = "Input Sequence Length"
    short_header = "ISL"
    unit = GenericMetricUnit.TOKENS
    display_order = 700
    flags = MetricFlags.PRODUCES_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        This method extracts the input token count from the record and returns it.

        Raises:
            ValueError: If the record does not have an input token count.
        """
        if record.input_token_count is None:
            raise NoMetricValue("Input Token Count is not available for the record.")

        return record.input_token_count


class TotalInputSequenceLengthMetric(DerivedSumMetric[int, InputSequenceLengthMetric]):
    """
    This is the total number of input tokens processed by the benchmark for valid records.

    Formula:
        ```
        Total Input Sequence Length = Sum(Input Sequence Lengths)
        ```
    """

    tag = "total_isl"
    header = "Total Input Sequence Length"
    short_header = "Total ISL"
    short_header_hide_unit = True
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )


class ErrorInputSequenceLengthMetric(InputSequenceLengthMetric):
    """
    Post-processor for calculating Input Sequence Length (ISL) metrics from error records.
    """

    tag = "error_isl"
    header = "Error Input Sequence Length"
    short_header = "Error ISL"
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
        | MetricFlags.ERROR_ONLY
    )


class TotalErrorInputSequenceLengthMetric(
    DerivedSumMetric[int, ErrorInputSequenceLengthMetric]
):
    """
    This is the total number of input tokens processed in the benchmark for error records.

    Formula:
        ```
        Total Error Input Sequence Length = Sum(Error Input Sequence Lengths)
        ```
    """

    tag = "total_error_isl"
    header = "Total Error Input Sequence Length"
    short_header = "Total Error ISL"
    short_header_hide_unit = True
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
        | MetricFlags.ERROR_ONLY
    )
