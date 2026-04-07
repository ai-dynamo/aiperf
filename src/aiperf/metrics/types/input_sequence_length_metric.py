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
    flags = MetricFlags.TOKENIZES_INPUT_ONLY | MetricFlags.LARGER_IS_BETTER
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        Extract the input token count from the record.

        Returns server-reported prompt token count when available (the common
        case). Falls back to client-side token_counts.input_local for the rare case
        where the server does not report prompt token counts.

        Raises:
            NoMetricValue: If neither server nor client input token count is available.
        """
        if record.token_counts is None:
            raise NoMetricValue("Input Token Count is not available for the record.")

        # Prefer server-reported prompt tokens
        if record.token_counts.input is not None:
            return record.token_counts.input

        # Rare fallback: server didn't report prompt tokens, use client-side
        if record.token_counts.input_local is not None:
            return record.token_counts.input_local

        raise NoMetricValue("Input Token Count is not available for the record.")


class InputSequenceLengthServerMetric(BaseRecordMetric[int]):
    """
    Server-reported Input Sequence Length — file-only metric.

    Returns the server-reported prompt token count (``token_counts.input``).
    Raises ``NoMetricValue`` when the server did not report a value.
    """

    tag = "input_sequence_length_server"
    header = "Input Sequence Length (Server)"
    short_header = "ISL Server"
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.TOKENIZES_INPUT_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        if record.token_counts is not None and record.token_counts.input is not None:
            return record.token_counts.input
        raise NoMetricValue("Server-reported input token count is not available.")


class InputSequenceLengthLocalMetric(BaseRecordMetric[int]):
    """
    Client-side tokenized Input Sequence Length — file-only metric.

    Returns the client-side tokenized prompt token count
    (``token_counts.input_local``).  Raises ``NoMetricValue`` when the
    client-side value is not available.
    """

    tag = "input_sequence_length_local"
    header = "Input Sequence Length (Local)"
    short_header = "ISL Local"
    unit = GenericMetricUnit.TOKENS
    flags = (
        MetricFlags.TOKENIZES_INPUT_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        if (
            record.token_counts is not None
            and record.token_counts.input_local is not None
        ):
            return record.token_counts.input_local
        raise NoMetricValue("Client-side input token count is not available.")


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
        MetricFlags.TOKENIZES_INPUT_ONLY
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
        MetricFlags.TOKENIZES_INPUT_ONLY
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
        MetricFlags.TOKENIZES_INPUT_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
        | MetricFlags.ERROR_ONLY
    )
