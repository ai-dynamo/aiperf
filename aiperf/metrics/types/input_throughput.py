# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.input_sequence_length_metric import InputSequenceLengthMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric


class InputThroughputMetric(BaseRecordMetric[float]):
    """
    Post-processor for calculating Input Throughput metrics from records. This is only applicable to streaming responses.

    Formula:
        Input Throughput = Input Sequence Length / Time to First Token (seconds)
    """

    tag = "input_throughput"
    header = "Input Throughput"
    short_header = "Input TPS"
    short_header_hide_unit = True
    unit = MetricOverTimeUnit.TOKENS_PER_SECOND
    flags = (
        MetricFlags.STREAMING_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.HIDDEN  # Hidden for now, as it is new and not yet validated
    )
    required_metrics = {
        InputSequenceLengthMetric.tag,
        TTFTMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """This method calculates the input throughput by dividing the input sequence length by the TTFT."""

        isl = record_metrics[InputSequenceLengthMetric.tag]
        ttft = record_metrics[TTFTMetric.tag]
        if ttft is None or ttft == 0:
            raise ValueError("Time to first token is not available for the record.")
        converted_ttft = record_metrics.get_converted(TTFTMetric, self.unit.time_unit)  # type: ignore
        return isl / converted_ttft  # type: ignore
