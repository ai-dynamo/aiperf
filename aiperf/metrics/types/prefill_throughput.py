# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.input_sequence_length_metric import InputSequenceLengthMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric


class PrefillThroughputMetric(BaseRecordMetric[float]):
    """
    Post-processor for calculating Prefill Throughput metrics from records. This is only applicable to streaming responses.

    Formula:
        Prefill Throughput = Prefill Sequence Length / Time to First Token (seconds)
    """

    tag = "prefill_throughput"
    header = "Prefill Throughput"
    short_header = "Prefill TPS"
    short_header_hide_unit = True
    unit = MetricOverTimeUnit.TOKENS_PER_SECOND
    flags = (
        MetricFlags.STREAMING_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.EXPERIMENTAL
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
        """This method calculates the prefill throughput by dividing the input sequence length by the TTFT."""

        isl = record_metrics.get_or_raise(InputSequenceLengthMetric)
        converted_ttft = record_metrics.get_converted_or_raise(
            TTFTMetric,
            self.unit.time_unit,  # type: ignore
        )
        return isl / converted_ttft  # type: ignore
