# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.input_sequence_length_metric import InputSequenceLengthMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric


class PrefillThroughputPerUserMetric(BaseRecordMetric[float]):
    """
    Post-processor for calculating Prefill Throughput Per User metrics from records. This is only applicable to streaming responses.

    Formula:
        Prefill Throughput Per User = Prefill Sequence Length / Time to First Token (seconds)
    """

    tag = "prefill_throughput_per_user"
    header = "Prefill Throughput Per User"
    short_header = "Prefill TPS/User"
    short_header_hide_unit = True
    unit = MetricOverTimeUnit.TOKENS_PER_SECOND_PER_USER
    flags = (
        MetricFlags.STREAMING_TOKENS_ONLY
        | MetricFlags.TOKENIZES_INPUT_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
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
        """This method calculates the prefill throughput per user by dividing the input sequence length by the TTFT."""

        isl = record_metrics.get_or_raise(InputSequenceLengthMetric)
        converted_ttft = record_metrics.get_converted_or_raise(
            TTFTMetric,
            self.unit.time_unit,  # type: ignore
        )
        if converted_ttft == 0:
            raise NoMetricValue(
                "TTFT is zero, cannot calculate prefill throughput per user metric"
            )
        return isl / converted_ttft  # type: ignore
