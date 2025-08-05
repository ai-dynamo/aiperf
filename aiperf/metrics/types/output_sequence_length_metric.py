# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class OutputSequenceLengthMetric(BaseRecordMetric[int]):
    """
    Post-processor for calculating Output Sequence Length (OSL) metrics from records.

    Formula:
        Output Sequence Length = Sum(Output Token Counts)
    """

    tag = "output_sequence_length"
    header = "Output Sequence Length"
    unit = GenericMetricUnit.TOKENS
    flags = MetricFlags.PRODUCES_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        This method extracts the output token count from the record and returns it.

        Raises:
            ValueError: If the record does not have an output token count.
        """
        if record.output_token_count is None:
            raise ValueError("Output token count is missing in the record.")

        return record.output_token_count
