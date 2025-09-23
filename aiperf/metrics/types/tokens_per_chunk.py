# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags
from aiperf.common.enums.metric_enums import GenericMetricUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)


class TokensPerChunkMetric(BaseRecordMetric[float]):
    """
    Post Processor for calculating Tokens Per Chunk (TPC) metric.
    This is the average number of tokens per chunk for a streaming response.

    Formula:
        Tokens Per Chunk = Output Sequence Length / Count(Responses)
    """

    tag = "tokens_per_chunk"
    header = "Tokens Per Chunk"
    short_header = "TPC"
    unit = GenericMetricUnit.TOKENS
    flags = MetricFlags.STREAMING_TOKENS_ONLY | MetricFlags.HIDDEN
    required_metrics = {
        OutputSequenceLengthMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """
        Calculates the Tokens Per Chunk (TPC) metric.
        """
        osl = record_metrics.get_or_raise(OutputSequenceLengthMetric)
        if osl < 1:  # type: ignore
            raise NoMetricValue(
                f"Output sequence length must be at least 1 for Tokens Per Chunk Metric, got {osl}"
            )
        if len(record.responses) < 2:
            raise NoMetricValue(
                f"Record must have at least 2 responses for Tokens Per Chunk Metric, got {len(record.responses)}"
            )

        return osl / len(record.responses)  # type: ignore
