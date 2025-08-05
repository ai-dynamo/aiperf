# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.common.models.record_models import ParsedResponseRecord
from aiperf.metrics.base_record_metric import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.inter_token_latency_metric import InterTokenLatencyMetric


class OutputInferenceSpeedMetric(BaseRecordMetric[float]):
    """
    Post Processor for calculating Output Inference Speed Metric.

    Formula:
        Output Inference Speed = 1 / Inter-Token Latency (seconds)
    """

    tag = "output_inference_speed"
    header = "Output Inference Speed"
    unit = MetricOverTimeUnit.TOKENS_PER_SECOND
    flags = MetricFlags.STREAMING_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    required_metrics = {
        InterTokenLatencyMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """This method calculates the output inference speed by computing the inverse of the inter-token latency."""
        converted_itl = record_metrics.get_converted(
            InterTokenLatencyMetric,
            self.unit.time_unit,  # type: ignore
        )
        if converted_itl == 0:
            raise ValueError(
                "Inter-token latency is 0, cannot compute output inference speed."
            )
        return 1 / converted_itl
