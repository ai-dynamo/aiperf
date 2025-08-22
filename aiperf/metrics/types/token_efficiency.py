# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.output_token_count import OutputTokenCountMetric
from aiperf.metrics.types.reasoning_token_count import ReasoningTokenCountMetric


class TokenEfficiencyMetric(BaseRecordMetric[float]):
    """
    Post-processor for calculating Token Efficiency metrics from records.

    Formula:
        Token Efficiency = Total Reasoning Tokens / Total Output Tokens

    References:
        @misc{lrm_token_economy_2025,
            title={Measuring Thinking Efficiency in Reasoning Models: The Missing Benchmark},
            author={TSB},
            year={2025},
            month={August},
            url={https://github.com/cpldcpu/LRMTokenEconomy}
        }
    """

    tag = "token_efficiency"
    header = "Token Efficiency"
    short_header_hide_unit = True
    unit = GenericMetricUnit.RATIO
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.SUPPORTS_REASONING
        | MetricFlags.EXPERIMENTAL
    )
    required_metrics = {
        ReasoningTokenCountMetric.tag,
        OutputTokenCountMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        reasoning_token_count = record_metrics.get(ReasoningTokenCountMetric.tag)
        if not reasoning_token_count:
            raise NoMetricValue("Reasoning token count is missing in the record.")

        output_token_count = record_metrics.get(OutputTokenCountMetric.tag)
        if not output_token_count:
            raise NoMetricValue("Output token count is missing in the record.")

        return reasoning_token_count / output_token_count  # type: ignore
