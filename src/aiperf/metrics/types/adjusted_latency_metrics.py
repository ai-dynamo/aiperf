# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.inter_token_latency_metric import InterTokenLatencyMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric


class BaseAdjustedLatencyMetric(BaseDerivedMetric[float]):
    """Synthetic adjusted latency distribution emitted from a parent record metric."""

    __is_abstract__ = True
    source_metric_tag: ClassVar[str]
    flags = MetricFlags.NO_INDIVIDUAL_RECORDS | MetricFlags.SYNTHETIC
    required_metrics = None

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        raise NoMetricValue(
            f"{self.tag} is emitted by MetricResultsProcessor summary logic"
        )


class AdjustedRequestLatencyMetric(BaseAdjustedLatencyMetric):
    """Request latency with failed requests modeled as unbounded latency."""

    __is_abstract__ = False
    tag = f"adj_{RequestLatencyMetric.tag}"
    source_metric_tag = RequestLatencyMetric.tag
    header = "Adjusted Request Latency"
    short_header = "Adj Req Latency"
    unit = RequestLatencyMetric.unit
    display_unit = RequestLatencyMetric.display_unit
    display_order = (RequestLatencyMetric.display_order or 0) + 1


class AdjustedTTFTMetric(BaseAdjustedLatencyMetric):
    """TTFT with failed requests modeled as unbounded latency."""

    __is_abstract__ = False
    tag = f"adj_{TTFTMetric.tag}"
    source_metric_tag = TTFTMetric.tag
    header = "Adjusted Time to First Token"
    short_header = "Adj TTFT"
    unit = TTFTMetric.unit
    display_unit = TTFTMetric.display_unit
    display_order = (TTFTMetric.display_order or 0) + 1
    flags = (
        BaseAdjustedLatencyMetric.flags
        | MetricFlags.STREAMING_ONLY
        | MetricFlags.PRODUCES_TOKENS_ONLY
    )


class AdjustedInterTokenLatencyMetric(BaseAdjustedLatencyMetric):
    """ITL with failed requests modeled as unbounded latency."""

    __is_abstract__ = False
    tag = f"adj_{InterTokenLatencyMetric.tag}"
    source_metric_tag = InterTokenLatencyMetric.tag
    header = "Adjusted Inter Token Latency"
    short_header = "Adj ITL"
    unit = InterTokenLatencyMetric.unit
    display_unit = InterTokenLatencyMetric.display_unit
    display_order = (InterTokenLatencyMetric.display_order or 0) + 1
    flags = (
        BaseAdjustedLatencyMetric.flags
        | MetricFlags.STREAMING_ONLY
        | MetricFlags.PRODUCES_TOKENS_ONLY
    )
