# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Network-RTT-adjusted latency metrics.

When network latency calibration is enabled, the mean client-to-endpoint RTT is
subtracted from the request-start-anchored latency metrics to produce these
``network_adjusted_*`` variants, making runs taken from different network locations
comparable. The raw metrics are always preserved.

These metrics are NOT computed per-record. The RTT is a run-level scalar that is only
known at the end of the run, so :class:`aiperf.post_processors.metric_results_processor`
injects the shifted distributions after aggregation (subtracting a constant shifts every
percentile and the mean by that constant and leaves the standard deviation unchanged).
``_derive_value`` therefore always defers.

Inter-token / inter-chunk latencies are intentionally NOT adjusted: the same RTT cancels
in ``(request_latency - ttft)``, so those metrics are already network-invariant.
"""

from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.types import MetricTagT
from aiperf.metrics import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.time_to_first_output_token_metric import (
    TimeToFirstOutputTokenMetric,
)
from aiperf.metrics.types.ttft_metric import TTFTMetric


class _NetworkAdjustedMixin:
    """Shared metadata and deferred-derivation behavior for injected network metrics.

    Not a metric itself (does not subclass BaseMetric), so it is never registered and
    does not interfere with the value-type auto-detection that reads the concrete
    class's parameterized ``BaseDerivedMetric[...]`` base.
    """

    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS

    def _derive_value(self, metric_results: MetricResultsDict):
        raise NoMetricValue(
            f"{self.tag} is injected post-aggregation by the network latency transform"  # type: ignore[attr-defined]
        )


class NetworkAdjustedRequestLatencyMetric(
    _NetworkAdjustedMixin, BaseDerivedMetric[int]
):
    tag = "network_adjusted_request_latency"
    header = "Network-Adjusted Request Latency"
    short_header = "Net-Adj Req Latency"
    display_order = 301
    flags = MetricFlags.NONE


class NetworkAdjustedTTFTMetric(_NetworkAdjustedMixin, BaseDerivedMetric[int]):
    tag = "network_adjusted_time_to_first_token"
    header = "Network-Adjusted Time to First Token"
    short_header = "Net-Adj TTFT"
    display_order = 101
    flags = MetricFlags.STREAMING_TOKENS_ONLY


class NetworkAdjustedTimeToFirstOutputTokenMetric(
    _NetworkAdjustedMixin, BaseDerivedMetric[int]
):
    tag = "network_adjusted_time_to_first_output_token"
    header = "Network-Adjusted Time to First Output Token"
    short_header = "Net-Adj TTFO"
    display_order = 211
    flags = MetricFlags.STREAMING_TOKENS_ONLY | MetricFlags.SUPPORTS_REASONING


class NetworkRttMetric(_NetworkAdjustedMixin, BaseDerivedMetric[float]):
    """The mean network RTT that was subtracted from the adjusted latency metrics.

    Single-value summary metric (no per-record distribution).
    """

    tag = "network_rtt"
    header = "Network RTT"
    short_header = "Net RTT"
    display_order = 305
    flags = MetricFlags.NO_INDIVIDUAL_RECORDS


NETWORK_ADJUSTED_SOURCES: dict[MetricTagT, MetricTagT] = {
    NetworkAdjustedRequestLatencyMetric.tag: RequestLatencyMetric.tag,
    NetworkAdjustedTTFTMetric.tag: TTFTMetric.tag,
    NetworkAdjustedTimeToFirstOutputTokenMetric.tag: TimeToFirstOutputTokenMetric.tag,
}
"""Maps each network_adjusted_* tag to the source latency metric it shifts.

Only metrics whose interval STARTS at the client request-send timestamp carry the
network RTT. Time to Second Token (second_response - first_response) and the
inter-token / inter-chunk latencies are intra-stream gaps that do not include the RTT,
so they are intentionally excluded.
"""
