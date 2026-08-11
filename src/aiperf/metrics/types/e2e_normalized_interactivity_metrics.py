# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E Normalized Interactivity metrics (AgentX / InferenceX Pareto x-axis).

E2E Normalized Interactivity is the per-request rate at which a user receives
output tokens INCLUDING the prefill wait: ``output_sequence_length /
request_latency``. Unlike ``output_token_throughput_per_user`` (``1 / ITL``,
decode-only), it charges the whole end-to-end latency, so a large TTFT relative
to the tokens produced pulls it down -- prefill-delaying cannot inflate it.

To reproduce InferenceX's number exactly, the percentile is taken over the
seconds-per-output-token ratio ``request_latency / OSL`` and then inverted:

    p90 interactivity = 1 / p90(request_latency_s / OSL)

This is the slow-tail convention InferenceX enforces for interactivity
(``1 / p(x)``, NOT ``p(1 / x)``): a higher percentile names a slower, more
pessimistic tail, so ``p90`` is the effective token rate of the 90th-percentile
worst request. Inverting the ratio's percentile (rather than percentiling the
per-request rate directly) is required because ``1 / p(x) != p(1 / x)`` -- the
two land the interpolation in different spaces. It also means the family cannot
be expressed as a monotonic percentile band (inversion flips the order), so it
is emitted as named scalars, mirroring InferenceX's ``p75``/``p90`` stored
values.

The values are injected post-aggregation from the ``request_latency`` and
``output_sequence_length`` columns by
:func:`aiperf.metrics.e2e_normalized_interactivity_analyzer.inject_e2e_normalized_interactivity_metrics`,
mirroring the replay-send-lag and derived-latency families.
"""

from __future__ import annotations

from typing import ClassVar

from aiperf.common.enums import (
    MetricConsoleGroup,
    MetricFlags,
    MetricOverTimeUnit,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric


class _E2ENormalizedInteractivityDeferMixin:
    """Deferred-derivation behavior for the injected interactivity metrics.

    Not a metric itself (does not subclass BaseMetric) so it is never
    registered. The whole family is ``1 / p(request_latency / OSL)`` over the
    per-record columns, which the scalar summarize path cannot express, so the
    derive defers and
    :func:`aiperf.metrics.e2e_normalized_interactivity_analyzer.inject_e2e_normalized_interactivity_metrics`
    fills the values post-aggregation from the column store.
    """

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        raise NoMetricValue(
            f"{self.tag} is injected post-aggregation from the "  # type: ignore[attr-defined]
            "request_latency / output_sequence_length columns"
        )


class E2ENormalizedInteractivityPercentileBase(
    _E2ENormalizedInteractivityDeferMixin, BaseDerivedMetric[float]
):
    """Shared metadata for the E2E Normalized Interactivity percentile metrics.

    ``percentile`` is the single source of truth for both the metric's identity
    and the ratio percentile the analyzer inverts.
    """

    __is_abstract__ = True
    percentile: float

    unit = MetricOverTimeUnit.TOKENS_PER_SECOND_PER_USER
    flags = MetricFlags.PRODUCES_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.NONE
    timeslice_derivable = False
    required_metrics: ClassVar[set[str]] = {
        RequestLatencyMetric.tag,
        OutputSequenceLengthMetric.tag,
    }


class E2ENormalizedInteractivityP90Metric(E2ENormalizedInteractivityPercentileBase):
    """P90 (slow-tail) E2E Normalized Interactivity, in tokens/sec/user.

    Formula:
        1 / p90(request_latency_s / output_sequence_length)
    """

    __is_abstract__ = False
    percentile = 90.0
    tag = "e2e_normalized_interactivity_p90"
    header = "E2E Normalized Interactivity p90"
    short_header = "E2E Norm Intvty p90"


class E2ENormalizedInteractivityP75Metric(E2ENormalizedInteractivityPercentileBase):
    """P75 (slow-tail) E2E Normalized Interactivity, in tokens/sec/user.

    Formula:
        1 / p75(request_latency_s / output_sequence_length)
    """

    __is_abstract__ = False
    percentile = 75.0
    tag = "e2e_normalized_interactivity_p75"
    header = "E2E Normalized Interactivity p75"
    short_header = "E2E Norm Intvty p75"
