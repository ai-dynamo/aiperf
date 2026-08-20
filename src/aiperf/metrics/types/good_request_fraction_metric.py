# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricConsoleGroup, MetricFlags
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.good_request_count_metric import GoodRequestCountMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric


class GoodRequestFractionMetric(BaseDerivedMetric[float]):
    """Fraction of all attempted requests that satisfied every per-request SLO.

    Formula:
        good_request_fraction = good_request_count
                              / (request_count + error_request_count)

    The denominator counts failed requests when they exist so the SLA
    gate penalises runs that drop traffic, not just runs that violate
    latency SLOs; otherwise a backend that errors out under load would
    look "good" simply because the survivors stayed under the latency
    budget.

    None of the counters is required because each can legitimately be absent:
    valid-request counters on an all-error run, and the error counter on a clean
    run. Deliberate client cancellations use their own counter and do not enter
    this denominator.

    Returns 0.0 when no requests were attempted (denominator == 0).
    Used by the `max-goodput-under-slo` search recipe as the
    SLA-feasibility gate (`good_request_fraction:avg:ge:<attainment>`);
    without this derived metric the recipe SLA filter dereferences a
    missing metric_tag and BO treats every iteration as infeasible.
    """

    tag = "good_request_fraction"
    header = "GoodRequestFraction"
    short_header = "GoodReqFrac"
    short_header_hide_unit = True
    unit = GenericMetricUnit.RATIO
    flags = MetricFlags.GOODPUT | MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.NONE
    required_metrics = None

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        good = int(metric_results.get(GoodRequestCountMetric.tag, 0) or 0)
        valid = int(metric_results.get(RequestCountMetric.tag, 0) or 0)
        errors = int(metric_results.get(ErrorRequestCountMetric.tag, 0) or 0)
        attempted = valid + errors
        if attempted == 0:
            return 0.0
        return good / attempted
