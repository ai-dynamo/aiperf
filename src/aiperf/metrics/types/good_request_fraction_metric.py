# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricConsoleGroup, MetricFlags
from aiperf.common.exceptions import NoMetricValue
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

    Note on `error_request_count`: it is *not* declared in
    `required_metrics` because it carries `MetricFlags.ERROR_ONLY`,
    meaning the upstream aggregate counter only emits a value when at
    least one error record is observed. On clean runs (zero errors)
    the metric is absent from `metric_results` entirely, so this
    derivation reads it opportunistically and treats a missing tag as
    `0`. Declaring it required would cause `_check_metrics` to raise
    `NoMetricValue` on every clean run and the framework would silently
    drop `good_request_fraction` from output -- the opposite of the
    desired SLA-gate behavior.

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
    required_metrics = {GoodRequestCountMetric.tag, RequestCountMetric.tag}

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        good = metric_results.get(GoodRequestCountMetric.tag)
        valid = metric_results.get(RequestCountMetric.tag)
        if good is None or valid is None:
            raise NoMetricValue(
                "good_request_fraction requires both good_request_count and request_count"
            )
        # error_request_count is ERROR_ONLY: absent on clean runs.
        errors = metric_results.get(ErrorRequestCountMetric.tag) or 0
        attempted = float(valid) + float(errors)
        if attempted == 0:
            return 0.0
        return float(good) / attempted
