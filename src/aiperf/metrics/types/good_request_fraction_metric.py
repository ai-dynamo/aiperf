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

    Client-cancelled requests (``--request-cancellation-rate``) are excluded
    from the denominator: the record-processor routes them to
    :class:`CancelledRequestCountMetric` (``CANCELLED_ONLY``) rather than the
    error path, so they never reach ``error_request_count`` or ``request_count``.
    Deliberate cancellations therefore do not deflate goodput, matching the
    credit-side ``cancelled`` bucket.

    No counter is declared in `required_metrics` because each one can be
    legitimately absent. `error_request_count` is `MetricFlags.ERROR_ONLY`,
    so it is absent on a clean zero-error run. `good_request_count` and
    `request_count` are computed only for *valid* records, so on a 100%-fail
    run (every record an error) both are absent from `metric_results`
    entirely. Declaring any of them required would make `_check_metrics`
    raise `NoMetricValue` and the framework would silently drop
    `good_request_fraction` from output -- the opposite of the desired
    SLA-gate behavior, and worst on the very runs (all traffic failing)
    where the gate should report 0.0. All three counters are read with a
    `0` default.

    Returns 0.0 when no requests were attempted (denominator == 0) and on a
    fully-failed run (no good requests, denominator == error count).
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
        # good_request_count / request_count are absent on a 100%-fail run
        # (valid-only counters); error_request_count is ERROR_ONLY and absent
        # on a clean run. All three default to 0 so the gate never vanishes.
        good = int(metric_results.get(GoodRequestCountMetric.tag, 0) or 0)
        valid = int(metric_results.get(RequestCountMetric.tag, 0) or 0)
        errors = int(metric_results.get(ErrorRequestCountMetric.tag, 0) or 0)
        attempted = valid + errors
        if attempted == 0:
            return 0.0
        return good / attempted
