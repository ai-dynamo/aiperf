# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric


class CompletedRequestCountMetric(BaseDerivedMetric[int]):
    """Successful plus failed requests that completed the benchmark pipeline.

    Distinct from :class:`RequestCountMetric`, which counts only valid (successful)
    inference results used for latency/token distributions. Surfaces the total
    completion volume so consumers can compute error rate without re-summing.

    See https://github.com/ai-dynamo/aiperf/issues/688.
    """

    tag = "completed_request_count"
    header = "Completed Requests (Success + Error)"
    short_header = "Completed"
    short_header_hide_unit = True
    unit = GenericMetricUnit.REQUESTS
    display_order = 1075
    flags = MetricFlags.NO_INDIVIDUAL_RECORDS
    # Both dependencies are declared so MetricRegistry's dependency-order
    # validator (``create_dependency_order_for``) ensures they are computed
    # before this metric runs. ``ErrorRequestCountMetric`` may legitimately
    # be absent (zero-error workloads); the derive falls back to 0 in that
    # case via ``.get(..., 0)``.
    required_metrics = frozenset(
        {
            RequestCountMetric.tag,
            ErrorRequestCountMetric.tag,
        }
    )

    def _derive_value(self, metric_results: MetricResultsDict) -> int:
        successes = int(metric_results.get_or_raise(RequestCountMetric))
        errors = int(metric_results.get(ErrorRequestCountMetric.tag, 0) or 0)
        return successes + errors
