# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric


class RequestErrorRateMetric(BaseDerivedMetric[float]):
    """Percentage of completed requests that ended in error.

    Uses :class:`ErrorRequestCountMetric` and :class:`CompletedRequestCountMetric`
    so latency percentiles (computed on successes only) can be read alongside
    the operational error rate.
    """

    tag = "request_error_rate"
    header = "Request Error Rate"
    short_header = "Err %"
    short_header_hide_unit = True
    unit = GenericMetricUnit.PERCENT
    display_order = 1080
    flags = MetricFlags.NO_INDIVIDUAL_RECORDS
    required_metrics = {
        RequestCountMetric.tag,
    }

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        successes = int(metric_results.get_or_raise(RequestCountMetric))
        errors = int(metric_results.get(ErrorRequestCountMetric.tag, 0) or 0)
        total = successes + errors
        if total <= 0:
            raise NoMetricValue("No completed requests for error rate")
        return 100.0 * errors / total
