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

    Reads :class:`ErrorRequestCountMetric` and :class:`RequestCountMetric`
    so latency percentiles (computed on successes only) can be read alongside
    the operational error rate. Pair with the ``adj_*`` percentile band on
    flagged latency metrics (see ``MetricFlags.PERCENTILE_INCLUDES_FAILED_REQUESTS``)
    for a full picture of failure-contaminated tail behavior.

    See https://github.com/ai-dynamo/aiperf/issues/688.
    """

    tag = "request_error_rate"
    header = "Request Error Rate"
    short_header = "Err %"
    short_header_hide_unit = True
    unit = GenericMetricUnit.PERCENT
    display_order = 1080
    flags = MetricFlags.NO_INDIVIDUAL_RECORDS
    # Neither counter is declared required. request_count (successes) is absent
    # when 100% of requests fail, and error_request_count (ERROR_ONLY) is absent
    # on a clean zero-error run. Declaring either in required_metrics would make
    # _check_metrics raise NoMetricValue and silently drop this metric from the
    # export exactly when it matters most -- a fully-down server should report a
    # 100% error rate, not omit it. Both counters are read with a 0 default; the
    # total <= 0 guard below still covers a truly-empty run.
    required_metrics = None

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        successes = int(metric_results.get(RequestCountMetric.tag, 0) or 0)
        errors = int(metric_results.get(ErrorRequestCountMetric.tag, 0) or 0)
        total = successes + errors
        if total <= 0:
            raise NoMetricValue("No completed requests for error rate")
        return 100.0 * errors / total
