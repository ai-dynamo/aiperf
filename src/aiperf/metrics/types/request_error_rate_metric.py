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

    Uses :class:`RequestCountMetric` for valid requests and
    :class:`ErrorRequestCountMetric` for failed requests. Because
    ``ErrorRequestCountMetric`` carries ``MetricFlags.ERROR_ONLY``, clean runs
    omit it instead of emitting zero; an omitted error count is therefore treated
    as zero.

    Latency percentiles are computed on successful requests only. Read this metric
    alongside the ``adj_*`` percentile band on metrics flagged with
    ``MetricFlags.PERCENTILE_INCLUDES_FAILED_REQUESTS`` for a fuller picture of
    failure-contaminated tail behavior.

    See https://github.com/ai-dynamo/aiperf/issues/688.
    """

    tag = "request_error_rate"
    header = "Request Error Rate"
    short_header = "Err %"
    short_header_hide_unit = True
    unit = GenericMetricUnit.PERCENT
    display_order = 1080
    flags = MetricFlags.NO_INDIVIDUAL_RECORDS
    required_metrics = frozenset({RequestCountMetric.tag})

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        successes = int(metric_results.get_or_raise(RequestCountMetric))
        errors = int(metric_results.get(ErrorRequestCountMetric.tag, 0) or 0)
        total = successes + errors
        if total <= 0:
            raise NoMetricValue("No completed requests for error rate")
        return 100.0 * errors / total
