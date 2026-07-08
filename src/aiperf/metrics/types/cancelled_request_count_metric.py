# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricConsoleGroup, MetricFlags
from aiperf.metrics.base_aggregate_counter_metric import BaseAggregateCounterMetric


class CancelledRequestCountMetric(BaseAggregateCounterMetric[int]):
    """Total number of client-cancelled requests processed by the benchmark.

    Incremented once per record whose request was cancelled client-side
    (``--request-cancellation-rate`` disconnection, surfaced as a code-499
    ``RequestCancellationError``). Mirrors the credit-side ``cancelled`` bucket
    (``CreditCounter``) so the metrics export agrees with the PhaseRunner log:
    a 40%-cancel/0-error run reports ``cancelled_request_count == 32`` alongside
    ``error_request_count == 0``, instead of counting the deliberate
    cancellations as server errors.

    A cancellation is deliberately NOT a server error, so this counter is kept
    separate from :class:`ErrorRequestCountMetric`. The ``CANCELLED_ONLY`` flag
    routes ``was_cancelled`` records here (via ``MetricRecordProcessor``) instead
    of the error path, which also keeps cancellations out of the
    ``request_error_rate`` numerator/denominator and the ``good_request_fraction``
    denominator.

    Formula:
        ```
        Cancelled Request Count = Sum(Cancelled Requests)
        ```
    """

    tag = "cancelled_request_count"
    header = "Cancelled Request Count"
    short_header = "Cancelled Count"
    short_header_hide_unit = True
    unit = GenericMetricUnit.REQUESTS
    display_order = 1076
    flags = MetricFlags.CANCELLED_ONLY | MetricFlags.NO_INDIVIDUAL_RECORDS
    console_group = MetricConsoleGroup.NONE
    required_metrics = None
