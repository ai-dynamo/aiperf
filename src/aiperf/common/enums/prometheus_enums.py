# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from typing_extensions import Self

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class PrometheusMetricType(CaseInsensitiveStrEnum):
    """Prometheus metric types as defined in the Prometheus exposition format.

    See: https://prometheus.io/docs/concepts/metric_types/
    """

    COUNTER = "counter"
    """Counter: A cumulative metric that represents a single monotonically increasing counter."""

    GAUGE = "gauge"
    """Gauge: A metric that represents a single numerical value that can arbitrarily go up and down."""

    HISTOGRAM = "histogram"
    """Histogram: Samples observations and counts them in configurable buckets."""

    SUMMARY = "summary"
    """Summary: Not supported for collection (quantiles are cumulative over server lifetime).

    Note: Summary metrics are intentionally skipped during collection because their
    quantiles are computed cumulatively over the entire server lifetime, making them
    unsuitable for benchmark-specific analysis. No major LLM inference servers use
    Summary metrics - they all use Histograms instead.
    """

    UNKNOWN = "unknown"
    """Unknown: Untyped metric (prometheus_client uses 'unknown' instead of 'untyped')."""

    @classmethod
    def _missing_(cls, value: Any) -> Self:
        """Handle unrecognized metric type values by returning UNKNOWN.

        Called automatically when constructing a PrometheusMetricType with a value
        that doesn't match any defined member. Attempts case-insensitive matching
        via parent class, then falls back to UNKNOWN for unrecognized types.

        This ensures robust parsing of Prometheus metrics where servers may expose
        non-standard or future metric types.

        Args:
            value: The value to match against enum members

        Returns:
            Matching enum member (case-insensitive) or PrometheusMetricType.UNKNOWN
        """
        try:
            return super()._missing_(value)
        except ValueError:
            return cls.UNKNOWN
