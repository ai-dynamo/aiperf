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
    """Summary: Similar to histogram, samples observations and provides quantiles."""

    INFO = "info"
    """Info: A gauge metric that contains various metadata via labels."""

    UNKNOWN = "unknown"
    """Unknown: Untyped metric (prometheus_client uses 'unknown' instead of 'untyped')."""

    @classmethod
    def _missing_(cls, value: Any) -> Self:
        """
        Handles cases where a value is not directly found in the enumeration.

        This method is called when an attempt is made to access an enumeration
        member using a value that does not directly match any of the defined
        members. It provides custom logic to handle such cases.

        Returns:
            The matching enumeration member if a case-insensitive match is found
            for string values; otherwise, returns PrometheusMetricType.UNKNOWN.
        """
        try:
            return super()._missing_(value)
        except ValueError:
            return cls.UNKNOWN
