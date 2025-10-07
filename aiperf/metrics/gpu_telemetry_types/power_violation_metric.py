# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.metric_enums import MetricFlags, MetricTimeUnit
from aiperf.common.models import TelemetryRecord
from aiperf.metrics.base_telemetry_metric import BaseTelemetryMetric


class PowerViolationMetric(BaseTelemetryMetric[float]):
    """GPU power violation metric.

    Tracks the throttling duration due to power constraints in microseconds.
    This metric indicates how long the GPU has been throttled due to power limits.
    """

    tag = "power_violation"
    header = "Power Violation"
    unit = MetricTimeUnit.MICROSECONDS
    display_order = None
    required_metrics = None
    flags = MetricFlags.GPU_TELEMETRY

    def _extract_value(self, record: TelemetryRecord) -> float | None:
        """Extract power violation duration from telemetry record.

        Args:
            record: The telemetry record containing GPU metrics

        Returns:
            Power violation duration in microseconds, or None if not available
        """
        return record.telemetry_data.power_violation
