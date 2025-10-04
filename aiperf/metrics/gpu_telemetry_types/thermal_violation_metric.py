# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.metric_enums import MetricFlags, MetricTimeUnit
from aiperf.common.models import TelemetryRecord
from aiperf.metrics.base_telemetry_metric import BaseTelemetryMetric


class ThermalViolationMetric(BaseTelemetryMetric[float]):
    """GPU thermal violation metric.

    Tracks the throttling duration due to thermal constraints in microseconds.
    This metric indicates how long the GPU has been throttled due to temperature limits.
    """

    tag = "thermal_violation"
    header = "Thermal Violation"
    unit = MetricTimeUnit.MICROSECONDS
    display_order = None
    required_metrics = None
    flags = MetricFlags.GPU_TELEMETRY

    def _extract_value(self, record: TelemetryRecord) -> float | None:
        """Extract thermal violation duration from telemetry record.

        Args:
            record: The telemetry record containing GPU metrics

        Returns:
            Thermal violation duration in microseconds, or None if not available
        """
        return record.telemetry_data.thermal_violation
