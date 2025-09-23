# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.metric_enums import MetricFlags, TemperatureMetricUnit
from aiperf.common.models import TelemetryRecord
from aiperf.metrics.base_telemetry_metric import BaseTelemetryMetric


class MemoryTemperatureMetric(BaseTelemetryMetric[float]):
    """Memory temperature metric from GPU telemetry.

    Tracks the memory temperature in degrees Celsius.
    """
    tag = "memory_temperature"
    header = "Memory Temperature"
    unit = TemperatureMetricUnit.CELSIUS
    display_order = None
    required_metrics = None
    flags = MetricFlags.GPU_TELEMETRY

    def _extract_value(self, record: TelemetryRecord) -> float | None:
        return record.memory_temperature