# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.metric_enums import MetricFlags, PowerMetricUnit
from aiperf.common.models import TelemetryRecord
from aiperf.metrics.base_telemetry_metric import BaseTelemetryMetric


class GpuPowerUsageMetric(BaseTelemetryMetric[float]):
    tag = "gpu_power_usage"
    header = "GPU Power Usage"
    unit = PowerMetricUnit.WATT
    display_order = None
    required_metrics = None
    flags = MetricFlags.GPU_TELEMETRY

    def _extract_value(self, record: TelemetryRecord) -> float | None:
        """
        Extracts GPU power usage from a TelemetryRecord.
        
        Parameters:
            record (TelemetryRecord): Telemetry record containing GPU telemetry fields.
        
        Returns:
            float | None: GPU power usage in watts, or None if the value is not present.
        """
        return record.gpu_power_usage
