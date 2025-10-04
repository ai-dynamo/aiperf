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
        return record.telemetry_data.gpu_power_usage
