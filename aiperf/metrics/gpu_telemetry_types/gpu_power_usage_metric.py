# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.metrics.base_telemetry_metric import BaseTelemetryMetric
from aiperf.common.enums.metric_enums import PowerMetricUnit, MetricFlags
from aiperf.common.models.telemetry_models import TelemetryRecord

class GpuPowerUsageMetric(BaseTelemetryMetric[float]):
    tag = "gpu_power_usage"
    header = "GPU Power Usage"
    unit = PowerMetricUnit.WATT
    display_order = None
    required_metrics = None
    flags = MetricFlags.GPU_TELEMETRY

    def _extract_value(self, record: TelemetryRecord) -> float | None:
        return record.gpu_power_usage
