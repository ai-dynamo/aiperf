# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.metrics.base_telemetry_metric import BaseTelemetryMetric
from aiperf.common.enums.metric_enums import MetricSizeUnit, MetricFlags
from aiperf.common.models.telemetry_models import TelemetryRecord

class GpuMemoryUsedMetric(BaseTelemetryMetric[float]):
    tag = "gpu_memory_used"
    header = "GPU Memory Used"
    unit = MetricSizeUnit.GIGABYTES
    display_order = None
    required_metrics = None
    flags = MetricFlags.GPU_TELEMETRY

    def _extract_value(self, record: TelemetryRecord) -> float | None:
        return record.gpu_memory_used
