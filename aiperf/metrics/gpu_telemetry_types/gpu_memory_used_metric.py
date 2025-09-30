# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricSizeUnit
from aiperf.common.models import TelemetryRecord
from aiperf.metrics.base_telemetry_metric import BaseTelemetryMetric


class GpuMemoryUsedMetric(BaseTelemetryMetric[float]):
    tag = "gpu_memory_used"
    header = "GPU Memory Used"
    unit = MetricSizeUnit.GIGABYTES
    display_order = None
    required_metrics = None
    flags = MetricFlags.GPU_TELEMETRY

    def _extract_value(self, record: TelemetryRecord) -> float | None:
        """
        Extracts the GPU memory used value from a telemetry record.
        
        Parameters:
            record (TelemetryRecord): Telemetry record containing GPU metrics.
        
        Returns:
            float | None: GPU memory used in gigabytes, or `None` if the value is not present.
        """
        return record.gpu_memory_used
