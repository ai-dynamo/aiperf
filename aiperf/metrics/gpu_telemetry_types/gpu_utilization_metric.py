# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.metric_enums import GenericMetricUnit, MetricFlags
from aiperf.common.models import TelemetryRecord
from aiperf.metrics.base_telemetry_metric import BaseTelemetryMetric


class GpuUtilizationMetric(BaseTelemetryMetric[float]):
    tag = "gpu_utilization"
    header = "GPU Utilization"
    unit = GenericMetricUnit.PERCENT
    display_order = None
    required_metrics = None
    flags = MetricFlags.GPU_TELEMETRY

    def _extract_value(self, record: TelemetryRecord) -> float | None:
        """
        Extracts the GPU utilization percentage from a telemetry record.
        
        Parameters:
            record (TelemetryRecord): Telemetry record containing GPU telemetry fields.
        
        Returns:
            float | None: GPU utilization percentage from the record, or `None` if not available.
        """
        return record.gpu_utilization
