# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.metric_enums import GenericMetricUnit, MetricFlags
from aiperf.common.models import TelemetryRecord
from aiperf.metrics.base_telemetry_metric import BaseTelemetryMetric


class MemoryCopyUtilizationMetric(BaseTelemetryMetric[float]):
    """GPU memory copy utilization metric.

    Tracks the memory copy engine utilization as a percentage (0-100).
    This metric indicates how busy the GPU's memory copy engines are.
    """

    tag = "memory_copy_utilization"
    header = "Memory Copy Utilization"
    unit = GenericMetricUnit.PERCENT
    display_order = None
    required_metrics = None
    flags = MetricFlags.GPU_TELEMETRY

    def _extract_value(self, record: TelemetryRecord) -> float | None:
        """Extract memory copy utilization from telemetry record.

        Args:
            record: The telemetry record containing GPU metrics

        Returns:
            Memory copy utilization percentage, or None if not available
        """
        return record.telemetry_data.memory_copy_utilization
