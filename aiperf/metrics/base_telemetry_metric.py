# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Generic

from aiperf.common.enums import MetricType, MetricValueTypeVarT
from aiperf.common.models.telemetry_models import TelemetryRecord
from aiperf.metrics.base_metric import BaseMetric
from aiperf.metrics.metric_dicts import MetricTelemetryDict


class BaseTelemetryMetric(
    Generic[MetricValueTypeVarT], BaseMetric[MetricValueTypeVarT], ABC
):
    """A base class for telemetry metrics that aggregate data per GPU.
    These metrics collect telemetry data from external sources like NVIDIA DCGM Exporter.

    Examples:
    ```python
    class GpuPowerUsageMetric(BaseTelemetryMetric[float]):
        tag = "gpu_power_usage"
        header = "GPU Power Usage"
        unit = MetricUnitT.WATT
        
        def _extract_value(self, record: TelemetryRecord) -> float | None:
            return record.gpu_power_usage
    ```
    """

    type = MetricType.TELEMETRY

    def process_telemetry_batch(
        self, telemetry_records: list[TelemetryRecord]
    ) -> dict[int, list[MetricValueTypeVarT]]:
        """Process batch of telemetry records, returning values grouped by GPU index."""
        gpu_values = {}
        for record in telemetry_records:
            value = self._extract_value(record)
            if value is not None:
                gpu_values.setdefault(record.gpu_index, []).append(value)
        return gpu_values
    
    @abstractmethod
    def _extract_value(self, record: TelemetryRecord) -> MetricValueTypeVarT | None:
        """Extract metric value from telemetry record.
        
        Args:
            record: The telemetry record to extract the value from
            
        Returns:
            The metric value, or None if the value is not available
        """
        raise NotImplementedError("Subclasses must implement this method")
