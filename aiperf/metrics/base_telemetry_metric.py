# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Generic

from aiperf.common.enums import MetricType, MetricValueTypeVarT
from aiperf.common.models import TelemetryRecord
from aiperf.metrics.base_metric import BaseMetric


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
        """
        Group extracted metric values from telemetry records by GPU index.
        
        Processes the provided telemetry records, extracts a metric value from each record via _extract_value, and collects non-missing values into lists keyed by the record's GPU index. Records for which no value can be extracted are ignored.
        
        Parameters:
            telemetry_records (list[TelemetryRecord]): Telemetry records to process.
        
        Returns:
            dict[int, list[MetricValueTypeVarT]]: Mapping from GPU index to a list of extracted metric values.
        """
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
