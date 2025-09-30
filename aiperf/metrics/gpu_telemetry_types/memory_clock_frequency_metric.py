# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.metric_enums import FrequencyMetricUnit, MetricFlags
from aiperf.common.models import TelemetryRecord
from aiperf.metrics.base_telemetry_metric import BaseTelemetryMetric


class MemoryClockFrequencyMetric(BaseTelemetryMetric[float]):
    """Memory clock frequency metric from GPU telemetry.

    Tracks the memory clock frequency in MHz.
    """

    tag = "memory_clock_frequency"
    header = "Memory Clock Frequency"
    unit = FrequencyMetricUnit.MEGAHERTZ
    display_order = None
    required_metrics = None
    flags = MetricFlags.GPU_TELEMETRY

    def _extract_value(self, record: TelemetryRecord) -> float | None:
        """
        Extract the memory clock frequency value from a telemetry record.
        
        Parameters:
            record (TelemetryRecord): Telemetry record containing GPU metrics.
        
        Returns:
            float | None: Memory clock frequency in megahertz, or `None` if the value is unavailable.
        """
        return record.memory_clock_frequency
