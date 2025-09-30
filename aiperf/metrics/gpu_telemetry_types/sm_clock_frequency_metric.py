# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.metric_enums import FrequencyMetricUnit, MetricFlags
from aiperf.common.models import TelemetryRecord
from aiperf.metrics.base_telemetry_metric import BaseTelemetryMetric


class SmClockFrequencyMetric(BaseTelemetryMetric[float]):
    """SM clock frequency metric from GPU telemetry.

    Tracks the streaming multiprocessor (SM) clock frequency in MHz.
    """

    tag = "sm_clock_frequency"
    header = "SM Clock Frequency"
    unit = FrequencyMetricUnit.MEGAHERTZ
    display_order = None
    required_metrics = None
    flags = MetricFlags.GPU_TELEMETRY

    def _extract_value(self, record: TelemetryRecord) -> float | None:
        """
        Return the SM (streaming multiprocessor) clock frequency from a telemetry record in megahertz.
        
        Parameters:
            record (TelemetryRecord): Telemetry record containing GPU telemetry fields.
        
        Returns:
            float | None: SM clock frequency in MHz if present, otherwise None.
        """
        return record.sm_clock_frequency
