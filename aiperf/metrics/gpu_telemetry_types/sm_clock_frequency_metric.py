# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.metric_enums import MetricFlags, FrequencyMetricUnit
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
        return record.sm_clock_frequency