# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.metric_enums import EnergyMetricUnit, MetricFlags
from aiperf.common.models import TelemetryRecord
from aiperf.metrics.base_telemetry_metric import BaseTelemetryMetric


class EnergyConsumptionMetric(BaseTelemetryMetric[float]):
    tag = "energy_consumption"
    header = "Energy Consumption"
    unit = EnergyMetricUnit.MILLIJOULE
    display_order = None
    required_metrics = None
    flags = MetricFlags.GPU_TELEMETRY

    def _extract_value(self, record: TelemetryRecord) -> float | None:
        return record.energy_consumption
