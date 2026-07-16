# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.power_efficiency_metrics import (
    AverageGpuPowerMetric,
    EnergyDelayProductMetric,
    EnergyPerOutputTokenMetric,
    EnergyPerRequestMetric,
    EnergyPerTotalTokenMetric,
    EnergyPerUserMetric,
    OutputTokensPerJouleMetric,
    PerformancePerWattMetric,
    TotalGpuEnergyMetric,
    TotalGpuPowerMetric,
)

_ENERGY_METRIC_CLASSES = [
    AverageGpuPowerMetric,
    EnergyDelayProductMetric,
    EnergyPerOutputTokenMetric,
    EnergyPerRequestMetric,
    EnergyPerTotalTokenMetric,
    EnergyPerUserMetric,
    OutputTokensPerJouleMetric,
    PerformancePerWattMetric,
    TotalGpuEnergyMetric,
    TotalGpuPowerMetric,
]


class TestPowerEfficiencyDeriveValueContract:
    """Pin the `_derive_value` invariant for the analyzer-injected energy metrics.

    The energy-efficiency classes inherit `BaseDerivedMetric` for registry
    integration, but their values are produced by
    `EnergyEfficiencyAnalyzer.analyze`, not by the metrics accumulator's
    derived-metric pass (they need GPU telemetry from a different accumulator).
    Calling `_derive_value` directly must raise `NoMetricValue` with a message
    that names the tag, the operation, and the injection site — so a future
    contributor copy-pasting this as the "derived metric pattern" sees the
    contract spelled out rather than a silent miscalculation.
    """

    @pytest.mark.parametrize(
        "metric_class",
        _ENERGY_METRIC_CLASSES,
        ids=lambda c: c.tag,
    )
    def test_derive_value_raises_no_metric_value(self, metric_class) -> None:
        with pytest.raises(NoMetricValue) as exc_info:
            metric_class()._derive_value(MetricResultsDict())

        msg = str(exc_info.value)
        assert metric_class.tag in msg, (
            f"error message must name the tag {metric_class.tag!r}"
        )
        assert "MetricResultsDict" in msg, (
            "error message must name the operation source so agents understand "
            "which derivation path is being rejected"
        )
        assert "EnergyEfficiencyAnalyzer" in msg, (
            "error message must point to the actual injection site so a future "
            "contributor doesn't copy this as the derived-metric pattern"
        )
