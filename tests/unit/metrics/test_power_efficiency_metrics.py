# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.power_efficiency_metrics import (
    EnergyPerUserMetric,
    OutputTokensPerJouleMetric,
    TotalGpuEnergyMetric,
    TotalGpuPowerMetric,
)


class TestPowerEfficiencyDeriveValueContract:
    """Pin the `_derive_value` invariant for externally-injected derived metrics.

    The three power-efficiency classes inherit `BaseDerivedMetric` for registry
    integration but their values are produced by
    `GPUTelemetryAccumulator.compute_efficiency_metrics`, not by the derivation
    walk in `MetricResultsProcessor.update_derived_metrics`. Calling
    `_derive_value` directly must raise `NoMetricValue` with a message that
    names the tag, the operation, and the injection site — so a future
    contributor copy-pasting this as the "derived metric pattern" sees the
    contract spelled out rather than a silent miscalculation.
    """

    @pytest.mark.parametrize(
        "metric_class",
        [
            TotalGpuPowerMetric,
            TotalGpuEnergyMetric,
            OutputTokensPerJouleMetric,
            EnergyPerUserMetric,
        ],
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
        assert "compute_efficiency_metrics" in msg, (
            "error message must point to the actual injection site so a future "
            "contributor doesn't copy this as the derived-metric pattern"
        )
