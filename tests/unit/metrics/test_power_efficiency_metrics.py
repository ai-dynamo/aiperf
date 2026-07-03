# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.power_efficiency_metrics import TotalGpuEnergyMetric


class TestTotalGpuEnergyDeriveValueContract:
    """Pin the `_derive_value` invariant for the externally-injected
    `total_gpu_energy` metric.

    `TotalGpuEnergyMetric` inherits `BaseDerivedMetric` for registry integration
    (tag / header / unit), but its value is produced controller-side by
    `analysis.energy_analyzer`, not by the derivation walk in
    `MetricResultsProcessor.update_derived_metrics`. Calling `_derive_value`
    directly must raise `NoMetricValue` with a message that names the tag, the
    operation, and the injection site — so a future contributor copy-pasting this
    as the "derived metric pattern" sees the contract spelled out rather than a
    silent miscalculation.
    """

    def test_derive_value_raises_no_metric_value(self) -> None:
        with pytest.raises(NoMetricValue) as exc_info:
            TotalGpuEnergyMetric()._derive_value(MetricResultsDict())

        msg = str(exc_info.value)
        assert TotalGpuEnergyMetric.tag in msg, (
            f"error message must name the tag {TotalGpuEnergyMetric.tag!r}"
        )
        assert "MetricResultsDict" in msg, (
            "error message must name the operation source so agents understand "
            "which derivation path is being rejected"
        )
        assert "energy_analyzer" in msg, (
            "error message must point to the actual injection site so a future "
            "contributor doesn't copy this as the derived-metric pattern"
        )
