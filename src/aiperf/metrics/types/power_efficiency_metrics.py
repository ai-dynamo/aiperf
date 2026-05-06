# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import (
    EnergyMetricUnit,
    GenericMetricUnit,
    MetricFlags,
    PowerMetricUnit,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict


class TotalGpuPowerMetric(BaseDerivedMetric[float]):
    """Sum of average GPU power across all GPUs during the benchmark phase, in watts.

    Computed externally by the GPU telemetry accumulator and injected as a pre-computed result.
    """

    tag = "total_gpu_power"
    header = "Total GPU Power"
    unit = PowerMetricUnit.WATT
    display_order = 900
    flags = MetricFlags.NONE

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        raise NoMetricValue(
            "total_gpu_power is computed by the GPU telemetry accumulator"
        )


class TotalGpuEnergyMetric(BaseDerivedMetric[float]):
    """Sum of GPU energy consumed across all GPUs during the benchmark phase, in joules.

    Computed externally by the GPU telemetry accumulator and injected as a pre-computed result.
    """

    tag = "total_gpu_energy"
    header = "Total GPU Energy"
    unit = EnergyMetricUnit.JOULE
    display_order = 901
    flags = MetricFlags.NONE

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        raise NoMetricValue(
            "total_gpu_energy is computed by the GPU telemetry accumulator"
        )


class OutputTokensPerJouleMetric(BaseDerivedMetric[float]):
    """Total output tokens divided by total GPU energy consumed, in tokens per joule.

    Computed externally by the GPU telemetry accumulator and injected as a pre-computed result.
    """

    tag = "output_tokens_per_joule"
    header = "Output Tokens per Joule"
    unit = GenericMetricUnit.TOKENS_PER_JOULE
    display_order = 902
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.PRODUCES_TOKENS_ONLY

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        raise NoMetricValue(
            "output_tokens_per_joule is computed by the GPU telemetry accumulator"
        )
