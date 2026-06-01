# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import NoReturn

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

    Invariant: externally injected by
    `GPUTelemetryAccumulator.compute_efficiency_metrics` from gpu_power_usage
    scrapes. `_derive_value` is intentionally non-functional;
    `MetricResultsProcessor.update_derived_metrics` is expected to catch
    NoMetricValue and skip the tag during its derivation walk.
    """

    tag = "total_gpu_power"
    header = "Total GPU Power"
    unit = PowerMetricUnit.WATT
    display_order = 900
    flags = MetricFlags.NONE

    def _derive_value(self, metric_results: MetricResultsDict) -> NoReturn:
        raise NoMetricValue(
            "Cannot derive 'total_gpu_power' from MetricResultsDict: this metric "
            "is externally injected by "
            "GPUTelemetryAccumulator.compute_efficiency_metrics. If this exception "
            "surfaces, the derivation walk is missing its NoMetricValue handler "
            "(see MetricResultsProcessor.update_derived_metrics)."
        )


class TotalGpuEnergyMetric(BaseDerivedMetric[float]):
    """Sum of GPU energy consumed across all GPUs during the benchmark phase, in joules.

    Invariant: externally injected by
    `GPUTelemetryAccumulator.compute_efficiency_metrics` from
    energy_consumption counter deltas. `_derive_value` is intentionally
    non-functional; `MetricResultsProcessor.update_derived_metrics` is
    expected to catch NoMetricValue and skip the tag during its
    derivation walk.
    """

    tag = "total_gpu_energy"
    header = "Total GPU Energy"
    unit = EnergyMetricUnit.JOULE
    display_order = 901
    flags = MetricFlags.NONE

    def _derive_value(self, metric_results: MetricResultsDict) -> NoReturn:
        raise NoMetricValue(
            "Cannot derive 'total_gpu_energy' from MetricResultsDict: this metric "
            "is externally injected by "
            "GPUTelemetryAccumulator.compute_efficiency_metrics. If this exception "
            "surfaces, the derivation walk is missing its NoMetricValue handler "
            "(see MetricResultsProcessor.update_derived_metrics)."
        )


class OutputTokensPerJouleMetric(BaseDerivedMetric[float]):
    """Total output tokens divided by total GPU energy consumed, in tokens per joule.

    Invariant: externally injected by
    `GPUTelemetryAccumulator.compute_efficiency_metrics` as
    `total_output_tokens / total_gpu_energy`. `_derive_value` is
    intentionally non-functional;
    `MetricResultsProcessor.update_derived_metrics` is expected to catch
    NoMetricValue and skip the tag during its derivation walk.
    """

    tag = "output_tokens_per_joule"
    header = "Output Tokens per Joule"
    unit = GenericMetricUnit.TOKENS_PER_JOULE
    display_order = 902
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.PRODUCES_TOKENS_ONLY

    def _derive_value(self, metric_results: MetricResultsDict) -> NoReturn:
        raise NoMetricValue(
            "Cannot derive 'output_tokens_per_joule' from MetricResultsDict: this "
            "metric is externally injected by "
            "GPUTelemetryAccumulator.compute_efficiency_metrics. If this exception "
            "surfaces, the derivation walk is missing its NoMetricValue handler "
            "(see MetricResultsProcessor.update_derived_metrics)."
        )


class EnergyPerUserMetric(BaseDerivedMetric[float]):
    """Total GPU energy divided by configured concurrency, in joules per user.

    Invariant: externally injected by
    `GPUTelemetryAccumulator.compute_efficiency_metrics` as
    `total_gpu_energy / profiling_phase.concurrency`. `_derive_value` is
    intentionally non-functional;
    `MetricResultsProcessor.update_derived_metrics` is expected to catch
    NoMetricValue and skip the tag during its derivation walk. Omitted
    when concurrency is unset (e.g. pure request-rate runs) or zero.
    """

    tag = "energy_per_user"
    header = "Energy per User"
    unit = GenericMetricUnit.JOULES_PER_USER
    display_order = 903
    flags = MetricFlags.NONE

    def _derive_value(self, metric_results: MetricResultsDict) -> NoReturn:
        raise NoMetricValue(
            "Cannot derive 'energy_per_user' from MetricResultsDict: this metric "
            "is externally injected by "
            "GPUTelemetryAccumulator.compute_efficiency_metrics. If this exception "
            "surfaces, the derivation walk is missing its NoMetricValue handler "
            "(see MetricResultsProcessor.update_derived_metrics)."
        )
