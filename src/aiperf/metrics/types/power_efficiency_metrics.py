# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registration classes for the GPU energy-efficiency metric family.

These are ``BaseDerivedMetric`` placeholders: they register each tag's
header/unit/display-order/flags so the summary and exporters recognize them, but
their values are injected post-aggregation by
:class:`aiperf.metrics.energy_efficiency_analyzer.EnergyEfficiencyAnalyzer`,
which joins GPU-telemetry energy/power to inference token totals across
accumulators. ``_derive_value`` therefore always defers with ``NoMetricValue`` so
``MetricsAccumulator._resolve_derived_metrics`` skips them during its derivation
walk; the analyzer supplies the real values.

See design doc ``0005-energy-efficiency-metrics.md``.
"""

from typing import NoReturn

from aiperf.common.enums import (
    EnergyMetricUnit,
    GenericMetricUnit,
    MetricConsoleGroup,
    MetricFlags,
    PowerMetricUnit,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict


class _InjectedEnergyMetric(BaseDerivedMetric[float]):
    """Shared deferred-derivation base for the analyzer-injected energy metrics.

    Not registered itself (abstract). The energy-efficiency metrics cannot be
    derived from a single ``MetricResultsDict`` because they need GPU telemetry
    that lives in a different accumulator; ``EnergyEfficiencyAnalyzer`` computes
    them at summarize time via the SummaryContext and injects the results.
    """

    __is_abstract__ = True

    def _derive_value(self, metric_results: MetricResultsDict) -> NoReturn:
        raise NoMetricValue(
            f"'{self.tag}' is injected post-aggregation by EnergyEfficiencyAnalyzer "
            "from GPU telemetry + inference token totals; it cannot be derived from "
            "a single MetricResultsDict. If this surfaces, the derivation walk is "
            "missing its NoMetricValue handler (MetricsAccumulator."
            "_resolve_derived_metrics)."
        )


class EnergyDelayProductMetric(_InjectedEnergyMetric):
    """Energy-delay product: ``energy_per_request_J * mean_request_latency_s`` (J*s)."""

    __is_abstract__ = False
    tag = "energy_delay_product"
    header = "Energy Delay Product"
    unit = GenericMetricUnit.JOULE_SECONDS
    display_order = 700
    flags = MetricFlags.NONE


class PerformancePerWattMetric(_InjectedEnergyMetric):
    """Request throughput per watt of GPU power draw (req/s/W)."""

    __is_abstract__ = False
    tag = "performance_per_watt"
    header = "Performance per Watt"
    unit = GenericMetricUnit.REQUESTS_PER_SECOND_PER_WATT
    display_order = 710
    flags = MetricFlags.LARGER_IS_BETTER


class OutputTokensPerSecondPerWattMetric(_InjectedEnergyMetric):
    """Output-token throughput per watt of GPU power draw (tokens/sec/W)."""

    __is_abstract__ = False
    tag = "output_tps_per_watt"
    header = "Output Tokens per Second per Watt"
    unit = GenericMetricUnit.TOKENS_PER_SECOND_PER_WATT
    display_order = 712
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.PRODUCES_TOKENS_ONLY


class GoodputPerWattMetric(_InjectedEnergyMetric):
    """Goodput (SLO-passing requests/sec) per watt of GPU power draw (good-req/s/W).

    Only meaningful when goodput SLOs are configured; omitted otherwise.
    """

    __is_abstract__ = False
    tag = "goodput_per_watt"
    header = "Goodput per Watt"
    unit = GenericMetricUnit.GOODPUT_PER_WATT
    display_order = 714
    flags = MetricFlags.LARGER_IS_BETTER


class AverageGpuPowerMetric(_InjectedEnergyMetric):
    """Time-averaged total GPU power over the profiling window (energy / duration) (W)."""

    __is_abstract__ = False
    tag = "average_gpu_power"
    header = "Average GPU Power"
    unit = PowerMetricUnit.WATT
    display_order = 720
    flags = MetricFlags.NONE


class TotalGpuEnergyMetric(_InjectedEnergyMetric):
    """Sum of GPU energy consumed across all GPUs during the phase (J)."""

    __is_abstract__ = False
    tag = "total_gpu_energy"
    header = "Total GPU Energy"
    unit = EnergyMetricUnit.JOULE
    display_order = 730
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY


class TotalGpuPowerMetric(_InjectedEnergyMetric):
    """Sum of average GPU power across all GPUs during the phase (W)."""

    __is_abstract__ = False
    tag = "total_gpu_power"
    header = "Total GPU Power"
    unit = PowerMetricUnit.WATT
    display_order = 740
    flags = MetricFlags.NONE


class EnergyPerTotalTokenMetric(_InjectedEnergyMetric):
    """GPU energy per total (input + output) token (mJ/token)."""

    __is_abstract__ = False
    tag = "energy_per_total_token"
    header = "Energy per Total Token"
    unit = GenericMetricUnit.MILLIJOULES_PER_TOKEN
    display_order = 745
    flags = MetricFlags.NONE


class EnergyPerOutputTokenMetric(_InjectedEnergyMetric):
    """GPU energy per output token — the primary efficiency metric (mJ/token)."""

    __is_abstract__ = False
    tag = "energy_per_output_token"
    header = "Energy per Output Token"
    unit = GenericMetricUnit.MILLIJOULES_PER_TOKEN
    display_order = 750
    flags = MetricFlags.PRODUCES_TOKENS_ONLY


class EnergyPerRequestMetric(_InjectedEnergyMetric):
    """GPU energy per request (J/request)."""

    __is_abstract__ = False
    tag = "energy_per_request"
    header = "Energy per Request"
    unit = GenericMetricUnit.JOULES_PER_REQUEST
    display_order = 755
    flags = MetricFlags.NONE


class OutputTokensPerJouleMetric(_InjectedEnergyMetric):
    """Total output tokens per joule of GPU energy (tokens/J)."""

    __is_abstract__ = False
    tag = "output_tokens_per_joule"
    header = "Output Tokens per Joule"
    unit = GenericMetricUnit.TOKENS_PER_JOULE
    display_order = 760
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.PRODUCES_TOKENS_ONLY
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY


class EnergyPerUserMetric(_InjectedEnergyMetric):
    """Total GPU energy divided by configured concurrency (J/user).

    Omitted when concurrency is unset (e.g. pure request-rate runs) or zero.
    """

    __is_abstract__ = False
    tag = "energy_per_user"
    header = "Energy per User"
    unit = GenericMetricUnit.JOULES_PER_USER
    display_order = 765
    flags = MetricFlags.NONE
