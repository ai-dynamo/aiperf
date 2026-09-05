# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vendor-split GPU energy-efficiency metric registration classes.

Each metric is a ``BaseDerivedMetric`` placeholder: it registers the tag,
header, unit, display-order, and flags so the summary and exporters recognize
it, but its value is injected post-aggregation by
:class:`aiperf.metrics.energy_efficiency_analyzer.EnergyEfficiencyAnalyzer`,
which joins GPU-telemetry energy/power (per vendor) to inference token/
throughput/latency totals via the ``SummaryContext``. ``_derive_value``
therefore always defers with ``NoMetricValue`` so the derivation walk skips
these tags; the analyzer supplies the real values.

Each metric exists in an NVIDIA variant (``nvidia_`` prefix,
``GPU_POWER_EFFICIENCY_NVIDIA`` console group) and an AMD variant (``amd_``
prefix, ``GPU_POWER_EFFICIENCY_AMD`` console group). The analyzer emits only
the variants whose vendor actually reported data during the run.

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


# --- NVIDIA ------------------------------------------------------------------


class NvidiaEnergyDelayProductMetric(_InjectedEnergyMetric):
    """NVIDIA energy-delay product: ``energy_per_request_J * mean_request_latency_s`` (J*s)."""

    __is_abstract__ = False
    tag = "nvidia_energy_delay_product"
    header = "Energy Delay Product"
    unit = GenericMetricUnit.JOULE_SECONDS
    display_order = 700
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA


class NvidiaPerformancePerWattMetric(_InjectedEnergyMetric):
    """NVIDIA request throughput per watt of GPU power draw (req/s/W)."""

    __is_abstract__ = False
    tag = "nvidia_performance_per_watt"
    header = "Performance per Watt"
    unit = GenericMetricUnit.REQUESTS_PER_SECOND_PER_WATT
    display_order = 710
    flags = MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA


class NvidiaOutputTokensPerSecondPerWattMetric(_InjectedEnergyMetric):
    """NVIDIA output-token throughput per watt of GPU power draw (tokens/sec/W)."""

    __is_abstract__ = False
    tag = "nvidia_output_tps_per_watt"
    header = "Output Tokens per Second per Watt"
    unit = GenericMetricUnit.TOKENS_PER_SECOND_PER_WATT
    display_order = 712
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.PRODUCES_TOKENS_ONLY
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA


class NvidiaGoodputPerWattMetric(_InjectedEnergyMetric):
    """NVIDIA goodput (SLO-passing requests/sec) per watt of GPU power draw (good-req/s/W).

    Only meaningful when goodput SLOs are configured; omitted otherwise.
    """

    __is_abstract__ = False
    tag = "nvidia_goodput_per_watt"
    header = "Goodput per Watt"
    unit = GenericMetricUnit.GOODPUT_PER_WATT
    display_order = 714
    flags = MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA


class NvidiaAverageGpuPowerMetric(_InjectedEnergyMetric):
    """Time-averaged total NVIDIA GPU power over the profiling window (energy / duration) (W)."""

    __is_abstract__ = False
    tag = "nvidia_average_gpu_power"
    header = "Average GPU Power"
    unit = PowerMetricUnit.WATT
    display_order = 720
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA


class NvidiaTotalGpuEnergyMetric(_InjectedEnergyMetric):
    """Sum of NVIDIA GPU energy consumed across all NVIDIA GPUs during the phase (J)."""

    __is_abstract__ = False
    tag = "nvidia_total_gpu_energy"
    header = "Total GPU Energy"
    unit = EnergyMetricUnit.JOULE
    display_order = 730
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA


class NvidiaTotalGpuPowerMetric(_InjectedEnergyMetric):
    """Sum of average NVIDIA GPU power across all NVIDIA GPUs during the phase (W)."""

    __is_abstract__ = False
    tag = "nvidia_total_gpu_power"
    header = "Total GPU Power"
    unit = PowerMetricUnit.WATT
    display_order = 740
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA


class NvidiaEnergyPerTotalTokenMetric(_InjectedEnergyMetric):
    """NVIDIA GPU energy per total (input + output) token (mJ/token)."""

    __is_abstract__ = False
    tag = "nvidia_energy_per_total_token"
    header = "Energy per Total Token"
    unit = GenericMetricUnit.MILLIJOULES_PER_TOKEN
    display_order = 745
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA


class NvidiaEnergyPerOutputTokenMetric(_InjectedEnergyMetric):
    """NVIDIA GPU energy per output token — the primary efficiency metric (mJ/token)."""

    __is_abstract__ = False
    tag = "nvidia_energy_per_output_token"
    header = "Energy per Output Token"
    unit = GenericMetricUnit.MILLIJOULES_PER_TOKEN
    display_order = 750
    flags = MetricFlags.PRODUCES_TOKENS_ONLY
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA


class NvidiaEnergyPerRequestMetric(_InjectedEnergyMetric):
    """NVIDIA GPU energy per request (J/request)."""

    __is_abstract__ = False
    tag = "nvidia_energy_per_request"
    header = "Energy per Request"
    unit = GenericMetricUnit.JOULES_PER_REQUEST
    display_order = 755
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA


class NvidiaOutputTokensPerJouleMetric(_InjectedEnergyMetric):
    """Total output tokens per joule of NVIDIA GPU energy (tokens/J)."""

    __is_abstract__ = False
    tag = "nvidia_output_tokens_per_joule"
    header = "Output Tokens per Joule"
    unit = GenericMetricUnit.TOKENS_PER_JOULE
    display_order = 760
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.PRODUCES_TOKENS_ONLY
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA


class NvidiaEnergyPerUserMetric(_InjectedEnergyMetric):
    """Total NVIDIA GPU energy divided by configured concurrency (J/user).

    Omitted when concurrency is unset (e.g. pure request-rate runs) or zero.
    """

    __is_abstract__ = False
    tag = "nvidia_energy_per_user"
    header = "Energy per User"
    unit = GenericMetricUnit.JOULES_PER_USER
    display_order = 765
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA


# --- AMD ---------------------------------------------------------------------


class AmdEnergyDelayProductMetric(_InjectedEnergyMetric):
    """AMD energy-delay product: ``energy_per_request_J * mean_request_latency_s`` (J*s)."""

    __is_abstract__ = False
    tag = "amd_energy_delay_product"
    header = "Energy Delay Product"
    unit = GenericMetricUnit.JOULE_SECONDS
    display_order = 800
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD


class AmdPerformancePerWattMetric(_InjectedEnergyMetric):
    """AMD request throughput per watt of GPU power draw (req/s/W)."""

    __is_abstract__ = False
    tag = "amd_performance_per_watt"
    header = "Performance per Watt"
    unit = GenericMetricUnit.REQUESTS_PER_SECOND_PER_WATT
    display_order = 810
    flags = MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD


class AmdOutputTokensPerSecondPerWattMetric(_InjectedEnergyMetric):
    """AMD output-token throughput per watt of GPU power draw (tokens/sec/W)."""

    __is_abstract__ = False
    tag = "amd_output_tps_per_watt"
    header = "Output Tokens per Second per Watt"
    unit = GenericMetricUnit.TOKENS_PER_SECOND_PER_WATT
    display_order = 812
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.PRODUCES_TOKENS_ONLY
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD


class AmdGoodputPerWattMetric(_InjectedEnergyMetric):
    """AMD goodput (SLO-passing requests/sec) per watt of GPU power draw (good-req/s/W).

    Only meaningful when goodput SLOs are configured; omitted otherwise.
    """

    __is_abstract__ = False
    tag = "amd_goodput_per_watt"
    header = "Goodput per Watt"
    unit = GenericMetricUnit.GOODPUT_PER_WATT
    display_order = 814
    flags = MetricFlags.LARGER_IS_BETTER
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD


class AmdAverageGpuPowerMetric(_InjectedEnergyMetric):
    """Time-averaged total AMD GPU power over the profiling window (energy / duration) (W)."""

    __is_abstract__ = False
    tag = "amd_average_gpu_power"
    header = "Average GPU Power"
    unit = PowerMetricUnit.WATT
    display_order = 820
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD


class AmdTotalGpuEnergyMetric(_InjectedEnergyMetric):
    """Sum of AMD GPU energy consumed across all AMD GPUs during the phase (J)."""

    __is_abstract__ = False
    tag = "amd_total_gpu_energy"
    header = "Total GPU Energy"
    unit = EnergyMetricUnit.JOULE
    display_order = 830
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD


class AmdTotalGpuPowerMetric(_InjectedEnergyMetric):
    """Sum of average AMD GPU power across all AMD GPUs during the phase (W)."""

    __is_abstract__ = False
    tag = "amd_total_gpu_power"
    header = "Total GPU Power"
    unit = PowerMetricUnit.WATT
    display_order = 840
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD


class AmdEnergyPerTotalTokenMetric(_InjectedEnergyMetric):
    """AMD GPU energy per total (input + output) token (mJ/token)."""

    __is_abstract__ = False
    tag = "amd_energy_per_total_token"
    header = "Energy per Total Token"
    unit = GenericMetricUnit.MILLIJOULES_PER_TOKEN
    display_order = 845
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD


class AmdEnergyPerOutputTokenMetric(_InjectedEnergyMetric):
    """AMD GPU energy per output token — the primary efficiency metric (mJ/token)."""

    __is_abstract__ = False
    tag = "amd_energy_per_output_token"
    header = "Energy per Output Token"
    unit = GenericMetricUnit.MILLIJOULES_PER_TOKEN
    display_order = 850
    flags = MetricFlags.PRODUCES_TOKENS_ONLY
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD


class AmdEnergyPerRequestMetric(_InjectedEnergyMetric):
    """AMD GPU energy per request (J/request)."""

    __is_abstract__ = False
    tag = "amd_energy_per_request"
    header = "Energy per Request"
    unit = GenericMetricUnit.JOULES_PER_REQUEST
    display_order = 855
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD


class AmdOutputTokensPerJouleMetric(_InjectedEnergyMetric):
    """Total output tokens per joule of AMD GPU energy (tokens/J)."""

    __is_abstract__ = False
    tag = "amd_output_tokens_per_joule"
    header = "Output Tokens per Joule"
    unit = GenericMetricUnit.TOKENS_PER_JOULE
    display_order = 860
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.PRODUCES_TOKENS_ONLY
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD


class AmdEnergyPerUserMetric(_InjectedEnergyMetric):
    """Total AMD GPU energy divided by configured concurrency (J/user).

    Omitted when concurrency is unset (e.g. pure request-rate runs) or zero.
    """

    __is_abstract__ = False
    tag = "amd_energy_per_user"
    header = "Energy per User"
    unit = GenericMetricUnit.JOULES_PER_USER
    display_order = 865
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD
