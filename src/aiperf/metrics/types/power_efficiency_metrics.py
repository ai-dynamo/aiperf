# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import NoReturn

from aiperf.common.enums import (
    EnergyMetricUnit,
    MetricFlags,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict


class TotalGpuEnergyMetric(BaseDerivedMetric[float]):
    """Sum of GPU energy consumed across all GPUs during the benchmark phase, in joules.

    Invariant: not derivable from ``MetricResultsDict``. The value is produced
    controller-side by
    ``analysis.energy_analyzer.compute_energy_efficiency_from_summaries``, which
    fans in the GPU telemetry summary and the inference profile results across
    process boundaries and attaches ``total_gpu_energy`` to the energy-efficiency
    export (not to ``ProfileResults.records``). This class only registers the
    ``total_gpu_energy`` tag / header / unit in the ``MetricRegistry``;
    ``_derive_value`` is intentionally non-functional, and
    ``MetricResultsProcessor.update_derived_metrics`` is expected to catch
    ``NoMetricValue`` and skip the tag during its derivation walk.
    """

    tag = "total_gpu_energy"
    header = "Total GPU Energy"
    unit = EnergyMetricUnit.JOULE
    display_order = 901
    flags = MetricFlags.NONE

    def _derive_value(self, metric_results: MetricResultsDict) -> NoReturn:
        raise NoMetricValue(
            "Cannot derive 'total_gpu_energy' from MetricResultsDict: this metric "
            "is computed controller-side by the energy analyzer "
            "(analysis.energy_analyzer.compute_energy_efficiency_from_summaries). "
            "If this exception surfaces, the derivation walk is missing its "
            "NoMetricValue handler (see MetricResultsProcessor.update_derived_metrics)."
        )
