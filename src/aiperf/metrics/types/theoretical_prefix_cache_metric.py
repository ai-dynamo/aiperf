# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import NoReturn

from aiperf.common.enums import (
    GenericMetricUnit,
    MetricConsoleGroup,
    MetricFlags,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict


class TheoreticalPrefixCacheHitMetric(BaseDerivedMetric[float]):
    """Cumulative infinite-cache prefix hit rate over a trace replay, in percent.

    Invariant: externally injected by `TheoreticalPrefixCacheAccumulator` from
    loader-stamped per-node / per-turn prefix-cache block counts. The
    accumulator sets `supports_phase_scoped_export`, so RecordsManager reaches
    it through `export_results(ctx)` (phase-scoped); the unscoped `summarize`
    path pools every phase. `_derive_value` is intentionally
    non-functional; `MetricsAccumulator._resolve_derived_metrics` is expected
    to catch NoMetricValue and skip the tag during its derivation walk. The
    class exists so display consumers (realtime dashboard table, console
    exporter) can resolve the tag's display metadata from the MetricRegistry.
    """

    tag = "theoretical_prefix_cache_hit"
    header = "Theoretical Prefix Cache Hit"
    unit = GenericMetricUnit.PERCENT
    display_order = 1018
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.CACHE

    def _derive_value(self, metric_results: MetricResultsDict) -> NoReturn:
        raise NoMetricValue(
            "Cannot derive 'theoretical_prefix_cache_hit' from MetricResultsDict: "
            "this metric is externally injected by "
            "TheoreticalPrefixCacheAccumulator.summarize. If this exception "
            "surfaces, the derivation walk is missing its NoMetricValue handler "
            "(see MetricsAccumulator._resolve_derived_metrics)."
        )
