# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import GenericMetricUnit, MetricConsoleGroup
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.theoretical_prefix_cache_metric import (
    TheoreticalPrefixCacheHitMetric,
)
from aiperf.post_processors.theoretical_prefix_cache import (
    THEORETICAL_PREFIX_CACHE_HIT_TAG,
)


class TestTheoreticalPrefixCacheHitMetricRegistration:
    """The tag must resolve in the MetricRegistry with CACHE display metadata.

    The value is produced by the standalone TheoreticalPrefixCacheAccumulator,
    but display consumers (RealtimeMetricsTable, ConsoleMetricsExporter) resolve
    the tag via strict/grouped registry lookups: an unregistered tag used to
    kill the realtime dashboard table and be silently omitted from the console
    table on every graph-IR run.
    """

    def test_tag_resolves_in_registry(self) -> None:
        metric_class = MetricRegistry.get_class("theoretical_prefix_cache_hit")
        assert metric_class is TheoreticalPrefixCacheHitMetric
        assert metric_class.tag == "theoretical_prefix_cache_hit"

    def test_accumulator_tag_constant_matches_registered_class(self) -> None:
        assert TheoreticalPrefixCacheHitMetric.tag == THEORETICAL_PREFIX_CACHE_HIT_TAG

    def test_display_metadata_matches_accumulator_output(self) -> None:
        assert TheoreticalPrefixCacheHitMetric.header == "Theoretical Prefix Cache Hit"
        assert TheoreticalPrefixCacheHitMetric.unit == GenericMetricUnit.PERCENT
        assert TheoreticalPrefixCacheHitMetric.console_group == MetricConsoleGroup.CACHE
        assert TheoreticalPrefixCacheHitMetric.display_order is not None

    def test_derive_value_raises_no_metric_value(self) -> None:
        """Externally-injected contract: the derivation walk must skip this tag."""
        with pytest.raises(NoMetricValue) as exc_info:
            TheoreticalPrefixCacheHitMetric()._derive_value(MetricResultsDict())

        msg = str(exc_info.value)
        assert TheoreticalPrefixCacheHitMetric.tag in msg
        assert "MetricResultsDict" in msg
        assert "TheoreticalPrefixCacheAccumulator.summarize" in msg
