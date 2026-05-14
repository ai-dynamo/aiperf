# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from typing import Any, ClassVar

from aiperf.common.config import UserConfig
from aiperf.common.exceptions import NoMetricValue, PostProcessorDisabled
from aiperf.common.models import MetricRecordMetadata, MetricResult
from aiperf.common.types import MetricTagT
from aiperf.metrics.base_metric import BaseMetric
from aiperf.metrics.display_units import to_display_unit
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.metric_results_processor import MetricResultsProcessor


class ArchetypeMetricResultsProcessor(MetricResultsProcessor):
    """Groups metrics by archetype name for media mix benchmarks.

    Same architecture as TimesliceMetricResultsProcessor, but the grouping
    key is MetricRecordMetadata.archetype_name (set during dataset
    generation by SyntheticDatasetComposer when media_mix is configured)
    instead of a time-window index.

    The InputConfig validator guarantees every archetype has a unique
    non-None name at config load time, so this processor can use the
    name directly as a dict key without defensive fallback logic.
    """

    result_kind: ClassVar[str] = "archetype"

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        super().__init__(user_config=user_config, **kwargs)

        if not user_config.input.media_mix:
            raise PostProcessorDisabled(
                "ArchetypeMetricResultsProcessor requires media_mix to be configured"
            )

        self._archetype_instances_maps: dict[str, dict[MetricTagT, BaseMetric]] = (
            defaultdict(
                lambda: {
                    tag: MetricRegistry.get_class(tag)()
                    for tag in MetricRegistry.all_tags()
                }
            )
        )

        self._archetype_results: dict[str, MetricResultsDict] = defaultdict(
            MetricResultsDict
        )

    async def get_instances_map(
        self, record_metadata: MetricRecordMetadata | None = None
    ) -> dict[MetricTagT, BaseMetric]:
        if record_metadata is None or record_metadata.archetype_name is None:
            raise ValueError(
                "ArchetypeMetricResultsProcessor::get_instances_map must be passed a "
                "record_metadata whose archetype_name is set. The InputConfig validator "
                "should have ensured every archetype has a non-None name."
            )
        return self._archetype_instances_maps[record_metadata.archetype_name]

    async def get_results(
        self, record_metadata: MetricRecordMetadata | None = None
    ) -> MetricResultsDict:
        if record_metadata is None or record_metadata.archetype_name is None:
            raise ValueError(
                "ArchetypeMetricResultsProcessor::get_results must be passed a "
                "record_metadata whose archetype_name is set."
            )
        return self._archetype_results[record_metadata.archetype_name]

    async def update_derived_metrics(self) -> None:
        for archetype_results in self._archetype_results.values():
            for tag, derive_func in self.derive_funcs.items():
                try:
                    archetype_results[tag] = derive_func(archetype_results)
                except NoMetricValue as e:
                    self.debug(f"No metric value for derived metric '{tag}': {e!r}")
                except Exception as e:
                    self.warning(f"Error deriving metric '{tag}': {e!r}")

    async def summarize(self) -> dict[str, list[MetricResult]]:
        """Per-archetype summarize.

        Computes derived metrics within each archetype's bucket, filters
        INTERNAL/EXPERIMENTAL metrics, and converts to display units.
        Returns dict mapping archetype_name -> list[MetricResult].
        """
        self.info("Summarizing per-archetype metric results...")
        await self.update_derived_metrics()

        out: dict[str, list[MetricResult]] = {}
        for archetype_name, results in self._archetype_results.items():
            out[archetype_name] = [
                to_display_unit(self._create_metric_result(tag, values), MetricRegistry)
                for tag, values in results.items()
                if self._should_include_in_summary(tag)
            ]
        self.info(f"Summarized {len(out)} archetype metric result(s)")
        return out
