# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rich.console import Console, Group, RenderableType

from aiperf.common.exceptions import ConsoleExporterDisabled
from aiperf.exporters.console_metrics_exporter import ConsoleMetricsExporter
from aiperf.exporters.exporter_config import ExporterConfig


class ConsoleArchetypeMetricsExporter(ConsoleMetricsExporter):
    """Renders one Rich table per archetype for media mix benchmarks.

    Sits alongside (not replacing) the existing ConsoleMetricsExporter,
    which still renders the across-archetype aggregate table. Each
    archetype gets the same column set and metric ordering as the
    aggregate so users learn one table layout and see it N+1 times.

    Self-disables when ProfileResults.archetype_metric_results is missing
    so users running non-media-mix benchmarks see no behavioral change.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        self._user_config = exporter_config.user_config

    def _check_enabled(self, exporter_config: ExporterConfig) -> None:
        archetype_results = getattr(
            exporter_config.results, "archetype_metric_results", None
        )
        if not archetype_results:
            raise ConsoleExporterDisabled(
                "ConsoleArchetypeMetricsExporter disabled: "
                "no archetype metric results found"
            )

    async def export(self, console: Console) -> None:
        archetype_results = self._results.archetype_metric_results
        if not archetype_results:
            self.debug("No archetype results to export")
            return

        weights = self._archetype_weights_by_name()
        total_weight = sum(weights.values()) or None
        tables = []
        for archetype_name in sorted(archetype_results.keys()):
            records = archetype_results[archetype_name]
            visible = [r for r in records if self._should_show(r)]
            if not visible:
                continue
            title = self._archetype_title(
                archetype_name, weights.get(archetype_name), total_weight
            )
            tables.append(self._build_table(title, visible))

        if not tables:
            return
        renderable: RenderableType = tables[0] if len(tables) == 1 else Group(*tables)
        self._print_renderable(console, renderable)

    def _archetype_title(
        self,
        archetype_name: str,
        weight: float | None,
        total_weight: float | None,
    ) -> str:
        """Build a per-archetype table title that mirrors the aggregate title shape."""
        from aiperf.plugin import plugins

        endpoint_metadata = plugins.get_endpoint_metadata(self._endpoint_type)
        prefix = f"NVIDIA AIPerf | {endpoint_metadata.metrics_title}: {archetype_name}"
        if weight is None or not total_weight:
            return prefix
        share = (weight / total_weight) * 100
        return f"{prefix} ({share:.0f}% of traffic)"

    def _archetype_weights_by_name(self) -> dict[str, float]:
        media_mix = self._user_config.input.media_mix or []
        return {a.name: a.weight for a in media_mix if a.name is not None}
