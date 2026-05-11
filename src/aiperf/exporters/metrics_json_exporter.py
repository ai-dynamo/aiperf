# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from datetime import datetime

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models import MetricResult
from aiperf.common.models.export_models import (
    ArchetypeData,
    JsonExportData,
    JsonMetricResult,
)
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter


class MetricsJsonExporter(MetricsBaseExporter):
    """
    A class to export records to a JSON file.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        self._file_path = exporter_config.user_config.output.profile_export_json_file
        self.trace_or_debug(
            lambda: f"Initializing MetricsJsonExporter with config: {exporter_config}",
            lambda: f"Initializing MetricsJsonExporter with file path: {self._file_path}",
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="JSON Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        """Generate JSON content string from inference and telemetry data.

        Uses instance data members self._results.records and self._telemetry_results.

        Returns:
            str: Complete JSON content with all sections formatted and ready to write
        """
        # Use helper method to prepare metrics
        prepared_json_metrics = self._prepare_metrics_for_json(self._results.records)

        start_time = (
            datetime.fromtimestamp(self._results.start_ns / NANOS_PER_SECOND)
            if self._results.start_ns
            else None
        )
        end_time = (
            datetime.fromtimestamp(self._results.end_ns / NANOS_PER_SECOND)
            if self._results.end_ns
            else None
        )

        from aiperf import __version__ as aiperf_version

        # Note: server_metrics_data is exported to a separate file via ServerMetricsJsonExporter
        export_data = JsonExportData(
            schema_version=JsonExportData.SCHEMA_VERSION,
            aiperf_version=aiperf_version,
            benchmark_id=self._user_config.benchmark_id,
            input_config=self._user_config,
            was_cancelled=self._results.was_cancelled,
            error_summary=self._results.error_summary,
            start_time=start_time,
            end_time=end_time,
            telemetry_data=self._telemetry_results,
            archetypes=self._build_archetype_blocks(),
        )

        # Add all prepared metrics dynamically
        for metric_tag, json_result in prepared_json_metrics.items():
            setattr(export_data, metric_tag, json_result)

        self.trace_or_debug(
            lambda: f"Exporting data to JSON file: {export_data}",
            lambda: f"Exporting data to JSON file: {self._file_path}",
        )
        return export_data.model_dump_json(
            indent=2, exclude_unset=True, exclude_none=True
        )

    def _build_archetype_blocks(self) -> list[ArchetypeData] | None:
        """Build the per-archetype metric blocks for the JSON export.

        Returns None when no archetype results exist so the field is
        omitted from the output for non-media-mix benchmarks.

        Each block carries the archetype's identity (name + configured
        weight) plus dynamic metric fields populated via setattr, the
        same pattern JsonExportData uses for the top-level aggregate.
        """
        archetype_results = getattr(self._results, "archetype_metric_results", None)
        if not archetype_results:
            return None

        weights = self._archetype_weights_by_name()

        blocks: list[ArchetypeData] = []
        for archetype_name, metric_results in archetype_results.items():
            block = ArchetypeData(
                archetype_name=archetype_name,
                archetype_weight=weights.get(archetype_name),
            )
            for tag, json_result in self._prepare_metrics_for_json(
                metric_results
            ).items():
                setattr(block, tag, json_result)
            blocks.append(block)
        return blocks

    def _archetype_weights_by_name(self) -> dict[str, float]:
        """Return a {name: weight} map from the configured media_mix archetypes.

        Returns an empty map when media_mix is unconfigured (the archetype
        results would also be empty in that case).
        """
        media_mix = self._user_config.input.media_mix or []
        return {a.name: a.weight for a in media_mix if a.name is not None}

    def _prepare_metrics_for_json(
        self, metric_results: Iterable[MetricResult]
    ) -> dict[str, JsonMetricResult]:
        """Prepare and convert metrics to JsonMetricResult objects.

        Applies unit conversion, filtering, and conversion to JSON format.

        Args:
            metric_results: Raw metric results to prepare

        Returns:
            dict mapping metric tags to JsonMetricResult objects ready for export
        """
        prepared = self._prepare_metrics(metric_results)
        return {tag: result.to_json_result() for tag, result in prepared.items()}
