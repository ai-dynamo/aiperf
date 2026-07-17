# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from datetime import datetime

import orjson

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.finite import scrub_non_finite
from aiperf.common.models import MetricResult
from aiperf.common.models.export_models import (
    JsonExportData,
    JsonMetricResult,
    RunInfo,
)
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter


class MetricsJsonExporter(MetricsBaseExporter):
    """
    A class to export records to a JSON file.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        summary = exporter_config.cfg.artifacts.summary
        if summary is False or "json" not in summary:
            raise DataExporterDisabled(
                "MetricsJsonExporter disabled: 'json' not in artifacts.summary"
            )
        super().__init__(exporter_config, **kwargs)
        self._file_path = exporter_config.cfg.artifacts.profile_export_json_file
        self.trace_or_debug(
            lambda: f"Initializing MetricsJsonExporter with config: {exporter_config}",
            lambda: (
                f"Initializing MetricsJsonExporter with file path: {self._file_path}"
            ),
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
        prepared_warmup_metrics = self._prepare_metrics_for_json(
            getattr(self._results, "warmup_records", None) or []
        )

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
            benchmark_id=self._run.benchmark_id if self._run is not None else None,
            input_config=self._cfg,
            run_info=RunInfo.from_run(self._run),
            was_cancelled=self._results.was_cancelled,
            error_summary=self._results.error_summary,
            start_time=start_time,
            end_time=end_time,
            telemetry_data=self._telemetry_results,
            warmup_metrics=prepared_warmup_metrics or None,
        )

        # Add all prepared metrics dynamically
        for metric_tag, json_result in prepared_json_metrics.items():
            setattr(export_data, metric_tag, json_result)

        # Fold the runtime context-overflow-rate contribution (InferenceX
        # AgentX RFC §7) into the lock-only ``submission_valid`` surfaced by
        # ``RunInfo.from_run``. ``apply_scenario`` stamps lock violations +
        # ``unsafe_override``; ``compute_submission_outcome`` additionally flips
        # the verdict when ``context_overflow_count / total_responses`` exceeds
        # the configured threshold (or the run was cancelled). Null-safe: only
        # runs out of a scenario carry a non-None ``scenario_name``.
        self._fold_runtime_submission_outcome(export_data, prepared_json_metrics)

        # Splice DAG branch orchestration counters when present. Non-DAG
        # runs leave ``branch_stats`` unset on ProfileResults so the
        # section is omitted entirely (model_dump_json with
        # ``exclude_none=True`` drops it).
        branch_stats = getattr(self._results, "branch_stats", None)
        if branch_stats is not None:
            export_data.branch_stats = branch_stats

        self.trace_or_debug(
            lambda: f"Exporting data to JSON file: {export_data}",
            lambda: f"Exporting data to JSON file: {self._file_path}",
        )
        # Pydantic's model_dump_json silently coerces NaN/inf to JSON null,
        # which collides with explicit-None ("metric was missing") semantics
        # downstream. Round-trip through model_dump + scrub_non_finite +
        # orjson.dumps so non-finite values are rewritten to null only when
        # they were genuinely numerically absent.
        payload = export_data.model_dump(
            mode="json", exclude_unset=True, exclude_none=True
        )
        return orjson.dumps(
            scrub_non_finite(payload), option=orjson.OPT_INDENT_2
        ).decode("utf-8")

    def _fold_runtime_submission_outcome(
        self,
        export_data: JsonExportData,
        prepared_json_metrics: dict[str, JsonMetricResult],
    ) -> None:
        """Fold the runtime overflow-rate contribution into ``run_info``.

        ``RunInfo.from_run`` surfaces the lock-only scenario outcome
        (invariant violations + ``--unsafe-override``). This re-derives the
        final ``submission_valid`` / ``submission_invalid_reasons`` via
        ``compute_submission_outcome``, which additionally flips the verdict
        when the runtime context-overflow rate exceeds
        ``Environment.AGENTX.CONTEXT_OVERFLOW_RATE_LIMIT`` (RFC §7) or the run
        was cancelled. No-op for non-scenario runs (``scenario_name`` None).

        Args:
            export_data: The export envelope whose ``run_info`` is mutated
                in place.
            prepared_json_metrics: Tag -> JsonMetricResult map; aggregate
                counters expose their value via ``.avg``.
        """
        from aiperf.common.scenario import compute_submission_outcome

        run_info = export_data.run_info
        if run_info is None or run_info.scenario_name is None:
            return

        def _metric_avg(tag: str) -> int:
            m = prepared_json_metrics.get(tag)
            if m is None or m.avg is None:
                return 0
            return int(m.avg)

        context_overflow_count = _metric_avg("context_overflow_count")
        total_responses = (
            _metric_avg("request_count")
            + _metric_avg("error_request_count")
            + context_overflow_count
        )

        submission_valid, submission_invalid_reasons = compute_submission_outcome(
            scenario_name=run_info.scenario_name,
            validator_submission_valid=run_info.submission_valid,
            validator_reasons=run_info.submission_invalid_reasons,
            total_responses=total_responses,
            context_overflow_count=context_overflow_count,
            was_cancelled=bool(self._results.was_cancelled),
        )
        run_info.submission_valid = submission_valid
        run_info.submission_invalid_reasons = submission_invalid_reasons or None

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
