# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from datetime import datetime

import orjson

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.finite import is_finite_value, scrub_non_finite
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

        from aiperf.dataset.provenance import public_dataset_provenance

        run_metadata: dict[str, object] = {}
        dataset = public_dataset_provenance(self._cfg)
        if dataset is not None:
            run_metadata["dataset"] = dataset
        metric_duration_coverage = getattr(
            self._results, "metric_duration_coverage", []
        )
        if metric_duration_coverage:
            run_metadata["metric_duration_coverage"] = [
                coverage.model_dump(mode="json")
                for coverage in metric_duration_coverage
            ]

        # ProfileResults.context_overflow_count is the AGENTIC_REPLAY skip-path
        # side channel only (not in error_request_count / ContextOverflowCountMetric).
        # Metric-path overflows are already in error_request_count (ERROR_ONLY).
        skipped_context_overflow_count = int(
            getattr(self._results, "context_overflow_count", 0) or 0
        )
        if skipped_context_overflow_count:
            existing_context_overflow = prepared_json_metrics.get(
                "context_overflow_count"
            )
            if existing_context_overflow is None:
                prepared_json_metrics["context_overflow_count"] = JsonMetricResult(
                    unit="requests",
                    avg=float(skipped_context_overflow_count),
                )
            else:
                prepared_json_metrics["context_overflow_count"] = (
                    existing_context_overflow.model_copy(
                        update={
                            "avg": float(
                                (existing_context_overflow.avg or 0)
                                + skipped_context_overflow_count
                            )
                        }
                    )
                )
            # Persist the skip-only count so aggregate re-summation can add it to
            # the denominator without double-counting metric-path overflows.
            prepared_json_metrics["skipped_context_overflow_count"] = JsonMetricResult(
                unit="requests",
                avg=float(skipped_context_overflow_count),
            )

        # Add all prepared metrics dynamically
        for metric_tag, json_result in prepared_json_metrics.items():
            setattr(export_data, metric_tag, json_result)

        # Attach optional run-level aggregates (branch_stats, pooled spec-decode
        # histogram) that live on ProfileResults outside the metric dict.
        self._splice_run_level_aggregates(export_data)

        # Stamp scenario submission metadata for single-run exports. Mirrors the
        # carrier-key contract used by AggregateConfidenceJsonExporter: the
        # validator outcome lives on ``run.resolved.scenario_outcome`` (set by
        # ScenarioResolver) and runtime totals are summed from the prepared
        # metric results. No-ops (metadata omitted) when no --scenario was set
        # or the outcome is absent.
        scenario_name = getattr(self._cfg, "scenario", None)
        resolved = self._run.resolved if self._run is not None else None
        outcome = getattr(resolved, "scenario_outcome", None)
        if scenario_name is not None and outcome is not None:
            from aiperf.exporters.aggregate.aggregate_base_exporter import (
                _build_run_metadata_dict,
                compute_submission_outcome,
            )

            validator_submission_valid = outcome.submission_valid
            validator_reasons = list(outcome.submission_invalid_reasons)

            def _metric_avg(tag: str) -> int:
                m = prepared_json_metrics.get(tag)
                if m is None or not is_finite_value(m.avg):
                    return 0
                return int(m.avg)

            # Numerator: all overflows (metric-path + skip-path, after merge above).
            # Denominator: successes + errors + skip-path-only overflows.
            # Metric-path overflows are already inside error_request_count
            # (ContextOverflowCountMetric is ERROR_ONLY); adding the merged
            # context_overflow_count again would double-count them.
            context_overflow_count = _metric_avg("context_overflow_count")
            total_responses = (
                _metric_avg("request_count")
                + _metric_avg("error_request_count")
                + skipped_context_overflow_count
            )

            submission_valid, submission_invalid_reasons = compute_submission_outcome(
                scenario_name=scenario_name,
                validator_submission_valid=validator_submission_valid,
                validator_reasons=validator_reasons,
                total_responses=total_responses,
                context_overflow_count=context_overflow_count,
                was_cancelled=bool(self._results.was_cancelled),
                runtime_invalid_reasons=list(
                    getattr(self._results, "runtime_submission_invalid_reasons", [])
                ),
            )
            run_metadata.update(
                _build_run_metadata_dict(
                    scenario_name=scenario_name,
                    submission_valid=submission_valid,
                    submission_invalid_reasons=submission_invalid_reasons,
                )
            )

        if run_metadata:
            export_data.metadata = run_metadata

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

    def _splice_run_level_aggregates(self, export_data: JsonExportData) -> None:
        """Attach optional run-level aggregates that live on ``ProfileResults``
        outside the metric dict. Each is omitted from the export (``exclude_none``)
        when absent: ``branch_stats`` on non-DAG runs, the pooled spec-decode
        acceptance histogram when spec decode is off.
        """
        branch_stats = getattr(self._results, "branch_stats", None)
        if branch_stats is not None:
            export_data.branch_stats = branch_stats

        spec_decode_histogram = getattr(
            self._results, "pooled_spec_decode_acceptance_histogram", None
        )
        if spec_decode_histogram is not None:
            export_data.pooled_spec_decode_acceptance_histogram = spec_decode_histogram

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
