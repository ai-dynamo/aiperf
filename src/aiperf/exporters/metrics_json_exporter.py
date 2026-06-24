# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from datetime import datetime

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models import MetricResult
from aiperf.common.models.export_models import (
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
            branch_stats=getattr(self._results, "branch_stats", None),
        )

        from aiperf.dataset.provenance import public_dataset_provenance

        run_metadata: dict[str, object] = {}
        dataset = public_dataset_provenance(self._user_config)
        if dataset is not None:
            run_metadata["dataset"] = dataset

        context_overflow_count = int(
            getattr(self._results, "context_overflow_count", 0) or 0
        )
        if context_overflow_count:
            existing_context_overflow = prepared_json_metrics.get(
                "context_overflow_count"
            )
            if existing_context_overflow is None:
                prepared_json_metrics["context_overflow_count"] = JsonMetricResult(
                    unit="requests",
                    avg=float(context_overflow_count),
                )
            else:
                prepared_json_metrics["context_overflow_count"] = (
                    existing_context_overflow.model_copy(
                        update={
                            "avg": float(
                                (existing_context_overflow.avg or 0)
                                + context_overflow_count
                            )
                        }
                    )
                )

        # Add all prepared metrics dynamically
        for metric_tag, json_result in prepared_json_metrics.items():
            setattr(export_data, metric_tag, json_result)

        # Stamp scenario submission metadata for single-run exports. Mirrors the
        # carrier-key contract used by AggregateConfidenceJsonExporter: validator
        # outcome lives on user_config._scenario_outcome (set by
        # UserConfig._run_scenario_validator) and runtime totals are summed from
        # the prepared metric results.
        scenario_name = getattr(self._user_config, "scenario", None)
        if scenario_name is not None:
            from aiperf.exporters.aggregate.aggregate_base_exporter import (
                _build_run_metadata_dict,
                compute_submission_outcome,
            )

            outcome = getattr(self._user_config, "_scenario_outcome", None)
            validator_submission_valid = (
                outcome.submission_valid if outcome is not None else True
            )
            validator_reasons = (
                list(outcome.submission_invalid_reasons) if outcome is not None else []
            )

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
                scenario_name=scenario_name,
                validator_submission_valid=validator_submission_valid,
                validator_reasons=validator_reasons,
                total_responses=total_responses,
                context_overflow_count=context_overflow_count,
                was_cancelled=bool(self._results.was_cancelled),
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
        return export_data.model_dump_json(
            indent=2, exclude_unset=True, exclude_none=True
        )

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
