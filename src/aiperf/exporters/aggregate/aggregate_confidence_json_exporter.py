# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON exporter for confidence aggregate results."""

from typing import TYPE_CHECKING, ClassVar

import orjson

from aiperf.common.finite import scrub_non_finite
from aiperf.exporters.aggregate.aggregate_base_exporter import AggregateBaseExporter

if TYPE_CHECKING:
    from aiperf.common.models.export_models import JsonExportData


class AggregateConfidenceJsonExporter(AggregateBaseExporter):
    """Exports confidence aggregate results to JSON format.

    Uses adapter pattern to convert AggregateResult to JsonExportData,
    then leverages Pydantic serialization for consistency with single-run exports.

    Design:
    - Reuses JsonExportData and JsonMetricResult models
    - Uses same serialization approach as MetricsJsonExporter
    - Owns its own SCHEMA_VERSION because the per-metric shape (mean, std, cv,
      se, ci_low, ci_high, t_critical) differs from the regular profile export
      and evolves on its own cadence.
    """

    # Bump on breaking changes to the aggregate-confidence on-disk shape only.
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def get_file_name(self) -> str:
        """Return JSON file name.

        Returns:
            str: "profile_export_aiperf_aggregate.json"
        """
        return "profile_export_aiperf_aggregate.json"

    def _generate_content(self) -> str:
        """Generate JSON content from aggregate result.

        Uses adapter pattern:
        1. Convert AggregateResult → JsonExportData
        2. Serialize using Pydantic (same as MetricsJsonExporter)

        Returns:
            str: JSON content string
        """
        # Convert to JsonExportData format (adapter pattern)
        export_data = self._aggregate_to_export_data()

        # Pydantic's model_dump_json silently coerces NaN/inf to JSON null,
        # which collides with the explicit-None "metric was missing"
        # contract. Round-trip via model_dump + scrub_non_finite + orjson
        # so null on disk only ever means "absent".
        payload = export_data.model_dump(
            mode="json", exclude_unset=True, exclude_none=True
        )
        return orjson.dumps(
            scrub_non_finite(payload), option=orjson.OPT_INDENT_2
        ).decode("utf-8")

    def _aggregate_to_export_data(self) -> "JsonExportData":
        """Convert AggregateResult to JsonExportData format.

        This is the adapter that bridges aggregate domain to export format.
        Reuses the same Pydantic models as single-run exports for consistency.

        Returns:
            JsonExportData with aggregate metrics and metadata
        """
        from aiperf import __version__ as aiperf_version
        from aiperf.common.models.export_models import JsonExportData
        from aiperf.exporters.aggregate.aggregate_base_exporter import (
            _build_run_metadata_dict,
            compute_submission_outcome,
        )

        # Use this exporter's own SCHEMA_VERSION, not JsonExportData.SCHEMA_VERSION,
        # because the aggregate file's per-metric shape evolves independently
        # from the regular profile export.
        export_data = JsonExportData(
            schema_version=self.SCHEMA_VERSION,
            aiperf_version=aiperf_version,
        )

        # Pull scenario-submission inputs out of the aggregate metadata
        # (see cli_runner / orchestrator: these underscore-prefixed keys are
        # the carrier from validator+runtime to exporter, and are stripped
        # before merging the rest of metadata into the output).
        result_metadata = dict(self._result.metadata)
        scenario_name = result_metadata.pop("_scenario_name", None)
        validator_submission_valid = result_metadata.pop(
            "_validator_submission_valid", None
        )
        validator_reasons = result_metadata.pop(
            "_validator_submission_invalid_reasons", None
        )
        total_responses = int(result_metadata.pop("_total_responses", 0) or 0)
        context_overflow_count = int(
            result_metadata.pop("_context_overflow_count", 0) or 0
        )
        was_cancelled = bool(result_metadata.pop("_was_cancelled", False))

        submission_valid, submission_invalid_reasons = compute_submission_outcome(
            scenario_name=scenario_name,
            validator_submission_valid=validator_submission_valid,
            validator_reasons=validator_reasons,
            total_responses=total_responses,
            context_overflow_count=context_overflow_count,
            was_cancelled=was_cancelled,
        )

        run_metadata = _build_run_metadata_dict(
            scenario_name=scenario_name,
            submission_valid=submission_valid,
            submission_invalid_reasons=submission_invalid_reasons,
        )

        # Add aggregate-specific metadata as extra field
        # (JsonExportData has extra="allow" to support this)
        aggregate_metadata = {
            "aggregation_type": self._result.aggregation_type,
            "num_profile_runs": self._result.num_runs,
            "num_successful_runs": self._result.num_successful_runs,
            "failed_runs": self._result.failed_runs,
            **result_metadata,
            **run_metadata,
        }
        export_data.metadata = aggregate_metadata

        # Convert metrics and group them under "metrics" key
        metrics_dict = {}
        for metric_name, metric in self._result.metrics.items():
            if hasattr(metric, "mean"):
                # ConfidenceMetric - include all fields directly
                metric_data = {
                    "mean": metric.mean,
                    "std": metric.std,
                    "min": metric.min,
                    "max": metric.max,
                    "cv": metric.cv,
                    "se": metric.se,
                    "ci_low": metric.ci_low,
                    "ci_high": metric.ci_high,
                    "t_critical": metric.t_critical,
                    "unit": metric.unit,
                }
                metrics_dict[metric_name] = metric_data
            else:
                # For other metric types, store as-is
                metrics_dict[metric_name] = metric

        export_data.metrics = metrics_dict

        return export_data
