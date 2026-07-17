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

        # Use this exporter's own SCHEMA_VERSION, not JsonExportData.SCHEMA_VERSION,
        # because the aggregate file's per-metric shape evolves independently
        # from the regular profile export.
        export_data = JsonExportData(
            schema_version=self.SCHEMA_VERSION,
            aiperf_version=aiperf_version,
        )

        # ``result_metadata`` is consumed (carrier keys popped) by the submission
        # fold below; the survivors flow into the public aggregate metadata.
        result_metadata = dict(self._result.metadata)
        run_metadata = self._build_submission_metadata(result_metadata)

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

    def _build_submission_metadata(self, result_metadata: dict) -> dict:
        """Fold the aggregate (multi-run) scenario-submission verdict.

        Mirrors the single-run ``MetricsJsonExporter._fold_runtime_submission_outcome``
        for the ``--num-profile-runs > 1`` / sweep path: the scenario-lock
        outcome (validator) is combined ACROSS RUNS with the cross-run
        context-overflow rate and cancellation via ``compute_submission_outcome``
        (InferenceX AgentX RFC §7). Null-safe: a non-scenario run carries no
        ``scenario_name`` and yields an empty dict (no submission fields).

        Args:
            result_metadata: A mutable copy of ``self._result.metadata`` whose
                underscore-prefixed carrier keys are popped in place so they do
                not leak into the public aggregate metadata.

        Returns:
            The ``scenario`` / ``submission_valid`` / ``submission_invalid_reasons``
            sub-dict to merge into the aggregate output, or ``{}`` for a
            non-scenario run.
        """
        from aiperf.exporters.aggregate.aggregate_base_exporter import (
            _build_run_metadata_dict,
            compute_submission_outcome,
        )

        scenario_name = self._pop_scenario_name(result_metadata)
        validator_submission_valid = result_metadata.pop(
            "_validator_submission_valid", None
        )
        validator_reasons = result_metadata.pop(
            "_validator_submission_invalid_reasons", None
        )
        total_responses, context_overflow_count = self._cross_run_response_counts(
            result_metadata
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
        return _build_run_metadata_dict(
            scenario_name=scenario_name,
            submission_valid=submission_valid,
            submission_invalid_reasons=submission_invalid_reasons,
        )

    def _pop_scenario_name(self, result_metadata: dict) -> str | None:
        """Resolve the active scenario name for the aggregate submission fold.

        Prefers the orchestrator-stamped ``_scenario_name`` carrier key (popped
        so it does not leak into the public metadata), falling back to the
        ``scenario`` key the aggregation strategy may already carry. Returns
        ``None`` for a non-scenario run, which short-circuits the whole
        submission fold to no output.

        Args:
            result_metadata: A mutable copy of ``self._result.metadata`` whose
                underscore-prefixed carrier keys are popped in place.

        Returns:
            The scenario name, or ``None`` when no scenario is active.
        """
        carrier = result_metadata.pop("_scenario_name", None)
        if carrier is not None:
            return str(carrier)
        existing = result_metadata.get("scenario")
        return str(existing) if existing is not None else None

    def _cross_run_response_counts(self, result_metadata: dict) -> tuple[int, int]:
        """Resolve ``(total_responses, context_overflow_count)`` across runs.

        Prefers the orchestrator-stamped ``_total_responses`` /
        ``_context_overflow_count`` carrier keys (summed across runs upstream),
        popping them so they do not leak into the public metadata. When those
        carriers are absent, derives the counts from the confidence aggregate's
        own per-run-mean count metrics (``request_count_avg``,
        ``error_request_count_avg``, ``context_overflow_count_avg``).

        The overflow RATE is rate-equivalent under both sourcing modes: the
        cross-run mean and the cross-run sum share the same run count, so
        ``mean(overflow) / mean(total) == sum(overflow) / sum(total)`` and
        ``compute_submission_outcome`` derives the identical verdict. The
        denominator is ``request_count + error_request_count + overflow`` per
        the InferenceX AgentX spec §4.8 / §7 (all responses received).

        Args:
            result_metadata: A mutable copy of ``self._result.metadata`` whose
                underscore-prefixed carrier keys are popped in place.

        Returns:
            A ``(total_responses, context_overflow_count)`` tuple of
            non-negative ints.
        """
        carrier_total = result_metadata.pop("_total_responses", None)
        carrier_overflow = result_metadata.pop("_context_overflow_count", None)
        if carrier_total is not None or carrier_overflow is not None:
            return (
                int(carrier_total or 0),
                int(carrier_overflow or 0),
            )

        overflow = self._aggregate_metric_mean("context_overflow_count")
        total = (
            self._aggregate_metric_mean("request_count")
            + self._aggregate_metric_mean("error_request_count")
            + overflow
        )
        return total, overflow

    def _aggregate_metric_mean(self, metric_tag: str) -> int:
        """Return a count metric's cross-run mean from the aggregate, or 0.

        The confidence aggregation flattens each metric to ``{tag}_{stat}``
        keys; the per-run average lands on ``{tag}_avg``. A ``ConfidenceMetric``
        exposes that value as ``.mean``; a non-confidence fallback object may
        expose ``.avg``. Missing or non-numeric -> 0.

        Args:
            metric_tag: The base metric tag (e.g. ``"request_count"``).

        Returns:
            The non-negative integer cross-run mean, or 0 when absent.
        """
        metric = self._result.metrics.get(f"{metric_tag}_avg")
        if metric is None:
            return 0
        value = getattr(metric, "mean", None)
        if value is None:
            value = getattr(metric, "avg", None)
        if value is None:
            return 0
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0
