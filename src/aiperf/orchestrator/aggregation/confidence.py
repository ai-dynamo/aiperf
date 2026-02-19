# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Confidence aggregation strategy for multi-run results."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from aiperf.common.constants import STAT_KEYS
from aiperf.orchestrator.aggregation.base import AggregateResult, AggregationStrategy
from aiperf.orchestrator.models import RunResult

if TYPE_CHECKING:
    from aiperf.common.models.export_models import JsonMetricResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ConfidenceMetric:
    """Statistics for a single metric across runs.

    Attributes:
        mean: Sample mean
        std: Sample standard deviation (ddof=1)
        min: Minimum value
        max: Maximum value
        cv: Coefficient of variation (std/mean)
        se: Standard error (std/sqrt(n))
        ci_low: Lower bound of confidence interval
        ci_high: Upper bound of confidence interval
        t_critical: t-distribution critical value used for CI
        unit: Unit of measurement (e.g., "ms", "requests/sec")
    """

    mean: float
    std: float
    min: float
    max: float
    cv: float
    se: float
    ci_low: float
    ci_high: float
    t_critical: float
    unit: str

    def to_json_result(self) -> "JsonMetricResult":
        """Convert to JsonMetricResult for export.

        Maps confidence statistics to JSON export format:
        - mean → avg (mean of run-level averages)
        - std → std (std of run-level averages)
        - min/max → min/max (across runs)

        Confidence-specific fields (cv, se, ci_low, ci_high, t_critical)
        are added as extra fields via JsonExportData's extra="allow" setting.

        Returns:
            JsonMetricResult compatible with existing exporters
        """
        from aiperf.common.models.export_models import JsonMetricResult

        return JsonMetricResult(
            unit=self.unit,
            avg=self.mean,
            std=self.std,
            min=self.min,
            max=self.max,
        )


class ConfidenceAggregation(AggregationStrategy):
    """Aggregation strategy for confidence reporting.

    Computes mean, std, CV, and confidence intervals for each metric.

    Attributes:
        confidence_level: Confidence level for intervals (default: 0.95)
    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        """Initialize ConfidenceAggregation.

        Args:
            confidence_level: Confidence level for intervals (0 < level < 1)

        Raises:
            ValueError: If confidence_level is not between 0 and 1
        """
        if not 0 < confidence_level < 1:
            raise ValueError(
                f"Invalid confidence level: {confidence_level}. "
                "Confidence level must be between 0 and 1 (exclusive). "
                "Common values: 0.90 (90%), 0.95 (95%), 0.99 (99%)."
            )
        self.confidence_level = confidence_level

    def get_aggregation_type(self) -> str:
        """Return aggregation type identifier."""
        return "confidence"

    def aggregate(self, results: list[RunResult]) -> AggregateResult:
        """Aggregate results for confidence reporting.

        Args:
            results: List of RunResult from orchestrator

        Returns:
            AggregateResult with confidence statistics

        Raises:
            ValueError: If fewer than 2 successful runs
        """
        # Separate successful and failed runs
        successful = [r for r in results if r.success]
        failed = [
            {"label": r.label, "error": r.error} for r in results if not r.success
        ]

        if len(successful) < 2:
            if len(successful) == 0:
                raise ValueError(
                    "All runs failed - cannot compute confidence statistics. "
                    f"Total runs: {len(results)}, Failed runs: {len(failed)}. "
                    "Please check the error messages in the logs and ensure your "
                    "benchmark configuration is correct."
                )
            else:
                raise ValueError(
                    f"Insufficient successful runs for confidence intervals. "
                    f"Got {len(successful)} successful run(s), but need at least 2. "
                    f"Total runs: {len(results)}, Failed runs: {len(failed)}. "
                    "Consider increasing --num-profile-runs or investigating why runs are failing."
                )

        # Aggregate each metric
        metrics = self._aggregate_metrics(successful)

        return AggregateResult(
            aggregation_type="confidence",
            num_runs=len(results),
            num_successful_runs=len(successful),
            failed_runs=failed,
            metrics=metrics,
            metadata={
                "confidence_level": self.confidence_level,
                "run_labels": [r.label for r in successful],
            },
        )

    def _aggregate_metrics(
        self, results: list[RunResult]
    ) -> dict[str, ConfidenceMetric]:
        """Aggregate each metric across runs.

        Args:
            results: List of successful RunResult

        Returns:
            Dict mapping flattened metric name to ConfidenceMetric
            (e.g., "time_to_first_token_avg", "time_to_first_token_p99")
        """
        # Get all metric names from first result
        if not results or not results[0].summary_metrics:
            return {}

        # Collect all unique metric names and stat keys across all runs
        metric_stat_pairs = set()
        for result in results:
            for metric_name, metric_result in result.summary_metrics.items():
                # Get all populated stat fields
                for stat_key in STAT_KEYS:
                    if getattr(metric_result, stat_key, None) is not None:
                        metric_stat_pairs.add((metric_name, stat_key))

        aggregated = {}
        for metric_name, stat_key in metric_stat_pairs:
            # Extract values for this metric+stat combination across all runs
            values = []
            unit = ""

            for result in results:
                if metric_name in result.summary_metrics:
                    metric_result = result.summary_metrics[metric_name]
                    value = getattr(metric_result, stat_key, None)
                    if value is not None:
                        values.append(value)
                        # Get unit from first occurrence
                        if not unit:
                            unit = metric_result.unit

            if not values:
                continue

            # Create flattened key for output (e.g., "time_to_first_token_p99")
            flattened_key = f"{metric_name}_{stat_key}"

            # Compute statistics
            aggregated[flattened_key] = self._compute_confidence_stats(
                values, flattened_key, unit
            )

        return aggregated

    def _compute_confidence_stats(
        self, values: list[float], metric_name: str, unit: str
    ) -> ConfidenceMetric:
        """Compute confidence statistics for a single metric.

        Args:
            values: List of metric values across runs
            metric_name: Name of the metric (e.g., "time_to_first_token_avg")
            unit: Unit of measurement (e.g., "ms", "requests/sec")

        Returns:
            ConfidenceMetric with computed statistics
        """
        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))  # Sample std (N-1)

        # Coefficient of variation (handle division by zero)
        # CV is expressed as a ratio (not percentage), so no *100
        cv = std / mean if mean != 0 else float("inf")

        # Standard error
        se = std / np.sqrt(n)

        # Confidence interval using t-distribution
        alpha = 1 - self.confidence_level
        df = n - 1
        t_critical = float(stats.t.ppf(1 - alpha / 2, df))

        margin = t_critical * se
        ci_low = mean - margin
        ci_high = mean + margin

        return ConfidenceMetric(
            mean=mean,
            std=std,
            min=float(min(values)),
            max=float(max(values)),
            cv=cv,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            t_critical=t_critical,
            unit=unit,
        )
