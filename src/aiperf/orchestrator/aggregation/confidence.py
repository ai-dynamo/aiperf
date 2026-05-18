# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Confidence aggregation strategy for multi-run results."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiperf.common.constants import STAT_KEYS
from aiperf.orchestrator.aggregation.base import AggregateResult, AggregationStrategy
from aiperf.orchestrator.models import RunResult

if TYPE_CHECKING:
    from aiperf.common.models.export_models import JsonMetricResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ConfidenceMetric:
    """Statistics for a single metric across runs."""

    mean: float
    """Sample mean."""

    std: float
    """Sample standard deviation (ddof=1)."""

    min: float
    """Minimum value across runs."""

    max: float
    """Maximum value across runs."""

    cv: float
    """Coefficient of variation (std/mean)."""

    se: float
    """Standard error (std/sqrt(n))."""

    ci_low: float
    """Lower bound of the confidence interval."""

    ci_high: float
    """Upper bound of the confidence interval."""

    t_critical: float
    """t-distribution critical value used for CI calculation."""

    unit: str
    """Unit of measurement (e.g., "ms", "requests/sec")."""

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
            ValueError: If zero successful runs (all runs failed). A single
                successful run is no longer fatal — see the single-run
                degraded path below.
        """
        # Separate successful and failed runs
        successful = [r for r in results if r.success]
        failed = [
            {"label": r.label, "error": r.error} for r in results if not r.success
        ]

        if len(successful) == 0:
            raise ValueError(
                "All runs failed - cannot compute confidence statistics. "
                f"Total runs: {len(results)}, Failed runs: {len(failed)}. "
                "Please check the error messages in the logs and ensure your "
                "benchmark configuration is correct."
            )

        if len(successful) == 1:
            # Single-run degraded mode. Confidence intervals require >= 2
            # observations (sample variance, t-distribution df=n-1), so we
            # cannot produce meaningful CIs from one run. But crashing the
            # whole sweep over numRuns=1 is hostile to the common cases:
            # (a) the user explicitly chose num_profile_runs=1 to iterate
            # quickly; (b) an outer parameter sweep happens to bottom out at
            # one cell. Report point estimates with std=0 and CI=[mean,mean]
            # so downstream exporters / SLA filters still receive each
            # metric's value, and flag the degenerate case via metadata so
            # CI-consuming UIs can render a "n=1, no CI" badge instead of a
            # zero-width error bar.
            logger.warning(
                "ConfidenceAggregation: only 1 successful run "
                "(num_successful=%d / total=%d); reporting point estimates "
                "with std=0 and CI collapsed to the mean. Set "
                "num_profile_runs >= 2 for meaningful confidence intervals.",
                len(successful),
                len(results),
            )
            metrics = self._aggregate_metrics_single_run(successful[0])
            return AggregateResult(
                aggregation_type="confidence",
                num_runs=len(results),
                num_successful_runs=1,
                failed_runs=failed,
                metrics=metrics,
                metadata={
                    "confidence_level": self.confidence_level,
                    "run_labels": [r.label for r in successful],
                    "single_run": True,
                },
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
        if not results or not results[0].summary_metrics:
            return {}

        metric_stat_pairs = self._collect_metric_stat_pairs(results)

        aggregated = {}
        for metric_name, stat_key in metric_stat_pairs:
            values, unit = self._extract_values_for_pair(results, metric_name, stat_key)
            if not values:
                continue

            flattened_key = f"{metric_name}_{stat_key}"
            aggregated[flattened_key] = self._compute_confidence_stats(
                values, flattened_key, unit
            )

        return aggregated

    @staticmethod
    def _aggregate_metrics_single_run(run: RunResult) -> dict[str, ConfidenceMetric]:
        """Build degenerate per-metric ConfidenceMetric records for a single run.

        Used by the ``len(successful) == 1`` branch of :meth:`aggregate`. Each
        populated stat surfaces as a ConfidenceMetric whose ``mean``, ``min``,
        and ``max`` all equal the single observation; ``std``, ``cv``, ``se``
        are 0; ``ci_low`` and ``ci_high`` collapse to the mean; ``t_critical``
        is NaN to flag the absence of a true CI to downstream UI code.

        This preserves the per-metric / per-stat keying contract of the
        multi-run path (``"time_to_first_token_avg"``, ``"time_to_first_token_p99"``,
        ...) so exporters and SLA-filter consumers don't need to special-case
        the single-run shape.
        """
        aggregated: dict[str, ConfidenceMetric] = {}
        if not run.summary_metrics:
            return aggregated
        for metric_name, metric_result in run.summary_metrics.items():
            for stat_key in STAT_KEYS:
                value = getattr(metric_result, stat_key, None)
                if value is None:
                    continue
                v = float(value)
                aggregated[f"{metric_name}_{stat_key}"] = ConfidenceMetric(
                    mean=v,
                    std=0.0,
                    min=v,
                    max=v,
                    cv=0.0,
                    se=0.0,
                    ci_low=v,
                    ci_high=v,
                    t_critical=float("nan"),
                    unit=metric_result.unit,
                )
        return aggregated

    @staticmethod
    def _collect_metric_stat_pairs(
        results: list[RunResult],
    ) -> set[tuple[str, str]]:
        """Collect all unique (metric_name, stat_key) pairs populated across runs."""
        pairs: set[tuple[str, str]] = set()
        for result in results:
            for metric_name, metric_result in result.summary_metrics.items():
                for stat_key in STAT_KEYS:
                    if getattr(metric_result, stat_key, None) is not None:
                        pairs.add((metric_name, stat_key))
        return pairs

    @staticmethod
    def _extract_values_for_pair(
        results: list[RunResult], metric_name: str, stat_key: str
    ) -> tuple[list[float], str]:
        """Extract values and unit for a (metric_name, stat_key) pair across runs.

        Skips ``None`` and non-finite values (NaN/+inf/-inf) so a single bad
        observation in one trial does not poison the aggregate via
        ``np.mean`` / ``np.std``. The metric's unit comes from the first
        finite trial encountered.
        """
        from aiperf.common.finite import is_finite_value

        values: list[float] = []
        unit = ""
        for result in results:
            if metric_name not in result.summary_metrics:
                continue
            metric_result = result.summary_metrics[metric_name]
            value = getattr(metric_result, stat_key, None)
            if value is None or not is_finite_value(value):
                continue
            values.append(float(value))
            if not unit:
                unit = metric_result.unit
        return values, unit

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
        import math

        from scipy import stats

        from aiperf.common.finite import nan_safe_mean, nan_safe_std

        # Defensive filter: callers (``_extract_values_for_pair``) already
        # skip non-finite samples, but using nan-safe aggregations here
        # ensures the discipline survives future callsite refactors. A
        # NaN slipping through would otherwise poison every downstream
        # field (CV, SE, CI) and silently round-trip to JSON ``null``.
        mean_opt = nan_safe_mean(values)
        std_opt = nan_safe_std(values, ddof=1)
        if mean_opt is None or std_opt is None:
            # Should never happen because the caller drops empty value lists
            # and the single-run branch handles n=1 separately, but degrade
            # to NaN sentinels rather than crashing if it does.
            mean = float("nan")
            std = float("nan")
        else:
            mean = float(mean_opt)
            std = float(std_opt)
        n = len(values)

        # Coefficient of variation (handle division by zero)
        # CV is expressed as a ratio (not percentage), so no *100
        cv = std / mean if mean != 0 else float("inf")

        # Standard error
        se = std / math.sqrt(n)

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
