# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CI width convergence criterion using Student's t confidence interval."""

import logging
import math

import numpy as np
from scipy.stats import t as t_dist

from aiperf.orchestrator.convergence.base import ConvergenceCriterion
from aiperf.orchestrator.models import RunResult

logger = logging.getLogger(__name__)


class CIWidthConvergence(ConvergenceCriterion):
    """Converges when the Student's t CI width relative to the mean falls below a threshold."""

    def __init__(
        self,
        *,
        metric: str,
        stat: str = "avg",
        threshold: float = 0.10,
        confidence_level: float = 0.95,
        min_runs: int = 3,
    ) -> None:
        self._metric = metric
        self._stat = stat
        self._threshold = threshold
        self._confidence_level = confidence_level
        self._min_runs = min_runs

    def is_converged(self, results: list[RunResult]) -> bool:
        """Check whether the CI width ratio is below the threshold.

        Returns False when fewer than min_runs (or 2, whichever is larger)
        successful runs have the metric, or when the mean is zero.
        """
        values = self._extract_values(results)

        if len(values) < max(self._min_runs, 2):
            if len(values) == 0 and len(results) >= self._min_runs:
                logger.warning(
                    "Convergence metric '%s' (stat '%s') not found in any run's summary metrics; "
                    "convergence will never trigger. Check --convergence-metric spelling. "
                    "Available metrics: %s",
                    self._metric,
                    self._stat,
                    sorted(
                        {k for r in results if r.success for k in r.summary_metrics}
                    ),
                )
            return False

        mean = np.mean(values)
        if mean == 0.0:
            return False

        n = len(values)
        std = np.std(values, ddof=1)
        se = std / math.sqrt(n)
        t_crit = t_dist.ppf((1 + self._confidence_level) / 2, df=n - 1)
        ci_half = t_crit * se
        ci_width_ratio = (2 * ci_half) / abs(mean)

        return bool(ci_width_ratio < self._threshold)

    def _extract_values(self, results: list[RunResult]) -> list[float]:
        """Extract the target metric/stat from successful runs."""
        values: list[float] = []
        for r in results:
            if not r.success:
                continue
            metric_result = r.summary_metrics.get(self._metric)
            if metric_result is None:
                continue
            val = getattr(metric_result, self._stat, None)
            if val is None:
                continue
            values.append(float(val))
        return values
