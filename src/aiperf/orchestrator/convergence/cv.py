# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CV convergence criterion using coefficient of variation."""

import logging

import numpy as np

from aiperf.orchestrator.convergence.base import ConvergenceCriterion
from aiperf.orchestrator.models import RunResult

logger = logging.getLogger(__name__)


class CVConvergence(ConvergenceCriterion):
    """Converges when the coefficient of variation across run-level averages falls below a threshold."""

    def __init__(
        self,
        metric: str,
        threshold: float = 0.05,
        min_runs: int = 3,
        stat: str = "avg",
    ) -> None:
        self._metric = metric
        self._threshold = threshold
        self._min_runs = min_runs
        self._stat = stat

    def is_converged(self, results: list[RunResult]) -> bool:
        """Check whether the CV is below the threshold.

        Returns False when fewer than min_runs successful runs have the metric
        or when the mean is zero.
        """
        values = self._extract_values(results)

        if len(values) < self._min_runs:
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

        cv = float(np.std(values, ddof=1) / abs(mean))
        return cv < self._threshold

    def _extract_values(self, results: list[RunResult]) -> list[float]:
        """Extract the target metric stat from successful runs."""
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
