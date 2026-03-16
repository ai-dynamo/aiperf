# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distribution convergence criterion using two-sample Kolmogorov-Smirnov test."""

import logging

import numpy as np
from scipy.stats import ks_2samp

from aiperf.orchestrator.convergence.base import (
    DEFAULT_JSONL_FILENAME,
    ConvergenceCriterion,
)
from aiperf.orchestrator.models import RunResult

logger = logging.getLogger(__name__)


class DistributionConvergence(ConvergenceCriterion):
    """Converges when the latest run's distribution matches prior runs via KS test."""

    def __init__(
        self,
        metric: str,
        p_value_threshold: float = 0.05,
        min_runs: int = 3,
        jsonl_filename: str = DEFAULT_JSONL_FILENAME,
    ) -> None:
        self._metric = metric
        self._p_value_threshold = p_value_threshold
        self._min_runs = min_runs
        self._jsonl_filename = jsonl_filename

    def is_converged(self, results: list[RunResult]) -> bool:
        """Check whether the latest run's distribution matches prior runs.

        Combines per-request metrics from runs 1..N-1 into distribution A and
        run N into distribution B. Returns True when the KS test p-value
        exceeds the threshold.

        Returns False when fewer than min_runs successful runs exist, when
        JSONL data is missing, or when either distribution has fewer than
        2 data points.
        """
        successful = [r for r in results if r.success and r.artifacts_path is not None]

        if len(successful) < self._min_runs:
            return False

        dist_a: list[float] = []
        for r in successful[:-1]:
            dist_a.extend(
                self._load_request_metrics(
                    r.artifacts_path, self._metric, self._jsonl_filename
                )
            )

        dist_b = self._load_request_metrics(
            successful[-1].artifacts_path, self._metric, self._jsonl_filename
        )

        if len(dist_a) < 2 or len(dist_b) < 2:
            if len(dist_a) == 0 and len(dist_b) == 0:
                logger.warning(
                    "Convergence metric '%s' not found in any run's JSONL data; "
                    "convergence will never trigger. Check --convergence-metric spelling.",
                    self._metric,
                )
            return False

        _, p_value = ks_2samp(np.array(dist_a), np.array(dist_b))
        return bool(p_value > self._p_value_threshold)
