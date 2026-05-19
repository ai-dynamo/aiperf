# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyperband-flavored replicate budget allocator.

The boundary probe dominates SLA-search accuracy because its margin variance
is what the bracket CI is computed over. ``replicate_count`` allocates more
replicates to noisy / tight constraints (``sigma_margin / |threshold|`` large)
and fewer to clean / loose ones, capped at 20 so a single degenerate
constraint can't consume the entire iteration budget.

``boundary_ci`` is a thin wrapper over ``scipy.stats.bootstrap`` (1.13+) that
handles the singleton-margin edge case — the percentile bootstrap is undefined
on a single observation, so we degenerate to a point CI rather than letting
scipy raise.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import bootstrap

__all__ = ["boundary_ci", "replicate_count"]

_REPLICATE_FLOOR = 3
_REPLICATE_CEIL = 20
_REPLICATE_GAIN = 4
_THRESHOLD_EPS = 1e-9


def replicate_count(sigma_margin: float, threshold: float, override: int = 0) -> int:
    """Hyperband-flavored replicate count for the boundary probe.

    Formula:

        R(x) = min(20, max(3, ceil(4 * (sigma_margin / max(|threshold|, eps))^2)))

    The ``min(20, ...)`` cap is load-bearing: when ``sigma_margin > |threshold|``
    the formula would explode (a noisy degenerate constraint), so we hard-cap
    to keep the iteration budget bounded. ``eps = 1e-9`` prevents
    zero-division when a constraint threshold is exactly zero (e.g. a margin
    expressed as ``observed >= 0``).

    ``override > 0`` short-circuits the formula and returns the override
    directly — wires the user-facing ``--sla-replicates N`` flag. ``0`` (the
    default) means "auto" and runs the formula.
    """
    if override > 0:
        return int(override)
    denom = max(abs(threshold), _THRESHOLD_EPS)
    ratio_sq = (sigma_margin / denom) ** 2
    raw = math.ceil(_REPLICATE_GAIN * ratio_sq)
    return min(_REPLICATE_CEIL, max(_REPLICATE_FLOOR, int(raw)))


def boundary_ci(margins: list[float], n_resamples: int = 10000) -> tuple[float, float]:
    """Bootstrap 95% CI on the mean of replicate margins.

    Returns ``(ci_low, ci_high)``. Singleton input ``len(margins) == 1``
    short-circuits to a point CI ``(margins[0], margins[0])`` because the
    percentile bootstrap is undefined on one observation. Empty input is
    forbidden (caller invariant — only invoked after at least one replicate
    has landed).
    """
    if len(margins) == 1:
        return (float(margins[0]), float(margins[0]))
    data = np.asarray(margins, dtype=float)
    result = bootstrap(
        (data,),
        statistic=np.mean,
        n_resamples=n_resamples,
        confidence_level=0.95,
    )
    return (
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )
