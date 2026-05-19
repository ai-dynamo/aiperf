# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PAVA-residual cliff guard. stdlib-only (statistics.pstdev), no numpy/ruptures dep.

Smooth-isotonic regression assumes some smoothness. Sarathi-Serve (arxiv
2403.02310 fig 8) shows prefill-prioritizing servers can have a 28x p99-TBT
cliff at saturation — no smooth fit is honest about a discontinuity.

The detector compares the residual at the most-recently-probed raw point
against ``3 * sigma_local`` (rolling stddev of the last few raw margins). When
the residual exceeds that AND the bracket gap is still wider than the user's
precision target, we declare a cliff so the planner can report
``boundary_type: "cliff"`` instead of pretending the spline interpolated a
discontinuity.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import pstdev

from aiperf.common.environment import Environment

__all__ = ["detect_cliff"]

_RESIDUAL_SIGMA_MULTIPLIER = 3.0
_LOCAL_WINDOW = 3
_MIN_LOCAL_POINTS = 3


def detect_cliff(
    raw_points: list[tuple[int, float]],
    fitted_curve: Callable[[float], float],
    *,
    feasible_max: int | None,
    infeasible_min: int | None,
    x_hi: int,
    precision: float | None = None,
) -> bool:
    """Return True iff the last raw probe sits outside the smooth fit and the bracket is still wide.

    ``raw_points`` is the chronological ``(x, margin)`` series of probes
    (bracket + fit-step internals). ``sigma_local`` is the population stddev of
    the last ``_LOCAL_WINDOW`` (3) raw margins; fewer than 3 points or
    ``sigma_local == 0`` returns False because we don't have enough evidence
    to call a discontinuity. ``bracket_gap = infeasible_min - feasible_max``
    falls back to 0 when either bound is unset (no bracket -> no cliff
    evidence). Cliff iff
    ``|raw_margin_last - fitted_curve(x_last)| > 3 * sigma_local`` AND
    ``bracket_gap > precision * x_hi``.

    ``precision`` defaults to
    ``Environment.SEARCH_PLANNER.SLA_PRECISION_DEFAULT`` when None.
    """
    if precision is None:
        precision = Environment.SEARCH_PLANNER.SLA_PRECISION_DEFAULT
    if len(raw_points) < _MIN_LOCAL_POINTS:
        return False
    recent = [m for _, m in raw_points[-_LOCAL_WINDOW:]]
    sigma_local = pstdev(recent)
    if sigma_local == 0.0:
        return False
    x_last, margin_last = raw_points[-1]
    fit_last = float(fitted_curve(float(x_last)))
    residual = abs(margin_last - fit_last)
    if residual <= _RESIDUAL_SIGMA_MULTIPLIER * sigma_local:
        return False
    bracket_gap = (
        infeasible_min - feasible_max
        if feasible_max is not None and infeasible_min is not None
        else 0
    )
    return bracket_gap > precision * x_hi
