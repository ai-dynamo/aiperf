# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fit / replicate / cliff-bisect step bodies for ``SmoothIsotonicSLAPlanner``.

Module-level functions taking the planner as the first argument; they read
planner state and may mutate ``_phase``, ``_probe_queue``, ``_candidate_x``,
``_fit_count``, ``binding_constraint``, ``boundary_type``,
``boundary_ci_low/_hi``, ``_convergence_reason``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from scipy.optimize import isotonic_regression

from aiperf.common.environment import Environment
from aiperf.orchestrator.search_planner._cliff_detect import detect_cliff
from aiperf.orchestrator.search_planner._margin_normalize import normalize_margins
from aiperf.orchestrator.search_planner._replicate_budget import (
    boundary_ci,
    replicate_count,
)
from aiperf.orchestrator.search_planner._smooth_isotonic_fit import (
    find_root,
    smooth_isotonic_fit,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiperf.orchestrator.search_planner.smooth_isotonic import (
        SmoothIsotonicSLAPlanner,
    )


__all__ = ["plan_cliff_bisect_step", "plan_fit_step", "plan_replicate_step"]


# Adaptive expansion: refit when PAVA collapses to <= this many distinct values.
_FIT_MIN_DISTINCT = 3
# Hard cap on fit-step refit cycles; further fits drop into terminate.
_MAX_REFIT_CYCLES = 3
# Relative-precision target — read from
# ``Environment.SEARCH_PLANNER.SLA_PRECISION_DEFAULT`` (default 5%, same shape
# as monotonic_sla / vLLM-PCHIP).


def plan_fit_step(planner: SmoothIsotonicSLAPlanner) -> None:
    """Drain fit-step internal probes; fit, root-find, transition."""
    if planner._probe_queue:
        return  # still draining internal probes
    planner._fit_count += 1
    candidate = _fit_and_solve(planner)
    if candidate is None:
        candidate = _bisection_fallback(planner)
        if candidate is None:
            _finalize_with_reason(planner, "smooth_isotonic_pchip_fallback_bisection")
            return
    if _needs_more_fit_data(planner):
        if planner._fit_count >= _MAX_REFIT_CYCLES:
            planner._candidate_x = candidate
            _enter_replicate_or_terminate(planner, candidate)
            return
        _queue_more_probes_for_refit(planner)
        return
    planner._candidate_x = candidate
    _enter_replicate_or_terminate(planner, candidate)


def plan_cliff_bisect_step(planner: SmoothIsotonicSLAPlanner) -> None:
    """Drain cliff-bisect probes; halve [feasible_max, infeasible_min] until precision.

    Pure bisection within the cliff bracket — no PCHIP refit, since the curve
    is discontinuous and the spline is meaningless across the step. Mirrors
    ``MonotonicSLASearchPlanner._plan_bisect_step`` shape.
    """
    if planner._probe_queue:
        return
    if _bracket_precision_reached(planner):
        _finalize_with_reason(planner, "smooth_isotonic_cliff_precision_reached")
        return
    mid = _cliff_bisect_midpoint(planner)
    if mid is None:
        _finalize_with_reason(planner, "smooth_isotonic_cliff_precision_reached")
        return
    planner._probe_queue.append(mid)


def plan_replicate_step(planner: SmoothIsotonicSLAPlanner) -> None:
    """Drain replicate-step trials; bootstrap CI + terminate or refit."""
    if planner._probe_queue:
        return
    if planner._candidate_x is None or not planner.binding_constraint:
        _finalize_with_reason(planner, "smooth_isotonic_precision_reached")
        return
    margins_at_candidate = [
        m[planner.binding_constraint]
        for m in planner._raw_probes.get(planner._candidate_x, [])
        if planner.binding_constraint in m
    ]
    if len(margins_at_candidate) < 2:
        _finalize_with_reason(planner, "smooth_isotonic_precision_reached")
        return
    ci_low, ci_high = boundary_ci(margins_at_candidate)
    planner.boundary_ci_low = float(ci_low)
    planner.boundary_ci_high = float(ci_high)
    if ci_low <= 0.0 <= ci_high:
        _queue_more_probes_for_refit(planner)
        planner._phase = "fit"
        return
    _finalize_with_reason(planner, "smooth_isotonic_precision_reached")


# ----------------------------------------------------------------------
# Fit-step internals
# ----------------------------------------------------------------------


def _fit_and_solve(planner: SmoothIsotonicSLAPlanner) -> int | None:
    xs, per_filter_curves, per_filter_margins, per_filter_sigmas = _build_fit(planner)
    if not xs:
        return None
    binding_key = _select_binding_constraint(
        planner, per_filter_margins, per_filter_sigmas
    )
    planner.binding_constraint = binding_key
    binding_curve = per_filter_curves.get(binding_key)
    if binding_curve is None:
        return None
    if planner.feasible_max is None or planner.infeasible_min is None:
        return None
    root = find_root(
        binding_curve, float(planner.feasible_max), float(planner.infeasible_min)
    )
    if root is None:
        return None
    candidate = int(round(root))
    if candidate <= planner.feasible_max:
        candidate = planner.feasible_max + 1
    if candidate >= planner.infeasible_min:
        candidate = planner.infeasible_min - 1
    return candidate


def _build_fit(
    planner: SmoothIsotonicSLAPlanner,
) -> tuple[
    list[int],
    dict[str, Callable[[float], float] | None],
    dict[str, float],
    dict[str, float],
]:
    xs = sorted(planner._raw_probes.keys())
    if len(xs) < 2:
        return [], {}, {}, {}

    curves: dict[str, Callable[[float], float] | None] = {}
    margins: dict[str, float] = {}
    sigmas: dict[str, float] = {}

    for key in planner._filter_keys:
        ys: list[float] = []
        for x in xs:
            samples = [m[key] for m in planner._raw_probes[x] if key in m]
            if not samples:
                continue
            ys.append(sum(samples) / len(samples))
        if len(ys) < len(xs):
            curves[key] = None
            margins[key] = 0.0
            sigmas[key] = 0.0
            continue
        curve_fn, _ = smooth_isotonic_fit(xs, ys)
        curves[key] = curve_fn
        margins[key] = ys[-1]
        last_x = xs[-1]
        samples_at_last = [m[key] for m in planner._raw_probes[last_x] if key in m]
        if len(samples_at_last) >= 2:
            mean = sum(samples_at_last) / len(samples_at_last)
            var = sum((s - mean) ** 2 for s in samples_at_last) / (
                len(samples_at_last) - 1
            )
            sigmas[key] = math.sqrt(var)
        else:
            sigmas[key] = 0.0

    return xs, curves, margins, sigmas


def _select_binding_constraint(
    planner: SmoothIsotonicSLAPlanner,
    margins: dict[str, float],
    sigmas: dict[str, float],
) -> str:
    thresholds = {
        key: float(planner._sla_filters[i].threshold)
        for i, key in enumerate(planner._filter_keys)
    }
    sigmas_or_none = sigmas if any(s > 0 for s in sigmas.values()) else None
    _, binding_key = normalize_margins(margins, sigmas_or_none, thresholds)
    return binding_key


def _needs_more_fit_data(planner: SmoothIsotonicSLAPlanner) -> bool:
    if not planner.binding_constraint:
        return False
    xs = sorted(planner._raw_probes.keys())
    ys: list[float] = []
    for x in xs:
        samples = [
            m[planner.binding_constraint]
            for m in planner._raw_probes[x]
            if planner.binding_constraint in m
        ]
        if samples:
            ys.append(sum(samples) / len(samples))
    if len(ys) < 4:
        return True
    result = isotonic_regression(ys, increasing=True)
    distinct = len({round(v, 9) for v in result.x})
    return distinct < _FIT_MIN_DISTINCT


def _queue_more_probes_for_refit(planner: SmoothIsotonicSLAPlanner) -> None:
    if planner.feasible_max is None or planner.infeasible_min is None:
        return
    gap = planner.infeasible_min - planner.feasible_max
    if gap <= 1:
        return
    for frac in (0.125, 0.625):
        x = planner.feasible_max + max(1, round(gap * frac))
        x = min(x, planner.infeasible_min - 1)
        x = max(x, planner.feasible_max + 1)
        if x not in planner._raw_probes and x not in planner._probe_queue:
            planner._probe_queue.append(int(x))


def _bisection_fallback(planner: SmoothIsotonicSLAPlanner) -> int | None:
    if planner.feasible_max is None or planner.infeasible_min is None:
        return None
    gap = planner.infeasible_min - planner.feasible_max
    if gap <= 1:
        return None
    return planner.feasible_max + gap // 2


# ----------------------------------------------------------------------
# Replicate / cliff-bisect / termination
# ----------------------------------------------------------------------


def _enter_replicate_or_terminate(
    planner: SmoothIsotonicSLAPlanner, candidate: int
) -> None:
    cliff = _check_cliff(planner, candidate)
    planner.boundary_type = "cliff" if cliff else "smooth"

    if cliff:
        # Cliff branch: bisect the bracket in-place. PCHIP is meaningless across
        # a discontinuity, and replicates won't narrow [feasible_max,
        # infeasible_min] — only halving the bracket will.
        if _bracket_precision_reached(planner):
            _finalize_with_reason(planner, "smooth_isotonic_cliff_precision_reached")
            return
        mid = _cliff_bisect_midpoint(planner)
        if mid is None:
            _finalize_with_reason(planner, "smooth_isotonic_cliff_precision_reached")
            return
        planner._probe_queue.append(mid)
        planner._phase = "cliff_bisect"
        return

    if _bracket_precision_reached(planner):
        _finalize_with_reason(planner, "smooth_isotonic_precision_reached")
        return

    budget = _replicate_budget(planner)
    if budget <= 0:
        if candidate not in planner._raw_probes:
            planner._probe_queue.append(candidate)
            planner._phase = "fit"
            return
        _finalize_with_reason(planner, "smooth_isotonic_precision_reached")
        return

    for _ in range(budget):
        planner._probe_queue.append(candidate)
    planner._phase = "replicate"


def _cliff_bisect_midpoint(planner: SmoothIsotonicSLAPlanner) -> int | None:
    """Midpoint of [feasible_max, infeasible_min], nudged inward on collision."""
    if planner.feasible_max is None or planner.infeasible_min is None:
        return None
    gap = planner.infeasible_min - planner.feasible_max
    if gap <= 1:
        return None
    mid = planner.feasible_max + gap // 2
    if mid <= planner.feasible_max:
        mid = planner.feasible_max + 1
    if mid >= planner.infeasible_min:
        mid = planner.infeasible_min - 1
    if mid <= planner.feasible_max or mid >= planner.infeasible_min:
        return None
    return int(mid)


def _check_cliff(planner: SmoothIsotonicSLAPlanner, candidate: int) -> bool:
    """PAVA-residual cliff guard. Returns False on degenerate fit input."""
    del candidate  # currently unused; reserved for future per-candidate residual.
    if not planner.binding_constraint:
        return False
    xs = sorted(planner._raw_probes.keys())
    raw_pts: list[tuple[int, float]] = []
    for x in xs:
        samples = [
            m[planner.binding_constraint]
            for m in planner._raw_probes[x]
            if planner.binding_constraint in m
        ]
        if samples:
            raw_pts.append((x, sum(samples) / len(samples)))
    ys = [v for _, v in raw_pts]
    if len(ys) < 2:
        return False
    try:
        curve_fn, _ = smooth_isotonic_fit([p[0] for p in raw_pts], ys)
    except (ValueError, RuntimeError):
        return False
    return detect_cliff(
        raw_points=raw_pts,
        fitted_curve=curve_fn,
        feasible_max=planner.feasible_max,
        infeasible_min=planner.infeasible_min,
        x_hi=planner._hi,
        precision=Environment.SEARCH_PLANNER.SLA_PRECISION_DEFAULT,
    )


def _replicate_budget(planner: SmoothIsotonicSLAPlanner) -> int:
    override = int(planner._cfg.sla_replicates)
    if override > 0:
        return min(20, override)
    if not planner.binding_constraint:
        return 0
    if not planner._raw_probes:
        return 0
    last_x = max(planner._raw_probes.keys())
    samples = [
        m[planner.binding_constraint]
        for m in planner._raw_probes[last_x]
        if planner.binding_constraint in m
    ]
    if len(samples) < 2:
        return 0
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / (len(samples) - 1)
    sigma = math.sqrt(var)
    threshold = float(
        planner._sla_filters[
            planner._filter_keys.index(planner.binding_constraint)
        ].threshold
    )
    return replicate_count(sigma, threshold, override=0)


def _bracket_precision_reached(planner: SmoothIsotonicSLAPlanner) -> bool:
    if planner.feasible_max is None or planner.infeasible_min is None:
        return False
    gap = planner.infeasible_min - planner.feasible_max
    if gap <= 1:
        return True
    return (
        gap / max(planner.infeasible_min, 1)
        < Environment.SEARCH_PLANNER.SLA_PRECISION_DEFAULT
    )


def _finalize_with_reason(planner: SmoothIsotonicSLAPlanner, reason: str) -> None:
    if planner._convergence_reason is None:
        planner._convergence_reason = reason
    if planner.boundary_type is None:
        planner.boundary_type = "smooth"
