# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared per-trial / per-iteration SLA-feasibility helpers.

Used by both ``bayesian.py`` and ``monotonic.py`` so both planners share the
same canonical interpretation of ``SLAFilter``. Pure functions on plain inputs
(no ``self``) so they're trivially testable and reusable from a third planner.

Boundary semantics: missing metric / missing stat is **infeasible** — silently
treating an unmeasurable filter as a pass would invert the planner's bracket.
The bayesian planner additionally feeds unmeasurable filters into its soft
penalty (separate concern); both planners agree that the boolean feasibility
flag stays False whenever any filter is unmeasurable.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Literal

from aiperf.common.finite import is_finite_value

if TYPE_CHECKING:
    from aiperf.config.sweep import AdaptiveSearchSweep
    from aiperf.config.sweep.adaptive import SLAFilter
    from aiperf.orchestrator.models import RunResult
    from aiperf.orchestrator.search_planner.base import SearchIteration

# Plateau-detection guard: coefficient of variation has no meaning when the
# mean magnitude collapses toward zero (the CV = std/|mean| ratio explodes).
# Floor the |mean| at this value before computing CV; below the floor we
# refuse to declare convergence on a "plateau" since the signal is too small
# to distinguish noise from genuine flatness.
PLATEAU_MEAN_EPSILON: float = 1e-9


logger = logging.getLogger(__name__)


__all__ = [
    "averaged_metric_value",
    "evaluate_three_signal_convergence",
    "first_failing_filter",
    "iteration_feasibility",
    "trial_satisfies",
]


_DIAGNOSIS_STAT_NAMES: tuple[str, ...] = (
    "avg",
    "p1",
    "p5",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
    "min",
    "max",
)


def _diagnose_no_successful_trials(all_results: list[RunResult] | None) -> str:
    """Branch (a): no successful trials. Enumerate failed-run errors when known."""
    failed = [r for r in (all_results or []) if not r.success]
    if not failed:
        return "no successful trials in this iteration"
    samples = [f"{r.label}: {r.error or 'no error message'}" for r in failed[:3]]
    sample = "; ".join(samples)
    if len(failed) > 3:
        sample += f"; ... (+{len(failed) - 3} more)"
    return f"no successful trials in this iteration; {len(failed)} failed: [{sample}]"


def _diagnose_missing_metric_tag(successful: list[RunResult], sla: SLAFilter) -> str:
    """Branch (b): metric tag absent from every successful run. List available tags."""
    sample_keys = sorted({tag for run in successful for tag in run.summary_metrics})
    keys_repr = ", ".join(sample_keys[:10])
    if len(sample_keys) > 10:
        keys_repr += f", ... (+{len(sample_keys) - 10} more)"
    return (
        f"metric_tag={sla.metric_tag!r} missing from every "
        f"successful run's summary_metrics; available tags: [{keys_repr}]"
    )


def _diagnose_missing_stat(successful: list[RunResult], sla: SLAFilter) -> str:
    """Branch (c): metric present but requested stat is None. List stats with values.

    Distinguishes "exporter dropped p95 from the export" (avg/p50/p99 present)
    from "user typoed p95 instead of a populated stat" — and names the boundary
    case "(no stats populated)" so a unit-only entry isn't mistaken for a
    missing tag.
    """
    present: set[str] = set()
    for run in successful:
        m = run.summary_metrics.get(sla.metric_tag)
        if m is None:
            continue
        for stat_name in _DIAGNOSIS_STAT_NAMES:
            if getattr(m, stat_name, None) is not None:
                present.add(stat_name)
    present_repr = ", ".join(sorted(present)) if present else "(no stats populated)"
    return (
        f"metric_tag={sla.metric_tag!r} present but stat={sla.stat!r} "
        f"is None on every successful run (likely a missing percentile in the "
        f"export); stats with values: [{present_repr}]"
    )


def _diagnose_unmeasurable(
    successful: list[RunResult],
    sla: SLAFilter,
    all_results: list[RunResult] | None = None,
) -> str:
    """Return a one-line diagnosis of *why* an SLA filter is unmeasurable.

    Distinguishes the three silent paths that previously all produced
    ``observed: null`` with no log: (a) no successful trials, (b) metric tag
    absent from every run's ``summary_metrics``, (c) metric present but the
    requested stat field is ``None`` on every run. The third case is the one
    that bit ``ajc-sweep-conc-may5`` on DGX 2026-05-06 — the planner read
    ``run.summary_metrics["time_to_first_token"].p95`` and got ``None`` even
    though the file on disk had a populated ``time_to_first_token`` entry,
    just without the ``p95`` percentile.

    ``all_results`` is the full per-iteration result list (failed + succeeded);
    when provided, branch (a) names the failed children's error strings so
    oncall sees *why* the iteration produced no successful trial without
    needing a second log query. Branch (c) lists the stats that DO have values
    on the present metric, so oncall can tell "p95 was never computed by the
    exporter" from "the user typoed p95 instead of p99".
    """
    if not successful:
        return _diagnose_no_successful_trials(all_results)
    metric_seen = any(
        run.summary_metrics.get(sla.metric_tag) is not None for run in successful
    )
    if not metric_seen:
        return _diagnose_missing_metric_tag(successful, sla)
    stat_seen = any(
        getattr(run.summary_metrics.get(sla.metric_tag), sla.stat, None) is not None
        for run in successful
    )
    if not stat_seen:
        return _diagnose_missing_stat(successful, sla)
    # Should not happen — caller has already established the filter is
    # unmeasurable, so at least one of the three branches must apply.
    return "unmeasurable for unknown reason"


def trial_satisfies(run: RunResult, sla: SLAFilter) -> bool:
    """Return True iff ``run`` satisfies the single SLA filter ``sla``.

    Missing metric/stat is treated as infeasible — the planner has no signal
    to rank against, so silently passing would invert the verdict. Boundary:
    strict ops (``lt``/``gt``) call ``value == threshold`` infeasible.

    A non-finite (NaN/+inf/-inf) measurement is treated identically to
    missing: NaN comparisons short-circuit to False which would silently flip
    feasibility, so we route it through the same "infeasible" branch.
    """
    metric = run.summary_metrics.get(sla.metric_tag)
    if metric is None:
        return False
    value = getattr(metric, sla.stat, None)
    if not is_finite_value(value):
        return False
    if sla.op == "lt":
        return value < sla.threshold
    if sla.op == "le":
        return value <= sla.threshold
    if sla.op == "gt":
        return value > sla.threshold
    return value >= sla.threshold


def iteration_feasibility(
    results: list[RunResult], sla_filters: list[SLAFilter]
) -> bool:
    """True iff at least one successful trial satisfies every SLA filter.

    Empty ``sla_filters`` is unconditionally feasible (BO can use the planner
    in objective-only mode). Empty ``results`` / all-failed runs are
    infeasible: a configuration that produced no measurable trial gives the
    planner no evidence the SLA holds.
    """
    if not sla_filters:
        return True
    for run in results:
        if not run.success:
            continue
        if all(trial_satisfies(run, f) for f in sla_filters):
            return True
    return False


def first_failing_filter(
    results: list[RunResult], sla_filters: list[SLAFilter]
) -> dict[str, Any] | None:
    """Return the first SLA filter (input order) breached at this iteration.

    Mirrors ``iteration_feasibility`` semantics — an iteration is feasible
    iff at least one successful trial satisfies every filter, so the
    "first-breach" filter is the first one with no satisfying trial. Each
    filter is reported with its echoed ``metric_tag``/``stat``/``op``/
    ``threshold`` plus an ``observed`` value averaged over successful runs;
    ``observed`` is null when no trial measured the metric (or when every
    trial failed entirely).

    The ``observed`` value is the arithmetic mean of ``stat(metric_tag)``
    across successful trials in the iteration. **This is a display-only
    summary**, not the feasibility-verdict input — the verdict itself uses
    ``iteration_feasibility``'s ANY-trial-passes semantics. Averaging
    percentiles across trials is mathematically loose; treat ``observed`` as
    a UI hint, not a ranking signal.
    """
    if not sla_filters:
        return None
    successful = [r for r in results if r.success]
    for sla in sla_filters:
        if any(trial_satisfies(run, sla) for run in successful):
            continue
        observed_values: list[float] = []
        for run in successful:
            metric = run.summary_metrics.get(sla.metric_tag)
            if metric is None:
                continue
            stat_value = getattr(metric, sla.stat, None)
            # Why: NaN/inf bypass `is None` checks but corrupt the mean. Treat
            # them as missing so `observed` reports null instead of leaking a
            # non-finite display value into the breach record.
            if not is_finite_value(stat_value):
                continue
            observed_values.append(float(stat_value))
        observed: float | None = (
            sum(observed_values) / len(observed_values) if observed_values else None
        )
        if observed is None:
            # Distinguish the three silent paths that all produce
            # ``observed: null`` so production diagnoses don't have to dig.
            # Pre-fix this was the *only* signal that the SLA bracket had
            # collapsed because the data never arrived — the planner just
            # latched ``infeasible_min`` and terminated with
            # ``no_pass_in_range``.
            logger.warning(
                "SLA filter %s.%s %s %s: %s",
                sla.metric_tag,
                sla.stat,
                sla.op,
                sla.threshold,
                _diagnose_unmeasurable(successful, sla, all_results=results),
            )
        return {
            "metric_tag": sla.metric_tag,
            "stat": sla.stat,
            "op": sla.op,
            "threshold": sla.threshold,
            "observed": observed,
        }
    return None


def averaged_metric_value(
    results: list[RunResult],
    metric_tag: str,
    stat: Literal["avg", "p50", "p90", "p95", "p99"],
) -> float | None:
    """Mean of stat(metric_tag) across successful trials, or None.

    None means no successful trial had a measurable value for the
    (metric_tag, stat) pair. Used by :class:`OptunaSearchPlanner` (and
    its :class:`BayesianSearchPlanner` curated-preset subclass) on the
    constraints-func path: caller writes the value (or None) onto
    ``trial.user_attrs`` so ``constraints_func`` can read it at
    ``study.tell()`` time. Also shared with the 1D-saturation
    :class:`MonotonicSLASearchPlanner` and
    :class:`SmoothIsotonicSLAPlanner` so all planners use byte-identical
    averaging semantics.
    """
    values: list[float] = []
    for run in results:
        if not run.success:
            continue
        metric = run.summary_metrics.get(metric_tag)
        if metric is None:
            continue
        value = getattr(metric, stat, None)
        # Why: filter NaN/+inf/-inf alongside None. NaN bypasses `is None`
        # guards in both planners (`max(0, NaN)==0`, `NaN > 0` is False), so
        # without this filter the soft-penalty silently zeroes and Optuna's
        # constraint-func reads the iteration as feasible while
        # `iteration_feasibility` records it as infeasible — a silent
        # split-brain between the GP model and the history.
        if not is_finite_value(value):
            continue
        values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def evaluate_three_signal_convergence(
    *,
    iter_count: int,
    history: list[SearchIteration],
    iters_since_improvement: int,
    cfg: AdaptiveSearchSweep,
) -> tuple[bool, str | None]:
    """Evaluate the three-signal convergence used by BO-style planners.

    Returns ``(is_converged, reason)``. ``reason`` is one of:

    - ``"max_iterations"`` — ``iter_count >= cfg.max_iterations``
    - ``"improvement_patience"`` — ``iters_since_improvement >= cfg.improvement_patience``
    - ``"plateau_cv"`` — sample CV of last ``cfg.plateau_window`` objectives < ``cfg.plateau_threshold``
    - ``None`` — not converged

    Reason strings are stable contract: ``search_history.json``'s
    ``convergence_reason`` field consumes them. Do not change.

    Math note: plateau test uses sample variance (Bessel's correction) so the
    ``n=5`` default doesn't trip ~12% prematurely vs. population variance.
    When ``|mean| < PLATEAU_MEAN_EPSILON`` we refuse to declare plateau —
    the user's threshold is a *relative* coefficient and applying it as an
    absolute compares unlike units.
    """
    if iter_count >= cfg.max_iterations:
        return True, "max_iterations"
    if iters_since_improvement >= cfg.improvement_patience:
        return True, "improvement_patience"
    window = cfg.plateau_window
    if len(history) < window:
        return False, None
    recent_objs = [
        h.objective_value for h in history[-window:] if h.objective_value is not None
    ]
    if len(recent_objs) < window:
        return False, None
    n = len(recent_objs)
    mean = sum(recent_objs) / n
    if abs(mean) < PLATEAU_MEAN_EPSILON:
        return False, None
    sample_variance = sum((v - mean) ** 2 for v in recent_objs) / (n - 1)
    cv = math.sqrt(sample_variance) / abs(mean)
    if cv < cfg.plateau_threshold:
        return True, "plateau_cv"
    return False, None
