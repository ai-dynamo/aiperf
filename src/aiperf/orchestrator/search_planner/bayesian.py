# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Skopt-backed Bayesian-Optimization outer-loop planner.

Treats `BenchmarkConfig` mutation as: model_dump → dict → _set_nested_value
→ model_validate. This sidesteps the complication that BenchmarkConfig has
deeply-nested Pydantic submodels and `_set_nested_value` only operates on
dicts. Round-trip is safe: BenchmarkConfig is the v2 validated form which
is round-trip stable by construction.

Skopt is unmaintained-ish (last release 2024); the SearchPlanner abstract
seam means swapping to optuna later is a single-file change with no API
break.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from aiperf.config.adaptive_search import (
    AdaptiveSearchConfig,
    SearchSpaceDimension,
    SLAFilter,
)
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import SweepVariation, _set_nested_value
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.search_planner.base import (
    SearchIteration,
    SearchPlanner,
)

if TYPE_CHECKING:
    from aiperf.orchestrator.models import RunResult

logger = logging.getLogger(__name__)

__all__ = ["BayesianSearchPlanner"]


class BayesianSearchPlanner(SearchPlanner):
    """skopt.Optimizer-backed adaptive outer-loop planner."""

    def __init__(self, base_config: BenchmarkConfig, cfg: AdaptiveSearchConfig) -> None:
        try:
            from skopt import Optimizer
            from skopt.space import Integer, Real
        except ImportError as e:
            raise ImportError(
                "Bayesian Optimization requires the 'bo' extra: "
                "`uv pip install -e '.[bo]'` (or add scikit-optimize to your env). "
                f"Underlying import error: {e}"
            ) from e

        self._base = base_config
        self._cfg = cfg
        self._iter = 0
        self._history: list[SearchIteration] = []
        # Track ask/tell pairs so skopt's tell sees the same X it returned.
        self._pending_x: list[Any] | None = None
        # Patience-based stop: track best loss in skopt's loss space and the
        # number of consecutive iterations since it last improved. Matches
        # skopt's HollowIterationsStopper and Hyperopt's no_progress_loss.
        self._best_loss: float | None = None
        self._iters_since_improvement: int = 0
        # Which signal caused the most recent True from is_converged(); read
        # by SearchPlanner.convergence_reason(). Set to None until is_converged
        # actually fires.
        self._convergence_reason: str | None = None
        # Largest |loss| seen so far. Used to scale the SLA-violation soft
        # penalty so it dominates typical objective values without poisoning
        # GP variance with infinities. Bootstrapped at 1.0 so the penalty has
        # a finite floor before any iteration completes.
        self._max_seen_loss: float = 1.0
        # Constraint metrics for which we've already logged the
        # "unmeasurable, treating as infeasible" warning. Throttles to one
        # message per (planner instance, metric tag) pair.
        self._warned_unmeasurable_metrics: set[str] = set()

        dims = []
        for d in cfg.search_space:
            if d.kind == "int":
                dims.append(Integer(int(d.lo), int(d.hi)))
            else:
                dims.append(Real(d.lo, d.hi))
        self._opt = Optimizer(
            dimensions=dims,
            n_initial_points=cfg.n_initial_points,
            random_state=cfg.random_seed,
        )

    def ask(self) -> tuple[BenchmarkConfig, SweepVariation] | None:
        if self._iter >= self._cfg.max_iterations:
            return None
        if self.is_converged():
            return None

        x = self._opt.ask()
        self._pending_x = x  # remember for tell() to call skopt.tell()
        values: dict[str, Any] = {}
        for dim, suggestion in zip(self._cfg.search_space, x, strict=True):
            values[dim.path] = _coerce_for_kind(suggestion, dim)

        cfg_dict = self._base.model_dump(mode="json", exclude_none=True)
        for path, val in values.items():
            _set_nested_value(cfg_dict, path, val)
        cfg = BenchmarkConfig.model_validate(cfg_dict)

        variation = SweepVariation(
            index=self._iter,
            label=f"search_iter_{self._iter:04d}",
            values=values,
        )
        return cfg, variation

    def tell(self, variation: SweepVariation, results: list[RunResult]) -> None:
        per_trial_objectives = self._extract_trial_objectives(results)

        if self._pending_x is None:
            raise RuntimeError("tell() called without matching ask()")

        feasible = self._iteration_feasibility(results)
        # Compute the soft-penalty term ONCE per iteration from the averaged
        # constraint-metric values; tell skopt the penalty-augmented loss but
        # record the raw objective in SearchIteration for honest reporting.
        penalty, has_unmeasurable = self._compute_constraint_penalty(results)

        if not per_trial_objectives:
            objective_for_history = self._tell_failed_iteration(variation, penalty)
        else:
            objective_for_history = self._tell_successful_iteration(
                per_trial_objectives, penalty
            )

        self._pending_x = None
        self._update_improvement_tracking(objective_for_history)

        # Iteration is infeasible whenever any constraint metric is
        # unmeasurable (treated as infeasible per the spec) or the per-trial
        # feasibility check returned False.
        iteration_feasible = feasible and not has_unmeasurable

        self._history.append(
            SearchIteration(
                iteration_idx=self._iter,
                variation_values=dict(variation.values),
                objective_value=objective_for_history,
                results=list(results),
                feasible=iteration_feasible,
            )
        )
        self._iter += 1

    def _tell_failed_iteration(
        self, variation: SweepVariation, penalty: float
    ) -> float | None:
        """Tell skopt a synthetic worse-than-worst loss + penalty for failures.

        Iteration produced no usable objective (every trial failed, or the
        configured metric/stat was missing). Skopt cannot accept None and the
        optimizer must remain consistent with the ask/tell pairing, so we
        synthesize a deliberately-bad loss in skopt's space.
        """
        tell_value = self._failed_iteration_loss() + penalty
        logger.warning(
            "Search iteration %d at %s produced no usable objective; "
            "telling skopt fallback loss=%s (penalty=%s) and continuing.",
            self._iter,
            variation.values,
            tell_value,
            penalty,
        )
        self._opt.tell(self._pending_x, float(tell_value))
        return None

    def _tell_successful_iteration(
        self, per_trial_objectives: list[float], penalty: float
    ) -> float:
        """Tell skopt per-trial losses + penalty; return arithmetic mean for history.

        Per-trial observations let skopt's GP estimate the noise term
        (sigma_n^2) properly via repeated observations at the same x — see
        Letham et al. 2017, "Constrained Bayesian Optimization with Noisy
        Experiments" (arXiv:1706.07094). Pre-averaging discards the
        within-point variance the GP could have used. History stores the
        arithmetic mean for plateau detection and user-facing summaries.
        """
        losses = [self._objective_to_loss(o) + penalty for o in per_trial_objectives]
        if len(losses) == 1:
            self._opt.tell(self._pending_x, float(losses[0]))
        else:
            xs = [list(self._pending_x) for _ in losses]
            self._opt.tell(xs, [float(loss) for loss in losses])
        # Bookkeeping for the penalty scale: track |loss| of the largest *raw*
        # (unpenalized) objective seen so a future penalty stays dominant over
        # typical successes without exploding to infinity.
        for raw_obj in per_trial_objectives:
            self._max_seen_loss = max(
                self._max_seen_loss, abs(self._objective_to_loss(raw_obj))
            )
        return sum(per_trial_objectives) / len(per_trial_objectives)

    def _update_improvement_tracking(self, objective_for_history: float | None) -> None:
        """Update the patience-based stop counter from this iteration's objective.

        Mirrors skopt's HollowIterationsStopper and Hyperopt's no_progress_loss:
        consecutive iterations without improvement-over-best trigger
        convergence. Computed in skopt's loss space so the comparison is
        direction-agnostic. A failed iteration (objective_for_history=None) is
        treated as no-improvement.
        """
        if objective_for_history is None:
            self._iters_since_improvement += 1
            return
        iter_loss = self._objective_to_loss(objective_for_history)
        if self._best_loss is None or iter_loss < self._best_loss:
            self._best_loss = iter_loss
            self._iters_since_improvement = 0
        else:
            self._iters_since_improvement += 1

    def is_converged(self) -> bool:
        if self._iter >= self._cfg.max_iterations:
            self._convergence_reason = "max_iterations"
            return True
        # Improvement-over-best patience stop. If no successful iteration has
        # ever improved on the running best for `improvement_patience`
        # consecutive iterations, declare converged. Idiom from skopt's
        # HollowIterationsStopper and Hyperopt's no_progress_loss; treats
        # "we've stopped finding better points" as a stronger termination
        # signal than "values stopped fluctuating" alone.
        if self._iters_since_improvement >= self._cfg.improvement_patience:
            self._convergence_reason = "improvement_patience"
            return True
        window = self._cfg.plateau_window
        if len(self._history) < window:
            return False
        recent_objs = [
            h.objective_value
            for h in self._history[-window:]
            if h.objective_value is not None
        ]
        if len(recent_objs) < window:
            return False
        # Plateau test: coefficient of variation (sample stddev / |mean|).
        # Sample variance uses Bessel's correction (n-1) for unbiasedness with
        # small windows; population variance (/n) underestimates by a factor of
        # (n-1)/n which trips the threshold ~12% prematurely at the n=5 default.
        n = len(recent_objs)
        mean = sum(recent_objs) / n
        # When |mean| is essentially zero we cannot form a coefficient of
        # variation: it has no scale. Refuse to declare convergence in that
        # regime — the user's threshold is a *relative* coefficient and applying
        # it as an absolute compares unlike units. Wait for non-zero mean.
        if abs(mean) < _PLATEAU_MEAN_EPSILON:
            return False
        sample_variance = sum((v - mean) ** 2 for v in recent_objs) / (n - 1)
        cv = math.sqrt(sample_variance) / abs(mean)
        if cv < self._cfg.plateau_threshold:
            self._convergence_reason = "plateau_cv"
            return True
        return False

    def convergence_reason(self) -> str | None:
        """The signal that caused the most recent True from is_converged().

        One of ``"max_iterations"``, ``"improvement_patience"``,
        ``"plateau_cv"``, or ``None`` if is_converged has never returned True.
        Stable across calls until the next is_converged() check.
        """
        return self._convergence_reason

    def history(self) -> list[SearchIteration]:
        return list(self._history)

    def _extract_trial_objectives(self, results: list[RunResult]) -> list[float]:
        """Return per-trial objective values (one float per successful trial).

        Pre-research-fix this returned the arithmetic mean as a single float;
        now returns the full list so skopt's GP can fit the noise term
        (sigma_n^2) properly via repeated observations at the same x. See
        Letham et al. 2017, "Constrained Bayesian Optimization with Noisy
        Experiments" (arXiv:1706.07094). Pre-averaging discards the
        within-point variance the GP could have used to estimate noise.

        summary_metrics keys are bare metric tags; the stat (avg/p99/...)
        is a JsonMetricResult field — NOT a suffix on the key.

        Math note on the chosen stat: per-trial percentiles are the *expected
        per-trial percentile*, not the percentile of pooled samples — these
        differ for skewed distributions. BO optimizes whichever quantity is
        fed in, so consistency across iterations is what matters; the framing
        (per-trial vs. pooled) is a measurement-philosophy choice the user
        makes via --search-stat.
        """
        successful = [
            r
            for r in results
            if r.success and self._cfg.objective_metric in r.summary_metrics
        ]
        values: list[float] = []
        for r in successful:
            mr = r.summary_metrics[self._cfg.objective_metric]
            stat_value = getattr(mr, self._cfg.objective_stat, None)
            if stat_value is not None:
                values.append(float(stat_value))
        return values

    def _objective_to_loss(self, objective: float) -> float:
        """Map objective-space value to skopt's loss-space (which it minimizes).

        Skopt minimizes; for MAXIMIZE we negate, for MINIMIZE we pass through.
        Single sign-flip site avoids the inconsistency where success and
        failure paths apply sign in different places.
        """
        if self._cfg.objective_direction == OptimizationDirection.MAXIMIZE:
            return -objective
        return objective

    def _failed_iteration_loss(self) -> float:
        """Loss (skopt-space) to tell skopt when an iteration has no usable objective.

        Strategy:

        - If we have prior successful objectives, return the worst-seen loss
          plus a 10%-or-1.0-absolute margin. This keeps the GP kernel matrix
          well-posed (no inf/nan) while telling skopt this point is unambiguously
          worse than anywhere it has actually seen succeed.
        - With no prior data, fall back to a large finite sentinel
          (``_NO_DATA_SENTINEL_LOSS``). BO is essentially random until the first
          successful iteration; the warning logged at the call site flags this.
        """
        prior_losses = [
            self._objective_to_loss(h.objective_value)
            for h in self._history
            if h.objective_value is not None
        ]
        if not prior_losses:
            return _NO_DATA_SENTINEL_LOSS
        worst_loss = max(prior_losses)
        margin = max(abs(worst_loss) * 0.1, 1.0)
        return worst_loss + margin

    def _trial_satisfies(self, run: RunResult, sla: SLAFilter) -> bool:
        """Return True iff ``run`` satisfies the single SLA filter ``sla``.

        Missing metric/stat is treated as infeasible (safer than silently
        passing); the caller logs each unmeasurable tag once. Boundary:
        strict ops (``lt``/``gt``) call ``value == threshold`` infeasible
        while the soft penalty is 0 there — feasibility flag drives best
        selection, GP penalty drives exploration; the two are intentionally
        separable for the rare exact-on-the-line case.
        """
        metric = run.summary_metrics.get(sla.metric_tag)
        if metric is None:
            return False
        value = getattr(metric, sla.stat, None)
        if value is None:
            return False
        if sla.op == "lt":
            return value < sla.threshold
        if sla.op == "le":
            return value <= sla.threshold
        if sla.op == "gt":
            return value > sla.threshold
        return value >= sla.threshold

    def _iteration_feasibility(self, results: list[RunResult]) -> bool:
        """True iff at least one trial in this iteration satisfied every SLA filter.

        Mirrors per-trial averaging — if any trial passed all filters, the
        configuration is reproducibly feasible (we report feasible-best on this
        same averaging). When ``self._cfg.sla_filters`` is empty the iteration
        is unconditionally feasible.
        """
        if not self._cfg.sla_filters:
            return True
        for run in results:
            if not run.success:
                continue
            if all(self._trial_satisfies(run, f) for f in self._cfg.sla_filters):
                return True
        return False

    def _averaged_metric_value(
        self, results: list[RunResult], metric_tag: str, stat: str
    ) -> float | None:
        """Return the mean of stat(metric_tag) across successful trials, or None.

        None means no successful trial had a measurable value for the
        (metric_tag, stat) pair. The caller treats None as unmeasurable and
        applies a fixed-magnitude penalty to steer the GP away from this region.
        """
        values: list[float] = []
        for run in results:
            if not run.success:
                continue
            metric = run.summary_metrics.get(metric_tag)
            if metric is None:
                continue
            value = getattr(metric, stat, None)
            if value is None:
                continue
            values.append(float(value))
        if not values:
            return None
        return sum(values) / len(values)

    def _compute_constraint_penalty(
        self, results: list[RunResult]
    ) -> tuple[float, bool]:
        """Soft-penalty term to add to skopt's loss for SLA violations.

        Per-iteration, computed from averaged constraint-metric values across
        successful trials. For each filter, the contribution is
        ``W * (max(0, signed_violation) / |threshold|)``; an unmeasurable
        constraint contributes a fixed ``W`` (treated as a 1.0× normalized
        violation) and is logged once per metric tag.

        ``W = _PENALTY_WEIGHT_MULTIPLIER * max(self._max_seen_loss, 1.0)`` keeps
        the penalty dominant over typical objective values without poisoning
        the GP kernel with infinities.

        Returns ``(penalty, has_unmeasurable)`` where ``has_unmeasurable`` is
        True if at least one filter could not be measured — this flag forces
        the iteration's ``feasible`` flag to False even if other filters
        coincidentally passed.
        """
        if not self._cfg.sla_filters:
            return 0.0, False

        weight = _PENALTY_WEIGHT_MULTIPLIER * max(self._max_seen_loss, 1.0)
        penalty = 0.0
        has_unmeasurable = False
        for sla in self._cfg.sla_filters:
            value = self._averaged_metric_value(results, sla.metric_tag, sla.stat)
            if value is None:
                has_unmeasurable = True
                penalty += weight
                if sla.metric_tag not in self._warned_unmeasurable_metrics:
                    self._warned_unmeasurable_metrics.add(sla.metric_tag)
                    logger.warning(
                        "SLA filter on metric %r (stat=%s) is unmeasurable on "
                        "iteration %d; treating as infeasible and applying a "
                        "fixed-magnitude penalty (=%s) to the BO loss. Likely "
                        "cause: a streaming-only metric on a non-streaming "
                        "endpoint, or a typo in --search-stat. Subsequent "
                        "iterations with the same tag will not re-log.",
                        sla.metric_tag,
                        sla.stat,
                        self._iter,
                        weight,
                    )
                continue
            violation = max(0.0, _signed_violation(value, sla))
            denom = abs(sla.threshold) if sla.threshold != 0 else 1.0
            penalty += weight * (violation / denom)
        return penalty, has_unmeasurable


# Plateau-detection numerical guard. Coefficient of variation has no meaning
# when |mean| collapses to zero; this floor lets us refuse to claim convergence
# in that regime without exposing yet another knob to users.
_PLATEAU_MEAN_EPSILON = 1e-9

# Sentinel loss told to skopt for an iteration that produced no usable objective
# AND for which we have no successful prior to scale against. Large enough to be
# unambiguously worse than any plausible real metric, finite enough not to
# poison the GP's kernel matrix. Not user-tunable: the no-data branch is a
# degraded mode and BO is essentially random until the first success.
_NO_DATA_SENTINEL_LOSS = 1.0e6

# Soft-penalty multiplier on max(|loss|) for SLA-filter violations. Finite
# (not ±inf / 1e18) to avoid GP variance distortion; tune up if BO ignores
# soft constraints, down if it dominates exploration too early.
_PENALTY_WEIGHT_MULTIPLIER: float = 100.0


def _coerce_for_kind(value: Any, dim: SearchSpaceDimension) -> Any:
    """Skopt returns numpy scalars; coerce to plain Python int/float."""
    if dim.kind == "int":
        return int(value)
    return float(value)


def _signed_violation(value: float, sla: SLAFilter) -> float:
    """Signed magnitude of how much ``value`` violates ``sla``.

    Positive = violation, negative = slack. Caller clamps to ``max(0, .)`` so
    only violations contribute to the penalty. Sign convention: ``lt``/``le``
    use ``value - threshold`` (positive when over the cap); ``gt``/``ge`` use
    ``threshold - value`` (positive when under the floor). Defined at module
    scope because it reads no planner state.
    """
    if sla.op in ("lt", "le"):
        return value - sla.threshold
    return sla.threshold - value
