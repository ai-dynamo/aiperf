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

from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension
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

        if not per_trial_objectives:
            # Iteration produced no usable objective (every trial failed, or the
            # configured metric/stat was missing). Skopt cannot accept None and
            # the optimizer must remain consistent with the ask/tell pairing,
            # so we synthesize a deliberately-bad loss in skopt's space.
            tell_value = self._failed_iteration_loss()
            logger.warning(
                "Search iteration %d at %s produced no usable objective; "
                "telling skopt fallback loss=%s and continuing.",
                self._iter,
                variation.values,
                tell_value,
            )
            self._opt.tell(self._pending_x, float(tell_value))
            objective_for_history: float | None = None
        else:
            # Per-trial observations let skopt's GP estimate the noise term
            # (sigma_n^2) properly. Pre-averaging the trials before telling
            # discards the within-point variance the GP could have used —
            # see Letham et al. 2017, "Constrained Bayesian Optimization with
            # Noisy Experiments" (arXiv:1706.07094). When we have N>=2 trials
            # we feed N copies of the same x with the N losses; skopt's
            # Optimizer.tell accepts repeated x's and the GP fits accordingly.
            losses = [self._objective_to_loss(o) for o in per_trial_objectives]
            if len(losses) == 1:
                self._opt.tell(self._pending_x, float(losses[0]))
            else:
                xs = [list(self._pending_x) for _ in losses]
                self._opt.tell(xs, [float(loss) for loss in losses])
            # History stores the arithmetic mean for plateau detection and
            # user-facing summaries; the GP itself sees per-trial values.
            objective_for_history = sum(per_trial_objectives) / len(
                per_trial_objectives
            )

        self._pending_x = None

        # Track improvement-over-best for the patience-based stop. Mirrors
        # skopt's HollowIterationsStopper and Hyperopt's no_progress_loss:
        # consecutive iterations without improvement-over-best trigger
        # convergence. Computed in skopt's loss space so the comparison is
        # direction-agnostic.
        if objective_for_history is not None:
            iter_loss = self._objective_to_loss(objective_for_history)
            if self._best_loss is None or iter_loss < self._best_loss:
                self._best_loss = iter_loss
                self._iters_since_improvement = 0
            else:
                self._iters_since_improvement += 1
        else:
            self._iters_since_improvement += 1

        self._history.append(
            SearchIteration(
                iteration_idx=self._iter,
                variation_values=dict(variation.values),
                objective_value=objective_for_history,
                results=list(results),
            )
        )
        self._iter += 1

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


def _coerce_for_kind(value: Any, dim: SearchSpaceDimension) -> Any:
    """Skopt returns numpy scalars; coerce to plain Python int/float."""
    if dim.kind == "int":
        return int(value)
    return float(value)
