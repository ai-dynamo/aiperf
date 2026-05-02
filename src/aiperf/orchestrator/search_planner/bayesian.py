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
        objective = self._extract_objective(results)
        if objective is None:
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
        else:
            tell_value = self._objective_to_loss(objective)

        if self._pending_x is None:
            raise RuntimeError("tell() called without matching ask()")
        self._opt.tell(self._pending_x, float(tell_value))
        self._pending_x = None

        self._history.append(
            SearchIteration(
                iteration_idx=self._iter,
                variation_values=dict(variation.values),
                objective_value=objective,
                results=list(results),
            )
        )
        self._iter += 1

    def is_converged(self) -> bool:
        if self._iter >= self._cfg.max_iterations:
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
        return cv < self._cfg.plateau_threshold

    def history(self) -> list[SearchIteration]:
        return list(self._history)

    def _extract_objective(self, results: list[RunResult]) -> float | None:
        """Average the configured stat across all successful trials.

        Returns the arithmetic mean of `JsonMetricResult.<stat>` over successful
        trials, or None when no trial produced the metric. summary_metrics keys
        are bare metric tags; the stat (avg/p99/...) is a JsonMetricResult field
        — NOT a suffix on the key.

        Math note: arithmetic mean of per-trial means is the unbiased estimator
        of the true mean iff trials carry equal weight (the usual case under
        --num-profile-runs N with consistent --request-count). For percentile
        stats (p50/p99/...) the result is the *expected per-trial percentile*,
        not the percentile of the pooled samples — these differ for skewed
        distributions. BO optimizes whichever quantity is fed in, so the
        percentile-of-trial-percentiles framing is consistent across iterations.
        """
        successful = [
            r
            for r in results
            if r.success and self._cfg.objective_metric in r.summary_metrics
        ]
        if not successful:
            return None
        values: list[float] = []
        for r in successful:
            mr = r.summary_metrics[self._cfg.objective_metric]
            stat_value = getattr(mr, self._cfg.objective_stat, None)
            if stat_value is not None:
                values.append(float(stat_value))
        if not values:
            return None
        return sum(values) / len(values)

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
