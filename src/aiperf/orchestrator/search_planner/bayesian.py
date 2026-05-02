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
            # Skopt cannot accept None; tell it the worst value seen so far,
            # or a baseline if none seen, so the iteration counter still advances.
            tell_value = self._fallback_tell_value()
            logger.warning(
                "BO iteration %d at %s produced no successful runs; "
                "telling skopt fallback %s and continuing.",
                self._iter,
                variation.values,
                tell_value,
            )
        else:
            sign = (
                -1.0
                if self._cfg.objective_direction == OptimizationDirection.MAXIMIZE
                else 1.0
            )
            tell_value = sign * objective

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
        mean = sum(recent_objs) / len(recent_objs)
        if mean == 0:
            return all(abs(v) < self._cfg.plateau_threshold for v in recent_objs)
        # Coefficient of variation: scale-free relative spread.
        variance = sum((v - mean) ** 2 for v in recent_objs) / len(recent_objs)
        cv = math.sqrt(variance) / abs(mean)
        return cv < self._cfg.plateau_threshold

    def history(self) -> list[SearchIteration]:
        return list(self._history)

    def _extract_objective(self, results: list[RunResult]) -> float | None:
        """Average the configured stat across all successful trials.

        summary_metrics keys are bare metric tags; the stat (avg/p99/...)
        is a field on JsonMetricResult — NOT a suffix on the key. This is
        the gotcha called out in the design doc.
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

    def _fallback_tell_value(self) -> float:
        """Worst-seen value for the running optimizer, or 0 if none seen."""
        seen = [
            h.objective_value for h in self._history if h.objective_value is not None
        ]
        if not seen:
            return 0.0
        if self._cfg.objective_direction == OptimizationDirection.MAXIMIZE:
            return -min(seen)  # worst (smallest) maximize value, sign-flipped
        return max(seen)


def _coerce_for_kind(value: Any, dim: SearchSpaceDimension) -> Any:
    """Skopt returns numpy scalars; coerce to plain Python int/float."""
    if dim.kind == "int":
        return int(value)
    return float(value)
