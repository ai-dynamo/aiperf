# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""1D-optimized SLA-saturation search planner (exponential probe + bisection).

Default planner for the ``max-concurrency-under-sla`` search recipe.
Mirrors perf_analyzer's ``--binary-search`` precedent: instead of fitting a
GP over a continuous objective surface, treat the swept dimension as a
monotonically degrading axis and binary-search the SLA-saturation boundary.
For 1D feasibility-only problems this is O(log N) where BO is O(N).

Algorithm
---------

1. **Exponential probe.** Starting at ``lo``, multiply by 2 each step until
   any SLA filter fails (or ``hi`` reached). Records ``lo'`` (last passing
   point) and ``hi'`` (first failing point).
2. **Bisection.** Bisect ``[lo', hi']`` until ``(hi' - lo') / hi' <
   precision`` (5%) or ``max_iterations`` exhausted.

Stability window
----------------

Each probed swept-value carries a per-trial verdict log; a verdict is
provisional until ``monotonic_stability_trials`` trials at that value
agree. ``plan.trials >= monotonic_stability_trials`` satisfies it
automatically; otherwise the planner re-asks the same swept value until
it has enough agreeing trials. Mirrors perf_analyzer's max/min-of-last-3
idiom.

Convergence reasons
-------------------

* ``monotonic_precision_reached`` — bisection narrowed the bracket to
  within 5%.
* ``monotonic_no_failure_in_range`` — every probed point passed; the SLA
  saturates above ``hi``.
* ``monotonic_no_pass_in_range`` — every probed point failed; the SLA
  saturates below ``lo``.
* ``max_iterations`` — bisection budget exhausted before precision was hit.

Example
-------

(Pseudo-code; not runnable — ``base_cfg`` and ``run_benchmark`` are
illustrative placeholders, not defined symbols.)

    >>> from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter
    >>> from aiperf.config.sweep import AdaptiveSearchSweep, Objective
    >>> from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
    >>> cfg = AdaptiveSearchSweep(
    ...     search_space=[SearchSpaceDimension(
    ...         path="phases.profiling.concurrency", lo=1, hi=1000, kind="int"
    ...     )],
    ...     objectives=[Objective(
    ...         metric="output_token_throughput",
    ...         direction=OptimizationDirection.MAXIMIZE,
    ...     )],
    ...     max_iterations=30,
    ...     n_initial_points=2,
    ...     sla_filters=[SLAFilter(
    ...         metric_tag="time_to_first_token", stat="p95",
    ...         op="lt", threshold=200.0,
    ...     )],
    ... )
    >>> planner = MonotonicSLASearchPlanner(base_cfg, cfg)
    >>> while not planner.is_converged():
    ...     proposal = planner.ask()
    ...     if proposal is None:
    ...         break
    ...     _, variation = proposal
    ...     planner.tell(variation, run_benchmark(variation))
    >>> planner.feasible_max, planner.infeasible_min
    (255, 256)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from aiperf.common.environment import Environment
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, SweepVariation, _set_nested_value
from aiperf.orchestrator.search_planner._monotonic_boundary import (
    compute_boundary_summary,
)
from aiperf.orchestrator.search_planner._sla_helpers import (
    iteration_feasibility,
)
from aiperf.orchestrator.search_planner.base import (
    SearchIteration,
    SearchPlanner,
)

if TYPE_CHECKING:
    from aiperf.orchestrator.models import RunResult

logger = logging.getLogger(__name__)

__all__ = ["MonotonicSLASearchPlanner"]


# Default relative-precision target for bisection: stop when
# ``(hi' - lo') / hi' < precision``. The 5% default mirrors perf_analyzer's
# --binary-search default and is fine-grained enough to land on the right
# log-spaced concurrency bin in practice. Configured via
# ``Environment.SEARCH_PLANNER.SLA_PRECISION_DEFAULT``.


class _PointLog:
    """Per-swept-value running tally of pass/fail trials.

    Stability-window bookkeeping: when ``required_agreements > 1`` the
    planner re-asks the same swept value until ``required_agreements``
    consecutive trials at that value agree. Single-trial verdicts (the
    perf_analyzer default at ``trials=1``) accept immediately.
    """

    def __init__(self, required_agreements: int) -> None:
        self.required = required_agreements
        self.passes = 0
        self.fails = 0

    def record(self, feasible: bool) -> None:
        if feasible:
            self.passes += 1
        else:
            self.fails += 1

    def verdict(self) -> bool | None:
        """Return latched verdict or None if still provisional."""
        if self.passes >= self.required:
            return True
        if self.fails >= self.required:
            return False
        return None


class MonotonicSLASearchPlanner(SearchPlanner):
    """Exponential-probe + bisection 1D SLA-saturation planner.

    See module docstring for the algorithm and ``perf_analyzer
    --binary-search`` precedent.
    """

    def __init__(self, base_config: BenchmarkConfig, cfg: AdaptiveSearchSweep) -> None:
        if len(cfg.search_space) != 1:
            raise ValueError(
                "monotonic_sla planner requires exactly one search-space "
                f"dimension; got {len(cfg.search_space)}. For multi-dimensional "
                "search use the bayesian (BO) planner via "
                "`--search-planner bayesian` — it handles multi-dim search "
                "spaces by GP-modeling the joint surface."
            )
        dim = cfg.search_space[0]
        if dim.kind != "int":
            raise ValueError(
                f"monotonic_sla planner v1 supports kind='int' dimensions only; "
                f"got kind={dim.kind!r} on path {dim.path!r}. Use the bayesian "
                "planner for real-valued dimensions."
            )
        if not cfg.sla_filters:
            raise ValueError(
                "monotonic_sla planner requires at least one SLA filter "
                "(sla_filters is empty). The planner has no scoring without "
                "feasibility constraints; supply --search-sla / --ttft-sla-ms / "
                "etc. or use the bayesian planner if you only have an "
                "objective metric."
            )

        self._base = base_config
        self._cfg = cfg
        if len(cfg.objectives) > 1:
            raise ValueError(
                f"{type(self).__name__} is single-objective only; "
                f"received {len(cfg.objectives)} objectives. For multi-objective "
                "Pareto BO use --search-planner optuna --optuna-sampler botorch "
                "--optuna-acquisition qlognehvi."
            )
        self._dim = dim
        self._lo: int = int(dim.lo)
        self._hi: int = int(dim.hi)
        self._stability_trials = cfg.monotonic_stability_trials

        # Bracket. ``feasible_max`` (lo') is the highest swept value with a
        # latched feasible verdict; ``infeasible_min`` (hi') is the lowest
        # with a latched infeasible verdict. Both None until first verdict.
        self.feasible_max: int | None = None
        self.infeasible_min: int | None = None

        # Per-point trial logs for the stability window.
        self._point_logs: dict[int, _PointLog] = {}

        # Algorithm phase: "probe" (exponential ramp) or "bisect".
        self._phase: Literal["probe", "bisect"] = "probe"
        # Next swept value to ask for. Starts at lo; doubled in probe phase.
        self._next_value: int = self._lo
        # ``ask`` returned a value but ``tell`` hasn't been called yet.
        self._pending_value: int | None = None

        self._iter = 0
        self._history: list[SearchIteration] = []
        self._convergence_reason: str | None = None

        # Set when a non-monotonicity is detected during bisection (a
        # feasible verdict appears above an infeasible one or vice versa).
        self.non_monotonic_warning: bool = False
        # Iteration indices that observed the non-monotonic transition; used
        # to flag the SearchIteration entries for downstream artifacts.
        self._warned_iterations: set[int] = set()

    # ------------------------------------------------------------------
    # SearchPlanner ABC
    # ------------------------------------------------------------------

    def ask(self) -> tuple[BenchmarkConfig, SweepVariation] | None:
        """Return the next 1D SLA probe and latch it as pending.

        Mutates `_pending_value` so the following `tell()` must report results for
        this exact variation. Returns None once the monotonic planner has latched a
        convergence reason.
        """
        if self.is_converged():
            return None

        value = self._next_value
        self._pending_value = value
        cfg = self._mutate_base(value)
        variation = SweepVariation(
            index=self._iter,
            label=f"search_iter_{self._iter:04d}",
            values={self._dim.path: value},
        )
        return cfg, variation

    def tell(self, variation: SweepVariation, results: list[RunResult]) -> None:
        """Absorb results for the pending monotonic probe.

        Requires a preceding `ask()`; raises RuntimeError otherwise. Updates the
        per-point stability log, latches feasible/infeasible boundary verdicts,
        records a SearchIteration, and plans the next probe or convergence reason.
        """
        if self._pending_value is None:
            raise RuntimeError("tell() called without matching ask()")
        value = self._pending_value
        self._pending_value = None

        feasible = iteration_feasibility(results, self._cfg.sla_filters)
        objective_value = self._extract_objective(results)
        log = self._point_logs.setdefault(value, _PointLog(self._stability_trials))
        log.record(feasible)

        non_monotonic_this_iter = False
        verdict = log.verdict()
        if verdict is not None:
            non_monotonic_this_iter = self._absorb_verdict(value, verdict)

        # Decide what value to probe next based on the latest verdict and the
        # current phase. Provisional verdicts re-ask the same value.
        self._plan_next_step(value, verdict)

        if non_monotonic_this_iter:
            self._warned_iterations.add(self._iter)
        iteration = SearchIteration(
            iteration_idx=self._iter,
            variation_values=dict(variation.values),
            objective_value=objective_value,
            objective_values=(
                [objective_value] if objective_value is not None else None
            ),
            results=list(results),
            feasible=feasible,
            non_monotonic_warning=non_monotonic_this_iter,
        )
        self._history.append(iteration)
        self._iter += 1

    def is_converged(self) -> bool:
        """Return True once a boundary reason or max_iterations has stopped the search.

        Calling this may latch `_convergence_reason = "max_iterations"` when the
        iteration budget is exhausted.
        """
        if self._convergence_reason is not None:
            return True
        if self._iter >= self._cfg.max_iterations:
            self._convergence_reason = "max_iterations"
            return True
        return False

    def convergence_reason(self) -> str | None:
        return self._convergence_reason

    def history(self) -> list[SearchIteration]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Algorithm internals
    # ------------------------------------------------------------------

    def _absorb_verdict(self, value: int, verdict: bool) -> bool:
        """Update ``feasible_max`` / ``infeasible_min`` from a latched verdict.

        Returns True if the verdict reveals a non-monotonic transition (a
        feasible point above an existing infeasible one or vice versa).
        """
        non_monotonic = False
        if verdict:
            if self.infeasible_min is not None and value >= self.infeasible_min:
                non_monotonic = True
                self.non_monotonic_warning = True
                logger.warning(
                    "monotonic_sla: feasible verdict at %s above existing "
                    "infeasible_min=%s; the SLA boundary is non-monotonic in "
                    "this run. Continuing with the boundary already found; "
                    "see search_history.json for the full trajectory.",
                    value,
                    self.infeasible_min,
                )
            if (self.feasible_max is None or value > self.feasible_max) and (
                self.infeasible_min is None or value < self.infeasible_min
            ):
                # Only widen feasible_max while it stays below infeasible_min;
                # keeps the bracket consistent if non-monotonicity surfaces.
                self.feasible_max = value
        else:
            if self.feasible_max is not None and value <= self.feasible_max:
                non_monotonic = True
                self.non_monotonic_warning = True
                logger.warning(
                    "monotonic_sla: infeasible verdict at %s at-or-below "
                    "existing feasible_max=%s; the SLA boundary is "
                    "non-monotonic. Continuing with the boundary already found.",
                    value,
                    self.feasible_max,
                )
            if (self.infeasible_min is None or value < self.infeasible_min) and (
                self.feasible_max is None or value > self.feasible_max
            ):
                self.infeasible_min = value
        return non_monotonic

    def _plan_next_step(self, value: int, verdict: bool | None) -> None:
        """Decide the next swept value to ask for, or latch a convergence reason."""
        if verdict is None:
            # Stability window: re-ask the same value until verdict latches.
            self._next_value = value
            return

        if self._phase == "probe":
            self._plan_probe_step(value, verdict)
        else:
            self._plan_bisect_step()

    def _plan_probe_step(self, value: int, verdict: bool) -> None:
        """Exponential ramp: double the swept value until a fail or hi reached."""
        if not verdict:
            # First failure during probing — bracket found.
            if self.feasible_max is None:
                # Failed at the very first point: every value fails.
                self._convergence_reason = "monotonic_no_pass_in_range"
                return
            self._phase = "bisect"
            self._plan_bisect_step()
            return

        # Passed: try double, capped at hi.
        next_value = value * 2
        if next_value >= self._hi:
            # Reached the top without seeing a failure: probe hi explicitly.
            if value >= self._hi:
                # Already at hi and it passed: every probed value passes.
                self.feasible_max = self._hi
                self._convergence_reason = "monotonic_no_failure_in_range"
                return
            self._next_value = self._hi
            return
        self._next_value = next_value

    def _plan_bisect_step(self) -> None:
        """Bisection: midpoint of [feasible_max, infeasible_min] until precision."""
        if self.feasible_max is None or self.infeasible_min is None:
            # Cannot bisect without both bounds; defer to the failsafes.
            self._convergence_reason = (
                "monotonic_no_pass_in_range"
                if self.feasible_max is None
                else "monotonic_no_failure_in_range"
            )
            return
        gap = self.infeasible_min - self.feasible_max
        if gap <= 1:
            self._convergence_reason = "monotonic_precision_reached"
            return
        relative = gap / max(self.infeasible_min, 1)
        if relative < Environment.SEARCH_PLANNER.SLA_PRECISION_DEFAULT:
            self._convergence_reason = "monotonic_precision_reached"
            return
        # Integer midpoint biased downward — keeps the bracket tightening.
        midpoint = self.feasible_max + gap // 2
        # Avoid re-probing a value we already have a latched verdict on; if
        # the midpoint coincides with one of the bounds, nudge inward.
        if midpoint <= self.feasible_max:
            midpoint = self.feasible_max + 1
        if midpoint >= self.infeasible_min:
            midpoint = self.infeasible_min - 1
        self._next_value = midpoint

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mutate_base(self, value: int) -> BenchmarkConfig:
        """Return a deep-copied BenchmarkConfig with ``value`` patched in at the dim path.

        mode="python" + context={"include_secrets": True} so neither the
        when_used="json" credential redactors nor the unconditional
        _redact_urls serializer fire mid-pipeline. See ``smooth_isotonic``
        for the full rationale.
        """
        cfg_dict = self._base.model_dump(
            mode="python",
            exclude_none=True,
            context={"include_secrets": True},
        )
        _set_nested_value(cfg_dict, self._dim.path, value)
        return BenchmarkConfig.model_validate(cfg_dict)

    def _extract_objective(self, results: list[RunResult]) -> float | None:
        """Return the mean of the objective metric across successful trials, or None.

        Recorded for ``search_history.json`` compatibility only — the
        planner's bisection logic ignores this and operates purely on
        feasibility verdicts.
        """
        values: list[float] = []
        for r in results:
            if not r.success:
                continue
            metric = r.summary_metrics.get(self._cfg.objectives[0].metric)
            if metric is None:
                continue
            stat_value = getattr(metric, self._cfg.objectives[0].stat, None)
            if stat_value is None:
                continue
            values.append(float(stat_value))
        if not values:
            return None
        return sum(values) / len(values)

    @property
    def warned_iterations(self) -> set[int]:
        """Read-only view of iteration indices flagged as non-monotonic."""
        return set(self._warned_iterations)

    @property
    def state(self) -> dict[str, Any]:
        """Snapshot of planner state for boundary_summary export.

        Returned shape mirrors the ``boundary_summary`` block so the
        post-process pipeline can serialize it without re-derivation.
        """
        return {
            "swept_dim_path": self._dim.path,
            "feasible_max": self.feasible_max,
            "infeasible_min": self.infeasible_min,
            "non_monotonic_warning": self.non_monotonic_warning,
            "convergence_reason": self._convergence_reason,
        }

    def boundary_summary(self) -> dict[str, Any] | None:
        """Boundary-summary block precomputed from internal state.

        The monotonic planner owns the truth for ``feasible_max`` /
        ``infeasible_min`` (latched directly during bisection from per-point
        verdict logs), so reverse-deriving from ``history()`` would force the
        exporter to redo per-iteration feasibility extraction. Surfacing the
        precomputed shape here lets ``write_search_history`` consume it via
        the ABC's ``boundary_summary()`` contract and skip the
        history-based fallback.

        Returns None when no verdict has latched yet (zero iterations or all
        provisional). When at least one bound exists, the returned dict has
        the same shape that ``_compute_boundary_summary`` would produce from
        ``history()``: ``{"swept_dim_path", "feasible_max", "infeasible_min"}``
        with each bound either None or ``{"value", "iteration_idx",
        "objective_value" | "first_breach"}``.
        """
        return compute_boundary_summary(self)
