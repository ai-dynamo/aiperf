# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-tier SLO boundary search planner.

Composition layer over existing search algorithms that resolves per-tier SLO
boundaries simultaneously by sharing observations, exploiting tier ordering,
and allocating probes to the widest-gap bracket.

# TODO: follow-up PR to update docs/tutorials with multi-tier search usage.

Phases
------

1. **Bracket (exponential ramp).** Start at ``dim.lo``, double concurrency
   until the first SLA failure for ANY tier. Observations from this phase
   seed initial bracket bounds for all tiers retroactively.
2. **Bisection.** Once the first failure is detected, delegate probe selection
   to ``ProbeAllocator`` which picks the midpoint of the widest bracket gap
   across all tiers.

The planner activates only when 2+ tiers are configured via ``--search-sla-tier``.
Single-tier behavior remains unchanged (existing planners run unmodified).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aiperf.common.environment import Environment
from aiperf.common.finite import is_finite_value
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, SweepVariation
from aiperf.config.sweep.adaptive import SLAFilter, SLOTier
from aiperf.orchestrator.search_planner._shared_warmup import mutate_base
from aiperf.orchestrator.search_planner._sla_helpers import (
    first_failing_filter,
    iteration_feasibility,
)
from aiperf.orchestrator.search_planner.base import SearchIteration, SearchPlanner
from aiperf.orchestrator.search_planner.multi_tier_allocator import ProbeAllocator
from aiperf.orchestrator.search_planner.multi_tier_models import (
    BracketState,
    TierResult,
)
from aiperf.orchestrator.search_planner.multi_tier_ordering import (
    TierOrderingDetector,
)
from aiperf.orchestrator.search_planner.multi_tier_store import (
    SharedObservationStore,
)

if TYPE_CHECKING:
    from aiperf.orchestrator.models import RunResult

logger = logging.getLogger(__name__)

__all__ = ["MultiTierPlanner"]

_ALL_STATS = (
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
    "std",
)


class MultiTierPlanner(SearchPlanner):
    """Multi-tier SLA-saturation planner using bracket/bisection search.

    Manages per-tier bracket state, cross-tier observation sharing, and
    ordering inference. Reuses the underlying algorithm's warmup strategy,
    precision settings, and convergence thresholds (via _shared_warmup.py),
    but owns the bracket/bisect loop directly.

    Unlike single-tier smooth_isotonic (which denoises via isotonic regression
    and replicates), this planner uses single-run verdicts per probe. When a
    non-monotonic observation is detected, it clears the stale bound and
    continues probing rather than re-running the same concurrency. This is
    intentional: multi-tier search trades per-probe denoising for faster
    global convergence across N tiers by sharing observations and exploiting
    tier ordering.

    Future: replicate-based stability can be added as an opt-in mode if
    noisy endpoints require it (tracked as a follow-up).
    """

    def __init__(
        self,
        base_config: BenchmarkConfig,
        cfg: AdaptiveSearchSweep,
        tiers: list[SLOTier],
    ) -> None:
        if len(cfg.search_space) != 1:
            raise ValueError(
                "multi_tier planner requires exactly one search-space "
                f"dimension; got {len(cfg.search_space)}."
            )
        dim = cfg.search_space[0]
        if dim.kind != "int":
            raise ValueError(
                f"multi_tier planner supports kind='int' dimensions only; "
                f"got kind={dim.kind!r} on path {dim.path!r}."
            )
        if len(tiers) < 2:
            raise ValueError(
                f"multi_tier planner requires at least 2 tiers; got {len(tiers)}."
            )

        self._base = base_config
        self._cfg = cfg
        self._dim = dim
        self._lo: int = int(dim.lo)
        self._hi: int = int(dim.hi)
        self._tiers = tiers
        self._max_iterations: int = cfg.max_iterations

        # Per-tier bracket state
        self._brackets: list[BracketState] = [BracketState(tier=t) for t in tiers]

        # Shared components
        self._store = SharedObservationStore()
        self._ordering = TierOrderingDetector(tiers)
        self._allocator = ProbeAllocator()

        # Detect ordering pairs at init
        self._ordering.detect_ordering()

        # Phase tracking
        self._phase: str = "bracket"
        self._next_value: int = self._lo
        self._pending_value: int | None = None

        # Iteration state
        self._iter = 0
        self._history: list[SearchIteration] = []
        self._convergence_reason: str | None = None

        # Warmup tracking (shared across all tiers)
        self._first_probe_at: set[int] = set()

        # Global SLA filters from --search-sla that compose with per-tier filters
        self._global_filters: list[SLAFilter] = (
            list(cfg.sla_filters) if cfg.sla_filters else []
        )

    # ------------------------------------------------------------------
    # SearchPlanner ABC
    # ------------------------------------------------------------------

    def ask(self) -> tuple[BenchmarkConfig, SweepVariation] | None:
        """Return the next probe config and variation, or None when done."""
        if self.is_converged():
            return None

        if self._phase == "bracket":
            value = self._next_value
        else:
            # Bisection phase: delegate to allocator
            probe = self._allocator.select_next_probe(self._brackets)
            if probe is None:
                # Check if any non-converged tier still needs bounds
                probe = self._find_missing_bound_probe()
                if probe is None:
                    self._convergence_reason = "multi_tier_all_converged"
                    return None
            value = probe

        self._pending_value = value
        bench_cfg = mutate_base(
            self._base,
            self._dim,
            value,
            cfg=self._cfg,
            first_probe_at=self._first_probe_at,
        )
        variation = SweepVariation(
            index=self._iter,
            label=f"search_iter_{self._iter:04d}",
            values={self._dim.path: value},
        )
        return bench_cfg, variation

    def tell(self, variation: SweepVariation, results: list[RunResult]) -> None:
        """Absorb results: store, evaluate all tiers, update brackets, propagate ordering."""
        if self._pending_value is None:
            raise RuntimeError("tell() called without matching ask()")
        value = self._pending_value
        self._pending_value = None

        self._store.store(value, results)
        self._log_no_successful_trials(value, results)

        tier_verdicts, non_monotonic_this_iter = self._evaluate_all_tiers(
            value, results
        )
        self._propagate_ordering(value, tier_verdicts)
        self._check_bracket_convergence()

        if self._phase == "bracket":
            self._advance_bracket_phase(value, tier_verdicts)

        self._record_iteration(
            variation, results, tier_verdicts, non_monotonic_this_iter
        )
        self._log_tier_progress(value, tier_verdicts)

    def is_converged(self) -> bool:
        """True when all tiers converged or max_iterations exhausted."""
        if self._convergence_reason is not None:
            return True
        if self._iter >= self._max_iterations:
            self._convergence_reason = "max_iterations"
            return True
        if all(b.converged for b in self._brackets):
            self._convergence_reason = "multi_tier_all_converged"
            return True
        return False

    def history(self) -> list[SearchIteration]:
        """All iterations recorded so far, in submission order."""
        return list(self._history)

    def convergence_reason(self) -> str | None:
        """Reason for convergence, or None if still running."""
        return self._convergence_reason

    def boundary_summary(self) -> dict[str, Any] | None:
        """Boundary summary from the most-lenient tier for backward compatibility."""
        if not self._history:
            return None

        # Most-lenient tier = highest feasible_max (or last tier if none converged)
        best_bracket: BracketState | None = None
        for b in self._brackets:
            if b.feasible_max is not None and (
                best_bracket is None or b.feasible_max > best_bracket.feasible_max
            ):
                best_bracket = b
        if best_bracket is None:
            best_bracket = self._brackets[-1]

        return {
            "swept_dim_path": self._dim.path,
            "feasible_max": (
                {
                    "value": best_bracket.feasible_max,
                    "iteration_idx": self._find_iteration_for_value(
                        best_bracket.feasible_max
                    ),
                    "objective_value": None,
                }
                if best_bracket.feasible_max is not None
                else None
            ),
            "infeasible_min": (
                {
                    "value": best_bracket.infeasible_min,
                    "iteration_idx": self._find_iteration_for_value(
                        best_bracket.infeasible_min
                    ),
                    "first_breach": best_bracket.binding_constraint,
                }
                if best_bracket.infeasible_min is not None
                else None
            ),
            "non_monotonic_warning": any(
                b.non_monotonic_warning for b in self._brackets
            ),
            "convergence_reason": self._convergence_reason,
        }

    # ------------------------------------------------------------------
    # Multi-tier specific API
    # ------------------------------------------------------------------

    def tier_metadata(self) -> dict[str, Any]:
        """Produce tier-level metadata for search_history.json output."""
        total_probes = sum(b.probe_count for b in self._brackets)
        actual_probes = len(self._history)
        active_pairs = self._ordering.ordered_pairs
        ordering_pairs: list[dict[str, str]] | None = None
        if active_pairs:
            ordering_pairs = [
                {
                    "strict": self._tiers[strict].label,
                    "lenient": self._tiers[lenient].label,
                }
                for strict, lenient in active_pairs
            ]
        return {
            "actual_probe_count": actual_probes,
            "tier_evaluation_count": total_probes,
            "ordering_detected": len(active_pairs) > 0,
            "ordering_pairs": ordering_pairs,
        }

    def tier_results(self) -> list[TierResult]:
        """Produce per-tier result records for output."""
        results: list[TierResult] = []
        for bracket in self._brackets:
            status = self._tier_convergence_status(bracket)
            boundary_metrics = self._extract_boundary_metrics(bracket)
            bracket_lower = bracket.feasible_max
            bracket_upper = bracket.infeasible_min
            if (
                bracket_lower is not None
                and bracket_upper is not None
                and bracket_upper <= bracket_lower
            ):
                bracket_upper = None  # inverted bracket is meaningless
            results.append(
                TierResult(
                    label=bracket.tier.label,
                    boundary_concurrency=bracket.feasible_max,
                    convergence_status=status,
                    convergence_reason=bracket.convergence_reason,
                    binding_constraint=bracket.binding_constraint,
                    bracket_lower=bracket_lower,
                    bracket_upper=bracket_upper,
                    confidence_interval=None,
                    probe_count=bracket.probe_count,
                    boundary_metrics=boundary_metrics,
                    filters=[
                        {
                            "metric_tag": f.metric_tag,
                            "stat": f.stat,
                            "op": f.op,
                            "threshold": f.threshold,
                        }
                        for f in self._effective_filters(bracket)
                    ],
                )
            )
        return results

    # ------------------------------------------------------------------
    # Internal: tell() decomposition
    # ------------------------------------------------------------------

    def _effective_filters(self, bracket: BracketState) -> list[SLAFilter]:
        """Return tier filters + global filters for evaluation."""
        return list(bracket.tier.filters) + self._global_filters

    def _log_no_successful_trials(self, value: int, results: list[RunResult]) -> None:
        """Log warning when all trials at a concurrency level failed (Req 10.1)."""
        successful = [r for r in results if r.success]
        if not successful:
            logger.warning(
                "multi_tier: no successful trials at concurrency=%d "
                "(%d trials failed); marking all tiers infeasible at this level",
                value,
                len(results),
            )

    def _evaluate_all_tiers(
        self, value: int, results: list[RunResult]
    ) -> tuple[list[bool], bool]:
        """Evaluate all tiers against an observation, updating brackets.

        Returns (tier_verdicts, non_monotonic_this_iter).
        """
        successful = [r for r in results if r.success]
        tier_verdicts: list[bool] = []
        non_monotonic_this_iter = False

        for bracket in self._brackets:
            all_filters = bracket.tier.filters + self._global_filters
            feasible = iteration_feasibility(results, all_filters)
            tier_verdicts.append(feasible)

            if not feasible and successful:
                self._warn_missing_metrics(bracket, successful)

            if bracket.converged:
                continue

            was_non_monotonic = bracket.non_monotonic_warning
            bracket.probe_count += 1
            self._update_bracket(bracket, value, feasible, results)
            if bracket.non_monotonic_warning and not was_non_monotonic:
                non_monotonic_this_iter = True

        return tier_verdicts, non_monotonic_this_iter

    def _propagate_ordering(self, value: int, tier_verdicts: list[bool]) -> None:
        """Propagate ordering inferences from this probe's verdicts."""
        for i, feasible in enumerate(tier_verdicts):
            if self._brackets[i].converged:
                continue
            if not feasible:
                self._ordering.propagate_fail(i, value, self._brackets)
            else:
                self._ordering.propagate_pass(i, value, self._brackets)

    def _record_iteration(
        self,
        variation: SweepVariation,
        results: list[RunResult],
        tier_verdicts: list[bool],
        non_monotonic_this_iter: bool,
    ) -> None:
        """Append a SearchIteration record to history."""
        objective_value = self._extract_objective(results)
        self._history.append(
            SearchIteration(
                iteration_idx=self._iter,
                variation_values=dict(variation.values),
                objective_value=objective_value,
                objective_values=(
                    [objective_value] if objective_value is not None else None
                ),
                results=list(results),
                feasible=any(tier_verdicts),
                non_monotonic_warning=non_monotonic_this_iter,
            )
        )
        self._iter += 1

    # ------------------------------------------------------------------
    # Internal: bracket management
    # ------------------------------------------------------------------

    def _update_bracket(
        self,
        bracket: BracketState,
        value: int,
        feasible: bool,
        results: list[RunResult],
    ) -> None:
        """Update a single tier's bracket bounds from a verdict."""
        if feasible:
            # Non-monotonic detection (Req 10.4): feasible at or above infeasible_min
            if bracket.infeasible_min is not None and value >= bracket.infeasible_min:
                bracket.non_monotonic_warning = True
                logger.warning(
                    "multi_tier[%s]: feasible verdict at concurrency=%d "
                    "above existing infeasible_min=%d; SLA boundary is "
                    "non-monotonic for this tier",
                    bracket.tier.label,
                    value,
                    bracket.infeasible_min,
                )
            if bracket.feasible_max is None or value > bracket.feasible_max:
                bracket.feasible_max = value
        else:
            # Non-monotonic detection (Req 10.4): infeasible at or below feasible_max
            if bracket.feasible_max is not None and value <= bracket.feasible_max:
                bracket.non_monotonic_warning = True
                logger.warning(
                    "multi_tier[%s]: infeasible verdict at concurrency=%d "
                    "at-or-below existing feasible_max=%d; SLA boundary is "
                    "non-monotonic for this tier",
                    bracket.tier.label,
                    value,
                    bracket.feasible_max,
                )
            if bracket.infeasible_min is None or value < bracket.infeasible_min:
                bracket.infeasible_min = value
                # Track binding constraint
                breach = first_failing_filter(results, self._effective_filters(bracket))
                if breach is not None:
                    bracket.binding_constraint = breach

    def _advance_bracket_phase(self, value: int, tier_verdicts: list[bool]) -> None:
        """Advance the exponential ramp or transition to bisection."""
        if not all(tier_verdicts):
            self._handle_first_failure(value)
            return
        self._handle_all_pass(value)

    def _handle_first_failure(self, value: int) -> None:
        """Transition to bisection on first failure, or terminate if no pass exists."""
        if not any(b.feasible_max is not None for b in self._brackets):
            logger.warning(
                "multi_tier: all tiers infeasible at lowest probed "
                "concurrency=%d; terminating with no_pass_in_range",
                value,
            )
            for b in self._brackets:
                if not b.converged:
                    b.converged = True
                    b.convergence_reason = "no_pass_in_range"
            self._convergence_reason = "multi_tier_all_converged"
            return
        self._phase = "bisect"

    def _handle_all_pass(self, value: int) -> None:
        """All tiers passed: double the probe value or cap at hi."""
        next_value = max(value * 2, value + 1)
        if next_value >= self._hi:
            if value >= self._hi:
                for b in self._brackets:
                    if not b.converged and b.feasible_max is not None:
                        b.feasible_max = max(b.feasible_max, self._hi)
                    elif not b.converged:
                        b.feasible_max = self._hi
                    if not b.converged:
                        b.converged = True
                        b.convergence_reason = "no_failure_in_range"
                self._convergence_reason = "multi_tier_all_converged"
                return
            self._next_value = self._hi
            return
        self._next_value = next_value

    def _find_missing_bound_probe(self) -> int | None:
        """Find a probe to establish missing bounds for tiers that need them.

        Called when the main allocator returns None (all fully-bracketed tiers
        are converged) but some tiers still lack one bound. Unlike the bracket
        ramp (which doubles during the initial exponential phase), this uses
        2x as a heuristic to find the missing bound efficiently.
        """
        for bracket in self._brackets:
            if bracket.converged:
                continue
            if bracket.feasible_max is not None and bracket.infeasible_min is None:
                candidate = min(bracket.feasible_max * 2, self._hi)
                if candidate > bracket.feasible_max:
                    return candidate
                # feasible_max >= hi with no failure found: mark no_failure_in_range
                bracket.feasible_max = self._hi
                bracket.converged = True
                bracket.convergence_reason = "no_failure_in_range"
                continue
            if bracket.infeasible_min is not None and bracket.feasible_max is None:
                candidate = max(bracket.infeasible_min // 2, self._lo)
                if candidate < bracket.infeasible_min:
                    return candidate
        return None

    def _check_bracket_convergence(self) -> None:
        """Mark tiers as converged when bracket gap is within precision."""
        precision = Environment.SEARCH_PLANNER.SLA_PRECISION_DEFAULT
        for bracket in self._brackets:
            if bracket.converged:
                continue
            if bracket.feasible_max is None or bracket.infeasible_min is None:
                continue
            # Inverted bracket: non-monotonic evidence contradicts the boundary
            if bracket.feasible_max >= bracket.infeasible_min:
                bracket.non_monotonic_warning = True
                logger.warning(
                    "multi_tier[%s]: inverted bracket (feasible_max=%d >= "
                    "infeasible_min=%d); clearing stale infeasible_min and "
                    "continuing search",
                    bracket.tier.label,
                    bracket.feasible_max,
                    bracket.infeasible_min,
                )
                bracket.infeasible_min = None
                bracket.binding_constraint = None
                continue
            gap = bracket.infeasible_min - bracket.feasible_max
            if gap <= 1:
                bracket.converged = True
                bracket.convergence_reason = "multi_tier_precision_reached"
            elif bracket.infeasible_min > 0:
                relative = gap / bracket.infeasible_min
                if relative < precision:
                    bracket.converged = True
                    bracket.convergence_reason = "multi_tier_precision_reached"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_boundary_metrics(self, bracket: BracketState) -> dict[str, Any] | None:
        """Extract metric summaries from the observation at the boundary concurrency."""
        if bracket.feasible_max is None:
            return None
        obs = self._store.get(bracket.feasible_max)
        if not obs:
            return None
        successful = [r for r in obs[-1] if r.success]
        if not successful:
            return None
        # Use first successful trial's metrics
        run = successful[0]
        result: dict[str, Any] = {}
        for tag, metric in run.summary_metrics.items():
            stat_values: dict[str, float] = {}
            for stat in _ALL_STATS:
                val = getattr(metric, stat, None)
                if isinstance(val, int | float) and is_finite_value(val):
                    stat_values[stat] = val
            if stat_values:
                result[tag] = stat_values
        return result or None

    def _warn_missing_metrics(
        self,
        bracket: BracketState,
        successful: list[RunResult],
    ) -> None:
        """Log a warning for each SLA filter referencing a missing metric.

        Called when a tier is infeasible but there are successful trials — detects
        the case where the filter failed because the metric was absent rather than
        because the metric exceeded the threshold (Req 10.2).
        """
        for sla in self._effective_filters(bracket):
            metric_present = any(
                run.summary_metrics.get(sla.metric_tag) is not None
                for run in successful
            )
            if not metric_present:
                logger.warning(
                    "multi_tier[%s]: metric %r missing from observation; "
                    "filter %s.%s %s %s treated as failed",
                    bracket.tier.label,
                    sla.metric_tag,
                    sla.metric_tag,
                    sla.stat,
                    sla.op,
                    sla.threshold,
                )
                continue
            stat_present = any(
                getattr(run.summary_metrics.get(sla.metric_tag), sla.stat, None)
                is not None
                for run in successful
            )
            if not stat_present:
                logger.warning(
                    "multi_tier[%s]: stat %r missing for metric %r; "
                    "filter %s.%s %s %s treated as failed",
                    bracket.tier.label,
                    sla.stat,
                    sla.metric_tag,
                    sla.metric_tag,
                    sla.stat,
                    sla.op,
                    sla.threshold,
                )

    def _extract_objective(self, results: list[RunResult]) -> float | None:
        """Mean of the objective metric across successful trials, or None."""
        if not self._cfg.objectives:
            return None
        obj = self._cfg.objectives[0]
        values: list[float] = []
        for r in results:
            if not r.success:
                continue
            metric = r.summary_metrics.get(obj.metric)
            if metric is None:
                continue
            stat_value = getattr(metric, obj.stat, None)
            if stat_value is None:
                continue
            values.append(float(stat_value))
        if not values:
            return None
        return sum(values) / len(values)

    def _tier_convergence_status(self, bracket: BracketState) -> str:
        """Derive convergence_status string for output."""
        if not bracket.converged:
            return "partial"
        reason = bracket.convergence_reason or ""
        if "precision_reached" in reason:
            return "converged"
        if "no_pass_in_range" in reason:
            return "no_pass_in_range"
        if "no_failure_in_range" in reason:
            return "no_failure_in_range"
        return "converged"

    def _find_iteration_for_value(self, value: int | None) -> int | None:
        """Find the iteration index that probed a given value."""
        if value is None:
            return None
        for h in self._history:
            for v in h.variation_values.values():
                if v == value:
                    return h.iteration_idx
        return None

    def _log_tier_progress(self, value: int, tier_verdicts: list[bool]) -> None:
        """Log per-tier convergence progress after each probe.

        Emits a single INFO line summarizing all tiers' bracket status so the
        user can track convergence in simple/none UI modes (Req 9.2).
        """
        parts: list[str] = []
        for bracket, verdict in zip(self._brackets, tier_verdicts, strict=True):
            status = "PASS" if verdict else "FAIL"
            if bracket.converged:
                tag = f"[converged@{bracket.feasible_max}]"
            elif (
                bracket.feasible_max is not None and bracket.infeasible_min is not None
            ):
                tag = f"[{bracket.feasible_max},{bracket.infeasible_min}]"
            elif bracket.feasible_max is not None:
                tag = f"[{bracket.feasible_max},?]"
            elif bracket.infeasible_min is not None:
                tag = f"[?,{bracket.infeasible_min}]"
            else:
                tag = "[?,?]"
            parts.append(f"{bracket.tier.label}:{status}{tag}")

        converged_count = sum(1 for b in self._brackets if b.converged)
        logger.info(
            "multi_tier probe@%d: %s (%d/%d tiers converged)",
            value,
            " | ".join(parts),
            converged_count,
            len(self._brackets),
        )
