# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smooth-isotonic SLA-saturation search planner.

Drop-in replacement for :class:`MonotonicSLASearchPlanner` that fixes the
accuracy gaps in deterministic bisection (sign-only feedback, no
denoising) and vLLM-style PCHIP interpolation (passes through every noisy
point). Composition: PAVA denoises, PCHIP interpolates the denoised
points, root-find recovers the SLA-saturation boundary.

Phases
------

1. **Bracket.** Exponential ramp from ``lo`` doubling until first SLA
   breach. Identical shape to ``MonotonicSLASearchPlanner``.
2. **Fit.** 3 internal probes inside ``[feasible_max, infeasible_min]``;
   per-filter PAVA + PCHIP root-find via ``_smooth_isotonic_fit``;
   sigma-normalized multi-SLO aggregation via ``_margin_normalize``.
3. **Replicate (opt-in).** When ``cfg.sla_replicates > 0`` (or the
   auto-formula triggers), re-probe the candidate boundary R times;
   bootstrap CI on the binding-constraint margin terminates when the
   CI does not bracket zero.
4. **Cliff guard (always-on, cheap).** PAVA-residual heuristic flags
   discontinuous curves; reported as ``boundary_type: "cliff"`` so the
   user gets honest output rather than a spline pretending to interpolate
   a step.
5. **Termination.** Bracket gap below 5% of ``infeasible_min`` (the latched
   upper bracket bound, not ``dim.hi``) — or, when the replicate step ran,
   replicate-CI excludes zero — latches ``smooth_isotonic_precision_reached``.

Failure modes
-------------

* No feasible probe seen -> ``smooth_isotonic_no_pass_in_range``.
* No infeasible probe seen -> ``smooth_isotonic_no_failure_in_range``.
* PCHIP root-find returns no crossing -> bisection-midpoint fallback,
  reason ``smooth_isotonic_pchip_fallback_bisection`` if it ends the run.
* ``cfg.max_iterations`` exhausted -> ``max_iterations``.

Module split: this file holds the planner class + ABC contract; the
fit/replicate/cliff-bisect step bodies live in ``_smooth_isotonic_phases``;
the ``boundary_summary`` projection lives in ``_smooth_isotonic_boundary``.
The split mirrors ``monotonic.py`` / ``_monotonic_boundary.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from aiperf.common.environment import Environment
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, SweepVariation, _set_nested_value
from aiperf.orchestrator.search_planner._sla_helpers import (
    averaged_metric_value,
    iteration_feasibility,
)
from aiperf.orchestrator.search_planner._smooth_isotonic_boundary import (
    compute_boundary_summary,
)
from aiperf.orchestrator.search_planner._smooth_isotonic_phases import (
    plan_cliff_bisect_step,
    plan_fit_step,
    plan_replicate_step,
)
from aiperf.orchestrator.search_planner.base import (
    SearchIteration,
    SearchPlanner,
)

if TYPE_CHECKING:
    from aiperf.config.sweep.adaptive import SLAFilter
    from aiperf.orchestrator.models import RunResult

logger = logging.getLogger(__name__)

__all__ = ["SmoothIsotonicSLAPlanner"]


# Number of internal fit-step probes inside [feasible_max, infeasible_min] on
# the first fit iteration. PCHIP needs >=4 distinct points to be meaningful;
# bracket bounds give 2, so 3 internal yields 5 total.
_FIT_INTERNAL_PROBES = 3


_PhaseLiteral = Literal["bracket", "fit", "replicate", "cliff_bisect"]


def _find_phase_index(phases: list[dict[str, Any]], name: str) -> int | None:
    """Return the index of the first phase with ``name`` field equal to ``name``.

    Defensive against malformed fixtures where a phase entry is not a dict
    (e.g. test stubs); such entries are skipped rather than raising.
    """
    for idx, phase in enumerate(phases):
        if isinstance(phase, dict) and phase.get("name") == name:
            return idx
    return None


class SmoothIsotonicSLAPlanner(SearchPlanner):
    """Smooth-isotonic SLA-saturation 1D planner.

    Synonyms (so grep finds it from any framing): ``boundary``,
    ``saturation``, ``capacity``, ``max-passing``, ``find-the-knee``.
    """

    def __init__(self, base_config: BenchmarkConfig, cfg: AdaptiveSearchSweep) -> None:
        if len(cfg.search_space) != 1:
            raise ValueError(
                "smooth_isotonic planner requires exactly one search-space "
                f"dimension; got {len(cfg.search_space)}. For multi-dimensional "
                "search use the bayesian planner via `--search-planner bayesian`."
            )
        dim = cfg.search_space[0]
        if dim.kind != "int":
            raise ValueError(
                f"smooth_isotonic planner v1 supports kind='int' dimensions only; "
                f"got kind={dim.kind!r} on path {dim.path!r}. Use the bayesian "
                "planner for real-valued dimensions."
            )
        if not cfg.sla_filters:
            raise ValueError(
                "smooth_isotonic planner requires at least one SLA filter "
                "(sla_filters is empty); pass --search-sla / --ttft-sla-ms / "
                "--tpot-sla-ms / --e2e-sla-ms."
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
        self._sla_filters: list[SLAFilter] = list(cfg.sla_filters)
        self._filter_keys: list[str] = [
            f"{i}:{f.metric_tag}.{f.stat}.{f.op}.{f.threshold}"
            for i, f in enumerate(self._sla_filters)
        ]

        self.feasible_max: int | None = None
        self.infeasible_min: int | None = None
        # Per-x raw signed margins per filter. "Signed" means negative=feasible
        # regardless of filter ``op`` (gt filters are negated so PAVA's
        # ``increasing=True`` assumption holds for all filters).
        self._raw_probes: dict[int, list[dict[str, float]]] = {}

        self._phase: _PhaseLiteral = "bracket"
        self._next_value: int = self._lo
        self._pending_value: int | None = None
        self._probe_queue: list[int] = []

        self._candidate_x: int | None = None
        self._fit_count: int = 0

        self._iter = 0
        self._history: list[SearchIteration] = []
        self._convergence_reason: str | None = None
        self.boundary_type: Literal["smooth", "cliff"] | None = None
        self.boundary_ci_low: float | None = None
        self.boundary_ci_high: float | None = None
        self.binding_constraint: str | None = None
        self.non_monotonic_warning: bool = False
        # Per-iteration latch: set by ``_flag_non_monotonic`` while
        # ``tell`` is absorbing the current verdict, then read + cleared
        # when constructing this iteration's ``SearchIteration``. Mirrors
        # MonotonicSLASearchPlanner's per-iteration semantics so
        # ``search_history.json`` flags the offending iteration rather
        # than only carrying the cumulative planner-level boolean.
        self._non_monotonic_this_iter: bool = False

        # Track which swept-dim values we have already probed so the warmup
        # floor downshifts to
        # ``Environment.SEARCH_PLANNER.REPLICATE_WARMUP_FLOOR`` for replicates.
        self._first_probe_at: set[int] = set()

    # ------------------------------------------------------------------
    # SearchPlanner ABC
    # ------------------------------------------------------------------

    def ask(self) -> tuple[BenchmarkConfig, SweepVariation] | None:
        """Return the next smooth-isotonic probe and latch it as pending.

        Drains queued internal probes before using `_next_value`, mutates
        `_pending_value`, and returns None after a convergence reason is latched.
        """
        if self.is_converged():
            return None
        if self._probe_queue:
            self._next_value = self._probe_queue.pop(0)
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
        """Absorb results for the pending smooth-isotonic probe.

        Requires a preceding `ask()`; raises RuntimeError otherwise. Records signed
        SLA margins, updates the feasible/infeasible bracket, advances the current
        phase, and appends the SearchIteration with any per-iteration non-monotonic
        warning.
        """
        if self._pending_value is None:
            raise RuntimeError("tell() called without matching ask()")
        value = self._pending_value
        self._pending_value = None

        feasible = iteration_feasibility(results, self._sla_filters)
        margins = self._signed_margins(results)
        self._raw_probes.setdefault(value, []).append(margins)

        # Clear the per-iteration latch before verdict absorption so
        # ``_flag_non_monotonic`` can set it for THIS iteration only.
        self._non_monotonic_this_iter = False
        # Bracket update is universal across phases.
        self._absorb_verdict(value, feasible)
        non_monotonic_this_iter = self._non_monotonic_this_iter

        if self._phase == "bracket":
            self._plan_bracket_step(value, feasible)
        elif self._phase == "fit":
            plan_fit_step(self)
        elif self._phase == "cliff_bisect":
            plan_cliff_bisect_step(self)
        else:  # "replicate"
            plan_replicate_step(self)

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
                feasible=feasible,
                non_monotonic_warning=non_monotonic_this_iter,
            )
        )
        self._iter += 1

    def is_converged(self) -> bool:
        """Return True once the smooth-isotonic planner has a stop reason.

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

    def boundary_summary(self) -> dict[str, Any] | None:
        """Boundary-summary block + smooth-isotonic-specific fields.

        Delegates to ``_smooth_isotonic_boundary.compute_boundary_summary``.
        """
        return compute_boundary_summary(self)

    # ------------------------------------------------------------------
    # Bracket step + verdict latching
    # ------------------------------------------------------------------

    def _absorb_verdict(self, value: int, feasible: bool) -> None:
        """Latch ``feasible_max`` / ``infeasible_min``; flag non-monotonicity."""
        if feasible:
            if self.infeasible_min is not None and value >= self.infeasible_min:
                self._flag_non_monotonic(
                    f"feasible verdict at {value} above existing "
                    f"infeasible_min={self.infeasible_min}"
                )
                return
            if self.feasible_max is None or value > self.feasible_max:
                self.feasible_max = value
        else:
            if self.feasible_max is not None and value <= self.feasible_max:
                self._flag_non_monotonic(
                    f"infeasible verdict at {value} at-or-below existing "
                    f"feasible_max={self.feasible_max}"
                )
                return
            if self.infeasible_min is None or value < self.infeasible_min:
                self.infeasible_min = value

    def _flag_non_monotonic(self, detail: str) -> None:
        self.non_monotonic_warning = True
        self._non_monotonic_this_iter = True
        logger.warning(
            "smooth_isotonic: %s; SLA boundary is non-monotonic in this run.", detail
        )

    def _plan_bracket_step(self, value: int, feasible: bool) -> None:
        if not feasible:
            if self.feasible_max is None:
                self._convergence_reason = "smooth_isotonic_no_pass_in_range"
                return
            self._enter_fit_phase()
            return
        next_value = value * 2
        if next_value >= self._hi:
            if value >= self._hi:
                self.feasible_max = self._hi
                self._convergence_reason = "smooth_isotonic_no_failure_in_range"
                return
            self._next_value = self._hi
            return
        self._next_value = next_value

    def _enter_fit_phase(self) -> None:
        self._phase = "fit"
        self._queue_internal_probes(_FIT_INTERNAL_PROBES)

    def _queue_internal_probes(self, count: int) -> None:
        """Schedule ``count`` evenly-spaced probes inside the bracket."""
        if self.feasible_max is None or self.infeasible_min is None:
            return
        gap = self.infeasible_min - self.feasible_max
        if gap <= 1:
            return
        for k in range(1, count + 1):
            frac = k / (count + 1)
            x = self.feasible_max + max(1, round(gap * frac))
            x = min(x, self.infeasible_min - 1)
            x = max(x, self.feasible_max + 1)
            if x not in self._raw_probes:
                self._probe_queue.append(int(x))
        # Dedupe while preserving order.
        seen: set[int] = set()
        deduped: list[int] = []
        for x in self._probe_queue:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        self._probe_queue = deduped

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _signed_margins(self, results: list[RunResult]) -> dict[str, float]:
        """Per-filter signed margin (negative=feasible, increasing in x)."""
        out: dict[str, float] = {}
        for key, sla in zip(self._filter_keys, self._sla_filters, strict=True):
            value = averaged_metric_value(results, sla.metric_tag, sla.stat)
            if value is None:
                continue
            if sla.op in ("lt", "le"):
                out[key] = float(value) - float(sla.threshold)
            else:
                out[key] = float(sla.threshold) - float(value)
        return out

    def _mutate_base(self, value: int) -> BenchmarkConfig:
        # mode="python" silences the when_used="json" credential redactors
        # (api_key / headers). context={"include_secrets": True} silences
        # the unconditional _redact_urls serializer that strips userinfo
        # (user:pass@host) from endpoint.urls regardless of mode. Both
        # are needed; without the context flag, URL-credentialed sweeps
        # like postgres / MLflow URIs would hit "<redacted>" in the
        # iteration's config and fail to authenticate. Mirrors the
        # pattern in config/loader/plan.py (PR #972).
        cfg_dict = self._base.model_dump(
            mode="python",
            exclude_none=True,
            context={"include_secrets": True},
        )
        _set_nested_value(cfg_dict, self._dim.path, value)
        self._apply_sla_precision(cfg_dict)
        self._apply_sla_warmup(cfg_dict, value)
        return BenchmarkConfig.model_validate(cfg_dict)

    def _apply_sla_precision(self, cfg_dict: dict[str, Any]) -> None:
        """Override profiling-phase ``requests`` per ``cfg.sla_precision``.

        Only fills in when the user did not specify ``requests`` on the
        profiling phase already; explicit user values always win. Defensive
        against degenerate fixtures where ``phases`` is missing/empty or the
        profiling phase is absent.
        """
        target = Environment.SEARCH_PLANNER.SLA_PRECISION_REQUESTS.get(
            self._cfg.sla_precision
        )
        if target is None:
            return
        phases = cfg_dict.get("phases")
        if not phases:
            return
        idx = _find_phase_index(phases, "profiling")
        if idx is None:
            return
        # ``exclude_none=True`` on dump means an unset ``requests`` is absent
        # from the dict; treat both ``None`` and missing key as user-unset.
        existing = phases[idx].get("requests")
        if existing is None:
            phases[idx]["requests"] = target

    def _apply_sla_warmup(self, cfg_dict: dict[str, Any], value: int) -> None:
        """Prepend a per-iteration ``warmup`` phase to ``cfg_dict["phases"]``.

        Skipped when ``cfg.sla_warmup_seconds == 0`` (explicit user opt-out)
        or when the profiling phase cannot be located. The warmup uses the
        same swept-dim value being probed and is excluded from results. If
        the user already declared a warmup phase, it is replaced — the
        unique-name validator forbids two ``warmup`` entries, and the swept
        value should drive the warmup during the search.
        """
        if self._cfg.sla_warmup_seconds == 0:
            self._first_probe_at.add(value)
            return
        phases = cfg_dict.get("phases")
        if not phases:
            return
        if _find_phase_index(phases, "profiling") is None:
            return

        base_warmup = (
            self._cfg.sla_warmup_seconds
            if self._cfg.sla_warmup_seconds is not None
            else Environment.SEARCH_PLANNER.DEFAULT_WARMUP_SECONDS
        )
        if value not in self._first_probe_at:
            duration = max(
                Environment.SEARCH_PLANNER.FIRST_PROBE_WARMUP_FLOOR, base_warmup
            )
            self._first_probe_at.add(value)
        else:
            duration = max(
                Environment.SEARCH_PLANNER.REPLICATE_WARMUP_FLOOR, base_warmup
            )

        warmup_phase: dict[str, Any] = {
            "name": "warmup",
            "type": "concurrency",
            "concurrency": value,
            "duration": duration,
            "exclude_from_results": True,
        }
        # Replace any user-declared warmup so the per-iteration sweep value
        # drives the warmup; the unique-name validator on BenchmarkConfig
        # forbids two phases named "warmup".
        if phases and phases[0].get("name") == "warmup":
            phases[0] = warmup_phase
        else:
            phases.insert(0, warmup_phase)

    def _extract_objective(self, results: list[RunResult]) -> float | None:
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
