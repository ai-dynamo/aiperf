# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optuna-backed Bayesian-Optimization outer-loop planner.

Expert-mode entry point under the SearchPlanner ABC. Selected via
``--search-planner optuna``; the underlying sampler (``gp`` / ``tpe`` /
``botorch``) is selected via ``--optuna-sampler``. Optuna core is a default
dependency; the BoTorch/Torch stack remains optional.

Constraint handling: SLA filters are passed to the sampler as a
``constraints_func`` that reads observations off ``trial.user_attrs``
(written during ``tell()``). This is Optuna's first-class constrained-BO
API — no penalty-merit reweighting — and works for all three samplers.
The :class:`BayesianSearchPlanner` curated preset subclasses this class
to lock the BoTorch sampler and the qLogNEI / qLogNEHVI acquisition;
both share the same ``ask`` / ``tell`` lifecycle and the same
convergence-reason strings.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

from aiperf.common.finite import is_finite_value
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, SweepVariation, _set_nested_value
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.search_planner._optuna_helpers import (
    _attr_key,
    build_sampler,
)
from aiperf.orchestrator.search_planner._pooled_percentile import (
    percentile_from_stat,
    pooled_percentile_from_results,
)
from aiperf.orchestrator.search_planner._sla_helpers import (
    averaged_metric_value,
    evaluate_three_signal_convergence,
    iteration_feasibility,
)
from aiperf.orchestrator.search_planner.base import (
    SearchIteration,
    SearchPlanner,
)

if TYPE_CHECKING:
    from aiperf.orchestrator.models import RunResult

logger = logging.getLogger(__name__)

# Synthetic worse-than-worst loss handed to Optuna when an iteration produced
# no usable objective. Magnitude is large enough to dominate any plausible
# real objective so the sampler treats the iteration as a failure, but small
# enough not to overflow downstream serialization.
NO_DATA_SENTINEL_LOSS: float = 1.0e6

__all__ = ["OptunaSearchPlanner"]


def _any_non_finite_observation(
    results: list[RunResult], metric_tag: str, stat: str
) -> bool:
    """True iff at least one successful run reports a non-finite ``stat``.

    Mirror of the helper in :mod:`bayesian` — used to distinguish "exporter
    dropped the field" (None on every run) from "exporter wrote NaN/inf"
    (one or more non-finite, none finite). The distinction drives whether
    the planner emits a NaN-specific warn-once.
    """
    for r in results:
        if not r.success:
            continue
        metric = r.summary_metrics.get(metric_tag)
        if metric is None:
            continue
        value = getattr(metric, stat, None)
        if value is None:
            continue
        if not is_finite_value(value):
            return True
    return False


def _build_terminator(mode: str) -> tuple[Any | None, str | None]:
    """Build an optuna.terminator.Terminator for the configured stop rule.

    Returns ``(terminator, reason_string)`` or ``(None, None)`` when the
    user did not opt in. ``reason_string`` is the value written into
    ``search_history.json``'s ``convergence_reason`` field when the
    terminator fires; matches the family naming in the user-facing doc
    (``"posterior_regret_bound"`` for Makarova 2022, ``"emmr"`` for
    Ishibashi 2023).

    Lazy import: ``optuna.terminator`` ships in core, but importing it
    early would make planner construction pay the terminator import cost.
    """
    if mode == "none":
        return None, None
    from optuna.terminator import EMMREvaluator, RegretBoundEvaluator, Terminator

    if mode == "regret":
        return Terminator(improvement_evaluator=RegretBoundEvaluator()), (
            "posterior_regret_bound"
        )
    if mode == "emmr":
        return Terminator(improvement_evaluator=EMMREvaluator()), "emmr"
    raise ValueError(f"unknown optuna_terminator: {mode!r}")


class OptunaSearchPlanner(SearchPlanner):
    """Optuna-backed adaptive outer-loop planner (expert mode).

    Default sampler when ``--optuna-sampler`` is unset: ``botorch``
    (see :attr:`AdaptiveSearchSweep.optuna_sampler`) — the preferred GP path.
    If that implicit default cannot import the optional BoTorch stack, the
    planner warns and falls back to Optuna's core ``TPESampler``. Explicit
    ``--optuna-sampler botorch`` requests raise instead. ``--optuna-sampler gp``
    selects Optuna's native GP-EI and requires ``torch``.

    The :class:`BayesianSearchPlanner` curated preset subclasses this
    class and tries ``optuna_sampler=botorch`` plus the qLogNEI /
    qLogNEHVI acquisition before falling back to TPE; users who don't want to choose sampler /
    acquisition should pick ``--search-planner bayesian`` instead.
    """

    def __init__(
        self,
        base_config: BenchmarkConfig,
        cfg: AdaptiveSearchSweep,
        *,
        allow_implicit_botorch_fallback: bool | None = None,
    ) -> None:
        try:
            import optuna
        except ImportError as e:
            raise ImportError(
                "Optuna planner requires the core `optuna` dependency. "
                "Install or resync AIPerf dependencies with `make first-time-setup`. "
                f"Underlying import error: {e}"
            ) from e

        self._base = base_config
        self._cfg = cfg
        self._iter = 0
        self._history: list[SearchIteration] = []
        self._convergence_reason: str | None = None
        # Patience-based stop: track best objective and iterations since the
        # last improvement. Same idiom as BayesianSearchPlanner so
        # search_history.json convergence reasons stay byte-identical across
        # planners.
        self._best_loss: float | None = None
        self._best_hypervolume: float | None = None
        self._iters_since_improvement: int = 0
        self._pending_trial: Any | None = None
        self._sla_filters = list(cfg.sla_filters)
        self._outcome_constraints = list(cfg.outcome_constraints)
        self._warned_unmeasurable_metrics: set[str] = set()
        # One-shot guard for NaN-as-missing warnings (mirror of bayesian.py).
        self._warned_nan: bool = False
        self._qnehvi_installed: bool = False

        sampler_cfg = cfg
        if allow_implicit_botorch_fallback is None:
            allow_implicit_botorch_fallback = (
                "optuna_sampler" not in cfg.model_fields_set
            )
        try:
            sampler = build_sampler(sampler_cfg)
        except ImportError as exc:
            if cfg.optuna_sampler != "botorch" or not allow_implicit_botorch_fallback:
                raise
            sampler_cfg = self._fallback_to_tpe_after_botorch_import_error(cfg, exc)
            sampler = build_sampler(sampler_cfg)
        self._cfg = sampler_cfg
        directions = [
            "maximize"
            if obj.direction == OptimizationDirection.MAXIMIZE
            else "minimize"
            for obj in cfg.objectives
        ]
        self._study = optuna.create_study(directions=directions, sampler=sampler)
        self._terminator, self._terminator_reason = _build_terminator(
            cfg.optuna_terminator
        )

    def ask(self) -> tuple[BenchmarkConfig, SweepVariation] | None:
        """Ask Optuna for the next trial and return its patched BenchmarkConfig.

        Creates and stores one pending Optuna trial; the next `tell()` must report
        results for the returned variation so `study.tell()` stays paired with
        `study.ask()`.
        """
        if self.is_converged():
            return None

        trial = self._study.ask()
        self._pending_trial = trial
        values: dict[str, Any] = {}
        for dim in self._cfg.search_space:
            log = dim.prior == "log-uniform"
            if dim.kind == "int":
                v: int | float = trial.suggest_int(
                    dim.path, int(dim.lo), int(dim.hi), log=log
                )
            else:
                v = trial.suggest_float(dim.path, dim.lo, dim.hi, log=log)
            values[dim.path] = v

        # mode="python" + context={"include_secrets": True} so neither the
        # when_used="json" credential redactors (api_key / headers) nor the
        # unconditional _redact_urls serializer fire mid-pipeline. See
        # smooth_isotonic._mutate_base for the full rationale.
        cfg_dict = self._base.model_dump(
            mode="python",
            exclude_none=True,
            context={"include_secrets": True},
        )
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
        """Report ``results`` for an ``ask()``-issued ``variation`` to the planner.

        Per iteration: writes per-SLA averaged observations to
        ``trial.user_attrs`` (read by ``constraints_func`` at
        ``study.tell()`` time), computes feasibility against
        ``self._cfg.sla_filters`` (warn-once dedup for unmeasurable
        constraints), tells Optuna the raw objective, and records a
        :class:`SearchIteration`. ``has_unmeasurable`` forces
        ``iteration_feasible=False`` even when the per-trial check
        coincidentally passed.

        ``variation`` must match the most recent ``ask()`` (no pending ask =
        ``RuntimeError``). An empty/all-failed ``results`` still tells
        Optuna a synthetic worse-than-worst objective so the ask/tell
        pairing stays consistent.
        """
        if self._pending_trial is None:
            raise RuntimeError("tell() called without matching ask()")
        trial = self._pending_trial

        has_unmeasurable = self._populate_user_attrs(trial, results)
        self._populate_outcome_user_attrs(trial, results)
        feasible = iteration_feasibility(results, self._sla_filters)
        objective_vec = self._extract_objective_vector(results)

        if objective_vec is None:
            tell_value = self._failure_sentinel_vector()
            logger.warning(
                "Search iteration %d at %s produced no usable objective; "
                "telling Optuna fallback objective=%s and continuing.",
                self._iter,
                variation.values,
                tell_value,
            )
            self._study.tell(trial, [float(v) for v in tell_value])
            objective_for_history: float | None = None
        else:
            self._study.tell(trial, [float(v) for v in objective_vec])
            objective_for_history = objective_vec[0]

        self._pending_trial = None
        self._track_improvement(objective_for_history, objective_vec)

        # Iteration is infeasible when any SLA-referenced metric was
        # unmeasurable OR the per-trial feasibility check returned False.
        # Mirrors BayesianSearchPlanner's ``feasible and not has_unmeasurable``.
        iteration_feasible = feasible and not has_unmeasurable

        self._history.append(
            SearchIteration(
                iteration_idx=self._iter,
                variation_values=dict(variation.values),
                objective_value=objective_for_history,
                objective_values=list(objective_vec)
                if objective_vec is not None
                else None,
                results=list(results),
                feasible=iteration_feasible,
            )
        )
        self._iter += 1
        self._maybe_install_qnehvi_candidates_func()

    def is_converged(self) -> bool:
        """Evaluate Optuna planner stop conditions and latch the reason when met.

        Checks the shared max-iterations / improvement-patience / plateau-CV signals
        first, then the optional Optuna terminator. Terminator fit failures are logged
        and treated as not-yet-converged so search can continue.
        """
        # Mirror of BayesianSearchPlanner.is_converged via the shared
        # evaluator: same three-signal contract (max_iterations /
        # improvement_patience / plateau_cv) and same convergence-reason
        # strings, so search_history.json consumers don't branch on planner
        # type.
        converged, reason = evaluate_three_signal_convergence(
            iter_count=self._iter,
            history=self._history,
            iters_since_improvement=self._iters_since_improvement,
            cfg=self._cfg,
        )
        if converged and reason is not None:
            self._convergence_reason = reason
            return True
        # Posterior-regret stopping (Makarova 2022 / Ishibashi 2023; same
        # family as Wilson 2024). Layered AFTER three-signal so the cheap
        # checks short-circuit; only fires once the terminator's internal
        # GP has enough data to bound regret with high probability.
        if self._terminator is not None and self._iter >= 2:
            try:
                if self._terminator.should_terminate(study=self._study):
                    self._convergence_reason = self._terminator_reason
                    return True
            except Exception as e:
                # Optuna's terminator can raise mid-run when its GP fails to
                # fit (e.g. degenerate observations); a stop-rule failure is
                # not a planner failure -- log and fall through to "not
                # converged" so the three-signal check is still authoritative.
                logger.debug(
                    "Optuna terminator %s raised during should_terminate at "
                    "iteration %d: %r; treating as not-yet-converged.",
                    self._cfg.optuna_terminator,
                    self._iter,
                    e,
                )
        return False

    def convergence_reason(self) -> str | None:
        """The signal that caused the most recent True from is_converged().

        One of ``"max_iterations"``, ``"improvement_patience"``,
        ``"plateau_cv"``, ``"posterior_regret_bound"`` (Makarova 2022),
        ``"emmr"`` (Ishibashi 2023), or ``None`` if is_converged has never
        returned True. Stable across calls until the next is_converged()
        check.
        """
        return self._convergence_reason

    def history(self) -> list[SearchIteration]:
        return list(self._history)

    def boundary_summary(self) -> dict[str, Any] | None:
        """Boundary-summary block derived from the iteration history.

        Delegates to the shared exporter helper, so the schema stays
        identical regardless of planner type. Returns None for non-1D
        search spaces or empty history.
        Override of the ``SearchPlanner`` ABC default — saves the exporter
        a re-derivation pass when this planner is supplied.
        """
        from aiperf.exporters.search_history import _compute_boundary_summary

        if not self._history or len(self._cfg.search_space) != 1:
            return None
        return _compute_boundary_summary(self._history, self._cfg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fallback_to_tpe_after_botorch_import_error(
        self, cfg: AdaptiveSearchSweep, exc: ImportError
    ) -> AdaptiveSearchSweep:
        fallback_cfg = cfg.model_copy(
            update={
                "optuna_sampler": "tpe",
                "optuna_acquisition": None,
            }
        )
        message = (
            "Implicit optuna_sampler='botorch' could not initialize because the "
            "optional BoTorch stack is unavailable; falling back to "
            "optuna_sampler='tpe'. Install aiperf[botorch] to enable "
            "BoTorch-backed optimization. Underlying import error: "
            f"{exc}"
        )
        logger.warning("%s", message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return fallback_cfg

    def _populate_user_attrs(self, trial: Any, results: list[RunResult]) -> bool:
        """Write per-SLA averaged observations onto ``trial.user_attrs``.

        Returns True iff at least one filter was unmeasurable on this
        iteration — the caller forces ``feasible=False`` in that case to
        match the soft-penalty planner's contract. Unmeasurable filters
        warn once per (planner instance, metric tag) pair.
        """
        has_unmeasurable = False
        for sla in self._sla_filters:
            observed = averaged_metric_value(results, sla.metric_tag, sla.stat)
            # Why: averaged_metric_value already filters NaN/inf to None, but
            # we still pass the cleaned value to Optuna's user_attrs so the
            # constraints_func sees None (which it interprets as missing /
            # infeasible) instead of a non-finite scalar that would be silently
            # treated as feasible (`NaN > 0` is False).
            trial.set_user_attr(_attr_key(sla), observed)
            if observed is None:
                has_unmeasurable = True
                if _any_non_finite_observation(results, sla.metric_tag, sla.stat):
                    self._warn_nan_once(
                        f"constraint metric {sla.metric_tag!r} (stat="
                        f"{sla.stat!r}) returned non-finite value(s) on "
                        f"iteration {self._iter}; treating as missing "
                        "(infeasible) for both Optuna and history."
                    )
                if sla.metric_tag not in self._warned_unmeasurable_metrics:
                    self._warned_unmeasurable_metrics.add(sla.metric_tag)
                    logger.warning(
                        "SLA filter on metric %r (stat=%s) is unmeasurable on "
                        "iteration %d; treating as infeasible. Likely cause: "
                        "a streaming-only metric on a non-streaming endpoint, "
                        "or a typo in --search-stat. Subsequent iterations "
                        "with the same tag will not re-log.",
                        sla.metric_tag,
                        sla.stat,
                        self._iter,
                    )
        return has_unmeasurable

    def _collect_finite_values_for(
        self, results: list[RunResult], metric: str, stat: str
    ) -> tuple[list[float], int]:
        """Pull ``metric.stat`` from each successful trial.

        Returns ``(finite_values, non_finite_dropped_count)``. Skips failed
        trials, missing metrics, missing stats, and non-finite values (Optuna
        rejects non-finite objectives, so we drop them uniformly).
        """
        values: list[float] = []
        non_finite_dropped = 0
        for r in results:
            if not r.success:
                continue
            metric_obj = r.summary_metrics.get(metric)
            if metric_obj is None:
                continue
            stat_value = getattr(metric_obj, stat, None)
            if stat_value is None:
                continue
            if not is_finite_value(stat_value):
                # Non-finite poisons the GP and the constraint user_attr; drop uniformly.
                non_finite_dropped += 1
                continue
            values.append(float(stat_value))
        return values, non_finite_dropped

    def _extract_objective_vector(self, results: list[RunResult]) -> list[float] | None:
        """Project benchmark results onto the configured objective vector.

        Returns ``None`` when any objective is missing across all trials; the
        caller treats this as a failed iteration and feeds a per-direction
        sentinel into ``study.tell``. For length-1 objectives the result is a
        1-element list (preserves vector shape end-to-end so single- and
        multi-objective code paths stay identical).

        When ``cfg.objective_pooling == "pooled"`` and an objective stat is a
        percentile, the pooled-percentile path is used per objective (requires
        ``--export-level records``); falls back to the mean of finite
        per-trial values when the JSONL is missing.
        """
        if not results:
            return None
        out: list[float] = []
        for obj in self._cfg.objectives:
            if self._cfg.objective_pooling == "pooled":
                pct = percentile_from_stat(obj.stat)
                if pct is not None:
                    pooled = pooled_percentile_from_results(results, obj.metric, pct)
                    if pooled is not None:
                        out.append(float(pooled))
                        continue
                    # Fall through: helper warned about missing JSONL.
            values, non_finite_dropped = self._collect_finite_values_for(
                results, obj.metric, obj.stat
            )
            if non_finite_dropped:
                self._warn_nan_once(
                    f"dropped {non_finite_dropped} non-finite objective value(s) "
                    f"for metric={obj.metric!r} stat={obj.stat!r} on iteration "
                    f"{self._iter}; treating as missing."
                )
            if not values:
                return None
            out.append(sum(values) / len(values))
        return out

    def _objective_to_loss(self, objective: float) -> float:
        """Map objective-space value to a minimization loss for tracking.

        Used only for ``_best_loss`` / ``_iters_since_improvement``
        bookkeeping (Optuna manages its own internal objective space). For
        MAXIMIZE we negate, for MINIMIZE we pass through. Same convention as
        BayesianSearchPlanner so convergence reasons compare apples-to-apples.
        """
        if self._cfg.objectives[0].direction == OptimizationDirection.MAXIMIZE:
            return -objective
        return objective

    def _failure_sentinel_vector(self) -> list[float]:
        """Per-objective sentinel vector to tell Optuna for failed iterations.

        One sentinel per configured objective. For each objective, picks a
        finite value strictly worse than any seen so far in that objective's
        direction (worst-of-prior plus a 10%-or-1.0 margin in the worse
        direction); falls back to ``+/- NO_DATA_SENTINEL_LOSS`` when no
        prior history exists for that objective.

        Why per-objective: Optuna's multi-objective tell expects a vector;
        each direction has its own sense of "worse". Reusing the
        single-objective scalar would force the same sign on every
        direction and silently corrupt the Pareto front.
        """
        sentinels: list[float] = []
        for idx, obj in enumerate(self._cfg.objectives):
            prior = self._prior_finite_objective_values(idx)
            if not prior:
                if obj.direction == OptimizationDirection.MAXIMIZE:
                    sentinels.append(-NO_DATA_SENTINEL_LOSS)
                else:
                    sentinels.append(NO_DATA_SENTINEL_LOSS)
                continue
            if obj.direction == OptimizationDirection.MAXIMIZE:
                worst = min(prior)
                margin = max(abs(worst) * 0.1, 1.0)
                sentinels.append(worst - margin)
            else:
                worst = max(prior)
                margin = max(abs(worst) * 0.1, 1.0)
                sentinels.append(worst + margin)
        return sentinels

    def _prior_finite_objective_values(self, idx: int) -> list[float]:
        """Finite objective values from prior history at index ``idx``.

        Reads ``objective_values[idx]`` (vector form) when populated; falls
        back to ``objective_value`` for the idx==0 case so single-objective
        history entries still contribute. Scrubs non-finite to keep
        ``min``/``max`` safe.
        """
        out: list[float] = []
        for h in self._history:
            if h.objective_values is not None and idx < len(h.objective_values):
                v = h.objective_values[idx]
            elif idx == 0:
                v = h.objective_value
            else:
                v = None
            if v is None or not is_finite_value(v):
                continue
            out.append(float(v))
        return out

    def _track_improvement(
        self,
        objective_for_history: float | None,
        objective_vec: list[float] | None,
    ) -> None:
        """Update the patience-based stop counter for this iteration.

        Single-objective dispatches to scalar best-loss tracking (mirrors
        :class:`BayesianSearchPlanner`). Multi-objective dispatches to
        hypervolume-improvement tracking — the scalar plateau signal is
        meaningless across a Pareto front. Both paths feed
        ``_iters_since_improvement``, so ``improvement_patience`` semantics
        are identical from the convergence-evaluator's point of view.
        """
        if len(self._cfg.objectives) == 1:
            self._track_scalar_improvement(objective_for_history)
            return
        self._track_hypervolume_improvement(objective_vec)

    def _track_scalar_improvement(self, objective_for_history: float | None) -> None:
        """Single-objective best-loss + iters-since-improvement bookkeeping.

        A failed iteration (objective_for_history=None) is treated as
        no-improvement.
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

    def _track_hypervolume_improvement(self, objective_vec: list[float] | None) -> None:
        """Multi-objective: update plateau counter from feasible-set hypervolume.

        Skips the first ``n_initial_points`` rounds where the front is
        meaningless. Reference point comes from
        :func:`derive_reference_point` over the same observed feasible
        history. A failed iteration (``objective_vec=None``) is treated as
        no-improvement, mirroring scalar tracking.
        """
        if objective_vec is None:
            self._iters_since_improvement += 1
            return
        scored = [
            h for h in self._history if h.objective_values is not None and h.feasible
        ]
        if len(scored) < self._cfg.n_initial_points:
            return
        from aiperf.orchestrator.search_planner._optuna_helpers import (
            compute_hypervolume,
            derive_reference_point,
        )

        observed = [list(h.objective_values) for h in scored if h.objective_values]
        try:
            ref_point = derive_reference_point(self._cfg.objectives, observed=observed)
        except ValueError:
            return
        try:
            hv = compute_hypervolume(observed, self._cfg.objectives, ref_point)
        except Exception as e:
            logger.debug(
                "compute_hypervolume failed at iter %d: %r; skipping plateau update.",
                self._iter,
                e,
            )
            return
        if self._best_hypervolume is None or hv > self._best_hypervolume * (1.0 + 1e-6):
            self._best_hypervolume = hv
            self._iters_since_improvement = 0
        else:
            self._iters_since_improvement += 1

    def _populate_outcome_user_attrs(
        self, trial: Any, results: list[RunResult]
    ) -> None:
        """Write per-OutcomeConstraint averaged observations onto ``trial.user_attrs``.

        Uses the same averaging path as SLA filters (``averaged_metric_value``)
        with ``stat="avg"`` since OutcomeConstraint has no stat field. Missing
        values flow through as ``None``; ``build_outcome_constraints_func``
        translates those to ``_UNMEASURABLE_VIOLATION``.
        """
        for c in self._outcome_constraints:
            observed = averaged_metric_value(results, c.metric, "avg")
            trial.set_user_attr(f"outcome:{c.metric}", observed)

    def _maybe_install_qnehvi_candidates_func(self) -> None:
        """Bind the qNEHVI candidates_func once Sobol initial round completes.

        BoTorchSampler stores the callable as ``_candidates_func`` (private):
        ``sample_relative`` reads from ``self._candidates_func`` and lazy-defaults
        it to ``qehvi_candidates_func`` on the first post-startup call.
        Writing to a public ``candidates_func`` attribute would be a no-op —
        Optuna never reads it. Reference point is derived from the
        now-observed feasible front; the install fires exactly once per
        planner instance.
        """
        if len(self._cfg.objectives) <= 1 or self._qnehvi_installed:
            return
        sampler = getattr(self._study, "sampler", None)
        if sampler is None or not hasattr(sampler, "_candidates_func"):
            return
        scored = [
            h for h in self._history if h.objective_values is not None and h.feasible
        ]
        if len(scored) < self._cfg.n_initial_points:
            return
        from aiperf.orchestrator.search_planner._optuna_helpers import (
            build_qnehvi_candidates_func,
            derive_reference_point,
        )

        observed = [list(h.objective_values) for h in scored if h.objective_values]
        try:
            ref_point = derive_reference_point(self._cfg.objectives, observed=observed)
        except ValueError as e:
            logger.debug(
                "derive_reference_point unavailable at iter %d: %r; "
                "deferring qNEHVI install.",
                self._iter,
                e,
            )
            return
        sampler._candidates_func = build_qnehvi_candidates_func(
            reference_point=ref_point
        )
        self._qnehvi_installed = True

    def _warn_nan_once(self, detail: str) -> None:
        """Emit ``warnings.warn`` exactly once per planner instance for NaN-as-missing.

        The Optuna ask/tell loop must keep going on NaN — terminating mid-search
        is strictly worse than treating one iteration as missing. Both
        ``logger.warning`` and ``warnings.warn`` are emitted because production
        deployments may capture only one of the two streams.
        """
        if self._warned_nan:
            return
        self._warned_nan = True
        message = f"OptunaSearchPlanner: {detail}"
        logger.warning("%s", message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
