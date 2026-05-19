# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``MaxConcurrencyUnderSLA`` recipe (closes GitHub issue #883).

Re-exported from ``builtins.py``.
"""

from __future__ import annotations

from typing import ClassVar

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension
from aiperf.plugin.enums import SearchPlannerType
from aiperf.search_recipes._base import (
    PostProcessSpec,
    SearchRecipe,
    SearchRecipeContext,
    SearchRecipeOutput,
    SLAFilter,
    get_inter_token_sla_ms,
    require_streaming,
    resolve_concurrency_bounds,
)
from aiperf.search_recipes._pareto_axes import ParetoAxesSpec

__all__ = ["MaxConcurrencyUnderSLA"]


class MaxConcurrencyUnderSLA(SearchRecipe):
    """Find the highest concurrency where all SLA filters pass.

    Default style: ``smooth_isotonic`` — PAVA + PCHIP smooth-isotonic
    regression-based 1D SLA-saturation search. Drop-in replacement for
    ``monotonic``; denoises before fitting and supports replicate-based
    bootstrap CIs.

    Alternative styles (``--search-style {smooth_isotonic|monotonic|bo|optuna|grid}``):

    - ``monotonic``: 1D binary-search via the ``MonotonicSLASearchPlanner``
      (~10-20 iterations on ``[1, 1000]``). Mirrors perf_analyzer's
      ``--binary-search``.
    - ``bo``: penalty-Bayesian-Optimization with the same SLA filters and
      ``output_token_throughput`` as the maximize-objective. Useful when
      the user wants to land at the throughput-maximizing feasible point
      rather than the boundary itself.
    - ``optuna``: same penalty-BO formulation as ``bo`` but routed through
      the ``OptunaSearchPlanner`` (TPE / GP / BoTorch samplers, selected
      via ``--optuna-sampler``). Optuna ships by default; BoTorch requires
      the optional ``botorch`` extra.
    - ``grid``: log-spaced 8-step sweep + ``sla_breach_knee`` post-process,
      emitting ``sla_breach.json`` with the boundary report.

    SLA filters compose from any of ``--ttft-sla-ms``,
    ``--tpot-sla-ms`` / ``--itl-sla-ms`` (aliases for the same
    inter-token-latency SLA), ``--e2e-sla-ms``, ``--error-rate-sla``. At
    least one must be set.

    Streaming is required iff at least one filter references a streaming-only
    metric (TTFT or ITL/TPOT). ``--e2e-sla-ms`` and ``--error-rate-sla`` work
    without streaming.

    Closes GitHub issue #883.

    Example:
        aiperf profile --search-recipe max-concurrency-under-sla \\
            --ttft-sla-ms 200
    """

    name: ClassVar[str] = "max-concurrency-under-sla"
    description: ClassVar[str] = (
        "Find the highest concurrency where all SLA filters pass. Default "
        "smooth_isotonic search; use --search-style "
        "{smooth_isotonic|monotonic|bo|optuna|grid}."
    )
    pareto_axes: ClassVar[ParetoAxesSpec | None] = ParetoAxesSpec(
        x_metric="request_latency",
        x_stat="p95",
        y_metric="concurrency",  # parameter-as-axis: read from variation params
        y_stat="value",  # sentinel: see _extract_axis_value in Task 6
    )

    _CONCURRENCY_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _CONCURRENCY_SWEEP_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _CONCURRENCY_LO: ClassVar[int] = 1
    _CONCURRENCY_HI: ClassVar[int] = 1000
    _MONOTONIC_MAX_ITERATIONS: ClassVar[int] = 20
    _SMOOTH_ISOTONIC_MAX_ITERATIONS: ClassVar[int] = 30
    _BO_MAX_ITERATIONS: ClassVar[int] = 30
    _BO_INITIAL_POINTS: ClassVar[int] = 5
    _OPTUNA_MAX_ITERATIONS: ClassVar[int] = 30
    _OPTUNA_INITIAL_POINTS: ClassVar[int] = 5
    _GRID_DEFAULT_STEPS: ClassVar[int] = 8

    _STREAMING_ONLY_METRIC_TAGS: ClassVar[frozenset[str]] = frozenset(
        {"time_to_first_token", "inter_token_latency"}
    )
    _VALID_STYLES: ClassVar[tuple[str, ...]] = (
        "smooth_isotonic",
        "monotonic",
        "bo",
        "optuna",
        "grid",
    )

    def expand(self, ctx: SearchRecipeContext) -> SearchRecipeOutput:
        """Compile the max-concurrency-under-SLA recipe for the selected search style.

        Requires at least one SLA target. Streaming is required only when the selected
        filters include streaming-only metrics. ``grid`` emits a sweep plus
        ``sla_breach_knee`` post-process; other styles emit adaptive-search config.
        """
        sla_filters = self._build_sla_filters(ctx)
        if not sla_filters:
            raise ValueError(
                f"recipe {self.name!r} requires at least one of --ttft-sla-ms / "
                "--tpot-sla-ms / --itl-sla-ms / --e2e-sla-ms / --error-rate-sla; "
                "pass at least one on the CLI alongside --search-recipe."
            )

        self._check_streaming_if_required(ctx, sla_filters)

        lo, hi = resolve_concurrency_bounds(
            ctx.sweep_overrides,
            recipe_name=self.name,
            default_lo=self._CONCURRENCY_LO,
            default_hi=self._CONCURRENCY_HI,
        )

        style = ctx.sweep_overrides.get("search_style") or "smooth_isotonic"
        if style == "smooth_isotonic":
            return self._build_smooth_isotonic_output(sla_filters, lo, hi)
        if style == "monotonic":
            return self._build_monotonic_output(sla_filters, lo, hi)
        if style == "bo":
            return self._build_bo_output(sla_filters, lo, hi)
        if style == "optuna":
            return self._build_optuna_output(sla_filters, lo, hi)
        if style == "grid":
            return self._build_grid_output(sla_filters, lo, hi)
        raise ValueError(
            f"recipe {self.name!r}: unknown --search-style {style!r}; expected "
            f"one of {self._VALID_STYLES}."
        )

    def _build_sla_filters(self, ctx: SearchRecipeContext) -> list[SLAFilter]:
        filters: list[SLAFilter] = []
        ttft = ctx.sla_targets.get("ttft_sla_ms")
        if ttft is not None:
            filters.append(
                SLAFilter(
                    metric_tag="time_to_first_token",
                    stat="p95",
                    op="lt",
                    threshold=float(ttft),
                )
            )
        tpot = get_inter_token_sla_ms(ctx.sla_targets)
        if tpot is not None:
            filters.append(
                SLAFilter(
                    metric_tag="inter_token_latency",
                    stat="p95",
                    op="lt",
                    threshold=float(tpot),
                )
            )
        e2e = ctx.sla_targets.get("e2e_sla_ms")
        if e2e is not None:
            filters.append(
                SLAFilter(
                    metric_tag="request_latency",
                    stat="p99",
                    op="lt",
                    threshold=float(e2e),
                )
            )
        err = ctx.sla_targets.get("error_rate_sla")
        if err is not None:
            filters.append(
                SLAFilter(
                    metric_tag="request_error_rate",
                    stat="p99",
                    op="lt",
                    threshold=float(err),
                )
            )
        return filters

    def _check_streaming_if_required(
        self, ctx: SearchRecipeContext, sla_filters: list[SLAFilter]
    ) -> None:
        # Conditional check: only TTFT/ITL filters need streaming; an e2e or
        # error-rate-only run is fine without it.
        needs_streaming = any(
            f.metric_tag in self._STREAMING_ONLY_METRIC_TAGS for f in sla_filters
        )
        if not needs_streaming:
            return
        require_streaming(
            ctx.benchmark_config.endpoint,
            recipe_name=self.name,
            reason="SLA filters reference streaming-only metrics (TTFT, ITL/TPOT)",
        )

    def _concurrency_dim(
        self,
        lo: int,
        hi: int,
        prior: str = "uniform",
    ) -> SearchSpaceDimension:
        return SearchSpaceDimension(
            path=self._CONCURRENCY_PATH,
            lo=lo,
            hi=hi,
            kind="int",
            prior=prior,  # type: ignore[arg-type]
        )

    def _build_smooth_isotonic_output(
        self, sla_filters: list[SLAFilter], lo: int, hi: int
    ) -> SearchRecipeOutput:
        # Resolved via dynamic-enum lookup against `plugins.yaml`.
        planner_value = SearchPlannerType("smooth_isotonic")
        adaptive_search = AdaptiveSearchSweep(
            planner=planner_value,
            search_space=[self._concurrency_dim(lo, hi)],
            objectives=[
                Objective(
                    metric="output_token_throughput",
                    stat="avg",
                    direction=OptimizationDirection.MAXIMIZE,
                )
            ],
            max_iterations=self._SMOOTH_ISOTONIC_MAX_ITERATIONS,
            n_initial_points=1,
            sla_filters=sla_filters,
        )
        return SearchRecipeOutput(
            adaptive_search=adaptive_search,
            sla_filters=sla_filters,
        )

    def _build_monotonic_output(
        self, sla_filters: list[SLAFilter], lo: int, hi: int
    ) -> SearchRecipeOutput:
        adaptive_search = AdaptiveSearchSweep(
            planner=SearchPlannerType.MONOTONIC_SLA,
            search_space=[self._concurrency_dim(lo, hi)],
            objectives=[
                Objective(
                    metric="output_token_throughput",
                    stat="avg",
                    direction=OptimizationDirection.MAXIMIZE,
                )
            ],
            max_iterations=self._MONOTONIC_MAX_ITERATIONS,
            n_initial_points=1,
            sla_filters=sla_filters,
        )
        return SearchRecipeOutput(
            adaptive_search=adaptive_search,
            sla_filters=sla_filters,
        )

    def _build_bo_output(
        self, sla_filters: list[SLAFilter], lo: int, hi: int
    ) -> SearchRecipeOutput:
        # Log-uniform prior: concurrency spans 3 orders of magnitude on the
        # default [1, 1000] range. With uniform Sobol, the expected fraction
        # of initial points in the [1, 16] knee region is ~1.6%, so the BO
        # planner essentially never sees the feasible decade and latches onto
        # an infeasible-but-high-throughput surface. Log-uniform splits the
        # initial points evenly across decades — matches the grid recipe's
        # log-spaced sweep and the scaling behavior of decode-batch knees.
        adaptive_search = AdaptiveSearchSweep(
            planner=SearchPlannerType.BAYESIAN,
            search_space=[self._concurrency_dim(lo, hi, prior="log-uniform")],
            objectives=[
                Objective(
                    metric="output_token_throughput",
                    stat="avg",
                    direction=OptimizationDirection.MAXIMIZE,
                )
            ],
            max_iterations=self._BO_MAX_ITERATIONS,
            n_initial_points=self._BO_INITIAL_POINTS,
            sla_filters=sla_filters,
        )
        return SearchRecipeOutput(
            adaptive_search=adaptive_search,
            sla_filters=sla_filters,
        )

    def _build_optuna_output(
        self, sla_filters: list[SLAFilter], lo: int, hi: int
    ) -> SearchRecipeOutput:
        # Same penalty-BO formulation as ``_build_bo_output`` but routed
        # through the Optuna planner so users can pick a TPE/GP/BoTorch
        # sampler via ``--optuna-sampler``. SLA feasibility is enforced via
        # the same ``sla_filters`` list the bayesian path uses. Log-uniform
        # prior for the same reason as the BO path: even decade coverage on
        # the default [1, 1000] concurrency range.
        adaptive_search = AdaptiveSearchSweep(
            planner=SearchPlannerType.OPTUNA,
            search_space=[self._concurrency_dim(lo, hi, prior="log-uniform")],
            objectives=[
                Objective(
                    metric="output_token_throughput",
                    stat="avg",
                    direction=OptimizationDirection.MAXIMIZE,
                )
            ],
            max_iterations=self._OPTUNA_MAX_ITERATIONS,
            n_initial_points=self._OPTUNA_INITIAL_POINTS,
            sla_filters=sla_filters,
        )
        return SearchRecipeOutput(
            adaptive_search=adaptive_search,
            sla_filters=sla_filters,
        )

    def _build_grid_output(
        self, sla_filters: list[SLAFilter], lo: int, hi: int
    ) -> SearchRecipeOutput:
        # Local import: ``_logspace_int_steps`` lives in ``builtins.py`` next
        # to the other grid recipes; importing here avoids a circular import
        # at module load (builtins.py re-exports this class).
        from aiperf.search_recipes.builtins import _logspace_int_steps

        concurrency_values = _logspace_int_steps(lo, hi, self._GRID_DEFAULT_STEPS)
        sweep_parameters = {self._CONCURRENCY_SWEEP_PATH: concurrency_values}
        post_process = PostProcessSpec(
            handler="sla_breach_knee",
            params={
                "sla_filters": [f.model_dump(mode="json") for f in sla_filters],
                "swept_param": self._CONCURRENCY_SWEEP_PATH,
            },
            output_filename="sla_breach.json",
        )
        return SearchRecipeOutput(
            sweep_parameters=sweep_parameters,
            sla_filters=sla_filters,
            post_process=post_process,
        )
