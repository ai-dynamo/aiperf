# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``MaxGoodputUnderSLO`` recipe (canonical DistServe goodput formulation).

Closes the canonical-formulation gap to GitHub issue #883: maximizes goodput
under simultaneous TTFT + TPOT + E2E per-request SLOs with a configurable
attainment fraction (default 0.95). Re-exported from ``builtins.py``.
"""

from __future__ import annotations

from typing import ClassVar

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension
from aiperf.search_recipes._base import (
    SearchRecipe,
    SearchRecipeContext,
    SearchRecipeOutput,
    SLAFilter,
    get_inter_token_sla_ms,
    require_streaming,
    resolve_concurrency_bounds,
)
from aiperf.search_recipes._pareto_axes import ParetoAxesSpec

__all__ = ["MaxGoodputUnderSLO"]


class MaxGoodputUnderSLO(SearchRecipe):
    """Maximize goodput where >=X% of requests satisfy TTFT/TPOT/E2E SLOs.

    Matches DistServe's canonical goodput formulation: the maximum request
    rate at which at least ``--slo-attainment-fraction`` of requests have
    TTFT <= tau_TTFT AND TPOT <= tau_TPOT AND E2E <= tau_E2E simultaneously.
    See https://arxiv.org/pdf/2401.09670 and the
    https://hao-ai-lab.github.io/blogs/distserve writeup.

    Bayesian-optimized over ``phases.profiling.concurrency`` in [1, 1000].

    The three SLO thresholds become per-request ``slos`` on the v2 config so
    ``GoodRequestCountMetric`` counts a request as good iff it meets all
    three. The attainment fraction lands as an ``SLAFilter`` on
    ``good_request_fraction`` (good / attempted, where attempted =
    request_count + error_request_count) so BO scoring
    treats configurations below the fraction as infeasible.

    Streaming MUST be enabled (TPOT is a streaming-only metric); the recipe
    rejects non-streaming configs at expand time.

    All three SLA-ms flags are required (``--tpot-sla-ms`` accepts
    ``--itl-sla-ms`` as an alias); ``--slo-attainment-fraction`` defaults
    to ``0.95`` (DistServe convention) and is bounded in (0, 1].

    Closes the canonical-formulation gap to GitHub issue #883.

    Example:
        aiperf profile --search-recipe max-goodput-under-slo \\
            --ttft-sla-ms 500 --tpot-sla-ms 15 --e2e-sla-ms 2000 \\
            --slo-attainment-fraction 0.95
    """

    name: ClassVar[str] = "max-goodput-under-slo"
    description: ClassVar[str] = (
        "Maximize goodput at the highest concurrency where >=X% of requests "
        "satisfy TTFT/TPOT/E2E SLOs. Bayesian-optimized; canonical DistServe "
        "formulation."
    )
    pareto_axes: ClassVar[ParetoAxesSpec | None] = ParetoAxesSpec(
        x_metric="request_latency",
        x_stat="p95",
        y_metric="goodput",
        y_stat="avg",
    )

    _CONCURRENCY_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _CONCURRENCY_SWEEP_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _CONCURRENCY_LO: ClassVar[int] = 1
    _CONCURRENCY_HI: ClassVar[int] = 1000
    _MAX_ITERATIONS: ClassVar[int] = 30
    _N_INITIAL_POINTS: ClassVar[int] = 5
    _DEFAULT_ATTAINMENT_FRACTION: ClassVar[float] = 0.95

    def _resolve_required_slos(
        self, ctx: SearchRecipeContext
    ) -> tuple[float, float, float]:
        """Pull the three required SLO targets off ctx; raise on any missing."""
        ttft_ms = ctx.sla_targets.get("ttft_sla_ms")
        tpot_ms = get_inter_token_sla_ms(ctx.sla_targets)
        e2e_ms = ctx.sla_targets.get("e2e_sla_ms")
        missing = [
            flag
            for flag, value in (
                ("--ttft-sla-ms", ttft_ms),
                ("--tpot-sla-ms", tpot_ms),
                ("--e2e-sla-ms", e2e_ms),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"recipe {self.name!r} requires {', '.join(missing)}; all three "
                "(--ttft-sla-ms, --tpot-sla-ms / --itl-sla-ms, --e2e-sla-ms) "
                "define what 'good' means per request for the goodput formula. "
                "Pass them on the CLI alongside "
                "--search-recipe max-goodput-under-slo."
            )
        return float(ttft_ms), float(tpot_ms), float(e2e_ms)

    def expand(self, ctx: SearchRecipeContext) -> SearchRecipeOutput:
        """Compile the DistServe-style goodput recipe into an adaptive-search sweep.

        Requires TTFT, TPOT/ITL, and E2E SLO thresholds plus streaming. The output
        installs per-request SLOs and constrains ``good_request_fraction`` to the
        configured attainment fraction while maximizing goodput.
        """
        ttft_ms, tpot_ms, e2e_ms = self._resolve_required_slos(ctx)

        attainment = ctx.sla_targets.get("slo_attainment_fraction")
        if attainment is None:
            attainment = self._DEFAULT_ATTAINMENT_FRACTION

        endpoint = ctx.benchmark_config.endpoint
        require_streaming(
            endpoint,
            recipe_name=self.name,
            reason="TPOT is a streaming-only metric",
        )

        lo, hi = resolve_concurrency_bounds(
            ctx.sweep_overrides,
            recipe_name=self.name,
            default_lo=self._CONCURRENCY_LO,
            default_hi=self._CONCURRENCY_HI,
        )

        # Per-request SLO thresholds keyed by metric tag. Consumed by
        # `GoodRequestCountMetric.set_slos()` to mark each request good/bad.
        slos = {
            "time_to_first_token": ttft_ms,
            "inter_token_latency": tpot_ms,
            "request_latency": e2e_ms,
        }

        # Attainment-fraction filter: BO treats configurations below this
        # threshold as infeasible. `good_request_fraction` is a derived
        # metric (good_request_count / (request_count + error_request_count))
        # that lives alongside the existing `goodput` rate metric.
        sla_filters = [
            SLAFilter(
                metric_tag="good_request_fraction",
                stat="avg",
                op="ge",
                threshold=float(attainment),
            ),
        ]

        adaptive_search = AdaptiveSearchSweep(
            search_space=[
                SearchSpaceDimension(
                    path=self._CONCURRENCY_PATH,
                    lo=lo,
                    hi=hi,
                    kind="int",
                    # Log-uniform: concurrency [1, 1000] spans 3 decades, so
                    # uniform Sobol sampling concentrates initial points at the
                    # high end where goodput typically collapses. Log-uniform
                    # mirrors the grid recipe and the decade-scale knee
                    # behavior — see _max_concurrency_under_sla._build_bo_output.
                    prior="log-uniform",
                ),
            ],
            objectives=[
                Objective(
                    metric="goodput",
                    stat="avg",
                    direction=OptimizationDirection.MAXIMIZE,
                )
            ],
            max_iterations=self._MAX_ITERATIONS,
            n_initial_points=self._N_INITIAL_POINTS,
        )
        return SearchRecipeOutput(
            adaptive_search=adaptive_search,
            sla_filters=sla_filters,
            slos=slos,
        )
