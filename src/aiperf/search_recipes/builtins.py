# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in Search Recipes.

Phase 1 ships exactly one recipe: ``MaxThroughputUnderTTFTSLA``. Additional
recipes (max-throughput-itl-sla, throughput-vs-latency-pareto, isl-elasticity,
saturation-curve) land in later phases.
"""

from __future__ import annotations

from typing import ClassVar

from aiperf.common.enums import OptimizationDirection
from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension
from aiperf.search_recipes._base import (
    SearchRecipe,
    SearchRecipeContext,
    SearchRecipeOutput,
    SLAFilter,
)

__all__ = ["MaxThroughputUnderTTFTSLA"]


class MaxThroughputUnderTTFTSLA(SearchRecipe):
    """Maximize output_token_throughput at the highest concurrency where p95 TTFT
    stays under ``--ttft-sla-ms``.

    Bayesian-optimized over ``phases.profiling.concurrency`` in [1, 1000]. The
    SLA constraint lands as an ``SLAFilter`` on ``SearchRecipeOutput.sla_filters``;
    Phase 2 wires it into ``BayesianSearchPlanner`` for lexicographic feasibility
    scoring. In Phase 1 the filter is carried through but not enforced.

    Streaming MUST be enabled on the user's config (TTFT is a streaming-only
    metric). The recipe rejects non-streaming configs at expand time.

    Example:
        aiperf profile --search-recipe max-throughput-ttft-sla --ttft-sla-ms 200
    """

    name: ClassVar[str] = "max-throughput-ttft-sla"
    description: ClassVar[str] = (
        "Maximize output_token_throughput at the highest concurrency where p95 TTFT "
        "stays under --ttft-sla-ms. Bayesian-optimized over concurrency."
    )

    _CONCURRENCY_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _CONCURRENCY_LO: ClassVar[float] = 1
    _CONCURRENCY_HI: ClassVar[float] = 1000
    _MAX_ITERATIONS: ClassVar[int] = 30
    _N_INITIAL_POINTS: ClassVar[int] = 5

    def expand(self, ctx: SearchRecipeContext) -> SearchRecipeOutput:
        threshold = ctx.sla_targets.get("ttft_sla_ms")
        if threshold is None:
            raise ValueError(
                f"recipe {self.name!r} requires --ttft-sla-ms (TTFT SLA threshold "
                "in milliseconds); pass it on the CLI alongside --search-recipe."
            )

        endpoint = ctx.user_config.endpoint
        # TTFT is a streaming-only metric; refusing non-streaming configs up front
        # avoids a confusing "unknown metric time_to_first_token" error mid-BO.
        if endpoint is not None and endpoint.streaming is False:
            raise ValueError(
                f"recipe {self.name!r} requires --streaming (TTFT is a streaming-only "
                "metric); enable streaming on the endpoint or pick a different recipe."
            )

        adaptive_search = AdaptiveSearchConfig(
            algorithm="bayes",
            search_space=[
                SearchSpaceDimension(
                    path=self._CONCURRENCY_PATH,
                    lo=self._CONCURRENCY_LO,
                    hi=self._CONCURRENCY_HI,
                    kind="int",
                ),
            ],
            objective_metric="output_token_throughput",
            objective_stat="avg",
            objective_direction=OptimizationDirection.MAXIMIZE,
            max_iterations=self._MAX_ITERATIONS,
            n_initial_points=self._N_INITIAL_POINTS,
        )
        sla_filters = [
            SLAFilter(
                metric_tag="time_to_first_token",
                stat="p95",
                op="lt",
                threshold=float(threshold),
            ),
        ]
        return SearchRecipeOutput(
            adaptive_search=adaptive_search,
            sla_filters=sla_filters,
        )
