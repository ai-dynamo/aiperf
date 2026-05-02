# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in Search Recipes.

Phase 1 ships ``MaxThroughputUnderTTFTSLA`` (BO).
Phase 3 adds the grid recipes ``ConcurrencyRamp`` and ``PrefillTTFTCurve``,
each pairing a swept parameter with a post-process handler that emits a
derived artifact under ``sweep_aggregate/``.

Additional recipes (max-throughput-itl-sla, throughput-vs-latency-pareto,
isl-elasticity, saturation-curve) land in later phases.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from aiperf.common.enums import OptimizationDirection
from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension
from aiperf.search_recipes._base import (
    PostProcessSpec,
    SearchRecipe,
    SearchRecipeContext,
    SearchRecipeOutput,
    SLAFilter,
)

__all__ = [
    "ConcurrencyRamp",
    "MaxThroughputUnderTTFTSLA",
    "PrefillTTFTCurve",
]


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
        # `is False` (not `not endpoint.streaming`) so an unset (None) streaming flag
        # falls through — only an explicit --no-streaming hard-rejects.
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


def _logspace_int_steps(lo: float, hi: float, steps: int) -> list[int]:
    """Return ``steps`` log-spaced integer values in ``[lo, hi]`` (inclusive).

    Endpoints are forced into the result so callers can rely on the lowest /
    highest swept value being exactly ``lo`` / ``hi``. Duplicates from rounding
    (e.g. log-spaced 1, 1, 2, ...) are collapsed; order is ascending.
    """
    if steps < 2:
        raise ValueError(
            f"_logspace_int_steps: steps must be >= 2 (got {steps}); a single-point "
            "ramp degenerates and post-process can't compute a baseline."
        )
    if hi <= lo:
        raise ValueError(
            f"_logspace_int_steps: hi ({hi}) must be > lo ({lo}); "
            "use --isl-min/--isl-max with hi > lo."
        )
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    raw = [math.exp(log_lo + (log_hi - log_lo) * i / (steps - 1)) for i in range(steps)]
    rounded = sorted({max(int(round(v)), 1) for v in raw})
    return rounded


class ConcurrencyRamp(SearchRecipe):
    """Ramp concurrency on a log scale and detect the latency degradation knee.

    Sweeps ``phases.profiling.concurrency`` over a default 8-step log-spaced
    grid in ``[1, 1000]``; the post-process handler reports the first
    concurrency where p99 ``request_latency`` exceeds
    ``baseline * (1 + threshold)`` (default ``--degradation-threshold 0.20``,
    i.e. 20%). Streaming is NOT required (request_latency is end-to-end).

    Override the grid via ``--isl-min`` / ``--isl-max`` style overrides
    surfaced through the recipe's ``sweep_overrides``: ``concurrency_min``,
    ``concurrency_max``, ``concurrency_steps`` (no CLI flags yet — these read
    from ctx.sweep_overrides if a future flag wires them).

    Example:
        aiperf profile --search-recipe concurrency-ramp --degradation-threshold 0.20
    """

    name: ClassVar[str] = "concurrency-ramp"
    description: ClassVar[str] = (
        "Ramp concurrency log-spaced over [1, 1000] and detect the first "
        "concurrency where p99 request_latency degrades past "
        "baseline * (1 + --degradation-threshold)."
    )

    _CONCURRENCY_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _DEFAULT_LO: ClassVar[int] = 1
    _DEFAULT_HI: ClassVar[int] = 1000
    _DEFAULT_STEPS: ClassVar[int] = 8
    _DEFAULT_THRESHOLD: ClassVar[float] = 0.20

    def expand(self, ctx: SearchRecipeContext) -> SearchRecipeOutput:
        overrides = ctx.sweep_overrides
        lo = int(overrides.get("concurrency_min", self._DEFAULT_LO))
        hi = int(overrides.get("concurrency_max", self._DEFAULT_HI))
        steps = int(overrides.get("concurrency_steps", self._DEFAULT_STEPS))
        threshold = float(
            overrides.get("degradation_threshold", self._DEFAULT_THRESHOLD)
        )

        concurrency_values = _logspace_int_steps(lo, hi, steps)
        sweep_variables = {self._CONCURRENCY_PATH: concurrency_values}
        post_process = PostProcessSpec(
            handler="degradation_knee_detect",
            params={
                "threshold_pct": threshold,
                "metric_tag": "request_latency",
                "stat": "p99",
                "swept_param": self._CONCURRENCY_PATH,
            },
            output_filename="degradation_knee.json",
        )
        return SearchRecipeOutput(
            sweep_variables=sweep_variables,
            post_process=post_process,
        )


class PrefillTTFTCurve(SearchRecipe):
    """Sweep ISL at concurrency=1 and fit a TTFT vs ISL curve.

    Sweeps ``phases.profiling.synthetic_input_tokens.mean`` log-spaced over
    ``[--isl-min, --isl-max]`` (defaults 256, 32768). Concurrency is forced to
    a fixed value of 1 to isolate prefill cost from queueing effects. The
    post-process handler fits ``TTFT = a * ISL + b`` and falls back to a
    quadratic fit when ``r^2 < 0.85``.

    Streaming MUST be enabled (TTFT is streaming-only); the recipe rejects
    non-streaming configs.

    Example:
        aiperf profile --search-recipe prefill-ttft-curve --streaming \\
            --isl-min 256 --isl-max 32768
    """

    name: ClassVar[str] = "prefill-ttft-curve"
    description: ClassVar[str] = (
        "Sweep ISL log-spaced at concurrency=1; fit TTFT vs ISL with a linear "
        "regression (quadratic fallback when r^2 < 0.85)."
    )

    _CONCURRENCY_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _ISL_PATH: ClassVar[str] = "phases.profiling.synthetic_input_tokens.mean"
    _DEFAULT_ISL_MIN: ClassVar[int] = 256
    _DEFAULT_ISL_MAX: ClassVar[int] = 32768
    _DEFAULT_STEPS: ClassVar[int] = 8

    def expand(self, ctx: SearchRecipeContext) -> SearchRecipeOutput:
        endpoint = ctx.user_config.endpoint
        if endpoint is not None and endpoint.streaming is False:
            raise ValueError(
                f"recipe {self.name!r} requires --streaming (TTFT is a streaming-only "
                "metric); enable streaming on the endpoint or pick a different recipe."
            )

        overrides = ctx.sweep_overrides
        isl_min = int(overrides.get("isl_min", self._DEFAULT_ISL_MIN))
        isl_max = int(overrides.get("isl_max", self._DEFAULT_ISL_MAX))
        steps = int(overrides.get("isl_steps", self._DEFAULT_STEPS))

        isl_values = _logspace_int_steps(isl_min, isl_max, steps)
        sweep_variables: dict[str, list[Any]] = {
            self._ISL_PATH: isl_values,
            # Single-element list is interpreted as "fixed value" by expand_sweep --
            # there's only one variation along this dimension, so the cartesian
            # product collapses and concurrency stays at 1 for every ISL row.
            self._CONCURRENCY_PATH: [1],
        }
        post_process = PostProcessSpec(
            handler="ttft_curve_fit",
            params={
                "metric_tag": "time_to_first_token",
                "stat": "avg",
                "swept_param": self._ISL_PATH,
            },
            output_filename="prefill_curve.json",
        )
        return SearchRecipeOutput(
            sweep_variables=sweep_variables,
            post_process=post_process,
        )
