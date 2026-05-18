# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in Search Recipes.

BO recipes (``MaxThroughputUnderTTFTSLA``, ``MaxThroughputUnderITLSLA``)
optimize ``output_token_throughput`` over concurrency under an SLA filter.

Grid recipes (``ConcurrencyRamp``, ``PrefillTTFTCurve``, ``DecodeITLCurve``)
pair a swept parameter with a post-process handler that emits a derived
artifact under ``sweep_aggregate/``.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension
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
from aiperf.search_recipes._max_concurrency_under_sla import MaxConcurrencyUnderSLA
from aiperf.search_recipes._max_goodput_under_slo import MaxGoodputUnderSLO
from aiperf.search_recipes._pareto_axes import ParetoAxesSpec
from aiperf.search_recipes._pareto_sweep import ParetoSweep

# Default dataset name used by CLI-created recipe grids.
_DEFAULT_DATASET_NAME = "main"

__all__ = [
    "ConcurrencyRamp",
    "DecodeITLCurve",
    "MaxConcurrencyUnderSLA",
    "MaxGoodputUnderSLO",
    "MaxThroughputUnderITLSLA",
    "MaxThroughputUnderTTFTSLA",
    "ParetoSweep",
    "PrefillTTFTCurve",
]


class MaxThroughputUnderTTFTSLA(SearchRecipe):
    """Maximize output_token_throughput at the highest concurrency where p95 TTFT
    stays under ``--ttft-sla-ms``.

    Bayesian-optimized over ``phases.profiling.concurrency`` in [1, 1000]
    (overridable via --concurrency-min/--concurrency-max). The
    SLA constraint lands as an ``SLAFilter`` on ``SearchRecipeOutput.sla_filters``
    and feeds ``BayesianSearchPlanner`` for lexicographic feasibility scoring.

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
    pareto_axes: ClassVar[ParetoAxesSpec | None] = ParetoAxesSpec(
        x_metric="time_to_first_token",
        x_stat="p99",
        y_metric="output_token_throughput",
        y_stat="avg",
    )

    _CONCURRENCY_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _CONCURRENCY_SWEEP_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _CONCURRENCY_LO: ClassVar[int] = 1
    _CONCURRENCY_HI: ClassVar[int] = 1000
    _MAX_ITERATIONS: ClassVar[int] = 30
    _N_INITIAL_POINTS: ClassVar[int] = 5

    def expand(self, ctx: SearchRecipeContext) -> SearchRecipeOutput:
        """Compile the TTFT-SLA throughput recipe into an adaptive-search sweep.

        Requires ``ctx.sla_targets["ttft_sla_ms"]`` and rejects explicit
        non-streaming configs. The output maximizes ``output_token_throughput`` with
        a TTFT SLA filter over the resolved concurrency bounds.
        """
        threshold = ctx.sla_targets.get("ttft_sla_ms")
        if threshold is None:
            raise ValueError(
                f"recipe {self.name!r} requires --ttft-sla-ms (TTFT SLA threshold "
                "in milliseconds); pass it on the CLI alongside --search-recipe."
            )

        endpoint = ctx.benchmark_config.endpoint
        # TTFT is a streaming-only metric; refusing non-streaming configs up
        # front avoids a confusing "unknown metric time_to_first_token" error
        # mid-BO.
        require_streaming(
            endpoint,
            recipe_name=self.name,
            reason="TTFT is a streaming-only metric",
        )

        lo, hi = resolve_concurrency_bounds(
            ctx.sweep_overrides,
            recipe_name=self.name,
            default_lo=self._CONCURRENCY_LO,
            default_hi=self._CONCURRENCY_HI,
        )

        adaptive_search = AdaptiveSearchSweep(
            search_space=[
                SearchSpaceDimension(
                    path=self._CONCURRENCY_PATH,
                    lo=lo,
                    hi=hi,
                    kind="int",
                    prior="log-uniform",
                ),
            ],
            objectives=[
                Objective(
                    metric="output_token_throughput",
                    stat="avg",
                    direction=OptimizationDirection.MAXIMIZE,
                )
            ],
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
    if lo <= 0:
        raise ValueError(
            f"_logspace_int_steps: lo must be > 0 (got {lo}); log-spaced "
            "ramps require a positive lower bound. The CLI Pydantic field "
            "(--concurrency-min ge=1) defends user-facing callers; "
            "programmatic recipe authors must pass a positive value too."
        )
    if hi <= lo:
        raise ValueError(
            f"_logspace_int_steps: hi ({hi}) must be > lo ({lo}); "
            "use --isl-min/--isl-max with hi > lo."
        )
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    raw = [math.exp(log_lo + (log_hi - log_lo) * i / (steps - 1)) for i in range(steps)]
    # ``max(..., 1)`` defends against ``lo < 1`` (e.g. a recipe author later
    # passing 0.5 for fractional concurrency); for the current callers (lo>=1)
    # this is always a no-op since round of a positive log-spaced value is >=1.
    rounded = sorted({max(int(round(v)), 1) for v in raw})
    return rounded


class ConcurrencyRamp(SearchRecipe):
    """Ramp concurrency on a log scale and detect the latency degradation knee.

    Sweeps ``phases.profiling.concurrency`` over a default 8-step log-spaced
    grid in ``[1, 1000]``; the post-process handler reports the first
    concurrency where the chosen metric / stat exceeds
    ``baseline * (1 + threshold)`` (default ``--degradation-threshold 0.20``,
    i.e. 20%). Streaming is NOT required (``request_latency`` is end-to-end);
    selecting a streaming-only metric like ``time_to_first_token`` via
    ``--degradation-metric-tag`` does require streaming on the user config.

    Override the grid via ``--concurrency-min`` / ``--concurrency-max`` /
    ``--concurrency-steps``. Override the post-process knee detection via
    ``--degradation-metric-tag`` (default ``request_latency``) and
    ``--degradation-stat`` (default ``p99``); both flow through the recipe's
    ``sweep_overrides`` and land in the
    ``degradation_knee_detect`` PostProcessSpec params.

    Example:
        aiperf profile --search-recipe concurrency-ramp --degradation-threshold 0.20
        aiperf profile --search-recipe concurrency-ramp \\
            --degradation-metric-tag time_to_first_token --degradation-stat p95
    """

    name: ClassVar[str] = "concurrency-ramp"
    description: ClassVar[str] = (
        "Ramp concurrency log-spaced over [1, 1000] (overridable via "
        "--concurrency-min/--concurrency-max) and detect the first "
        "concurrency where p99 request_latency degrades past "
        "baseline * (1 + --degradation-threshold)."
    )
    pareto_axes: ClassVar[ParetoAxesSpec | None] = ParetoAxesSpec(
        x_metric="request_latency",
        x_stat="p95",
        y_metric="output_token_throughput",
        y_stat="avg",
    )
    auto_plot_default: ClassVar[bool] = True

    _CONCURRENCY_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _CONCURRENCY_SWEEP_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _DEFAULT_LO: ClassVar[int] = 1
    _DEFAULT_HI: ClassVar[int] = 1000
    _DEFAULT_STEPS: ClassVar[int] = 8
    _DEFAULT_THRESHOLD: ClassVar[float] = 0.20
    _DEFAULT_METRIC_TAG: ClassVar[str] = "request_latency"
    _DEFAULT_STAT: ClassVar[str] = "p99"

    def expand(self, ctx: SearchRecipeContext) -> SearchRecipeOutput:
        """Compile the concurrency-ramp recipe into a grid sweep plus knee detector.

        Consumes concurrency and degradation override keys from ``ctx.sweep_overrides``.
        The output sweeps profiling concurrency and attaches a
        ``degradation_knee_detect`` post-process spec for the aggregate artifact.
        """
        overrides = ctx.sweep_overrides
        lo, hi = resolve_concurrency_bounds(
            overrides,
            recipe_name=self.name,
            default_lo=self._DEFAULT_LO,
            default_hi=self._DEFAULT_HI,
        )
        steps = int(overrides.get("concurrency_steps", self._DEFAULT_STEPS))
        threshold = float(
            overrides.get("degradation_threshold", self._DEFAULT_THRESHOLD)
        )
        metric_tag = str(
            overrides.get("degradation_metric_tag", self._DEFAULT_METRIC_TAG)
        )
        stat = str(overrides.get("degradation_stat", self._DEFAULT_STAT))

        concurrency_values = _logspace_int_steps(lo, hi, steps)
        sweep_parameters = {self._CONCURRENCY_SWEEP_PATH: concurrency_values}
        post_process = PostProcessSpec(
            handler="degradation_knee_detect",
            params={
                "threshold_pct": threshold,
                "metric_tag": metric_tag,
                "stat": stat,
                "swept_param": self._CONCURRENCY_SWEEP_PATH,
            },
            output_filename="degradation_knee.json",
        )
        return SearchRecipeOutput(
            sweep_parameters=sweep_parameters,
            post_process=post_process,
        )


class PrefillTTFTCurve(SearchRecipe):
    """Sweep ISL at concurrency=1 and fit a TTFT vs ISL curve.

    Sweeps ``datasets.main.prompts.isl`` log-spaced over
    ``[--isl-min, --isl-max]`` (defaults 256, 32768). Concurrency is forced to
    a fixed value of 1 to isolate prefill cost from queueing effects. The
    post-process handler fits ``TTFT = a * ISL + b`` and falls back to a
    quadratic fit when ``r^2 < 0.85``.

    The dataset key is the CLI recipe default dataset name (see
    ``_DEFAULT_DATASET_NAME``); recipe grids target the body-rooted path
    ``datasets.<name>``.

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
    auto_plot_default: ClassVar[bool] = True

    _CONCURRENCY_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _CONCURRENCY_SWEEP_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _ISL_PATH: ClassVar[str] = f"datasets.{_DEFAULT_DATASET_NAME}.prompts.isl"
    _DEFAULT_ISL_MIN: ClassVar[int] = 256
    _DEFAULT_ISL_MAX: ClassVar[int] = 32768
    _DEFAULT_STEPS: ClassVar[int] = 8

    def expand(self, ctx: SearchRecipeContext) -> SearchRecipeOutput:
        """Compile the prefill TTFT curve recipe into an ISL grid sweep.

        Requires streaming, fixes profiling concurrency to 1, sweeps
        ``datasets.<default>.prompts.isl``, and attaches ``ttft_curve_fit`` so the
        aggregate export includes the fitted TTFT curve artifact.
        """
        endpoint = ctx.benchmark_config.endpoint
        require_streaming(
            endpoint,
            recipe_name=self.name,
            reason="TTFT is a streaming-only metric",
        )

        overrides = ctx.sweep_overrides
        isl_min = int(overrides.get("isl_min", self._DEFAULT_ISL_MIN))
        isl_max = int(overrides.get("isl_max", self._DEFAULT_ISL_MAX))
        steps = int(overrides.get("isl_steps", self._DEFAULT_STEPS))

        isl_values = _logspace_int_steps(isl_min, isl_max, steps)
        sweep_parameters: dict[str, list[Any]] = {
            self._ISL_PATH: isl_values,
            # Single-element list is interpreted as "fixed value" by expand_sweep --
            # there's only one variation along this dimension, so the cartesian
            # product collapses and concurrency stays at 1 for every ISL row.
            self._CONCURRENCY_SWEEP_PATH: [1],
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
            sweep_parameters=sweep_parameters,
            post_process=post_process,
        )


class MaxThroughputUnderITLSLA(SearchRecipe):
    """Maximize output_token_throughput at the highest concurrency where p95 ITL
    stays under ``--itl-sla-ms``.

    Bayesian-optimized over ``phases.profiling.concurrency`` in [1, 1000]
    (overridable via --concurrency-min/--concurrency-max); the
    SLA constraint lands as an ``SLAFilter`` on
    ``SearchRecipeOutput.sla_filters`` and feeds ``BayesianSearchPlanner`` for
    lexicographic feasibility scoring (same wiring as the TTFT-SLA twin).

    Streaming MUST be enabled (ITL is a streaming-only metric); the recipe
    rejects non-streaming configs at expand time.

    Accepts ``--itl-sla-ms`` or its alias ``--tpot-sla-ms`` (same underlying
    inter-token-latency SLA). Passing both with different values raises.

    Example:
        aiperf profile --search-recipe max-throughput-itl-sla --itl-sla-ms 50
    """

    name: ClassVar[str] = "max-throughput-itl-sla"
    description: ClassVar[str] = (
        "Maximize output_token_throughput at the highest concurrency where p95 ITL "
        "stays under --itl-sla-ms (alias --tpot-sla-ms). Bayesian-optimized over "
        "concurrency."
    )
    pareto_axes: ClassVar[ParetoAxesSpec | None] = ParetoAxesSpec(
        x_metric="inter_token_latency",
        x_stat="p99",
        y_metric="output_token_throughput",
        y_stat="avg",
    )

    _CONCURRENCY_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _CONCURRENCY_SWEEP_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _CONCURRENCY_LO: ClassVar[int] = 1
    _CONCURRENCY_HI: ClassVar[int] = 1000
    _MAX_ITERATIONS: ClassVar[int] = 30
    _N_INITIAL_POINTS: ClassVar[int] = 5

    def expand(self, ctx: SearchRecipeContext) -> SearchRecipeOutput:
        """Compile the ITL/TPOT-SLA throughput recipe into an adaptive-search sweep.

        Accepts either ``itl_sla_ms`` or ``tpot_sla_ms`` from ``ctx.sla_targets`` and
        rejects conflicting values. Requires streaming and maximizes
        ``output_token_throughput`` under the resolved inter-token-latency SLA filter.
        """
        threshold = get_inter_token_sla_ms(ctx.sla_targets)
        if threshold is None:
            raise ValueError(
                f"recipe {self.name!r} requires --itl-sla-ms (or its alias "
                "--tpot-sla-ms); pass one on the CLI alongside --search-recipe."
            )

        endpoint = ctx.benchmark_config.endpoint
        # ITL is a streaming-only metric (per-token timing emerges from SSE
        # chunks); refusing non-streaming configs here avoids a late "unknown
        # metric inter_token_latency" error mid-BO.
        require_streaming(
            endpoint,
            recipe_name=self.name,
            reason="ITL is a streaming-only metric",
        )

        lo, hi = resolve_concurrency_bounds(
            ctx.sweep_overrides,
            recipe_name=self.name,
            default_lo=self._CONCURRENCY_LO,
            default_hi=self._CONCURRENCY_HI,
        )

        adaptive_search = AdaptiveSearchSweep(
            search_space=[
                SearchSpaceDimension(
                    path=self._CONCURRENCY_PATH,
                    lo=lo,
                    hi=hi,
                    kind="int",
                    prior="log-uniform",
                ),
            ],
            objectives=[
                Objective(
                    metric="output_token_throughput",
                    stat="avg",
                    direction=OptimizationDirection.MAXIMIZE,
                )
            ],
            max_iterations=self._MAX_ITERATIONS,
            n_initial_points=self._N_INITIAL_POINTS,
        )
        sla_filters = [
            SLAFilter(
                metric_tag="inter_token_latency",
                stat="p95",
                op="lt",
                threshold=float(threshold),
            ),
        ]
        return SearchRecipeOutput(
            adaptive_search=adaptive_search,
            sla_filters=sla_filters,
        )


class DecodeITLCurve(SearchRecipe):
    """Sweep concurrency x OSL grid and fit an ITL surface.

    Sweeps ``phases.profiling.concurrency`` (6 log-spaced points in [1, 200])
    against ``datasets.main.prompts.osl`` (4 log-spaced
    points in [64, 1024]); ``itl_surface_fit`` post-process emits
    ``decode_itl_surface.json`` with raw points and a bilinear-grid surface.

    The dataset key is the CLI recipe default dataset name (see
    ``_DEFAULT_DATASET_NAME``); recipe grids target the body-rooted path
    ``datasets.<name>``.

    Override the grid via ``ctx.sweep_overrides`` keys
    ``concurrency_min`` / ``concurrency_max`` / ``concurrency_steps`` /
    ``osl_min`` / ``osl_max`` / ``osl_steps``. Streaming MUST be enabled.

    Example:
        aiperf profile --search-recipe decode-itl-curve --streaming
    """

    name: ClassVar[str] = "decode-itl-curve"
    description: ClassVar[str] = (
        "Sweep concurrency x OSL grid; fit ITL surface (bilinear) and emit "
        "decode_itl_surface.json with raw points."
    )
    auto_plot_default: ClassVar[bool] = True

    _CONCURRENCY_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _CONCURRENCY_SWEEP_PATH: ClassVar[str] = "phases.profiling.concurrency"
    _OSL_PATH: ClassVar[str] = f"datasets.{_DEFAULT_DATASET_NAME}.prompts.osl"
    _DEFAULT_CONCURRENCY_MIN: ClassVar[int] = 1
    _DEFAULT_CONCURRENCY_MAX: ClassVar[int] = 200
    _DEFAULT_CONCURRENCY_STEPS: ClassVar[int] = 6
    _DEFAULT_OSL_MIN: ClassVar[int] = 64
    _DEFAULT_OSL_MAX: ClassVar[int] = 1024
    _DEFAULT_OSL_STEPS: ClassVar[int] = 4

    def expand(self, ctx: SearchRecipeContext) -> SearchRecipeOutput:
        """Compile the decode ITL curve recipe into a concurrency-by-OSL grid sweep.

        Requires streaming, sweeps profiling concurrency against
        ``datasets.<default>.prompts.osl``, and attaches ``itl_surface_fit`` so the
        aggregate export includes the raw measured ITL surface.
        """
        endpoint = ctx.benchmark_config.endpoint
        require_streaming(
            endpoint,
            recipe_name=self.name,
            reason="ITL is a streaming-only metric",
        )

        overrides = ctx.sweep_overrides
        c_lo, c_hi = resolve_concurrency_bounds(
            overrides,
            recipe_name=self.name,
            default_lo=self._DEFAULT_CONCURRENCY_MIN,
            default_hi=self._DEFAULT_CONCURRENCY_MAX,
        )
        c_steps = int(
            overrides.get("concurrency_steps", self._DEFAULT_CONCURRENCY_STEPS)
        )
        o_lo = int(overrides.get("osl_min", self._DEFAULT_OSL_MIN))
        o_hi = int(overrides.get("osl_max", self._DEFAULT_OSL_MAX))
        o_steps = int(overrides.get("osl_steps", self._DEFAULT_OSL_STEPS))

        concurrency_values = _logspace_int_steps(c_lo, c_hi, c_steps)
        osl_values = _logspace_int_steps(o_lo, o_hi, o_steps)
        sweep_parameters: dict[str, list[Any]] = {
            self._CONCURRENCY_SWEEP_PATH: concurrency_values,
            self._OSL_PATH: osl_values,
        }
        post_process = PostProcessSpec(
            handler="itl_surface_fit",
            params={
                "metric_tag": "inter_token_latency",
                "stat": "avg",
                "concurrency_param": self._CONCURRENCY_SWEEP_PATH,
                "osl_param": self._OSL_PATH,
            },
            output_filename="decode_itl_surface.json",
        )
        return SearchRecipeOutput(
            sweep_parameters=sweep_parameters,
            post_process=post_process,
        )
