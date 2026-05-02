# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-process plugin handlers for grid Search Recipes.

Handlers run after :func:`SweepAnalyzer.compute` in
``aggregate_sweep_and_export``; they consume the sweep aggregate dict + the
recipe's params and emit a JSON artifact under ``sweep_aggregate/`` (filename
chosen by the recipe via :class:`PostProcessSpec.output_filename`).

Built-in handlers:

- :class:`DegradationKneeDetect` -- p99 latency knee for ``concurrency-ramp``.
- :class:`TTFTCurveFit` -- linear/quadratic TTFT vs ISL fit for
  ``prefill-ttft-curve``.
- :class:`ItlSurfaceFit` -- 2D ITL(concurrency, OSL) surface for
  ``decode-itl-curve``.

Handlers are registered under the ``search_recipe_post_process`` plugin
category and looked up by name at the hook site.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np

# Statistic name accepted by sweep-aggregate readers. Mirrors the Literal on
# ``SLAFilter.stat`` (``aiperf.config.adaptive_search``); a typo here at type-
# check time beats a silent infeasible-cell at runtime.
StatLiteral: TypeAlias = Literal["avg", "p50", "p90", "p95", "p99"]
_STAT_VALUES: tuple[str, ...] = ("avg", "p50", "p90", "p95", "p99")


def _stat_or_raise(value: Any, *, handler: str) -> StatLiteral:
    """Validate a runtime ``params['stat']`` against the allowed values.

    Handler ``params`` arrive as ``dict[str, Any]`` (recipes inject heterogeneous
    payloads); this gate narrows to the ``StatLiteral`` so ``_extract_points``
    and friends keep an honest type. Raises ``ValueError`` on unknown values
    naming the handler so a typo (``p98``) surfaces with full context.
    """
    if value in _STAT_VALUES:
        return value  # type: ignore[return-value]
    raise ValueError(
        f"{handler}: params['stat']={value!r} is not a recognized statistic; "
        f"expected one of {_STAT_VALUES}."
    )


__all__ = [
    "DegradationKneeDetect",
    "ItlSurfaceFit",
    "PostProcessHandler",
    "TTFTCurveFit",
]


@runtime_checkable
class PostProcessHandler(Protocol):
    """Handles a post-process step for a grid recipe.

    Receives the :meth:`SweepAnalyzer.compute` output dict and the recipe's
    params, and returns a dict that ``aggregate_sweep_and_export`` serializes
    as JSON to ``<sweep_aggregate>/<output_filename>``. Stateless: one instance
    is constructed at the hook site and discarded.

    Implementations register under the ``search_recipe_post_process`` plugin
    category.

    Example:
        >>> handler = DegradationKneeDetect()
        >>> handler.process(
        ...     sweep_aggregate={"per_combination_metrics": [...]},
        ...     params={"threshold_pct": 0.20, "metric_tag": "request_latency", "stat": "p99"},
        ... )  # doctest: +SKIP
        {'baseline_concurrency': 1, 'knee_concurrency': 200, ...}
    """

    name: ClassVar[str]
    description: ClassVar[str]

    def process(
        self,
        sweep_aggregate: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Process the sweep aggregate and return the artifact dict."""
        ...


def _extract_points(
    sweep_aggregate: dict[str, Any],
    *,
    swept_param: str,
    metric_tag: str,
    stat: StatLiteral,
) -> list[tuple[float, float]]:
    """Pull ``(swept_value, metric_value)`` pairs from the sweep aggregate.

    Supports both per-combination layouts produced by ``SweepAnalyzer.compute``:

    - Multi-trial path: keys are flattened ``<metric_tag>_<stat>`` and the
      block carries ``{mean, std, min, max, cv, ci_low, ci_high, unit}`` --
      we read ``mean`` (the multi-trial average of the stat).
    - Single-trial path: keys are the metric tag alone and the block carries
      a collapsed ``{mean, std=0, min, max, ...}``; here we use ``mean``
      (which equals the JsonMetricResult.avg) regardless of the requested
      ``stat``. Single-trial sweeps don't carry per-stat percentiles.

    Skips rows missing the swept-parameter key or the requested metric;
    raises ``ValueError`` when nothing is left after filtering so handlers
    fail loudly rather than emit an empty artifact silently.
    """
    rows = sweep_aggregate.get("per_combination_metrics") or []
    flat_key = f"{metric_tag}_{stat}"
    points: list[tuple[float, float]] = []
    for row in rows:
        params = row.get("parameters") or {}
        metrics = row.get("metrics") or {}
        if swept_param not in params:
            continue
        block = metrics.get(flat_key)
        if block is None or "mean" not in block:
            block = metrics.get(metric_tag)
        if block is None or "mean" not in block:
            continue
        points.append((float(params[swept_param]), float(block["mean"])))
    if not points:
        raise ValueError(
            f"post-process: sweep aggregate has no rows with parameter "
            f"{swept_param!r} and metric {metric_tag!r} (flat key {flat_key!r}); "
            f"check that the recipe swept that parameter and that the metric is "
            f"enabled (e.g. --streaming for time_to_first_token)."
        )
    points.sort(key=lambda pair: pair[0])
    return points


class DegradationKneeDetect:
    """Find the first swept-parameter value where p99 latency degrades past a threshold.

    Used by the ``concurrency-ramp`` recipe to detect the elbow of a saturation
    curve. ``baseline`` is taken from the lowest swept value (typically
    concurrency=1); the knee is the first swept value at which the p99 latency
    exceeds ``baseline * (1 + threshold_pct)``.

    Required ``params`` keys:

    - ``threshold_pct`` (float): degradation cutoff, e.g. ``0.20`` for 20%.
    - ``metric_tag`` (str): metric tag to inspect, e.g. ``"request_latency"``.
    - ``stat`` (str): statistic, e.g. ``"p99"``.
    - ``swept_param`` (str): parameter name from the sweep, e.g. ``"phases.profiling.concurrency"``.

    Returns a dict with ``baseline_*``, ``knee_*`` (``null`` when no knee found
    in the swept range), ``threshold_pct``, ``swept_metric``, ``stat``, and
    ``all_points``.

    Example:
        >>> agg = {
        ...     "per_combination_metrics": [
        ...         {"parameters": {"phases.profiling.concurrency": 1}, "metrics": {"request_latency_p99": {"mean": 10.0}}},
        ...         {"parameters": {"phases.profiling.concurrency": 100}, "metrics": {"request_latency_p99": {"mean": 13.0}}},
        ...     ]
        ... }
        >>> out = DegradationKneeDetect().process(
        ...     agg,
        ...     {"threshold_pct": 0.2, "metric_tag": "request_latency", "stat": "p99",
        ...      "swept_param": "phases.profiling.concurrency"},
        ... )
        >>> out["knee_concurrency"]
        100
    """

    name: ClassVar[str] = "degradation_knee_detect"
    description: ClassVar[str] = (
        "Find the first swept value where the chosen metric exceeds "
        "baseline * (1 + threshold_pct)."
    )

    def process(
        self,
        sweep_aggregate: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        threshold_pct = float(params["threshold_pct"])
        metric_tag = str(params["metric_tag"])
        stat = _stat_or_raise(params["stat"], handler="degradation_knee_detect")
        swept_param = str(params["swept_param"])

        points = _extract_points(
            sweep_aggregate,
            swept_param=swept_param,
            metric_tag=metric_tag,
            stat=stat,
        )
        baseline_x, baseline_y = points[0]
        cutoff = baseline_y * (1.0 + threshold_pct)
        knee_x: float | None = None
        knee_y: float | None = None
        for x, y in points[1:]:
            if y > cutoff:
                knee_x, knee_y = x, y
                break

        # Use a short alias for the swept-parameter leaf so downstream readers
        # can reference `knee_concurrency` rather than the dotted-path key.
        leaf = swept_param.split(".")[-1]
        return {
            f"baseline_{leaf}": baseline_x,
            f"baseline_{stat}": baseline_y,
            f"knee_{leaf}": knee_x,
            f"knee_{stat}": knee_y,
            "threshold_pct": threshold_pct,
            "swept_metric": metric_tag,
            "stat": stat,
            "swept_param": swept_param,
            "all_points": [{leaf: x, stat: y} for x, y in points],
        }


class TTFTCurveFit:
    """Fit TTFT vs ISL with a linear regression; fall back to quadratic on poor fit.

    Used by the ``prefill-ttft-curve`` recipe. Default form is
    ``TTFT = a*ISL + b``; when ``r^2 < 0.85`` we refit a quadratic
    ``TTFT = a*ISL^2 + b*ISL + c`` and return whichever has the higher r^2.

    Required ``params`` keys:

    - ``metric_tag`` (str): TTFT metric tag, typically ``"time_to_first_token"``.
    - ``stat`` (str): statistic, e.g. ``"avg"``.
    - ``swept_param`` (str): parameter name swept on the ISL axis, e.g.
      ``"datasets.main.prompts.isl"``.

    Optional ``params`` keys:

    - ``r_squared_floor`` (float): threshold below which we refit quadratic.
      Defaults to ``0.85``.

    Example:
        >>> handler = TTFTCurveFit()
        >>> agg = {
        ...     "per_combination_metrics": [
        ...         {"parameters": {"datasets.main.prompts.isl": 256},
        ...          "metrics": {"time_to_first_token_avg": {"mean": 12.0}}},
        ...         {"parameters": {"datasets.main.prompts.isl": 512},
        ...          "metrics": {"time_to_first_token_avg": {"mean": 24.0}}},
        ...         {"parameters": {"datasets.main.prompts.isl": 1024},
        ...          "metrics": {"time_to_first_token_avg": {"mean": 48.0}}},
        ...     ]
        ... }
        >>> out = handler.process(agg, {
        ...     "metric_tag": "time_to_first_token", "stat": "avg",
        ...     "swept_param": "datasets.main.prompts.isl",
        ... })
        >>> out["fit_form"]
        'linear'
    """

    name: ClassVar[str] = "ttft_curve_fit"
    description: ClassVar[str] = (
        "Fit TTFT vs ISL with linear regression; refit quadratic when r^2 is poor."
    )

    _R2_FLOOR_DEFAULT: ClassVar[float] = 0.85

    def process(
        self,
        sweep_aggregate: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        metric_tag = str(params["metric_tag"])
        stat = _stat_or_raise(params["stat"], handler="ttft_curve_fit")
        swept_param = str(params["swept_param"])
        r2_floor = float(params.get("r_squared_floor", self._R2_FLOOR_DEFAULT))

        points = _extract_points(
            sweep_aggregate,
            swept_param=swept_param,
            metric_tag=metric_tag,
            stat=stat,
        )
        if len(points) < 2:
            raise ValueError(
                f"ttft_curve_fit: need >= 2 sweep points to fit a curve "
                f"(got {len(points)} for {metric_tag!r}/{stat!r}); widen the "
                "recipe's ISL range or sweep more steps."
            )

        x = np.asarray([pt[0] for pt in points], dtype=float)
        y = np.asarray([pt[1] for pt in points], dtype=float)

        linear_coeffs, linear_r2 = _polyfit_with_r2(x, y, deg=1)
        fit_form = "linear"
        coefficients = linear_coeffs
        r_squared = linear_r2
        below_floor = False
        if linear_r2 < r2_floor and len(points) >= 3:
            quadratic_coeffs, quadratic_r2 = _polyfit_with_r2(x, y, deg=2)
            if quadratic_r2 > linear_r2:
                fit_form = "quadratic"
                coefficients = quadratic_coeffs
                r_squared = quadratic_r2
        if r_squared < r2_floor:
            # Either we had only 2 points (can't quadratic-refit) or the quadratic
            # was no improvement; either way the chosen fit doesn't meet the
            # configured floor. Surface it so downstream consumers can flag the
            # curve as low-confidence rather than silently trusting bad coefficients.
            below_floor = True

        return {
            "fit_form": fit_form,
            "coefficients": [float(c) for c in coefficients],
            "r_squared": float(r_squared),
            "below_floor": below_floor,
            "r_squared_floor": r2_floor,
            "raw_points": [{"isl": x_i, "ttft_ms": y_i} for x_i, y_i in points],
            "swept_metric": metric_tag,
            "stat": stat,
            "swept_param": swept_param,
        }


def _polyfit_with_r2(
    x: np.ndarray, y: np.ndarray, *, deg: int
) -> tuple[list[float], float]:
    """Fit a polynomial of degree ``deg`` and return (coefficients, r^2).

    ``coefficients`` are returned highest-degree-first to match
    :func:`numpy.polyfit` so callers can render them naturally
    (``[a, b, c]`` for ``a*x^2 + b*x + c``). ``r^2`` collapses to ``0.0`` when
    ``y`` has zero variance (a degenerate "constant" fit) so we never divide by
    zero in the residual ratio.
    """
    coeffs = np.polyfit(x, y, deg=deg)
    y_hat = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_hat) ** 2))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r_squared = 0.0 if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot)
    return list(coeffs), r_squared


def _extract_2d_points(
    sweep_aggregate: dict[str, Any],
    *,
    concurrency_param: str,
    osl_param: str,
    metric_tag: str,
    stat: StatLiteral,
) -> list[tuple[float, float, float]]:
    """Pull ``(concurrency, osl, metric_value)`` triples from the sweep aggregate.

    Mirrors :func:`_extract_points` for two swept dimensions. Tolerates both
    flat-key (``<metric_tag>_<stat>``) and tag-only blocks per sweep-aggregate
    layouts produced by single-trial vs multi-trial paths in
    :class:`SweepAnalyzer`. Skips rows missing either swept-parameter key or
    the requested metric block.

    Returns a flat list of triples sorted by ``(concurrency, osl)`` for stable
    grid construction. Raises ``ValueError`` when no rows remain after filtering
    so handlers fail loudly rather than emit an empty surface silently.
    """
    rows = sweep_aggregate.get("per_combination_metrics") or []
    flat_key = f"{metric_tag}_{stat}"
    triples: list[tuple[float, float, float]] = []
    for row in rows:
        params = row.get("parameters") or {}
        metrics = row.get("metrics") or {}
        if concurrency_param not in params or osl_param not in params:
            continue
        block = metrics.get(flat_key)
        if block is None or "mean" not in block:
            block = metrics.get(metric_tag)
        if block is None or "mean" not in block:
            continue
        triples.append(
            (
                float(params[concurrency_param]),
                float(params[osl_param]),
                float(block["mean"]),
            )
        )
    if not triples:
        raise ValueError(
            f"itl_surface_fit: sweep aggregate has no rows with parameters "
            f"{concurrency_param!r} + {osl_param!r} and metric "
            f"{metric_tag!r} (flat key {flat_key!r}); check that the recipe "
            "swept both axes and streaming was enabled."
        )
    triples.sort(key=lambda t: (t[0], t[1]))
    return triples


class ItlSurfaceFit:
    """Build a 2D ITL(concurrency, OSL) surface from a grid sweep.

    Used by the ``decode-itl-curve`` recipe. Walks
    ``per_combination_metrics`` for ``(concurrency, OSL, ITL)`` triples,
    builds an axis-aligned grid keyed by the unique sorted concurrency and
    OSL values found in the sweep, and emits ``null`` (JSON) for cells where
    no triple was measured.

    The "bilinear" surface is the as-measured grid itself; downstream
    consumers (Dynamo profiler, plotting tools) interpolate between cells.
    Genuinely missing cells stay ``null`` -- the handler refuses to invent
    values for them.

    Required ``params`` keys:

    - ``metric_tag`` (str): ITL metric tag, typically ``"inter_token_latency"``.
    - ``stat`` (str): statistic, e.g. ``"avg"``.
    - ``concurrency_param`` (str): dotted-path swept on the concurrency axis,
      e.g. ``"phases.profiling.concurrency"``.
    - ``osl_param`` (str): dotted-path swept on the OSL axis, e.g.
      ``"datasets.main.prompts.osl"``.

    Returns a dict with ``swept_metric``, ``stat``, ``swept_params``,
    ``raw_points``, and a ``surface`` block:
    ``{"concurrency_axis": [...], "osl_axis": [...], "itl_grid": [[...]]}``.
    ``itl_grid[i][j]`` is the ITL value at ``concurrency_axis[i]``,
    ``osl_axis[j]`` (or ``None`` when no triple measured).

    Example:
        >>> handler = ItlSurfaceFit()
        >>> agg = {"per_combination_metrics": [
        ...     {"parameters": {"phases.profiling.concurrency": 1,
        ...                     "datasets.main.prompts.osl": 64},
        ...      "metrics": {"inter_token_latency_avg": {"mean": 10.0}}},
        ...     {"parameters": {"phases.profiling.concurrency": 1,
        ...                     "datasets.main.prompts.osl": 256},
        ...      "metrics": {"inter_token_latency_avg": {"mean": 12.0}}},
        ... ]}
        >>> out = handler.process(agg, {
        ...     "metric_tag": "inter_token_latency", "stat": "avg",
        ...     "concurrency_param": "phases.profiling.concurrency",
        ...     "osl_param": "datasets.main.prompts.osl",
        ... })
        >>> out["surface"]["concurrency_axis"]
        [1.0]
    """

    name: ClassVar[str] = "itl_surface_fit"
    description: ClassVar[str] = (
        "Build an axis-aligned ITL(concurrency, OSL) surface from a 2D grid "
        "sweep; emit raw points + grid with nulls for unmeasured cells."
    )

    def process(
        self,
        sweep_aggregate: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        metric_tag = str(params["metric_tag"])
        stat = _stat_or_raise(params["stat"], handler="itl_surface_fit")
        concurrency_param = str(params["concurrency_param"])
        osl_param = str(params["osl_param"])

        triples = _extract_2d_points(
            sweep_aggregate,
            concurrency_param=concurrency_param,
            osl_param=osl_param,
            metric_tag=metric_tag,
            stat=stat,
        )

        # Build axes from observed unique values rather than from recipe
        # defaults so missing cells are detected (not silently filled).
        concurrency_axis = sorted({t[0] for t in triples})
        osl_axis = sorted({t[1] for t in triples})
        cell_index: dict[tuple[float, float], float] = {
            (c, o): v for c, o, v in triples
        }
        itl_grid: list[list[float | None]] = [
            [cell_index.get((c, o)) for o in osl_axis] for c in concurrency_axis
        ]

        return {
            "swept_metric": metric_tag,
            "stat": stat,
            "swept_params": [concurrency_param, osl_param],
            "raw_points": [
                {"concurrency": c, "osl": o, "itl_ms": v} for c, o, v in triples
            ],
            "surface": {
                "concurrency_axis": concurrency_axis,
                "osl_axis": osl_axis,
                "itl_grid": itl_grid,
            },
        }
