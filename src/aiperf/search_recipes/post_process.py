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
- :class:`ParetoSweepExport` -- Pareto-frontier export for ``pareto-sweep``.
- :class:`SLABreachKnee` -- first-breach detection for SLA-bound ramps.

Handlers are registered under the ``search_recipe_post_process`` plugin
category and looked up by name at the hook site. Several classes live in
sibling ``_<name>.py`` modules and are re-exported here so the public import
path matches the other built-in handlers and ``plugins.yaml`` resolves them
directly.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

from aiperf.common.finite import is_finite_value
from aiperf.search_recipes._itl_surface_fit import ItlSurfaceFit
from aiperf.search_recipes._pareto_sweep_export import ParetoSweepExport
from aiperf.search_recipes._post_process_shared import _stat_or_raise
from aiperf.search_recipes._sla_breach_knee import SLABreachKnee
from aiperf.search_recipes._sweep_extract import _extract_points
from aiperf.search_recipes._ttft_curve_fit import TTFTCurveFit

__all__ = [
    "DegradationKneeDetect",
    "ItlSurfaceFit",
    "ParetoSweepExport",
    "PostProcessHandler",
    "SLABreachKnee",
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
        """Transform a sweep aggregate into one JSON-serializable artifact payload.

        Implementations read ``SweepAnalyzer.compute`` output plus handler-specific
        ``params`` and return the exact dict that the aggregation hook writes under
        ``sweep_aggregate/<output_filename>``.
        """
        ...


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
        """Return the degradation-knee artifact, preserving all extracted points."""
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
        # Why: `< 0` alone misses NaN (`nan < 0` is False) and +inf. NaN/inf
        # baseline silently produces `cutoff = baseline_y * (1 + threshold)`
        # which is also NaN/inf, every `y > cutoff` is False, and the handler
        # returns `knee=None` indistinguishable from a robust system. Reject
        # both up front with a loud ValueError so the caller sees the data is
        # junk, not that the system survived the sweep.
        if not is_finite_value(baseline_y):
            raise ValueError(
                f"degradation_knee_detect: baseline {stat} for {metric_tag!r} "
                f"is non-finite ({baseline_y!r}); a NaN/inf baseline collapses "
                "the cutoff comparison to False so the handler can't tell "
                "'no knee in range' from 'data is junk'. Check the exporter "
                "and the lowest-swept-value trial."
            )
        if baseline_y <= 0:
            raise ValueError(
                f"degradation_knee_detect: baseline {stat} for {metric_tag!r} "
                f"must be positive (got {baseline_y}, which is zero or "
                "negative); a zero baseline produces cutoff=0 so any positive "
                "value is an 'infinite' degradation, and a negative baseline "
                "flips the cutoff sign so 'degradation' becomes meaningless. "
                "Latency metrics must be strictly positive."
            )
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
        # Coerce concurrency-like swept axes back to int for stable JSON
        # output: _extract_points reads via float() to tolerate "256.0"-style
        # serialization, but emitting `1.0` / `100.0` for an integer-valued
        # axis (the canonical case) is a type degradation.
        baseline_x_out = (
            int(baseline_x) if baseline_x == int(baseline_x) else baseline_x
        )
        knee_x_out: int | float | None
        if knee_x is None:
            knee_x_out = None
        else:
            knee_x_out = int(knee_x) if knee_x == int(knee_x) else knee_x
        return {
            f"baseline_{leaf}": baseline_x_out,
            f"baseline_{stat}": baseline_y,
            f"knee_{leaf}": knee_x_out,
            f"knee_{stat}": knee_y,
            "threshold_pct": threshold_pct,
            "swept_metric": metric_tag,
            "stat": stat,
            "swept_param": swept_param,
            "all_points": [
                {
                    leaf: int(x) if x == int(x) else x,
                    stat: y,
                }
                for x, y in points
            ],
        }
