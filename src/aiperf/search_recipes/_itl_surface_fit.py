# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ItlSurfaceFit`` post-process handler for the ``decode-itl-curve`` recipe.

Re-exported from :mod:`aiperf.search_recipes.post_process` so the public
import path matches the other built-in handlers.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, ClassVar

from aiperf.common.finite import is_finite_value
from aiperf.search_recipes._post_process_shared import StatLiteral, _stat_or_raise

logger = logging.getLogger(__name__)


def _extract_2d_points(
    sweep_aggregate: dict[str, Any],
    *,
    concurrency_param: str,
    osl_param: str,
    metric_tag: str,
    stat: StatLiteral,
) -> tuple[list[tuple[float, float, float]], int, int]:
    """Pull ``(concurrency, osl, metric_value)`` triples from the sweep aggregate.

    Mirrors :func:`_extract_points` for two swept dimensions. Tolerates both
    flat-key (``<metric_tag>_<stat>``) and tag-only blocks per sweep-aggregate
    layouts produced by single-trial vs multi-trial paths in
    :class:`SweepAnalyzer`. Skips rows missing either swept-parameter key or
    the requested metric block.

    Recipes pass full dotted paths (``phases.profiling.concurrency``,
    ``datasets.main.prompts.osl``) but the per-combination ``parameters`` dict
    keys on the leaf name (``concurrency``, ``osl``). Accept either form so
    the handler stays in sync with :func:`_extract_points`.

    Drops rows whose metric value is non-finite (NaN/+inf/-inf) or negative.
    Negative ITL is a sensor error: the surface represents a non-negative
    decode latency, and propagating a negative cell into the grid produces
    nonsense plots downstream. NaN/inf rows would corrupt ``orjson`` output
    silently (orjson coerces both to JSON ``null``), making them
    indistinguishable from genuinely-missing cells.

    Returns ``(triples, dropped_non_finite, dropped_negative)`` so the caller
    can surface a sentinel block when too few finite-positive rows remain.
    Raises ``ValueError`` only when the sweep aggregate had zero candidate
    rows at all (the original "missing parameters / wrong metric tag" case).
    """
    rows = sweep_aggregate.get("per_combination_metrics") or []
    flat_key = f"{metric_tag}_{stat}"
    concurrency_short = concurrency_param.rsplit(".", 1)[-1]
    osl_short = osl_param.rsplit(".", 1)[-1]
    triples: list[tuple[float, float, float]] = []
    dropped_non_finite = 0
    dropped_negative = 0
    candidate_count = 0
    for row in rows:
        params = row.get("parameters") or {}
        metrics = row.get("metrics") or {}
        if concurrency_param in params:
            concurrency_value = params[concurrency_param]
        elif concurrency_short in params:
            concurrency_value = params[concurrency_short]
        else:
            continue
        if osl_param in params:
            osl_value = params[osl_param]
        elif osl_short in params:
            osl_value = params[osl_short]
        else:
            continue
        block = metrics.get(flat_key)
        if block is None or "mean" not in block:
            block = metrics.get(metric_tag)
        if block is None or "mean" not in block:
            continue
        candidate_count += 1
        raw_value = block["mean"]
        if not is_finite_value(raw_value):
            dropped_non_finite += 1
            continue
        value = float(raw_value)
        if value < 0:
            dropped_negative += 1
            continue
        triples.append(
            (
                float(concurrency_value),
                float(osl_value),
                value,
            )
        )
    if candidate_count == 0:
        raise ValueError(
            f"itl_surface_fit: sweep aggregate has no rows with parameters "
            f"{concurrency_param!r} + {osl_param!r} and metric "
            f"{metric_tag!r} (flat key {flat_key!r}); check that the recipe "
            "swept both axes and streaming was enabled."
        )
    triples.sort(key=lambda t: (t[0], t[1]))
    return triples, dropped_non_finite, dropped_negative


class ItlSurfaceFit:
    """Build a 2D ITL(concurrency, OSL) surface from a grid sweep.

    Used by the ``decode-itl-curve`` recipe. Walks
    ``per_combination_metrics`` for ``(concurrency, OSL, ITL)`` triples,
    builds an axis-aligned grid keyed by the unique sorted concurrency and
    OSL values found in the sweep, and emits ``null`` (JSON) for cells where
    no triple was measured.

    The emitted surface is the as-measured grid itself; the resulting grid
    is consumed by downstream interpolators (Dynamo profiler, plotting
    tools), and this handler itself performs no interpolation.
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
        """Build the ITL surface artifact from per-combination sweep metrics.

        Reads ``metric_tag``, ``stat``, ``concurrency_param``, and ``osl_param`` from
        ``params``. Missing measured cells remain ``None`` in the emitted grid; this
        method does not interpolate or synthesize values.
        """
        metric_tag = str(params["metric_tag"])
        stat = _stat_or_raise(params["stat"], handler="itl_surface_fit")
        concurrency_param = str(params["concurrency_param"])
        osl_param = str(params["osl_param"])

        triples, dropped_non_finite, dropped_negative = _extract_2d_points(
            sweep_aggregate,
            concurrency_param=concurrency_param,
            osl_param=osl_param,
            metric_tag=metric_tag,
            stat=stat,
        )

        if dropped_non_finite or dropped_negative:
            _warn_dropped_rows(metric_tag, dropped_non_finite, dropped_negative)

        # Mirror :class:`TTFTCurveFit`'s sentinel pattern: when too few
        # finite-positive rows survive the filters, emit a structured failure
        # marker rather than a hollow surface so downstream consumers can
        # short-circuit with a real reason.
        if len(triples) < 2:
            return _too_few_rows_sentinel(
                triples=triples,
                metric_tag=metric_tag,
                stat=stat,
                concurrency_param=concurrency_param,
                osl_param=osl_param,
                dropped_non_finite=dropped_non_finite,
                dropped_negative=dropped_negative,
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
            "surface_fit_failed": False,
        }


def _warn_dropped_rows(
    metric_tag: str, dropped_non_finite: int, dropped_negative: int
) -> None:
    """Emit warning + log for rows quarantined by ``_extract_2d_points``.

    Split out so :meth:`ItlSurfaceFit.process` stays under the function-size
    ceiling; identical text/levels to the inline version.
    """
    warnings.warn(
        f"itl_surface_fit: dropped {dropped_non_finite} non-finite "
        f"and {dropped_negative} negative ITL row(s) for metric "
        f"{metric_tag!r}; non-finite cells would coerce to JSON null "
        "and become indistinguishable from measurement-missing cells, "
        "and negative ITL is a sensor error.",
        UserWarning,
        stacklevel=2,
    )
    logger.warning(
        "itl_surface_fit: dropped %d non-finite and %d negative ITL "
        "row(s) for metric %r before surface fit.",
        dropped_non_finite,
        dropped_negative,
        metric_tag,
    )


def _too_few_rows_sentinel(
    *,
    triples: list[tuple[float, float, float]],
    metric_tag: str,
    stat: StatLiteral,
    concurrency_param: str,
    osl_param: str,
    dropped_non_finite: int,
    dropped_negative: int,
) -> dict[str, Any]:
    """Build the structured failure artifact for the < 2 finite-rows case.

    Same shape as the success path but with empty axes, a populated
    ``error_reason``, and ``surface_fit_failed=True``.
    """
    return {
        "swept_metric": metric_tag,
        "stat": stat,
        "swept_params": [concurrency_param, osl_param],
        "raw_points": [
            {"concurrency": c, "osl": o, "itl_ms": v} for c, o, v in triples
        ],
        "surface": {
            "concurrency_axis": [],
            "osl_axis": [],
            "itl_grid": [],
        },
        "surface_fit_failed": True,
        "error_reason": (
            f"itl_surface_fit: fewer than 2 finite-positive rows "
            f"after dropping non-finite/negative ITL "
            f"(kept {len(triples)}, dropped "
            f"{dropped_non_finite} non-finite + "
            f"{dropped_negative} negative); check that streaming "
            "was enabled and that swept cells produced successful "
            "requests."
        ),
    }
