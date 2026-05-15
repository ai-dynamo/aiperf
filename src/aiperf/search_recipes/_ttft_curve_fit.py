# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``TTFTCurveFit`` post-process handler for the ``prefill-ttft-curve`` recipe.

Re-exported from :mod:`aiperf.search_recipes.post_process` -- consumers should
import :class:`TTFTCurveFit` from there to match the other built-in handlers.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from aiperf.search_recipes._post_process_shared import _stat_or_raise
from aiperf.search_recipes._sweep_extract import _extract_points


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


def _too_few_finite_points_sentinel(
    *,
    points: list[tuple[float, float]],
    finite_count: int,
    metric_tag: str,
    stat: str,
    swept_param: str,
    r2_floor: float,
) -> dict[str, Any]:
    """Build the structured failure artifact when < 2 finite trial points remain.

    Mirrors the "below_floor" success-path shape so downstream consumers
    don't have to special-case the failure mode -- they just see
    ``below_floor=True`` plus an ``error_reason``.
    """
    return {
        "fit_form": "linear",
        "coefficients": [],
        "r_squared": 0.0,
        "below_floor": True,
        "r_squared_floor": r2_floor,
        "error_reason": (
            f"ttft_curve_fit: fewer than 2 finite trial points after "
            f"dropping non-finite metric values "
            f"(got {finite_count} of {len(points)}); "
            "check that swept cells produced successful requests."
        ),
        "raw_points": [{"isl": x_i, "ttft_ms": y_i} for x_i, y_i in points],
        "swept_metric": metric_tag,
        "stat": stat,
        "swept_param": swept_param,
    }


class TTFTCurveFit:
    """Fit TTFT vs ISL with a linear regression; fall back to quadratic on poor fit.

    Used by the ``prefill-ttft-curve`` recipe. Default form is
    ``TTFT = a*ISL + b``; when ``r^2 < 0.85`` and at least 3 finite points
    remain, we refit a quadratic ``TTFT = a*ISL^2 + b*ISL + c`` and return
    whichever has the higher r^2. When neither fit clears the floor or the
    fit produces non-finite coefficients, ``below_floor=True`` is set on the
    result.

    Failure modes: raises ``ValueError`` when fewer than 2 raw sweep points
    are available; returns a ``below_floor=True`` sentinel block (with
    ``error_reason``) when fewer than 2 *finite* points survive non-finite
    filtering.

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
        """Fit TTFT against swept ISL, or emit a below-floor sentinel."""
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

        # Drop trial rows with non-finite x or y BEFORE polyfit. np.polyfit
        # propagates NaN/inf into every coefficient and into r^2, and the
        # below_floor check (`r_squared < r2_floor`) is False for NaN -- so a
        # single failing sweep cell silently emits a "healthy fit" with NaN
        # coefficients. Quarantine the bad rows; if too few finite points
        # remain, return below_floor=True with an error_reason instead.
        finite_mask = np.isfinite(x) & np.isfinite(y)
        if not finite_mask.all():
            x = x[finite_mask]
            y = y[finite_mask]
        if len(x) < 2:
            return _too_few_finite_points_sentinel(
                points=points,
                finite_count=int(finite_mask.sum()),
                metric_tag=metric_tag,
                stat=stat,
                swept_param=swept_param,
                r2_floor=r2_floor,
            )

        linear_coeffs, linear_r2 = _polyfit_with_r2(x, y, deg=1)
        fit_form = "linear"
        coefficients = linear_coeffs
        r_squared = linear_r2
        below_floor = False
        if linear_r2 < r2_floor and len(x) >= 3:
            quadratic_coeffs, quadratic_r2 = _polyfit_with_r2(x, y, deg=2)
            if quadratic_r2 > linear_r2:
                fit_form = "quadratic"
                coefficients = quadratic_coeffs
                r_squared = quadratic_r2
        # Defense-in-depth: even after dropping non-finite inputs, polyfit can
        # return NaN coefficients on degenerate fits (e.g. all-equal x). Treat
        # any non-finite output as below_floor so downstream consumers don't
        # trust junk coefficients.
        if (
            np.isnan(r_squared)
            or any(not math.isfinite(c) for c in coefficients)
            or r_squared < r2_floor
        ):
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
