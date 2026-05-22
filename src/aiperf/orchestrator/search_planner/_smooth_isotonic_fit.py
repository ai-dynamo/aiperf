# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PAVA + PCHIP root-find for smooth-isotonic SLA search.

PAVA (``scipy.optimize.isotonic_regression``) denoises a noisy monotone series
into a piecewise-constant fit. PCHIP through the *denoised* points then yields
a smooth, strictly-monotone interpolant whose root recovers the SLA-saturation
boundary without being pulled by any single noisy probe.

Composition matters: PCHIP alone interpolates exactly through every input
(vLLM-PCHIP's noise-fragility); PAVA alone produces flat runs ambiguous about
where the curve actually crosses zero. PAVA -> strictify (break ties with a
range-scaled epsilon) -> PCHIP fixes both.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, isotonic_regression

__all__ = ["find_root", "smooth_isotonic_fit"]

_STRICTIFY_EPS_FRAC = 1e-9


def smooth_isotonic_fit(
    xs: list[int], ys: list[float]
) -> tuple[Callable[[float], float], list[float]]:
    """Fit a monotone-denoised smooth curve through ``(xs, ys)``.

    Pipeline: ``scipy.optimize.isotonic_regression(ys, increasing=True)``
    produces denoised values ``y_hat``; ``_strictify(y_hat)`` breaks flat runs
    by adding ``eps = 1e-9 * range(y_hat)`` so the spline stays strictly
    monotone; ``scipy.interpolate.PchipInterpolator(xs, y_hat_strict)`` yields
    the smooth interpolant.

    Returns ``(curve_fn, y_hat)`` where ``curve_fn(x)`` evaluates the PCHIP
    spline at ``x`` and ``y_hat`` is the denoised series before strictify
    (callers may use it for cliff-detection residuals).

    Example
    -------

        >>> xs = [1, 2, 3, 4, 5]
        >>> ys = [-2.0, -0.9, -1.1, 1.0, 2.0]
        >>> curve, y_hat = smooth_isotonic_fit(xs, ys)
        >>> # PAVA pools the noisy second/third points into their average.
        >>> bool(curve(3.5) > -1.0)
        True
    """
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    result = isotonic_regression(ys_arr, increasing=True)
    y_hat = result.x
    y_hat_strict = _strictify(y_hat)
    spline = PchipInterpolator(xs_arr, y_hat_strict)
    return spline, list(map(float, y_hat))


def find_root(
    curve_fn: Callable[[float], float], x_lo: float, x_hi: float
) -> float | None:
    """Solve ``curve_fn(x) == 0`` in ``[x_lo, x_hi]``; return None if no crossing.

    Fast path for ``PchipInterpolator``: uses its built-in
    ``solve(0.0, extrapolate=False)`` which returns every real root inside the
    domain in O(n). Fallback for arbitrary callables: ``scipy.optimize.brentq``
    after a sign-change check on the bracket endpoints. ``None`` means no root
    in ``[x_lo, x_hi]`` (curve doesn't cross zero or is degenerate).
    """
    solve = getattr(curve_fn, "solve", None)
    if callable(solve):
        roots = solve(0.0, extrapolate=False)
        roots_in_bracket = [
            float(r) for r in np.atleast_1d(roots) if x_lo <= float(r) <= x_hi
        ]
        if not roots_in_bracket:
            return None
        return roots_in_bracket[0]
    f_lo = curve_fn(x_lo)
    f_hi = curve_fn(x_hi)
    if f_lo == 0.0:
        return float(x_lo)
    if f_hi == 0.0:
        return float(x_hi)
    if f_lo * f_hi > 0.0:
        return None
    return float(brentq(curve_fn, x_lo, x_hi))


def _strictify(y_hat: np.ndarray) -> np.ndarray:
    """Add a strictly increasing ramp ``bumps = arange(n) * eps`` (where ``eps = 1e-9 * span(y_hat)``) to every index so PCHIP stays strict across PAVA's flat runs without perturbing strictly-monotone runs beyond floating-point noise.

    PAVA produces piecewise-constant fits: equal-valued neighbors collapse the
    PCHIP derivative to zero in the run, which makes ``solve()`` return the
    entire run as a (degenerate) root region. Adding a tiny range-scaled
    increment per index restores strict monotonicity without perturbing the
    fit beyond floating-point noise. Degenerate inputs (single point or
    all-equal) get an evenly-spaced ramp at the same eps so the spline is
    still well-defined.
    """
    n = len(y_hat)
    if n == 0:
        return y_hat.copy()
    span = float(y_hat[-1] - y_hat[0])
    eps = _STRICTIFY_EPS_FRAC * span if span > 0.0 else _STRICTIFY_EPS_FRAC
    bumps = np.arange(n, dtype=float) * eps
    return y_hat + bumps
