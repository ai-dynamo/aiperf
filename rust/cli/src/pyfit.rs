// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! In-process Python numerical seam for the scipy-backed dynamic search styles.
//!
//! Feature-gated (`search-pyo3`): when enabled, the native binary embeds a
//! CPython interpreter (pyo3 `auto-initialize`) and calls the EXACT scipy
//! primitives the Python planners use, so the `smooth_isotonic` fit is byte-for-
//! byte identical rather than a fragile Rust reimplementation of PAVA + PCHIP +
//! root-solve. Everything else — the bracket ramp, phase state machine, probe
//! queue, feasibility, and the run loop — stays pure Rust.
//!
//! The embedded Python mirrors, line-for-line, the scipy calls in
//! `src/aiperf/orchestrator/search_planner/_smooth_isotonic_fit.py`
//! (`smooth_isotonic_fit` + `find_root`) and the `isotonic_regression`
//! distinct-count in `_smooth_isotonic_phases._needs_more_fit_data`. Ported
//! callsites are cited in each function's docs.

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// The embedded helper module source. Kept byte-faithful to the scipy calls in
/// `_smooth_isotonic_fit.py` so the fit/root and distinct-count match Python.
const PYFIT_SRC: &str = r#"
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, isotonic_regression

_STRICTIFY_EPS_FRAC = 1e-9


def _strictify(y_hat):
    n = len(y_hat)
    if n == 0:
        return y_hat.copy()
    span = float(y_hat[-1] - y_hat[0])
    eps = _STRICTIFY_EPS_FRAC * span if span > 0.0 else _STRICTIFY_EPS_FRAC
    bumps = np.arange(n, dtype=float) * eps
    return y_hat + bumps


def _smooth_isotonic_fit(xs, ys):
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    result = isotonic_regression(ys_arr, increasing=True)
    y_hat = result.x
    y_hat_strict = _strictify(y_hat)
    spline = PchipInterpolator(xs_arr, y_hat_strict)
    return spline


def _find_root(curve_fn, x_lo, x_hi):
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


def fit_root(xs, ys, x_lo, x_hi):
    """PAVA + PCHIP fit through (xs, ys), then the first root in [x_lo, x_hi]."""
    curve = _smooth_isotonic_fit(list(xs), list(ys))
    return _find_root(curve, float(x_lo), float(x_hi))


def isotonic_distinct(ys):
    """Count distinct round(y_hat, 9) values of the PAVA fit (_needs_more_fit_data)."""
    result = isotonic_regression(np.asarray(list(ys), dtype=float), increasing=True)
    return len({round(float(v), 9) for v in result.x})


from statistics import pstdev

_RESIDUAL_SIGMA_MULTIPLIER = 3.0
_LOCAL_WINDOW = 3
_MIN_LOCAL_POINTS = 3


def detect_cliff(xs, ys, feasible_max, infeasible_min, x_hi, precision):
    """Byte-faithful port of `_check_cliff` + `_cliff_detect.detect_cliff`.

    `xs`/`ys` are the sorted-by-x, per-x averaged (x, margin) points for the
    binding constraint. Builds the PAVA+PCHIP fit and applies the residual
    guard. `feasible_max`/`infeasible_min` may be None.
    """
    raw_points = list(zip([int(x) for x in xs], [float(y) for y in ys]))
    if len(raw_points) < 2:
        return False
    curve = _smooth_isotonic_fit([p[0] for p in raw_points], [p[1] for p in raw_points])
    if len(raw_points) < _MIN_LOCAL_POINTS:
        return False
    recent = [m for _, m in raw_points[-_LOCAL_WINDOW:]]
    sigma_local = pstdev(recent)
    if sigma_local == 0.0:
        return False
    x_last, margin_last = raw_points[-1]
    fit_last = float(curve(float(x_last)))
    residual = abs(margin_last - fit_last)
    if residual <= _RESIDUAL_SIGMA_MULTIPLIER * sigma_local:
        return False
    bracket_gap = (
        infeasible_min - feasible_max
        if feasible_max is not None and infeasible_min is not None
        else 0
    )
    return bracket_gap > precision * x_hi


def boundary_ci(margins, n_resamples=10000):
    """Bootstrap 95% CI on the mean of replicate margins (`_replicate_budget.boundary_ci`).

    NOTE: unseeded scipy.stats.bootstrap -> non-deterministic (as in Python).
    """
    from scipy.stats import bootstrap

    margins = [float(m) for m in margins]
    if len(margins) == 1:
        return (margins[0], margins[0])
    data = np.asarray(margins, dtype=float)
    result = bootstrap(
        (data,), statistic=np.mean, n_resamples=n_resamples, confidence_level=0.95
    )
    return (float(result.confidence_interval.low), float(result.confidence_interval.high))
"#;

/// Run `f` with the embedded pyfit module, importing scipy once per call. pyo3's
/// `auto-initialize` starts the interpreter on first `Python::with_gil`.
fn with_pyfit<T>(f: impl FnOnce(&Bound<'_, PyModule>) -> PyResult<T>) -> anyhow::Result<T> {
    Python::with_gil(|py| {
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new(PYFIT_SRC)?.as_c_str(),
            std::ffi::CString::new("pyfit.py")?.as_c_str(),
            std::ffi::CString::new("aiperf_pyfit")?.as_c_str(),
        )?;
        f(&module)
    })
    .map_err(|e: PyErr| {
        anyhow::anyhow!("scipy search seam failed (is scipy importable in this Python?): {e}")
    })
}

/// PAVA + PCHIP fit through `(xs, ys)`, returning the first root in `[x_lo, x_hi]`
/// or `None` if the curve does not cross zero. Byte-faithful to
/// `_smooth_isotonic_fit::smooth_isotonic_fit` + `find_root`.
pub fn fit_root(xs: &[i64], ys: &[f64], x_lo: f64, x_hi: f64) -> anyhow::Result<Option<f64>> {
    with_pyfit(|m| {
        let out = m
            .getattr("fit_root")?
            .call1((xs.to_vec(), ys.to_vec(), x_lo, x_hi))?;
        if out.is_none() {
            Ok(None)
        } else {
            Ok(Some(out.extract::<f64>()?))
        }
    })
}

/// Number of distinct `round(y_hat, 9)` values of the PAVA fit of `ys`. Mirrors
/// the distinct-count in `_smooth_isotonic_phases._needs_more_fit_data`.
pub fn isotonic_distinct(ys: &[f64]) -> anyhow::Result<usize> {
    with_pyfit(|m| {
        Ok(m.getattr("isotonic_distinct")?
            .call1((ys.to_vec(),))?
            .extract::<usize>()?)
    })
}

/// PAVA-residual cliff guard over the sorted-by-x averaged points. Byte-faithful
/// to `_smooth_isotonic_phases._check_cliff` + `_cliff_detect::detect_cliff`.
pub fn detect_cliff(
    xs: &[i64],
    ys: &[f64],
    feasible_max: Option<i64>,
    infeasible_min: Option<i64>,
    x_hi: i64,
    precision: f64,
) -> anyhow::Result<bool> {
    with_pyfit(|m| {
        Ok(m.getattr("detect_cliff")?
            .call1((
                xs.to_vec(),
                ys.to_vec(),
                feasible_max,
                infeasible_min,
                x_hi,
                precision,
            ))?
            .extract::<bool>()?)
    })
}

/// Bootstrap 95% CI on the mean of replicate margins. Wraps
/// `_replicate_budget::boundary_ci`. NOTE: unseeded scipy bootstrap →
/// non-deterministic (as in Python); the replicate branch is behaviorally, not
/// byte, verifiable.
pub fn boundary_ci(margins: &[f64]) -> anyhow::Result<(f64, f64)> {
    with_pyfit(|m| {
        Ok(m.getattr("boundary_ci")?
            .call1((margins.to_vec(),))?
            .extract::<(f64, f64)>()?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_root_crosses_zero() {
        // A monotone series crossing zero between x=3 and x=4: the PAVA+PCHIP
        // root must land in (3, 4). Matches the _smooth_isotonic_fit docstring
        // example shape.
        let xs = [1, 2, 3, 4, 5];
        let ys = [-2.0, -0.9, -1.1, 1.0, 2.0];
        let root = fit_root(&xs, &ys, 1.0, 5.0).unwrap().expect("a root");
        assert!(root > 3.0 && root < 4.0, "root {root} should be in (3,4)");
    }

    #[test]
    fn fit_root_no_crossing_is_none() {
        // All-negative margins never cross zero -> no root.
        let xs = [1, 2, 3];
        let ys = [-3.0, -2.0, -1.0];
        assert!(fit_root(&xs, &ys, 1.0, 3.0).unwrap().is_none());
    }

    #[test]
    fn isotonic_distinct_pools_violators() {
        // PAVA pools the middle violator pair (-0.9, -1.1) into one value.
        let ys = [-2.0, -0.9, -1.1, 1.0, 2.0];
        assert_eq!(isotonic_distinct(&ys).unwrap(), 4);
    }
}
