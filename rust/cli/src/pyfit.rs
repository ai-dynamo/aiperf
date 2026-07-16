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
