// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The `ttft_curve_fit` post-process handler for the `prefill-ttft-curve` recipe.
//!
//! Pure-Rust port of `aiperf.search_recipes._ttft_curve_fit::TTFTCurveFit.process`
//! (`src/aiperf/search_recipes/_ttft_curve_fit.py:75-208`): fit `TTFT = a*ISL + b`
//! with an ordinary-least-squares linear regression, and — when the linear
//! `r^2 < 0.85` and at least 3 finite points remain — refit a quadratic
//! `a*ISL^2 + b*ISL + c`, keeping whichever has the higher `r^2`. Written as
//! `prefill_curve.json`.
//!
//! `numpy.polyfit` solves a column-scaled Vandermonde least-squares via SVD; this
//! port solves the equivalent normal equations (deg 1 closed form; deg 2 via a
//! 3x3 Gaussian-elimination solve). For a full-rank system the least-squares
//! solution is unique, so the coefficients and `r^2` are numerically equal to
//! numpy's to floating-point rounding — not guaranteed byte-identical (same
//! caveat as the scipy-backed planner fits), but exact in algorithm.

use serde_json::{Map, Value};

use crate::search::extract::{extract_points, num};

/// Default `r^2` floor below which the linear fit refits quadratic (`_R2_FLOOR_DEFAULT`).
const R2_FLOOR_DEFAULT: f64 = 0.85;

/// Resolved params for the `ttft_curve_fit` handler.
pub struct CurveSpec {
    /// TTFT metric tag (`time_to_first_token`).
    pub metric_tag: String,
    /// Statistic label (`avg`).
    pub stat: String,
    /// Dotted swept ISL path (`datasets.main.prompts.isl`).
    pub swept_param: String,
}

/// Ordinary-least-squares polynomial fit of degree `deg` (1 or 2), returning the
/// coefficients highest-degree-first (matching `numpy.polyfit`) plus `r^2`.
/// `r^2` collapses to `0.0` when `y` has zero variance (degenerate constant fit).
fn polyfit_with_r2(x: &[f64], y: &[f64], deg: usize) -> Option<(Vec<f64>, f64)> {
    let n = x.len();
    if n <= deg {
        return None;
    }
    let ncols = deg + 1;
    // Normal equations A^T A c = A^T y, with A's columns [x^deg, ..., x, 1]
    // (highest power first, matching numpy's Vandermonde column order).
    let mut ata = vec![vec![0.0f64; ncols]; ncols];
    let mut atb = vec![0.0f64; ncols];
    for i in 0..n {
        // powers[k] = x^(deg - k)
        let mut powers = vec![0.0f64; ncols];
        for (k, p) in powers.iter_mut().enumerate() {
            *p = x[i].powi((deg - k) as i32);
        }
        for r in 0..ncols {
            for c in 0..ncols {
                ata[r][c] += powers[r] * powers[c];
            }
            atb[r] += powers[r] * y[i];
        }
    }
    let coeffs = solve_linear(ata, atb)?;
    // r^2 from residuals.
    let mut ss_res = 0.0;
    for i in 0..n {
        let mut yhat = 0.0;
        for (k, c) in coeffs.iter().enumerate() {
            yhat += c * x[i].powi((deg - k) as i32);
        }
        ss_res += (y[i] - yhat).powi(2);
    }
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    let r_squared = if ss_tot == 0.0 {
        0.0
    } else {
        1.0 - ss_res / ss_tot
    };
    Some((coeffs, r_squared))
}

/// Solve a small dense system `a x = b` via Gaussian elimination with partial
/// pivoting. `None` when the matrix is singular.
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        // Partial pivot.
        let mut pivot = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        // Eliminate below.
        for r in (col + 1)..n {
            let factor = a[r][col] / a[col][col];
            for c in col..n {
                a[r][c] -= factor * a[col][c];
            }
            b[r] -= factor * b[col];
        }
    }
    // Back-substitute.
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for c in (i + 1)..n {
            sum -= a[i][c] * x[c];
        }
        x[i] = sum / a[i][i];
    }
    Some(x)
}

/// Build the TTFT-curve artifact from a parsed sweep aggregate. Errors only when
/// fewer than 2 raw sweep points are available; otherwise (including the
/// too-few-finite-points and below-floor cases) returns a populated payload.
pub fn process(sweep_json: &Value, spec: &CurveSpec) -> anyhow::Result<Value> {
    let points = extract_points(sweep_json, &spec.swept_param, &spec.metric_tag, &spec.stat)?;
    anyhow::ensure!(
        points.len() >= 2,
        "ttft_curve_fit: need >= 2 sweep points to fit a curve (got {} for {:?}/{:?}); \
         widen the recipe's ISL range or sweep more steps.",
        points.len(),
        spec.metric_tag,
        spec.stat
    );

    // raw_points carries ALL extracted points (pre finite-filtering), as floats.
    let raw_points: Vec<Value> = points
        .iter()
        .map(|&(x, y)| serde_json::json!({ "isl": num(x), "ttft_ms": num(y) }))
        .collect();

    let finite: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|&(x, y)| x.is_finite() && y.is_finite())
        .collect();

    if finite.len() < 2 {
        // Structured too-few-finite-points sentinel (mirrors the below_floor shape).
        return Ok(serde_json::json!({
            "fit_form": "linear",
            "coefficients": [],
            "r_squared": 0.0,
            "below_floor": true,
            "r_squared_floor": R2_FLOOR_DEFAULT,
            "error_reason": format!(
                "ttft_curve_fit: fewer than 2 finite trial points after dropping non-finite \
                 metric values (got {} of {}); check that swept cells produced successful \
                 requests.",
                finite.len(),
                points.len()
            ),
            "raw_points": raw_points,
            "swept_metric": spec.metric_tag,
            "stat": spec.stat,
            "swept_param": spec.swept_param,
        }));
    }

    let xs: Vec<f64> = finite.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = finite.iter().map(|p| p.1).collect();

    let (linear_coeffs, linear_r2) = polyfit_with_r2(&xs, &ys, 1).unwrap_or((vec![0.0, 0.0], 0.0));
    let mut fit_form = "linear";
    let mut coefficients = linear_coeffs;
    let mut r_squared = linear_r2;
    if linear_r2 < R2_FLOOR_DEFAULT && xs.len() >= 3 {
        if let Some((quad_coeffs, quad_r2)) = polyfit_with_r2(&xs, &ys, 2) {
            if quad_r2 > linear_r2 {
                fit_form = "quadratic";
                coefficients = quad_coeffs;
                r_squared = quad_r2;
            }
        }
    }
    let below_floor = r_squared.is_nan()
        || coefficients.iter().any(|c| !c.is_finite())
        || r_squared < R2_FLOOR_DEFAULT;

    Ok(serde_json::json!({
        "fit_form": fit_form,
        "coefficients": coefficients.iter().map(|&c| num(c)).collect::<Vec<_>>(),
        "r_squared": num(r_squared),
        "below_floor": below_floor,
        "r_squared_floor": R2_FLOOR_DEFAULT,
        "raw_points": raw_points,
        "swept_metric": spec.metric_tag,
        "stat": spec.stat,
        "swept_param": spec.swept_param,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agg(rows: &[(i64, f64)]) -> Value {
        let combos: Vec<Value> = rows
            .iter()
            .map(|&(isl, y)| {
                serde_json::json!({
                    "parameters": {"isl": isl, "concurrency": 1},
                    "metrics": {"time_to_first_token": {"mean": y}},
                })
            })
            .collect();
        serde_json::json!({ "per_combination_metrics": combos })
    }

    fn spec() -> CurveSpec {
        CurveSpec {
            metric_tag: "time_to_first_token".into(),
            stat: "avg".into(),
            swept_param: "datasets.main.prompts.isl".into(),
        }
    }

    #[test]
    fn perfect_linear_fit() {
        // TTFT = 0.05 * ISL exactly.
        let out = process(
            &agg(&[(256, 12.8), (512, 25.6), (1024, 51.2), (2048, 102.4)]),
            &spec(),
        )
        .unwrap();
        assert_eq!(out["fit_form"], "linear");
        assert!(out["r_squared"].as_f64().unwrap() > 0.999, "{out}");
        // slope ~= 0.05, intercept ~= 0.
        let coeffs = out["coefficients"].as_array().unwrap();
        assert!((coeffs[0].as_f64().unwrap() - 0.05).abs() < 1e-6, "{out}");
        assert_eq!(out["raw_points"].as_array().unwrap().len(), 4);
    }

    #[test]
    fn quadratic_fallback_on_curved_data() {
        // U-shaped parabola y = (ISL-3)^2: the linear fit is flat (slope 0,
        // r^2 = 0, well below the 0.85 floor), so the quadratic refit wins.
        // A monotonic y = ISL^2 would NOT work here — its linear r^2 is ~0.96,
        // above the floor, so the refit branch never runs.
        let out = process(
            &agg(&[(1, 4.0), (2, 1.0), (3, 0.0), (4, 1.0), (5, 4.0)]),
            &spec(),
        )
        .unwrap();
        assert_eq!(out["fit_form"], "quadratic");
        assert_eq!(out["coefficients"].as_array().unwrap().len(), 3);
    }
}
