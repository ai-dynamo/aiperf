// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Multi-run confidence statistics.
//!
//! For a set of successful runs of the same configuration, pool each
//! `(metric_tag, stat_key)` scalar across runs and compute a
//! [`ConfidenceMetric`] (mean / sample-std / CV / SE / t-distribution CI). The
//! Student-t inverse CDF ([`t_ppf`]) uses regularized incomplete beta and
//! bisection to approximately `1e-10`. The invariants
//! `ci_low <= mean <= ci_high`, `t_critical > 0`, `cv == std/mean` are what
//! define the numerical contract.

use std::path::Path;

use serde_json::{Map, Value};

use crate::jsonnum::num as scrub;
use crate::model::export::AIPERF_V1_VERSION;

/// Per-metric statistic keys pooled across runs in artifact order. In the native v1
/// summary (`profile_export_aiperf.json`) every one of these is a flat key on a
/// metric object (percentiles are NOT nested under `percentiles`).
pub const STAT_KEYS: &[&str] = &[
    "avg", "min", "max", "sum", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "std",
];

/// Confidence statistics for one flattened `{metric_tag}_{stat_key}` series
/// across runs.
#[derive(Clone, Debug)]
pub struct ConfidenceMetric {
    /// Sample mean of the run-level values.
    pub mean: f64,
    /// Sample standard deviation (ddof=1); `0.0` for a single run.
    pub std: f64,
    /// Minimum across runs.
    pub min: f64,
    /// Maximum across runs.
    pub max: f64,
    /// Coefficient of variation (`std/mean`; `+inf` when `mean == 0`).
    pub cv: f64,
    /// Standard error (`std/sqrt(n)`).
    pub se: f64,
    /// Lower confidence bound (`mean - t_critical*se`).
    pub ci_low: f64,
    /// Upper confidence bound (`mean + t_critical*se`).
    pub ci_high: f64,
    /// Student-t critical value (`NaN` for the single-run degraded record).
    pub t_critical: f64,
    /// Unit of the first finite run's metric.
    pub unit: String,
}

impl ConfidenceMetric {
    /// The full 10-field confidence projection used by the confidence JSON
    /// exporter. Non-finite floats render as JSON `null`.
    fn to_full_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("mean".into(), scrub(self.mean));
        m.insert("std".into(), scrub(self.std));
        m.insert("min".into(), scrub(self.min));
        m.insert("max".into(), scrub(self.max));
        m.insert("cv".into(), scrub(self.cv));
        m.insert("se".into(), scrub(self.se));
        m.insert("ci_low".into(), scrub(self.ci_low));
        m.insert("ci_high".into(), scrub(self.ci_high));
        m.insert("t_critical".into(), scrub(self.t_critical));
        m.insert("unit".into(), Value::String(self.unit.clone()));
        Value::Object(m)
    }
}

/// Compute the confidence statistics for one metric's run-level values.
///
/// `values` must be non-empty. `n == 1` produces the degraded single-run record
/// with a collapsed CI; `n >= 2` uses the Student-t interval.
pub fn compute_confidence_stats(values: &[f64], unit: &str, confidence: f64) -> ConfidenceMetric {
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if n == 1 {
        return ConfidenceMetric {
            mean,
            std: 0.0,
            min: mean,
            max: mean,
            cv: 0.0,
            se: 0.0,
            ci_low: mean,
            ci_high: mean,
            t_critical: f64::NAN,
            unit: unit.to_string(),
        };
    }

    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let std = var.sqrt();
    let cv = if mean != 0.0 {
        std / mean
    } else {
        f64::INFINITY
    };
    let se = std / (n as f64).sqrt();
    let alpha = 1.0 - confidence;
    let df = n as f64 - 1.0;
    let t_critical = t_ppf(1.0 - alpha / 2.0, df);
    let margin = t_critical * se;

    ConfidenceMetric {
        mean,
        std,
        min,
        max,
        cv,
        se,
        ci_low: mean - margin,
        ci_high: mean + margin,
        t_critical,
        unit: unit.to_string(),
    }
}

/// Read `profile_export_aiperf.json` from a cell's artifact directory.
pub fn read_summary(dir: &Path) -> Option<Value> {
    let bytes = std::fs::read(dir.join("profile_export_aiperf.json")).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// True iff a top-level summary entry is a metric object (carries a `unit`
/// string). This distinguishes real metrics from `input_config` / `run_info` /
/// `telemetry_data` / scalar version fields.
fn is_metric_object(v: &Value) -> bool {
    v.get("unit").and_then(Value::as_str).is_some()
}

/// Classify a summary as complete when `request_count.avg` is nonzero.
pub fn classify_summary(summary: &Value) -> Result<(), String> {
    let request_avg = summary
        .get("request_count")
        .and_then(|rc| rc.get("avg"))
        .and_then(Value::as_f64);
    if matches!(request_avg, Some(v) if v != 0.0) {
        return Ok(());
    }
    let error_avg = summary
        .get("error_request_count")
        .and_then(|ec| ec.get("avg"))
        .and_then(Value::as_f64);
    match error_avg {
        Some(v) if v != 0.0 => Err(format!("All {} requests failed", v as i64)),
        _ => Err("No requests completed".to_string()),
    }
}

/// Pool every `(metric_tag, stat_key)` scalar across the given run summaries and
/// compute a [`ConfidenceMetric`] for each. Tags iterate in first-seen order,
/// stats in [`STAT_KEYS`] order.
pub fn collect_confidence_metrics(
    summaries: &[&Value],
    confidence: f64,
) -> Vec<(String, ConfidenceMetric)> {
    let mut tags: Vec<String> = Vec::new();
    for s in summaries {
        if let Some(obj) = s.as_object() {
            for (k, v) in obj {
                if is_metric_object(v) && !tags.iter().any(|t| t == k) {
                    tags.push(k.clone());
                }
            }
        }
    }

    let mut out = Vec::new();
    for tag in &tags {
        for stat in STAT_KEYS {
            let mut values = Vec::new();
            let mut unit = String::new();
            for s in summaries {
                let Some(mobj) = s.get(tag).and_then(Value::as_object) else {
                    continue;
                };
                let Some(val) = mobj.get(*stat).and_then(Value::as_f64) else {
                    continue;
                };
                if !val.is_finite() {
                    continue;
                }
                values.push(val);
                if unit.is_empty() {
                    unit = mobj
                        .get("unit")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                }
            }
            if values.is_empty() {
                continue;
            }
            out.push((
                format!("{tag}_{stat}"),
                compute_confidence_stats(&values, &unit, confidence),
            ));
        }
    }
    out
}

/// One failed run's `{label, error}` record (`failed_runs`).
pub struct FailedRun {
    /// The run/trial label (`run_NNNN`).
    pub label: String,
    /// The failure detail (`None` renders as JSON `null`).
    pub error: Option<String>,
}

/// Write the confidence aggregate JSON+CSV pair into `dir`
/// (`profile_export_aiperf_aggregate.{json,csv}`). `extra_metadata` is appended
/// after the base metadata block (per-variation cells add `sweep_mode` /
/// `variation_*`; the non-sweep path passes an empty slice).
#[allow(clippy::too_many_arguments)]
pub fn write_confidence_aggregate(
    dir: &Path,
    num_runs: usize,
    num_successful: usize,
    failed_runs: &[FailedRun],
    run_labels: &[String],
    confidence: f64,
    cooldown: f64,
    single_run: bool,
    metrics: &[(String, ConfidenceMetric)],
    extra_metadata: &[(String, Value)],
) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;

    let failed_json: Vec<Value> = failed_runs
        .iter()
        .map(|f| {
            serde_json::json!({
                "label": f.label,
                "error": f.error,
            })
        })
        .collect();

    let mut metadata = Map::new();
    metadata.insert(
        "aggregation_type".into(),
        Value::String("confidence".into()),
    );
    metadata.insert("num_profile_runs".into(), Value::from(num_runs as u64));
    metadata.insert(
        "num_successful_runs".into(),
        Value::from(num_successful as u64),
    );
    metadata.insert("failed_runs".into(), Value::Array(failed_json));
    metadata.insert("confidence_level".into(), scrub(confidence));
    metadata.insert(
        "run_labels".into(),
        Value::Array(run_labels.iter().cloned().map(Value::String).collect()),
    );
    if single_run {
        metadata.insert("single_run".into(), Value::Bool(true));
    }
    metadata.insert("cooldown_seconds".into(), scrub(cooldown));
    for (k, v) in extra_metadata {
        metadata.insert(k.clone(), v.clone());
    }

    let mut metrics_json = Map::new();
    for (name, metric) in metrics {
        metrics_json.insert(name.clone(), metric.to_full_json());
    }

    let mut root = Map::new();
    root.insert("schema_version".into(), Value::String("1.0".into()));
    root.insert(
        "aiperf_version".into(),
        Value::String(AIPERF_V1_VERSION.into()),
    );
    root.insert("metadata".into(), Value::Object(metadata));
    root.insert("metrics".into(), Value::Object(metrics_json));

    std::fs::write(
        dir.join("profile_export_aiperf_aggregate.json"),
        serde_json::to_string_pretty(&Value::Object(root))?,
    )?;
    std::fs::write(
        dir.join("profile_export_aiperf_aggregate.csv"),
        confidence_csv(num_runs, num_successful, confidence, cooldown, metrics),
    )?;
    Ok(())
}

/// Build the confidence aggregate CSV with metrics and metadata sections.
fn confidence_csv(
    num_runs: usize,
    num_successful: usize,
    confidence: f64,
    cooldown: f64,
    metrics: &[(String, ConfidenceMetric)],
) -> String {
    let mut w = super::aggregate::CsvWriter::new();
    w.row(&[
        "metric",
        "mean",
        "std",
        "min",
        "max",
        "cv",
        "se",
        "ci_low",
        "ci_high",
        "t_critical",
        "unit",
    ]);
    for (name, m) in metrics {
        w.row(&[
            name.clone(),
            fmt_csv(m.mean, 2),
            fmt_csv(m.std, 2),
            fmt_csv(m.min, 2),
            fmt_csv(m.max, 2),
            fmt_csv(m.cv, 4),
            fmt_csv(m.se, 4),
            fmt_csv(m.ci_low, 2),
            fmt_csv(m.ci_high, 2),
            fmt_csv(m.t_critical, 4),
            m.unit.clone(),
        ]);
    }
    w.row(&[] as &[String]);
    w.row(&["Aggregation Type".to_string(), "confidence".to_string()]);
    w.row(&["Total Runs".to_string(), num_runs.to_string()]);
    w.row(&["Successful Runs".to_string(), num_successful.to_string()]);
    w.row(&["Confidence Level".to_string(), py_float_str(confidence)]);
    w.row(&["Cooldown Seconds".to_string(), py_float_str(cooldown)]);
    w.finish()
}

/// Render `inf`, `-inf`, and `nan` literally; otherwise use fixed decimals.
fn fmt_csv(v: f64, decimals: usize) -> String {
    if v == f64::INFINITY {
        "inf".to_string()
    } else if v == f64::NEG_INFINITY {
        "-inf".to_string()
    } else if v.is_nan() {
        "nan".to_string()
    } else {
        format!("{v:.decimals$}")
    }
}

/// Render integral metadata floats with one decimal place.
fn py_float_str(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

/// Lanczos approximation of `ln Γ(x)` for `x > 0`.
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula.
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Continued-fraction expansion for the incomplete beta (Numerical Recipes
/// `betacf`).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 200;
    const EPS: f64 = 3.0e-12;
    const FPMIN: f64 = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta `I_x(a, b)` (Numerical Recipes `betai`).
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// CDF of the Student-t distribution with `df` degrees of freedom at `t`.
fn t_cdf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    let ib = 0.5 * betai(df / 2.0, 0.5, x);
    if t >= 0.0 { 1.0 - ib } else { ib }
}

/// Inverse CDF (quantile) of the Student-t distribution: the `t` with
/// `P(T <= t) = p` for `df` degrees of freedom. Bisection on the CDF to ~1e-10.
pub fn t_ppf(p: f64, df: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let mut lo = -1.0e7_f64;
    let mut hi = 1.0e7_f64;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if t_cdf(mid, df) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_ppf_matches_known_values() {
        assert!((t_ppf(0.975, 2.0) - 4.302_653).abs() < 1e-4);
        assert!((t_ppf(0.975, 4.0) - 2.776_445).abs() < 1e-4);
        assert!((t_ppf(0.975, 9.0) - 2.262_157).abs() < 1e-4);
        assert!((t_ppf(0.5, 5.0)).abs() < 1e-6);
    }

    #[test]
    fn confidence_stats_are_internally_consistent() {
        let values = [100.0, 110.0, 105.0];
        let m = compute_confidence_stats(&values, "requests/sec", 0.95);
        assert!((m.mean - 105.0).abs() < 1e-9);
        assert!((m.min - 100.0).abs() < 1e-9);
        assert!((m.max - 110.0).abs() < 1e-9);
        assert!(m.std > 0.0);
        assert!(m.se > 0.0);
        assert!(m.t_critical > 0.0);
        assert!(m.ci_low <= m.mean && m.mean <= m.ci_high);
        assert!((m.cv - m.std / m.mean).abs() < 1e-9);
    }

    #[test]
    fn single_run_is_degraded() {
        let m = compute_confidence_stats(&[42.0], "ms", 0.95);
        assert_eq!(m.mean, 42.0);
        assert_eq!(m.std, 0.0);
        assert_eq!(m.ci_low, 42.0);
        assert_eq!(m.ci_high, 42.0);
        assert!(m.t_critical.is_nan());
    }
}
