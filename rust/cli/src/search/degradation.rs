// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! `degradation_knee_detect` handler for the `concurrency-ramp` recipe.
//!
//! Uses the lowest swept value as the baseline and reports the first metric
//! above `baseline * (1 + threshold_pct)` in `degradation_knee.json`.

use serde_json::{Map, Value};

use crate::search::extract::{extract_points, int_or_float, num};

/// Resolved params for the `degradation_knee_detect` handler.
pub struct DegradationKneeSpec {
    /// Degradation cutoff fraction (`--degradation-threshold`, default 0.20).
    pub threshold_pct: f64,
    /// Metric tag inspected (`--degradation-metric-tag`, default `request_latency`).
    pub metric_tag: String,
    /// Statistic label (`--degradation-stat`, default `p99`).
    pub stat: String,
    /// Dotted swept path (`phases.profiling.concurrency`).
    pub swept_param: String,
}

/// Build the degradation-knee artifact from a parsed sweep aggregate.
///
/// Errors when the aggregate carries no rows for the swept parameter/metric, or
/// when the baseline is non-finite or non-positive, which makes the cutoff
/// comparison undefined.
pub fn process(sweep_json: &Value, spec: &DegradationKneeSpec) -> anyhow::Result<Value> {
    let points = extract_points(sweep_json, &spec.swept_param, &spec.metric_tag, &spec.stat)?;
    let (baseline_x, baseline_y) = points[0];
    anyhow::ensure!(
        baseline_y.is_finite(),
        "degradation_knee_detect: baseline {} for {:?} is non-finite ({baseline_y}); a \
         NaN/inf baseline collapses the cutoff comparison so the handler can't tell \
         'no knee in range' from 'data is junk'.",
        spec.stat,
        spec.metric_tag
    );
    anyhow::ensure!(
        baseline_y > 0.0,
        "degradation_knee_detect: baseline {} for {:?} must be positive (got {baseline_y}); \
         latency metrics must be strictly positive.",
        spec.stat,
        spec.metric_tag
    );

    let cutoff = baseline_y * (1.0 + spec.threshold_pct);
    let mut knee: Option<(f64, f64)> = None;
    for &(x, y) in &points[1..] {
        if y > cutoff {
            knee = Some((x, y));
            break;
        }
    }

    let leaf = spec
        .swept_param
        .rsplit('.')
        .next()
        .unwrap_or(&spec.swept_param);

    let all_points: Vec<Value> = points
        .iter()
        .map(|&(x, y)| {
            let mut m = Map::new();
            m.insert(leaf.to_string(), int_or_float(x));
            m.insert(spec.stat.clone(), num(y));
            Value::Object(m)
        })
        .collect();

    let mut out = Map::new();
    out.insert(format!("baseline_{leaf}"), int_or_float(baseline_x));
    out.insert(format!("baseline_{}", spec.stat), num(baseline_y));
    out.insert(
        format!("knee_{leaf}"),
        knee.map(|(x, _)| int_or_float(x)).unwrap_or(Value::Null),
    );
    out.insert(
        format!("knee_{}", spec.stat),
        knee.map(|(_, y)| num(y)).unwrap_or(Value::Null),
    );
    out.insert("threshold_pct".into(), num(spec.threshold_pct));
    out.insert(
        "swept_metric".into(),
        Value::String(spec.metric_tag.clone()),
    );
    out.insert("stat".into(), Value::String(spec.stat.clone()));
    out.insert(
        "swept_param".into(),
        Value::String(spec.swept_param.clone()),
    );
    out.insert("all_points".into(), Value::Array(all_points));
    Ok(Value::Object(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agg(rows: &[(i64, f64)]) -> Value {
        let combos: Vec<Value> = rows
            .iter()
            .map(|&(c, y)| {
                serde_json::json!({
                    "parameters": {"concurrency": c},
                    "metrics": {"request_latency": {"mean": y}},
                })
            })
            .collect();
        serde_json::json!({ "per_combination_metrics": combos })
    }

    fn spec() -> DegradationKneeSpec {
        DegradationKneeSpec {
            threshold_pct: 0.20,
            metric_tag: "request_latency".into(),
            stat: "p99".into(),
            swept_param: "phases.profiling.concurrency".into(),
        }
    }

    #[test]
    fn finds_first_knee_past_cutoff() {
        let out = process(
            &agg(&[(1, 10.0), (7, 11.5), (19, 13.0), (52, 40.0)]),
            &spec(),
        )
        .unwrap();
        assert_eq!(out["baseline_concurrency"].as_i64(), Some(1));
        assert_eq!(out["knee_concurrency"].as_i64(), Some(19));
        assert_eq!(out["swept_metric"], "request_latency");
        assert_eq!(out["stat"], "p99");
        assert_eq!(out["all_points"].as_array().unwrap().len(), 4);
    }

    #[test]
    fn no_knee_in_range_is_null() {
        let out = process(&agg(&[(1, 10.0), (7, 10.5), (19, 11.0)]), &spec()).unwrap();
        assert!(out["knee_concurrency"].is_null());
        assert!(out["knee_p99"].is_null());
    }

    #[test]
    fn non_positive_baseline_errors() {
        assert!(process(&agg(&[(1, 0.0), (7, 5.0)]), &spec()).is_err());
    }
}
