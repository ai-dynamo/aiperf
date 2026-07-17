// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! `itl_surface_fit` handler for the `decode-itl-curve` recipe.
//!
//! Builds an axis-aligned `(concurrency, OSL, ITL)` grid from observed values.
//! Unmeasured cells are `null`; values are not interpolated.

use serde_json::{Map, Value};

use crate::search::extract::{extract_2d_points, num};

/// Resolved params for the `itl_surface_fit` handler.
pub struct SurfaceSpec {
    /// ITL metric tag (`inter_token_latency`).
    pub metric_tag: String,
    /// Statistic label (`avg`).
    pub stat: String,
    /// Dotted swept concurrency path (`phases.profiling.concurrency`).
    pub concurrency_param: String,
    /// Dotted swept OSL path (`datasets.main.prompts.osl`).
    pub osl_param: String,
}

/// Build the ITL-surface artifact from a parsed sweep aggregate. Errors only when
/// the aggregate carried zero candidate rows; a too-few-finite-rows outcome
/// returns the structured `surface_fit_failed=true` sentinel.
pub fn process(sweep_json: &Value, spec: &SurfaceSpec) -> anyhow::Result<Value> {
    let (triples, dropped_non_finite, dropped_negative) = extract_2d_points(
        sweep_json,
        &spec.concurrency_param,
        &spec.osl_param,
        &spec.metric_tag,
        &spec.stat,
    )?;

    let raw_points: Vec<Value> = triples
        .iter()
        .map(|&(c, o, v)| serde_json::json!({ "concurrency": num(c), "osl": num(o), "itl_ms": num(v) }))
        .collect();
    let swept_params = vec![
        Value::String(spec.concurrency_param.clone()),
        Value::String(spec.osl_param.clone()),
    ];

    if triples.len() < 2 {
        return Ok(serde_json::json!({
            "swept_metric": spec.metric_tag,
            "stat": spec.stat,
            "swept_params": swept_params,
            "raw_points": raw_points,
            "surface": {
                "concurrency_axis": Vec::<Value>::new(),
                "osl_axis": Vec::<Value>::new(),
                "itl_grid": Vec::<Value>::new(),
            },
            "surface_fit_failed": true,
            "error_reason": format!(
                "itl_surface_fit: fewer than 2 finite-positive rows after dropping \
                 non-finite/negative ITL (kept {}, dropped {} non-finite + {} negative); \
                 check that streaming was enabled and that swept cells produced successful \
                 requests.",
                triples.len(),
                dropped_non_finite,
                dropped_negative
            ),
        }));
    }

    let concurrency_axis = sorted_unique(triples.iter().map(|t| t.0));
    let osl_axis = sorted_unique(triples.iter().map(|t| t.1));
    let cell = |c: f64, o: f64| -> Value {
        triples
            .iter()
            .find(|t| t.0 == c && t.1 == o)
            .map(|t| num(t.2))
            .unwrap_or(Value::Null)
    };
    let itl_grid: Vec<Value> = concurrency_axis
        .iter()
        .map(|&c| Value::Array(osl_axis.iter().map(|&o| cell(c, o)).collect()))
        .collect();

    let mut surface = Map::new();
    surface.insert(
        "concurrency_axis".into(),
        Value::Array(concurrency_axis.iter().map(|&c| num(c)).collect()),
    );
    surface.insert(
        "osl_axis".into(),
        Value::Array(osl_axis.iter().map(|&o| num(o)).collect()),
    );
    surface.insert("itl_grid".into(), Value::Array(itl_grid));

    Ok(serde_json::json!({
        "swept_metric": spec.metric_tag,
        "stat": spec.stat,
        "swept_params": swept_params,
        "raw_points": raw_points,
        "surface": Value::Object(surface),
        "surface_fit_failed": false,
    }))
}

/// Ascending unique values (`sorted({...})`), NaN-tolerant.
fn sorted_unique(it: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut v: Vec<f64> = it.collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v.dedup();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> SurfaceSpec {
        SurfaceSpec {
            metric_tag: "inter_token_latency".into(),
            stat: "avg".into(),
            concurrency_param: "phases.profiling.concurrency".into(),
            osl_param: "datasets.main.prompts.osl".into(),
        }
    }

    #[test]
    fn builds_grid_with_nulls_for_missing_cells() {
        let agg = serde_json::json!({
            "per_combination_metrics": [
                {"parameters": {"concurrency": 1, "osl": 64},
                 "metrics": {"inter_token_latency": {"mean": 10.0}}},
                {"parameters": {"concurrency": 1, "osl": 256},
                 "metrics": {"inter_token_latency": {"mean": 12.0}}},
                {"parameters": {"concurrency": 4, "osl": 64},
                 "metrics": {"inter_token_latency": {"mean": 15.0}}},
            ],
        });
        let out = process(&agg, &spec()).unwrap();
        assert_eq!(out["swept_metric"], "inter_token_latency");
        assert_eq!(out["surface_fit_failed"], false);
        let grid = out["surface"]["itl_grid"].as_array().unwrap();
        assert_eq!(grid.len(), 2);
        assert_eq!(grid[0][0].as_f64(), Some(10.0));
        assert!(grid[1][1].is_null());
        assert_eq!(out["raw_points"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn too_few_rows_sentinel() {
        let agg = serde_json::json!({
            "per_combination_metrics": [
                {"parameters": {"concurrency": 1, "osl": 64},
                 "metrics": {"inter_token_latency": {"mean": 10.0}}},
            ],
        });
        let out = process(&agg, &spec()).unwrap();
        assert_eq!(out["surface_fit_failed"], true);
        assert!(out["error_reason"].is_string());
    }
}
