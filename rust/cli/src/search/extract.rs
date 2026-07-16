// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Shared sweep-aggregate point extraction for the curve/surface post-process
//! handlers.
//!
//! Pure-Rust port of `aiperf.search_recipes._sweep_extract::_extract_points`
//! (`src/aiperf/search_recipes/_sweep_extract.py:19-74`) and
//! `aiperf.search_recipes._itl_surface_fit::_extract_2d_points`
//! (`src/aiperf/search_recipes/_itl_surface_fit.py:22-107`).
//!
//! Both walk the sweep aggregate's `per_combination_metrics` rows and pull
//! `(swept_value, metric_mean)` pairs (or `(concurrency, osl, metric_mean)`
//! triples), tolerating the two layouts `SweepAnalyzer.compute` produces: the
//! multi-trial flat key `"<metric_tag>_<stat>"` block and the single-trial
//! tag-only block. In BOTH cases the value read is the block's `mean` — the
//! requested `stat` is never read here (single-trial blocks carry no per-stat
//! percentiles), matching the Python contract exactly.

use serde_json::Value;

/// A row's swept-parameter value keyed by either the full dotted path or the
/// leaf display name (`_extract_points`: `swept_param in params` else
/// `short_key in params`). Returns `None` when neither key is present.
fn param_value<'a>(params: &'a Value, dotted: &str, leaf: &str) -> Option<&'a Value> {
    let obj = params.as_object()?;
    obj.get(dotted).or_else(|| obj.get(leaf))
}

/// The block `mean` for `metric_tag`/`stat`, preferring the multi-trial flat key
/// `"<metric_tag>_<stat>"` then the single-trial tag-only block (both read
/// `mean`). `None` when neither block carries a `mean`.
fn metric_mean(metrics: &Value, metric_tag: &str, stat: &str) -> Option<f64> {
    let obj = metrics.as_object()?;
    let flat_key = format!("{metric_tag}_{stat}");
    if let Some(m) = obj
        .get(&flat_key)
        .and_then(|b| b.get("mean"))
        .and_then(Value::as_f64)
    {
        return Some(m);
    }
    obj.get(metric_tag)
        .and_then(|b| b.get("mean"))
        .and_then(Value::as_f64)
}

/// Pull `(swept_value, metric_mean)` pairs from the sweep aggregate, ascending by
/// swept value. Byte-faithful to `_extract_points`: skips rows missing the swept
/// key or the metric block; errors when nothing survives.
pub fn extract_points(
    sweep_aggregate: &Value,
    swept_param: &str,
    metric_tag: &str,
    stat: &str,
) -> anyhow::Result<Vec<(f64, f64)>> {
    let leaf = swept_param.rsplit('.').next().unwrap_or(swept_param);
    let empty: Vec<Value> = Vec::new();
    let rows = sweep_aggregate
        .get("per_combination_metrics")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    let mut points: Vec<(f64, f64)> = Vec::new();
    for row in rows {
        let Some(params) = row.get("parameters") else {
            continue;
        };
        let Some(pv) = param_value(params, swept_param, leaf).and_then(Value::as_f64) else {
            continue;
        };
        let Some(metrics) = row.get("metrics") else {
            continue;
        };
        let Some(mean) = metric_mean(metrics, metric_tag, stat) else {
            continue;
        };
        points.push((pv, mean));
    }
    anyhow::ensure!(
        !points.is_empty(),
        "post-process: sweep aggregate has no rows with parameter {swept_param:?} and \
         metric {metric_tag:?} (flat key \"{metric_tag}_{stat}\"); check that the recipe \
         swept that parameter and that the metric is enabled (e.g. --streaming for \
         time_to_first_token)."
    );
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(points)
}

/// Pull `(concurrency, osl, metric_mean)` triples, sorted ascending by
/// `(concurrency, osl)`. Byte-faithful to `_extract_2d_points`: non-finite and
/// negative metric cells are dropped (returned as counts); errors only when the
/// aggregate had zero candidate rows.
pub fn extract_2d_points(
    sweep_aggregate: &Value,
    concurrency_param: &str,
    osl_param: &str,
    metric_tag: &str,
    stat: &str,
) -> anyhow::Result<(Vec<(f64, f64, f64)>, usize, usize)> {
    let c_leaf = concurrency_param
        .rsplit('.')
        .next()
        .unwrap_or(concurrency_param);
    let o_leaf = osl_param.rsplit('.').next().unwrap_or(osl_param);
    let empty: Vec<Value> = Vec::new();
    let rows = sweep_aggregate
        .get("per_combination_metrics")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    let mut triples: Vec<(f64, f64, f64)> = Vec::new();
    let mut dropped_non_finite = 0usize;
    let mut dropped_negative = 0usize;
    let mut candidate_count = 0usize;
    for row in rows {
        let Some(params) = row.get("parameters") else {
            continue;
        };
        let Some(cv) = param_value(params, concurrency_param, c_leaf).and_then(Value::as_f64)
        else {
            continue;
        };
        let Some(ov) = param_value(params, osl_param, o_leaf).and_then(Value::as_f64) else {
            continue;
        };
        let Some(metrics) = row.get("metrics") else {
            continue;
        };
        let Some(mean) = metric_mean(metrics, metric_tag, stat) else {
            continue;
        };
        candidate_count += 1;
        if !mean.is_finite() {
            dropped_non_finite += 1;
            continue;
        }
        if mean < 0.0 {
            dropped_negative += 1;
            continue;
        }
        triples.push((cv, ov, mean));
    }
    anyhow::ensure!(
        candidate_count > 0,
        "itl_surface_fit: sweep aggregate has no rows with parameters {concurrency_param:?} \
         + {osl_param:?} and metric {metric_tag:?} (flat key \"{metric_tag}_{stat}\"); \
         check that the recipe swept both axes and streaming was enabled."
    );
    triples.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    Ok((triples, dropped_non_finite, dropped_negative))
}

/// Emit an integral `f64` as a JSON integer, else as a JSON float — mirrors the
/// Python `int(x) if x == int(x) else x` coercion used across the handlers.
pub fn int_or_float(x: f64) -> Value {
    if x.is_finite() && x == x.trunc() && x.abs() < 9.007_199_254_740_992e15 {
        Value::from(x as i64)
    } else {
        serde_json::Number::from_f64(x)
            .map(Value::Number)
            .unwrap_or(Value::Null)
    }
}

/// A finite `f64` as a JSON number, else JSON `null`.
pub fn num(v: f64) -> Value {
    serde_json::Number::from_f64(v)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}
