// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! `sla_breach_knee` handler for the grid recipe.
//!
//! Evaluates every SLA filter in swept-value order, collapses equal-axis rows
//! with all-pass semantics, and writes the boundary to `sla_breach.json`.

use serde_json::{Map, Value};

use crate::jsonnum::num;
use crate::search::{SlaFilter, op_str};

/// Read one metric value from either sweep-aggregate layout:
///
/// 1. Multi-trial flat key `"<metric_tag>_<stat>"` → read `mean`.
/// 2. Single-trial tag-only block + direct `stat` field.
/// 3. Single-trial tag-only block + `mean` fallback (single-trial blocks may omit
///    per-percentile keys, so a `p95`/`p99` filter would otherwise read `None`).
fn read_metric_value(metrics: &Map<String, Value>, metric_tag: &str, stat: &str) -> Option<f64> {
    let flat_key = format!("{metric_tag}_{stat}");
    if let Some(block) = metrics.get(&flat_key).and_then(Value::as_object)
        && let Some(mean) = block.get("mean").and_then(Value::as_f64)
    {
        return Some(mean);
    }
    if let Some(block) = metrics.get(metric_tag).and_then(Value::as_object) {
        if let Some(raw) = block.get(stat).and_then(Value::as_f64) {
            return Some(raw);
        }
        if let Some(mean) = block.get("mean").and_then(Value::as_f64) {
            return Some(mean);
        }
    }
    None
}

/// Per-filter breach records for one row, preserving filter order. A filter
/// breaches when the metric is missing or the
/// comparison fails; the breach record carries the raw observation (`null` when
/// missing).
fn evaluate_breaches(metrics: Option<&Map<String, Value>>, filters: &[SlaFilter]) -> Vec<Value> {
    let mut breaches = Vec::new();
    for f in filters {
        let observed = metrics.and_then(|m| read_metric_value(m, &f.metric_tag, &f.stat));
        let passed = f.satisfied_by(observed);
        if !passed {
            let mut b = Map::new();
            b.insert("metric_tag".into(), Value::String(f.metric_tag.clone()));
            b.insert("stat".into(), Value::String(f.stat.clone()));
            b.insert("op".into(), Value::String(op_str(f.op).to_string()));
            b.insert("threshold".into(), num(f.threshold));
            b.insert("observed".into(), observed.map(num).unwrap_or(Value::Null));
            breaches.push(Value::Object(b));
        }
    }
    breaches
}

/// One collapsed swept-x point.
struct Point {
    /// Sort key (float coercion of the swept value).
    x: f64,
    /// Raw swept value, type-preserved for JSON output.
    raw: Value,
    /// Feasible iff every underlying row passed every filter.
    feasible: bool,
    /// Concatenated breach records (empty when feasible).
    breaches: Vec<Value>,
}

/// Locate the SLA-feasibility boundary along the swept parameter and return the
/// `sla_breach.json` document. `sweep_json` is the parsed
/// `profile_export_aiperf_sweep.json`; `swept_param` is the dotted axis path;
/// `filters` are the recipe's SLA filters.
pub fn process(sweep_json: &Value, swept_param: &str, filters: &[SlaFilter]) -> Value {
    let leaf = swept_param.rsplit('.').next().unwrap_or(swept_param);
    let empty: Vec<Value> = Vec::new();
    let rows = sweep_json
        .get("per_combination_metrics")
        .and_then(Value::as_array)
        .unwrap_or(&empty);

    // Recipes may stamp either the dotted path or its leaf display name.
    let param_key: &str = rows
        .first()
        .and_then(|r| r.get("parameters"))
        .and_then(Value::as_object)
        .filter(|p| p.contains_key(swept_param))
        .map(|_| swept_param)
        .unwrap_or(leaf);

    let mut kept: Vec<&Value> = rows
        .iter()
        .filter(|r| {
            r.get("parameters")
                .and_then(Value::as_object)
                .is_some_and(|p| p.contains_key(param_key))
        })
        .collect();
    kept.sort_by(|a, b| {
        let av = row_x(a, param_key);
        let bv = row_x(b, param_key);
        av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
    });

    // A repeated axis value is feasible only when every row passes every filter.
    let mut points: Vec<Point> = Vec::new();
    for r in &kept {
        let raw = r
            .get("parameters")
            .and_then(|p| p.get(param_key))
            .cloned()
            .unwrap_or(Value::Null);
        let x = raw.as_f64().unwrap_or(f64::NAN);
        let metrics = r.get("metrics").and_then(Value::as_object);
        let row_breaches = evaluate_breaches(metrics, filters);
        let row_feasible = row_breaches.is_empty();
        if let Some(p) = points.iter_mut().find(|p| p.x == x) {
            p.feasible = p.feasible && row_feasible;
            p.breaches.extend(row_breaches);
        } else {
            points.push(Point {
                x,
                raw,
                feasible: row_feasible,
                breaches: row_breaches,
            });
        }
    }

    let all_points: Vec<Value> = points
        .iter()
        .map(|p| {
            let mut m = Map::new();
            m.insert(leaf.to_string(), p.raw.clone());
            m.insert("feasible".into(), Value::Bool(p.feasible));
            m.insert("breaches".into(), Value::Array(p.breaches.clone()));
            Value::Object(m)
        })
        .collect();

    let max_passing = points
        .iter()
        .filter(|p| p.feasible)
        .max_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
        .map(|p| p.raw.clone())
        .unwrap_or(Value::Null);
    let first_failing_point = points
        .iter()
        .filter(|p| !p.feasible)
        .min_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
    let first_failing = first_failing_point
        .map(|p| p.raw.clone())
        .unwrap_or(Value::Null);
    let first_failing_breach = first_failing_point
        .and_then(|p| p.breaches.first().cloned())
        .unwrap_or(Value::Null);

    // Monotonicity: ascending swept order, no feasible may follow an infeasible.
    let mut seen_infeasible = false;
    let mut monotonic = true;
    for p in &points {
        if !p.feasible {
            seen_infeasible = true;
        } else if seen_infeasible {
            monotonic = false;
            break;
        }
    }

    let mut out = Map::new();
    out.insert("swept_param".into(), Value::String(swept_param.to_string()));
    out.insert(format!("max_passing_{leaf}"), max_passing);
    out.insert(format!("first_failing_{leaf}"), first_failing);
    out.insert("first_failing_breach".into(), first_failing_breach);
    out.insert("all_points".into(), Value::Array(all_points));
    out.insert("monotonicity_check".into(), Value::Bool(monotonic));
    out.insert(
        "filters".into(),
        Value::Array(filters.iter().map(SlaFilter::to_dict).collect()),
    );
    Value::Object(out)
}

/// Float coercion of a row's swept value (the sort key).
fn row_x(row: &Value, param_key: &str) -> f64 {
    row.get("parameters")
        .and_then(|p| p.get(param_key))
        .and_then(Value::as_f64)
        .unwrap_or(f64::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::SlaOp;

    fn combo(concurrency: i64, itl_p95: f64) -> Value {
        serde_json::json!({
            "parameters": {"concurrency": concurrency},
            "metrics": {"inter_token_latency": {"mean": itl_p95, "p95": itl_p95}},
        })
    }

    #[test]
    fn locates_knee_between_feasible_and_infeasible() {
        let sweep = serde_json::json!({
            "per_combination_metrics": [
                combo(1, 5.0), combo(3, 8.0), combo(7, 11.0),
                combo(19, 20.0), combo(52, 40.0),
            ],
        });
        let filters = vec![SlaFilter {
            metric_tag: "inter_token_latency".into(),
            stat: "p95".into(),
            op: SlaOp::Lt,
            threshold: 12.0,
        }];
        let out = process(&sweep, "phases.profiling.concurrency", &filters);
        assert_eq!(out["max_passing_concurrency"].as_i64(), Some(7));
        assert_eq!(out["first_failing_concurrency"].as_i64(), Some(19));
        assert_eq!(out["monotonicity_check"].as_bool(), Some(true));
        let breach = &out["first_failing_breach"];
        assert_eq!(breach["metric_tag"].as_str(), Some("inter_token_latency"));
        assert_eq!(breach["op"].as_str(), Some("lt"));
        assert_eq!(breach["threshold"].as_f64(), Some(12.0));
        assert!(breach["observed"].as_f64().unwrap() > 12.0);
    }

    #[test]
    fn mean_fallback_when_percentile_absent() {
        let sweep = serde_json::json!({
            "per_combination_metrics": [
                {"parameters": {"concurrency": 1}, "metrics": {"inter_token_latency": {"mean": 5.0}}},
                {"parameters": {"concurrency": 19}, "metrics": {"inter_token_latency": {"mean": 20.0}}},
            ],
        });
        let filters = vec![SlaFilter {
            metric_tag: "inter_token_latency".into(),
            stat: "p95".into(),
            op: SlaOp::Lt,
            threshold: 12.0,
        }];
        let out = process(&sweep, "phases.profiling.concurrency", &filters);
        assert_eq!(out["max_passing_concurrency"].as_i64(), Some(1));
        assert_eq!(out["first_failing_concurrency"].as_i64(), Some(19));
    }
}
