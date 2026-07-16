// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! `search_history.json` writer for the adaptive SLA-search recipes.
//!
//! Port of `aiperf.exporters.search_history::write_search_history`
//! (`src/aiperf/exporters/search_history.py:66-332`): the top-level `config`
//! block, the `iterations` trajectory, `best_trials` (feasibility-first argmax
//! over the single objective — `_compute_best_trials` / `_serialize_trial`,
//! lines 142-183), the 1D `boundary_summary`, the `recipe` id, and the
//! `convergence_reason`. Sits next to `sweep_aggregate/` in the artifact dir.
//!
//! The native 1D-SLA planners (`crate::search::MonotonicPlanner`,
//! `crate::isotonic::SmoothIsotonicPlanner`, `crate::bayes::OptunaPlanner`) hold
//! the iteration history + latched boundary; each search loop in
//! `crate::profile` records one [`IterationRecord`] per probe and calls
//! [`write_search_history`] once the loop converges.

use std::path::Path;

use serde_json::{Map, Value};

use crate::search::SlaFilter;

/// One recorded search iteration (a single `aiperf --execute` probe).
pub struct IterationRecord {
    /// Zero-based iteration index.
    pub iteration_idx: i64,
    /// The probed concurrency (the sole swept dimension value).
    pub concurrency: i64,
    /// The averaged objective-metric value at this probe (`None` = unscored).
    pub objective: Option<f64>,
    /// Whether every configured SLA filter was satisfied at this probe.
    pub feasible: bool,
    /// True iff this probe revealed a non-monotonic SLA boundary.
    pub non_monotonic_warning: bool,
}

/// Static config projected into the search-history `config` block.
pub struct HistoryConfig {
    /// Planner id (`"monotonic_sla"` / `"smooth_isotonic"` / `"bayesian"` / `"optuna"`).
    pub planner: String,
    /// Objective metric tag (`"output_token_throughput"` / `"goodput"`).
    pub objective_metric: String,
    /// Objective stat (`"avg"`).
    pub objective_stat: String,
    /// Optimization direction (`"MAXIMIZE"` / `"MINIMIZE"`).
    pub direction: String,
    /// Max search iterations.
    pub max_iterations: i64,
    /// Initial-point budget (BO planners).
    pub n_initial_points: i64,
    /// The recipe's SLA filters (echoed into `config.sla_filters`).
    pub sla_filters: Vec<SlaFilter>,
    /// Concurrency search lower bound.
    pub lo: i64,
    /// Concurrency search upper bound.
    pub hi: i64,
    /// Search-space dimension kind (`"int"`).
    pub kind: String,
    /// Swept dotted path (`"phases.profiling.concurrency"`).
    pub swept_dim_path: String,
    /// Search random seed, if any.
    pub random_seed: Option<u64>,
}

/// A finite `f64` as a JSON number, else JSON `null`.
fn num(v: f64) -> Value {
    serde_json::Number::from_f64(v)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

/// `[objective]` as a JSON array (`objective_values` vector form), or `null`.
fn objective_values(objective: Option<f64>) -> Value {
    match objective {
        Some(v) => Value::Array(vec![num(v)]),
        None => Value::Null,
    }
}

/// The `variation_values` object: `{ swept_dim_path: concurrency }`.
fn variation_values(config: &HistoryConfig, concurrency: i64) -> Value {
    let mut m = Map::new();
    m.insert(config.swept_dim_path.clone(), Value::from(concurrency));
    Value::Object(m)
}

/// Serialize one trial into a `best_trials` entry (`_serialize_trial`).
fn serialize_trial(r: &IterationRecord, feasible_count: usize, config: &HistoryConfig) -> Value {
    serde_json::json!({
        "iteration_idx": r.iteration_idx,
        "objective_values": objective_values(r.objective),
        "variation_values": variation_values(config, r.concurrency),
        "feasible": r.feasible,
        "feasible_count": feasible_count,
        "pareto_rank": 0,
    })
}

/// Compute `best_trials` for the single-objective recipes (`_compute_best_trials`
/// with `n_obj == 1`): the feasibility-first argmax/argmin over the scored
/// iterations. `null` when no iteration carried an objective value.
fn compute_best_trials(records: &[IterationRecord], config: &HistoryConfig) -> Value {
    let scored: Vec<&IterationRecord> = records.iter().filter(|r| r.objective.is_some()).collect();
    let feasible: Vec<&IterationRecord> = scored.iter().copied().filter(|r| r.feasible).collect();
    let pool: &[&IterationRecord] = if feasible.is_empty() {
        &scored
    } else {
        &feasible
    };
    if pool.is_empty() {
        return Value::Null;
    }
    let maximize = config.direction != "MINIMIZE";
    let best = pool
        .iter()
        .copied()
        .reduce(|a, b| {
            let (av, bv) = (a.objective.unwrap(), b.objective.unwrap());
            // `max`/`min` keep the FIRST element on ties (Python semantics).
            let take_b = if maximize { bv > av } else { bv < av };
            if take_b { b } else { a }
        })
        .unwrap();
    Value::Array(vec![serialize_trial(best, feasible.len(), config)])
}

/// The 1D `boundary_summary` block (`_compute_boundary_summary`): the highest
/// feasible and lowest infeasible swept value. `null` when no iteration ran.
fn boundary_summary(records: &[IterationRecord], config: &HistoryConfig) -> Value {
    let feasible: Vec<&IterationRecord> = records.iter().filter(|r| r.feasible).collect();
    let infeasible: Vec<&IterationRecord> = records.iter().filter(|r| !r.feasible).collect();
    if feasible.is_empty() && infeasible.is_empty() {
        return Value::Null;
    }
    let feasible_max = feasible
        .iter()
        .max_by_key(|r| r.concurrency)
        .map(|r| {
            serde_json::json!({
                "value": r.concurrency,
                "iteration_idx": r.iteration_idx,
                "objective_value": r.objective.map(num).unwrap_or(Value::Null),
            })
        })
        .unwrap_or(Value::Null);
    let infeasible_min = infeasible
        .iter()
        .min_by_key(|r| r.concurrency)
        .map(|r| {
            serde_json::json!({
                "value": r.concurrency,
                "iteration_idx": r.iteration_idx,
                "first_breach": Value::Null,
            })
        })
        .unwrap_or(Value::Null);
    serde_json::json!({
        "swept_dim_path": config.swept_dim_path,
        "feasible_max": feasible_max,
        "infeasible_min": infeasible_min,
    })
}

/// The `config` block (`_build_config_block`).
fn config_block(config: &HistoryConfig) -> Value {
    serde_json::json!({
        "planner": config.planner,
        "objectives": [{
            "metric": config.objective_metric,
            "stat": config.objective_stat,
            "direction": config.direction,
            "threshold": Value::Null,
        }],
        "outcome_constraints": [],
        "max_iterations": config.max_iterations,
        "n_initial_points": config.n_initial_points,
        "random_seed": config.random_seed,
        "search_space": [{
            "path": config.swept_dim_path,
            "lo": config.lo,
            "hi": config.hi,
            "kind": config.kind,
        }],
        "sla_filters": config.sla_filters.iter().map(SlaFilter::to_dict).collect::<Vec<_>>(),
    })
}

/// Write `search_history.json` under `base_dir`. See module docs for the schema.
pub fn write_search_history(
    base_dir: &Path,
    records: &[IterationRecord],
    config: &HistoryConfig,
    convergence_reason: Option<&str>,
    recipe: Option<&str>,
) -> anyhow::Result<()> {
    let iterations: Vec<Value> = records
        .iter()
        .map(|r| {
            serde_json::json!({
                "iteration_idx": r.iteration_idx,
                "variation_values": variation_values(config, r.concurrency),
                "objective_values": objective_values(r.objective),
                "feasible": r.feasible,
                "non_monotonic_warning": r.non_monotonic_warning,
            })
        })
        .collect();

    let payload = serde_json::json!({
        "config": config_block(config),
        "iterations": iterations,
        "best_trials": compute_best_trials(records, config),
        "boundary_summary": boundary_summary(records, config),
        "recipe": recipe,
        "convergence_reason": convergence_reason,
    });

    std::fs::create_dir_all(base_dir)?;
    let path = base_dir.join("search_history.json");
    std::fs::write(&path, serde_json::to_string_pretty(&payload)?)
        .map_err(|e| anyhow::anyhow!("failed to write {}: {e}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> HistoryConfig {
        HistoryConfig {
            planner: "monotonic_sla".into(),
            objective_metric: "output_token_throughput".into(),
            objective_stat: "avg".into(),
            direction: "MAXIMIZE".into(),
            max_iterations: 20,
            n_initial_points: 1,
            sla_filters: Vec::new(),
            lo: 1,
            hi: 1000,
            kind: "int".into(),
            swept_dim_path: "phases.profiling.concurrency".into(),
            random_seed: Some(42),
        }
    }

    #[test]
    fn best_trials_picks_feasible_argmax() {
        let records = vec![
            IterationRecord {
                iteration_idx: 0,
                concurrency: 1,
                objective: Some(10.0),
                feasible: true,
                non_monotonic_warning: false,
            },
            IterationRecord {
                iteration_idx: 1,
                concurrency: 8,
                objective: Some(50.0),
                feasible: true,
                non_monotonic_warning: false,
            },
            IterationRecord {
                iteration_idx: 2,
                concurrency: 64,
                objective: Some(90.0),
                feasible: false,
                non_monotonic_warning: false,
            },
        ];
        let cfg = config();
        let best = compute_best_trials(&records, &cfg);
        let arr = best.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        // Highest-throughput FEASIBLE point (c=8), not the infeasible c=64.
        assert_eq!(
            arr[0]["variation_values"]["phases.profiling.concurrency"].as_i64(),
            Some(8)
        );
        assert_eq!(arr[0]["feasible_count"].as_i64(), Some(2));
    }

    #[test]
    fn best_trials_falls_back_to_all_when_none_feasible() {
        let records = vec![IterationRecord {
            iteration_idx: 0,
            concurrency: 1,
            objective: Some(10.0),
            feasible: false,
            non_monotonic_warning: false,
        }];
        let cfg = config();
        let best = compute_best_trials(&records, &cfg);
        assert_eq!(
            best.as_array().unwrap()[0]["variation_values"]["phases.profiling.concurrency"]
                .as_i64(),
            Some(1)
        );
    }
}
