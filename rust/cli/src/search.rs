// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native adaptive-search recipes (`--search-recipe`).
//!
//! Ports Python's `aiperf.search_recipes` GRID recipes (`concurrency-ramp`,
//! `prefill-ttft-curve`, `decode-itl-curve`). A grid recipe expands its search
//! space into a STATIC grid sweep at config time — log-spaced value lists over
//! config paths — which is then run like any sweep. Because a recipe sweeps
//! CONFIG paths (a scalar `datasets.main.prompts.isl = N` becomes `{value:N}`,
//! not the `--isl` mean), the recipe path resolves the base run once and mutates
//! the built `cfg` per variation, mirroring Python's raw-config override.
//!
//! `bayes` recipes (Bayesian optimization) and the `smooth_isotonic` recipe run
//! a dynamic ask-tell loop rather than a static sweep; those are handled by the
//! search runtime (in-process pyo3 optuna), not here.

use serde_json::Value;

use crate::flags::ProfileFlags;

/// How a recipe axis value maps onto the built `cfg`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AxisKind {
    /// Set the `profiling` phase's `concurrency` (integer).
    PhaseConcurrency,
    /// Replace `datasets[0].prompts.isl` with a fixed scalar `{value:N}`.
    IslScalar,
    /// Replace `datasets[0].prompts.osl` with a fixed scalar `{value:N}`.
    OslScalar,
}

/// One recipe sweep axis: its config dotted path (for the label), directory
/// segment, the log-spaced values, and how the value maps onto `cfg`.
pub struct RecipeAxis {
    pub path: &'static str,
    pub seg: &'static str,
    pub values: Vec<i64>,
    pub kind: AxisKind,
}

/// An expanded grid recipe: axes sorted by dotted path (label/order stability).
pub struct RecipeSweep {
    pub axes: Vec<RecipeAxis>,
}

/// One expanded recipe variation.
pub struct RecipeVariation {
    pub index: usize,
    pub label: String,
    pub dir_name: String,
    /// `(kind, value)` overrides to apply to the built `cfg`.
    pub overrides: Vec<(AxisKind, i64)>,
    /// `(dotted_path, value)` for the stamped `variation.values`.
    pub values: Vec<(String, i64)>,
}

/// `steps` log-spaced integer values in `[lo, hi]` inclusive, endpoints forced,
/// rounding duplicates collapsed, ascending — byte-exact port of Python
/// `aiperf.search_recipes.builtins::_logspace_int_steps`.
pub fn logspace_int_steps(lo: f64, hi: f64, steps: i64) -> anyhow::Result<Vec<i64>> {
    anyhow::ensure!(steps >= 2, "search steps must be >= 2 (got {steps})");
    anyhow::ensure!(lo > 0.0, "search lower bound must be > 0 (got {lo})");
    anyhow::ensure!(hi > lo, "search upper bound ({hi}) must be > lower ({lo})");
    let (log_lo, log_hi) = (lo.ln(), hi.ln());
    let mut vals: Vec<i64> = (0..steps)
        .map(|i| {
            let v = (log_lo + (log_hi - log_lo) * i as f64 / (steps - 1) as f64).exp();
            (python_round(v) as i64).max(1)
        })
        .collect();
    vals.sort_unstable();
    vals.dedup();
    Ok(vals)
}

/// Python 3 `round()` — round-half-to-even (banker's rounding).
fn python_round(v: f64) -> f64 {
    let floor = v.floor();
    let diff = v - floor;
    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

/// Expand a grid `--search-recipe` into its axes. `Ok(None)` when no recipe is
/// set; errors on an unknown / not-yet-supported (bayes/isotonic) recipe.
pub fn expand_recipe(flags: &ProfileFlags) -> anyhow::Result<Option<RecipeSweep>> {
    let Some(recipe) = flags.search_recipe.as_deref() else {
        return Ok(None);
    };
    let axes = match recipe {
        "concurrency-ramp" => vec![RecipeAxis {
            path: "phases.profiling.concurrency",
            seg: "concurrency",
            values: logspace_int_steps(
                flags.concurrency_min.unwrap_or(1) as f64,
                flags.concurrency_max.unwrap_or(1000) as f64,
                flags.concurrency_steps.unwrap_or(8),
            )?,
            kind: AxisKind::PhaseConcurrency,
        }],
        "prefill-ttft-curve" => vec![
            RecipeAxis {
                path: "datasets.main.prompts.isl",
                seg: "isl",
                values: logspace_int_steps(
                    flags.isl_min.unwrap_or(256) as f64,
                    flags.isl_max.unwrap_or(32768) as f64,
                    flags.isl_steps.unwrap_or(8),
                )?,
                kind: AxisKind::IslScalar,
            },
            RecipeAxis {
                path: "phases.profiling.concurrency",
                seg: "concurrency",
                values: vec![1],
                kind: AxisKind::PhaseConcurrency,
            },
        ],
        "decode-itl-curve" => vec![
            RecipeAxis {
                path: "phases.profiling.concurrency",
                seg: "concurrency",
                values: logspace_int_steps(
                    flags.concurrency_min.unwrap_or(1) as f64,
                    flags.concurrency_max.unwrap_or(200) as f64,
                    flags.concurrency_steps.unwrap_or(6),
                )?,
                kind: AxisKind::PhaseConcurrency,
            },
            RecipeAxis {
                path: "datasets.main.prompts.osl",
                seg: "osl",
                values: logspace_int_steps(
                    flags.osl_min.unwrap_or(64) as f64,
                    flags.osl_max.unwrap_or(1024) as f64,
                    flags.osl_steps.unwrap_or(4),
                )?,
                kind: AxisKind::OslScalar,
            },
        ],
        other => anyhow::bail!(
            "search recipe {other:?} is not yet supported natively (grid recipes: \
             concurrency-ramp, prefill-ttft-curve, decode-itl-curve)"
        ),
    };
    Ok(Some(RecipeSweep { axes }))
}

impl RecipeSweep {
    /// Cartesian-product expansion of the axes (sorted by dotted path, last axis
    /// fastest — Python `itertools.product` over sorted `sweep_parameters`).
    pub fn expand(&self) -> Vec<RecipeVariation> {
        // Sort axes by dotted path (stable label + combination order).
        let mut order: Vec<usize> = (0..self.axes.len()).collect();
        order.sort_by_key(|&i| self.axes[i].path);

        let mut combos: Vec<Vec<usize>> = vec![vec![]];
        for &ai in &order {
            let mut next = Vec::new();
            for prefix in &combos {
                for vi in 0..self.axes[ai].values.len() {
                    let mut p = prefix.clone();
                    p.push(vi);
                    next.push(p);
                }
            }
            combos = next;
        }

        combos
            .into_iter()
            .enumerate()
            .map(|(index, combo)| {
                let mut label = Vec::new();
                let mut dir = Vec::new();
                let mut overrides = Vec::new();
                let mut values = Vec::new();
                for (&ai, &vi) in order.iter().zip(combo.iter()) {
                    let axis = &self.axes[ai];
                    let v = axis.values[vi];
                    label.push(format!("{}={v}", axis.path));
                    dir.push(format!("{}_{v}", axis.seg));
                    overrides.push((axis.kind, v));
                    values.push((axis.path.to_string(), v));
                }
                RecipeVariation {
                    index,
                    label: label.join(", "),
                    dir_name: dir.join("__"),
                    overrides,
                    values,
                }
            })
            .collect()
    }
}

/// Apply one recipe override onto a built `cfg` value (mirrors Python's raw
/// config override + resolution): concurrency sets the profiling phase's
/// `concurrency`; isl/osl replace the prompts distribution with a fixed scalar.
pub fn apply_override(cfg: &mut Value, kind: AxisKind, value: i64) {
    match kind {
        AxisKind::PhaseConcurrency => {
            if let Some(phases) = cfg.get_mut("phases").and_then(Value::as_array_mut) {
                for p in phases.iter_mut() {
                    if p.get("name").and_then(Value::as_str) == Some("profiling")
                        && let Some(o) = p.as_object_mut()
                    {
                        o.insert("concurrency".into(), Value::from(value));
                    }
                }
            }
        }
        AxisKind::IslScalar | AxisKind::OslScalar => {
            let field = if kind == AxisKind::IslScalar { "isl" } else { "osl" };
            if let Some(prompts) = cfg
                .get_mut("datasets")
                .and_then(Value::as_array_mut)
                .and_then(|d| d.first_mut())
                .and_then(|d| d.get_mut("prompts"))
                .and_then(Value::as_object_mut)
            {
                prompts.insert(
                    field.into(),
                    serde_json::json!({ "value": value as f64 }),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logspace_matches_python() {
        assert_eq!(logspace_int_steps(1.0, 100.0, 5).unwrap(), vec![1, 3, 10, 32, 100]);
        assert_eq!(
            logspace_int_steps(1.0, 1000.0, 8).unwrap(),
            vec![1, 3, 7, 19, 52, 139, 373, 1000]
        );
        assert_eq!(
            logspace_int_steps(256.0, 32768.0, 8).unwrap(),
            vec![256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        );
    }
}
