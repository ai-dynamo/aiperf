// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native adaptive-search recipes (`--search-recipe`).
//!
//! Ports Python's `aiperf.search_recipes` grid recipes. A **grid** recipe
//! (`concurrency-ramp`, `prefill-ttft-curve`, `decode-itl-curve`, `pareto-sweep`)
//! expands its search space into a STATIC sweep at config time — a log-spaced
//! value list over a config path — which the existing sweep engine then runs.
//! Because those axes coincide with the comma-list sweep flags, a grid recipe is
//! implemented by precomputing the value list(s) and injecting them as the
//! corresponding flag(s); the normal `sweep::expand` path takes over unchanged.
//!
//! `bayes` recipes (Bayesian optimization over concurrency) and the
//! `smooth_isotonic` recipe run a dynamic ask-tell loop rather than a static
//! sweep; those are handled by the search runtime (in-process pyo3 optuna for the
//! ML planner), not here.

use crate::flags::ProfileFlags;

/// `steps` log-spaced integer values in `[lo, hi]` inclusive, endpoints forced,
/// rounding duplicates collapsed, ascending — byte-exact port of Python
/// `aiperf.search_recipes.builtins::_logspace_int_steps`.
pub fn logspace_int_steps(lo: f64, hi: f64, steps: i64) -> anyhow::Result<Vec<i64>> {
    anyhow::ensure!(steps >= 2, "search steps must be >= 2 (got {steps})");
    anyhow::ensure!(lo > 0.0, "search lower bound must be > 0 (got {lo})");
    anyhow::ensure!(hi > lo, "search upper bound ({hi}) must be > lower ({lo})");
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    let mut vals: Vec<i64> = (0..steps)
        .map(|i| {
            let v = (log_lo + (log_hi - log_lo) * i as f64 / (steps - 1) as f64).exp();
            // `round-half-to-even`? Python `round()` is banker's rounding; but for
            // the positive log-spaced magnitudes here it matches `.round()` in
            // every builtin-recipe range. Clamp to >= 1 (defends lo < 1).
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
    } else {
        // Exactly .5 → round to even.
        if (floor as i64) % 2 == 0 { floor } else { floor + 1.0 }
    }
}

/// The result of expanding a grid search recipe: the swept-flag mutations to
/// apply before the normal sweep path runs.
pub struct GridExpansion {
    /// Comma-list value to inject into `--concurrency` (when the recipe sweeps it).
    pub concurrency: Option<String>,
    /// Comma-list value to inject into `--isl` (when the recipe sweeps it).
    pub isl: Option<String>,
    /// Comma-list value to inject into `--osl` (when the recipe sweeps it).
    pub osl: Option<String>,
}

/// Expand a grid search recipe into flag mutations. Returns `Ok(None)` when the
/// recipe is not a (yet-supported) grid recipe. Errors on an unknown recipe.
pub fn expand_grid_recipe(flags: &ProfileFlags) -> anyhow::Result<Option<GridExpansion>> {
    let Some(recipe) = flags.search_recipe.as_deref() else {
        return Ok(None);
    };
    let join = |v: &[i64]| v.iter().map(i64::to_string).collect::<Vec<_>>().join(",");
    match recipe {
        "concurrency-ramp" => {
            let lo = flags.concurrency_min.unwrap_or(1) as f64;
            let hi = flags.concurrency_max.unwrap_or(1000) as f64;
            let steps = flags.concurrency_steps.unwrap_or(8);
            let values = logspace_int_steps(lo, hi, steps)?;
            Ok(Some(GridExpansion {
                concurrency: Some(join(&values)),
                isl: None,
                osl: None,
            }))
        }
        other => anyhow::bail!(
            "search recipe {other:?} is not yet supported natively (grid recipes: \
             concurrency-ramp)"
        ),
    }
}

impl GridExpansion {
    /// Apply the recipe's swept-flag mutations onto a clone of the base flags.
    pub fn apply(&self, base: &ProfileFlags) -> ProfileFlags {
        let mut flags = base.clone();
        if let Some(c) = &self.concurrency {
            flags.concurrency = Some(c.clone());
        }
        if let Some(i) = &self.isl {
            flags.isl = Some(i.clone());
        }
        if let Some(o) = &self.osl {
            flags.osl = Some(o.clone());
        }
        // The recipe owns the sweep; clear the recipe id so the normal sweep
        // path doesn't recurse.
        flags.search_recipe = None;
        flags
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logspace_matches_python() {
        // _logspace_int_steps(1, 100, 5) -> [1, 3, 10, 32, 100].
        assert_eq!(logspace_int_steps(1.0, 100.0, 5).unwrap(), vec![1, 3, 10, 32, 100]);
        // _logspace_int_steps(1, 1000, 8) -> [1, 3, 7, 19, 52, 139, 373, 1000].
        assert_eq!(
            logspace_int_steps(1.0, 1000.0, 8).unwrap(),
            vec![1, 3, 7, 19, 52, 139, 373, 1000]
        );
    }
}
