// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Sweep orchestration: build one `BenchmarkRun` per `(variation, trial)` cell.
//!
//! Each cell carries the single-run projection plus `sweep_id`, `trial`,
//! `variation`, `artifact_dir`, and `random_seed = base + variation.index`.

use crate::flags::ProfileFlags;
use crate::load::Inputs;
use crate::model::BenchmarkRun;
use crate::sweep::artifact_dir::{self, IterationOrder};
use crate::sweep::{Expansion, Variation};

/// Default base run seed.
pub const DEFAULT_SWEEP_SEED: u64 = 42;

/// Per-variation seed policy.
#[derive(Clone, Copy)]
pub struct SeedPolicy {
    /// Base seed: `None` disables seeding (`--no-set-consistent-seed`, no
    /// `--random-seed`); otherwise the run seed (authored or the `42` default).
    pub base: Option<u64>,
    /// When true, every variation shares `base` (`--parameter-sweep-same-seed`);
    /// otherwise variation N gets `base + N`.
    pub same_seed: bool,
    /// When true, trial T additionally offsets the seed by `T`
    /// (`--vary-seed-per-trial`).
    pub vary_per_trial: bool,
}

impl SeedPolicy {
    /// The seed for variation `index` and trial `trial` (`None` when seeding is
    /// disabled).
    pub fn seed(&self, index: usize, trial: u32) -> Option<u64> {
        self.base.map(|b| {
            let variation = if self.same_seed { 0 } else { index as u64 };
            let trial_offset = if self.vary_per_trial {
                u64::from(trial)
            } else {
                0
            };
            b + variation + trial_offset
        })
    }
}

/// One planned cell: the fully-built run request plus its coordinates.
pub struct Cell {
    /// Zero-based variation index.
    pub index: usize,
    /// Zero-based trial number.
    pub trial: u32,
    /// Variation label (`"path=value, ..."`).
    pub label: String,
    /// The built run (envelope stamped). Used for per-cell planning metadata, the
    /// sweep aggregate (`variation`/`artifact_dir`), and the parity oracle.
    pub run: BenchmarkRun,
    /// Authoring inputs the cell ships on the `--execute` wire so the runtime
    /// resolves at execute time. Every planned cell (flag/YAML sweeps, multi-run
    /// trials, and recipe sweeps) carries `Some`; the CLI never lowers before the
    /// child launch.
    pub inputs: Option<Inputs>,
}

/// Build every `(variation, trial)` cell for a sweep plan.
///
/// `sweep_id` is a caller-supplied stable id (one per plan). Each cell clones the
/// base flags, applies the variation overrides, resolves the single-run request,
/// and stamps the sweep envelope + artifact dir + seed.
#[allow(clippy::too_many_arguments)]
pub fn plan_cells(
    base_flags: &ProfileFlags,
    expansion: &Expansion,
    trials: u32,
    order: IterationOrder,
    sweep_id: &str,
    seed: SeedPolicy,
    disable_warmup_after_first: bool,
    resolve: impl Fn(&ProfileFlags) -> anyhow::Result<BenchmarkRun>,
) -> anyhow::Result<Vec<Cell>> {
    let mut cells = Vec::new();
    for trial in 0..trials {
        for variation in &expansion.variations {
            let mut flags = variation.apply(base_flags);
            if let Some(entries) = expansion.entries_override {
                flags.num_dataset_entries = Some(entries.to_string());
            }
            // Authoring inputs the cell ships on the wire (the runtime resolves at
            // `--execute`), plus a CLI-side resolved run for planning metadata and the
            // parity oracle. Both derive from the same per-variation flags.
            let mut inputs = crate::load::resolve_inputs(&flags)?;
            let mut run = resolve(&flags)?;
            let base = run.artifact_dir.clone();
            let dir = artifact_dir::resolve(
                &base,
                expansion.is_sweep,
                trials,
                &variation.dir_name,
                trial,
                order,
            );
            stamp(&mut run, variation, trial, sweep_id, seed, &dir);
            // Mirror the per-cell artifact dir + run seed onto the authoring inputs so
            // the runtime's resolution reproduces the CLI-side run byte-for-byte.
            inputs.artifact_dir = dir;
            inputs.random_seed = seed.seed(variation.index, trial);
            if trial > 0 && disable_warmup_after_first {
                drop_warmup(&mut run);
                // Match the resolved run's dropped warmup phase on the authoring side.
                inputs.warmup = None;
            }
            cells.push(Cell {
                index: variation.index,
                trial,
                label: variation.label.clone(),
                run,
                inputs: Some(inputs),
            });
        }
    }
    Ok(cells)
}

/// Render a variation's rendered scalar back to its config-typed JSON: an integer
/// text (`"10"`) becomes a JSON integer, a fractional text (`"2.0"`) a JSON float
/// to preserve the config value type.
fn axis_value_json(rendered: &str) -> serde_json::Value {
    if !rendered.contains('.')
        && let Ok(i) = rendered.parse::<i64>()
    {
        return serde_json::Value::from(i);
    }
    if let Ok(f) = rendered.parse::<f64>() {
        return serde_json::Value::from(f);
    }
    serde_json::Value::String(rendered.to_string())
}

/// Remove the warmup phase from a run's `cfg.phases` (used for trials past the
/// first when `disable_warmup_after_first` is set).
fn drop_warmup(run: &mut BenchmarkRun) {
    if let Some(phases) = run.cfg.phases.as_mut() {
        phases.retain(|p| p.common.name != "warmup");
    }
}

/// Stamp the sweep envelope onto a per-cell run.
///
/// Every cell in a sweep or multi-run plan carries `sweep_id`, `variation`, and
/// `random_seed`, including the base variation
/// (`{index:0,label:"base",values:{}}`). Values render
/// as their config-typed JSON: an integer axis is a JSON number, a float axis a
/// JSON float. `plan_cells` only enters this path for a real sweep or `trials>1`,
/// so unconditional stamping is correct.
fn stamp(
    run: &mut BenchmarkRun,
    variation: &Variation,
    trial: u32,
    sweep_id: &str,
    seed: SeedPolicy,
    dir: &std::path::Path,
) {
    run.sweep_id = Some(sweep_id.to_string());
    let values: serde_json::Map<String, serde_json::Value> = variation
        .values
        .iter()
        .map(|(k, v)| (k.clone(), axis_value_json(v)))
        .collect();
    run.variation = Some(serde_json::json!({
        "index": variation.index,
        "label": variation.label,
        "values": values,
    }));
    run.random_seed = seed.seed(variation.index, trial);
    run.trial = trial;
    run.artifact_dir = dir.to_path_buf();
    if let Some(cfg_artifacts) = run.cfg.artifacts.as_mut() {
        // Per-record paths remain relative to the cell's artifact directory.
        let _ = cfg_artifacts;
    }
}
