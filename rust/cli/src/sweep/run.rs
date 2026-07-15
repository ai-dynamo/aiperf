// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Sweep orchestration: build one `BenchmarkRun` per `(variation, trial)` cell.
//!
//! The per-cell run is the byte-exact single-run projection with the sweep
//! envelope stamped: `sweep_id`, `trial`, `variation`, `artifact_dir`, and a
//! `random_seed = base + variation.index`.

use crate::flags::ProfileFlags;
use crate::model::BenchmarkRun;
use crate::sweep::artifact_dir::{self, IterationOrder};
use crate::sweep::{Expansion, Variation};

/// Default base run seed when none is authored (matches Python's config default).
pub const DEFAULT_SWEEP_SEED: u64 = 42;

/// One planned cell: the fully-built run request plus its coordinates.
pub struct Cell {
    /// Zero-based variation index.
    pub index: usize,
    /// Zero-based trial number.
    pub trial: u32,
    /// Variation label (`"path=value, ..."`).
    pub label: String,
    /// The built run (envelope stamped).
    pub run: BenchmarkRun,
}

/// Build every `(variation, trial)` cell for a sweep plan.
///
/// `sweep_id` is a caller-supplied stable id (one per plan). Each cell clones the
/// base flags, applies the variation overrides, resolves the single-run request,
/// and stamps the sweep envelope + artifact dir + seed.
pub fn plan_cells(
    base_flags: &ProfileFlags,
    expansion: &Expansion,
    trials: u32,
    order: IterationOrder,
    sweep_id: &str,
    base_seed: u64,
    resolve: impl Fn(&ProfileFlags) -> anyhow::Result<BenchmarkRun>,
) -> anyhow::Result<Vec<Cell>> {
    let mut cells = Vec::new();
    for trial in 0..trials {
        for variation in &expansion.variations {
            // Each cell resolves the single-run request with the swept scalar
            // pinned; the resolved run's `artifact_dir` is the base target, from
            // which the per-cell dir is derived.
            let mut flags = variation.apply(base_flags);
            // A swept count axis shares one entries pool (`max`) across cells.
            if let Some(entries) = expansion.entries_override {
                flags.num_dataset_entries = Some(entries.to_string());
            }
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
            stamp(&mut run, expansion, variation, trial, sweep_id, base_seed, &dir);
            cells.push(Cell {
                index: variation.index,
                trial,
                label: variation.label.clone(),
                run,
            });
        }
    }
    Ok(cells)
}

/// Stamp the sweep envelope onto a per-cell run.
fn stamp(
    run: &mut BenchmarkRun,
    expansion: &Expansion,
    variation: &Variation,
    trial: u32,
    sweep_id: &str,
    base_seed: u64,
    dir: &std::path::Path,
) {
    if expansion.is_sweep {
        run.sweep_id = Some(sweep_id.to_string());
        let values: serde_json::Map<String, serde_json::Value> = variation
            .values
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect();
        run.variation = Some(serde_json::json!({
            "index": variation.index,
            "label": variation.label,
            "values": values,
        }));
        run.random_seed = Some(base_seed + variation.index as u64);
    }
    run.trial = trial;
    run.artifact_dir = dir.to_path_buf();
    if let Some(cfg_artifacts) = run.cfg.artifacts.as_mut() {
        // Keep `cfg.artifacts.dir` == run.artifact_dir (the runner reprojects
        // relative paths against it). Only the base dir differs per cell; the
        // per-record filenames stay relative.
        let _ = cfg_artifacts;
    }
}
