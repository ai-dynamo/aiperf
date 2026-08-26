// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The cross-run index the dashboard server browses.
//!
//! A [`RunEntry`] is one benchmark run summarized for the run list: its artifact
//! dir, its `native-v2.json` path, and the headline metrics parsed from that report.
//! Two sources feed the index and merge (session wins on a dir collision):
//! - the LIVE session — the orchestrator pushes a [`RunEntry`] as each sweep cell
//!   completes (`profile::run_cells`), so an in-flight sweep is browsable;
//! - HISTORICAL runs on disk — a walk of the results root for `native-v2.json`
//!   files, so past runs (this or prior sessions) show up with no index/db.
//!
//! Metric parsing reuses the sweep aggregate's `native-v2.json` readers
//! (`crate::sweep::aggregate`) so the dashboard and the sweep table can never
//! disagree on how a report projects.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::path::Path;

use serde::Serialize;
use serde_json::Value;

use crate::sweep::aggregate::{HEADLINE, headline_value, read_report_path};

/// The report filename every run commits (`report::finalize_and_write_native_report_json`).
pub const NATIVE_REPORT_NAME: &str = "native-v2.json";

/// One run summarized for the cross-run list.
#[derive(Serialize, Clone)]
pub struct RunEntry {
    /// Stable URL-safe id (hash of the artifact dir) — the run's REST key.
    pub id: String,
    /// Human label: the sweep variation label, else the artifact dir name.
    pub label: String,
    /// Absolute artifact directory.
    pub artifact_dir: String,
    /// Absolute `native-v2.json` path, when the run committed a report.
    pub report_path: Option<String>,
    /// Whether the run succeeded (session runs carry the real flag; a disk run with
    /// a readable report is treated as successful).
    pub success: bool,
    /// Trial index within a repeated sweep cell (0 for single-trial).
    pub trial: u32,
    /// The sweep this run belongs to, when it was part of one.
    pub sweep_id: Option<String>,
    /// Headline metrics (tag → stat value), the same set the sweep table shows.
    pub headline: BTreeMap<String, Option<f64>>,
    /// `"session"` (live, in-memory) or `"disk"` (scanned from the results root).
    pub source: &'static str,
}

/// A stable id for a run keyed by its artifact dir, so the list and the detail
/// endpoints agree without leaking filesystem paths into URLs.
pub fn id_for(artifact_dir: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    artifact_dir.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// The headline metric map for a report (absent report → all `None`).
fn headline_map(report: Option<&Value>) -> BTreeMap<String, Option<f64>> {
    HEADLINE
        .iter()
        .map(|(tag, stat, _label)| {
            (
                (*tag).to_string(),
                report.and_then(|r| headline_value(r, tag, stat)),
            )
        })
        .collect()
}

impl RunEntry {
    /// Build a [`RunEntry`] from a run's artifact dir + report path, parsing the
    /// report's headline metrics.
    pub fn build(
        artifact_dir: &Path,
        report_path: Option<String>,
        label: String,
        success: bool,
        trial: u32,
        sweep_id: Option<String>,
        source: &'static str,
    ) -> Self {
        let report = report_path.as_deref().and_then(read_report_path);
        let dir = artifact_dir.display().to_string();
        Self {
            id: id_for(&dir),
            label,
            artifact_dir: dir,
            report_path,
            success,
            trial,
            sweep_id,
            headline: headline_map(report.as_ref()),
            source,
        }
    }
}

/// Walk `root` recursively for `native-v2.json` reports, one [`RunEntry`] per run.
/// Bounded by `max_depth` so a stray deep tree cannot stall the scan; symlinks are
/// not followed. The run's label is its directory name (variations encode the axis
/// values there via the sweep `dir_name` convention).
pub fn scan_disk(root: &Path, max_depth: usize) -> Vec<RunEntry> {
    let mut out = Vec::new();
    walk(root, max_depth, &mut out);
    out
}

fn walk(dir: &Path, depth_left: usize, out: &mut Vec<RunEntry>) {
    let report = dir.join(NATIVE_REPORT_NAME);
    if report.is_file() {
        let label = dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| dir.display().to_string());
        out.push(RunEntry::build(
            dir,
            Some(report.display().to_string()),
            label,
            true,
            0,
            disk_sweep_id(dir),
            "disk",
        ));
    }
    if depth_left == 0 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        // Don't follow symlinks; only descend real directories.
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            walk(&path, depth_left - 1, out);
        }
    }
}

/// Infer a disk run's sweep membership from the artifact-dir layout: a sweep writes
/// its cells as siblings of a `sweep_aggregate/` directory (`sweep::artifact_dir`),
/// so a run whose parent (or grandparent, for the trial-nested layouts) holds a
/// `sweep_aggregate/` is a sweep cell keyed by that base dir. Returns `None` for a
/// standalone run. (Session runs carry the real `sweep_id`; this recovers grouping
/// for runs browsed off disk, where the per-run report has no sweep field.)
fn disk_sweep_id(run_dir: &Path) -> Option<String> {
    for base in run_dir.ancestors().skip(1).take(2) {
        if base.join("sweep_aggregate").is_dir() {
            return Some(id_for(&base.display().to_string()));
        }
    }
    None
}

/// The merged cross-run list: historical disk runs under `results_root` (if any),
/// overlaid by the live `session` runs (session wins on an artifact-dir collision so
/// an in-flight run's real success/label/sweep replaces its disk shadow). Sorted by
/// label for a stable list.
pub fn merged(
    session: &[RunEntry],
    results_root: Option<&Path>,
    max_depth: usize,
) -> Vec<RunEntry> {
    let historical = results_root
        .map(|root| scan_disk(root, max_depth))
        .unwrap_or_default();
    merge_session_and_historical(session, historical)
}

/// The same merge as [`merged`] over an already-resolved `historical` list, so a
/// non-filesystem source (the operator results API behind
/// `crate::server::HistoricalSource`) reaches the run list through one policy.
pub fn merge_session_and_historical(
    session: &[RunEntry],
    historical: Vec<RunEntry>,
) -> Vec<RunEntry> {
    let mut by_dir: BTreeMap<String, RunEntry> = BTreeMap::new();
    for entry in historical {
        by_dir.insert(entry.artifact_dir.clone(), entry);
    }
    for entry in session {
        by_dir.insert(entry.artifact_dir.clone(), entry.clone());
    }
    let mut runs: Vec<RunEntry> = by_dir.into_values().collect();
    runs.sort_by(|a, b| {
        a.label
            .cmp(&b.label)
            .then(a.artifact_dir.cmp(&b.artifact_dir))
    });
    runs
}
