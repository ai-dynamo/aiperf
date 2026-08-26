// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The dashboard HTTP API: cross-run browse over the [`super::index`] plane.
//!
//! Browse-plane routes: list every run (session + historical), fetch a run's full
//! `native-v2.json`, and fetch its projected metric summary. The live in-flight
//! streaming plane is `/api/live` over SSE, heartbeat-fed by [`super::live`].
//! Static UI assets are served from the [`super::assets`] fallback.

use std::path::Path;

use axum::Json;
use axum::extract::{Path as AxPath, State};
use axum::http::StatusCode;
use axum::routing::get;
use serde_json::{Value, json};

use super::AppState;
use super::index::{self, RunEntry};

/// Build the dashboard router over the shared [`AppState`].
pub fn router(state: AppState) -> axum::Router {
    axum::Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/api/meta", get(meta))
        .route("/api/runs", get(list_runs))
        .route("/api/runs/{id}", get(get_run))
        .route("/api/runs/{id}/summary", get(get_run_summary))
        .route("/api/live", get(super::live::live_stream))
        .fallback(super::assets::serve)
        .with_state(state)
}

/// Snapshot the merged cross-run list under the current state.
fn snapshot(state: &AppState) -> Vec<RunEntry> {
    let session = state.session.lock().expect("session mutex").clone();
    index::merged(
        &session,
        state.results_root.as_deref(),
        state.scan_max_depth,
    )
}

/// `GET /api/meta` — server + session identity for the UI header.
async fn meta(State(state): State<AppState>) -> Json<Value> {
    let session_count = state.session.lock().expect("session mutex").len();
    Json(json!({
        "service": "aiperf-dashboard",
        "started_unix": state.started_unix,
        "results_root": state.results_root.as_ref().map(|p| p.display().to_string()),
        "session_runs": session_count,
    }))
}

/// `GET /api/runs` — the merged cross-run list (session + historical).
async fn list_runs(State(state): State<AppState>) -> Json<Vec<RunEntry>> {
    Json(snapshot(&state))
}

/// Resolve a run id to its [`RunEntry`] in the current snapshot.
fn find_run(state: &AppState, id: &str) -> Option<RunEntry> {
    snapshot(state).into_iter().find(|r| r.id == id)
}

/// `GET /api/runs/:id` — the run's full committed `native-v2.json`.
async fn get_run(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let run = find_run(&state, &id).ok_or((StatusCode::NOT_FOUND, format!("unknown run {id}")))?;
    let path = run
        .report_path
        .ok_or((StatusCode::NOT_FOUND, format!("run {id} has no report")))?;
    let report = crate::sweep::aggregate::read_report_path(&path).ok_or((
        StatusCode::NOT_FOUND,
        format!("could not read report for {id}"),
    ))?;
    Ok(Json(report))
}

/// `GET /api/runs/:id/summary` — the run's projected per-metric stats (the same
/// projection the sweep aggregate uses), plus its identity/headline.
async fn get_run_summary(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let run = find_run(&state, &id).ok_or((StatusCode::NOT_FOUND, format!("unknown run {id}")))?;
    let report = run
        .report_path
        .as_deref()
        .and_then(crate::sweep::aggregate::read_report_path);
    let metrics: serde_json::Map<String, Value> = report
        .as_ref()
        .map(|r| {
            crate::sweep::aggregate::project_summary(r)
                .into_iter()
                .map(|(tag, stats)| (tag, Value::Object(with_nested_percentiles(stats))))
                .collect()
        })
        .unwrap_or_default();
    Ok(Json(json!({
        "run": {
            "id": run.id,
            "label": run.label,
            "artifact_dir": run.artifact_dir,
            "success": run.success,
            "trial": run.trial,
            "sweep_id": run.sweep_id,
            "source": run.source,
        },
        "headline": run.headline,
        "metrics": metrics,
    })))
}

/// The percentile fields the projection flattens onto a metric (Python's
/// `_json_metric_to_stats` order).
const PERCENTILE_KEYS: &[&str] = &["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"];

/// `project_summary` flattens percentiles onto the metric as top-level `pNN` keys
/// (the byte-exact Python sweep-aggregate shape). The dashboard reads them nested,
/// so mirror the present `pNN` values into a `percentiles` object (leaving the flat
/// keys in place) — a UI convenience layered over the untouched projection.
fn with_nested_percentiles(
    mut stats: serde_json::Map<String, Value>,
) -> serde_json::Map<String, Value> {
    let mut pct = serde_json::Map::new();
    for key in PERCENTILE_KEYS {
        if let Some(v) = stats.get(*key)
            && v.is_number()
        {
            pct.insert((*key).to_string(), v.clone());
        }
    }
    if !pct.is_empty() {
        stats.insert("percentiles".to_string(), Value::Object(pct));
    }
    stats
}

/// Whether `root` currently holds any browseable run (a `native-v2.json` anywhere
/// under it). Used by callers to decide whether serving is worthwhile.
pub fn has_runs(root: &Path, max_depth: usize) -> bool {
    !index::scan_disk(root, max_depth).is_empty()
}
