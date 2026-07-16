// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MLflow run tracker (native Rust, MLflow REST + local FileStore uploader).
//!
//! Ports the Python MLflow exporter (`exporters/mlflow_data_exporter.py`) to the
//! runner: creates/attaches an MLflow run and logs the same params, per-stat
//! metrics (`metric.tag` for avg, `metric.tag.<stat>` for the rest), tags
//! (`aiperf.version`, `benchmark_id`, `aiperf.was_cancelled`, user tags), and
//! uploads the artifact bundle. Two tracking backends are supported directly,
//! without the Python SDK:
//!
//! * `http(s)://` — the MLflow REST API (`/api/2.0/mlflow/*`), with artifact
//!   upload through the `mlflow-artifacts` proxy when the created run's
//!   `artifact_uri` uses the `mlflow-artifacts:` scheme.
//! * `file://` — the on-disk `FileStore` layout MLflow itself reads
//!   (`<root>/<experiment_id>/<run_id>/{meta.yaml,metrics,params,tags,artifacts}`),
//!   so an e2e proof pointed at a local `file://` store works with no server.
//!
//! Parity oracle (read the actual Python, cited at `path:line` throughout): the
//! logged metric-key scheme, param set, and tag set must match the Python
//! exporter for an identical run. Because this sink reads the native-v2
//! [`NativeReport`] rather than Python's `ProfileResults`, the metric *values*
//! come from the report's typed stats while the *keys* reproduce
//! `_build_metric_payload` (`mlflow_data_exporter.py:335`). Config-only params
//! (endpoint/timing/CLI — none of which live in the report) are projected by the
//! Python frontend into [`MlflowExportConfig::params`] and forwarded verbatim.
//!
//! Spec §6: the commit site is synchronous with no ambient tokio runtime, so the
//! REST path runs on a short-lived `current_thread` runtime under one overall
//! [`tokio::time::timeout`] — an unreachable tracking server logs a warning and
//! returns `Err` (best-effort per the dispatcher) rather than hanging shutdown.
//! No `spawn`/`Queue`/subprocess apparatus is used.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::{MetricEntry, NativeReport, ReportStats, ReportValue};

/// Percentile stat keys the Python exporter pushes to MLflow, in
/// `_STAT_FIELDS` order (`mlflow_data_exporter.py:333`). Only these percentiles
/// are logged; any other percentile carried by the report is ignored so the key
/// set matches Python exactly.
const PERCENTILE_STAT_KEYS: &[&str] =
    &["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"];

/// Default hard wall-clock budget for the whole REST conversation (spec §6),
/// mirroring the Python `Environment.MLFLOW.EXPORT_TIMEOUT_SECONDS` default.
const DEFAULT_EXPORT_TIMEOUT_SECONDS: u64 = 60;

/// Per-call MLflow `log-batch` limits (server-enforced): at most 1000 metrics,
/// 100 params, and 100 tags per request. The uploader chunks to stay under them.
const MAX_METRICS_PER_BATCH: usize = 1000;
const MAX_PARAMS_PER_BATCH: usize = 100;
const MAX_TAGS_PER_BATCH: usize = 100;

/// MLflow export policy. `enabled` iff a tracking URI is provided (matching the
/// Python `MLflowConfig.enabled`, `config/mlflow.py:99`).
///
/// The report carries neither the endpoint/timing configuration nor the
/// benchmark identity, so the Python frontend projects those into the fields
/// below (see the module docs and the crate-level projection note). Metric
/// values are read from the report; params/benchmark_id/tags are projected.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MlflowExportConfig {
    /// Whether MLflow tracking is enabled.
    pub enabled: bool,
    /// MLflow tracking URI (`file://…` or `http(s)://…`).
    pub tracking_uri: Option<String>,
    /// Experiment name (Python default `"aiperf"`, `config/mlflow.py:30`).
    pub experiment: Option<String>,
    /// Optional run name.
    pub run_name: Option<String>,
    /// Optional parent run id.
    pub parent_run_id: Option<String>,
    /// User tags to attach to the run (Python `MLflowConfig.tags_dict`).
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
    /// Artifact globs to upload, relative to the artifact dir. When empty, the
    /// Python defaults (`MLflowDefaults.DEFAULT_ARTIFACT_GLOBS`) are used.
    #[serde(default)]
    pub artifact_globs: Vec<String>,
    /// Benchmark identity projected from `run.benchmark_id`; drives the
    /// `benchmark_id` tag and the derived default run name (Python
    /// `_derive_default_run_name`, `mlflow_data_exporter.py:324`). Absent from
    /// the report, so it must be projected.
    pub benchmark_id: Option<String>,
    /// AIPerf package version (`aiperf.__version__`) projected by the frontend
    /// for the `aiperf.version` tag. The native report carries only the Rust
    /// crate version, so the authoritative package version is projected here and
    /// used in preference to `report.aiperf_version` when present.
    pub aiperf_version: Option<String>,
    /// Pre-built param payload projected verbatim from the Python
    /// `_build_param_payload` (`mlflow_data_exporter.py:357`). These are pure
    /// frontend config (endpoint.type/models/urls, output.artifact_directory,
    /// timing.mode, loadgen.*, aiperf.cli_command) and do not appear in the
    /// native report, so the frontend assembles and redacts them.
    #[serde(default)]
    pub params: BTreeMap<String, String>,
    /// Total expected requests, projected from `ProfileResults.total_expected`
    /// (`mlflow_data_exporter.py:351`); absent from the report.
    pub total_expected_requests: Option<f64>,
    /// Overall REST timeout override in seconds (spec §6).
    pub export_timeout_seconds: Option<u64>,
}

/// The MLflow [`Exporter`] (REST or local FileStore).
pub struct MlflowExporter;

impl Exporter for MlflowExporter {
    fn name(&self) -> &'static str {
        "mlflow"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.mlflow.enabled && cfg.mlflow.tracking_uri.is_some()
    }

    fn export(
        &self,
        report: &NativeReport,
        artifact_dir: &Path,
        cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        let mlflow = &cfg.mlflow;
        let tracking_uri = mlflow
            .tracking_uri
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("MLflow tracking URI missing"))?;

        // Payloads are pure functions of the report + projected config, so build
        // them once up front (identical for REST and FileStore backends).
        let plan = ExportPlan::build(report, artifact_dir, mlflow);

        if let Some(root) = file_store_root(tracking_uri) {
            // Local on-disk store: no network, no timeout needed.
            return file_store::write(&root, mlflow, &plan);
        }
        if tracking_uri.starts_with("http://") || tracking_uri.starts_with("https://") {
            return rest::upload(tracking_uri, mlflow, &plan);
        }
        anyhow::bail!(
            "unsupported MLflow tracking URI scheme (expected file://, http://, or https://): {tracking_uri}"
        );
    }
}

/// The fully-resolved set of things to log, independent of the backend.
struct ExportPlan {
    /// Ordered `key -> value` metric payload (`_build_metric_payload`).
    metrics: BTreeMap<String, f64>,
    /// Ordered `key -> value` param payload (forwarded from the frontend).
    params: BTreeMap<String, String>,
    /// Ordered `key -> value` tag payload (`_build_tag_payload`).
    tags: BTreeMap<String, String>,
    /// Resolved run name (never a server-generated placeholder).
    run_name: String,
    /// Artifact files to upload, each with its resolved MLflow artifact subpath.
    artifacts: Vec<ResolvedArtifact>,
}

/// One artifact file paired with the MLflow artifact directory it lands in.
struct ResolvedArtifact {
    /// Absolute source path under the run artifact directory.
    source: PathBuf,
    /// Destination artifact subdirectory (`""`, `plots`, `exports`, or nested).
    artifact_path: String,
    /// Basename recorded under `artifact_path`.
    file_name: String,
}

impl ExportPlan {
    fn build(report: &NativeReport, artifact_dir: &Path, cfg: &MlflowExportConfig) -> Self {
        Self {
            metrics: build_metric_payload(report, cfg),
            params: cfg.params.clone(),
            tags: build_tag_payload(report, cfg),
            run_name: resolve_run_name(cfg),
            artifacts: collect_artifacts(artifact_dir, cfg),
        }
    }
}

/// Resolve the run name: the CLI name, else the benchmark-derived default, else
/// an epoch-stamped fallback (`_derive_default_run_name`,
/// `mlflow_data_exporter.py:324`).
fn resolve_run_name(cfg: &MlflowExportConfig) -> String {
    if let Some(name) = cfg.run_name.as_deref()
        && !name.is_empty()
    {
        return name.to_string();
    }
    derive_default_run_name(cfg.benchmark_id.as_deref())
}

fn derive_default_run_name(benchmark_id: Option<&str>) -> String {
    match benchmark_id {
        Some(id) if !id.is_empty() => format!("aiperf-{}", &id[..id.len().min(8)]),
        _ => format!("aiperf-{}", unix_seconds()),
    }
}

/// Reproduce `_build_metric_payload` (`mlflow_data_exporter.py:335`): for each
/// metric, emit `metric.tag` for `avg` and `metric.tag.<stat>` for the rest,
/// skipping non-finite/absent values. Values come from the report's typed
/// per-metric stats (the first series is the aggregate the Python record
/// mirrors); the KEY scheme is byte-identical to Python.
fn build_metric_payload(report: &NativeReport, cfg: &MlflowExportConfig) -> BTreeMap<String, f64> {
    let mut payload = BTreeMap::new();
    for (name, entry) in &report.metrics {
        let Some(stats) = entry.series.first().map(|series| &series.stats) else {
            continue;
        };
        push_stat_fields(&mut payload, name, stats);
    }

    // `aiperf.completed_requests` = the completed request count. The report's
    // `request_count` metric carries it; Python reads `ProfileResults.completed`
    // (`mlflow_data_exporter.py:350`).
    if let Some(completed) = report
        .metrics
        .get("request_count")
        .and_then(representative_value)
    {
        payload.insert("aiperf.completed_requests".to_string(), completed);
    }
    // `aiperf.total_expected_requests` is not in the native report; the frontend
    // projects it (`mlflow_data_exporter.py:351`).
    if let Some(total) = cfg.total_expected_requests
        && total.is_finite()
    {
        payload.insert("aiperf.total_expected_requests".to_string(), total);
    }
    payload
}

/// Emit every present-and-finite `_STAT_FIELDS` value for one metric under the
/// Python key scheme (bare tag for `avg`, `tag.<stat>` otherwise).
fn push_stat_fields(payload: &mut BTreeMap<String, f64>, tag: &str, stats: &ReportStats) {
    match stats {
        ReportStats::Distribution(dist) => {
            if let Some(avg) = dist.avg {
                put(payload, tag, None, avg);
            }
            if let Some(min) = dist.min {
                put(payload, tag, Some("min"), min);
            }
            if let Some(max) = dist.max {
                put(payload, tag, Some("max"), max);
            }
            if let Some(std) = dist.std {
                put(payload, tag, Some("std"), std);
            }
            if let Some(count) = dist.count {
                payload.insert(format!("{tag}.count"), count as f64);
            }
            for stat in PERCENTILE_STAT_KEYS {
                if let Some(value) = dist.percentiles.get(*stat) {
                    put(payload, tag, Some(stat), *value);
                }
            }
        }
        // Derived/min-max scalars store their single value in the `avg` slot in
        // the Python `JsonMetricResult` (`export_models.py:37`), so log it bare.
        ReportStats::Scalar(scalar) => put(payload, tag, None, scalar.value),
        // Counters (request_count/good_request_count) log their total in the
        // `avg` slot; `rate` is not a `_STAT_FIELD` and is never suffixed.
        ReportStats::Counter(counter) => put(payload, tag, None, counter.total),
        // Server-telemetry histograms expose avg/count/sum/percentiles.
        ReportStats::Histogram(hist) => {
            if let Some(avg) = hist.avg {
                put(payload, tag, None, avg);
            }
            put(payload, tag, Some("sum"), hist.sum);
            payload.insert(format!("{tag}.count"), hist.count as f64);
            for stat in PERCENTILE_STAT_KEYS {
                if let Some(value) = hist.percentiles.get(*stat) {
                    put(payload, tag, Some(stat), *value);
                }
            }
        }
    }
}

/// Insert one finite stat under the Python key scheme (bare tag for `avg`,
/// `tag.<field>` otherwise); non-finite/absent values are dropped.
fn put(payload: &mut BTreeMap<String, f64>, tag: &str, field: Option<&str>, value: ReportValue) {
    if let Some(value) = finite(value) {
        let key = match field {
            None => tag.to_string(),
            Some(field) => format!("{tag}.{field}"),
        };
        payload.insert(key, value);
    }
}

/// The single representative value of a metric's first series (its `avg`/value/
/// total/`avg`), used for `aiperf.completed_requests`.
fn representative_value(entry: &MetricEntry) -> Option<f64> {
    let stats = entry.series.first().map(|series| &series.stats)?;
    match stats {
        ReportStats::Distribution(dist) => dist.avg.and_then(finite),
        ReportStats::Scalar(scalar) => finite(scalar.value),
        ReportStats::Counter(counter) => finite(counter.total),
        ReportStats::Histogram(hist) => hist.avg.and_then(finite),
    }
}

fn finite(value: ReportValue) -> Option<f64> {
    match value {
        ReportValue::Finite(value) if value.is_finite() => Some(value),
        _ => None,
    }
}

/// Reproduce `_build_tag_payload` (`mlflow_data_exporter.py:386`): version,
/// was_cancelled, benchmark_id, then user tags (which may override).
fn build_tag_payload(report: &NativeReport, cfg: &MlflowExportConfig) -> BTreeMap<String, String> {
    let mut tags = BTreeMap::new();
    let aiperf_version = cfg
        .aiperf_version
        .clone()
        .unwrap_or_else(|| report.aiperf_version.clone());
    tags.insert("aiperf.version".to_string(), aiperf_version);
    tags.insert(
        "aiperf.was_cancelled".to_string(),
        report.summary.was_cancelled.to_string(),
    );
    if let Some(id) = cfg.benchmark_id.as_deref()
        && !id.is_empty()
    {
        tags.insert("benchmark_id".to_string(), id.to_string());
    }
    for (key, value) in &cfg.tags {
        tags.insert(key.clone(), value.clone());
    }
    tags
}

/// Enumerate artifact files matching the configured globs under `artifact_dir`,
/// deduped and sorted, each classified into its MLflow artifact subpath
/// (`log_artifacts`/`resolve_artifact_path`, `mlflow_data_exporter.py:94`).
/// Path safety: only files that canonicalize to a descendant of `artifact_dir`
/// are kept, so a glob can never escape the run's artifact tree.
fn collect_artifacts(artifact_dir: &Path, cfg: &MlflowExportConfig) -> Vec<ResolvedArtifact> {
    let globs = resolved_globs(cfg);
    let dir_canon = artifact_dir.canonicalize().ok();
    let mut seen = std::collections::BTreeSet::new();
    let mut relatives: Vec<PathBuf> = Vec::new();
    let mut files = Vec::new();
    walk_files(artifact_dir, artifact_dir, &mut files);
    files.sort();
    for (abs, rel) in files {
        let rel_posix = rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        // The Python exporter writes/uploads its own metadata file separately;
        // never sweep a stale one back in.
        if rel_posix == "mlflow_export.json" {
            continue;
        }
        if !globs.iter().any(|pattern| glob_match(pattern, &rel_posix)) {
            continue;
        }
        if let (Some(dir_canon), Ok(abs_canon)) = (&dir_canon, abs.canonicalize())
            && !abs_canon.starts_with(dir_canon)
        {
            continue;
        }
        if !seen.insert(rel_posix.clone()) {
            continue;
        }
        relatives.push(rel);
    }

    relatives
        .into_iter()
        .map(|rel| {
            let source = artifact_dir.join(&rel);
            let file_name = rel
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            ResolvedArtifact {
                artifact_path: resolve_artifact_path(&rel),
                file_name,
                source,
            }
        })
        .collect()
}

/// Suffixes routed to the `plots` artifact subtree (`_PLOT_SUFFIXES`,
/// `mlflow_data_exporter.py:30`).
const PLOT_SUFFIXES: &[&str] = &["png", "jpg", "jpeg", "svg", "gif", "webp", "html"];

/// Classify one artifact's destination subpath, mirroring
/// `resolve_artifact_path` (`mlflow_data_exporter.py:94`): plots vs exports,
/// with a leading duplicate base segment stripped.
fn resolve_artifact_path(relative: &Path) -> String {
    let is_plot = relative
        .extension()
        .map(|ext| PLOT_SUFFIXES.contains(&ext.to_string_lossy().to_lowercase().as_str()))
        .unwrap_or(false);
    let base = if is_plot { "plots" } else { "exports" };

    let mut parts: Vec<String> = relative
        .parent()
        .map(|parent| {
            parent
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    if parts.first().map(String::as_str) == Some(base) {
        parts.remove(0);
    }
    if parts.is_empty() {
        base.to_string()
    } else {
        format!("{base}/{}", parts.join("/"))
    }
}

fn resolved_globs(cfg: &MlflowExportConfig) -> Vec<String> {
    if cfg.artifact_globs.is_empty() {
        // `MLflowDefaults.DEFAULT_ARTIFACT_GLOBS` (`config/mlflow.py:34`).
        [
            "*.json",
            "*.csv",
            "*.jsonl",
            "*.parquet",
            "*_timeslices.*",
            "**/*.png",
            "**/*.jpg",
            "**/*.jpeg",
            "**/*.svg",
            "**/*.html",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    } else {
        cfg.artifact_globs.clone()
    }
}

/// Recursively collect `(absolute, relative-to-root)` file paths under `dir`.
fn walk_files(root: &Path, dir: &Path, out: &mut Vec<(PathBuf, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_files(root, &path, out);
        } else if path.is_file()
            && let Ok(rel) = path.strip_prefix(root)
        {
            out.push((path.clone(), rel.to_path_buf()));
        }
    }
}

/// Minimal glob matcher over a `/`-separated relative path. Supports `**`
/// (any run of path segments), `*` (any run within a segment), `?` (one
/// non-`/` char), and literal chars — enough for the MLflow default globs.
fn glob_match(pattern: &str, path: &str) -> bool {
    let pat: Vec<&str> = pattern.split('/').collect();
    let text: Vec<&str> = path.split('/').collect();
    segment_match(&pat, &text)
}

fn segment_match(pat: &[&str], text: &[&str]) -> bool {
    match pat.first() {
        None => text.is_empty(),
        Some(&"**") => {
            // `**` matches zero or more leading segments.
            (0..=text.len()).any(|skip| segment_match(&pat[1..], &text[skip..]))
        }
        Some(seg) => {
            if let Some(first) = text.first() {
                wildcard_match(seg, first) && segment_match(&pat[1..], &text[1..])
            } else {
                false
            }
        }
    }
}

/// Single-segment wildcard match with `*` and `?` (no `/`).
fn wildcard_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    wildcard_rec(&p, &t)
}

fn wildcard_rec(p: &[char], t: &[char]) -> bool {
    match p.first() {
        None => t.is_empty(),
        Some('*') => (0..=t.len()).any(|skip| wildcard_rec(&p[1..], &t[skip..])),
        Some('?') => !t.is_empty() && wildcard_rec(&p[1..], &t[1..]),
        Some(&c) => t.first() == Some(&c) && wildcard_rec(&p[1..], &t[1..]),
    }
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Interpret a `file://` tracking URI as a local FileStore root path.
fn file_store_root(tracking_uri: &str) -> Option<PathBuf> {
    let rest = tracking_uri.strip_prefix("file://")?;
    // `file:///abs/path` -> `/abs/path`; a `localhost` authority is tolerated.
    let path = rest.strip_prefix("localhost").unwrap_or(rest);
    Some(PathBuf::from(path))
}

// ---------------------------------------------------------------------------
// REST backend
// ---------------------------------------------------------------------------

mod rest {
    use super::{
        DEFAULT_EXPORT_TIMEOUT_SECONDS, ExportPlan, MAX_METRICS_PER_BATCH, MAX_PARAMS_PER_BATCH,
        MAX_TAGS_PER_BATCH, MlflowExportConfig, ResolvedArtifact, unix_millis,
    };
    use std::sync::Arc;
    use std::time::Duration;

    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::{Method, Request};
    use hyper_util::rt::TokioIo;

    /// Drive the whole MLflow REST conversation under one hard timeout on a
    /// short-lived `current_thread` runtime (spec §6).
    pub(super) fn upload(
        tracking_uri: &str,
        cfg: &MlflowExportConfig,
        plan: &ExportPlan,
    ) -> anyhow::Result<()> {
        let base = tracking_uri.trim_end_matches('/').to_string();
        let timeout = Duration::from_secs(
            cfg.export_timeout_seconds
                .unwrap_or(DEFAULT_EXPORT_TIMEOUT_SECONDS),
        );
        let experiment = cfg
            .experiment
            .clone()
            .unwrap_or_else(|| "aiperf".to_string());
        let parent_run_id = cfg.parent_run_id.clone();

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| anyhow::anyhow!("failed to build MLflow runtime: {e}"))?;

        let result = runtime.block_on(async {
            match tokio::time::timeout(
                timeout,
                run_conversation(&base, &experiment, parent_run_id.as_deref(), plan),
            )
            .await
            {
                Ok(inner) => inner,
                Err(_) => Err(anyhow::anyhow!(
                    "MLflow REST export timed out after {}s",
                    timeout.as_secs()
                )),
            }
        });
        if let Err(error) = &result {
            tracing::warn!("MLflow REST export failed: {error:#}");
        }
        result
    }

    async fn run_conversation(
        base: &str,
        experiment: &str,
        parent_run_id: Option<&str>,
        plan: &ExportPlan,
    ) -> anyhow::Result<()> {
        let experiment_id = resolve_experiment(base, experiment).await?;
        let run = create_run(base, &experiment_id, &plan.run_name, parent_run_id).await?;

        log_batch(base, &run.run_id, plan).await?;
        upload_artifacts(base, &run, &plan.artifacts).await?;
        terminate_run(base, &run.run_id).await?;

        tracing::debug!(
            run_id = run.run_id,
            metrics = plan.metrics.len(),
            artifacts = plan.artifacts.len(),
            "MLflow run logged"
        );
        Ok(())
    }

    /// One created run's identifiers.
    struct RunHandle {
        run_id: String,
        artifact_uri: String,
    }

    async fn resolve_experiment(base: &str, name: &str) -> anyhow::Result<String> {
        // Look up the experiment first. `get-by-name` is a GET in the MLflow REST
        // API (POST returns 405), with the name passed as a query parameter.
        if let Some(id) = get_experiment_by_name(base, name).await? {
            return Ok(id);
        }
        // Not found: create it.
        let url = format!("{base}/api/2.0/mlflow/experiments/create");
        let body = serde_json::json!({ "name": name });
        let (status, bytes) = send_json(Method::POST, &url, body.to_string().into_bytes()).await?;
        if status == 200 {
            let value: serde_json::Value = serde_json::from_slice(&bytes)?;
            return value
                .get("experiment_id")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .ok_or_else(|| anyhow::anyhow!("MLflow create-experiment response missing id"));
        }
        // A concurrent create losing the race (or a name registered between the
        // lookup and the create) surfaces as RESOURCE_ALREADY_EXISTS; re-fetch.
        if let Some(id) = get_experiment_by_name(base, name).await? {
            return Ok(id);
        }
        anyhow::bail!(
            "MLflow create-experiment failed (HTTP {status}): {}",
            String::from_utf8_lossy(&bytes)
        )
    }

    /// GET `experiments/get-by-name`, returning the experiment id when it exists.
    /// A non-200 (e.g. 404 RESOURCE_DOES_NOT_EXIST) maps to `None` so the caller
    /// creates it.
    async fn get_experiment_by_name(base: &str, name: &str) -> anyhow::Result<Option<String>> {
        let encoded: String = url::form_urlencoded::byte_serialize(name.as_bytes()).collect();
        let url =
            format!("{base}/api/2.0/mlflow/experiments/get-by-name?experiment_name={encoded}");
        let (status, bytes) = send_json(Method::GET, &url, Vec::new()).await?;
        if status != 200 {
            return Ok(None);
        }
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        Ok(value
            .get("experiment")
            .and_then(|e| e.get("experiment_id"))
            .and_then(|v| v.as_str())
            .map(str::to_string))
    }

    async fn create_run(
        base: &str,
        experiment_id: &str,
        run_name: &str,
        parent_run_id: Option<&str>,
    ) -> anyhow::Result<RunHandle> {
        let mut tags = vec![serde_json::json!({ "key": "mlflow.runName", "value": run_name })];
        if let Some(parent) = parent_run_id {
            // MLflow encodes the parent linkage as a run tag.
            tags.push(serde_json::json!({ "key": "mlflow.parentRunId", "value": parent }));
        }
        let url = format!("{base}/api/2.0/mlflow/runs/create");
        let body = serde_json::json!({
            "experiment_id": experiment_id,
            "start_time": unix_millis(),
            "run_name": run_name,
            "tags": tags,
        });
        let (status, bytes) = send_json(Method::POST, &url, body.to_string().into_bytes()).await?;
        if status != 200 {
            anyhow::bail!(
                "MLflow create-run failed (HTTP {status}): {}",
                String::from_utf8_lossy(&bytes)
            );
        }
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        let info = value
            .get("run")
            .and_then(|r| r.get("info"))
            .ok_or_else(|| anyhow::anyhow!("MLflow create-run response missing run.info"))?;
        let run_id = info
            .get("run_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("MLflow create-run response missing run_id"))?
            .to_string();
        let artifact_uri = info
            .get("artifact_uri")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        Ok(RunHandle {
            run_id,
            artifact_uri,
        })
    }

    /// Log all metrics/params/tags, chunked to the server's per-call limits.
    /// Each category is sent independently so no single call exceeds a limit.
    async fn log_batch(base: &str, run_id: &str, plan: &ExportPlan) -> anyhow::Result<()> {
        let timestamp = unix_millis();
        let metrics: Vec<serde_json::Value> = plan
            .metrics
            .iter()
            .map(|(key, value)| {
                serde_json::json!({ "key": key, "value": value, "timestamp": timestamp, "step": 0 })
            })
            .collect();
        let params: Vec<serde_json::Value> = plan
            .params
            .iter()
            .map(|(key, value)| serde_json::json!({ "key": key, "value": value }))
            .collect();
        let tags: Vec<serde_json::Value> = plan
            .tags
            .iter()
            .map(|(key, value)| serde_json::json!({ "key": key, "value": value }))
            .collect();

        for chunk in metrics.chunks(MAX_METRICS_PER_BATCH) {
            send_log_batch(base, run_id, chunk, &[], &[]).await?;
        }
        for chunk in params.chunks(MAX_PARAMS_PER_BATCH) {
            send_log_batch(base, run_id, &[], chunk, &[]).await?;
        }
        for chunk in tags.chunks(MAX_TAGS_PER_BATCH) {
            send_log_batch(base, run_id, &[], &[], chunk).await?;
        }
        Ok(())
    }

    async fn send_log_batch(
        base: &str,
        run_id: &str,
        metrics: &[serde_json::Value],
        params: &[serde_json::Value],
        tags: &[serde_json::Value],
    ) -> anyhow::Result<()> {
        if metrics.is_empty() && params.is_empty() && tags.is_empty() {
            return Ok(());
        }
        let url = format!("{base}/api/2.0/mlflow/runs/log-batch");
        let body = serde_json::json!({
            "run_id": run_id,
            "metrics": metrics,
            "params": params,
            "tags": tags,
        });
        let (status, bytes) = send_json(Method::POST, &url, body.to_string().into_bytes()).await?;
        if status != 200 {
            anyhow::bail!(
                "MLflow log-batch failed (HTTP {status}): {}",
                String::from_utf8_lossy(&bytes)
            );
        }
        Ok(())
    }

    /// Upload artifacts through the `mlflow-artifacts` proxy when the run's
    /// artifact store is proxied. Other backends (S3/GCS/local server disk) are
    /// not reachable over the tracking REST API, so upload is skipped with a
    /// warning rather than failing the run.
    async fn upload_artifacts(
        base: &str,
        run: &RunHandle,
        artifacts: &[ResolvedArtifact],
    ) -> anyhow::Result<()> {
        if artifacts.is_empty() {
            return Ok(());
        }
        let Some(prefix) = run.artifact_uri.strip_prefix("mlflow-artifacts:") else {
            tracing::warn!(
                artifact_uri = run.artifact_uri,
                "MLflow artifact store is not the mlflow-artifacts proxy; skipping artifact upload"
            );
            return Ok(());
        };
        // `mlflow-artifacts:/<path>` -> proxy path `<path>`.
        let base_path = prefix.trim_start_matches('/').trim_end_matches('/');
        for artifact in artifacts {
            let dest = join_artifact_dest(base_path, &artifact.artifact_path, &artifact.file_name);
            let url = format!("{base}/api/2.0/mlflow-artifacts/artifacts/{dest}");
            let bytes = match std::fs::read(&artifact.source) {
                Ok(bytes) => bytes,
                Err(error) => {
                    tracing::warn!(
                        path = %artifact.source.display(),
                        "failed to read artifact for MLflow upload: {error}"
                    );
                    continue;
                }
            };
            let (status, body) = send_bytes(Method::PUT, &url, bytes).await?;
            if status != 200 {
                tracing::warn!(
                    dest,
                    "MLflow artifact upload failed (HTTP {status}): {}",
                    String::from_utf8_lossy(&body)
                );
            }
        }
        Ok(())
    }

    fn join_artifact_dest(base_path: &str, artifact_path: &str, file_name: &str) -> String {
        let mut parts: Vec<&str> = Vec::new();
        for segment in [base_path, artifact_path] {
            let segment = segment.trim_matches('/');
            if !segment.is_empty() {
                parts.push(segment);
            }
        }
        parts.push(file_name);
        parts.join("/")
    }

    async fn terminate_run(base: &str, run_id: &str) -> anyhow::Result<()> {
        let url = format!("{base}/api/2.0/mlflow/runs/update");
        let body = serde_json::json!({
            "run_id": run_id,
            "status": "FINISHED",
            "end_time": unix_millis(),
        });
        let (status, bytes) = send_json(Method::POST, &url, body.to_string().into_bytes()).await?;
        if status != 200 {
            anyhow::bail!(
                "MLflow update-run failed (HTTP {status}): {}",
                String::from_utf8_lossy(&bytes)
            );
        }
        Ok(())
    }

    async fn send_json(method: Method, url: &str, body: Vec<u8>) -> anyhow::Result<(u16, Vec<u8>)> {
        send_request(method, url, body, "application/json").await
    }

    async fn send_bytes(
        method: Method,
        url: &str,
        body: Vec<u8>,
    ) -> anyhow::Result<(u16, Vec<u8>)> {
        send_request(method, url, body, "application/octet-stream").await
    }

    /// One request/response over a fresh connection. `http://` uses plain TCP;
    /// `https://` layers tokio-rustls with webpki roots (the same crypto
    /// provider `transport::http` uses).
    async fn send_request(
        method: Method,
        url: &str,
        body: Vec<u8>,
        content_type: &str,
    ) -> anyhow::Result<(u16, Vec<u8>)> {
        let uri: hyper::Uri = url
            .parse()
            .map_err(|e| anyhow::anyhow!("bad URL {url}: {e}"))?;
        let host = uri
            .host()
            .ok_or_else(|| anyhow::anyhow!("URL missing host: {url}"))?
            .to_string();
        let https = uri.scheme_str() == Some("https");
        let port = uri.port_u16().unwrap_or(if https { 443 } else { 80 });
        let host_header = match uri.port_u16() {
            Some(p) => format!("{host}:{p}"),
            None => host.clone(),
        };
        let target = uri
            .path_and_query()
            .map(|pq| pq.as_str())
            .unwrap_or("/")
            .to_string();

        let request = Request::builder()
            .method(method)
            .uri(target)
            .header(hyper::header::HOST, host_header)
            .header(hyper::header::CONTENT_TYPE, content_type)
            .header(hyper::header::CONTENT_LENGTH, body.len())
            .body(Full::new(Bytes::from(body)))
            .map_err(|e| anyhow::anyhow!("failed to build request: {e}"))?;

        let tcp = tokio::net::TcpStream::connect((host.as_str(), port))
            .await
            .map_err(|e| anyhow::anyhow!("connect to {host}:{port} failed: {e}"))?;

        if https {
            let connector = tokio_rustls::TlsConnector::from(tls_config());
            let server_name = rustls::pki_types::ServerName::try_from(host.clone())
                .map_err(|e| anyhow::anyhow!("invalid TLS server name {host}: {e}"))?;
            let stream = connector
                .connect(server_name, tcp)
                .await
                .map_err(|e| anyhow::anyhow!("TLS handshake failed: {e}"))?;
            drive(TokioIo::new(stream), request).await
        } else {
            drive(TokioIo::new(tcp), request).await
        }
    }

    async fn drive<I>(io: I, request: Request<Full<Bytes>>) -> anyhow::Result<(u16, Vec<u8>)>
    where
        I: hyper::rt::Read + hyper::rt::Write + Unpin + Send + 'static,
    {
        let (mut sender, connection) = hyper::client::conn::http1::handshake(io)
            .await
            .map_err(|e| anyhow::anyhow!("HTTP handshake failed: {e}"))?;
        // Drive the connection concurrently; it completes when the response body
        // is consumed. On a current_thread runtime this runs on the same thread.
        let pump = tokio::spawn(async move {
            let _ = connection.await;
        });
        let response = sender
            .send_request(request)
            .await
            .map_err(|e| anyhow::anyhow!("request failed: {e}"))?;
        let status = response.status().as_u16();
        let bytes = response
            .into_body()
            .collect()
            .await
            .map_err(|e| anyhow::anyhow!("reading response failed: {e}"))?
            .to_bytes()
            .to_vec();
        pump.abort();
        Ok((status, bytes))
    }

    fn tls_config() -> Arc<rustls::ClientConfig> {
        let mut roots = rustls::RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        // Match `transport::http`'s aws-lc-rs provider so both TLS clients share
        // one crypto backend within the process.
        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        let config = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .expect("aws-lc supports rustls safe default protocol versions")
            .with_root_certificates(roots)
            .with_no_client_auth();
        Arc::new(config)
    }
}

// ---------------------------------------------------------------------------
// Local FileStore backend
// ---------------------------------------------------------------------------

mod file_store {
    use super::{ExportPlan, MlflowExportConfig, ResolvedArtifact, unix_millis};
    use std::path::Path;

    /// Write one run into the MLflow on-disk `FileStore` layout under `root`,
    /// matching `store/tracking/file_store.py`: resolve/create the experiment
    /// directory, create the run directory with `meta.yaml` and the
    /// `metrics/params/tags` subtrees, write each fact as a file, copy artifacts,
    /// then stamp the run FINISHED.
    pub(super) fn write(
        root: &Path,
        cfg: &MlflowExportConfig,
        plan: &ExportPlan,
    ) -> anyhow::Result<()> {
        std::fs::create_dir_all(root)?;
        let experiment = cfg.experiment.as_deref().unwrap_or("aiperf");
        let experiment_id = resolve_experiment(root, experiment)?;

        let run_id = new_run_id();
        let run_dir = root.join(&experiment_id).join(&run_id);
        for sub in ["metrics", "params", "tags", "artifacts"] {
            std::fs::create_dir_all(run_dir.join(sub))?;
        }

        let start_time = unix_millis();
        let artifact_uri = run_dir.join("artifacts").display().to_string();

        // Metrics: `metrics/<key>` holds `<timestamp> <value> <step>` lines
        // (`file_store.py:1101`).
        for (key, value) in &plan.metrics {
            std::fs::write(
                run_dir.join("metrics").join(key),
                format!("{start_time} {value} 0\n"),
            )?;
        }
        // Params/tags: one file per key holding the raw value (`file_store.py`).
        for (key, value) in &plan.params {
            std::fs::write(run_dir.join("params").join(key), value)?;
        }
        let mut tags = plan.tags.clone();
        // MLflow records the run name as a tag as well.
        tags.insert("mlflow.runName".to_string(), plan.run_name.clone());
        for (key, value) in &tags {
            std::fs::write(run_dir.join("tags").join(key), value)?;
        }

        copy_artifacts(&run_dir.join("artifacts"), &plan.artifacts)?;

        // `meta.yaml` for the run (`_make_persisted_run_info_dict`,
        // `file_store.py:149`). Status 3 == FINISHED (RunStatus proto).
        let end_time = unix_millis();
        let meta = run_meta_yaml(
            &run_id,
            &experiment_id,
            &plan.run_name,
            &artifact_uri,
            start_time,
            end_time,
        );
        std::fs::write(run_dir.join("meta.yaml"), meta)?;
        Ok(())
    }

    /// Find an existing experiment directory by name, else create one with a
    /// fresh integer id (`file_store.py:461`).
    fn resolve_experiment(root: &Path, name: &str) -> anyhow::Result<String> {
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                if entry.file_name().to_string_lossy().starts_with('.') {
                    continue;
                }
                if let Ok(meta) = std::fs::read_to_string(path.join("meta.yaml"))
                    && yaml_field(&meta, "name").as_deref() == Some(name)
                    && let Some(id) = yaml_field(&meta, "experiment_id")
                {
                    return Ok(id);
                }
            }
        }
        let experiment_id = new_experiment_id();
        let exp_dir = root.join(&experiment_id);
        std::fs::create_dir_all(&exp_dir)?;
        let creation_time = unix_millis();
        let artifact_location = exp_dir.display().to_string();
        let meta = experiment_meta_yaml(&experiment_id, name, &artifact_location, creation_time);
        std::fs::write(exp_dir.join("meta.yaml"), meta)?;
        Ok(experiment_id)
    }

    fn copy_artifacts(dest_root: &Path, artifacts: &[ResolvedArtifact]) -> anyhow::Result<()> {
        for artifact in artifacts {
            let dir = if artifact.artifact_path.is_empty() {
                dest_root.to_path_buf()
            } else {
                dest_root.join(&artifact.artifact_path)
            };
            std::fs::create_dir_all(&dir)?;
            let dest = dir.join(&artifact.file_name);
            if let Err(error) = std::fs::copy(&artifact.source, &dest) {
                tracing::warn!(
                    path = %artifact.source.display(),
                    "failed to copy artifact into MLflow FileStore: {error}"
                );
            }
        }
        Ok(())
    }

    /// Random 32-hex run id, matching MLflow's `uuid.uuid4().hex`.
    fn new_run_id() -> String {
        uuid::Uuid::new_v4().simple().to_string()
    }

    /// Random positive integer experiment id, matching MLflow's
    /// `_generate_unique_integer_id`.
    fn new_experiment_id() -> String {
        // 15 decimal digits stays within MLflow's random-int range while
        // remaining collision-free for practical use.
        let value = uuid::Uuid::new_v4().as_u128() % 1_000_000_000_000_000;
        format!("{value}")
    }

    fn run_meta_yaml(
        run_id: &str,
        experiment_id: &str,
        run_name: &str,
        artifact_uri: &str,
        start_time: u64,
        end_time: u64,
    ) -> String {
        // Keys mirror `_make_persisted_run_info_dict`; status 3 == FINISHED.
        format!(
            "artifact_uri: {artifact_uri}\n\
             end_time: {end_time}\n\
             entry_point_name: ''\n\
             experiment_id: '{experiment_id}'\n\
             lifecycle_stage: active\n\
             run_id: {run_id}\n\
             run_name: {run_name}\n\
             run_uuid: {run_id}\n\
             source_name: ''\n\
             source_type: 4\n\
             source_version: ''\n\
             start_time: {start_time}\n\
             status: 3\n\
             user_id: aiperf\n"
        )
    }

    fn experiment_meta_yaml(
        experiment_id: &str,
        name: &str,
        artifact_location: &str,
        creation_time: u64,
    ) -> String {
        format!(
            "artifact_location: {artifact_location}\n\
             creation_time: {creation_time}\n\
             experiment_id: '{experiment_id}'\n\
             last_update_time: {creation_time}\n\
             lifecycle_stage: active\n\
             name: {name}\n"
        )
    }

    /// Extract a scalar `key: value` field from a minimal YAML document,
    /// stripping surrounding quotes. Sufficient for the flat meta files we read.
    fn yaml_field(yaml: &str, key: &str) -> Option<String> {
        for line in yaml.lines() {
            let line = line.trim_end();
            if let Some(rest) = line.strip_prefix(&format!("{key}:")) {
                let value = rest.trim().trim_matches('\'').trim_matches('"');
                return Some(value.to_string());
            }
        }
        None
    }
}

#[cfg(test)]
mod tests;
