// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MLflow run tracker (native Rust, MLflow REST uploader).
//!
//! Ports the Python MLflow exporter (`exporters/mlflow_data_exporter.py`) to the
//! runner: creates/attaches an MLflow run and logs the same params, per-stat
//! metrics (`metric.tag` for avg, `metric.tag.<stat>` for the rest), tags
//! (`aiperf.version`, `benchmark_id`, user tags), and uploads the artifact
//! bundle — all via the MLflow REST API (`/api/2.0/mlflow/*` + artifact upload)
//! rather than the Python SDK. Parity oracle: the logged param/metric-key/tag set
//! and values must match the Python exporter for an identical run, verified
//! against a local `file://`/http tracking store.
//!
//! Spec §6: run under a hard wall-clock timeout on a short-lived runtime so an
//! unreachable tracking server cannot hang shutdown — the durable requirement the
//! Python subprocess apparatus existed to satisfy, without `spawn`/`Queue`/pickle.
//!
//! STATUS: config + gating are wired; the REST uploader body is unimplemented
//! (Worker C).

use std::path::Path;

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::NativeReport;

/// MLflow export policy. `enabled` iff a tracking URI is provided (matching the
/// Python `MLflowConfig.enabled`). Worker C extends this struct as needed.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MlflowExportConfig {
    /// Whether MLflow tracking is enabled.
    pub enabled: bool,
    /// MLflow tracking URI (`file://…` or `http(s)://…`).
    pub tracking_uri: Option<String>,
    /// Experiment name (Python default `"aiperf"`).
    pub experiment: Option<String>,
    /// Optional run name.
    pub run_name: Option<String>,
    /// Optional parent run id.
    pub parent_run_id: Option<String>,
    /// User tags to attach to the run.
    #[serde(default)]
    pub tags: std::collections::BTreeMap<String, String>,
    /// Artifact globs to upload (relative to the artifact dir).
    #[serde(default)]
    pub artifact_globs: Vec<String>,
}

/// The MLflow REST [`Exporter`].
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
        _report: &NativeReport,
        _artifact_dir: &Path,
        _cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        // Worker C: create/attach the MLflow run and log params/metrics/tags +
        // upload artifacts via REST under a hard timeout. Inert at foundation.
        anyhow::bail!("native MLflow uploader not yet implemented");
    }
}
