// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native-Rust server-metrics summary sink: `server_metrics_export.json` + `.csv`.
//!
//! Ports the Python `server_metrics/{json_exporter,csv_exporter}.py` to the
//! runner. The native-v2 report already carries the server-metrics metadata
//! (`summary.server_metrics`) and the labeled/typed series (gauge/counter/
//! histogram) under `report.server_metrics`; this sink serializes them to the
//! two legacy files byte-for-byte. Parity oracle: the current Python
//! `server_metrics_export.{json,csv}` (the JSON carries `# schema_version: 1.0`
//! semantics; the CSV has a `# schema_version: 1.0` header line).
//!
//! STATUS: registered-but-inert stub (Worker D fills the body).

use std::path::Path;

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::NativeReport;

/// Server-metrics summary export policy. Enabled when the run collected server
/// metrics and the operator requested the json/csv summary formats.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerMetricsExportConfig {
    /// Emit `server_metrics_export.json`.
    pub json: bool,
    /// Emit `server_metrics_export.csv`.
    pub csv: bool,
}

/// The server-metrics summary [`Exporter`] (JSON + CSV).
pub struct ServerMetricsExporter;

impl Exporter for ServerMetricsExporter {
    fn name(&self) -> &'static str {
        "server_metrics"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.server_metrics.json || cfg.server_metrics.csv
    }

    fn export(
        &self,
        _report: &NativeReport,
        _artifact_dir: &Path,
        _cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        anyhow::bail!("native server-metrics json/csv sink not yet implemented");
    }
}
