// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native-Rust accuracy sink: `accuracy_results.csv`.
//!
//! Ports the Python `accuracy/accuracy_data_exporter.py`. The native-v2 report
//! carries the per-problem grading records (`report.accuracy_records`) plus the
//! accuracy analysis summary (`report.accuracy`); this sink writes the per-problem
//! CSV byte-for-byte. Only emitted in accuracy mode. Parity oracle: the current
//! Python `accuracy_results.csv`.
//!
//! STATUS: registered-but-inert stub (Worker F fills the body).

use std::path::Path;

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::NativeReport;

/// Accuracy CSV export policy. Enabled in accuracy mode.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AccuracyCsvExportConfig {
    /// Emit `accuracy_results.csv`.
    pub enabled: bool,
}

/// The accuracy CSV [`Exporter`].
pub struct AccuracyCsvExporter;

impl Exporter for AccuracyCsvExporter {
    fn name(&self) -> &'static str {
        "accuracy_csv"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.accuracy_csv.enabled
    }

    fn export(
        &self,
        _report: &NativeReport,
        _artifact_dir: &Path,
        _cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        anyhow::bail!("native accuracy csv sink not yet implemented");
    }
}
