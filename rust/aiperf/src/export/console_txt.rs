// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native-Rust fixed-width console artifact + warning/insight sink:
//! `profile_export_console.txt`.
//!
//! Ports the width-pinned console record from Python
//! `exporters/exporter_manager.py::_write_console_txt` plus the domain-logic
//! renderers it captures: the grouped metrics tables
//! (`console_metrics_exporter.py`), the error-summary table
//! (`console_error_exporter.py`), and the "earned-in-blood" warning/insight
//! detectors (spec §3): OSL-mismatch, usage-discrepancy, and the API-error
//! insights (MaxCompletionTokens, DynamoSessionControl — exact trigger + fix
//! text/version lore). Render to a fixed `CONSOLE_EXPORT_WIDTH` (140) buffer,
//! decoupled from terminal width; the LIVE terminal Rich rendering stays in the
//! Python parent (the runner subprocess reserves stdout for one JSON line).
//!
//! All trigger data is in the native-v2 report (metric `avg`s + `report.errors`).
//! Factor one `warning_panel(title, insight)` helper + `fn detect(&NativeReport)
//! -> Option<Warning>` functions (spec §3), not a class-per-detector. Parity
//! oracle: a fixed report → the exact 140-col `.txt` golden + per-detector
//! byte-exact message-string fixtures.
//!
//! STATUS: registered-but-inert stub (Worker G fills the body).

use std::path::Path;

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::NativeReport;

/// Console-artifact export policy. Enabled by default (the `.txt` artifact is a
/// stable CI-log surface); the fixed render width is carried here.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConsoleTxtExportConfig {
    /// Emit `profile_export_console.txt`.
    pub enabled: bool,
    /// Fixed render width (Python `CONSOLE_EXPORT_WIDTH`, default 140).
    pub width: u16,
    /// Include INTERNAL/EXPERIMENTAL metrics (dev mode).
    pub dev: bool,
}

impl Default for ConsoleTxtExportConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            width: 140,
            dev: false,
        }
    }
}

/// The fixed-width console-artifact [`Exporter`].
pub struct ConsoleTxtExporter;

impl Exporter for ConsoleTxtExporter {
    fn name(&self) -> &'static str {
        "console_txt"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.console_txt.enabled
    }

    fn export(
        &self,
        _report: &NativeReport,
        _artifact_dir: &Path,
        _cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        anyhow::bail!("native console-txt / warning-insight sink not yet implemented");
    }
}
