// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native-Rust timeslice sink: `profile_export_aiperf_timeslices.{json,csv}`.
//!
//! Ports the Python `exporters/timeslice_metrics_{json,csv}_exporter.py`. The
//! native-v2 report already embeds per-series timeslices
//! (`MetricSeries.timeslices`, each `{start_ns,end_ns,complete,stats}`); this
//! sink regroups them into the legacy per-slice metric map and serializes to the
//! two files byte-for-byte. Only emitted when `slice_duration` was set. Parity
//! oracle: the current Python timeslice JSON/CSV output.
//!
//! STATUS: registered-but-inert stub (Worker E fills the body).

use std::path::Path;

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::NativeReport;

/// Timeslice export policy. Enabled when the run produced timeslices.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TimesliceExportConfig {
    /// Emit `profile_export_aiperf_timeslices.json`.
    pub json: bool,
    /// Emit `profile_export_aiperf_timeslices.csv`.
    pub csv: bool,
    /// Filename stem (before the `_timeslices` suffix); default `profile_export_aiperf`.
    pub stem: Option<String>,
}

/// The timeslice [`Exporter`] (JSON + CSV).
pub struct TimesliceExporter;

impl Exporter for TimesliceExporter {
    fn name(&self) -> &'static str {
        "timeslice"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.timeslice.json || cfg.timeslice.csv
    }

    fn export(
        &self,
        _report: &NativeReport,
        _artifact_dir: &Path,
        _cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        anyhow::bail!("native timeslice json/csv sink not yet implemented");
    }
}
