// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native-Rust server-metrics Parquet sink: `server_metrics_export.parquet`.
//!
//! Ports the Python `server_metrics/parquet_exporter.py` (currently driven by
//! `native_report.py::_render_server_metrics_parquet`, which reads a Rust-emitted
//! `.aiperf-server-metrics-parquet-wire.jsonl` wire file and renders Parquet in
//! Python). Doing this natively in the runner lets us DELETE that wire round-trip:
//! the runner already owns the raw `ServerMetricsRecord` rows (see
//! `rust/runner/src/server_metrics.rs`) and can write Parquet directly via the
//! `parquet`/`arrow` crates. Parity oracle: the current Python Parquet file
//! (`aiperf.schema_version = 1.0` key-value metadata + column schema).
//!
//! NOTE: this is the one server-metrics artifact needing data NOT in the
//! aggregated native-v2 report (raw per-record rows), so this sink likely reads
//! the raw rows the runner holds rather than `NativeReport` alone — Worker I
//! decides the exact data path (report vs a runner-provided raw-row handle).
//!
//! STATUS: registered-but-inert stub (Worker I fills the body).

use std::path::Path;

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::NativeReport;

/// Server-metrics Parquet export policy.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ParquetExportConfig {
    /// Emit `server_metrics_export.parquet`.
    pub enabled: bool,
}

/// The server-metrics Parquet [`Exporter`].
pub struct ParquetExporter;

impl Exporter for ParquetExporter {
    fn name(&self) -> &'static str {
        "server_metrics_parquet"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.parquet.enabled
    }

    fn export(
        &self,
        _report: &NativeReport,
        _artifact_dir: &Path,
        _cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        anyhow::bail!("native server-metrics parquet sink not yet implemented");
    }
}
