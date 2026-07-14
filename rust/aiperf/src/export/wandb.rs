// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native-Rust Weights & Biases sink.
//!
//! Ports the Python `exporters/wandb_data_exporter.py`. W&B has no official Rust
//! SDK; the recommended route is the **offline `.wandb` transaction-log file**
//! (a length-prefixed protobuf `Record` stream — Run/History/Summary/Config/
//! Files/Artifact records — written under `wandb/offline-run-*/`, later shipped
//! by `wandb sync`). This is network-free (no live socket to hang shutdown,
//! spec §6) and has a deterministic oracle (`wandb sync --dryrun`). The logged
//! content matches the Python exporter: the summary-table rows (STAT_COLUMN_KEYS,
//! display-order, finite-guarded), the full redacted config blob, the tag set
//! (`aiperf-<version>`, `benchmark-<id8>`, user tags), and the artifact bundle.
//!
//! STATUS: registered-but-inert stub (Worker H fills the body).

use std::path::Path;

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::NativeReport;

/// W&B export policy. Enabled iff a project is set (matching Python
/// `WandbConfig.enabled = project is not None`).
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WandbExportConfig {
    /// W&B project (enables the sink when present).
    pub project: Option<String>,
    /// Optional entity/team.
    pub entity: Option<String>,
    /// Optional run name.
    pub run_name: Option<String>,
    /// User tags.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// The W&B [`Exporter`] (offline `.wandb` transaction log).
pub struct WandbExporter;

impl Exporter for WandbExporter {
    fn name(&self) -> &'static str {
        "wandb"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.wandb.project.is_some()
    }

    fn export(
        &self,
        _report: &NativeReport,
        _artifact_dir: &Path,
        _cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        anyhow::bail!("native W&B (.wandb offline) sink not yet implemented");
    }
}
