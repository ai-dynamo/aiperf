// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! genai-perf v1 byte-exact compatibility sink (`--export-genai-perf`).
//!
//! Translates the native-v2 [`NativeReport`] into the exact legacy artifacts the
//! NVIDIA genai-perf tool emits — `<stem>_genai_perf.json` and
//! `<stem>_genai_perf.csv` — for downstream tooling frozen on that contract. The
//! parity oracle is the genai-perf tool itself: an `aiperf profile` run and a
//! `genai-perf profile` run against the same target must produce byte-identical
//! compat files (modulo run-specific values), guarded by golden fixtures and an
//! e2e diff.
//!
//! Grounding (byte-exact source of truth):
//! - JSON: `genai_perf/export_data/json_exporter.py` — `json.dumps(indent=2)`,
//!   flat metric dict + `input_config`, scalar metrics carry only `{unit, avg}`.
//! - CSV: `genai_perf/export_data/csv_exporter.py` — `REQUEST_METRICS_HEADER`
//!   (`Metric,avg,min,max,p99,p95,p90,p75,p50,p25,p10,p5,p1`), blank row,
//!   `SYSTEM_METRICS_HEADER` (`Metric,Value`), then telemetry.
//! - Stats: `genai_perf/metrics/statistics.py` — percentile list `[1,5,10,25,50,
//!   75,90,95,99]`, ns→ms scale for time metrics, system metrics get only avg.
//!
//! STATUS: config + gating are wired; the translation body is unimplemented
//! (Worker A). Until implemented, an enabled run logs and writes nothing.

use std::path::Path;

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::NativeReport;

/// genai-perf v1 compat export policy. Disabled unless the frontend passes
/// `--export-genai-perf`; `stem` is the profile-export filename stem
/// (`profile_export` → `profile_export_genai_perf.{json,csv}`).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GenaiPerfExportConfig {
    /// Emit the genai-perf v1 compat JSON/CSV artifacts.
    pub enabled: bool,
    /// Filename stem for the compat artifacts (before the `_genai_perf` suffix).
    pub stem: String,
    /// Whether goodput was requested (gates the `request_goodput` system-metric
    /// row, matching genai-perf's `--goodput` behavior).
    pub goodput: bool,
    /// Whether the run streamed (gates streaming-only request metrics in CSV).
    pub streaming: bool,
    /// Endpoint type string (e.g. `chat`, `embeddings`) — drives genai-perf's
    /// per-endpoint metric-skip rules.
    pub endpoint_type: String,
}

impl Default for GenaiPerfExportConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            stem: "profile_export".to_owned(),
            goodput: false,
            streaming: false,
            endpoint_type: String::new(),
        }
    }
}

/// The genai-perf v1 compat [`Exporter`].
pub struct GenaiPerfV1Exporter;

impl Exporter for GenaiPerfV1Exporter {
    fn name(&self) -> &'static str {
        "genai_perf_v1"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.genai_perf.enabled
    }

    fn export(
        &self,
        _report: &NativeReport,
        _artifact_dir: &Path,
        _cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        // Worker A: emit `<stem>_genai_perf.json` and `<stem>_genai_perf.csv`
        // byte-identical to the genai-perf tool. Left unimplemented at the
        // foundation so the plane compiles with the sink registered-but-inert.
        anyhow::bail!("genai-perf v1 compat sink not yet implemented");
    }
}
