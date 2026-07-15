// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native-Rust post-report exporter plane.
//!
//! The runner commits the authoritative native-v2 report (`aiperf::report`) and
//! then hands the finalized [`NativeReport`] to this plane, which fans it out to
//! a static set of [`Exporter`] impls behind one trait. This replaces the legacy
//! Python exporter machinery (plugins.yaml registry, exception-as-disable
//! constructors, `asyncio` fan-out, subprocess uploaders) with a single-process
//! Rust sink list — see `specs/2026-07-11-aiperf-rust-exporters-overhaul-design.md`
//! §5 for the design and §4 for the accidental complexity deleted.
//!
//! # The seam (extend here)
//! Every output format / destination is an [`Exporter`]: a byte-exact
//! compatibility sink (genai-perf v1 JSON/CSV), a telemetry emitter (OTLP/HTTP
//! metrics), or a run tracker (MLflow, W&B). Each declares [`Exporter::enabled`]
//! against the typed [`ExportConfig`] and writes/uploads in [`Exporter::export`].
//! Adding a destination is a new module + one entry in [`registry`]; nothing in
//! the runner call site changes.
//!
//! # Failure discipline
//! The native-v2 report is the authority and is already committed before any
//! exporter runs. Exporter failures are therefore **best-effort**: each is logged
//! and does not abort the run (mirroring the Python `try/except` per exporter and
//! the §6 requirement that an unreachable tracking server never hangs shutdown).
//! A sink that must be authoritative should surface its own hard error inside
//! `export` and the operator reads the warning; the run's success is pinned to
//! the committed report, not to compat/telemetry side outputs.

use std::path::Path;

use crate::metrics_core::NativeReport;

pub mod accuracy_csv;
pub mod console_txt;
pub mod genai_perf;
pub mod mlflow;
pub mod otel;
/// Server-metrics Parquet sink. Gated behind the `parquet` feature: it links
/// `arrow` + `parquet` (~2.6 MiB of `.text`), which a lite/online-only build
/// (e.g. a lightweight nightly wheel) can drop. When the feature is off the
/// [`ParquetExporter`] is not registered and the public-dataset loader rejects
/// `.parquet` inputs; [`ParquetExportConfig`] stays present so the wire
/// `cfg.export.parquet` block still decodes (it is simply inert).
#[cfg(feature = "parquet")]
pub mod parquet;
/// Wide, per-request Parquet sidecar to `profile_export.jsonl`. Gated behind the
/// `parquet` feature (links `arrow` + `parquet`): a lite build drops it and the
/// runner skips the artifact with a warning. Unlike the sinks in [`registry`],
/// this is not an [`Exporter`] over the aggregated [`NativeReport`] — the
/// per-record data lives only at the runner's `CapturedRecord` callsites, so the
/// runner drives this writer directly.
#[cfg(feature = "parquet")]
pub mod per_record_parquet;
pub mod server_metrics;
pub mod timeslice;
pub mod wandb;

pub use accuracy_csv::AccuracyCsvExportConfig;
pub use console_txt::ConsoleTxtExportConfig;
pub use genai_perf::GenaiPerfExportConfig;
pub use mlflow::MlflowExportConfig;
pub use otel::OtelExportConfig;
pub use server_metrics::ServerMetricsExportConfig;
pub use timeslice::TimesliceExportConfig;
pub use wandb::WandbExportConfig;

/// Server-metrics Parquet export policy.
///
/// Defined here (not in the feature-gated [`parquet`] module) so the wire
/// `cfg.export.parquet` block always decodes: a lite build with the `parquet`
/// feature off still accepts the config, it just registers no Parquet exporter.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ParquetExportConfig {
    /// Emit `server_metrics_export.parquet`.
    pub enabled: bool,
}

/// Typed export policy projected by the Python frontend onto the wire `cfg.export`
/// block and decoded once by the runner. Each sub-config is independently gated;
/// an absent block decodes to all-disabled defaults so the base path emits only
/// the native-v2 report.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExportConfig {
    /// aiperf v1 summary sink (`profile_export_aiperf.{json,csv}`).
    pub genai_perf: GenaiPerfExportConfig,
    /// OpenTelemetry OTLP/HTTP metrics emitter.
    pub otel: OtelExportConfig,
    /// MLflow run tracker (REST uploader).
    pub mlflow: MlflowExportConfig,
    /// Server-metrics summary sink (`server_metrics_export.{json,csv}`).
    pub server_metrics: ServerMetricsExportConfig,
    /// Timeslice sink (`profile_export_aiperf_timeslices.{json,csv}`).
    pub timeslice: TimesliceExportConfig,
    /// Accuracy sink (`accuracy_results.csv`).
    pub accuracy_csv: AccuracyCsvExportConfig,
    /// Fixed-width console artifact + warning/insight sink
    /// (`profile_export_console.txt`).
    pub console_txt: ConsoleTxtExportConfig,
    /// Weights & Biases sink (offline `.wandb`).
    pub wandb: WandbExportConfig,
    /// Server-metrics Parquet sink (`server_metrics_export.parquet`).
    pub parquet: ParquetExportConfig,
}

/// One output format or destination for the finalized native-v2 report.
///
/// Object-safe (`&dyn Exporter`) so the [`registry`] is a heterogeneous static
/// list. The call site is synchronous with no ambient tokio runtime (the
/// execution runtime is already torn down by the time the report is committed),
/// so a sink needing async network I/O drives its own short-lived
/// `current_thread` runtime internally.
pub trait Exporter {
    /// Stable identifier for logs and error context.
    fn name(&self) -> &'static str;

    /// Whether this sink runs for the given export policy. Replaces the Python
    /// exception-as-disable constructor with an explicit predicate.
    fn enabled(&self, cfg: &ExportConfig) -> bool;

    /// Emit the report to this sink's format/destination. `artifact_dir` is the
    /// run's exclusive, already-created artifact target; file sinks join their
    /// output name onto it via the runner's path-traversal-safe helper.
    fn export(
        &self,
        report: &NativeReport,
        artifact_dir: &Path,
        cfg: &ExportConfig,
    ) -> anyhow::Result<()>;
}

/// The frozen static exporter list, assembled once per run. Order is
/// local-file writers first, then network uploaders (so uploaded artifact
/// bundles observe the on-disk files). Registering a sink is one line here.
fn registry() -> Vec<Box<dyn Exporter>> {
    vec![
        // Local-file writers first (so uploaders below see the on-disk files).
        Box::new(genai_perf::GenaiPerfV1Exporter),
        Box::new(server_metrics::ServerMetricsExporter),
        Box::new(timeslice::TimesliceExporter),
        Box::new(accuracy_csv::AccuracyCsvExporter),
        #[cfg(feature = "parquet")]
        Box::new(parquet::ParquetExporter),
        Box::new(console_txt::ConsoleTxtExporter),
        // Network / deferred uploaders.
        Box::new(otel::OtelExporter),
        Box::new(mlflow::MlflowExporter),
        Box::new(wandb::WandbExporter),
    ]
}

/// Run every enabled exporter over the finalized report. Best-effort: an
/// exporter error is logged and does not abort the run (the native-v2 report is
/// the committed authority). Returns the number of exporters that ran without
/// error, for the caller's provenance/telemetry.
pub fn run_exporters(report: &NativeReport, artifact_dir: &Path, cfg: &ExportConfig) -> usize {
    let mut succeeded = 0usize;
    for exporter in registry() {
        if !exporter.enabled(cfg) {
            continue;
        }
        match exporter.export(report, artifact_dir, cfg) {
            Ok(()) => {
                succeeded += 1;
                tracing::debug!(exporter = exporter.name(), "exporter completed");
            }
            Err(error) => {
                tracing::warn!(
                    exporter = exporter.name(),
                    "exporter failed (native-v2 report is unaffected): {error:#}"
                );
            }
        }
    }
    succeeded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_export_block_decodes_to_all_disabled() {
        let cfg = ExportConfig::default();
        for exporter in registry() {
            assert!(
                !exporter.enabled(&cfg),
                "{} should be disabled by default",
                exporter.name()
            );
        }
    }

    #[test]
    fn disabled_config_runs_no_exporters() {
        let summary = crate::metrics_core::AccumulatorSummary::new();
        let report = NativeReport::new(&summary, None);
        let dir = std::env::temp_dir();
        assert_eq!(run_exporters(&report, &dir, &ExportConfig::default()), 0);
    }
}
