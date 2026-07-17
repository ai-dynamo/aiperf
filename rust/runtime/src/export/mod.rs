// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native-Rust post-report exporter plane.
//!
//! The runner commits the authoritative native-v2 report (`aiperf_runtime::report`) and
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
//! Adding a destination is a new module + one registration in
//! [`ExporterRegistry::with_builtin_exporters`]; nothing in the runner call site
//! changes.
//!
//! # Failure discipline
//! The native-v2 report is the authority and is already committed before any
//! exporter runs. Exporter failures are therefore **best-effort**: each is logged
//! and does not abort the run (mirroring the Python `try/except` per exporter and
//! the §6 requirement that an unreachable tracking server never hangs shutdown).
//! A sink that must be authoritative should surface its own hard error inside
//! `export` and the operator reads the warning; the run's success is pinned to
//! the committed report, not to compat/telemetry side outputs.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use crate::extensions::DuplicateName;
use crate::metrics_core::NativeReport;
use crate::metrics_core::report::{MetricSeries, ReportValue};

/// Classification of a metric's series for summary selection, mirroring Python
/// `native_report._summary_series`.
///
/// The exporters disagree only on how they *react* to the degenerate cases: the
/// genai-perf and console tables skip the metric (best-effort), while the
/// timeslice exporter treats an empty or ambiguous metric as a hard error. This
/// classifier owns the shared selection rule; each caller maps the outcome to
/// its own policy (and error text).
pub(crate) enum SummarySeries<'a> {
    /// The metric carried no series at all.
    Empty,
    /// The selected summary series: the sole series, or the unique unlabeled
    /// aggregate among several labeled series.
    Selected(&'a MetricSeries),
    /// Several series, none unlabeled — there is no aggregate to summarize.
    NoAggregate,
    /// Several series with more than one unlabeled aggregate — ambiguous.
    Ambiguous,
}

/// Build a CSV writer with CRLF line terminators, matching the Python exporters'
/// `\r\n`-terminated output. Every native CSV emitter (genai-perf, timeslice,
/// server-metrics, accuracy) shares this one builder over its own sink.
pub(crate) fn crlf_csv_writer<W: std::io::Write>(writer: W) -> csv::Writer<W> {
    csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(writer)
}

/// Ports `aiperf/exporters/utils.py::normalize_endpoint_display`: drop the URL
/// scheme, keep the `netloc` plus a `/metrics`-trimmed path, discarding any query
/// or fragment (as `urlparse`'s netloc/path split does).
///
/// Shared by the genai-perf telemetry summary and the server-metrics exporter so
/// both render the same endpoint keys. Netloc (host, port, any userinfo) is
/// preserved verbatim, so `http://127.0.0.1:9400/dcgm1/metrics` becomes
/// `127.0.0.1:9400/dcgm1`.
pub(crate) fn normalize_endpoint_display(url: &str) -> String {
    let after_scheme = match url.find("://") {
        Some(index) => &url[index + 3..],
        None => url,
    };
    let netloc_end = after_scheme
        .find(['/', '?', '#'])
        .unwrap_or(after_scheme.len());
    let netloc = &after_scheme[..netloc_end];
    let rest = &after_scheme[netloc_end..];
    let path_end = rest.find(['?', '#']).unwrap_or(rest.len());
    let path = &rest[..path_end];
    let path = if path.starts_with('/') { path } else { "" };
    let path = path.strip_suffix("/metrics").unwrap_or(path);
    let mut display = netloc.to_string();
    if !path.is_empty() {
        display.push_str(path);
    }
    display
}

/// Select a metric's summary series (`native_report._summary_series`): the sole
/// series, or the single unlabeled aggregate when several labeled series exist.
pub(crate) fn summary_series(series: &[MetricSeries]) -> SummarySeries<'_> {
    match series {
        [] => SummarySeries::Empty,
        [single] => SummarySeries::Selected(single),
        many => {
            let mut unlabeled = many.iter().filter(|series| series.labels.is_none());
            let first = unlabeled.next();
            if unlabeled.next().is_some() {
                return SummarySeries::Ambiguous;
            }
            match first {
                Some(series) => SummarySeries::Selected(series),
                None => SummarySeries::NoAggregate,
            }
        }
    }
}

/// Lower a [`ReportValue`] to its inner `f64` by passthrough: `Finite` always
/// yields its payload, `NonFinite` yields `None`. The `Finite` payload is
/// trusted as-is (no `is_finite` re-check) — the projection that constructed the
/// value already decided finiteness. Sinks that additionally reject a non-finite
/// `Finite` payload use [`finite_guarded`] instead.
pub(crate) fn finite_passthrough(value: ReportValue) -> Option<f64> {
    match value {
        ReportValue::Finite(value) => Some(value),
        ReportValue::NonFinite => None,
    }
}

/// Lower a [`ReportValue`] to its inner `f64`, additionally dropping a `Finite`
/// payload that is not `is_finite` (a defensively-guarded NaN/inf). Sinks that
/// trust the constructor's finiteness use [`finite_passthrough`] instead.
pub(crate) fn finite_guarded(value: ReportValue) -> Option<f64> {
    match value {
        ReportValue::Finite(value) if value.is_finite() => Some(value),
        _ => None,
    }
}

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
/// Shared Arrow column builders and Parquet writer properties for the two
/// Parquet sinks. Gated behind the `parquet` feature alongside its consumers.
#[cfg(feature = "parquet")]
pub(crate) mod parquet_util;
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
/// Object-safe (`&dyn Exporter`) so the [`ExporterRegistry`] is a heterogeneous
/// static list. The call site is synchronous with no ambient tokio runtime (the
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

/// Emit-order band for the local-file writers. Every sink in this band runs
/// before any sink in [`ORDER_UPLOADER`] so the network uploaders below observe
/// the on-disk files (§6 of the exporters design). The per-sink offsets keep the
/// intra-band order stable and leave gaps for a future extension to slot into.
const ORDER_FILE_WRITER: u32 = 0;
/// Emit-order band for the network / deferred uploaders, run after every
/// local-file writer has produced its on-disk artifact.
const ORDER_UPLOADER: u32 = 1_000;

/// One exporter plus its explicit emit-order key.
#[derive(Clone)]
struct OrderedExporter {
    /// Ascending sort key; ties break on [`Exporter::name`] for determinism.
    order: u32,
    /// Shared thread-safe (`Arc<… + Send + Sync>`) so the registry stays [`Clone`]
    /// (which the transactional staging of the enclosing [`AIPerfRegistry`]
    /// requires) and keeps that aggregate `Send + Sync` alongside its transport /
    /// workload factories; the exporters themselves are stateless.
    exporter: Arc<dyn Exporter + Send + Sync>,
}

/// Name-keyed, explicitly ordered registry of report exporters.
///
/// The emit order is LOAD-BEARING: local-file writers must run before network
/// uploaders so an uploaded artifact bundle observes the on-disk files. Each
/// entry therefore carries an explicit `order` key that decouples registration
/// order from emit order — a later extension can slot a sink into the correct
/// band without depending on insertion position. Names are unique (keyed by
/// [`Exporter::name`]); a duplicate registration is rejected.
///
/// [`Clone`] (via shared `Arc` exporters) so it can be a field of the
/// transactionally staged [`AIPerfRegistry`](crate::extensions::AIPerfRegistry),
/// which the `BuiltinExportersExtension` populates.
#[derive(Clone)]
pub struct ExporterRegistry {
    entries: BTreeMap<String, OrderedExporter>,
}

impl ExporterRegistry {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Register one exporter under `order`, rejecting a name already present.
    ///
    /// Lower `order` values emit first; ties break on [`Exporter::name`].
    pub fn register(
        &mut self,
        order: u32,
        exporter: Arc<dyn Exporter + Send + Sync>,
    ) -> Result<(), DuplicateName> {
        let name = exporter.name().to_string();
        if self.entries.contains_key(&name) {
            return Err(DuplicateName(name));
        }
        self.entries
            .insert(name, OrderedExporter { order, exporter });
        Ok(())
    }

    /// Populate the complete native in-tree exporter set into this registry in
    /// canonical emit order: local-file writers first, then network uploaders.
    ///
    /// This is the single source of truth for the built-in exporter set; both the
    /// [`Self::with_builtin_exporters`] convenience constructor and the
    /// `BuiltinExportersExtension` that folds these sinks into the unified
    /// [`AIPerfRegistry`](crate::extensions::AIPerfRegistry) delegate here.
    pub fn register_builtins(&mut self) -> Result<(), DuplicateName> {
        // Local-file writers (so the uploaders below see the on-disk files). The
        // server-metrics Parquet sink is gated behind the `parquet` feature: a
        // lite/online-only build drops `arrow`/`parquet` and leaves the slot empty
        // (its `cfg.export.parquet` block still decodes but is inert).
        let mut builtins: Vec<(u32, Arc<dyn Exporter + Send + Sync>)> = vec![
            (ORDER_FILE_WRITER, Arc::new(genai_perf::GenaiPerfV1Exporter)),
            (
                ORDER_FILE_WRITER + 1,
                Arc::new(server_metrics::ServerMetricsExporter),
            ),
            (
                ORDER_FILE_WRITER + 2,
                Arc::new(timeslice::TimesliceExporter),
            ),
            (
                ORDER_FILE_WRITER + 3,
                Arc::new(accuracy_csv::AccuracyCsvExporter),
            ),
        ];
        #[cfg(feature = "parquet")]
        builtins.push((ORDER_FILE_WRITER + 4, Arc::new(parquet::ParquetExporter)));
        builtins.extend([
            (
                ORDER_FILE_WRITER + 5,
                Arc::new(console_txt::ConsoleTxtExporter) as Arc<dyn Exporter + Send + Sync>,
            ),
            // Network / deferred uploaders.
            (ORDER_UPLOADER, Arc::new(otel::OtelExporter)),
            (ORDER_UPLOADER + 1, Arc::new(mlflow::MlflowExporter)),
            (ORDER_UPLOADER + 2, Arc::new(wandb::WandbExporter)),
        ]);
        for (order, exporter) in builtins {
            self.register(order, exporter)?;
        }
        Ok(())
    }

    /// Construct the complete native in-tree exporter set in canonical emit
    /// order: local-file writers first, then network uploaders.
    pub fn with_builtin_exporters() -> Self {
        let mut registry = Self::new();
        registry
            .register_builtins()
            .expect("built-in exporter names are unique");
        registry
    }

    /// Exporters in emit order: ascending `order`, [`Exporter::name`] as the
    /// deterministic tie-break.
    pub fn iter(&self) -> impl Iterator<Item = &dyn Exporter> {
        let mut ordered: Vec<&OrderedExporter> = self.entries.values().collect();
        ordered.sort_by(|a, b| {
            a.order
                .cmp(&b.order)
                .then_with(|| a.exporter.name().cmp(b.exporter.name()))
        });
        ordered
            .into_iter()
            .map(|entry| entry.exporter.as_ref() as &dyn Exporter)
    }

    /// Run every enabled exporter over the finalized report in emit order.
    /// Best-effort: an exporter error is logged and does not abort the run (the
    /// native-v2 report is the committed authority). Returns the number of
    /// exporters that ran without error, for provenance/telemetry.
    pub fn run(&self, report: &NativeReport, artifact_dir: &Path, cfg: &ExportConfig) -> usize {
        let mut succeeded = 0usize;
        for exporter in self.iter() {
            if !exporter.enabled(cfg) {
                continue;
            }
            match exporter.export(report, artifact_dir, cfg) {
                Ok(()) => {
                    succeeded += 1;
                    // INFO so the export step is visible on a normal run, mirroring
                    // Python's per-exporter "Exported … to: …" lines (the native
                    // exporters own their own paths, so we name the exporter).
                    tracing::info!(
                        exporter = exporter.name(),
                        "Exported {} data",
                        exporter.name()
                    );
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
}

impl Default for ExporterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Run every enabled built-in exporter over the finalized report. Thin wrapper
/// over [`ExporterRegistry::with_builtin_exporters`] + [`ExporterRegistry::run`]
/// for the runner call sites that emit the stock exporter set.
pub fn run_exporters(report: &NativeReport, artifact_dir: &Path, cfg: &ExportConfig) -> usize {
    ExporterRegistry::with_builtin_exporters().run(report, artifact_dir, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_export_block_decodes_to_all_disabled() {
        let cfg = ExportConfig::default();
        for exporter in ExporterRegistry::with_builtin_exporters().iter() {
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

    /// The emit order is byte-for-byte load-bearing: every local-file writer must
    /// run before every network uploader so an uploaded bundle observes the
    /// on-disk files. Guard both the exact canonical sequence and the band split.
    #[test]
    fn builtin_exporter_emit_order_is_preserved() {
        let names: Vec<&str> = ExporterRegistry::with_builtin_exporters()
            .iter()
            .map(Exporter::name)
            .collect();

        let file_writers = [
            "genai_perf_v1",
            "server_metrics",
            "timeslice",
            "accuracy_csv",
            "server_metrics_parquet",
            "console_txt",
        ];
        let uploaders = ["otel", "mlflow", "wandb"];
        let expected: Vec<&str> = file_writers
            .iter()
            .chain(uploaders.iter())
            .copied()
            .collect();
        assert_eq!(names, expected, "canonical exporter emit order drifted");

        // Regression guard for the band split independent of the exact sequence:
        // the last file writer must precede the first uploader.
        let last_writer = file_writers
            .iter()
            .map(|name| names.iter().position(|n| n == name).unwrap())
            .max()
            .unwrap();
        let first_uploader = uploaders
            .iter()
            .map(|name| names.iter().position(|n| n == name).unwrap())
            .min()
            .unwrap();
        assert!(
            last_writer < first_uploader,
            "a network uploader was ordered before a local-file writer"
        );
    }

    #[test]
    fn duplicate_exporter_name_is_rejected() {
        let mut registry = ExporterRegistry::new();
        registry
            .register(0, Arc::new(genai_perf::GenaiPerfV1Exporter))
            .unwrap();
        let error = registry
            .register(5, Arc::new(genai_perf::GenaiPerfV1Exporter))
            .unwrap_err();
        assert_eq!(error, DuplicateName("genai_perf_v1".to_owned()));
    }
}
