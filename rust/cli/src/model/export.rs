// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed `export` section — the post-report sink policy the runner consumes.
//!
//! Mirrors the runner's `aiperf_runtime::export::ExportConfig` (and its `console_txt` /
//! `genai_perf` sub-configs) exactly; the runner decodes this block with
//! `deny_unknown_fields`, so only fields it owns are emitted.
//!
//! Wire shape ported from `src/aiperf/orchestrator/rust_wire.py::_export`,
//! `_console_txt_frontend_projection`, `_genai_perf_frontend_projection`.
//!
//! Two provenance classes:
//! * **Static, byte-exact** — the per-metric console metadata and the
//!   header_map/filtered_tags/scalar_tags come from Python's `MetricRegistry`
//!   (intentionally divergent from the Rust `metrics_core` catalog, per that
//!   projection's own docstring). They are identical across runs, so they are
//!   captured once into `resources/metric_metadata.json` and embedded here
//!   byte-exact rather than derived from the divergent Rust catalog.
//! * **Best-effort** — the genai-perf `envelope.input_config` is Python's
//!   `JsonExportData.model_dump(exclude_unset=True)`, whose set-tracking a flat
//!   native object cannot reproduce. It is projected functionally (opaque
//!   `Value`, echoed verbatim by the runner) so the aiperf-v1 exports are
//!   *emitted*; the metric VALUES come from the authoritative native-v2 report.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

/// aiperf-v1 exports echo the Python package version, not the Rust crate version.
pub const AIPERF_V1_VERSION: &str = "0.11.0";
/// Fixed console render width (`Environment.UI.CONSOLE_EXPORT_WIDTH`).
const CONSOLE_EXPORT_WIDTH: u16 = 140;

/// One console-metric's display metadata (mirrors `console_txt::ConsoleMetricMeta`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsoleMetricMeta {
    /// Display header.
    pub header: String,
    /// Console group id.
    pub group: String,
    /// Explicit display order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_order: Option<u32>,
    /// Internal (dev-only) metric.
    #[serde(default, skip_serializing_if = "is_false")]
    pub internal: bool,
    /// Experimental (dev-only) metric.
    #[serde(default, skip_serializing_if = "is_false")]
    pub experimental: bool,
    /// Error-only metric.
    #[serde(default, skip_serializing_if = "is_false")]
    pub error_only: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Static, registry-derived metric metadata captured from Python once.
#[derive(Debug, Deserialize)]
struct StaticMeta {
    console_metrics: BTreeMap<String, ConsoleMetricMeta>,
    header_map: BTreeMap<String, String>,
    filtered_tags: Vec<String>,
    scalar_tags: Vec<String>,
}

static META: LazyLock<StaticMeta> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../../resources/metric_metadata.json"))
        .expect("embedded metric_metadata.json is valid")
});

/// The fixed-width console artifact policy (mirrors `ConsoleTxtExportConfig`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsoleTxt {
    /// The `.txt` artifact is always written.
    pub enabled: bool,
    /// Fixed render width.
    pub width: u16,
    /// Dev (internal/experimental) visibility.
    pub dev: bool,
    /// Base metrics title.
    pub title: String,
    /// Per-registered-tag console metadata.
    pub metrics: BTreeMap<String, ConsoleMetricMeta>,
}

/// The genai-perf-v1 envelope (opaque values echoed by the runner sink).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenaiPerfEnvelope {
    /// aiperf-v1 package version.
    pub aiperf_version: String,
    /// Run identifier.
    pub benchmark_id: serde_json::Value,
    /// Input-config echo (best-effort; see module docs).
    pub input_config: serde_json::Value,
    /// Run info.
    pub run_info: serde_json::Value,
}

/// The aiperf-v1 summary sink policy (mirrors `GenaiPerfExportConfig`; `stem`
/// omitted so the runner keeps its default).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenaiPerf {
    /// Whether `profile_export_aiperf.{json,csv}` are written.
    pub enabled: bool,
    /// Per-tag display headers.
    pub header_map: BTreeMap<String, String>,
    /// Registered tags the file exporters drop (sorted).
    pub filtered_tags: Vec<String>,
    /// Registered scalar-tier tags (sorted).
    pub scalar_tags: Vec<String>,
    /// Frontend-owned envelope values.
    pub envelope: GenaiPerfEnvelope,
}

/// The OTLP/HTTP metrics sink (mirrors `OtelExportConfig`, projected fields).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OtelExport {
    /// Whether the OTLP sink runs.
    pub enabled: bool,
    /// OTLP metrics endpoint (`<url>/v1/metrics`).
    pub endpoint: String,
    /// GenAI provider label (`--gen-ai-provider`), present when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Resource attributes attached to every metric.
    pub resource_attributes: std::collections::BTreeMap<String, String>,
}

impl OtelExport {
    /// Build the OTLP sink from a collector URL and run identity (mirrors
    /// `_otel_frontend_projection`): append `/v1/metrics` when absent, attach the
    /// canonical aiperf resource attributes, merge any custom
    /// `--otel-resource-attributes`, and record the `--gen-ai-provider`.
    pub fn build(
        url: &str,
        benchmark_id: &str,
        endpoint_type: &str,
        model: &str,
        provider: Option<&str>,
        extra_attrs: &[(String, String)],
    ) -> Self {
        let endpoint = if url.ends_with("/v1/metrics") {
            url.to_string()
        } else {
            format!("{}/v1/metrics", url.trim_end_matches('/'))
        };
        let mut attrs = std::collections::BTreeMap::new();
        attrs.insert("aiperf.benchmark.id".to_string(), benchmark_id.to_string());
        attrs.insert(
            "aiperf.endpoint.type".to_string(),
            endpoint_type.to_string(),
        );
        attrs.insert("aiperf.model.name".to_string(), model.to_string());
        attrs.insert(
            "service.instance.id".to_string(),
            "records-manager".to_string(),
        );
        for (k, v) in extra_attrs {
            attrs.insert(k.clone(), v.clone());
        }
        Self {
            enabled: true,
            endpoint,
            provider: provider.map(str::to_string),
            resource_attributes: attrs,
        }
    }
}

/// Fixed MLflow artifact glob list (`_mlflow_frontend_projection`).
fn mlflow_artifact_globs() -> Vec<String> {
    [
        "*.json",
        "*.csv",
        "*.jsonl",
        "*.parquet",
        "*_timeslices.*",
        "**/*.png",
        "**/*.jpg",
        "**/*.jpeg",
        "**/*.svg",
        "**/*.html",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// The MLflow REST uploader sink (mirrors `MlflowExportConfig`). Envelope-ish
/// fields (`params`) are best-effort; the config fields are byte-exact.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MlflowExport {
    /// Whether the sink runs.
    pub enabled: bool,
    /// aiperf-v1 version.
    pub aiperf_version: String,
    /// Artifact globs to upload.
    pub artifact_globs: Vec<String>,
    /// Run identifier.
    pub benchmark_id: String,
    /// Tracking server URI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tracking_uri: Option<String>,
    /// Experiment name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experiment: Option<String>,
    /// Run name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_name: Option<String>,
    /// Parent MLflow run id (`--mlflow-parent-run-id`), present when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<String>,
    /// Total expected requests (the run's request bound), present when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_expected_requests: Option<f64>,
    /// Run tags.
    pub tags: std::collections::BTreeMap<String, String>,
    /// Logged params (best-effort; Python includes cli_command).
    pub params: std::collections::BTreeMap<String, String>,
}

/// The Weights & Biases sink (mirrors `WandbExportConfig`). `config_json` /
/// `cli_command` are best-effort; the config fields are byte-exact.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WandbExport {
    /// aiperf-v1 version.
    pub aiperf_version: String,
    /// Run identifier.
    pub benchmark_id: String,
    /// Redacted invoking command (best-effort).
    pub cli_command: String,
    /// Serialized config (best-effort).
    pub config_json: String,
    /// W&B entity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity: Option<String>,
    /// W&B project.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// W&B run name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_name: Option<String>,
    /// W&B run tags (`--wandb-tag`), present when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
}

/// The typed `export` policy. Only the sinks the frontend enables are modeled;
/// omitted sinks decode to the runner's all-disabled defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Export {
    /// aiperf-v1 summary sink.
    pub genai_perf: GenaiPerf,
    /// Console artifact sink.
    pub console_txt: ConsoleTxt,
    /// OTLP/HTTP metrics sink (present when `--otel-url` is set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub otel: Option<OtelExport>,
    /// MLflow uploader (present when `--mlflow-tracking-uri` is set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mlflow: Option<MlflowExport>,
    /// W&B sink (present when `--wandb-project` is set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wandb: Option<WandbExport>,
    /// Server-metrics summary sink (`server_metrics_export.{json,csv}`), present
    /// when server-metrics collection is enabled and the JSON and/or CSV format is
    /// selected. Projects `aiperf::export::ServerMetricsExportConfig`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_metrics: Option<ServerMetricsExport>,
    /// Server-metrics Parquet sink (`server_metrics_export.parquet`), present when
    /// server-metrics collection is enabled and the `parquet` format is selected.
    /// Projects `aiperf::export::ParquetExportConfig`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parquet: Option<ParquetExport>,
}

/// The server-metrics summary sink policy (`cfg.export.server_metrics`).
///
/// Projects the runner's `aiperf::export::ServerMetricsExportConfig`: the
/// JSON/CSV toggles plus the three frontend-owned envelope values the native
/// report cannot reconstruct (`aiperf_version`, `benchmark_id`, `input_config`).
/// Ported from
/// `src/aiperf/orchestrator/rust_wire.py::_server_metrics_frontend_projection`
/// (`rust_wire.py:1158`): enabled iff collection is on and JSON and/or CSV is a
/// selected format; `input_config` is echoed only when JSON is enabled. The field
/// set matches the runner struct exactly (which decodes with
/// `deny_unknown_fields`), so unset optional values are skipped on the wire.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerMetricsExport {
    /// Emit `server_metrics_export.json`.
    pub json: bool,
    /// Emit `server_metrics_export.csv`.
    pub csv: bool,
    /// AIPerf package version rendered into the JSON `aiperf_version` field and
    /// the CSV `# aiperf_version:` header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aiperf_version: Option<String>,
    /// Run identity rendered into the JSON `benchmark_id` field and the CSV
    /// `# benchmark_id:` header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmark_id: Option<String>,
    /// User configuration echoed verbatim into the JSON `input_config` object
    /// (JSON only; omitted, and thus left `Null`, when JSON is disabled).
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub input_config: serde_json::Value,
}

impl ServerMetricsExport {
    /// Build the server-metrics summary policy for a run.
    ///
    /// Returns `None` (the block is omitted from `cfg.export`, decoding to the
    /// runner's all-disabled default) when collection is disabled or neither JSON
    /// nor CSV is selected, mirroring
    /// `_server_metrics_frontend_projection`. `input_config` is carried only when
    /// JSON is enabled (the CSV export never reads it).
    pub fn build(
        formats: &[String],
        server_metrics_enabled: bool,
        aiperf_version: &str,
        benchmark_id: &str,
        input_config: serde_json::Value,
    ) -> Option<Self> {
        if !server_metrics_enabled {
            return None;
        }
        let json = formats.iter().any(|format| format == "json");
        let csv = formats.iter().any(|format| format == "csv");
        if !(json || csv) {
            return None;
        }
        Some(Self {
            json,
            csv,
            aiperf_version: Some(aiperf_version.to_string()),
            benchmark_id: Some(benchmark_id.to_string()),
            input_config: if json {
                input_config
            } else {
                serde_json::Value::Null
            },
        })
    }
}

/// The server-metrics Parquet sink toggle (`cfg.export.parquet`).
///
/// Projects the runner's `aiperf::export::ParquetExportConfig`; `enabled` is the
/// sole field (the native Parquet sink reads the runner-emitted wire JSONL and
/// the profiling boundary from the report). Ported from
/// `src/aiperf/orchestrator/rust_wire.py::_parquet_frontend_projection`
/// (`rust_wire.py:1244`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParquetExport {
    /// Emit `server_metrics_export.parquet`.
    pub enabled: bool,
}

impl ParquetExport {
    /// Build the Parquet sink toggle: enabled iff server-metrics collection is on
    /// and the `parquet` format is selected (the same gate under which the sidecar
    /// lowers the `parquet_wire_path` the sink consumes). Returns `None` otherwise
    /// so the block is omitted and the runner sink stays disabled.
    pub fn build(formats: &[String], server_metrics_enabled: bool) -> Option<Self> {
        if server_metrics_enabled && formats.iter().any(|format| format == "parquet") {
            Some(Self { enabled: true })
        } else {
            None
        }
    }
}

/// Parameters for building the optional MLflow sink.
pub struct MlflowParams {
    /// Tracking server URI (the enable signal).
    pub tracking_uri: Option<String>,
    /// Experiment name.
    pub experiment: Option<String>,
    /// Run name.
    pub run_name: Option<String>,
    /// Parent MLflow run id.
    pub parent_run_id: Option<String>,
    /// Run tags (`--mlflow-tag k:v`).
    pub tags: Vec<(String, String)>,
    /// Artifact glob override (`--mlflow-artifact-glob`); empty keeps the default.
    pub artifact_globs: Vec<String>,
    /// Total expected requests (run request bound).
    pub total_expected_requests: Option<f64>,
}

/// Parameters for building the optional W&B sink.
pub struct WandbParams {
    /// Project (the enable signal).
    pub project: Option<String>,
    /// Entity.
    pub entity: Option<String>,
    /// Run name.
    pub run_name: Option<String>,
    /// Run tags (`--wandb-tag`).
    pub tags: Vec<String>,
}

impl MlflowExport {
    /// Build the MLflow sink when a tracking URI is configured.
    pub fn build(params: &MlflowParams, benchmark_id: &str) -> Option<Self> {
        params.tracking_uri.as_ref()?;
        Some(Self {
            enabled: true,
            aiperf_version: AIPERF_V1_VERSION.to_string(),
            artifact_globs: if params.artifact_globs.is_empty() {
                mlflow_artifact_globs()
            } else {
                params.artifact_globs.clone()
            },
            benchmark_id: benchmark_id.to_string(),
            tracking_uri: params.tracking_uri.clone(),
            experiment: params.experiment.clone(),
            run_name: params.run_name.clone(),
            parent_run_id: params.parent_run_id.clone(),
            total_expected_requests: params.total_expected_requests,
            tags: params.tags.iter().cloned().collect(),
            params: std::collections::BTreeMap::new(),
        })
    }
}

impl WandbExport {
    /// Build the W&B sink when a project is configured.
    pub fn build(params: &WandbParams, benchmark_id: &str) -> Option<Self> {
        params.project.as_ref()?;
        Some(Self {
            aiperf_version: AIPERF_V1_VERSION.to_string(),
            benchmark_id: benchmark_id.to_string(),
            cli_command: String::new(),
            config_json: "{}".to_string(),
            entity: params.entity.clone(),
            project: params.project.clone(),
            run_name: params.run_name.clone(),
            tags: (!params.tags.is_empty()).then(|| params.tags.clone()),
        })
    }
}

impl Export {
    /// Build the export policy for a run.
    ///
    /// `endpoint_type` selects the console title; `genai_perf_enabled` follows the
    /// `"json" in artifacts.summary` signal (default on). The envelope is
    /// projected functionally from the supplied opaque values.
    pub fn build(
        endpoint_type: &str,
        genai_perf_enabled: bool,
        benchmark_id: &str,
        input_config: serde_json::Value,
        run_info: serde_json::Value,
    ) -> Self {
        let envelope = GenaiPerfEnvelope {
            aiperf_version: AIPERF_V1_VERSION.to_string(),
            benchmark_id: serde_json::Value::String(benchmark_id.to_string()),
            input_config,
            run_info,
        };
        Export {
            genai_perf: GenaiPerf {
                enabled: genai_perf_enabled,
                header_map: META.header_map.clone(),
                filtered_tags: META.filtered_tags.clone(),
                scalar_tags: META.scalar_tags.clone(),
                envelope,
            },
            console_txt: ConsoleTxt {
                enabled: true,
                width: CONSOLE_EXPORT_WIDTH,
                dev: false,
                title: console_title(endpoint_type),
                metrics: META.console_metrics.clone(),
            },
            otel: None,
            mlflow: None,
            wandb: None,
            server_metrics: None,
            parquet: None,
        }
    }
}

/// Reproduce `ConsoleMetricsExporter._get_title`: `NVIDIA AIPerf | <metrics
/// title>` from the endpoint plugin metadata, degrading to `NVIDIA AIPerf` for a
/// runner-only dialect with no Python metadata.
fn console_title(endpoint_type: &str) -> String {
    match endpoint_metrics_title(endpoint_type) {
        Some(title) => format!("NVIDIA AIPerf | {title}"),
        None => "NVIDIA AIPerf".to_string(),
    }
}

/// The per-endpoint metrics title from Python's endpoint plugin metadata.
fn endpoint_metrics_title(endpoint_type: &str) -> Option<&'static str> {
    Some(match endpoint_type {
        "chat" | "completions" | "huggingface_generate" | "raw" | "responses" | "template" => {
            "LLM Metrics"
        }
        "chat_embeddings" | "embeddings" => "Embeddings Metrics",
        "cohere_rankings" | "hf_tei_rankings" => "Ranking Metrics",
        "image_edit" => "Image Edit Metrics",
        "image_generation" => "Image Generation Metrics",
        "image_retrieval" => "Image Retrieval Metrics",
        "nim_embeddings" => "NIM Embeddings Metrics",
        "nim_rankings" => "Rankings Metrics",
        "solido_rag" => "SOLIDO RAG Metrics",
        "video_generation" => "Video Generation Metrics",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_metadata_loads() {
        assert!(META.console_metrics.len() > 50);
        assert_eq!(META.header_map.len(), META.console_metrics.len());
    }

    #[test]
    fn chat_title() {
        assert_eq!(console_title("chat"), "NVIDIA AIPerf | LLM Metrics");
        assert_eq!(console_title("dynosim_offline"), "NVIDIA AIPerf");
    }

    #[test]
    fn server_metrics_export_default_formats_enable_json_and_csv() {
        let sm = ServerMetricsExport::build(
            &["json".to_string(), "csv".to_string()],
            true,
            "0.11.0",
            "abc123",
            serde_json::json!({"model": "m"}),
        )
        .expect("json/csv selected");
        assert!(sm.json && sm.csv);
        // Only the five runner-known keys are serialized (deny_unknown_fields).
        let value = serde_json::to_value(&sm).unwrap();
        let keys: std::collections::BTreeSet<&str> = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            [
                "aiperf_version",
                "benchmark_id",
                "csv",
                "input_config",
                "json"
            ]
            .into_iter()
            .collect()
        );
    }

    #[test]
    fn server_metrics_export_csv_only_drops_input_config() {
        let sm = ServerMetricsExport::build(
            &["csv".to_string()],
            true,
            "0.11.0",
            "abc123",
            serde_json::json!({"model": "m"}),
        )
        .expect("csv selected");
        assert!(!sm.json && sm.csv);
        assert!(sm.input_config.is_null());
        let value = serde_json::to_value(&sm).unwrap();
        assert!(
            value.get("input_config").is_none(),
            "csv export omits input_config"
        );
    }

    #[test]
    fn server_metrics_export_omitted_when_disabled_or_no_summary_format() {
        assert!(
            ServerMetricsExport::build(&["json".into()], false, "v", "id", serde_json::Value::Null)
                .is_none()
        );
        // jsonl/parquet-only selects no summary sink (those are runner-owned).
        assert!(
            ServerMetricsExport::build(
                &["jsonl".into(), "parquet".into()],
                true,
                "v",
                "id",
                serde_json::Value::Null
            )
            .is_none()
        );
    }

    #[test]
    fn parquet_export_gated_on_format_and_enabled() {
        assert_eq!(
            ParquetExport::build(&["json".into(), "parquet".into()], true).map(|p| p.enabled),
            Some(true)
        );
        assert!(ParquetExport::build(&["json".into(), "csv".into()], true).is_none());
        assert!(ParquetExport::build(&["parquet".into()], false).is_none());
    }
}
