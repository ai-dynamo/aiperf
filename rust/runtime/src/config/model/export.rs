// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed post-report export policy.
//!
//! Static console metadata is embedded byte-exact from
//! `resources/metric_metadata.json`. It intentionally differs from the native
//! metrics catalog and must not be regenerated from that catalog.
//! Envelope values are opaque and echoed verbatim by the runner.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

/// Aiperf-v1 export compatibility version, independent of the Rust crate version.
pub const AIPERF_V1_VERSION: &str = "0.11.0";
/// Default fixed console render width when unset.
const CONSOLE_EXPORT_WIDTH: u16 = 140;
/// Minimum authored console export width.
const CONSOLE_EXPORT_WIDTH_MIN: u16 = 40;
/// Maximum authored console export width.
const CONSOLE_EXPORT_WIDTH_MAX: u16 = 10000;

/// Resolve the fixed console-export render width from
/// `AIPERF_UI_CONSOLE_EXPORT_WIDTH`, defaulting to [`CONSOLE_EXPORT_WIDTH`] and
/// clamped to `[40, 10000]` to mirror Python's bounded `_UISettings` field. The
/// width pins `profile_export_console.txt` (and the non-tty live console)
/// independent of the terminal, so CI logs match the saved artifact.
fn console_export_width() -> u16 {
    std::env::var("AIPERF_UI_CONSOLE_EXPORT_WIDTH")
        .ok()
        .and_then(|value| value.trim().parse::<u16>().ok())
        .map(|value| value.clamp(CONSOLE_EXPORT_WIDTH_MIN, CONSOLE_EXPORT_WIDTH_MAX))
        .unwrap_or(CONSOLE_EXPORT_WIDTH)
}

/// Display metadata for one console metric.
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

/// Embedded metric display metadata.
#[derive(Debug, Deserialize)]
struct StaticMeta {
    console_metrics: BTreeMap<String, ConsoleMetricMeta>,
    header_map: BTreeMap<String, String>,
    filtered_tags: Vec<String>,
    scalar_tags: Vec<String>,
}

static META: LazyLock<StaticMeta> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../../../resources/metric_metadata.json"))
        .expect("embedded metric_metadata.json is valid")
});

/// Fixed-width console artifact policy.
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

/// Opaque values echoed by the genai-perf-v1 sink.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenaiPerfEnvelope {
    /// aiperf-v1 package version.
    pub aiperf_version: String,
    /// Run identifier.
    pub benchmark_id: serde_json::Value,
    /// Input configuration echoed verbatim.
    pub input_config: serde_json::Value,
    /// Run info.
    pub run_info: serde_json::Value,
}

/// Serde default for the summary-format toggles: both formats ship unless the
/// authored `artifacts.summary` list narrows them.
pub(crate) fn default_true() -> bool {
    true
}

/// Aiperf-v1 summary sink policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenaiPerf {
    /// Whether the sink runs at all (either format selected).
    pub enabled: bool,
    /// Whether `profile_export_aiperf.json` is written.
    #[serde(default = "crate::config::model::export::default_true")]
    pub json: bool,
    /// Whether `profile_export_aiperf.csv` is written.
    #[serde(default = "crate::config::model::export::default_true")]
    pub csv: bool,
    /// Per-tag display headers.
    pub header_map: BTreeMap<String, String>,
    /// Registered tags the file exporters drop (sorted).
    pub filtered_tags: Vec<String>,
    /// Registered scalar-tier tags (sorted).
    pub scalar_tags: Vec<String>,
    /// Envelope values echoed by the sink.
    pub envelope: GenaiPerfEnvelope,
}

/// OTLP/HTTP metrics sink.
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
    /// Build the OTLP sink, appending `/v1/metrics` when absent and merging
    /// `--otel-resource-attributes`.
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

/// Fixed MLflow artifact globs.
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

/// MLflow REST uploader sink.
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
    /// Best-effort logged parameters; currently empty.
    pub params: std::collections::BTreeMap<String, String>,
}

/// Weights & Biases sink.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WandbExport {
    /// aiperf-v1 version.
    pub aiperf_version: String,
    /// Run identifier.
    pub benchmark_id: String,
    /// Best-effort redacted invoking command; currently empty.
    pub cli_command: String,
    /// Best-effort serialized configuration; currently `{}`.
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
    /// Optional AIPerf datastore receiver URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sync_url: Option<String>,
}

/// Export policy; omitted sinks decode as disabled.
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
    /// Server-metrics JSON/CSV summary sink.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_metrics: Option<ServerMetricsExport>,
    /// Server-metrics Parquet sink.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parquet: Option<ParquetExport>,
}

/// Server-metrics summary sink policy.
///
/// Enabled only for selected JSON or CSV output. `input_config` is carried only
/// for JSON output, and unset optional values are omitted.
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
    /// Build the summary policy, or omit it when collection or summary formats
    /// are disabled.
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

/// Server-metrics Parquet sink toggle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParquetExport {
    /// Emit `server_metrics_export.parquet`.
    pub enabled: bool,
}

impl ParquetExport {
    /// Enable the sink only when collection and Parquet output are selected.
    pub fn build(formats: &[String], server_metrics_enabled: bool) -> Option<Self> {
        if server_metrics_enabled && formats.iter().any(|format| format == "parquet") {
            Some(Self { enabled: true })
        } else {
            None
        }
    }
}

/// Parameters for building the optional MLflow sink.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WandbParams {
    /// Project (the enable signal).
    pub project: Option<String>,
    /// Entity.
    pub entity: Option<String>,
    /// Run name.
    pub run_name: Option<String>,
    /// Run tags (`--wandb-tag`).
    pub tags: Vec<String>,
    /// Optional AIPerf datastore receiver URL.
    pub sync_url: Option<String>,
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
            sync_url: params.sync_url.clone(),
        })
    }
}

impl Export {
    /// Build the export policy for a run.
    ///
    /// `endpoint_type` selects the console title. Envelope values remain opaque.
    ///
    /// `summary_formats` is the authored `artifacts.summary` list; an empty list
    /// means unauthored and ships both formats.
    pub fn build(
        endpoint_type: &str,
        summary_formats: &[String],
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
        let unauthored = summary_formats.is_empty();
        let json = unauthored || summary_formats.iter().any(|f| f == "json");
        let csv = unauthored || summary_formats.iter().any(|f| f == "csv");
        Export {
            genai_perf: GenaiPerf {
                enabled: json || csv,
                json,
                csv,
                header_map: META.header_map.clone(),
                filtered_tags: META.filtered_tags.clone(),
                scalar_tags: META.scalar_tags.clone(),
                envelope,
            },
            console_txt: ConsoleTxt {
                enabled: true,
                width: console_export_width(),
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

/// Build the endpoint-specific console title, falling back to `NVIDIA AIPerf`.
fn console_title(endpoint_type: &str) -> String {
    match endpoint_metrics_title(endpoint_type) {
        Some(title) => format!("NVIDIA AIPerf | {title}"),
        None => "NVIDIA AIPerf".to_string(),
    }
}

/// The per-endpoint metrics title.
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
        // `console_metrics` is a superset of `header_map`: the sweepline metrics
        // (`effective_*`, `active_*`, `tokens_in_flight`) are keyed only for
        // console grouping.
        assert!(META.console_metrics.len() >= META.header_map.len());
        // Every genai-perf header tag also carries console metadata.
        for tag in META.header_map.keys() {
            assert!(
                META.console_metrics.contains_key(tag),
                "header_map tag {tag} missing console metadata"
            );
        }
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
