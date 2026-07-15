// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed `export` section — the post-report sink policy the runner consumes.
//!
//! Mirrors the runner's `aiperf::export::ExportConfig` (and its `console_txt` /
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

/// The typed `export` policy. Only the sinks the frontend enables are modeled;
/// omitted sinks decode to the runner's all-disabled defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Export {
    /// aiperf-v1 summary sink.
    pub genai_perf: GenaiPerf,
    /// Console artifact sink.
    pub console_txt: ConsoleTxt,
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
}
