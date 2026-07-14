// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry OTLP/HTTP metrics emitter (native Rust).
//!
//! Ports the Python OTel plane (`post_processors/otel_metrics_results_processor.py`,
//! `otel_streaming_fanout.py`, `strategies/genai_semconv.py`) to the runner: emits
//! GenAI-semconv histograms/counters (`gen_ai.client.operation.duration`,
//! `gen_ai.client.token.usage`, TTFT/ITL client metrics) and `aiperf.*` timing
//! counters over OTLP/HTTP to `--otel-url`, with the same resource attributes
//! (`service.name`, `aiperf.benchmark.id`, `aiperf.model.name`, …) and the same
//! histogram bucket boundaries. Parity oracle: the OTLP `ExportMetricsServiceRequest`
//! bodies a collector receives must carry the same metric names, attributes, and
//! bucket layout as the Python emitter for an identical run.
//!
//! The commit site is synchronous with no ambient runtime; this sink drives its
//! own short-lived `current_thread` tokio runtime for the OTLP POST and enforces
//! a hard wall-clock timeout so an unreachable collector cannot hang shutdown
//! (spec §6). `aiperf` already carries `hyper`/`prost`/`tonic`, so no new HTTP
//! dependency is required.
//!
//! STATUS: config + gating are wired; the OTLP emission body is unimplemented
//! (Worker B).

use std::path::Path;

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::NativeReport;

/// OTLP/HTTP metrics export policy. Disabled unless the frontend provides an
/// OTLP endpoint. Fields mirror the Python `config/otel.py` surface the emitter
/// needs; Worker B extends this struct as required (own-file edit only).
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OtelExportConfig {
    /// Whether OTLP metric export is enabled (frontend sets true when an OTLP
    /// endpoint is configured).
    pub enabled: bool,
    /// Normalized OTLP/HTTP metrics endpoint (`http(s)://host[:port]/v1/metrics`).
    pub endpoint: Option<String>,
    /// Optional GenAI provider-name override (`gen_ai.provider.name`); inferred
    /// from the endpoint host when absent.
    pub provider: Option<String>,
    /// Extra OTLP resource attributes (`--otel-resource-attributes`).
    #[serde(default)]
    pub resource_attributes: std::collections::BTreeMap<String, String>,
}

/// The OTLP/HTTP metrics [`Exporter`].
pub struct OtelExporter;

impl Exporter for OtelExporter {
    fn name(&self) -> &'static str {
        "otel"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.otel.enabled && cfg.otel.endpoint.is_some()
    }

    fn export(
        &self,
        _report: &NativeReport,
        _artifact_dir: &Path,
        _cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        // Worker B: build the OTLP ExportMetricsServiceRequest from the report's
        // typed metrics and POST it under a hard timeout. Inert at foundation.
        anyhow::bail!("native OTLP metrics sink not yet implemented");
    }
}
