// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The typed native `BenchmarkConfig` — the runner-consumed `cfg` tree.
//!
//! Each section (`endpoint`, `datasets`, `phases`, `transport`, …) is added as a
//! fully-typed struct as it is ported from `src/aiperf/config/*.py` (input keys)
//! and `src/aiperf/orchestrator/rust_wire.py` (wire shape). Serializing this
//! struct yields the exact `run.cfg` subtree the runner consumes.
//!
//! `deny_unknown_fields` is intentionally omitted: deserializing a Python golden
//! through this type drops the sections not yet ported, which is exactly the
//! parity filter (see `crate::model`). Fields present here are fully typed.

use serde::{Deserialize, Serialize};

use super::artifacts::Artifacts;
use super::dataset::Dataset;
use super::endpoint::Endpoint;
use super::export::Export;
use super::metrics::Metrics;
use super::models::Models;
use super::phase::Phase;
use super::runtime::Runtime;
use super::telemetry::{GpuTelemetryConfig, NetworkLatencyConfig, ServerMetricsConfig, Sidecars};
use super::tokenizer::Tokenizer;
use super::transport::Transport;

/// The canonical benchmark configuration (runner-consumed projection).
///
/// Grows one typed section per port task. Sections not yet ported are simply
/// absent from this struct; a Python golden deserialized through it drops them
/// (no `deny_unknown_fields`), which is the parity filter. Every section field
/// is `Option` so a partial config (and a filtered golden) both round-trip;
/// `skip_serializing_if` keeps an unset section out of the serialized request
/// exactly as Python omits an unprojected section.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Model-selection policy (`cfg.models`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<Models>,
    /// Default endpoint profile (`cfg.endpoint`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<Endpoint>,
    /// Tokenizer acquisition policy (`cfg.tokenizer`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<Tokenizer>,
    /// Inline transport selection (`cfg.transport`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<Transport>,
    /// Worker/cell runtime policy (`cfg.runtime`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<Runtime>,
    /// Native metrics policy (`cfg.metrics`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Metrics>,
    /// Goodput SLO thresholds (`cfg.slos`, metric→threshold; open bag).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slos: Option<serde_json::Map<String, serde_json::Value>>,
    /// Native output policy (`cfg.artifacts`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<Artifacts>,
    /// Canonical single-dataset list (`cfg.datasets`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub datasets: Option<Vec<Dataset>>,
    /// Ordered phase policy (`cfg.phases`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phases: Option<Vec<Phase>>,
    /// Post-report export policy (`cfg.export`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export: Option<Export>,
    /// Raw GPU-telemetry policy (`cfg.gpu_telemetry`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_telemetry: Option<GpuTelemetryConfig>,
    /// Raw server-metrics policy (`cfg.server_metrics`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_metrics: Option<ServerMetricsConfig>,
    /// Raw network-latency policy (`cfg.network_latency`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network_latency: Option<NetworkLatencyConfig>,
    /// Lowered side-channel sidecars (`cfg.sidecars`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sidecars: Option<Sidecars>,
    // Further sections (slos, …) are added here as ported.
}
