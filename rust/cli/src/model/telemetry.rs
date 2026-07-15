// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed telemetry/side-channel config blocks and the lowered `sidecars` block.
//!
//! GPU telemetry and server-metrics scraping are **default-enabled** in Config
//! v2, so their `cfg.*` blocks and the lowered `cfg.sidecars` entries appear on
//! every non-DynoSim run. Ported from
//! `src/aiperf/orchestrator/rust_wire.py::{_gpu_telemetry,_server_metrics,
//! _network_latency,_authored_sidecars}` and the `GPU`/`SERVER_METRICS`/
//! `NETWORK_LATENCY` environment defaults.

use serde::{Deserialize, Serialize};

/// Default sidecar cadence: 0.333 s in nanoseconds.
const COLLECTION_INTERVAL_NS: u64 = 333_000_000;
/// Default reachability timeout: 10 s in nanoseconds.
const REACHABILITY_TIMEOUT_NS: u64 = 10_000_000_000;
/// Default DCGM exporter endpoints (`Environment.GPU.DEFAULT_DCGM_ENDPOINTS`).
const DEFAULT_DCGM_ENDPOINTS: [&str; 2] = ["localhost:9400", "localhost:9401"];

/// K8s pod-discovery policy for server metrics (`cfg.server_metrics.discovery`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerMetricsDiscovery {
    /// Discovery mode (`auto` by default).
    pub mode: String,
    /// K8s namespace filter.
    pub namespace: Option<String>,
    /// K8s label selector.
    pub label_selector: Option<String>,
}

impl Default for ServerMetricsDiscovery {
    fn default() -> Self {
        Self {
            mode: "auto".to_string(),
            namespace: None,
            label_selector: None,
        }
    }
}

/// The raw `cfg.server_metrics` policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerMetricsConfig {
    /// Whether server-metrics scraping is enabled (default true).
    pub enabled: bool,
    /// Output formats.
    pub formats: Vec<String>,
    /// Explicit scrape URLs (endpoint-derived at lowering time).
    pub urls: Vec<String>,
    /// Pod-discovery policy.
    pub discovery: ServerMetricsDiscovery,
}

impl Default for ServerMetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            formats: vec!["json".to_string(), "csv".to_string()],
            urls: Vec::new(),
            discovery: ServerMetricsDiscovery::default(),
        }
    }
}

/// The raw `cfg.gpu_telemetry` policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuTelemetryConfig {
    /// Whether GPU telemetry is enabled (default true).
    pub enabled: bool,
    /// Collector backend id.
    pub collector: String,
    /// Summary vs dashboard mode.
    pub mode: String,
    /// Custom-metrics CSV path.
    pub metrics_file: Option<String>,
    /// Explicit DCGM URLs (merged with the defaults at lowering time).
    pub urls: Vec<String>,
}

impl Default for GpuTelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collector: "dcgm".to_string(),
            mode: "summary".to_string(),
            metrics_file: None,
            urls: Vec::new(),
        }
    }
}

/// The raw `cfg.network_latency` policy (disabled by default).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkLatencyConfig {
    /// Whether RTT calibration runs.
    pub enabled: bool,
    /// Fixed mean RTT override, milliseconds.
    pub mean_ms: Option<f64>,
    /// Probe ping interval, seconds.
    pub ping_interval: f64,
}

impl Default for NetworkLatencyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mean_ms: None,
            ping_interval: 1.0,
        }
    }
}

/// One lowered GPU telemetry source.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuSource {
    /// Source type (`dcgm`).
    #[serde(rename = "type")]
    pub source_type: String,
    /// Scrape URL.
    pub url: String,
}

/// The lowered `sidecars.gpu_telemetry` block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuTelemetrySidecar {
    /// Scrape cadence, nanoseconds.
    pub collection_interval_ns: u64,
    /// Reachability timeout, nanoseconds.
    pub request_timeout_ns: u64,
    /// Output-relative records path.
    pub records_path: String,
    /// Lowered sources.
    pub sources: Vec<GpuSource>,
}

/// The lowered `sidecars.server_metrics` block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerMetricsSidecar {
    /// Scrape cadence, nanoseconds.
    pub collection_interval_ns: u64,
    /// Reachability timeout, nanoseconds.
    pub reachability_timeout_ns: u64,
    /// Scrape URLs (endpoint-derived).
    pub urls: Vec<String>,
    /// Output formats.
    pub formats: Vec<String>,
    /// Per-scrape JSONL path (present when `jsonl` in formats).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jsonl_path: Option<String>,
    /// Parquet wire-record path (present when `parquet` in formats).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parquet_wire_path: Option<String>,
}

/// The RTT-calibration probe (`_network_latency` automatic branch).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkLatencyProbe {
    /// Ping interval, nanoseconds.
    pub ping_interval_ns: u64,
    /// TCP connect timeout, nanoseconds.
    pub connect_timeout_ns: u64,
    /// Completion top-up timeout, nanoseconds.
    pub complete_topup_timeout_ns: u64,
    /// Minimum successful samples.
    pub min_successful_samples: u64,
    /// Output-relative records path.
    pub records_path: String,
}

/// The lowered `sidecars.network_latency` block (fixed mean or a probe).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkLatencySidecar {
    /// Fixed mean RTT, nanoseconds (fixed-mean mode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_rtt_ns: Option<u64>,
    /// RTT probe (automatic mode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe: Option<NetworkLatencyProbe>,
}

impl NetworkLatencySidecar {
    /// Build a fixed-mean sidecar (`--network-latency-mean` ms → ns).
    pub fn fixed(mean_ms: f64) -> Self {
        Self {
            mean_rtt_ns: Some((mean_ms / 1000.0 * 1e9).round() as u64),
            probe: None,
        }
    }

    /// Build a probe sidecar (`--network-latency-automatic`).
    pub fn probe(ping_interval_seconds: f64) -> Self {
        Self {
            mean_rtt_ns: None,
            probe: Some(NetworkLatencyProbe {
                ping_interval_ns: (ping_interval_seconds * 1e9).round() as u64,
                connect_timeout_ns: 5_000_000_000,
                complete_topup_timeout_ns: 3_000_000_000,
                min_successful_samples: 5,
                records_path: "profile_export_network_latency.jsonl".to_string(),
            }),
        }
    }
}

/// The lowered `cfg.sidecars` block.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Sidecars {
    /// GPU telemetry sidecar (present when enabled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_telemetry: Option<GpuTelemetrySidecar>,
    /// Server-metrics sidecar (present when enabled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_metrics: Option<ServerMetricsSidecar>,
    /// Network-latency sidecar (present when enabled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network_latency: Option<NetworkLatencySidecar>,
}

impl GpuTelemetrySidecar {
    /// Build the default DCGM sidecar (the enabled-by-default path); `extra` are
    /// custom DCGM URLs appended after the defaults (deduped).
    pub fn default_dcgm(extra: &[String]) -> Self {
        let mut urls: Vec<String> = DEFAULT_DCGM_ENDPOINTS
            .iter()
            .map(|e| normalize_metrics_url(e))
            .collect();
        for u in extra {
            let n = normalize_metrics_url(u);
            if !urls.contains(&n) {
                urls.push(n);
            }
        }
        Self {
            collection_interval_ns: COLLECTION_INTERVAL_NS,
            request_timeout_ns: REACHABILITY_TIMEOUT_NS,
            records_path: "gpu_telemetry_export.jsonl".to_string(),
            sources: urls
                .into_iter()
                .map(|url| GpuSource {
                    source_type: "dcgm".to_string(),
                    url,
                })
                .collect(),
        }
    }
}

impl ServerMetricsSidecar {
    /// Build the server-metrics sidecar scraping each endpoint URL's `/metrics`.
    pub fn from_endpoint_urls(endpoint_urls: &[String]) -> Self {
        let mut urls: Vec<String> = Vec::new();
        for url in endpoint_urls {
            let normalized = normalize_metrics_url(url);
            if !urls.contains(&normalized) {
                urls.push(normalized);
            }
        }
        Self {
            collection_interval_ns: COLLECTION_INTERVAL_NS,
            reachability_timeout_ns: REACHABILITY_TIMEOUT_NS,
            urls,
            formats: vec!["json".to_string(), "csv".to_string()],
            jsonl_path: None,
            parquet_wire_path: None,
        }
    }

    /// Apply the output formats, deriving the per-format artifact paths
    /// (`_server_metrics`: `jsonl` → jsonl_path, `parquet` → parquet_wire_path).
    pub fn with_formats(mut self, formats: Vec<String>) -> Self {
        self.jsonl_path = formats
            .iter()
            .any(|f| f == "jsonl")
            .then(|| "server_metrics_export.jsonl".to_string());
        self.parquet_wire_path = formats
            .iter()
            .any(|f| f == "parquet")
            .then(|| ".aiperf-server-metrics-parquet-wire.jsonl".to_string());
        self.formats = formats;
        self
    }
}

/// Normalize a scrape target to end with `/metrics`, exactly reproducing
/// `normalize_metrics_endpoint_url`: prepend `http://` only when the URL does
/// not already start with `http://`/`https://` (so a `grpc://` URL becomes the
/// intentional `http://grpc://.../metrics`), strip trailing slashes, then append
/// `/metrics` unless already present.
pub fn normalize_metrics_url(url: &str) -> String {
    let mut url = if url.starts_with("http://") || url.starts_with("https://") {
        url.to_string()
    } else {
        format!("http://{url}")
    };
    let trimmed_len = url.trim_end_matches('/').len();
    url.truncate(trimmed_len);
    if !url.ends_with("/metrics") {
        url.push_str("/metrics");
    }
    url
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_url_normalization() {
        assert_eq!(
            normalize_metrics_url("http://127.0.0.1:8000"),
            "http://127.0.0.1:8000/metrics"
        );
        assert_eq!(
            normalize_metrics_url("localhost:9400"),
            "http://localhost:9400/metrics"
        );
        // A non-http scheme is treated as a bare host and prefixed (the same
        // quirk Python has for grpc:// endpoints).
        assert_eq!(
            normalize_metrics_url("grpc://127.0.0.1:8001"),
            "http://grpc://127.0.0.1:8001/metrics"
        );
        // A custom path still gets /metrics appended.
        assert_eq!(
            normalize_metrics_url("http://h:9/custom"),
            "http://h:9/custom/metrics"
        );
    }
}
