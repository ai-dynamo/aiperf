// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-neutral sidecar policy and direct authored-input adapters.
//!
//! Protocol v1 re-exports the policy values for compatibility. Protocol v2
//! retains each sidecar body as raw JSON until the injected resolver selects
//! exactly one adapter, performs the sole strict decode, and stores the typed
//! result in [`PreparedSidecarInputs`]. Runtime resource preparation is a
//! separate seam: it may open sockets or supervise Python workers only after
//! every authored input has passed this side-effect-free stage.

use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

/// Stable built-in GPU telemetry sidecar ID.
pub const GPU_TELEMETRY_SIDECAR_ID: &str = "gpu_telemetry";
/// Stable built-in generated-content HTTP server sidecar ID.
pub const CONTENT_SERVER_SIDECAR_ID: &str = "content_server";
/// Stable built-in network-latency sidecar ID.
pub const NETWORK_LATENCY_SIDECAR_ID: &str = "network_latency";
/// Stable built-in server-metrics sidecar ID.
pub const SERVER_METRICS_SIDECAR_ID: &str = "server_metrics";
/// Stable built-in live-results sidecar ID.
pub const LIVE_STREAMING_SIDECAR_ID: &str = "live_streaming";

/// Run-owned HTTP content-server and synthetic-media publication policy.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContentServerSpec {
    /// Host/interface to bind and advertise in generated media URLs.
    pub host: String,
    /// TCP port.
    pub port: u16,
    /// Existing directory to serve. Absence creates a temporary serving root and
    /// leaves synthetic media inline.
    #[serde(default)]
    pub content_dir: Option<PathBuf>,
    /// Bounded recent-request record capacity.
    pub max_tracked_records: usize,
}

impl ContentServerSpec {
    /// HTTP base URL embedded in generated image/video values.
    pub fn base_url(&self) -> String {
        let host = if self.host.parse::<std::net::Ipv6Addr>().is_ok() {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        format!("http://{host}:{}", self.port)
    }
}

/// Canonical Python live-results extension configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LiveStreamingSpec {
    /// Absolute interpreter selected by the Python Config-v2 parent.
    pub python_executable: PathBuf,
    /// Importable strict-stdio worker module.
    #[serde(default = "default_live_streaming_worker_module")]
    pub worker_module: String,
    /// Bounded Rust-to-Python queue capacity with drop-oldest overflow.
    pub buffer_capacity: usize,
    /// Canonical OpenTelemetry streaming settings.
    pub otel: OTelStreamingSpec,
    /// Canonical live-MLflow settings.
    pub mlflow: MLflowStreamingSpec,
}

fn default_live_streaming_worker_module() -> String {
    "aiperf.post_processors.native_streaming_worker".to_string()
}

/// OpenTelemetry settings forwarded to the canonical Python processor.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OTelStreamingSpec {
    /// OTLP/HTTP metrics endpoint.
    #[serde(default)]
    pub metrics_url: Option<String>,
    /// Emit terminal request metric records.
    pub stream_metrics_enabled: bool,
    /// Emit phase lifecycle and progress records.
    pub stream_timing_enabled: bool,
    /// User-authored OTel resource attributes.
    #[serde(default)]
    pub custom_resource_attributes: BTreeMap<String, String>,
    /// Optional GenAI semantic-convention provider override.
    #[serde(default)]
    pub gen_ai_provider: Option<String>,
}

/// Live MLflow settings forwarded to the canonical Python fanout.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MLflowStreamingSpec {
    /// MLflow tracking server URI.
    #[serde(default)]
    pub tracking_uri: Option<String>,
    /// Experiment name.
    pub experiment: String,
    /// Optional run name.
    #[serde(default)]
    pub run_name: Option<String>,
    /// Optional run tags.
    #[serde(default)]
    pub tags: Option<BTreeMap<String, String>>,
    /// Optional parent run identity.
    #[serde(default)]
    pub parent_run_id: Option<String>,
    /// Optional post-run artifact selection retained in fanout metadata.
    #[serde(default)]
    pub artifact_globs: Option<Vec<String>>,
}

/// Low-rate GPU telemetry synchronized to the profiling phase.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuTelemetrySpec {
    /// Clock cadence between continuous scrapes.
    pub collection_interval_ns: i64,
    /// Clock deadline applied independently to each telemetry HTTP request.
    pub request_timeout_ns: i64,
    /// Per-GPU JSONL path relative to the run directory.
    pub records_path: PathBuf,
    /// Ordered source list after Config-v2 default expansion and deduplication.
    pub sources: Vec<GpuTelemetrySourceSpec>,
    /// Config-v2 custom DCGM fields registered for native sidecar reporting.
    #[serde(default)]
    pub custom_metrics: Vec<GpuTelemetryMetricSpec>,
    /// Optional custom DCGM metrics CSV (`--gpu-telemetry <file>.csv`). When set,
    /// the native DCGM decoder is extended with the CSV's additional exporter
    /// fields and the accumulator registers each parsed metric spec.
    #[serde(default)]
    pub metrics_file: Option<PathBuf>,
}

/// Run-level network RTT calibration lowered from Config v2.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkLatencySpec {
    /// Fixed mean RTT in nanoseconds; mutually exclusive with active probing.
    #[serde(default)]
    pub mean_rtt_ns: Option<f64>,
    /// Active fresh-TCP probe policy; mutually exclusive with a fixed mean.
    #[serde(default)]
    pub probe: Option<NetworkLatencyProbeSpec>,
}

/// Profiling-bounded fresh-TCP probe policy.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkLatencyProbeSpec {
    /// Clock cadence between probe issuances.
    pub ping_interval_ns: i64,
    /// Per-connect Clock deadline.
    pub connect_timeout_ns: i64,
    /// Global Clock budget for phase-end sample top-up.
    pub complete_topup_timeout_ns: i64,
    /// Successful-sample floor applied independently to every unique target.
    pub min_successful_samples: usize,
    /// Per-sample JSONL path relative to the run directory.
    pub records_path: PathBuf,
}

/// Inference-server Prometheus collection and artifact policy.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerMetricsSpec {
    /// Clock cadence between sequential continuous scrapes.
    pub collection_interval_ns: i64,
    /// Clock deadline used for source reachability/connect attempts.
    pub reachability_timeout_ns: i64,
    /// Ordered normalized endpoints after inference and explicit URL expansion.
    pub urls: Vec<String>,
    /// Canonical compatibility artifacts requested by Config v2.
    pub formats: Vec<ServerMetricsFormatSpec>,
    /// Slim JSONL output relative to the run directory when requested.
    #[serde(default)]
    pub jsonl_path: Option<PathBuf>,
    /// Full-record handoff relative to the run directory for Python Parquet rendering.
    #[serde(default)]
    pub parquet_wire_path: Option<PathBuf>,
}

/// Config-v2 server-metrics export formats.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ServerMetricsFormatSpec {
    /// Aggregate JSON rendered by Python's canonical exporter.
    Json,
    /// Aggregate CSV rendered by Python's canonical exporter.
    Csv,
    /// Slim per-scrape JSONL written by Rust.
    Jsonl,
    /// Raw time-series Parquet rendered by Python's canonical exporter.
    Parquet,
}

/// One injected GPU telemetry source.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum GpuTelemetrySourceSpec {
    /// NVIDIA DCGM Prometheus endpoint collected by Rust HTTP.
    Dcgm {
        /// Metrics endpoint; `/metrics` is appended when absent.
        url: String,
    },
    /// Canonical Python collector or user extension supervised by Rust.
    Python {
        /// Registered Config-v2 collector name.
        collector: String,
        /// Optional remote endpoint used by the DCGM collector.
        #[serde(default)]
        url: Option<String>,
        /// Optional custom DCGM metrics definition.
        #[serde(default)]
        metrics_file: Option<PathBuf>,
        /// Absolute interpreter selected by the Python orchestrator.
        python_executable: PathBuf,
        /// Importable strict-stdio worker module.
        worker_module: String,
    },
}

/// One Config-v2 custom GPU signal exposed in native-v2 output.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuTelemetryMetricSpec {
    /// Stable normalized field name emitted by the Python collector.
    pub name: String,
    /// Human-readable metric label.
    pub header: String,
    /// Native report unit.
    pub unit: GpuTelemetryUnitSpec,
}

/// Config-v2 GPU unit vocabulary accepted by the native report engine.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuTelemetryUnitSpec {
    /// Unitless count.
    Count,
    /// Kibibytes.
    Kilobyte,
    /// Mebibytes.
    Megabyte,
    /// Gibibytes.
    Gigabyte,
    /// Microseconds.
    Microsecond,
    /// Milliseconds.
    Millisecond,
    /// Seconds.
    Second,
    /// Percentage.
    Percent,
    /// Watts.
    Watt,
    /// Joules.
    Joule,
    /// Megajoules.
    Megajoule,
    /// Megahertz.
    Megahertz,
    /// Gigahertz.
    Gigahertz,
    /// Celsius.
    Celsius,
}

/// One raw sidecar input paired with the open adapter ID selected by its key.
#[derive(Clone, Copy)]
pub struct AuthoredSidecarInput<'a> {
    /// Stable adapter identity.
    pub id: &'a str,
    /// Adapter-owned strict JSON body.
    pub config: &'a RawValue,
}

impl fmt::Debug for AuthoredSidecarInput<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthoredSidecarInput")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

/// Type-erased, strictly validated sidecar input retained after selection.
pub trait ValidatedSidecarInput: fmt::Debug + Send + Sync {
    /// Borrow the concrete adapter-owned value for startup-only downcasting.
    fn as_any(&self) -> &dyn Any;
    /// Consume the value for startup-only typed extraction.
    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync>;
}

impl<T> ValidatedSidecarInput for T
where
    T: Any + fmt::Debug + Send + Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync> {
        self
    }
}

/// Direct adapter from one authored sidecar body to its retained typed input.
pub trait SidecarInputAdapter: fmt::Debug + Send + Sync {
    /// Stable sidecar input ID.
    fn input_id(&self) -> &'static str;

    /// Perform the sole full strict decode and semantic validation.
    fn validate(&self, raw: &RawValue) -> Result<Box<dyn ValidatedSidecarInput>>;
}

/// Injected resolver over an open, deterministic sidecar adapter registry.
pub trait SidecarInputAdapterResolver: fmt::Debug + Send + Sync {
    /// Select and validate every authored input exactly once.
    fn prepare(&self, authored: &[AuthoredSidecarInput<'_>]) -> Result<PreparedSidecarInputs>;
}

/// One retained set of direct sidecar adapter outputs.
#[derive(Default)]
pub struct PreparedSidecarInputs {
    entries: BTreeMap<String, Box<dyn ValidatedSidecarInput>>,
}

impl fmt::Debug for PreparedSidecarInputs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedSidecarInputs")
            .field("ids", &self.entries.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl PreparedSidecarInputs {
    /// Iterate prepared IDs in deterministic order.
    pub fn ids(&self) -> impl ExactSizeIterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Borrow one exact adapter-owned value without re-decoding its wire body.
    pub fn get<T: Any>(&self, id: &str) -> Result<Option<&T>> {
        self.entries
            .get(id)
            .map(|value| {
                value.as_ref().as_any().downcast_ref::<T>().ok_or_else(|| {
                    anyhow!("prepared sidecar input {id:?} has a different concrete adapter type")
                })
            })
            .transpose()
    }

    /// Consume one exact adapter-owned value for runtime resource preparation.
    pub fn take<T: Any + Send + Sync>(&mut self, id: &str) -> Result<Option<T>> {
        self.entries
            .remove(id)
            .map(|value| {
                value
                    .into_any()
                    .downcast::<T>()
                    .map(|value| *value)
                    .map_err(|_| {
                        anyhow!(
                            "prepared sidecar input {id:?} has a different concrete adapter type"
                        )
                    })
            })
            .transpose()
    }

    /// Return whether no sidecar input was authored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return whether one exact adapter ID was authored.
    pub fn contains(&self, id: &str) -> bool {
        self.entries.contains_key(id)
    }

    /// Return whether every authored input is in the supplied allowlist.
    pub fn contains_only(&self, allowed: &[&str]) -> bool {
        self.entries.keys().all(|id| allowed.contains(&id.as_str()))
    }
}

/// Deterministic built-in sidecar-input adapter composition.
pub struct BuiltinRunnerSidecarInputAdapterResolver {
    adapters: BTreeMap<&'static str, Arc<dyn SidecarInputAdapter>>,
}

impl fmt::Debug for BuiltinRunnerSidecarInputAdapterResolver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BuiltinRunnerSidecarInputAdapterResolver")
            .field("ids", &self.adapters.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl Default for BuiltinRunnerSidecarInputAdapterResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl BuiltinRunnerSidecarInputAdapterResolver {
    /// Compose the built-in adapters in stable ID order.
    pub fn new() -> Self {
        let adapters: [Arc<dyn SidecarInputAdapter>; 5] = [
            Arc::new(ContentServerInputAdapter),
            Arc::new(GpuTelemetryInputAdapter),
            Arc::new(LiveStreamingInputAdapter),
            Arc::new(NetworkLatencyInputAdapter),
            Arc::new(ServerMetricsInputAdapter),
        ];
        Self {
            adapters: adapters
                .into_iter()
                .map(|adapter| (adapter.input_id(), adapter))
                .collect(),
        }
    }
}

impl SidecarInputAdapterResolver for BuiltinRunnerSidecarInputAdapterResolver {
    fn prepare(&self, authored: &[AuthoredSidecarInput<'_>]) -> Result<PreparedSidecarInputs> {
        let mut entries = BTreeMap::new();
        for input in authored {
            ensure!(
                !input.id.is_empty() && input.id.trim() == input.id,
                "sidecar input ID must be non-empty and contain no surrounding whitespace"
            );
            ensure!(
                !entries.contains_key(input.id),
                "duplicate authored sidecar input ID {:?}",
                input.id
            );
            let adapter = self.adapters.get(input.id).ok_or_else(|| {
                anyhow!(
                    "no sidecar-input adapter is registered for {:?}; available: {}",
                    input.id,
                    self.adapters.keys().copied().collect::<Vec<_>>().join(", ")
                )
            })?;
            let prepared = adapter
                .validate(input.config)
                .with_context(|| format!("preparing sidecar input {:?}", input.id))?;
            entries.insert(input.id.to_owned(), prepared);
        }
        Ok(PreparedSidecarInputs { entries })
    }
}

#[derive(Debug)]
struct ContentServerInputAdapter;
#[derive(Debug)]
struct GpuTelemetryInputAdapter;
#[derive(Debug)]
struct NetworkLatencyInputAdapter;
#[derive(Debug)]
struct ServerMetricsInputAdapter;
#[derive(Debug)]
struct LiveStreamingInputAdapter;

impl SidecarInputAdapter for ContentServerInputAdapter {
    fn input_id(&self) -> &'static str {
        CONTENT_SERVER_SIDECAR_ID
    }

    fn validate(&self, raw: &RawValue) -> Result<Box<dyn ValidatedSidecarInput>> {
        let spec = strict_decode::<ContentServerSpec>(raw, self.input_id())?;
        ensure_nonempty(&spec.host, "host")?;
        ensure!(
            !(spec.host.starts_with('[') || spec.host.ends_with(']')),
            "host must use a bare IPv6 literal rather than URL bracket syntax"
        );
        ensure!(spec.port > 0, "port must be between 1 and 65535");
        ensure!(
            (100..=1_000_000).contains(&spec.max_tracked_records),
            "max_tracked_records must be between 100 and 1000000"
        );
        let base_url = spec.base_url();
        let parsed = url::Url::parse(&base_url).context("parsing derived content-server URL")?;
        ensure!(
            parsed.host_str().is_some()
                && parsed.username().is_empty()
                && parsed.password().is_none()
                && parsed.port_or_known_default() == Some(spec.port)
                && parsed.path() == "/"
                && parsed.query().is_none()
                && parsed.fragment().is_none(),
            "host does not produce a plain HTTP origin"
        );
        if let Some(path) = &spec.content_dir {
            ensure!(!path.as_os_str().is_empty(), "content_dir cannot be empty");
            ensure!(path.is_absolute(), "content_dir must be absolute");
        }
        Ok(Box::new(spec))
    }
}

impl SidecarInputAdapter for GpuTelemetryInputAdapter {
    fn input_id(&self) -> &'static str {
        GPU_TELEMETRY_SIDECAR_ID
    }

    fn validate(&self, raw: &RawValue) -> Result<Box<dyn ValidatedSidecarInput>> {
        let spec = strict_decode::<GpuTelemetrySpec>(raw, self.input_id())?;
        ensure!(
            spec.collection_interval_ns > 0,
            "collection_interval_ns must be positive"
        );
        ensure!(
            spec.request_timeout_ns > 0,
            "request_timeout_ns must be positive"
        );
        ensure!(!spec.sources.is_empty(), "at least one source is required");
        validate_relative_path(&spec.records_path, "records_path")?;
        for source in &spec.sources {
            match source {
                GpuTelemetrySourceSpec::Dcgm { url } => ensure_nonempty(url, "DCGM url")?,
                GpuTelemetrySourceSpec::Python {
                    collector,
                    url,
                    python_executable,
                    worker_module,
                    ..
                } => {
                    ensure_nonempty(collector, "Python collector")?;
                    if let Some(url) = url {
                        ensure_nonempty(url, "Python collector url")?;
                    }
                    ensure!(
                        python_executable.is_absolute(),
                        "python_executable must be absolute"
                    );
                    ensure_nonempty(worker_module, "worker_module")?;
                }
            }
        }
        for metric in &spec.custom_metrics {
            ensure_nonempty(&metric.name, "custom metric name")?;
            ensure_nonempty(&metric.header, "custom metric header")?;
        }
        Ok(Box::new(spec))
    }
}

impl SidecarInputAdapter for NetworkLatencyInputAdapter {
    fn input_id(&self) -> &'static str {
        NETWORK_LATENCY_SIDECAR_ID
    }

    fn validate(&self, raw: &RawValue) -> Result<Box<dyn ValidatedSidecarInput>> {
        let spec = strict_decode::<NetworkLatencySpec>(raw, self.input_id())?;
        ensure!(
            spec.mean_rtt_ns.is_some() ^ spec.probe.is_some(),
            "exactly one of mean_rtt_ns or probe is required"
        );
        if let Some(mean_rtt_ns) = spec.mean_rtt_ns {
            ensure!(
                mean_rtt_ns.is_finite() && mean_rtt_ns >= 0.0,
                "mean_rtt_ns must be finite and non-negative"
            );
        }
        if let Some(probe) = &spec.probe {
            ensure!(
                probe.ping_interval_ns > 0,
                "ping_interval_ns must be positive"
            );
            ensure!(
                probe.connect_timeout_ns > 0,
                "connect_timeout_ns must be positive"
            );
            ensure!(
                probe.complete_topup_timeout_ns >= 0,
                "complete_topup_timeout_ns must be non-negative"
            );
            ensure!(
                probe.min_successful_samples > 0,
                "min_successful_samples must be positive"
            );
            validate_relative_path(&probe.records_path, "probe.records_path")?;
        }
        Ok(Box::new(spec))
    }
}

impl SidecarInputAdapter for ServerMetricsInputAdapter {
    fn input_id(&self) -> &'static str {
        SERVER_METRICS_SIDECAR_ID
    }

    fn validate(&self, raw: &RawValue) -> Result<Box<dyn ValidatedSidecarInput>> {
        let spec = strict_decode::<ServerMetricsSpec>(raw, self.input_id())?;
        ensure!(
            spec.collection_interval_ns > 0,
            "collection_interval_ns must be positive"
        );
        ensure!(
            spec.reachability_timeout_ns > 0,
            "reachability_timeout_ns must be positive"
        );
        ensure!(
            !spec.urls.is_empty(),
            "at least one metrics URL is required"
        );
        for url in &spec.urls {
            ensure_nonempty(url, "metrics URL")?;
        }
        ensure!(
            !spec.formats.is_empty(),
            "at least one export format is required"
        );
        ensure!(
            spec.formats.contains(&ServerMetricsFormatSpec::Jsonl) == spec.jsonl_path.is_some(),
            "jsonl_path must be present exactly when the jsonl format is selected"
        );
        ensure!(
            spec.formats.contains(&ServerMetricsFormatSpec::Parquet)
                == spec.parquet_wire_path.is_some(),
            "parquet_wire_path must be present exactly when the parquet format is selected"
        );
        if let Some(path) = &spec.jsonl_path {
            validate_relative_path(path, "jsonl_path")?;
        }
        if let Some(path) = &spec.parquet_wire_path {
            validate_relative_path(path, "parquet_wire_path")?;
        }
        Ok(Box::new(spec))
    }
}

impl SidecarInputAdapter for LiveStreamingInputAdapter {
    fn input_id(&self) -> &'static str {
        LIVE_STREAMING_SIDECAR_ID
    }

    fn validate(&self, raw: &RawValue) -> Result<Box<dyn ValidatedSidecarInput>> {
        let spec = strict_decode::<LiveStreamingSpec>(raw, self.input_id())?;
        ensure!(
            spec.python_executable.is_absolute(),
            "python_executable must be absolute"
        );
        ensure_nonempty(&spec.worker_module, "worker_module")?;
        ensure!(spec.buffer_capacity > 0, "buffer_capacity must be positive");
        ensure!(
            spec.otel.metrics_url.is_some() || spec.mlflow.tracking_uri.is_some(),
            "at least one OTel or MLflow destination is required"
        );
        Ok(Box::new(spec))
    }
}

fn strict_decode<T: DeserializeOwned>(raw: &RawValue, id: &str) -> Result<T> {
    // Buffer through `serde_json::Value` so `#[serde(flatten)]`/`#[serde(untagged)]`
    // f64 fields decode correctly under the build-wide `arbitrary_precision`
    // feature (see `registry::strict_decode` for the full rationale).
    let value: serde_json::Value = serde_json::from_str(raw.get())
        .with_context(|| format!("decoding {id:?} sidecar input"))?;
    serde_json::from_value(value).with_context(|| format!("decoding {id:?} sidecar input"))
}

fn ensure_nonempty(value: &str, field: &str) -> Result<()> {
    ensure!(
        !value.trim().is_empty() && value.trim() == value,
        "{field} must be non-empty and contain no surrounding whitespace"
    );
    Ok(())
}

fn validate_relative_path(path: &Path, field: &str) -> Result<()> {
    ensure!(!path.as_os_str().is_empty(), "{field} cannot be empty");
    ensure!(!path.is_absolute(), "{field} must be relative");
    ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "{field} must contain only normal relative path components"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(value: serde_json::Value) -> Box<RawValue> {
        RawValue::from_string(value.to_string()).unwrap()
    }

    #[test]
    fn builtins_prepare_direct_typed_inputs_in_deterministic_order() {
        let content = raw(serde_json::json!({
            "host": "0.0.0.0",
            "port": 8090,
            "content_dir": "/tmp/aiperf-content",
            "max_tracked_records": 10000
        }));
        let gpu = raw(serde_json::json!({
            "collection_interval_ns": 333_000_000,
            "request_timeout_ns": 10_000_000_000_i64,
            "records_path": "gpu.jsonl",
            "sources": [{"type": "dcgm", "url": "http://gpu:9400/metrics"}]
        }));
        let network = raw(serde_json::json!({"mean_rtt_ns": 2_500_000.0}));
        let server = raw(serde_json::json!({
            "collection_interval_ns": 333_000_000,
            "reachability_timeout_ns": 10_000_000_000_i64,
            "urls": ["http://server:8000/metrics"],
            "formats": ["jsonl", "parquet"],
            "jsonl_path": "server.jsonl",
            "parquet_wire_path": ".wire.jsonl"
        }));
        let live = raw(serde_json::json!({
            "python_executable": "/usr/bin/python3",
            "worker_module": "aiperf.post_processors.native_streaming_worker",
            "buffer_capacity": 100,
            "otel": {
                "metrics_url": "http://otel:4318/v1/metrics",
                "stream_metrics_enabled": true,
                "stream_timing_enabled": true
            },
            "mlflow": {"tracking_uri": null, "experiment": "aiperf"}
        }));
        let inputs = [
            AuthoredSidecarInput {
                id: CONTENT_SERVER_SIDECAR_ID,
                config: &content,
            },
            AuthoredSidecarInput {
                id: SERVER_METRICS_SIDECAR_ID,
                config: &server,
            },
            AuthoredSidecarInput {
                id: GPU_TELEMETRY_SIDECAR_ID,
                config: &gpu,
            },
            AuthoredSidecarInput {
                id: LIVE_STREAMING_SIDECAR_ID,
                config: &live,
            },
            AuthoredSidecarInput {
                id: NETWORK_LATENCY_SIDECAR_ID,
                config: &network,
            },
        ];

        let mut prepared = BuiltinRunnerSidecarInputAdapterResolver::new()
            .prepare(&inputs)
            .unwrap();

        assert_eq!(
            prepared.ids().collect::<Vec<_>>(),
            vec![
                CONTENT_SERVER_SIDECAR_ID,
                GPU_TELEMETRY_SIDECAR_ID,
                LIVE_STREAMING_SIDECAR_ID,
                NETWORK_LATENCY_SIDECAR_ID,
                SERVER_METRICS_SIDECAR_ID,
            ]
        );
        assert!(
            prepared
                .get::<ContentServerSpec>(CONTENT_SERVER_SIDECAR_ID)
                .unwrap()
                .is_some()
        );
        assert!(
            prepared
                .get::<GpuTelemetrySpec>(GPU_TELEMETRY_SIDECAR_ID)
                .unwrap()
                .is_some()
        );
        assert_eq!(
            prepared
                .take::<NetworkLatencySpec>(NETWORK_LATENCY_SIDECAR_ID)
                .unwrap()
                .unwrap()
                .mean_rtt_ns,
            Some(2_500_000.0)
        );
    }

    #[test]
    fn content_server_validation_is_strict_and_side_effect_free() {
        let root = tempfile::tempdir().unwrap();
        let missing = root.path().join("not-created");
        let valid = raw(serde_json::json!({
            "host": "127.0.0.1",
            "port": 8090,
            "content_dir": missing,
            "max_tracked_records": 100
        }));
        let prepared = BuiltinRunnerSidecarInputAdapterResolver::new()
            .prepare(&[AuthoredSidecarInput {
                id: CONTENT_SERVER_SIDECAR_ID,
                config: &valid,
            }])
            .unwrap();
        let spec = prepared
            .get::<ContentServerSpec>(CONTENT_SERVER_SIDECAR_ID)
            .unwrap()
            .unwrap();
        assert_eq!(spec.base_url(), "http://127.0.0.1:8090");
        assert!(!missing.exists());

        let default_http_port = raw(serde_json::json!({
            "host": "127.0.0.1",
            "port": 80,
            "max_tracked_records": 100
        }));
        BuiltinRunnerSidecarInputAdapterResolver::new()
            .prepare(&[AuthoredSidecarInput {
                id: CONTENT_SERVER_SIDECAR_ID,
                config: &default_http_port,
            }])
            .unwrap();

        let bare_ipv6 = raw(serde_json::json!({
            "host": "::1",
            "port": 8090,
            "max_tracked_records": 100
        }));
        let prepared = BuiltinRunnerSidecarInputAdapterResolver::new()
            .prepare(&[AuthoredSidecarInput {
                id: CONTENT_SERVER_SIDECAR_ID,
                config: &bare_ipv6,
            }])
            .unwrap();
        assert_eq!(
            prepared
                .get::<ContentServerSpec>(CONTENT_SERVER_SIDECAR_ID)
                .unwrap()
                .unwrap()
                .base_url(),
            "http://[::1]:8090"
        );

        for (field, value) in [
            ("port", serde_json::json!(0)),
            ("max_tracked_records", serde_json::json!(99)),
        ] {
            let mut config = serde_json::json!({
                "host": "127.0.0.1",
                "port": 8090,
                "max_tracked_records": 100
            });
            config[field] = value;
            let raw_config = raw(config);
            assert!(
                BuiltinRunnerSidecarInputAdapterResolver::new()
                    .prepare(&[AuthoredSidecarInput {
                        id: CONTENT_SERVER_SIDECAR_ID,
                        config: &raw_config,
                    }])
                    .is_err()
            );
        }

        for host in ["user@127.0.0.1", "127.0.0.1/content", " 127.0.0.1", "[::1]"] {
            let raw_config = raw(serde_json::json!({
                "host": host,
                "port": 8090,
                "max_tracked_records": 100
            }));
            assert!(
                BuiltinRunnerSidecarInputAdapterResolver::new()
                    .prepare(&[AuthoredSidecarInput {
                        id: CONTENT_SERVER_SIDECAR_ID,
                        config: &raw_config,
                    }])
                    .is_err()
            );
        }
    }

    #[test]
    fn selected_adapter_owns_the_only_strict_full_decode() {
        let gpu = raw(serde_json::json!({
            "collection_interval_ns": 1,
            "request_timeout_ns": 1,
            "records_path": "gpu.jsonl",
            "sources": [{"type": "dcgm", "url": "http://gpu/metrics"}],
            "silently_ignored": true
        }));
        let error = BuiltinRunnerSidecarInputAdapterResolver::new()
            .prepare(&[AuthoredSidecarInput {
                id: GPU_TELEMETRY_SIDECAR_ID,
                config: &gpu,
            }])
            .unwrap_err();

        assert!(format!("{error:#}").contains("unknown field"));
    }

    #[test]
    fn unknown_and_duplicate_adapter_ids_fail_closed() {
        let raw_config = raw(serde_json::json!({}));
        let resolver = BuiltinRunnerSidecarInputAdapterResolver::new();
        let unknown = resolver
            .prepare(&[AuthoredSidecarInput {
                id: "future_sidecar",
                config: &raw_config,
            }])
            .unwrap_err();
        assert!(unknown.to_string().contains("no sidecar-input adapter"));

        let network = raw(serde_json::json!({"mean_rtt_ns": 1.0}));
        let duplicate = resolver
            .prepare(&[
                AuthoredSidecarInput {
                    id: NETWORK_LATENCY_SIDECAR_ID,
                    config: &network,
                },
                AuthoredSidecarInput {
                    id: NETWORK_LATENCY_SIDECAR_ID,
                    config: &network,
                },
            ])
            .unwrap_err();
        assert!(duplicate.to_string().contains("duplicate"));
    }
}
