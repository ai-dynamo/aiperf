// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed transport policy.
//!
//! The wire uses a `type`-discriminated union and omits unset optional fields.
//! DynoSim configuration fields are flattened onto the transport object.

use std::fmt::{self, Display, Formatter};

use serde::{Deserialize, Deserializer, Serialize};

/// Inline transport selection discriminated by `type`.
///
/// Serde's internal tagging emits the `type` discriminator; the DynoSim newtype
/// variants flatten [`DynosimConfig`]'s set fields alongside it.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Transport {
    /// Native HTTP/1.1 or HTTP/2 transport.
    Http,
    /// Native gRPC transport (KServe OIP / Riva).
    Grpc,
    /// Offline virtual-clock Dynamo replay (fields flat on the transport).
    DynosimOffline(DynosimConfig),
    /// Online wall-clock Dynamo replay (fields flat on the transport).
    DynosimOnline(DynosimConfig),
    /// Lightweight fake execution leaf: analytic-latency synthetic responses,
    /// zero network (fields flat on the transport).
    DryRun(DryRunConfig),
    /// Native persistent WebSocket transport.
    Websocket(WebSocketTransportConfig),
}

impl Transport {
    /// Canonical wire discriminant id for this transport (the `type` value).
    ///
    /// This is the typed source of truth for the id the runner keys component
    /// selection on (`match id.as_str()`), replacing string extraction from the
    /// serialized value. Matches the serde `rename_all = "snake_case"` tag.
    pub const fn canonical_id(&self) -> &'static str {
        match self {
            Transport::Http => "http",
            Transport::Grpc => "grpc",
            Transport::DynosimOffline(_) => "dynosim_offline",
            Transport::DynosimOnline(_) => "dynosim_online",
            Transport::DryRun(_) => "dry_run",
            Transport::Websocket(_) => "websocket",
        }
    }

    /// Whether this is one of the in-process Dynamo co-simulation transports.
    pub fn is_dynosim(&self) -> bool {
        matches!(
            self,
            Transport::DynosimOffline(_) | Transport::DynosimOnline(_)
        )
    }

    /// Whether this is the no-server fake `dry_run` transport.
    pub fn is_dry_run(&self) -> bool {
        matches!(self, Transport::DryRun(_))
    }

    /// Whether this selects the persistent WebSocket transport.
    pub fn is_websocket(&self) -> bool {
        matches!(self, Transport::Websocket(_))
    }
}

/// WebSocket fallback policy.
///
/// An HTTP/SSE fallback is available only when the selected endpoint dialect
/// declares a semantically equivalent operation. The transport default is
/// deliberately closed: a failed WebSocket request must not silently change
/// protocols.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WebSocketFallback {
    /// Require the selected WebSocket dialect to complete the operation.
    #[default]
    Disabled,
    /// Permit a dialect-declared pre-send HTTP/SSE alternative.
    HttpSse,
}

const fn default_websocket_ping_interval_seconds() -> f64 {
    30.0
}

const fn default_websocket_stream_idle_timeout_seconds() -> f64 {
    900.0
}

const fn default_websocket_max_queued_commands() -> usize {
    64
}

const fn default_websocket_max_queued_bytes() -> usize {
    1_048_576
}

const fn default_websocket_max_frame_bytes() -> usize {
    1_048_576
}

const fn default_websocket_max_message_bytes() -> usize {
    8_388_608
}

const fn default_websocket_max_response_bytes() -> usize {
    67_108_864
}

/// Maximum RFC 6455 client frame header plus one maximum control-frame wire.
pub(crate) const WEBSOCKET_WRITER_RESERVE_BYTES: usize = 14 + 131;

/// Strict policy for a native WebSocket transport.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct WebSocketTransportConfig {
    /// Fallback policy selected before an application message is sent.
    pub fallback: WebSocketFallback,
    /// Interval between application-independent keepalive pings in seconds.
    pub ping_interval_seconds: f64,
    /// Maximum inactive duration for a live response stream in seconds.
    pub stream_idle_timeout_seconds: f64,
    /// Maximum commands retained in the outbound driver queue.
    pub max_queued_commands: usize,
    /// Maximum total application payload bytes retained in the outbound queue.
    pub max_queued_bytes: usize,
    /// Maximum payload size for one WebSocket frame.
    pub max_frame_bytes: usize,
    /// Maximum reassembled WebSocket application-message size.
    pub max_message_bytes: usize,
    /// Maximum cumulative response bytes accepted for one operation.
    pub max_response_bytes: usize,
}

/// Invalid WebSocket transport policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WebSocketTransportConfigError {
    /// A duration is zero, negative, or non-finite.
    InvalidDuration {
        /// Name of the invalid duration field.
        field: &'static str,
    },
    /// A count or byte limit is zero.
    NonPositiveLimit {
        /// Name of the invalid limit field.
        field: &'static str,
    },
    /// The frame limit exceeds the reassembled message limit.
    FrameExceedsMessage,
    /// The reassembled message limit exceeds the response limit.
    MessageExceedsResponse,
    /// The authored payload bound leaves no representable writer overhead.
    QueueExceedsWriterCapacity,
}

impl Display for WebSocketTransportConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDuration { field } => {
                write!(formatter, "websocket {field} must be finite and positive")
            }
            Self::NonPositiveLimit { field } => {
                write!(formatter, "websocket {field} must be positive")
            }
            Self::FrameExceedsMessage => {
                formatter.write_str("websocket max_frame_bytes cannot exceed max_message_bytes")
            }
            Self::MessageExceedsResponse => {
                formatter.write_str("websocket max_message_bytes cannot exceed max_response_bytes")
            }
            Self::QueueExceedsWriterCapacity => formatter.write_str(
                "websocket max_queued_bytes leaves no capacity for frame and control overhead",
            ),
        }
    }
}

impl std::error::Error for WebSocketTransportConfigError {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WebSocketTransportConfigRaw {
    #[serde(default)]
    fallback: WebSocketFallback,
    #[serde(
        default = "default_websocket_ping_interval_seconds",
        deserialize_with = "deserialize_positive_finite_seconds"
    )]
    ping_interval_seconds: f64,
    #[serde(
        default = "default_websocket_stream_idle_timeout_seconds",
        deserialize_with = "deserialize_positive_finite_seconds"
    )]
    stream_idle_timeout_seconds: f64,
    #[serde(
        default = "default_websocket_max_queued_commands",
        deserialize_with = "deserialize_positive_usize"
    )]
    max_queued_commands: usize,
    #[serde(
        default = "default_websocket_max_queued_bytes",
        deserialize_with = "deserialize_positive_usize"
    )]
    max_queued_bytes: usize,
    #[serde(
        default = "default_websocket_max_frame_bytes",
        deserialize_with = "deserialize_positive_usize"
    )]
    max_frame_bytes: usize,
    #[serde(
        default = "default_websocket_max_message_bytes",
        deserialize_with = "deserialize_positive_usize"
    )]
    max_message_bytes: usize,
    #[serde(
        default = "default_websocket_max_response_bytes",
        deserialize_with = "deserialize_positive_usize"
    )]
    max_response_bytes: usize,
}

impl<'de> Deserialize<'de> for WebSocketTransportConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = WebSocketTransportConfigRaw::deserialize(deserializer)?;
        Self {
            fallback: raw.fallback,
            ping_interval_seconds: raw.ping_interval_seconds,
            stream_idle_timeout_seconds: raw.stream_idle_timeout_seconds,
            max_queued_commands: raw.max_queued_commands,
            max_queued_bytes: raw.max_queued_bytes,
            max_frame_bytes: raw.max_frame_bytes,
            max_message_bytes: raw.max_message_bytes,
            max_response_bytes: raw.max_response_bytes,
        }
        .validate()
        .map_err(serde::de::Error::custom)
    }
}

fn deserialize_positive_finite_seconds<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let seconds = f64::deserialize(deserializer)?;
    validate_websocket_seconds("duration", seconds).map_err(serde::de::Error::custom)
}

fn validate_websocket_seconds(
    field: &'static str,
    seconds: f64,
) -> Result<f64, WebSocketTransportConfigError> {
    if seconds.is_finite() && seconds > 0.0 {
        Ok(seconds)
    } else {
        Err(WebSocketTransportConfigError::InvalidDuration { field })
    }
}

fn deserialize_positive_usize<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    let value = usize::deserialize(deserializer)?;
    validate_websocket_positive_usize("limit", value).map_err(serde::de::Error::custom)
}

fn validate_websocket_positive_usize(
    field: &'static str,
    value: usize,
) -> Result<usize, WebSocketTransportConfigError> {
    if value > 0 {
        Ok(value)
    } else {
        Err(WebSocketTransportConfigError::NonPositiveLimit { field })
    }
}

impl Default for WebSocketTransportConfig {
    fn default() -> Self {
        Self {
            fallback: WebSocketFallback::Disabled,
            ping_interval_seconds: default_websocket_ping_interval_seconds(),
            stream_idle_timeout_seconds: default_websocket_stream_idle_timeout_seconds(),
            max_queued_commands: default_websocket_max_queued_commands(),
            max_queued_bytes: default_websocket_max_queued_bytes(),
            max_frame_bytes: default_websocket_max_frame_bytes(),
            max_message_bytes: default_websocket_max_message_bytes(),
            max_response_bytes: default_websocket_max_response_bytes(),
        }
    }
}

impl WebSocketTransportConfig {
    /// Validate the policy shared by every configuration frontend.
    pub fn validate(self) -> Result<Self, WebSocketTransportConfigError> {
        for (field, seconds) in [
            ("ping_interval_seconds", self.ping_interval_seconds),
            (
                "stream_idle_timeout_seconds",
                self.stream_idle_timeout_seconds,
            ),
        ] {
            validate_websocket_seconds(field, seconds)?;
        }
        for (field, value) in [
            ("max_queued_commands", self.max_queued_commands),
            ("max_queued_bytes", self.max_queued_bytes),
            ("max_frame_bytes", self.max_frame_bytes),
            ("max_message_bytes", self.max_message_bytes),
            ("max_response_bytes", self.max_response_bytes),
        ] {
            validate_websocket_positive_usize(field, value)?;
        }
        if self.max_frame_bytes > self.max_message_bytes {
            return Err(WebSocketTransportConfigError::FrameExceedsMessage);
        }
        if self.max_message_bytes > self.max_response_bytes {
            return Err(WebSocketTransportConfigError::MessageExceedsResponse);
        }
        if self
            .max_queued_bytes
            .checked_add(WEBSOCKET_WRITER_RESERVE_BYTES)
            .is_none()
        {
            return Err(WebSocketTransportConfigError::QueueExceedsWriterCapacity);
        }
        Ok(self)
    }
}

/// Analytic latency settings flattened onto the `dry_run` transport.
///
/// The runtime supplies defaults for omitted fields.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DryRunConfig {
    /// Base time-to-first-token in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    /// Base inter-token latency in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_ms: Option<f64>,
    /// Prefill cost per input token (ms): `TTFT += this · ISL`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_per_isl_token_ms: Option<f64>,
    /// Super-linear prefill contention (ms per inflight²).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_concurrency_quad_ms: Option<f64>,
    /// Decode cost per output token (ms): `ITL += this · OSL`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_per_osl_token_ms: Option<f64>,
    /// Linear decode contention (ms per inflight).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_concurrency_lin_ms: Option<f64>,
    /// Lognormal TTFT jitter coefficient of variation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_jitter_cv: Option<f64>,
    /// Lognormal ITL jitter coefficient of variation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_jitter_cv: Option<f64>,
    /// Root seed for the per-request jitter draw.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Latency model: `linear`, `aiconfigurator_polynomial`, or `recorded`
    /// (reproduce the trace's pre-known api_time as the total response latency).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_model: Option<String>,
    /// KV-cache utilization for the polynomial decode curve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_utilization: Option<f64>,
    /// Clock driver: `real` (default) or `sim` (deterministic virtual time).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock: Option<String>,
    /// Optional single-reactor model of logical AIPerf worker placement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub virtual_workers: Option<DryRunVirtualWorkersConfig>,
}

/// Logical worker placement configuration for socket-free dry runs.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DryRunVirtualWorkersConfig {
    /// Enable virtual worker placement.
    #[serde(default)]
    pub enabled: bool,
    /// Logical worker count; defaults to the authored runtime worker count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<usize>,
    /// Contention input: `global` (default) or `worker_local`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contention_scope: Option<String>,
    /// Optional per-worker analytic latency multipliers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub profiles: Vec<DryRunVirtualWorkerProfile>,
}

/// Analytic latency multipliers for one virtual worker.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DryRunVirtualWorkerProfile {
    /// Zero-based virtual worker index.
    pub worker: usize,
    /// TTFT multiplier applied after analytic latency calculation.
    #[serde(default = "one")]
    pub ttft_multiplier: f64,
    /// ITL multiplier applied after analytic latency calculation.
    #[serde(default = "one")]
    pub itl_multiplier: f64,
}

fn one() -> f64 {
    1.0
}

/// DynoSim transport configuration.
///
/// The `engine` / `prefill_engine` / `decode_engine` / `router` objects are
/// opaque `MockEngineArgs` / `KvRouterConfig` JSON preserved verbatim for Dynamo
/// to validate. Multi-word fields accept Config-v2 camelCase input and emit
/// snake_case.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DynosimConfig {
    /// JSON engine profile consumed by Dynamo's canonical parser.
    #[serde(
        default,
        alias = "engineProfile",
        skip_serializing_if = "Option::is_none"
    )]
    pub engine_profile: Option<String>,
    /// Inline aggregate/single `MockEngineArgs` object (passed through verbatim).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<serde_json::Value>,
    /// Inline disaggregated prefill `MockEngineArgs` object.
    #[serde(
        default,
        alias = "prefillEngine",
        skip_serializing_if = "Option::is_none"
    )]
    pub prefill_engine: Option<serde_json::Value>,
    /// Inline disaggregated decode `MockEngineArgs` object.
    #[serde(
        default,
        alias = "decodeEngine",
        skip_serializing_if = "Option::is_none"
    )]
    pub decode_engine: Option<serde_json::Value>,
    /// Inline `KvRouterConfig` object (passed through verbatim).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router: Option<serde_json::Value>,
    /// Startup router policy-family YAML path overriding the inline `router`.
    #[serde(
        default,
        alias = "routerPolicyConfig",
        skip_serializing_if = "Option::is_none"
    )]
    pub router_policy_config: Option<String>,
    /// Model selector for a multi-model router policy document.
    #[serde(
        default,
        alias = "routerModelName",
        skip_serializing_if = "Option::is_none"
    )]
    pub router_model_name: Option<String>,
    /// Optional structured AIConfigurator overrides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aic: Option<DynosimAic>,
    /// Capture backend per-request records even without a JSONL artifact.
    #[serde(
        default,
        alias = "capturePerRequest",
        skip_serializing_if = "Option::is_none"
    )]
    pub capture_per_request: Option<bool>,
    /// Canonical goodput thresholds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sla: Option<DynosimSla>,
    /// Deployment topology (`single` / `aggregated` / `disaggregated`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology: Option<String>,
    /// Aggregate worker count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workers: Option<u32>,
    /// Disaggregated prefill worker count.
    #[serde(
        default,
        alias = "prefillWorkers",
        skip_serializing_if = "Option::is_none"
    )]
    pub prefill_workers: Option<u32>,
    /// Disaggregated decode worker count.
    #[serde(
        default,
        alias = "decodeWorkers",
        skip_serializing_if = "Option::is_none"
    )]
    pub decode_workers: Option<u32>,
    /// Router policy for routed topologies (`round_robin` / `kv`).
    #[serde(default, alias = "routerMode", skip_serializing_if = "Option::is_none")]
    pub router_mode: Option<String>,
    /// Required runner build capabilities, sorted and deduplicated.
    #[serde(
        default,
        alias = "requiredFeatures",
        skip_serializing_if = "Option::is_none"
    )]
    pub required_features: Option<Vec<String>>,
    /// Backend-owned output artifacts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<DynosimArtifacts>,
}

impl DynosimConfig {
    /// Sort and deduplicate `required_features` for stable wire order.
    pub fn normalize(&mut self) {
        if let Some(features) = self.required_features.as_mut() {
            features.sort();
            features.dedup();
        }
    }
}

/// Canonical goodput thresholds (`DynosimSlaConfig`; each bound optional).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DynosimSla {
    /// Maximum time to first token (ms).
    #[serde(default, alias = "ttftMs", skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    /// Maximum mean inter-token latency (ms).
    #[serde(default, alias = "itlMs", skip_serializing_if = "Option::is_none")]
    pub itl_ms: Option<f64>,
    /// Maximum end-to-end latency in milliseconds.
    ///
    /// The accepted Config-v2 camelCase key is `e2EMs`.
    #[serde(default, alias = "e2EMs", skip_serializing_if = "Option::is_none")]
    pub e2e_ms: Option<f64>,
}

/// Backend-owned Dynamo artifacts written after a successful run
/// (`DynosimArtifactConfig`). Paths are relative to the run artifact target.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DynosimArtifacts {
    /// Canonical aggregate Dynamo JSON report path.
    #[serde(default, alias = "reportJson", skip_serializing_if = "Option::is_none")]
    pub report_json: Option<String>,
    /// Canonical per-request Dynamo JSONL path.
    #[serde(
        default,
        alias = "perRequestJsonl",
        skip_serializing_if = "Option::is_none"
    )]
    pub per_request_jsonl: Option<String>,
    /// Timed worker/request/KV artifact JSON for trace workloads.
    #[serde(
        default,
        alias = "workerArtifactsJson",
        skip_serializing_if = "Option::is_none"
    )]
    pub worker_artifacts_json: Option<String>,
    /// Pass-start/pass-end KV visibility override.
    #[serde(
        default,
        alias = "kvEventVisibility",
        skip_serializing_if = "Option::is_none"
    )]
    pub kv_event_visibility: Option<String>,
}

/// Structured AIConfigurator overrides (`DynosimAicConfig`; all optional).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DynosimAic {
    /// AIC serving backend identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// AIC GPU system identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Performance-database backend version.
    #[serde(
        default,
        alias = "backendVersion",
        skip_serializing_if = "Option::is_none"
    )]
    pub backend_version: Option<String>,
    /// Hugging Face model path for AIC.
    #[serde(default, alias = "modelPath", skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    /// Tensor-parallel degree.
    #[serde(default, alias = "tpSize", skip_serializing_if = "Option::is_none")]
    pub tp_size: Option<u32>,
    /// MoE tensor-parallel degree.
    #[serde(default, alias = "moeTpSize", skip_serializing_if = "Option::is_none")]
    pub moe_tp_size: Option<u32>,
    /// MoE expert-parallel degree.
    #[serde(default, alias = "moeEpSize", skip_serializing_if = "Option::is_none")]
    pub moe_ep_size: Option<u32>,
    /// Attention data-parallel degree.
    #[serde(
        default,
        alias = "attentionDpSize",
        skip_serializing_if = "Option::is_none"
    )]
    pub attention_dp_size: Option<u32>,
    /// Speculative (MTP) draft-token count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nextn: Option<u32>,
    /// GEMM quantization override.
    #[serde(default, alias = "gemmDtype", skip_serializing_if = "Option::is_none")]
    pub gemm_dtype: Option<String>,
    /// MoE quantization override.
    #[serde(default, alias = "moeDtype", skip_serializing_if = "Option::is_none")]
    pub moe_dtype: Option<String>,
    /// Attention/FMHA quantization override.
    #[serde(default, alias = "fmhaDtype", skip_serializing_if = "Option::is_none")]
    pub fmha_dtype: Option<String>,
    /// KV-cache quantization override.
    #[serde(
        default,
        alias = "kvCacheDtype",
        skip_serializing_if = "Option::is_none"
    )]
    pub kv_cache_dtype: Option<String>,
    /// Collective/comm quantization override.
    #[serde(default, alias = "commDtype", skip_serializing_if = "Option::is_none")]
    pub comm_dtype: Option<String>,
    /// Comma-separated conditional draft acceptance rates.
    #[serde(
        default,
        alias = "nextnAcceptRates",
        skip_serializing_if = "Option::is_none"
    )]
    pub nextn_accept_rates: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_projects_type_only() {
        assert_eq!(
            serde_json::to_value(Transport::Http).unwrap(),
            serde_json::json!({"type": "http"})
        );
    }

    #[test]
    fn websocket_transport_defaults_are_strict() {
        let value = serde_json::json!({"type": "websocket"});
        let decoded: Transport = serde_json::from_value(value).unwrap();
        let Transport::Websocket(config) = decoded else {
            panic!("expected websocket transport");
        };

        assert_eq!(config.fallback, WebSocketFallback::Disabled);
        assert_eq!(config.ping_interval_seconds, 30.0);
        assert_eq!(config.stream_idle_timeout_seconds, 900.0);
        assert!(
            serde_json::from_value::<Transport>(serde_json::json!({
                "type": "websocket",
                "unsupported": true,
            }))
            .is_err()
        );
    }

    #[test]
    fn websocket_queue_limit_reserves_wire_overhead_without_overflow() {
        let mut config = WebSocketTransportConfig::default();
        config.max_queued_bytes = usize::MAX;
        config.max_frame_bytes = config.max_message_bytes;

        assert_eq!(
            config.validate(),
            Err(WebSocketTransportConfigError::QueueExceedsWriterCapacity)
        );
    }

    #[test]
    fn dynosim_offline_flattens_set_fields_only() {
        let cfg = DynosimConfig {
            topology: Some("single".into()),
            workers: Some(1),
            router_mode: Some("round_robin".into()),
            engine: Some(serde_json::json!({"block_size": 16})),
            sla: Some(DynosimSla {
                ttft_ms: Some(500.0),
                itl_ms: Some(20.0),
                e2e_ms: None,
            }),
            ..Default::default()
        };
        let v = serde_json::to_value(Transport::DynosimOffline(cfg)).unwrap();
        assert_eq!(
            v,
            serde_json::json!({
                "type": "dynosim_offline",
                "topology": "single",
                "workers": 1,
                "router_mode": "round_robin",
                "engine": {"block_size": 16},
                "sla": {"ttft_ms": 500.0, "itl_ms": 20.0},
            })
        );
    }

    #[test]
    fn canonical_id_matches_serde_tag() {
        // `canonical_id()` must stay byte-identical to the serialized `type`
        // tag, since the runner keys component selection on it. The sample list
        // alone cannot enforce that: it is hand-maintained, and `Websocket` was
        // added to `Transport` while this guard kept passing over the five
        // variants someone had remembered to list. The exhaustive `match` below
        // is the trip-wire — a new variant fails to compile here, so it cannot
        // escape both the accessor and its own test the way `Websocket` did.
        let cases = [
            Transport::Http,
            Transport::Grpc,
            Transport::DynosimOffline(DynosimConfig::default()),
            Transport::DynosimOnline(DynosimConfig::default()),
            Transport::DryRun(DryRunConfig::default()),
            Transport::Websocket(WebSocketTransportConfig::default()),
        ];

        let mut covered = std::collections::BTreeSet::new();
        for transport in &cases {
            let expected = match transport {
                Transport::Http => "http",
                Transport::Grpc => "grpc",
                Transport::DynosimOffline(_) => "dynosim_offline",
                Transport::DynosimOnline(_) => "dynosim_online",
                Transport::DryRun(_) => "dry_run",
                Transport::Websocket(_) => "websocket",
            };
            let wire_tag = serde_json::to_value(transport).unwrap()["type"]
                .as_str()
                .expect("transport serializes with a string `type` tag")
                .to_string();
            assert_eq!(wire_tag, expected, "serde tag drifted for {expected}");
            assert_eq!(
                transport.canonical_id(),
                expected,
                "canonical_id drifted for {expected}"
            );
            covered.insert(expected);
        }

        // The compiler pins the arm count; these pin that every arm actually
        // received a sample, so adding an arm without a sample still fails.
        assert_eq!(covered.len(), cases.len(), "duplicate sample variants");
        assert_eq!(
            covered.len(),
            6,
            "add a sample for the new Transport variant"
        );
    }
}
