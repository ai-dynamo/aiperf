// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed endpoint configuration.
//!
//! Endpoint dialects are registry-validated and extensible. Optional fields are
//! omitted when absent.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// An extensible endpoint dialect id.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EndpointType(pub String);

/// Connection-reuse policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionReuse {
    /// Shared connection pool.
    #[serde(rename = "pooled")]
    Pooled,
    /// New connection per request.
    #[serde(rename = "never")]
    Never,
    /// Per-user sticky sessions.
    #[serde(rename = "sticky-user-sessions")]
    StickyUserSessions,
}

fn default_timeout_seconds() -> f64 {
    21_600.0
}

fn default_connection_reuse() -> ConnectionReuse {
    ConnectionReuse::Pooled
}

fn default_true() -> bool {
    true
}

/// Must equal `resolve::DEFAULT_CONNECTION_LIMIT` and the protocol-v2 endpoint
/// DTO's `default_connection_limit`, which in turn track
/// `ClientConfig::default().max_connections_per_origin`. The typed model is
/// re-serialized into the protocol-v2 endpoint profile, so a divergent default
/// here silently rewrites an omitted field into non-default client policy — and
/// gRPC rejects any profile whose client policy differs from the HTTP default.
fn default_connection_limit() -> u32 {
    2_500
}

/// Must equal `resolve::DEFAULT_KEEPALIVE_TIMEOUT` and the protocol-v2 endpoint
/// DTO's `default_keepalive_timeout` (300 s =
/// `ClientConfig::default().keepalive_ns`). See [`default_connection_limit`].
fn default_keepalive_timeout() -> f64 {
    300.0
}

fn default_wait_interval() -> f64 {
    5.0
}

fn default_wait_mode() -> WaitForModelMode {
    WaitForModelMode::Inference
}

/// Readiness-probe mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WaitForModelMode {
    /// Probe the `/models` listing only.
    Models,
    /// Probe with a real inference request.
    Inference,
    /// Probe both.
    Both,
}

/// Request body content type in wire spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestContentType {
    /// `application/json`.
    ApplicationJson,
    /// `multipart/form-data`.
    MultipartFormData,
}

/// Endpoint-local reset-KV-cache hook policy.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ResetKvCacheConfig {
    /// Optional request timeout, in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<f64>,
    /// Optional origin-relative request path override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// Endpoint-local server-profiler hook policy.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ServerProfilerConfig {
    /// Optional request timeout, in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<f64>,
    /// Optional origin-relative request path override for profiler start.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_path: Option<String>,
    /// Optional origin-relative request path override for profiler stop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_path: Option<String>,
}

/// The typed `endpoint` section.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Endpoint {
    /// Target base URLs (normalized to include a scheme, e.g. `http://host:port`).
    pub urls: Vec<String>,
    /// Endpoint dialect id.
    #[serde(rename = "type")]
    pub endpoint_type: EndpointType,
    /// Whether to request server-sent-events streaming.
    #[serde(default)]
    pub streaming: bool,
    /// Emit `max_tokens` instead of `max_completion_tokens`.
    #[serde(
        default,
        rename = "use_legacy_max_tokens",
        alias = "useLegacyMaxTokens"
    )]
    pub use_legacy_max_tokens: bool,
    /// Trust the server's reported token counts over local tokenization.
    #[serde(default)]
    pub use_server_token_count: bool,
    /// Per-request timeout, in seconds.
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: f64,
    /// Connection-reuse policy.
    #[serde(default = "default_connection_reuse")]
    pub connection_reuse: ConnectionReuse,
    /// Verify TLS certificates.
    #[serde(default = "default_true", alias = "sslVerify")]
    pub ssl_verify: bool,
    /// Maximum concurrent connections.
    #[serde(default = "default_connection_limit")]
    pub connection_limit: u32,
    /// Idle keep-alive timeout, in seconds.
    #[serde(default = "default_keepalive_timeout")]
    pub keepalive_timeout: f64,
    /// Download video content referenced by responses.
    #[serde(default)]
    pub download_video_content: bool,
    /// Vendor-specific request-body extras (open bag).
    #[serde(default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
    /// Extra request headers (name → value).
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    /// Use HTTP/2 (h2/h2c).
    #[serde(default)]
    pub http2: bool,
    /// Readiness-probe timeout in seconds.
    #[serde(default)]
    pub wait_for_model_timeout: f64,
    /// Readiness-probe poll interval, in seconds.
    #[serde(default = "default_wait_interval")]
    pub wait_for_model_interval: f64,
    /// Readiness-probe mode.
    #[serde(default = "default_wait_mode")]
    pub wait_for_model_mode: WaitForModelMode,

    /// Optional URL path override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Optional API key (bearer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Optional session-affinity header name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_header: Option<String>,
    /// Optional request content type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_content_type: Option<RequestContentType>,
    /// Optional custom request-body template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    /// Optional response field to extract (paired with `template`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_field: Option<String>,
    /// Optional reset-KV-cache hook policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reset_kv_cache: Option<ResetKvCacheConfig>,
    /// Optional server-profiler hook policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_profiler: Option<ServerProfilerConfig>,
    /// Optional forward-proxy URL for benchmark traffic (explicit opt-in).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy: Option<String>,
    /// Honor the ambient proxy environment for benchmark traffic. Ignored when
    /// `proxy` is set.
    #[serde(default)]
    pub proxy_from_env: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_reuse_wire_spellings() {
        assert_eq!(
            serde_json::to_value(ConnectionReuse::StickyUserSessions).unwrap(),
            serde_json::json!("sticky-user-sessions")
        );
        assert_eq!(
            serde_json::to_value(ConnectionReuse::Pooled).unwrap(),
            serde_json::json!("pooled")
        );
    }

    #[test]
    fn request_content_type_wire_spellings() {
        assert_eq!(
            serde_json::to_value(RequestContentType::ApplicationJson).unwrap(),
            serde_json::json!("application_json")
        );
        assert_eq!(
            serde_json::to_value(RequestContentType::MultipartFormData).unwrap(),
            serde_json::json!("multipart_form_data")
        );
    }
}
