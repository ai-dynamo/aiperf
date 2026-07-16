// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed `endpoint` section of the native `BenchmarkConfig`.
//!
//! Wire shape ported from `src/aiperf/orchestrator/rust_wire.py::_authored_endpoint`
//! (with `include_readiness=True`, as `dump_benchmark_run` uses). Input keys /
//! defaults come from `src/aiperf/config/endpoint.py`. Serializing [`Endpoint`]
//! yields the exact `cfg.endpoint` subtree the runner consumes.
//!
//! Typing: `endpoint.type` is an open, registry-validated dialect id in Python
//! (annotated `str`, not a closed enum), so it is a transparent newtype rather
//! than an enum; `connection_reuse` / `wait_for_model_mode` /
//! `request_content_type` ARE closed sets and are real enums. Optional fields
//! use `_set_optional` semantics in Python (omitted when absent), so they are
//! `skip_serializing_if = "Option::is_none"`; the always-present fields emit
//! their value (including readiness fields).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// An OpenAI-compatible / KServe / Riva endpoint dialect id (e.g. `chat`,
/// `completions`, `embeddings`, `rankings`). Open/extensible in Python
/// (registry-validated), so a transparent newtype rather than a closed enum.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EndpointType(pub String);

/// Connection-reuse policy. Closed set (`ConnectionReuseStrategy`).
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

/// Readiness-probe mode. Closed set (Python `Literal['models','inference','both']`).
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

/// Request body content type, in wire spelling (Python maps the MIME string to
/// these tokens in `_authored_endpoint`). Closed set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestContentType {
    /// `application/json`.
    ApplicationJson,
    /// `multipart/form-data`.
    MultipartFormData,
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
    pub streaming: bool,
    /// Emit the legacy `max_tokens` field instead of `max_completion_tokens`.
    pub use_legacy_max_tokens: bool,
    /// Trust the server's reported token counts over local tokenization.
    pub use_server_token_count: bool,
    /// Per-request timeout, in seconds.
    pub timeout_seconds: f64,
    /// Connection-reuse policy.
    pub connection_reuse: ConnectionReuse,
    /// Verify TLS certificates.
    pub ssl_verify: bool,
    /// Maximum concurrent connections.
    pub connection_limit: u32,
    /// Idle keep-alive timeout, in seconds.
    pub keepalive_timeout: f64,
    /// Download video content referenced by responses.
    pub download_video_content: bool,
    /// Vendor-specific request-body extras (open bag).
    pub extra: serde_json::Map<String, serde_json::Value>,
    /// Extra request headers (name → value).
    pub headers: BTreeMap<String, String>,
    /// Use HTTP/2 (h2/h2c).
    pub http2: bool,
    /// Readiness-probe timeout, in seconds (always present via include_readiness).
    pub wait_for_model_timeout: f64,
    /// Readiness-probe poll interval, in seconds.
    pub wait_for_model_interval: f64,
    /// Readiness-probe mode.
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
