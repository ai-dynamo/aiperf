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
    /// Readiness-probe timeout in seconds.
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
