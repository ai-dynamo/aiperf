// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Endpoint configuration validation and normalization.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use url::Url;

use crate::endpoints::metadata::{EndpointDescriptor, EndpointType};
use crate::endpoints::models::{EndpointError, EndpointResult};
use crate::endpoints::registry::legacy_descriptor_for;

/// Wire request content type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestContentType {
    /// JSON request body.
    ApplicationJson,
    /// Multipart form-data request body.
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

/// Authored endpoint policy with identity selected separately by [`crate::endpoints::EndpointId`].
///
/// [`EndpointConfig`] adds the protocol-v1 closed-enum identity.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RawEndpointConfig {
    /// Base URLs.
    pub urls: Vec<String>,
    /// Optional path override.
    pub path: Option<String>,
    /// Optional reset-KV-cache hook policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reset_kv_cache: Option<ResetKvCacheConfig>,
    /// Optional server-profiler hook policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_profiler: Option<ServerProfilerConfig>,
    /// Whether streaming is requested.
    pub streaming: bool,
    /// Request content type override or derived value.
    pub request_content_type: Option<RequestContentType>,
    /// Template body for template endpoints.
    pub template: Option<String>,
    /// Optional JMESPath response selector for raw/template endpoints.
    pub response_field: Option<String>,
    /// Whole polling-lifecycle timeout in seconds.
    pub timeout_seconds: f64,
    /// Delay between async-job status polls in seconds.
    pub polling_interval_seconds: f64,
    /// Download completed video bytes after polling when true.
    pub download_video_content: bool,
    /// Wait-for-model probe timeout in seconds; zero disables probing.
    pub wait_for_model_timeout: f64,
    /// Wait-for-model probe interval.
    pub wait_for_model_interval: f64,
    /// Wait-for-model probe mode.
    pub wait_for_model_mode: String,
    /// Whether the interval was supplied explicitly.
    pub wait_for_model_interval_set: bool,
    /// Whether the mode was supplied explicitly.
    pub wait_for_model_mode_set: bool,
    /// Emit `max_tokens` instead of `max_completion_tokens` for chat.
    #[serde(rename = "use_legacy_max_tokens")]
    pub use_legacy_max_tokens: bool,
    /// Request usage in streaming frames when supported.
    pub use_server_token_count: bool,
    /// Headers merged into every request before per-turn header overrides.
    #[serde(default, skip_serializing)]
    pub headers: BTreeMap<String, String>,
    /// Endpoint API key. It is deliberately never serialized into artifacts.
    #[serde(default, skip_serializing)]
    pub api_key: Option<String>,
    /// Endpoint-level extra body fields.
    pub extra: Option<Map<String, Value>>,
}

impl Default for RawEndpointConfig {
    fn default() -> Self {
        Self {
            urls: Vec::new(),
            path: None,
            reset_kv_cache: None,
            server_profiler: None,
            streaming: false,
            request_content_type: None,
            template: None,
            response_field: None,
            timeout_seconds: 6.0 * 60.0 * 60.0,
            polling_interval_seconds: 0.1,
            download_video_content: false,
            wait_for_model_timeout: 0.0,
            wait_for_model_interval: 5.0,
            wait_for_model_mode: "inference".to_string(),
            wait_for_model_interval_set: false,
            wait_for_model_mode_set: false,
            use_legacy_max_tokens: false,
            use_server_token_count: false,
            headers: BTreeMap::new(),
            api_key: None,
            extra: None,
        }
    }
}

impl fmt::Debug for RawEndpointConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RawEndpointConfig")
            .field("urls", &self.urls)
            .field("path", &self.path)
            .field("reset_kv_cache", &self.reset_kv_cache)
            .field("server_profiler", &self.server_profiler)
            .field("streaming", &self.streaming)
            .field("request_content_type", &self.request_content_type)
            .field("template", &self.template)
            .field("response_field", &self.response_field)
            .field("timeout_seconds", &self.timeout_seconds)
            .field("polling_interval_seconds", &self.polling_interval_seconds)
            .field("download_video_content", &self.download_video_content)
            .field("wait_for_model_timeout", &self.wait_for_model_timeout)
            .field("wait_for_model_interval", &self.wait_for_model_interval)
            .field("wait_for_model_mode", &self.wait_for_model_mode)
            .field(
                "wait_for_model_interval_set",
                &self.wait_for_model_interval_set,
            )
            .field("wait_for_model_mode_set", &self.wait_for_model_mode_set)
            .field("use_legacy_max_tokens", &self.use_legacy_max_tokens)
            .field("use_server_token_count", &self.use_server_token_count)
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .field("has_api_key", &self.api_key.is_some())
            .field("extra", &self.extra)
            .finish()
    }
}

/// Endpoint configuration used by endpoint formatters and validators.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct EndpointConfig {
    /// Endpoint type.
    #[serde(rename = "type")]
    pub endpoint_type: EndpointType,
    /// Base URLs.
    pub urls: Vec<String>,
    /// Optional path override.
    pub path: Option<String>,
    /// Optional reset-KV-cache hook policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reset_kv_cache: Option<ResetKvCacheConfig>,
    /// Optional server-profiler hook policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_profiler: Option<ServerProfilerConfig>,
    /// Whether streaming is enabled after validation.
    pub streaming: bool,
    /// Request content type override or derived value.
    pub request_content_type: Option<RequestContentType>,
    /// Template body for template endpoints.
    pub template: Option<String>,
    /// Optional JMESPath response selector for raw/template endpoints.
    pub response_field: Option<String>,
    /// Whole polling-lifecycle timeout in seconds.
    pub timeout_seconds: f64,
    /// Delay between async-job status polls in seconds.
    pub polling_interval_seconds: f64,
    /// Download completed video bytes after polling when true.
    pub download_video_content: bool,
    /// Wait-for-model probe timeout in seconds; zero disables probing.
    pub wait_for_model_timeout: f64,
    /// Wait-for-model probe interval.
    pub wait_for_model_interval: f64,
    /// Wait-for-model probe mode.
    pub wait_for_model_mode: String,
    /// Whether the interval was supplied explicitly.
    pub wait_for_model_interval_set: bool,
    /// Whether the mode was supplied explicitly.
    pub wait_for_model_mode_set: bool,
    /// Emit `max_tokens` instead of `max_completion_tokens` for chat.
    #[serde(rename = "use_legacy_max_tokens")]
    pub use_legacy_max_tokens: bool,
    /// Request usage in streaming frames when supported.
    pub use_server_token_count: bool,
    /// Headers merged into every request before per-turn header overrides.
    #[serde(default, skip_serializing)]
    pub headers: BTreeMap<String, String>,
    /// Endpoint API key. It is deliberately never serialized into artifacts.
    #[serde(default, skip_serializing)]
    pub api_key: Option<String>,
    /// Endpoint-level extra body fields.
    pub extra: Option<Map<String, Value>>,
}

impl fmt::Debug for EndpointConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EndpointConfig")
            .field("endpoint_type", &self.endpoint_type)
            .field("urls", &self.urls)
            .field("path", &self.path)
            .field("reset_kv_cache", &self.reset_kv_cache)
            .field("server_profiler", &self.server_profiler)
            .field("streaming", &self.streaming)
            .field("request_content_type", &self.request_content_type)
            .field("template", &self.template)
            .field("response_field", &self.response_field)
            .field("timeout_seconds", &self.timeout_seconds)
            .field("polling_interval_seconds", &self.polling_interval_seconds)
            .field("download_video_content", &self.download_video_content)
            .field("wait_for_model_timeout", &self.wait_for_model_timeout)
            .field("wait_for_model_interval", &self.wait_for_model_interval)
            .field("wait_for_model_mode", &self.wait_for_model_mode)
            .field(
                "wait_for_model_interval_set",
                &self.wait_for_model_interval_set,
            )
            .field("wait_for_model_mode_set", &self.wait_for_model_mode_set)
            .field("use_legacy_max_tokens", &self.use_legacy_max_tokens)
            .field("use_server_token_count", &self.use_server_token_count)
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .field("has_api_key", &self.api_key.is_some())
            .field("extra", &self.extra)
            .finish()
    }
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self::from_raw(EndpointType::Chat, RawEndpointConfig::default())
    }
}

impl From<&EndpointConfig> for RawEndpointConfig {
    fn from(config: &EndpointConfig) -> Self {
        Self {
            urls: config.urls.clone(),
            path: config.path.clone(),
            reset_kv_cache: config.reset_kv_cache.clone(),
            server_profiler: config.server_profiler.clone(),
            streaming: config.streaming,
            request_content_type: config.request_content_type,
            template: config.template.clone(),
            response_field: config.response_field.clone(),
            timeout_seconds: config.timeout_seconds,
            polling_interval_seconds: config.polling_interval_seconds,
            download_video_content: config.download_video_content,
            wait_for_model_timeout: config.wait_for_model_timeout,
            wait_for_model_interval: config.wait_for_model_interval,
            wait_for_model_mode: config.wait_for_model_mode.clone(),
            wait_for_model_interval_set: config.wait_for_model_interval_set,
            wait_for_model_mode_set: config.wait_for_model_mode_set,
            use_legacy_max_tokens: config.use_legacy_max_tokens,
            use_server_token_count: config.use_server_token_count,
            headers: config.headers.clone(),
            api_key: config.api_key.clone(),
            extra: config.extra.clone(),
        }
    }
}

impl From<EndpointConfig> for RawEndpointConfig {
    fn from(config: EndpointConfig) -> Self {
        Self {
            urls: config.urls,
            path: config.path,
            reset_kv_cache: config.reset_kv_cache,
            server_profiler: config.server_profiler,
            streaming: config.streaming,
            request_content_type: config.request_content_type,
            template: config.template,
            response_field: config.response_field,
            timeout_seconds: config.timeout_seconds,
            polling_interval_seconds: config.polling_interval_seconds,
            download_video_content: config.download_video_content,
            wait_for_model_timeout: config.wait_for_model_timeout,
            wait_for_model_interval: config.wait_for_model_interval,
            wait_for_model_mode: config.wait_for_model_mode,
            wait_for_model_interval_set: config.wait_for_model_interval_set,
            wait_for_model_mode_set: config.wait_for_model_mode_set,
            use_legacy_max_tokens: config.use_legacy_max_tokens,
            use_server_token_count: config.use_server_token_count,
            headers: config.headers,
            api_key: config.api_key,
            extra: config.extra,
        }
    }
}

impl EndpointConfig {
    /// Construct the protocol-v1 compatibility shape from identity-free policy.
    pub fn from_raw(endpoint_type: EndpointType, raw: RawEndpointConfig) -> Self {
        Self {
            endpoint_type,
            urls: raw.urls,
            path: raw.path,
            reset_kv_cache: raw.reset_kv_cache,
            server_profiler: raw.server_profiler,
            streaming: raw.streaming,
            request_content_type: raw.request_content_type,
            template: raw.template,
            response_field: raw.response_field,
            timeout_seconds: raw.timeout_seconds,
            polling_interval_seconds: raw.polling_interval_seconds,
            download_video_content: raw.download_video_content,
            wait_for_model_timeout: raw.wait_for_model_timeout,
            wait_for_model_interval: raw.wait_for_model_interval,
            wait_for_model_mode: raw.wait_for_model_mode,
            wait_for_model_interval_set: raw.wait_for_model_interval_set,
            wait_for_model_mode_set: raw.wait_for_model_mode_set,
            use_legacy_max_tokens: raw.use_legacy_max_tokens,
            use_server_token_count: raw.use_server_token_count,
            headers: raw.headers,
            api_key: raw.api_key,
            extra: raw.extra,
        }
    }

    /// Validate and normalize config fields.
    pub fn validate(mut self) -> EndpointResult<Self> {
        if self.template.is_some() && self.endpoint_type == EndpointType::Chat {
            self.endpoint_type = EndpointType::Template;
        }
        let endpoint_type = self.endpoint_type;
        let descriptor = legacy_descriptor_for(endpoint_type);
        let raw = RawEndpointConfig::from(self).validate_against(
            descriptor.supports_streaming,
            descriptor.requires_form_data,
            endpoint_type.canonical_id(),
            endpoint_type == EndpointType::Template,
        )?;
        Ok(Self::from_raw(endpoint_type, raw))
    }
}

impl RawEndpointConfig {
    pub(crate) fn validate_for_descriptor(
        self,
        descriptor: &EndpointDescriptor,
    ) -> EndpointResult<Self> {
        self.validate_against(
            descriptor.supports_streaming,
            descriptor.requires_form_data,
            descriptor.id,
            false,
        )
    }

    fn validate_against(
        mut self,
        supports_streaming: bool,
        requires_form_data: bool,
        endpoint_id: &str,
        require_template: bool,
    ) -> EndpointResult<Self> {
        if self.streaming && !supports_streaming {
            self.streaming = false;
        }
        for url in &self.urls {
            validate_url(url)?;
        }
        if let Some(path) = &self.path
            && !path.starts_with('/')
        {
            return Err(EndpointError::InvalidConfig(
                "endpoint.path must start with a leading slash".to_string(),
            ));
        }
        if let Some(config) = &self.reset_kv_cache {
            validate_control_hook_timeout(
                config.timeout_seconds,
                "endpoint.reset_kv_cache.timeout_seconds",
            )?;
            if let Some(path) = &config.path {
                validate_origin_relative_path(path, "endpoint.reset_kv_cache.path")?;
            }
        }
        if let Some(config) = &self.server_profiler {
            validate_control_hook_timeout(
                config.timeout_seconds,
                "endpoint.server_profiler.timeout_seconds",
            )?;
            if let Some(path) = &config.start_path {
                validate_origin_relative_path(path, "endpoint.server_profiler.start_path")?;
            }
            if let Some(path) = &config.stop_path {
                validate_origin_relative_path(path, "endpoint.server_profiler.stop_path")?;
            }
        }
        let legacy_template = self
            .extra
            .as_ref()
            .and_then(|extra| extra.get("payload_template"))
            .and_then(Value::as_str);
        if require_template && self.template.is_none() && legacy_template.is_none() {
            return Err(EndpointError::InvalidConfig(
                "template or extra.payload_template is required when endpoint type is 'template'"
                    .to_string(),
            ));
        }
        if !self.timeout_seconds.is_finite() || self.timeout_seconds < 0.0 {
            return Err(EndpointError::InvalidConfig(
                "timeout_seconds must be finite and non-negative".to_string(),
            ));
        }
        if !self.polling_interval_seconds.is_finite()
            || !(0.001..=10.0).contains(&self.polling_interval_seconds)
        {
            return Err(EndpointError::InvalidConfig(
                "polling_interval_seconds must be finite and between 0.001 and 10 seconds"
                    .to_string(),
            ));
        }
        if self.wait_for_model_timeout <= 0.0 {
            let mut flags = Vec::new();
            if self.wait_for_model_interval_set && self.wait_for_model_interval != 5.0 {
                flags.push("--wait-for-model-interval");
            }
            if self.wait_for_model_mode_set && self.wait_for_model_mode != "inference" {
                flags.push("--wait-for-model-mode");
            }
            if !flags.is_empty() {
                return Err(EndpointError::InvalidConfig(format!(
                    "{} has no effect unless --wait-for-model-timeout is set to a positive value",
                    flags.join(", ")
                )));
            }
        }
        match self.request_content_type {
            None if requires_form_data => {
                self.request_content_type = Some(RequestContentType::MultipartFormData);
            }
            None => {}
            Some(RequestContentType::ApplicationJson) if requires_form_data => {
                return Err(EndpointError::InvalidConfig(format!(
                    "endpoint {endpoint_id:?} requires multipart/form-data; application/json is not supported"
                )));
            }
            Some(RequestContentType::ApplicationJson) => {}
            Some(RequestContentType::MultipartFormData) if !requires_form_data => {
                return Err(EndpointError::InvalidConfig(format!(
                    "request_content_type=multipart_form_data is only supported for endpoints that accept form-data; endpoint {endpoint_id:?} does not"
                )));
            }
            Some(RequestContentType::MultipartFormData) => {}
        }
        Ok(self)
    }
}

/// Validated endpoint policy bound to one selected endpoint factory.
///
/// Fields remain private so callers cannot assemble an adapter/config pair by
/// struct literal. The frozen registry is the only constructor.
#[derive(Clone, PartialEq)]
pub struct EffectiveEndpointConfig {
    inner: RawEndpointConfig,
}

impl EffectiveEndpointConfig {
    pub(crate) fn from_validated(inner: RawEndpointConfig) -> Self {
        Self { inner }
    }

    /// Borrow validated identity-free endpoint policy.
    pub fn as_raw(&self) -> &RawEndpointConfig {
        &self.inner
    }

    /// Clone validated identity-free endpoint policy.
    pub fn to_raw(&self) -> RawEndpointConfig {
        self.inner.clone()
    }

    /// Effective streaming policy after descriptor normalization.
    pub const fn streaming(&self) -> bool {
        self.inner.streaming
    }

    /// Effective request content type after descriptor normalization.
    pub const fn request_content_type(&self) -> Option<RequestContentType> {
        self.inner.request_content_type
    }
}

impl fmt::Debug for EffectiveEndpointConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("EffectiveEndpointConfig")
            .field(&self.inner)
            .finish()
    }
}

fn validate_url(raw: &str) -> EndpointResult<()> {
    if raw != raw.trim() {
        return Err(EndpointError::InvalidConfig(format!(
            "URL {raw:?} has leading or trailing whitespace"
        )));
    }
    if raw.chars().any(char::is_whitespace) {
        return Err(EndpointError::InvalidConfig(format!(
            "URL {raw:?} contains whitespace"
        )));
    }
    let parsed = Url::parse(raw).map_err(|_| {
        EndpointError::InvalidConfig(format!(
            "URL {raw:?} is missing scheme or host. Expected an http(s) or grpc(s) URL."
        ))
    })?;
    // `dynosim://offline` materializes in process and opens no socket.
    if !matches!(
        parsed.scheme(),
        "http" | "https" | "grpc" | "grpcs" | "dynosim"
    ) {
        return Err(EndpointError::InvalidConfig(format!(
            "URL {raw:?} has unsupported scheme {:?}. Expected 'http', 'https', 'grpc', 'grpcs', or 'dynosim'.",
            parsed.scheme()
        )));
    }
    if parsed.host_str().is_none() {
        return Err(EndpointError::InvalidConfig(format!(
            "URL {raw:?} is missing scheme or host. Expected an http(s) or grpc(s) URL."
        )));
    }
    if let Some(port) = parsed.port()
        && port == 0
    {
        return Err(EndpointError::InvalidConfig(format!(
            "URL {raw:?} has port {port} outside the valid range 1..65535."
        )));
    }
    Ok(())
}

fn validate_control_hook_timeout(value: Option<f64>, field: &str) -> EndpointResult<()> {
    if let Some(value) = value
        && (!value.is_finite() || value < 0.0)
    {
        return Err(EndpointError::InvalidConfig(format!(
            "{field} must be finite and non-negative"
        )));
    }
    Ok(())
}

fn validate_origin_relative_path(raw: &str, field: &str) -> EndpointResult<()> {
    if raw != raw.trim() {
        return Err(EndpointError::InvalidConfig(format!(
            "{field} must not have leading or trailing whitespace"
        )));
    }
    if raw.chars().any(char::is_whitespace) {
        return Err(EndpointError::InvalidConfig(format!(
            "{field} must not contain whitespace"
        )));
    }
    if !raw.starts_with('/') || raw.starts_with("//") || raw.contains("://") {
        return Err(EndpointError::InvalidConfig(format!(
            "{field} must be an origin-relative path beginning with '/'"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_control_hook_paths_must_be_relative() {
        let error = RawEndpointConfig {
            urls: vec!["http://127.0.0.1:8000".to_string()],
            reset_kv_cache: Some(ResetKvCacheConfig {
                path: Some("http://bad.example/reset_prefix_cache".to_string()),
                ..ResetKvCacheConfig::default()
            }),
            ..RawEndpointConfig::default()
        }
        .validate_against(true, false, "chat", false)
        .unwrap_err()
        .to_string();
        assert!(error.contains("relative"), "{error}");
    }
}
