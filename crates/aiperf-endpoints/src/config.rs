// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Endpoint configuration validation and normalization.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use url::Url;

use crate::metadata::{EndpointType, metadata_for};
use crate::models::{EndpointError, EndpointResult};

/// Wire request content type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestContentType {
    /// JSON request body.
    ApplicationJson,
    /// Multipart form-data request body.
    MultipartFormData,
}

/// Endpoint configuration used by endpoint formatters and validators.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EndpointConfig {
    /// Endpoint type.
    #[serde(rename = "type")]
    pub endpoint_type: EndpointType,
    /// Base URLs.
    pub urls: Vec<String>,
    /// Optional path override.
    pub path: Option<String>,
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
    /// Use legacy `max_tokens` for chat.
    pub use_legacy_max_tokens: bool,
    /// Request usage in streaming frames when supported.
    pub use_server_token_count: bool,
    /// Endpoint-level extra body fields.
    pub extra: Option<Map<String, Value>>,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            endpoint_type: EndpointType::Chat,
            urls: Vec::new(),
            path: None,
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
            extra: None,
        }
    }
}

impl EndpointConfig {
    /// Validate and normalize config fields.
    pub fn validate(mut self) -> EndpointResult<Self> {
        if self.template.is_some() && self.endpoint_type == EndpointType::Chat {
            self.endpoint_type = EndpointType::Template;
        }
        let metadata = metadata_for(self.endpoint_type);
        if self.streaming && !metadata.supports_streaming {
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
        let legacy_template = self
            .extra
            .as_ref()
            .and_then(|extra| extra.get("payload_template"))
            .and_then(Value::as_str);
        if self.endpoint_type == EndpointType::Template
            && self.template.is_none()
            && legacy_template.is_none()
        {
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
            None if metadata.requires_form_data => {
                self.request_content_type = Some(RequestContentType::MultipartFormData);
            }
            None => {}
            Some(RequestContentType::ApplicationJson) if metadata.requires_form_data => {
                return Err(EndpointError::InvalidConfig(format!(
                    "endpoint type {:?} requires multipart/form-data; application/json is not supported",
                    self.endpoint_type
                )));
            }
            Some(RequestContentType::ApplicationJson) => {}
            Some(RequestContentType::MultipartFormData) if !metadata.requires_form_data => {
                return Err(EndpointError::InvalidConfig(format!(
                    "request_content_type=multipart_form_data is only supported for endpoint types that accept form-data; endpoint type {:?} does not",
                    self.endpoint_type
                )));
            }
            Some(RequestContentType::MultipartFormData) => {}
        }
        Ok(self)
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
            "URL {raw:?} is missing scheme or host. Expected 'http://host:port' or 'https://host:port'."
        ))
    })?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err(EndpointError::InvalidConfig(format!(
            "URL {raw:?} has unsupported scheme {:?}. Expected 'http' or 'https'.",
            parsed.scheme()
        )));
    }
    if parsed.host_str().is_none() {
        return Err(EndpointError::InvalidConfig(format!(
            "URL {raw:?} is missing scheme or host. Expected 'http://host:port' or 'https://host:port'."
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
