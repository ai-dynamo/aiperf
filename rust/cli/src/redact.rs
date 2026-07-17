// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Credential redaction for the config echoed into export artifacts.
//!
//! `profile_export_aiperf.json` embeds the authored `BenchmarkConfig` as
//! `input_config`. Only this exported copy is redacted; the runtime config keeps
//! the credentials needed for dispatch.

use serde_json::Value;

/// Placeholder substituted for every redacted credential value.
pub const REDACTED_VALUE: &str = "<redacted>";

/// Header names whose values carry credentials.
const SENSITIVE_HEADER_NAMES: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "ocp-apim-subscription-key",
    "x-goog-api-key",
    "x-functions-key",
    "aeg-sas-key",
    "x-amz-security-token",
];

/// Redact credentials in the serialized `input_config` copy, in place.
///
/// Redacts `api_key`, sensitive headers, and URL userinfo. Unexpected shapes are
/// left untouched.
pub fn redact_input_config(value: &mut Value) {
    let Some(endpoint) = value.get_mut("endpoint").and_then(Value::as_object_mut) else {
        return;
    };

    if let Some(api_key) = endpoint.get_mut("api_key")
        && !api_key.is_null()
    {
        *api_key = Value::String(REDACTED_VALUE.to_string());
    }

    if let Some(headers) = endpoint.get_mut("headers").and_then(Value::as_object_mut) {
        for (name, header_value) in headers.iter_mut() {
            if is_sensitive_header(name) {
                *header_value = Value::String(REDACTED_VALUE.to_string());
            }
        }
    }

    if let Some(urls) = endpoint.get_mut("urls").and_then(Value::as_array_mut) {
        for url in urls.iter_mut() {
            if let Some(text) = url.as_str() {
                let redacted = redact_url(text);
                *url = Value::String(redacted);
            }
        }
    }
}

fn is_sensitive_header(name: &str) -> bool {
    SENSITIVE_HEADER_NAMES
        .iter()
        .any(|sensitive| name.eq_ignore_ascii_case(sensitive))
}

/// Strip `user:password@` userinfo from a URL, preserving everything else.
///
/// Handles `scheme://` URIs and bare `user:pass@host` forms.
fn redact_url(url: &str) -> String {
    if let Some(scheme_end) = url.find("://") {
        let after_scheme = scheme_end + 3;
        let rest = &url[after_scheme..];
        let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        if let Some(at) = rest[..authority_end].find('@') {
            let mut out = String::with_capacity(url.len());
            out.push_str(&url[..after_scheme]);
            out.push_str(REDACTED_VALUE);
            out.push_str(&rest[at..]);
            return out;
        }
        return url.to_string();
    }
    if let Some(at) = url.find('@') {
        let prefix = &url[..at];
        if prefix.contains(':') && !prefix.contains('/') {
            let mut out = String::with_capacity(url.len());
            out.push_str(REDACTED_VALUE);
            out.push_str(&url[at..]);
            return out;
        }
    }
    url.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn redacts_api_key_headers_and_url_userinfo() {
        let mut cfg = json!({
            "endpoint": {
                "urls": ["http://user:secret@host:8000", "http://127.0.0.1:8000"],
                "api_key": "sk-integration-secret-REDACT-12345",
                "headers": {
                    "Authorization": "Bearer sk-integration-secret-REDACT-12345",
                    "X-Api-Key": "sk-integration-secret-REDACT-12345",
                    "X-Custom-Tracking": "trace-abc-123"
                }
            }
        });

        redact_input_config(&mut cfg);
        let serialized = serde_json::to_string(&cfg).unwrap();

        assert!(
            !serialized.contains("sk-integration-secret-REDACT-12345"),
            "api key leaked: {serialized}"
        );
        assert!(
            !serialized.contains("user:secret"),
            "userinfo leaked: {serialized}"
        );
        let endpoint = &cfg["endpoint"];
        assert_eq!(endpoint["api_key"], REDACTED_VALUE);
        assert_eq!(endpoint["headers"]["Authorization"], REDACTED_VALUE);
        assert_eq!(endpoint["headers"]["X-Api-Key"], REDACTED_VALUE);
        assert_eq!(endpoint["headers"]["X-Custom-Tracking"], "trace-abc-123");
        assert_eq!(endpoint["urls"][1], "http://127.0.0.1:8000");
        assert_eq!(endpoint["urls"][0], "http://<redacted>@host:8000");
    }

    #[test]
    fn null_api_key_and_absent_endpoint_are_safe() {
        let mut null_key = json!({"endpoint": {"api_key": null}});
        redact_input_config(&mut null_key);
        assert!(null_key["endpoint"]["api_key"].is_null());

        let mut no_endpoint = json!({"models": {"items": []}});
        redact_input_config(&mut no_endpoint);
        assert_eq!(no_endpoint, json!({"models": {"items": []}}));
    }

    #[test]
    fn is_idempotent() {
        let mut cfg = json!({"endpoint": {"api_key": "s3cr3t", "urls": [], "headers": {}}});
        redact_input_config(&mut cfg);
        let once = cfg.clone();
        redact_input_config(&mut cfg);
        assert_eq!(cfg, once);
    }
}
