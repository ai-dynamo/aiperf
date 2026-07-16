// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Credential redaction for the config echoed into export artifacts.
//!
//! The native front door embeds a JSON dump of the authored `BenchmarkConfig`
//! as the genai-perf-v1 `input_config` envelope (see `load.rs`, where the run
//! request is built). That dump is written verbatim into
//! `profile_export_aiperf.json`, so the raw API key and any credentialed
//! headers would otherwise leak into a user-visible artifact.
//!
//! Python avoids this with per-field JSON serializers on `EndpointConfig`
//! (`src/aiperf/config/endpoint.py`): `api_key` -> `<redacted>`, `headers`
//! filtered through `redact_headers`, and `urls` stripped of userinfo via
//! `redact_url`. The runtime config the child actually dispatches with keeps the
//! real credentials; only the `input_config` COPY embedded for export is
//! redacted. This module reproduces that redaction on the serialized copy,
//! porting the header-name set and URL logic from `src/aiperf/common/redact.py`.

use serde_json::Value;

/// Placeholder substituted for every redacted credential value. Matches
/// `aiperf.common.redact.REDACTED_VALUE`.
pub const REDACTED_VALUE: &str = "<redacted>";

/// Header names (compared case-insensitively) whose values carry credentials.
/// Mirrors `_SENSITIVE_HEADER_NAMES` in `src/aiperf/common/redact.py`.
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
/// Navigates to the `endpoint` object and redacts `api_key`, `headers`, and
/// `urls`, matching the field serializers on Python's `EndpointConfig`. Absent
/// or unexpectedly-typed fields are left untouched, so the function is safe to
/// call on any config shape and is idempotent.
pub fn redact_input_config(value: &mut Value) {
    let Some(endpoint) = value.get_mut("endpoint").and_then(Value::as_object_mut) else {
        return;
    };

    // api_key: any non-null value becomes the placeholder (null stays null).
    if let Some(api_key) = endpoint.get_mut("api_key")
        && !api_key.is_null()
    {
        *api_key = Value::String(REDACTED_VALUE.to_string());
    }

    // headers: replace the value of every credential-carrying header name.
    if let Some(headers) = endpoint.get_mut("headers").and_then(Value::as_object_mut) {
        for (name, header_value) in headers.iter_mut() {
            if is_sensitive_header(name) {
                *header_value = Value::String(REDACTED_VALUE.to_string());
            }
        }
    }

    // urls: strip any `user:password@` userinfo from each entry.
    if let Some(urls) = endpoint.get_mut("urls").and_then(Value::as_array_mut) {
        for url in urls.iter_mut() {
            if let Some(text) = url.as_str() {
                let redacted = redact_url(text);
                *url = Value::String(redacted);
            }
        }
    }
}

/// Case-insensitive membership test against [`SENSITIVE_HEADER_NAMES`].
fn is_sensitive_header(name: &str) -> bool {
    SENSITIVE_HEADER_NAMES
        .iter()
        .any(|sensitive| name.eq_ignore_ascii_case(sensitive))
}

/// Strip `user:password@` userinfo from a URL, preserving everything else.
///
/// Ported from `aiperf.common.redact.redact_url`: handle any `scheme://` URI
/// (userinfo bounded by `/`, `?`, `#`, `@`) plus a bare scheme-less
/// `user:pass@host` form. Returns the input unchanged when no userinfo is
/// present.
fn redact_url(url: &str) -> String {
    if let Some(scheme_end) = url.find("://") {
        let after_scheme = scheme_end + 3;
        let rest = &url[after_scheme..];
        // Userinfo ends at the first `@` that precedes any `/`, `?`, or `#`.
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
    // Bare userinfo: `user:pass@host` (must contain `:` before the `@`).
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
        // Non-sensitive header and clean URL preserved.
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
