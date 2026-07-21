// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP header composition.

use std::collections::BTreeMap;

use crate::transport::http::models::RequestConfig;

/// Read the `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID` opt-in
/// toggle. Mirrors Python's `Environment.HTTP` settings, which are likewise
/// plain env-var-gated globals rather than per-endpoint Config v2 surface.
/// Callers read this once at transport-construction time (see
/// `HttpTransport::with_dynamo_session_id_from_correlation_id`), not per
/// request, so `build_headers` itself stays a pure function of its
/// arguments and is directly unit-testable without mutating process env.
pub fn dynamo_session_id_from_correlation_id_enabled() -> bool {
    std::env::var("AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID")
        .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true"))
}

/// Compose the final header set: base (User-Agent) -> correlation/request-id ->
/// endpoint headers -> transport headers (Accept, Content-Type) -> derived
/// Dynamo session headers (opt-in, always last/authoritative). Later sources
/// override earlier ones.
pub fn build_headers(
    cfg: &RequestConfig,
    streaming: bool,
    session_header: Option<&str>,
    user_agent: &str,
    dynamo_session_id_from_correlation_id: bool,
) -> BTreeMap<String, String> {
    let mut h: BTreeMap<String, String> = BTreeMap::new();
    h.insert("User-Agent".to_string(), user_agent.to_string());

    if let Some(req_id) = &cfg.request_id {
        h.insert("X-Request-ID".to_string(), req_id.clone());
    }
    if let Some(corr) = &cfg.correlation_id {
        let name = session_header.unwrap_or("X-Correlation-ID");
        h.insert(name.to_string(), corr.clone());
    }

    for (k, v) in &cfg.headers {
        h.insert(k.clone(), v.clone());
    }

    h.insert(
        "Accept".to_string(),
        if streaming {
            "text/event-stream"
        } else {
            "application/json"
        }
        .to_string(),
    );
    h.entry("Content-Type".to_string())
        .or_insert_with(|| "application/json".to_string());

    // Apply derived Dynamo session headers last so they are authoritative and
    // cannot be overwritten by endpoint or transport headers above. Use this
    // with a Dynamo frontend running --router-session-affinity-ttl-secs to
    // pin every turn of a session to the replica holding its KV prefix.
    // Strip any caller-supplied variants case-insensitively first, since HTTP
    // header names are case-insensitive and `h` is a plain string-keyed map.
    if let Some(corr) = &cfg.correlation_id
        && dynamo_session_id_from_correlation_id
    {
        h.retain(|k, _| {
            !k.eq_ignore_ascii_case("X-Dynamo-Session-ID")
                && !k.eq_ignore_ascii_case("X-Dynamo-Parent-Session-ID")
        });
        h.insert("X-Dynamo-Session-ID".to_string(), corr.clone());
        if let Some(parent) = &cfg.parent_correlation_id {
            h.insert("X-Dynamo-Parent-Session-ID".to_string(), parent.clone());
        }
    }

    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::http::models::RequestConfig;

    #[test]
    fn sets_user_agent_accept_and_content_type_for_streaming() {
        let cfg = RequestConfig::new("http://h/x");
        let h = build_headers(&cfg, true, None, "aiperf/test", false);
        assert_eq!(h.get("User-Agent").map(String::as_str), Some("aiperf/test"));
        assert_eq!(
            h.get("Accept").map(String::as_str),
            Some("text/event-stream")
        );
        assert_eq!(
            h.get("Content-Type").map(String::as_str),
            Some("application/json")
        );
    }

    #[test]
    fn non_streaming_accept_is_json() {
        let cfg = RequestConfig::new("http://h/x");
        let h = build_headers(&cfg, false, None, "aiperf/test", false);
        assert_eq!(
            h.get("Accept").map(String::as_str),
            Some("application/json")
        );
    }

    #[test]
    fn correlation_and_request_id_headers() {
        let cfg = RequestConfig::new("http://h/x")
            .correlation_id("sess-1")
            .request_id("req-1");
        let h = build_headers(&cfg, true, Some("X-Session"), "aiperf/test", false);
        assert_eq!(h.get("X-Session").map(String::as_str), Some("sess-1"));
        assert_eq!(h.get("X-Request-ID").map(String::as_str), Some("req-1"));
    }

    #[test]
    fn endpoint_headers_win_over_base() {
        let cfg = RequestConfig::new("http://h/x").header("User-Agent", "override");
        let h = build_headers(&cfg, true, None, "aiperf/test", false);
        assert_eq!(h.get("User-Agent").map(String::as_str), Some("override"));
    }

    #[test]
    fn dynamo_session_id_from_correlation_id_disabled_by_default() {
        let cfg = RequestConfig::new("http://h/x").correlation_id("sess-1");
        let h = build_headers(&cfg, true, None, "aiperf/test", false);
        assert_eq!(h.get("X-Dynamo-Session-ID"), None);
    }

    #[test]
    fn dynamo_session_id_from_correlation_id_derives_session_header() {
        let cfg = RequestConfig::new("http://h/x").correlation_id("sess-1");
        let h = build_headers(&cfg, true, None, "aiperf/test", true);
        assert_eq!(
            h.get("X-Dynamo-Session-ID").map(String::as_str),
            Some("sess-1")
        );
        assert_eq!(h.get("X-Dynamo-Parent-Session-ID"), None);
    }

    #[test]
    fn dynamo_session_id_from_correlation_id_derives_parent_header() {
        let cfg = RequestConfig::new("http://h/x")
            .correlation_id("child-1")
            .parent_correlation_id("parent-1");
        let h = build_headers(&cfg, true, None, "aiperf/test", true);
        assert_eq!(
            h.get("X-Dynamo-Session-ID").map(String::as_str),
            Some("child-1")
        );
        assert_eq!(
            h.get("X-Dynamo-Parent-Session-ID").map(String::as_str),
            Some("parent-1")
        );
    }

    #[test]
    fn dynamo_session_id_from_correlation_id_overrides_caller_supplied_headers() {
        // Case-insensitive strip-then-set: a caller-supplied variant (any
        // case) must not survive alongside the derived value.
        let cfg = RequestConfig::new("http://h/x")
            .correlation_id("sess-1")
            .header("x-dynamo-session-id", "stale")
            .header("X-DYNAMO-PARENT-SESSION-ID", "stale-parent");
        let h = build_headers(&cfg, true, None, "aiperf/test", true);
        assert_eq!(
            h.get("X-Dynamo-Session-ID").map(String::as_str),
            Some("sess-1")
        );
        assert!(!h.contains_key("x-dynamo-session-id"));
        assert!(!h.contains_key("X-DYNAMO-PARENT-SESSION-ID"));
    }
}
