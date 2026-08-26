// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP header composition.

use std::collections::BTreeMap;

use crate::transport::http::models::RequestConfig;

/// Read the `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID` opt-in
/// toggle. Mirrors Python's `Environment.HTTP` settings, which are likewise
/// plain env-var-gated globals rather than per-endpoint Config v2 surface.
/// Callers read this once at transport-construction time (see
/// `HttpTransport::new`), not per request, so `build_headers` itself stays a
/// pure function of its arguments and is directly unit-testable without
/// mutating process env.
pub fn dynamo_session_id_from_correlation_id_enabled() -> bool {
    env_flag_enabled("AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID")
}

/// Read the `AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID` opt-in toggle.
/// When enabled, `X-Session-ID` is sent ADDITIVELY alongside the correlation
/// header (distinct from `--session-header`, which RENAMES the single
/// correlation header). Use this when an external router requires an
/// `X-Session-ID` session-affinity header. Read once at transport construction.
pub fn x_session_id_from_correlation_id_enabled() -> bool {
    env_flag_enabled("AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID")
}

/// Read the `AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID` opt-in toggle.
/// When enabled, `X-SMG-Routing-Key` is sent ADDITIVELY with the stable
/// correlation value for the SGLang Model Gateway manual routing policy
/// (co-locates a session's requests on one worker). Read once at construction.
pub fn x_smg_routing_key_from_correlation_id_enabled() -> bool {
    env_flag_enabled("AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID")
}

/// Shared parse for the boolean session-affinity env toggles: `1`/`true`
/// (case-insensitive, trimmed) enables; anything else (or unset) disables.
fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true"))
}

/// Add the default router-facing affinity header for a stable correlation ID.
///
/// This policy is shared by the direct transport facade and the native
/// endpoint-binding path. Both paths receive caller-owned headers, so remove
/// every case-insensitive spelling before inserting the canonical derived
/// value.
pub fn apply_default_session_affinity_header(
    headers: &mut BTreeMap<String, String>,
    correlation_id: Option<&str>,
) {
    let Some(correlation_id) = correlation_id else {
        headers.retain(|name, _| !name.eq_ignore_ascii_case("X-Session-Affinity"));
        return;
    };
    let has_conflict = headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("X-Session-Affinity")
            && (name != "X-Session-Affinity" || value != correlation_id)
    });
    if !has_conflict
        && headers
            .get("X-Session-Affinity")
            .is_some_and(|value| value == correlation_id)
    {
        return;
    }
    headers.retain(|name, _| !name.eq_ignore_ascii_case("X-Session-Affinity"));
    headers.insert("X-Session-Affinity".to_string(), correlation_id.to_string());
}

/// endpoint headers -> transport headers (`Accept` overrides, `Content-Type`
/// only fills in when absent) -> derived session-affinity headers (always
/// last/authoritative) -> derived Dynamo session headers (opt-in, also
/// authoritative). Later sources otherwise override earlier ones.
pub fn build_headers(
    cfg: &RequestConfig,
    streaming: bool,
    session_header: Option<&str>,
    user_agent: &str,
    dynamo_session_id_from_correlation_id: bool,
    x_session_id_from_correlation_id: bool,
    x_smg_routing_key_from_correlation_id: bool,
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

    // Apply derived session-affinity headers last so they are authoritative and
    // cannot be silently overwritten by endpoint or transport headers above.
    // These are ADDITIVE (distinct from `session_header`, which RENAMES the
    // single correlation header): some external routers require a dedicated
    // session-affinity header ALONGSIDE the correlation header. Strip any
    // caller-supplied variants case-insensitively first, since HTTP header
    // names are case-insensitive and `h` is a plain string-keyed map.
    apply_default_session_affinity_header(&mut h, cfg.correlation_id.as_deref());
    if let Some(corr) = &cfg.correlation_id {
        if x_session_id_from_correlation_id {
            h.retain(|k, _| !k.eq_ignore_ascii_case("X-Session-ID"));
            h.insert("X-Session-ID".to_string(), corr.clone());
        }
        if x_smg_routing_key_from_correlation_id {
            h.retain(|k, _| !k.eq_ignore_ascii_case("X-SMG-Routing-Key"));
            h.insert("X-SMG-Routing-Key".to_string(), corr.clone());
        }
    }

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
        let h = build_headers(&cfg, true, None, "aiperf/test", false, false, false);
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
        let h = build_headers(&cfg, false, None, "aiperf/test", false, false, false);
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
        let h = build_headers(
            &cfg,
            true,
            Some("X-Session"),
            "aiperf/test",
            false,
            false,
            false,
        );
        assert_eq!(h.get("X-Session").map(String::as_str), Some("sess-1"));
        assert_eq!(h.get("X-Request-ID").map(String::as_str), Some("req-1"));
    }

    #[test]
    fn endpoint_headers_win_over_base() {
        let cfg = RequestConfig::new("http://h/x").header("User-Agent", "override");
        let h = build_headers(&cfg, true, None, "aiperf/test", false, false, false);
        assert_eq!(h.get("User-Agent").map(String::as_str), Some("override"));
    }

    #[test]
    fn dynamo_session_id_from_correlation_id_disabled_by_default() {
        let cfg = RequestConfig::new("http://h/x").correlation_id("sess-1");
        let h = build_headers(&cfg, true, None, "aiperf/test", false, false, false);
        assert_eq!(h.get("X-Dynamo-Session-ID"), None);
    }

    #[test]
    fn dynamo_session_id_from_correlation_id_derives_session_header() {
        let cfg = RequestConfig::new("http://h/x").correlation_id("sess-1");
        let h = build_headers(&cfg, true, None, "aiperf/test", true, false, false);
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
        let h = build_headers(&cfg, true, None, "aiperf/test", true, false, false);
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
    fn x_session_id_disabled_by_default() {
        let cfg = RequestConfig::new("http://h/x").correlation_id("sess-1");
        let h = build_headers(&cfg, true, None, "aiperf/test", false, false, false);
        assert_eq!(h.get("X-Session-ID"), None);
    }

    #[test]
    fn session_affinity_is_default_and_independent_of_session_id_opt_in() {
        let cfg = RequestConfig::new("http://h/x").correlation_id("sess-1");
        let h = build_headers(&cfg, true, None, "aiperf/test", false, false, false);
        assert_eq!(
            h.get("X-Correlation-ID").map(String::as_str),
            Some("sess-1")
        );
        assert_eq!(
            h.get("X-Session-Affinity").map(String::as_str),
            Some("sess-1")
        );
        assert_eq!(h.get("X-Session-ID"), None);
    }

    #[test]
    fn session_affinity_replaces_caller_header_case_insensitively() {
        let cfg = RequestConfig::new("http://h/x")
            .correlation_id("sess-1")
            .header("x-session-affinity", "stale-lowercase")
            .header("X-Session-Affinity", "stale-canonical");
        let h = build_headers(&cfg, true, None, "aiperf/test", false, false, false);
        assert_eq!(
            h.get("X-Session-Affinity").map(String::as_str),
            Some("sess-1")
        );
        assert!(!h.contains_key("x-session-affinity"));
    }

    #[test]
    fn session_affinity_removes_authored_variants_without_correlation_id() {
        let cfg = RequestConfig::new("http://h/x")
            .header("x-session-affinity", "stale-lowercase")
            .header("X-Session-Affinity", "stale-canonical");
        let h = build_headers(&cfg, true, None, "aiperf/test", false, false, false);
        assert!(
            h.keys()
                .all(|name| !name.eq_ignore_ascii_case("X-Session-Affinity")),
            "affinity must be absent without a correlation ID: {h:?}"
        );
    }

    #[test]
    fn default_session_affinity_normalization_is_idempotent() {
        let mut headers =
            BTreeMap::from([("X-Session-Affinity".to_string(), "sess-1".to_string())]);
        apply_default_session_affinity_header(&mut headers, Some("sess-1"));
        assert_eq!(
            headers,
            BTreeMap::from([("X-Session-Affinity".to_string(), "sess-1".to_string(),)])
        );
    }

    #[test]
    fn x_session_id_is_additive_with_correlation_header() {
        // Additive: the correlation header stays under its own name AND
        // X-Session-ID is sent with the same stable value.
        let cfg = RequestConfig::new("http://h/x").correlation_id("sess-1");
        let h = build_headers(&cfg, true, None, "aiperf/test", false, true, false);
        assert_eq!(
            h.get("X-Correlation-ID").map(String::as_str),
            Some("sess-1")
        );
        assert_eq!(h.get("X-Session-ID").map(String::as_str), Some("sess-1"));
    }

    #[test]
    fn x_session_id_overrides_caller_supplied_header_case_insensitively() {
        let cfg = RequestConfig::new("http://h/x")
            .correlation_id("sess-1")
            .header("x-session-id", "stale");
        let h = build_headers(&cfg, true, None, "aiperf/test", false, true, false);
        assert_eq!(h.get("X-Session-ID").map(String::as_str), Some("sess-1"));
        assert!(!h.contains_key("x-session-id"));
    }

    #[test]
    fn x_smg_routing_key_disabled_by_default() {
        let cfg = RequestConfig::new("http://h/x").correlation_id("sess-1");
        let h = build_headers(&cfg, true, None, "aiperf/test", false, false, false);
        assert_eq!(h.get("X-SMG-Routing-Key"), None);
    }

    #[test]
    fn x_smg_routing_key_derives_from_correlation_id() {
        let cfg = RequestConfig::new("http://h/x").correlation_id("sess-1");
        let h = build_headers(&cfg, true, None, "aiperf/test", false, false, true);
        assert_eq!(
            h.get("X-SMG-Routing-Key").map(String::as_str),
            Some("sess-1")
        );
    }

    #[test]
    fn session_affinity_headers_absent_without_correlation_id() {
        // No correlation id => nothing to derive from, even when both flags on.
        let cfg = RequestConfig::new("http://h/x");
        let h = build_headers(&cfg, true, None, "aiperf/test", false, true, true);
        assert_eq!(h.get("X-Session-ID"), None);
        assert_eq!(h.get("X-SMG-Routing-Key"), None);
    }

    #[test]
    fn dynamo_session_id_from_correlation_id_overrides_caller_supplied_headers() {
        // Case-insensitive strip-then-set: a caller-supplied variant (any
        // case) must not survive alongside the derived value.
        let cfg = RequestConfig::new("http://h/x")
            .correlation_id("sess-1")
            .header("x-dynamo-session-id", "stale")
            .header("X-DYNAMO-PARENT-SESSION-ID", "stale-parent");
        let h = build_headers(&cfg, true, None, "aiperf/test", true, false, false);
        assert_eq!(
            h.get("X-Dynamo-Session-ID").map(String::as_str),
            Some("sess-1")
        );
        assert!(!h.contains_key("x-dynamo-session-id"));
        assert!(!h.contains_key("X-DYNAMO-PARENT-SESSION-ID"));
    }
}
