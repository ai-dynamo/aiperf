// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Header composition. Port of `BaseTransport.build_headers` +
//! `AioHttpTransport.get_transport_headers`.

use std::collections::BTreeMap;

use crate::transport::http::models::RequestConfig;

/// Compose the final header set: base (User-Agent) -> correlation/request-id ->
/// endpoint headers -> transport headers (Accept, Content-Type). Later sources
/// override earlier ones, matching Python's merge order.
pub fn build_headers(
    cfg: &RequestConfig,
    streaming: bool,
    session_header: Option<&str>,
    user_agent: &str,
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

    // Endpoint headers override base/correlation.
    for (k, v) in &cfg.headers {
        h.insert(k.clone(), v.clone());
    }

    // Transport headers.
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

    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::http::models::RequestConfig;

    #[test]
    fn sets_user_agent_accept_and_content_type_for_streaming() {
        let cfg = RequestConfig::new("http://h/x");
        let h = build_headers(&cfg, true, None, "aiperf/test");
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
        let h = build_headers(&cfg, false, None, "aiperf/test");
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
        let h = build_headers(&cfg, true, Some("X-Session"), "aiperf/test");
        assert_eq!(h.get("X-Session").map(String::as_str), Some("sess-1"));
        assert_eq!(h.get("X-Request-ID").map(String::as_str), Some("req-1"));
    }

    #[test]
    fn endpoint_headers_win_over_base() {
        let cfg = RequestConfig::new("http://h/x").header("User-Agent", "override");
        let h = build_headers(&cfg, true, None, "aiperf/test");
        assert_eq!(h.get("User-Agent").map(String::as_str), Some("override"));
    }
}
