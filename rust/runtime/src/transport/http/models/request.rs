// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request configuration + strategy enums.

use std::collections::BTreeMap;

use crate::transport::core::ConnectionReuseStrategy;

/// Which HTTP protocol / handshake to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HttpVersion {
    /// ALPN on TLS; HTTP/1.1 on cleartext.
    #[default]
    Auto,
    /// Force HTTP/1.1.
    Http1Only,
    /// Force HTTP/2, including h2c prior-knowledge on cleartext.
    Http2PriorKnowledge,
}

/// Everything needed to dispatch one request. Built with the fluent methods.
#[derive(Debug, Clone)]
pub struct RequestConfig {
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub params: BTreeMap<String, String>,
    pub cancel_after_ns: Option<i64>,
    pub correlation_id: Option<String>,
    /// The parent session's `correlation_id`, when this request is a DAG
    /// (`dag_jsonl` Fork/Spawn) child of another session. `None` for
    /// ordinary sampled turns. Feeds `X-Dynamo-Parent-Session-ID` under
    /// `--dynamo-session-id-from-correlation-id`; see `headers::build_headers`.
    pub parent_correlation_id: Option<String>,
    pub request_id: Option<String>,
    pub is_final_turn: bool,
    pub reuse: ConnectionReuseStrategy,
}

impl RequestConfig {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            headers: BTreeMap::new(),
            params: BTreeMap::new(),
            cancel_after_ns: None,
            correlation_id: None,
            parent_correlation_id: None,
            request_id: None,
            is_final_turn: true,
            reuse: ConnectionReuseStrategy::Pooled,
        }
    }
    pub fn header(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.headers.insert(k.into(), v.into());
        self
    }
    pub fn param(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.params.insert(k.into(), v.into());
        self
    }
    pub fn cancel_after_ns(mut self, ns: i64) -> Self {
        self.cancel_after_ns = Some(ns);
        self
    }
    pub fn correlation_id(mut self, s: impl Into<String>) -> Self {
        self.correlation_id = Some(s.into());
        self
    }
    pub fn parent_correlation_id(mut self, s: impl Into<String>) -> Self {
        self.parent_correlation_id = Some(s.into());
        self
    }
    pub fn request_id(mut self, s: impl Into<String>) -> Self {
        self.request_id = Some(s.into());
        self
    }
    pub fn final_turn(mut self, v: bool) -> Self {
        self.is_final_turn = v;
        self
    }
    pub fn reuse(mut self, r: ConnectionReuseStrategy) -> Self {
        self.reuse = r;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_accumulates_fields() {
        let c = RequestConfig::new("http://h/v1/chat/completions")
            .header("X-A", "1")
            .param("stream", "true")
            .cancel_after_ns(5_000)
            .correlation_id("sess-1")
            .reuse(ConnectionReuseStrategy::StickyUserSessions);
        assert_eq!(c.url, "http://h/v1/chat/completions");
        assert_eq!(c.headers.get("X-A").map(String::as_str), Some("1"));
        assert_eq!(c.params.get("stream").map(String::as_str), Some("true"));
        assert_eq!(c.cancel_after_ns, Some(5_000));
        assert_eq!(c.correlation_id.as_deref(), Some("sess-1"));
        assert_eq!(c.reuse, ConnectionReuseStrategy::StickyUserSessions);
    }

    #[test]
    fn defaults_are_pooled_and_auto() {
        assert_eq!(
            ConnectionReuseStrategy::default(),
            ConnectionReuseStrategy::Pooled
        );
        assert_eq!(HttpVersion::default(), HttpVersion::Auto);
    }
}
