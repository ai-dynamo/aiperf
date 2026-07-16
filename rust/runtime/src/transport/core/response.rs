// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response variants collected during a request.

use bytes::Bytes;

use crate::transport::core::sse::SseMessage;

/// A raw text (non-SSE) response body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextResponse {
    /// Clock-nanoseconds when the body finished being read.
    pub perf_ns: i64,
    /// The raw text body.
    pub text: String,
    /// Exact response bytes. Unlike `text`, this preserves binary control-plane
    /// payloads such as public benchmark Parquet files.
    pub body: Bytes,
    /// The response `Content-Type`, if known.
    pub content_type: Option<String>,
}

impl TextResponse {
    /// Parse the body as JSON, or `None` if empty/invalid.
    pub fn json(&self) -> Option<serde_json::Value> {
        if self.text.is_empty() {
            return None;
        }
        serde_json::from_str(&self.text).ok()
    }
}

/// A single response item — an SSE message or a full text body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Response {
    Sse(SseMessage),
    Text(TextResponse),
}

impl Response {
    /// The arrival/read timestamp of this response item (clock-ns).
    pub fn perf_ns(&self) -> i64 {
        match self {
            Response::Sse(m) => m.perf_ns,
            Response::Text(t) => t.perf_ns,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::core::sse::SseMessage;

    #[test]
    fn text_response_parses_json() {
        let r = TextResponse {
            perf_ns: 5,
            text: "{\"ok\":true}".into(),
            body: Bytes::from_static(b"{\"ok\":true}"),
            content_type: Some("application/json".into()),
        };
        assert_eq!(r.json().unwrap()["ok"], serde_json::json!(true));
    }

    #[test]
    fn text_response_invalid_json_is_none() {
        let r = TextResponse {
            perf_ns: 5,
            text: "not json".into(),
            body: Bytes::from_static(b"not json"),
            content_type: None,
        };
        assert!(r.json().is_none());
    }

    #[test]
    fn response_perf_ns_delegates() {
        let sse = Response::Sse(SseMessage::parse("data: x", 7));
        assert_eq!(sse.perf_ns(), 7);
        let text = Response::Text(TextResponse {
            perf_ns: 9,
            text: "y".into(),
            body: Bytes::from_static(b"y"),
            content_type: None,
        });
        assert_eq!(text.perf_ns(), 9);
    }
}
