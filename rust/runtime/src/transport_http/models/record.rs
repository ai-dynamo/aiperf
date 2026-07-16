// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The output of a request: responses + timing + trace + error.

use std::collections::BTreeMap;

use bytes::Bytes;

use crate::transport_http::models::{ErrorDetails, Response, TraceData};

/// A completed (or failed) request with its responses and timing.
#[derive(Debug, Clone, Default)]
pub struct RequestRecord {
    /// Clock-ns when dispatch started.
    pub start_ns: i64,
    /// Exact request body bytes handed to the HTTP client.
    ///
    /// Keeping the reference-counted [`Bytes`] handle beside the terminal
    /// record lets optional raw exporters preserve the authored JSON without a
    /// decode/re-encode pass. Empty-body requests, such as control-plane GETs,
    /// retain an empty value.
    pub request_body: Bytes,
    /// Actual request headers after transport defaults and endpoint overrides
    /// were composed.
    pub request_headers: BTreeMap<String, String>,
    /// Clock-ns when the request completed, if it did.
    pub end_ns: Option<i64>,
    /// Clock-ns of the first response byte (streaming start).
    pub recv_start_ns: Option<i64>,
    /// HTTP status code, if a response header was received.
    pub status: Option<u16>,
    /// Response headers, normalized to lowercase names. Control-plane clients use
    /// this for explicit redirect and cache handling.
    pub response_headers: BTreeMap<String, String>,
    /// Collected responses (SSE messages or an exact text-body record).
    ///
    /// A completely read non-2xx body remains here even though [`Self::error`]
    /// carries the typed HTTP failure, so control-plane callers can preserve
    /// source evidence without treating metric-looking error bytes as success.
    pub responses: Vec<Response>,
    /// Failure detail, if any.
    pub error: Option<ErrorDetails>,
    /// Fine-grained trace timing.
    pub trace: Option<TraceData>,
    /// Clock-ns when the request was cancelled, if applicable.
    pub cancellation_ns: Option<i64>,
    /// The response body was fully drained even though dispatch returned an
    /// error (a non-2xx status), so the underlying HTTP/1 connection is clean
    /// and may be returned to the pool. Lets the transport reuse a lease on
    /// 4xx/5xx instead of forcing a fresh connect during error storms.
    pub reusable_connection: bool,
}

impl RequestRecord {
    /// A fresh record stamped at `start_ns`.
    pub fn started(start_ns: i64) -> Self {
        Self {
            start_ns,
            ..Self::default()
        }
    }
    pub fn was_cancelled(&self) -> bool {
        self.cancellation_ns.is_some()
    }
    pub fn has_error(&self) -> bool {
        self.error.is_some()
    }
    /// Valid when there is no error and at least one response was collected.
    pub fn is_valid(&self) -> bool {
        !self.has_error() && self.start_ns >= 0 && !self.responses.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport_http::models::{ErrorDetails, Response, TextResponse};

    #[test]
    fn valid_when_has_response_and_no_error() {
        let r = RequestRecord {
            start_ns: 10,
            request_body: bytes::Bytes::from_static(b"{\"prompt\":\"x\"}"),
            request_headers: BTreeMap::from([("Content-Type".into(), "application/json".into())]),
            responses: vec![Response::Text(TextResponse {
                perf_ns: 20,
                text: "x".into(),
                body: bytes::Bytes::from_static(b"x"),
                content_type: None,
            })],
            ..RequestRecord::started(10)
        };
        assert!(r.is_valid());
        assert!(!r.has_error());
        assert!(!r.was_cancelled());
        assert_eq!(r.request_body.as_ref(), b"{\"prompt\":\"x\"}");
        assert_eq!(
            r.request_headers.get("Content-Type").map(String::as_str),
            Some("application/json")
        );
    }

    #[test]
    fn invalid_when_error_present() {
        let mut r = RequestRecord::started(10);
        r.error = Some(ErrorDetails::http(500, "boom"));
        assert!(!r.is_valid());
        assert!(r.has_error());
    }

    #[test]
    fn cancellation_flag() {
        let mut r = RequestRecord::started(10);
        r.cancellation_ns = Some(99);
        assert!(r.was_cancelled());
    }
}
