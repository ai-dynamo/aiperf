// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in bounded capture of raw HTTP requests for full-stack parity tests.

use std::collections::VecDeque;
use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use parking_lot::Mutex;

use crate::state::AppState;

const MAX_CAPTURE_BODY_BYTES: usize = 8 * 1024 * 1024;

/// One raw request as received by the Axum frontend, before endpoint decoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestCapture {
    /// HTTP method.
    pub method: String,
    /// URI path without query parameters.
    pub route: String,
    /// Ordered raw header names and values.
    pub headers: Vec<(String, Vec<u8>)>,
    /// Exact request body bytes.
    pub body: Vec<u8>,
}

impl RequestCapture {
    /// Look up the first header value case-insensitively.
    pub fn header(&self, name: &str) -> Option<&[u8]> {
        self.headers
            .iter()
            .find(|(candidate, _)| candidate.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_slice())
    }
}

pub(crate) struct RequestCaptureStore {
    capacity: usize,
    captures: Mutex<VecDeque<RequestCapture>>,
}

impl RequestCaptureStore {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            captures: Mutex::new(VecDeque::with_capacity(capacity)),
        }
    }

    fn push(&self, capture: RequestCapture) {
        if self.capacity == 0 {
            return;
        }
        let mut captures = self.captures.lock();
        if captures.len() == self.capacity {
            captures.pop_front();
        }
        captures.push_back(capture);
    }

    pub(crate) fn snapshot(&self) -> Vec<RequestCapture> {
        self.captures.lock().iter().cloned().collect()
    }

    pub(crate) fn clear(&self) {
        self.captures.lock().clear();
    }
}

pub(crate) async fn capture_request(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let (parts, body) = request.into_parts();
    let body = match to_bytes(body, MAX_CAPTURE_BODY_BYTES).await {
        Ok(body) => body,
        Err(error) => {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("request capture body exceeds {MAX_CAPTURE_BODY_BYTES} bytes: {error}"),
            )
                .into_response();
        }
    };
    state.request_capture_store.push(RequestCapture {
        method: parts.method.to_string(),
        route: parts.uri.path().to_string(),
        headers: parts
            .headers
            .iter()
            .map(|(name, value)| (name.as_str().to_string(), value.as_bytes().to_vec()))
            .collect(),
        body: body.to_vec(),
    });
    next.run(Request::from_parts(parts, Body::from(body))).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture(body: u8) -> RequestCapture {
        RequestCapture {
            method: "POST".into(),
            route: "/v1/chat/completions".into(),
            headers: vec![("content-type".into(), b"application/json".to_vec())],
            body: vec![body],
        }
    }

    #[test]
    fn bounded_store_retains_arrival_order_and_can_be_cleared() {
        let store = RequestCaptureStore::new(2);
        store.push(capture(1));
        store.push(capture(2));
        store.push(capture(3));

        assert_eq!(
            store
                .snapshot()
                .iter()
                .map(|capture| capture.body[0])
                .collect::<Vec<_>>(),
            [2, 3]
        );
        assert_eq!(
            store.snapshot()[0].header("Content-Type"),
            Some(b"application/json".as_slice())
        );
        store.clear();
        assert!(store.snapshot().is_empty());
    }
}
