// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The transport facade tying URL/header building to the request path and the
//! connection pool. Port of `AioHttpTransport.send_request`.

use std::rc::Rc;

use bytes::Bytes;
use http::Method;

use crate::clock::Clock;

use crate::transport::http::client::cancellation::{CancelOutcome, race_cancel_after_send};
use crate::transport::http::client::connection::{SendCompletion, with_timeout};
use crate::transport::http::client::http_client::{
    HttpClient, SseMessageFilter, SynchronousSseMessageFilter,
};
use crate::transport::http::client::pool::{ConnectionManager, ConnectionPool};
use crate::transport::http::config::ClientConfig;
use crate::transport::http::models::{
    ConnectionReuseStrategy, ErrorDetails, RequestConfig, RequestRecord, SseMessage, TraceData,
};
use crate::transport::http::transport::headers::build_headers;
use crate::transport::http::transport::url::build_url;

pub struct HttpTransport {
    clock: Rc<dyn Clock>,
    client: HttpClient,
    client_cfg: ClientConfig,
    connections: Rc<dyn ConnectionManager>,
    user_agent: String,
    session_header: Option<String>,
}

impl HttpTransport {
    pub fn new(clock: Rc<dyn Clock>, cfg: ClientConfig) -> Self {
        Self::with_connection_manager(clock, cfg, Rc::new(ConnectionPool::new()))
    }

    /// Build over an injected connection-management policy.
    pub fn with_connection_manager(
        clock: Rc<dyn Clock>,
        mut cfg: ClientConfig,
        connections: Rc<dyn ConnectionManager>,
    ) -> Self {
        if let Some(total) = positive_timeout(cfg.total_timeout_ns) {
            cfg.connect_timeout_ns = minimum_timeout(cfg.connect_timeout_ns, Some(total));
        }
        Self {
            client: HttpClient::new(clock.clone(), cfg.clone()),
            client_cfg: cfg,
            clock,
            connections,
            user_agent: "aiperf-transport-http/0".to_string(),
            session_header: None,
        }
    }

    /// Override the correlation-id header name (Python `session_header`).
    pub fn with_session_header(mut self, name: impl Into<String>) -> Self {
        self.session_header = Some(name.into());
        self
    }

    /// Override the User-Agent string.
    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }

    /// Build and send a request from a [`RequestConfig`] + JSON payload,
    /// honoring `cfg.reuse` via the connection pool: `Pooled` reuses a shared
    /// origin connection, `StickyUserSessions` keeps one connection per
    /// correlation id (released on the final turn), `Never` opens a fresh one.
    pub async fn send_request(
        &self,
        cfg: &RequestConfig,
        payload: serde_json::Value,
        streaming: bool,
        mut on_first_token: impl FnMut(i64),
    ) -> RequestRecord {
        self.send_request_with_first_token_filter(
            cfg,
            payload,
            streaming,
            move |ttft_ns, _message| {
                on_first_token(ttft_ns);
                true
            },
        )
        .await
    }

    /// Send JSON while inspecting successive SSE messages until the callback
    /// returns `true` for meaningful first-token content.
    ///
    /// This is the live prefill-release contract: role-only, usage-only,
    /// or otherwise non-token messages return `false`, so the callback is tried
    /// again on the next SSE message. The ordinary [`send_request`](Self::send_request)
    /// wrapper preserves its first-message callback behavior.
    pub async fn send_request_with_first_token_filter(
        &self,
        cfg: &RequestConfig,
        payload: serde_json::Value,
        streaming: bool,
        first_token_filter: impl FnMut(i64, &SseMessage) -> bool,
    ) -> RequestRecord {
        // Serialize the JSON payload. On failure, return an error record rather
        // than silently sending an empty body (mirrors the bad-url handling).
        let body = match serde_json::to_vec(&payload) {
            Ok(b) => Bytes::from(b),
            Err(e) => {
                let start_ns = self.clock.now_ns();
                let mut r = RequestRecord::started(start_ns);
                r.error = Some(ErrorDetails::other(format!("serialize payload: {e}")));
                r.end_ns = Some(self.clock.now_ns());
                return r;
            }
        };
        let mut first_token_filter = SynchronousSseMessageFilter::new(first_token_filter);
        self.send_body(cfg, Method::POST, body, streaming, &mut first_token_filter)
            .await
    }

    /// Send an already-serialized JSON request body without decoding or
    /// reserializing it. Dataset raw replay and segment-slice materialization use
    /// this path to preserve authored bytes and avoid a hot-path JSON tree.
    pub async fn send_request_bytes(
        &self,
        cfg: &RequestConfig,
        body: Bytes,
        streaming: bool,
        mut on_first_token: impl FnMut(i64),
    ) -> RequestRecord {
        self.send_request_bytes_with_first_token_filter(
            cfg,
            body,
            streaming,
            move |ttft_ns, _message| {
                on_first_token(ttft_ns);
                true
            },
        )
        .await
    }

    /// Send an already-serialized JSON request body while inspecting successive
    /// SSE messages until `first_token_filter` accepts meaningful content.
    ///
    /// Dataset-backed dispatch uses this entry point so preformatted request
    /// bytes are never decoded and serialized again merely to retain the
    /// first-token admission hook.
    pub async fn send_request_bytes_with_first_token_filter(
        &self,
        cfg: &RequestConfig,
        body: Bytes,
        streaming: bool,
        first_token_filter: impl FnMut(i64, &SseMessage) -> bool,
    ) -> RequestRecord {
        let mut first_token_filter = SynchronousSseMessageFilter::new(first_token_filter);
        self.send_body(cfg, Method::POST, body, streaming, &mut first_token_filter)
            .await
    }

    /// Send serialized JSON while awaiting a backpressured SSE response filter.
    pub async fn send_request_bytes_with_sse_filter(
        &self,
        cfg: &RequestConfig,
        body: Bytes,
        streaming: bool,
        first_token_filter: &mut impl SseMessageFilter,
    ) -> RequestRecord {
        self.send_body(cfg, Method::POST, body, streaming, first_token_filter)
            .await
    }

    /// Send a non-streaming GET request through the same Clock-injected client
    /// and connection pool. This is intended for control-plane inputs such as
    /// public benchmark datasets; inference dispatch remains [`send_request`](Self::send_request).
    pub async fn get(&self, cfg: &RequestConfig) -> RequestRecord {
        let mut first_token_filter =
            SynchronousSseMessageFilter::new(|_: i64, _: &SseMessage| true);
        self.send_body(
            cfg,
            Method::GET,
            Bytes::new(),
            false,
            &mut first_token_filter,
        )
        .await
    }

    async fn send_body(
        &self,
        cfg: &RequestConfig,
        method: Method,
        body: Bytes,
        streaming: bool,
        first_token_filter: &mut impl SseMessageFilter,
    ) -> RequestRecord {
        let start_ns = self.clock.now_ns();
        let headers = build_headers(
            cfg,
            streaming,
            self.session_header.as_deref(),
            &self.user_agent,
        );
        let mut record = RequestRecord {
            request_body: body.clone(),
            request_headers: headers.clone(),
            ..RequestRecord::started(start_ns)
        };
        let full = match build_url(&cfg.url, "", &cfg.params) {
            Ok(f) => f,
            Err(e) => {
                record.error = Some(ErrorDetails::other(format!("bad url {}: {e}", cfg.url)));
                record.end_ns = Some(self.clock.now_ns());
                return record;
            }
        };
        let url = match url::Url::parse(&full) {
            Ok(u) => u,
            Err(e) => {
                record.error = Some(ErrorDetails::other(format!("bad url {full}: {e}")));
                record.end_ns = Some(self.clock.now_ns());
                return record;
            }
        };
        let body_len = body.len();
        let reuse = cfg.reuse;
        let corr = cfg.correlation_id.as_deref();
        let total_timeout_ns = positive_timeout(self.client_cfg.total_timeout_ns);
        let deadline_ns = total_timeout_ns.map(|timeout| start_ns.saturating_add(timeout));

        let mut trace = TraceData {
            // Match aiohttp's request lifecycle: request start precedes pool
            // queueing/reuse/connection creation.
            request_send_start_ns: Some(start_ns),
            ..TraceData::default()
        };
        let send_completion = Rc::new(SendCompletion::new());
        let completion_for_dispatch = send_completion.clone();
        let completion_for_record = send_completion.clone();

        // Acquire a connection per the reuse strategy, then dispatch on it.
        let dispatch = async {
            let acquire_remaining_ns = remaining_timeout(deadline_ns, self.clock.now_ns())?;
            let acquire_timeout_ns =
                minimum_timeout(self.client_cfg.connect_timeout_ns, acquire_remaining_ns);
            let mut lease = with_timeout(
                self.clock.clone(),
                acquire_timeout_ns,
                self.connections.acquire(
                    &url,
                    &self.client_cfg,
                    self.clock.clone(),
                    reuse,
                    corr,
                    &mut trace,
                ),
                || ErrorDetails {
                    kind: crate::transport::http::models::ErrorKind::Timeout,
                    code: None,
                    message: format!(
                        "connection acquisition timeout after {}ns",
                        acquire_timeout_ns.unwrap_or_default()
                    ),
                },
            )
            .await?;
            let remaining_ns = remaining_timeout(deadline_ns, self.clock.now_ns())?;
            let dispatch_timeout_ns =
                minimum_timeout(self.client_cfg.request_timeout_ns, remaining_ns);
            let res = self
                .client
                .dispatch_with_method_and_completion_timeout(
                    method,
                    lease.sender_mut(),
                    &url,
                    &headers,
                    body,
                    streaming,
                    &mut trace,
                    &mut record,
                    first_token_filter,
                    body_len,
                    completion_for_dispatch,
                    dispatch_timeout_ns,
                )
                .await;
            // A successful fully-drained response makes an HTTP/1 lease reusable.
            // A non-2xx response whose body was fully drained is equally clean
            // (`reusable_connection`), so it is pooled too rather than forcing a
            // reconnect per 4xx/5xx. The lease itself owns cleanup, so
            // cancellation/error paths cannot leak pool capacity.
            if res.is_ok() || record.reusable_connection {
                let keep = match reuse {
                    ConnectionReuseStrategy::StickyUserSessions => !cfg.is_final_turn,
                    _ => true,
                };
                if keep {
                    lease.mark_reusable();
                } else if let (ConnectionReuseStrategy::StickyUserSessions, Some(c)) = (reuse, corr)
                {
                    self.connections.release_session(c);
                }
            } else if let (ConnectionReuseStrategy::StickyUserSessions, Some(c)) = (reuse, corr) {
                self.connections.release_session(c);
            }
            res
        };

        // Optional post-send cancellation.
        let result = match cfg.cancel_after_ns {
            Some(cancel_after) => {
                match race_cancel_after_send(
                    self.clock.clone(),
                    cancel_after,
                    send_completion,
                    dispatch,
                )
                .await
                {
                    CancelOutcome::Completed(res) => res,
                    CancelOutcome::Cancelled => {
                        let now = self.clock.now_ns();
                        record.cancellation_ns = Some(now);
                        if let Some(sent_ns) = completion_for_record.sent_ns()
                            && trace.request_send_end_ns.is_none()
                        {
                            trace.request_send_end_ns = Some(sent_ns);
                            trace.request_headers_sent_ns = completion_for_record.headers_ns();
                            trace.request_bytes_total = body_len as u64;
                            trace.request_chunks_count = 1;
                            if self.client_cfg.collect_trace_chunks {
                                trace.request_chunks.push((sent_ns, body_len as u64));
                            }
                        }
                        if let (ConnectionReuseStrategy::StickyUserSessions, Some(c)) =
                            (reuse, corr)
                        {
                            self.connections.release_session(c);
                        }
                        Err(ErrorDetails::cancelled(format!(
                            "RequestCancellationError: request cancelled {cancel_after}ns after being sent"
                        )))
                    }
                }
            }
            None => dispatch.await,
        };

        if let Err(mut e) = result {
            if e.kind == crate::transport::http::models::ErrorKind::Timeout
                && let Some(total) = total_timeout_ns
            {
                e.message = format!("request timeout after {total}ns");
            }
            if e.kind == crate::transport::http::models::ErrorKind::Timeout
                && let (ConnectionReuseStrategy::StickyUserSessions, Some(c)) = (reuse, corr)
            {
                self.connections.release_session(c);
            }
            trace.error_timestamp_ns = Some(self.clock.now_ns());
            record.error = Some(e);
        }
        record.end_ns = Some(self.clock.now_ns());
        record.trace = Some(trace);
        record
    }
}

fn positive_timeout(timeout_ns: Option<i64>) -> Option<i64> {
    timeout_ns.filter(|timeout| *timeout > 0)
}

fn minimum_timeout(first: Option<i64>, second: Option<i64>) -> Option<i64> {
    match (positive_timeout(first), positive_timeout(second)) {
        (Some(first), Some(second)) => Some(first.min(second)),
        (Some(timeout), None) | (None, Some(timeout)) => Some(timeout),
        (None, None) => None,
    }
}

fn remaining_timeout(deadline_ns: Option<i64>, now_ns: i64) -> Result<Option<i64>, ErrorDetails> {
    let Some(deadline_ns) = deadline_ns else {
        return Ok(None);
    };
    let remaining = deadline_ns.saturating_sub(now_ns);
    if remaining <= 0 {
        return Err(ErrorDetails {
            kind: crate::transport::http::models::ErrorKind::Timeout,
            code: None,
            message: "request deadline elapsed before HTTP dispatch".to_string(),
        });
    }
    Ok(Some(remaining))
}
