// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The transport facade tying URL/header building to the request path and the
//! connection pool. Port of `AioHttpTransport.send_request`.

use std::rc::Rc;

use bytes::Bytes;
use http::Method;

use aiperf_clock::Clock;

use crate::client::cancellation::{CancelOutcome, race_cancel_after_send};
use crate::client::connection::SendCompletion;
use crate::client::http_client::HttpClient;
use crate::client::pool::ConnectionPool;
use crate::config::ClientConfig;
use crate::models::{
    ConnectionReuseStrategy, ErrorDetails, RequestConfig, RequestRecord, SseMessage, TraceData,
};
use crate::transport::headers::build_headers;
use crate::transport::url::build_url;

pub struct HttpTransport {
    clock: Rc<dyn Clock>,
    client: HttpClient,
    client_cfg: ClientConfig,
    pool: ConnectionPool,
    user_agent: String,
    session_header: Option<String>,
}

impl HttpTransport {
    pub fn new(clock: Rc<dyn Clock>, cfg: ClientConfig) -> Self {
        Self {
            client: HttpClient::new(clock.clone(), cfg.clone()),
            client_cfg: cfg,
            clock,
            pool: ConnectionPool::new(),
            user_agent: "aiperf-transport/0".to_string(),
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
    /// This is the live prefill-release contract ported from
    /// `src/aiperf/transports/aiohttp_client.py:210-224`: role-only, usage-only,
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
        self.send_body(cfg, Method::POST, body, streaming, first_token_filter)
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
        self.send_body(cfg, Method::POST, body, streaming, first_token_filter)
            .await
    }

    /// Send a non-streaming GET request through the same Clock-injected client
    /// and connection pool. This is intended for control-plane inputs such as
    /// public benchmark datasets; inference dispatch remains [`send_request`](Self::send_request).
    pub async fn get(&self, cfg: &RequestConfig) -> RequestRecord {
        self.send_body(cfg, Method::GET, Bytes::new(), false, |_, _| true)
            .await
    }

    async fn send_body(
        &self,
        cfg: &RequestConfig,
        method: Method,
        body: Bytes,
        streaming: bool,
        mut first_token_filter: impl FnMut(i64, &SseMessage) -> bool,
    ) -> RequestRecord {
        let start_ns = self.clock.now_ns();
        let full = match build_url(&cfg.url, "", &cfg.params) {
            Ok(f) => f,
            Err(e) => {
                let mut r = RequestRecord::started(start_ns);
                r.error = Some(ErrorDetails::other(format!("bad url {}: {e}", cfg.url)));
                r.end_ns = Some(self.clock.now_ns());
                return r;
            }
        };
        let headers = build_headers(
            cfg,
            streaming,
            self.session_header.as_deref(),
            &self.user_agent,
        );
        let url = match url::Url::parse(&full) {
            Ok(u) => u,
            Err(e) => {
                let mut r = RequestRecord::started(start_ns);
                r.error = Some(ErrorDetails::other(format!("bad url {full}: {e}")));
                r.end_ns = Some(self.clock.now_ns());
                return r;
            }
        };
        let body_len = body.len();
        let reuse = cfg.reuse;
        let corr = cfg.correlation_id.as_deref();

        let mut record = RequestRecord::started(start_ns);
        let mut trace = TraceData::default();
        let send_completion = Rc::new(SendCompletion::new());
        let completion_for_dispatch = send_completion.clone();
        let completion_for_record = send_completion.clone();

        // Acquire a connection per the reuse strategy, then dispatch on it.
        let dispatch = async {
            let mut sender = self
                .pool
                .acquire(
                    &url,
                    &self.client_cfg,
                    self.clock.clone(),
                    reuse,
                    corr,
                    &mut trace,
                )
                .await?;
            let res = self
                .client
                .dispatch_with_method_and_completion(
                    method,
                    &mut sender,
                    &url,
                    &headers,
                    body,
                    streaming,
                    &mut trace,
                    &mut record,
                    &mut first_token_filter,
                    body_len,
                    completion_for_dispatch,
                )
                .await;
            // On success, decide whether the connection is returned to the pool.
            if res.is_ok() {
                let keep = match reuse {
                    ConnectionReuseStrategy::StickyUserSessions => !cfg.is_final_turn,
                    _ => true,
                };
                if keep {
                    self.pool.put(&url, corr, reuse, sender);
                } else if let (ConnectionReuseStrategy::StickyUserSessions, Some(c)) = (reuse, corr)
                {
                    self.pool.release(c);
                }
            } else if let (ConnectionReuseStrategy::StickyUserSessions, Some(c)) = (reuse, corr) {
                // Failed connections are never reused; drop the sticky lease.
                self.pool.release(c);
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
                        if let Some(sent_ns) = completion_for_record.sent_ns() {
                            trace.request_send_end_ns = Some(sent_ns);
                            trace.request_headers_sent_ns = Some(sent_ns);
                            trace.request_bytes_total = body_len as u64;
                            trace.request_chunks_count = 1;
                        }
                        if let (ConnectionReuseStrategy::StickyUserSessions, Some(c)) =
                            (reuse, corr)
                        {
                            self.pool.release(c);
                        }
                        Err(ErrorDetails::cancelled(format!(
                            "RequestCancellationError: request cancelled {cancel_after}ns after being sent"
                        )))
                    }
                }
            }
            None => dispatch.await,
        };

        if let Err(e) = result {
            trace.error_timestamp_ns = Some(self.clock.now_ns());
            record.error = Some(e);
        }
        record.end_ns = Some(self.clock.now_ns());
        record.trace = Some(trace);
        record
    }
}
