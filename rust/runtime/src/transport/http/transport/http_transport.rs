// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The transport facade tying URL/header building to the request path and the
//! connection pool.

use std::rc::Rc;

use bytes::Bytes;
use http::Method;

use crate::clock::Clock;

use crate::transport::core::SseMessage;
use crate::transport::core::{ConnectionReuseStrategy, ErrorDetails, RequestRecord, TraceData};
use crate::transport::http::client::cancellation::{CancelOutcome, race_cancel_after_send};
use crate::transport::http::client::connection::{SendCompletion, with_timeout};
use crate::transport::http::client::http_client::{
    HttpClient, SseMessageFilter, SynchronousSseMessageFilter,
};
use crate::transport::http::client::pool::{ConnectionManager, ConnectionPool};
use crate::transport::http::config::ClientConfig;
use crate::transport::http::models::RequestConfig;
use crate::transport::http::transport::headers::{
    build_headers, dynamo_session_id_from_correlation_id_enabled,
};
use crate::transport::http::transport::url::build_url;

pub struct HttpTransport {
    clock: Rc<dyn Clock>,
    client: HttpClient,
    client_cfg: ClientConfig,
    connections: Rc<dyn ConnectionManager>,
    user_agent: String,
    session_header: Option<String>,
    dynamo_session_id_from_correlation_id: bool,
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
            dynamo_session_id_from_correlation_id: dynamo_session_id_from_correlation_id_enabled(),
        }
    }

    /// Override the correlation-id header name.
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
    /// reserializing it. Graph dispatch uses this path to preserve preassembled
    /// wire bytes and avoid a hot-path JSON tree.
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
            self.dynamo_session_id_from_correlation_id,
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
            // Request timing includes pool queueing and connection creation.
            request_send_start_ns: Some(start_ns),
            ..TraceData::default()
        };
        let send_completion = Rc::new(SendCompletion::new());
        let completion_for_dispatch = send_completion.clone();
        let completion_for_record = send_completion.clone();

        let dispatch = async {
            let acquire_remaining_ns = remaining_timeout(deadline_ns, self.clock.now_ns())?;
            // `connect_timeout_ns` bounds each *attempt* inside the connection
            // manager's retry loop; the outer acquisition cap must therefore
            // cover every attempt plus the linear backoff between them, or a
            // single-attempt cap would short-circuit the retries. With the
            // default zero retries this budget collapses back to
            // `connect_timeout_ns`, leaving established behavior unchanged.
            let acquire_timeout_ns = minimum_timeout(
                connect_acquire_budget_ns(&self.client_cfg),
                acquire_remaining_ns,
            );
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
                    kind: crate::transport::core::ErrorKind::Timeout,
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
            if e.kind == crate::transport::core::ErrorKind::Timeout
                && let Some(total) = total_timeout_ns
            {
                e.message = format!("request timeout after {total}ns");
            }
            if e.kind == crate::transport::core::ErrorKind::Timeout
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

/// Total connection-acquisition budget covering every connect attempt plus the
/// linear backoff waited between them.
///
/// Each attempt is separately bounded by `connect_timeout_ns` inside the
/// connection manager's retry loop; this outer budget must span all
/// `max_connect_retries + 1` attempts so the acquisition cap does not truncate
/// the retry sequence. Returns `None` when no per-attempt connect deadline is
/// set (the unbounded connect hot path), leaving only the total-request
/// deadline to bound acquisition. With the default zero retries the result is
/// exactly `connect_timeout_ns`.
fn connect_acquire_budget_ns(cfg: &ClientConfig) -> Option<i64> {
    let per_attempt_ns = positive_timeout(cfg.connect_timeout_ns)?;
    let attempts = i64::from(cfg.max_connect_retries) + 1;
    // Linear backoff sums to `backoff * (1 + 2 + ... + retries)`.
    let retries = i64::from(cfg.max_connect_retries);
    let backoff_total_ns = if cfg.connect_retry_backoff_ns > 0 {
        cfg.connect_retry_backoff_ns
            .saturating_mul(retries.saturating_mul(retries + 1) / 2)
    } else {
        0
    };
    Some(
        per_attempt_ns
            .saturating_mul(attempts)
            .saturating_add(backoff_total_ns),
    )
}

fn remaining_timeout(deadline_ns: Option<i64>, now_ns: i64) -> Result<Option<i64>, ErrorDetails> {
    let Some(deadline_ns) = deadline_ns else {
        return Ok(None);
    };
    let remaining = deadline_ns.saturating_sub(now_ns);
    if remaining <= 0 {
        return Err(ErrorDetails {
            kind: crate::transport::core::ErrorKind::Timeout,
            code: None,
            message: "request deadline elapsed before HTTP dispatch".to_string(),
        });
    }
    Ok(Some(remaining))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connect_budget_defaults_to_connect_timeout_without_retries() {
        // With the default zero retries the acquisition budget must equal
        // `connect_timeout_ns` exactly, so established single-attempt behavior
        // is preserved.
        let cfg = ClientConfig {
            connect_timeout_ns: Some(2_000_000),
            ..ClientConfig::default()
        };
        assert_eq!(connect_acquire_budget_ns(&cfg), Some(2_000_000));
    }

    #[test]
    fn connect_budget_spans_all_attempts_and_backoff() {
        // The outer acquisition cap must cover every attempt plus the linear
        // backoff, otherwise the per-attempt `connect_timeout_ns` would
        // truncate the retry sequence. 3 attempts * 1_000_000 per attempt +
        // backoff (500_000 * (1 + 2)) = 3_000_000 + 1_500_000.
        let cfg = ClientConfig {
            connect_timeout_ns: Some(1_000_000),
            max_connect_retries: 2,
            connect_retry_backoff_ns: 500_000,
            ..ClientConfig::default()
        };
        assert_eq!(connect_acquire_budget_ns(&cfg), Some(4_500_000));
    }

    #[test]
    fn connect_budget_is_none_without_a_connect_deadline() {
        // No per-attempt connect deadline leaves acquisition bounded only by the
        // total-request deadline; retries still fire (unbounded per attempt).
        let cfg = ClientConfig {
            connect_timeout_ns: None,
            max_connect_retries: 3,
            connect_retry_backoff_ns: 1_000,
            ..ClientConfig::default()
        };
        assert_eq!(connect_acquire_budget_ns(&cfg), None);
    }
}
