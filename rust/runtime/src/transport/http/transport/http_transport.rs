// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The transport facade tying URL/header building to the request path and the
//! connection pool.

use std::rc::Rc;

use bytes::Bytes;
use http::{HeaderMap, HeaderName, HeaderValue, Method};

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
    x_session_id_from_correlation_id_enabled, x_smg_routing_key_from_correlation_id_enabled,
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
    x_session_id_from_correlation_id: bool,
    x_smg_routing_key_from_correlation_id: bool,
    capture_raw: bool,
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
            x_session_id_from_correlation_id: x_session_id_from_correlation_id_enabled(),
            x_smg_routing_key_from_correlation_id: x_smg_routing_key_from_correlation_id_enabled(),
            capture_raw: true,
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

    /// Select whether request headers are retained for a raw HTTP artifact.
    pub fn with_raw_capture(mut self, capture_raw: bool) -> Self {
        self.capture_raw = capture_raw;
        self
    }

    fn prepared_headers(
        &self,
        cfg: &RequestConfig,
        static_headers: &HeaderMap,
        streaming: bool,
    ) -> Result<HeaderMap, ErrorDetails> {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::USER_AGENT,
            header_value(&self.user_agent, "User-Agent")?,
        );
        if let Some(request_id) = cfg.request_id.as_deref() {
            headers.insert(
                HeaderName::from_static("x-request-id"),
                header_value(request_id, "X-Request-ID")?,
            );
        }
        if let Some(correlation_id) = cfg.correlation_id.as_deref() {
            let name = self.session_header.as_deref().unwrap_or("X-Correlation-ID");
            headers.insert(
                HeaderName::try_from(name).map_err(|error| {
                    ErrorDetails::other(format!("invalid session header {name:?}: {error}"))
                })?,
                header_value(correlation_id, name)?,
            );
        }
        headers.extend(static_headers.clone());
        headers.insert(
            http::header::ACCEPT,
            HeaderValue::from_static(if streaming {
                "text/event-stream"
            } else {
                "application/json"
            }),
        );
        headers
            .entry(http::header::CONTENT_TYPE)
            .or_insert(HeaderValue::from_static("application/json"));
        if let Some(correlation_id) = cfg.correlation_id.as_deref() {
            headers.insert(
                HeaderName::from_static("x-session-affinity"),
                header_value(correlation_id, "X-Session-Affinity")?,
            );
            if self.x_session_id_from_correlation_id {
                headers.insert(
                    HeaderName::from_static("x-session-id"),
                    header_value(correlation_id, "X-Session-ID")?,
                );
            }
            if self.x_smg_routing_key_from_correlation_id {
                headers.insert(
                    HeaderName::from_static("x-smg-routing-key"),
                    header_value(correlation_id, "X-SMG-Routing-Key")?,
                );
            }
            if self.dynamo_session_id_from_correlation_id {
                headers.insert(
                    HeaderName::from_static("x-dynamo-session-id"),
                    header_value(correlation_id, "X-Dynamo-Session-ID")?,
                );
                if let Some(parent) = cfg.parent_correlation_id.as_deref() {
                    headers.insert(
                        HeaderName::from_static("x-dynamo-parent-session-id"),
                        header_value(parent, "X-Dynamo-Parent-Session-ID")?,
                    );
                }
            }
        }
        Ok(headers)
    }

    fn artifact_request_headers(
        &self,
        headers: &HeaderMap,
    ) -> std::collections::BTreeMap<String, String> {
        self.capture_raw
            .then(|| header_record(headers))
            .unwrap_or_default()
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
        self.send_body(
            cfg,
            Method::POST,
            body,
            streaming,
            &mut first_token_filter,
            None,
        )
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
        self.send_body(
            cfg,
            Method::POST,
            body,
            streaming,
            &mut first_token_filter,
            None,
        )
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
        self.send_body(cfg, Method::POST, body, streaming, first_token_filter, None)
            .await
    }

    /// Dispatch a request whose endpoint URL and endpoint-owned headers were
    /// parsed during request preparation.
    pub async fn send_prepared_request_bytes_with_first_token_filter(
        &self,
        cfg: &RequestConfig,
        url: &url::Url,
        static_headers: &HeaderMap,
        body: Bytes,
        streaming: bool,
        first_token_filter: impl FnMut(i64, &SseMessage) -> bool,
    ) -> RequestRecord {
        let mut first_token_filter = SynchronousSseMessageFilter::new(first_token_filter);
        let headers = match self.prepared_headers(cfg, static_headers, streaming) {
            Ok(headers) => headers,
            Err(error) => return request_error(self.clock.now_ns(), self.clock.now_ns(), error),
        };
        self.send_body(
            cfg,
            Method::POST,
            body,
            streaming,
            &mut first_token_filter,
            Some((url, &headers)),
        )
        .await
    }

    /// Dispatch a prepared request through a backpressured SSE filter.
    pub async fn send_prepared_request_bytes_with_sse_filter(
        &self,
        cfg: &RequestConfig,
        url: &url::Url,
        static_headers: &HeaderMap,
        body: Bytes,
        streaming: bool,
        first_token_filter: &mut impl SseMessageFilter,
    ) -> RequestRecord {
        let headers = match self.prepared_headers(cfg, static_headers, streaming) {
            Ok(headers) => headers,
            Err(error) => return request_error(self.clock.now_ns(), self.clock.now_ns(), error),
        };
        self.send_body(
            cfg,
            Method::POST,
            body,
            streaming,
            first_token_filter,
            Some((url, &headers)),
        )
        .await
    }

    /// Send one streaming request without creating a terminal request record.
    ///
    /// Bounded decision consumers use this narrow path to admit each decoded
    /// frame before any response/raw-record accumulation exists. Requests with
    /// a post-send cancellation policy fail closed until this no-record path
    /// has an equivalent cancellation lifecycle.
    pub async fn send_request_bytes_streaming(
        &self,
        cfg: &RequestConfig,
        body: Bytes,
        max_sse_frame_bytes: usize,
        on_first_token: &mut dyn FnMut(i64),
        on_message: &mut dyn FnMut(&SseMessage) -> Result<bool, ErrorDetails>,
    ) -> Result<u16, ErrorDetails> {
        if cfg.cancel_after_ns.is_some() {
            return Err(ErrorDetails::cancelled(
                "bounded decision dispatch does not support post-send cancellation",
            ));
        }

        let start_ns = self.clock.now_ns();
        let headers = header_map(build_headers(
            cfg,
            true,
            self.session_header.as_deref(),
            &self.user_agent,
            self.dynamo_session_id_from_correlation_id,
            self.x_session_id_from_correlation_id,
            self.x_smg_routing_key_from_correlation_id,
        ))?;
        let full = build_url(&cfg.url, "", &cfg.params)
            .map_err(|error| ErrorDetails::other(format!("bad url {}: {error}", cfg.url)))?;
        let url = url::Url::parse(&full)
            .map_err(|error| ErrorDetails::other(format!("bad url {full}: {error}")))?;
        let reuse = cfg.reuse;
        let correlation_id = cfg.correlation_id.as_deref();
        let total_timeout_ns = positive_timeout(self.client_cfg.total_timeout_ns);
        let deadline_ns = total_timeout_ns.map(|timeout| start_ns.saturating_add(timeout));
        let mut trace = TraceData {
            request_send_start_ns: Some(start_ns),
            ..TraceData::default()
        };

        let result = async {
            let acquire_remaining_ns = remaining_timeout(deadline_ns, self.clock.now_ns())?;
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
                    correlation_id,
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
            let code = with_timeout(
                self.clock.clone(),
                dispatch_timeout_ns,
                self.client.dispatch_bounded_streaming_with_handler(
                    lease.sender_mut(),
                    &url,
                    &headers,
                    body,
                    max_sse_frame_bytes,
                    on_first_token,
                    on_message,
                ),
                || ErrorDetails {
                    kind: crate::transport::core::ErrorKind::Timeout,
                    code: None,
                    message: "bounded decision request timed out".to_string(),
                },
            )
            .await?;
            let keep = match reuse {
                ConnectionReuseStrategy::StickyUserSessions => !cfg.is_final_turn,
                _ => true,
            };
            if keep {
                lease.mark_reusable();
            } else if let (ConnectionReuseStrategy::StickyUserSessions, Some(correlation_id)) =
                (reuse, correlation_id)
            {
                self.connections.release_session(correlation_id);
            }
            Ok(code)
        }
        .await;

        if result.is_err()
            && let (ConnectionReuseStrategy::StickyUserSessions, Some(correlation_id)) =
                (reuse, correlation_id)
        {
            self.connections.release_session(correlation_id);
        }
        result
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
            None,
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
        prepared: Option<(&url::Url, &HeaderMap)>,
    ) -> RequestRecord {
        let start_ns = self.clock.now_ns();
        let (url, headers) = match prepared {
            Some((url, headers)) => (url.clone(), headers.clone()),
            None => {
                let headers = match header_map(build_headers(
                    cfg,
                    streaming,
                    self.session_header.as_deref(),
                    &self.user_agent,
                    self.dynamo_session_id_from_correlation_id,
                    self.x_session_id_from_correlation_id,
                    self.x_smg_routing_key_from_correlation_id,
                )) {
                    Ok(headers) => headers,
                    Err(error) => return request_error(start_ns, self.clock.now_ns(), error),
                };
                let full = match build_url(&cfg.url, "", &cfg.params) {
                    Ok(full) => full,
                    Err(error) => {
                        return request_error(
                            start_ns,
                            self.clock.now_ns(),
                            ErrorDetails::other(format!("bad url {}: {error}", cfg.url)),
                        );
                    }
                };
                match url::Url::parse(&full) {
                    Ok(url) => (url, headers),
                    Err(error) => {
                        return request_error(
                            start_ns,
                            self.clock.now_ns(),
                            ErrorDetails::other(format!("bad url {full}: {error}")),
                        );
                    }
                }
            }
        };
        let mut record = RequestRecord {
            request_headers: self.artifact_request_headers(&headers),
            ..RequestRecord::started(start_ns)
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

fn header_value(value: &str, name: &str) -> Result<HeaderValue, ErrorDetails> {
    HeaderValue::try_from(value).map_err(|error| {
        ErrorDetails::other(format!("invalid request header value for {name}: {error}"))
    })
}

fn header_map(
    headers: std::collections::BTreeMap<String, String>,
) -> Result<HeaderMap, ErrorDetails> {
    headers
        .into_iter()
        .map(|(name, value)| {
            let parsed_name = HeaderName::try_from(name.as_str()).map_err(|error| {
                ErrorDetails::other(format!("invalid request header name {name:?}: {error}"))
            })?;
            Ok((parsed_name, header_value(&value, &name)?))
        })
        .collect()
}

fn header_record(headers: &HeaderMap) -> std::collections::BTreeMap<String, String> {
    headers
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_string(),
                String::from_utf8_lossy(value.as_bytes()).into_owned(),
            )
        })
        .collect()
}

fn request_error(start_ns: i64, end_ns: i64, error: ErrorDetails) -> RequestRecord {
    let mut record = RequestRecord::started(start_ns);
    record.error = Some(error);
    record.end_ns = Some(end_ns);
    record
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
    fn raw_capture_gate_skips_request_header_artifact_map() {
        let transport = HttpTransport::new(
            Rc::new(crate::clock::SimClock::new()),
            ClientConfig::default(),
        )
        .with_raw_capture(false);
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-static",
            HeaderValue::from_static("retained-only-for-raw"),
        );

        assert!(transport.artifact_request_headers(&headers).is_empty());
    }

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
