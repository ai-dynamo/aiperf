// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The transport facade tying URL/header building to the request path and the
//! connection pool. Port of `AioHttpTransport.send_request`.

use std::rc::Rc;

use bytes::Bytes;

use aiperf_clock::Clock;

use crate::client::cancellation::{CancelOutcome, race_cancel};
use crate::client::http_client::HttpClient;
use crate::client::pool::ConnectionPool;
use crate::config::ClientConfig;
use crate::models::{
    ConnectionReuseStrategy, ErrorDetails, RequestConfig, RequestRecord, TraceData,
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
        let body = Bytes::from(serde_json::to_vec(&payload).unwrap_or_default());
        let body_len = body.len();
        let reuse = cfg.reuse;
        let corr = cfg.correlation_id.as_deref();

        let mut record = RequestRecord::started(start_ns);
        let mut trace = TraceData::default();

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
                .dispatch(
                    &mut sender,
                    &url,
                    &headers,
                    body,
                    streaming,
                    &mut trace,
                    &mut record,
                    &mut on_first_token,
                    body_len,
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
                match race_cancel(self.clock.clone(), cancel_after, dispatch).await {
                    CancelOutcome::Completed(res) => res,
                    CancelOutcome::Cancelled => {
                        let now = self.clock.now_ns();
                        record.cancellation_ns = Some(now);
                        if let (ConnectionReuseStrategy::StickyUserSessions, Some(c)) =
                            (reuse, corr)
                        {
                            self.pool.release(c);
                        }
                        Err(ErrorDetails::cancelled(format!(
                            "Request cancelled {cancel_after}ns after being sent"
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
