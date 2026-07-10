// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The transport facade tying URL/header building to the request path and the
//! connection pool. Port of `AioHttpTransport.send_request`.

use std::rc::Rc;

use bytes::Bytes;

use aiperf_clock::Clock;

use crate::client::http_client::HttpClient;
use crate::config::ClientConfig;
use crate::models::{RequestConfig, RequestRecord};
use crate::transport::headers::build_headers;
use crate::transport::url::build_url;

pub struct HttpTransport {
    clock: Rc<dyn Clock>,
    client: HttpClient,
    user_agent: String,
    session_header: Option<String>,
}

impl HttpTransport {
    pub fn new(clock: Rc<dyn Clock>, cfg: ClientConfig) -> Self {
        Self {
            client: HttpClient::new(clock.clone(), cfg),
            clock,
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

    /// Build and send a request from a [`RequestConfig`] + JSON payload.
    pub async fn send_request(
        &self,
        cfg: &RequestConfig,
        payload: serde_json::Value,
        streaming: bool,
        on_first_token: impl FnMut(i64),
    ) -> RequestRecord {
        // cfg.url is treated as the full base+path already (the caller may also
        // pass a bare base; build_url dedups). Endpoint sub-path is empty here
        // because the OpenAI path is part of cfg.url in this facade.
        let full = build_url(&cfg.url, "", &cfg.params);
        let headers = build_headers(
            cfg,
            streaming,
            self.session_header.as_deref(),
            &self.user_agent,
        );
        let url = match url::Url::parse(&full) {
            Ok(u) => u,
            Err(e) => {
                let mut r = RequestRecord::started(self.clock.now_ns());
                r.error = Some(crate::models::ErrorDetails::other(format!(
                    "bad url {full}: {e}"
                )));
                return r;
            }
        };
        let body = Bytes::from(serde_json::to_vec(&payload).unwrap_or_default());

        if let Some(cancel_after) = cfg.cancel_after_ns {
            self.client
                .request_cancellable(
                    &url,
                    &headers,
                    body,
                    streaming,
                    cancel_after,
                    on_first_token,
                )
                .await
        } else {
            self.client
                .request(&url, &headers, body, streaming, on_first_token)
                .await
        }
    }
}
