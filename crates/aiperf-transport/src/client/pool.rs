// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connection-reuse strategies. Port of `ConnectionReuseStrategy` handling +
//! `ConnectionLeaseManager`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use url::Url;

use aiperf_clock::Clock;

use crate::client::connection::{Sender, establish};
use crate::config::ClientConfig;
use crate::models::{ConnectionReuseStrategy, ErrorDetails, TraceData};

/// A single-threaded store of live senders keyed by a reuse key.
#[derive(Default)]
pub struct ConnectionPool {
    /// Shared pool (one live sender per origin) for `Pooled`.
    pooled: RefCell<HashMap<String, Sender>>,
    /// Per-session leases for `StickyUserSessions`.
    sticky: RefCell<HashMap<String, Sender>>,
}

fn origin_key(url: &Url) -> String {
    format!(
        "{}://{}:{}",
        url.scheme(),
        url.host_str().unwrap_or(""),
        url.port_or_known_default().unwrap_or(0)
    )
}

impl ConnectionPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Acquire a ready sender per the reuse strategy, establishing a new
    /// connection when needed and recording connect vs reuse timings.
    pub async fn acquire(
        &self,
        url: &Url,
        cfg: &ClientConfig,
        clock: Rc<dyn Clock>,
        reuse: ConnectionReuseStrategy,
        correlation_id: Option<&str>,
        trace: &mut TraceData,
    ) -> Result<Sender, ErrorDetails> {
        match reuse {
            ConnectionReuseStrategy::Never => {
                let (s, _) = establish(url, cfg, clock, trace).await?;
                Ok(s)
            }
            ConnectionReuseStrategy::Pooled => {
                let key = origin_key(url);
                if let Some(s) = self.take_ready(&self.pooled, &key) {
                    trace.connection_reused_ns = Some(clock.now_ns());
                    return Ok(s);
                }
                let (s, _) = establish(url, cfg, clock, trace).await?;
                Ok(s)
            }
            ConnectionReuseStrategy::StickyUserSessions => {
                let key = correlation_id
                    .ok_or_else(|| ErrorDetails::other("sticky strategy requires correlation_id"))?
                    .to_string();
                if let Some(s) = self.take_ready(&self.sticky, &key) {
                    trace.connection_reused_ns = Some(clock.now_ns());
                    return Ok(s);
                }
                let (s, _) = establish(url, cfg, clock, trace).await?;
                Ok(s)
            }
        }
    }

    fn take_ready(&self, map: &RefCell<HashMap<String, Sender>>, key: &str) -> Option<Sender> {
        let mut m = map.borrow_mut();
        match m.remove(key) {
            Some(s) if s.is_ready() => Some(s),
            _ => None,
        }
    }

    /// Return a still-usable sender to the pool for future reuse. Only ready
    /// connections are pooled: a streaming response left un-drained (stopped at
    /// `[DONE]`) is not ready and is correctly dropped rather than reused.
    pub fn put(
        &self,
        url: &Url,
        correlation_id: Option<&str>,
        reuse: ConnectionReuseStrategy,
        sender: Sender,
    ) {
        if !sender.is_ready() {
            return;
        }
        match reuse {
            ConnectionReuseStrategy::Never => { /* drop: closes on task exit */ }
            ConnectionReuseStrategy::Pooled => {
                self.pooled.borrow_mut().insert(origin_key(url), sender);
            }
            ConnectionReuseStrategy::StickyUserSessions => {
                if let Some(cid) = correlation_id {
                    self.sticky.borrow_mut().insert(cid.to_string(), sender);
                }
            }
        }
    }

    /// Release (drop) a sticky lease — call on final turn / cancellation / error.
    pub fn release(&self, correlation_id: &str) {
        self.sticky.borrow_mut().remove(correlation_id);
    }
}
