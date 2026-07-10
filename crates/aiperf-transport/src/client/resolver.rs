// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DNS resolution, bracketed with clock timestamps into TraceData.

use std::net::SocketAddr;
use std::rc::Rc;

use aiperf_clock::Clock;

use crate::models::{ErrorDetails, ErrorKind, TraceData};

/// Resolve `host:port`, recording `dns_lookup_start/end_ns`. Returns the first
/// address. For a literal IP the lookup is trivial but still timestamped.
pub async fn resolve(
    host: &str,
    port: u16,
    clock: &Rc<dyn Clock>,
    trace: &mut TraceData,
) -> Result<SocketAddr, ErrorDetails> {
    trace.dns_lookup_start_ns = Some(clock.now_ns());
    let mut addrs = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| ErrorDetails {
            kind: ErrorKind::Connect,
            code: None,
            message: format!("dns: {e}"),
        })?;
    trace.dns_lookup_end_ns = Some(clock.now_ns());
    addrs
        .next()
        .ok_or_else(|| ErrorDetails::other(format!("no address for {host}")))
}
