// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defaults. Port of `AioHttpDefaults` / `SocketDefaults`.
//!
//! The pool, DNS, keepalive, and TLS defaults follow
//! `src/aiperf/transports/http_defaults.py:131-169` and are pinned by the
//! source tests in `tests/unit/transports/test_tcp_connector.py:32-88`.

use crate::models::HttpVersion;

/// Client-wide configuration. Timeouts are clock-nanoseconds.
///
/// `connect_timeout_ns` is enforced in `client::connection::establish` (races
/// the DNS/TCP/TLS/handshake phase against a Clock timer) and `request_timeout_ns`
/// is enforced in `client::http_client::HttpClient::dispatch` (races the
/// send + response phase). `total_timeout_ns` wraps connection acquisition,
/// send, and the complete response lifecycle with one deadline, matching
/// Config-v2's endpoint request timeout. For all three, `None` or a non-positive
/// value means "no deadline".
///
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Deadline for DNS, TCP, TLS, and HTTP handshake establishment.
    pub connect_timeout_ns: Option<i64>,
    /// Deadline for request send plus the complete response body.
    pub request_timeout_ns: Option<i64>,
    /// One end-to-end request deadline including connection establishment.
    pub total_timeout_ns: Option<i64>,
    /// Verify the server certificate and hostname for HTTPS connections.
    pub ssl_verify: bool,
    /// HTTP protocol selection and cleartext prior-knowledge policy.
    pub http_version: HttpVersion,
    /// Maximum idle lifetime of a pooled connection. `None` disables expiry.
    pub keepalive_ns: Option<i64>,
    /// Maximum number of simultaneous HTTP/1 connections per origin.
    ///
    /// HTTP/2 uses one multiplexed connection per origin; this bound applies to
    /// protocols that require an exclusive connection while a request is live.
    pub max_connections_per_origin: usize,
    /// Whether hostname resolutions are cached by the transport.
    pub use_dns_cache: bool,
    /// DNS cache lifetime. `None` retains entries until the transport is dropped.
    pub dns_cache_ttl_ns: Option<i64>,
    /// Retain per-wire-chunk `(clock_ns, size_bytes)` trace vectors.
    ///
    /// Counts, byte totals, and first/last timestamps are always collected.
    pub collect_trace_chunks: bool,
    /// When set, connect over this Unix-domain socket path instead of TCP
    /// (co-located high-throughput: bypasses the TCP/IP loopback softirq tax).
    /// HTTP/1.1 is used over UDS. The request URL still supplies the path + Host.
    pub uds_path: Option<String>,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            connect_timeout_ns: None,
            request_timeout_ns: None,
            total_timeout_ns: None,
            ssl_verify: true,
            http_version: HttpVersion::Auto,
            keepalive_ns: Some(300_000_000_000),
            max_connections_per_origin: 2_500,
            use_dns_cache: true,
            dns_cache_ttl_ns: Some(300_000_000_000),
            collect_trace_chunks: false,
            uds_path: None,
        }
    }
}

/// Apply low-latency streaming socket options (TCP_NODELAY, keepalive,
/// SO_REUSEADDR). Linux-only extras (buffer sizes) are `cfg`-gated. Port of
/// `SocketDefaults.apply_to_socket`. Operates on a borrowed [`socket2::SockRef`]
/// so it never takes ownership of the fd.
pub fn apply_socket_opts(sock: &socket2::SockRef<'_>) -> std::io::Result<()> {
    sock.set_nodelay(true)?;
    sock.set_keepalive(true)?;
    let _ = sock.set_reuse_address(true);
    #[cfg(target_os = "linux")]
    {
        // Buffer tuning is best-effort; ignore failures.
        let _ = sock.set_recv_buffer_size(1 << 20);
        let _ = sock.set_send_buffer_size(1 << 20);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::HttpVersion;

    #[test]
    fn defaults_match_python_aiohttp_defaults() {
        let c = ClientConfig::default();
        assert!(c.ssl_verify);
        assert_eq!(c.http_version, HttpVersion::Auto);
        assert_eq!(c.connect_timeout_ns, None);
        assert_eq!(c.request_timeout_ns, None);
        assert_eq!(c.total_timeout_ns, None);
        assert_eq!(c.keepalive_ns, Some(300_000_000_000));
        assert_eq!(c.max_connections_per_origin, 2_500);
        assert!(c.use_dns_cache);
        assert_eq!(c.dns_cache_ttl_ns, Some(300_000_000_000));
        assert!(!c.collect_trace_chunks);
    }
}
