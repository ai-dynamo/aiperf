// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defaults. Port of `AioHttpDefaults` / `SocketDefaults`.

use crate::models::HttpVersion;

/// Client-wide configuration. Timeouts are clock-nanoseconds.
///
/// `connect_timeout_ns` is enforced in `client::connection::establish` (races
/// the DNS/TCP/TLS/handshake phase against a Clock timer) and `request_timeout_ns`
/// is enforced in `client::http_client::HttpClient::dispatch` (races the
/// send + response phase). For both, `None` or a non-positive value means "no
/// deadline".
///
/// NOTE: `keepalive_ns` is accepted but NOT enforced: it is meant to bound the
/// idle lifetime of a pooled connection, but `client::pool::ConnectionPool`
/// currently stores senders in a plain map with no per-entry insertion timestamp
/// or idle-age check, so there is no pool knob to attach a keepalive deadline to.
/// Enforcing it requires adding an idle-since timestamp to the pool's stored
/// entries and evicting on `take_ready` when `now - idle_since > keepalive_ns`
/// (pool.rs is outside this change's scope). It is retained for API/config
/// compatibility.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub connect_timeout_ns: Option<i64>,
    pub request_timeout_ns: Option<i64>,
    pub ssl_verify: bool,
    pub http_version: HttpVersion,
    pub keepalive_ns: Option<i64>,
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
            ssl_verify: true,
            http_version: HttpVersion::Auto,
            keepalive_ns: None,
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
    }
}
