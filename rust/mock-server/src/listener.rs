// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared TCP listener construction.
//!
//! Both the single-process server (`main::serve`) and the multi-process
//! round-robin balancer (`balancer`) need a listener tuned identically —
//! `SO_REUSEADDR` + `SO_REUSEPORT` and a deep accept backlog — so the socket
//! setup lives here rather than being duplicated at each bind site.

use std::net::SocketAddr;

use socket2::{Domain, Protocol, Socket, Type};

/// Listen backlog. Large enough that a C10K burst of SYNs doesn't get dropped;
/// the kernel silently clamps to `/proc/sys/net/core/somaxconn` (4096 on modern
/// Linux).
pub const LISTEN_BACKLOG: i32 = 16_384;

/// Build a non-blocking, `SO_REUSEADDR`/`SO_REUSEPORT` TCP listener bound to
/// `addr` with a deep backlog, returned as a tokio `TcpListener`.
///
/// `SO_REUSEPORT` (Linux) lets several independent processes bind the *same*
/// port — the balancer relies on it only as a defensive fallback; its backends
/// each own a distinct loopback port and it round-robins across them itself.
pub fn build_listener(addr: SocketAddr) -> anyhow::Result<tokio::net::TcpListener> {
    let domain = if addr.is_ipv4() {
        Domain::IPV4
    } else {
        Domain::IPV6
    };
    let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
    socket.set_nonblocking(true)?;
    socket.set_reuse_address(true)?;
    // SO_REUSEPORT on Linux lets multiple sockets bind the same port so the
    // kernel load-balances accepts across them. Safe fallback if unsupported.
    #[cfg(target_os = "linux")]
    let _ = socket.set_reuse_port(true);
    socket.bind(&addr.into())?;
    socket.listen(LISTEN_BACKLOG)?;
    let std_listener: std::net::TcpListener = socket.into();
    let listener = tokio::net::TcpListener::from_std(std_listener)?;
    Ok(listener)
}
