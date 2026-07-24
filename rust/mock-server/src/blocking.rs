// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Blocking thread-per-connection HTTP engine for the request hot path
//! (`--blocking`).
//!
//! No async runtime at all: N `SO_REUSEPORT` accept loops on OS threads, each
//! accepted connection handled on its own blocking thread. This is `--plaid`'s
//! I/O model (which reaches the raw transport ceiling) but running the *real*
//! non-streaming chat path — a control point isolating "how much does the async
//! runtime cost?" from "how much does the real per-request work cost?". Request
//! framing/routing is shared with the io_uring engine via [`crate::http_core`].
//! Implies `--fast` semantics.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

use socket2::{Domain, Protocol, Socket, Type};

use crate::config::MockServerConfig;
use crate::http_core::{build_engine_state, parse_head, route};
use crate::state::AppState;

/// Launch `workers` `SO_REUSEPORT` accept loops (default = CPU count).
pub fn run(config: &MockServerConfig) -> anyhow::Result<()> {
    let accept_loops = if config.workers > 0 {
        config.workers
    } else {
        num_cpus::get().max(1)
    };
    let host: std::net::IpAddr = config
        .host
        .parse()
        .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
    let addr = std::net::SocketAddr::new(host, config.port);
    let state = build_engine_state(config);

    tracing::info!(
        %addr, accept_loops,
        "Starting AIPerf Mock Server (blocking thread-per-connection engine); \
         non-streaming chat path, --fast semantics"
    );

    let mut handles = Vec::with_capacity(accept_loops);
    for _ in 0..accept_loops {
        let state = state.clone();
        handles.push(std::thread::spawn(move || accept_loop(addr, state)));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

/// One `SO_REUSEPORT` listener; the kernel load-balances new connections across
/// all accept loops. Each connection gets its own handler thread.
fn accept_loop(addr: std::net::SocketAddr, state: Arc<AppState>) {
    let listener = match build_reuseport_listener(addr) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(%addr, "blocking bind failed: {e}");
            return;
        }
    };
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);
                let state = state.clone();
                std::thread::spawn(move || handle_conn(stream, &state));
            }
            Err(_) => continue,
        }
    }
}

fn build_reuseport_listener(addr: std::net::SocketAddr) -> std::io::Result<std::net::TcpListener> {
    let domain = if addr.is_ipv4() {
        Domain::IPV4
    } else {
        Domain::IPV6
    };
    let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
    socket.set_reuse_address(true)?;
    socket.set_reuse_port(true)?;
    socket.bind(&addr.into())?;
    socket.listen(16384)?;
    Ok(socket.into())
}

fn handle_conn(mut stream: TcpStream, state: &AppState) {
    let mut acc: Vec<u8> = Vec::with_capacity(16384);
    let mut buf = [0u8; 65536];
    loop {
        // Serve every fully-buffered request before blocking on another read.
        loop {
            let head = match parse_head(&acc) {
                Ok(Some(h)) => h,
                Ok(None) => break,
                Err(_) => return,
            };
            let total = head.head_len + head.body_len;
            if acc.len() < total {
                break;
            }
            let resp = route(state, &head, &acc);
            let close = !head.keep_alive;
            if stream.write_all(&resp).is_err() || close {
                return;
            }
            acc.drain(..total);
        }
        match stream.read(&mut buf) {
            Ok(0) => return,
            Ok(n) => acc.extend_from_slice(&buf[..n]),
            Err(_) => return,
        }
    }
}
