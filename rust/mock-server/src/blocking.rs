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

use axum::Router;
use socket2::{Domain, Protocol, Socket, Type};

use crate::config::MockServerConfig;
use crate::http_core::{Head, build_engine_state, parse_head, route_fast};
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
    // The real axum Router handles every non-hot endpoint (full fidelity); the
    // hot chat/text/embeddings paths are served by the fast synchronous
    // renderers. Both share the same `AppState`.
    let router = crate::app::build_router(state.clone());

    tracing::info!(
        %addr, accept_loops,
        "Starting AIPerf Mock Server (blocking thread-per-connection engine); \
         fast chat/text/embeddings path + axum fallback for other routes, --fast semantics"
    );

    let mut handles = Vec::with_capacity(accept_loops);
    for _ in 0..accept_loops {
        let state = state.clone();
        let router = router.clone();
        handles.push(std::thread::spawn(move || accept_loop(addr, state, router)));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

/// One `SO_REUSEPORT` listener; the kernel load-balances new connections across
/// all accept loops. Each connection gets its own handler thread.
fn accept_loop(addr: std::net::SocketAddr, state: Arc<AppState>, router: Router) {
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
                let router = router.clone();
                std::thread::spawn(move || handle_conn(stream, &state, &router));
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

fn handle_conn(mut stream: TcpStream, state: &AppState, router: &Router) {
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
            // Hot endpoints take the fast synchronous path; everything else is
            // served by the real axum Router for full fidelity.
            let resp = route_fast(state, &head, &acc)
                .unwrap_or_else(|| axum_fallback(router, &acc[..total], &head));
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

thread_local! {
    /// One current-thread Tokio runtime per connection thread, built lazily on
    /// the first non-hot request so connections that only hit the fast path
    /// never create a runtime.
    static FALLBACK_RT: tokio::runtime::Runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build fallback runtime");
}

/// Drive one request through the real axum `Router` and serialize its response
/// to HTTP/1.1 bytes. Used for every endpoint the fast path does not serve.
fn axum_fallback(router: &Router, raw: &[u8], head: &Head) -> Vec<u8> {
    use tower::ServiceExt;

    let mut headers = [httparse::EMPTY_HEADER; 96];
    let mut parsed = httparse::Request::new(&mut headers);
    if parsed.parse(raw).is_err() {
        return crate::http_core::http_response(
            "400 Bad Request",
            "text/plain",
            b"",
            head.keep_alive,
        );
    }
    let method = parsed.method.unwrap_or("GET");
    let path = parsed.path.unwrap_or("/");
    let body = raw[head.head_len..head.head_len + head.body_len].to_vec();
    let mut builder = http::Request::builder().method(method).uri(path);
    for h in parsed.headers.iter() {
        if !h.name.is_empty() {
            builder = builder.header(h.name, h.value);
        }
    }
    let request = match builder.body(axum::body::Body::from(body)) {
        Ok(r) => r,
        Err(_) => {
            return crate::http_core::http_response(
                "400 Bad Request",
                "text/plain",
                b"",
                head.keep_alive,
            );
        }
    };

    FALLBACK_RT.with(|rt| {
        rt.block_on(async move {
            let response = match router.clone().oneshot(request).await {
                Ok(r) => r,
                Err(_) => {
                    return crate::http_core::http_response(
                        "500 Internal Server Error",
                        "text/plain",
                        b"",
                        head.keep_alive,
                    );
                }
            };
            let status = response.status();
            let (parts, body) = response.into_parts();
            let bytes = axum::body::to_bytes(body, usize::MAX)
                .await
                .unwrap_or_default();

            let mut out = Vec::with_capacity(bytes.len() + 256);
            out.extend_from_slice(b"HTTP/1.1 ");
            out.extend_from_slice(status.as_str().as_bytes());
            out.push(b' ');
            out.extend_from_slice(status.canonical_reason().unwrap_or("").as_bytes());
            out.extend_from_slice(b"\r\n");
            for (name, value) in parts.headers.iter() {
                // Re-frame with our own Content-Length/Connection; collecting the
                // body converts any chunked/streaming transfer to fixed length.
                if name == http::header::CONTENT_LENGTH
                    || name == http::header::TRANSFER_ENCODING
                    || name == http::header::CONNECTION
                {
                    continue;
                }
                out.extend_from_slice(name.as_str().as_bytes());
                out.extend_from_slice(b": ");
                out.extend_from_slice(value.as_bytes());
                out.extend_from_slice(b"\r\n");
            }
            out.extend_from_slice(b"Content-Length: ");
            out.extend_from_slice(bytes.len().to_string().as_bytes());
            out.extend_from_slice(b"\r\nConnection: ");
            out.extend_from_slice(if head.keep_alive {
                b"keep-alive"
            } else {
                b"close"
            });
            out.extend_from_slice(b"\r\n\r\n");
            out.extend_from_slice(&bytes);
            out
        })
    })
}
