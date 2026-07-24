// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! io_uring thread-per-core HTTP engine for the request hot path (`--uring`).
//!
//! The default tokio/hyper server tops out at the async runtime's per-request
//! scheduling/reactor overhead (~1.15M rps on a 32-core box under saturating
//! load), well below the raw transport ceiling. This engine runs the *real*
//! non-streaming chat-completions path — HTTP framing, JSON parse, tokenization,
//! metrics, deterministic response — on a monoio io_uring runtime per core, each
//! with its own `SO_REUSEPORT` listener, so there is no cross-core scheduling on
//! the hot path.
//!
//! It reuses the shared generation and metric code (`AppState`,
//! `render_chat_completion_nonstream_fast`), so responses and metrics match the
//! tokio path. It implies `--fast` semantics (zero simulated latency) on the
//! served path. Non-chat endpoints get minimal handlers; streaming is not yet
//! served here.

use std::sync::Arc;

use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::{ListenerOpts, TcpListener, TcpStream};

use crate::config::MockServerConfig;
use crate::state::AppState;

/// Launch one io_uring runtime per core, each serving the shared `AppState`.
pub fn run(config: &MockServerConfig) -> anyhow::Result<()> {
    if !config.no_tokenizer {
        crate::tokens::load_corpus();
    }
    let cores = if config.workers > 0 {
        config.workers
    } else {
        num_cpus::get().max(1)
    };
    let host = config.host.clone();
    let port = config.port;
    // build_state spawns tokio tasks for the DCGM throughput sampler and the
    // batch scheduler; neither exists under the io_uring runtime and both are
    // irrelevant to this --fast raw-throughput engine. Disable them so setup
    // does not touch a Tokio reactor.
    let mut cfg = config.clone();
    cfg.dcgm_auto_load = false;
    cfg.scheduler_enabled = false;
    let state = crate::app::build_state(cfg);

    tracing::info!(
        %host, port, cores,
        "Starting AIPerf Mock Server (io_uring engine); non-streaming chat path, --fast semantics"
    );

    let mut handles = Vec::with_capacity(cores);
    for _ in 0..cores {
        let state = state.clone();
        let host = host.clone();
        handles.push(std::thread::spawn(move || {
            let mut rt = monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
                .with_entries(4096)
                .enable_timer()
                .build()
                .expect("build io_uring runtime");
            rt.block_on(serve(host, port, state));
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

async fn serve(host: String, port: u16, state: Arc<AppState>) {
    let addr = format!("{host}:{port}");
    let opts = ListenerOpts::default().reuse_port(true).reuse_addr(true);
    let listener = match TcpListener::bind_with_config(addr.as_str(), &opts) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(%addr, "io_uring bind failed: {e}");
            return;
        }
    };
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let _ = stream.set_nodelay(true);
                let state = state.clone();
                monoio::spawn(handle_conn(stream, state));
            }
            Err(_) => continue,
        }
    }
}

/// A parsed request head: byte length of the head, method/path, body length,
/// and whether the peer asked to close the connection.
struct Head {
    head_len: usize,
    body_len: usize,
    is_post: bool,
    path_start: usize,
    path_end: usize,
    keep_alive: bool,
}

/// Parse one HTTP/1.1 request head out of `buf`. Returns `Ok(None)` if more
/// bytes are needed, `Err(())` on malformed input.
fn parse_head(buf: &[u8]) -> Result<Option<Head>, ()> {
    let mut headers = [httparse::EMPTY_HEADER; 48];
    let mut req = httparse::Request::new(&mut headers);
    match req.parse(buf) {
        Ok(httparse::Status::Complete(head_len)) => {
            let method = req.method.unwrap_or("");
            let is_post = method.eq_ignore_ascii_case("POST");
            // Locate the path span within `buf` so the caller can match routes
            // without allocating.
            let path = req.path.unwrap_or("/");
            let path_start = path.as_ptr() as usize - buf.as_ptr() as usize;
            let path_end = path_start + path.len();
            let mut body_len = 0usize;
            let mut keep_alive = true; // HTTP/1.1 default
            for h in req.headers.iter() {
                if h.name.eq_ignore_ascii_case("content-length") {
                    body_len = std::str::from_utf8(h.value)
                        .ok()
                        .and_then(|v| v.trim().parse().ok())
                        .unwrap_or(0);
                } else if h.name.eq_ignore_ascii_case("connection") {
                    if h.value.eq_ignore_ascii_case(b"close") {
                        keep_alive = false;
                    }
                }
            }
            Ok(Some(Head {
                head_len,
                body_len,
                is_post,
                path_start,
                path_end,
                keep_alive,
            }))
        }
        Ok(httparse::Status::Partial) => Ok(None),
        Err(_) => Err(()),
    }
}

const CHAT_PATHS: &[&str] = &["/v1/chat/completions", "/openai/v1/chat/completions"];

fn http_response(status: &str, content_type: &str, body: &[u8], keep_alive: bool) -> Vec<u8> {
    let conn = if keep_alive { "keep-alive" } else { "close" };
    let mut out = Vec::with_capacity(body.len() + 160);
    out.extend_from_slice(b"HTTP/1.1 ");
    out.extend_from_slice(status.as_bytes());
    out.extend_from_slice(b"\r\nContent-Type: ");
    out.extend_from_slice(content_type.as_bytes());
    out.extend_from_slice(b"\r\nContent-Length: ");
    out.extend_from_slice(body.len().to_string().as_bytes());
    out.extend_from_slice(b"\r\nConnection: ");
    out.extend_from_slice(conn.as_bytes());
    out.extend_from_slice(b"\r\n\r\n");
    out.extend_from_slice(body);
    out
}

/// Build the response bytes for one fully-received request.
fn route(state: &AppState, head: &Head, buf: &[u8]) -> Vec<u8> {
    let path = &buf[head.path_start..head.path_end];
    let ka = head.keep_alive;
    if head.is_post {
        let is_chat = CHAT_PATHS.iter().any(|p| path == p.as_bytes());
        if is_chat {
            let body = &buf[head.head_len..head.head_len + head.body_len];
            match serde_json::from_slice::<crate::models::ChatCompletionRequest>(body) {
                Ok(req) => {
                    let json = crate::handlers::render_chat_completion_nonstream_fast(state, &req);
                    return http_response("200 OK", "application/json", &json, ka);
                }
                Err(e) => {
                    let msg = format!("{{\"error\":\"invalid request: {e}\"}}");
                    return http_response("422 Unprocessable Entity", "application/json", msg.as_bytes(), ka);
                }
            }
        }
        return http_response(
            "404 Not Found",
            "application/json",
            b"{\"error\":\"path not served by --uring engine\"}",
            ka,
        );
    }
    // GET routes.
    match path {
        b"/health" => http_response("200 OK", "application/json", b"{\"status\":\"healthy\"}", ka),
        b"/v1/models" | b"/openai/v1/models" => http_response(
            "200 OK",
            "application/json",
            b"{\"object\":\"list\",\"data\":[{\"id\":\"mock-model\",\"object\":\"model\",\"owned_by\":\"aiperf-mock\"}]}",
            ka,
        ),
        b"/metrics" => {
            let body = crate::prom::encode(&state.recorder.metrics.aiperf.registry);
            http_response("200 OK", "text/plain; version=0.0.4", &body, ka)
        }
        _ => http_response("404 Not Found", "application/json", b"{\"error\":\"not found\"}", ka),
    }
}

async fn handle_conn(mut stream: TcpStream, state: Arc<AppState>) {
    let mut acc: Vec<u8> = Vec::with_capacity(16384);
    let mut buf: Vec<u8> = vec![0u8; 65536];
    loop {
        // Drain any fully-buffered pipelined requests before another read.
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
            let resp = route(&state, &head, &acc);
            let close = !head.keep_alive;
            let (wres, _) = stream.write_all(resp).await;
            if wres.is_err() || close {
                return;
            }
            acc.drain(..total);
        }
        let (res, b) = stream.read(buf).await;
        buf = b;
        match res {
            Ok(0) => return,
            Ok(n) => acc.extend_from_slice(&buf[..n]),
            Err(_) => return,
        }
    }
}
