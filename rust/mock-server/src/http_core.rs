// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-agnostic HTTP/1.1 request framing and routing for the hand-rolled
//! `--blocking` and `--uring` engines.
//!
//! These engines bypass axum/hyper to serve the real request hot path with the
//! least per-request machinery. Parsing, routing, and response assembly are pure
//! functions here so both the blocking std-thread loop and the io_uring monoio
//! loop share identical behavior; each engine supplies only its own I/O.
//!
//! Only the non-streaming OpenAI chat-completions path is served with full real
//! behavior (via [`crate::handlers::render_chat_completion_nonstream_fast`]);
//! these engines imply `--fast` semantics. A few GET routes get minimal
//! handlers; streaming is not served here.

use crate::state::AppState;

/// A parsed request head: byte length of the head, body length, method, the
/// path span within the buffer, and whether to keep the connection alive.
pub struct Head {
    pub head_len: usize,
    pub body_len: usize,
    pub is_post: bool,
    pub path_start: usize,
    pub path_end: usize,
    pub keep_alive: bool,
}

/// Parse one HTTP/1.1 request head out of `buf`. `Ok(None)` means more bytes are
/// needed; `Err(())` means malformed input (close the connection).
pub fn parse_head(buf: &[u8]) -> Result<Option<Head>, ()> {
    let mut headers = [httparse::EMPTY_HEADER; 48];
    let mut req = httparse::Request::new(&mut headers);
    match req.parse(buf) {
        Ok(httparse::Status::Complete(head_len)) => {
            let is_post = req.method.is_some_and(|m| m.eq_ignore_ascii_case("POST"));
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
                } else if h.name.eq_ignore_ascii_case("connection")
                    && h.value.eq_ignore_ascii_case(b"close")
                {
                    keep_alive = false;
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
const TEXT_PATHS: &[&str] = &["/v1/completions", "/openai/v1/completions"];
const EMBED_PATHS: &[&str] = &["/v1/embeddings", "/openai/v1/embeddings"];

/// Assemble an HTTP/1.1 response with the given status line, content type, and body.
pub fn http_response(status: &str, content_type: &str, body: &[u8], keep_alive: bool) -> Vec<u8> {
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

fn bad_request(e: &serde_json::Error, keep_alive: bool) -> Vec<u8> {
    let msg = format!("{{\"error\":\"invalid request: {e}\"}}");
    http_response("422 Unprocessable Entity", "application/json", msg.as_bytes(), keep_alive)
}

/// Build the response bytes for one fully-received request.
pub fn route(state: &AppState, head: &Head, buf: &[u8]) -> Vec<u8> {
    let path = &buf[head.path_start..head.path_end];
    let ka = head.keep_alive;
    if head.is_post {
        let body = &buf[head.head_len..head.head_len + head.body_len];
        if CHAT_PATHS.iter().any(|p| path == p.as_bytes()) {
            return match serde_json::from_slice::<crate::models::ChatCompletionRequest>(body) {
                Ok(req) => {
                    let (ct, out) = crate::handlers::render_chat_completion_fast(state, &req);
                    http_response("200 OK", ct, &out, ka)
                }
                Err(e) => bad_request(&e, ka),
            };
        }
        if TEXT_PATHS.iter().any(|p| path == p.as_bytes()) {
            return match serde_json::from_slice::<crate::models::CompletionRequest>(body) {
                Ok(req) => {
                    let (ct, out) = crate::handlers::render_text_completion_fast(state, &req);
                    http_response("200 OK", ct, &out, ka)
                }
                Err(e) => bad_request(&e, ka),
            };
        }
        if EMBED_PATHS.iter().any(|p| path == p.as_bytes()) {
            return match serde_json::from_slice::<crate::models::EmbeddingRequest>(body) {
                Ok(req) => {
                    let out = crate::handlers::render_embeddings_fast(state, &req);
                    http_response("200 OK", "application/json", &out, ka)
                }
                Err(e) => bad_request(&e, ka),
            };
        }
        return http_response(
            "404 Not Found",
            "application/json",
            b"{\"error\":\"path not served by hand-rolled engine\"}",
            ka,
        );
    }
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

/// Prepare the shared `AppState` for a hand-rolled engine. Disables the
/// DCGM throughput sampler and batch scheduler, which spawn Tokio tasks during
/// build and have no runtime under these engines (and are irrelevant to the
/// `--fast` raw-throughput path they serve).
pub fn build_engine_state(config: &crate::config::MockServerConfig) -> std::sync::Arc<AppState> {
    if !config.no_tokenizer {
        crate::tokens::load_corpus();
    }
    let mut cfg = config.clone();
    cfg.dcgm_auto_load = false;
    cfg.scheduler_enabled = false;
    crate::app::build_state(cfg)
}
