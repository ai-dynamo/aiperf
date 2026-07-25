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

/// Fast path: serve the throughput-critical endpoints (chat/text completions,
/// embeddings) with the hand-rolled synchronous renderers. Returns `None` for
/// any other request so the caller can fall back to the full axum `Router`
/// (`--blocking`) or a minimal handler (`--uring`).
pub fn route_fast(state: &AppState, head: &Head, buf: &[u8]) -> Option<Vec<u8>> {
    if !head.is_post {
        return None;
    }
    let path = &buf[head.path_start..head.path_end];
    let ka = head.keep_alive;
    let body = &buf[head.head_len..head.head_len + head.body_len];
    if CHAT_PATHS.iter().any(|p| path == p.as_bytes()) {
        return Some(match serde_json::from_slice::<crate::models::ChatCompletionRequest>(body) {
            Ok(req) => {
                let (ct, out) = crate::handlers::render_chat_completion_fast(state, &req);
                http_response("200 OK", ct, &out, ka)
            }
            Err(e) => bad_request(&e, ka),
        });
    }
    if TEXT_PATHS.iter().any(|p| path == p.as_bytes()) {
        return Some(match serde_json::from_slice::<crate::models::CompletionRequest>(body) {
            Ok(req) => {
                let (ct, out) = crate::handlers::render_text_completion_fast(state, &req);
                http_response("200 OK", ct, &out, ka)
            }
            Err(e) => bad_request(&e, ka),
        });
    }
    if EMBED_PATHS.iter().any(|p| path == p.as_bytes()) {
        return Some(match serde_json::from_slice::<crate::models::EmbeddingRequest>(body) {
            Ok(req) => {
                let out = crate::handlers::render_embeddings_fast(state, &req);
                http_response("200 OK", "application/json", &out, ka)
            }
            Err(e) => bad_request(&e, ka),
        });
    }
    None
}

/// Minimal fallback for engines without an axum fallback (`--uring`): a few GET
/// routes, otherwise 404.
pub fn route_minimal(state: &AppState, head: &Head, buf: &[u8]) -> Vec<u8> {
    let path = &buf[head.path_start..head.path_end];
    let ka = head.keep_alive;
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
        _ => http_response(
            "404 Not Found",
            "application/json",
            b"{\"error\":\"path not served by --uring engine (use --blocking for full endpoint coverage)\"}",
            ka,
        ),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MockServerConfig;

    fn fast_state() -> std::sync::Arc<AppState> {
        let config = MockServerConfig {
            fast: true,
            no_tokenizer: true,
            ..MockServerConfig::default()
        }
        .apply_flags();
        build_engine_state(&config)
    }

    /// Drive `route_fast` end-to-end for one POST body and return the framed
    /// HTTP/1.1 response bytes (the same bytes the hand-rolled `--blocking`/
    /// `--uring` engines write to the socket).
    fn route_post(state: &AppState, path: &str, json: &str) -> Vec<u8> {
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{json}",
            json.len(),
        );
        let buf = request.into_bytes();
        let head = parse_head(&buf).unwrap().unwrap();
        route_fast(state, &head, &buf).expect("fast path serves this route")
    }

    #[test]
    fn route_fast_streaming_chat_emits_sse_frames() {
        let state = fast_state();
        let out = route_post(
            &state,
            "/v1/chat/completions",
            r#"{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}"#,
        );
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.contains("Content-Type: text/event-stream"),
            "streaming chat must set the SSE content type: {text:?}"
        );
        assert!(text.contains("data: "), "missing SSE data frames");
        assert!(text.contains("chat.completion.chunk"), "missing chunk object");
        assert!(text.contains("[DONE]"), "missing terminal [DONE] frame");
        assert!(text.contains("\"usage\""), "missing usage frame");
    }

    #[test]
    fn route_fast_streaming_completions_emits_sse_frames() {
        let state = fast_state();
        let out = route_post(
            &state,
            "/v1/completions",
            r#"{"model":"gpt-4","prompt":"hi","stream":true}"#,
        );
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.contains("Content-Type: text/event-stream"),
            "streaming completions must set the SSE content type: {text:?}"
        );
        assert!(text.contains("data: "), "missing SSE data frames");
        assert!(text.contains("[DONE]"), "missing terminal [DONE] frame");
    }

    #[test]
    fn route_fast_non_streaming_chat_emits_json() {
        let state = fast_state();
        let out = route_post(
            &state,
            "/v1/chat/completions",
            r#"{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}"#,
        );
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.contains("Content-Type: application/json"),
            "non-streaming chat must be JSON: {text:?}"
        );
        assert!(text.contains("chat.completion"), "missing completion object");
        // Body must be well-formed JSON after the header/body split.
        let body = text.split("\r\n\r\n").nth(1).expect("has a body");
        let value: serde_json::Value = serde_json::from_str(body).expect("valid JSON body");
        assert_eq!(value["object"], "chat.completion");
    }
}

