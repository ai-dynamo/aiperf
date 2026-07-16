// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for the mock server's Unix-domain-socket (UDS) HTTP/1.1
//! listener (`--uds`, env `MOCK_SERVER_UDS`).
//!
//! # Why this is a Rust-level e2e, not `aiperf profile`
//!
//! The runner *transport* supports HTTP/1.1 over a Unix socket
//! (`rust/aiperf/src/transport_http/client/connection.rs` connects a
//! `tokio::net::UnixStream` and negotiates h1 whenever `ClientConfig.uds_path`
//! is set), but **nothing on the product path wires a URL or flag through to
//! `uds_path`**:
//!
//! - The Python frontend (`src/aiperf/`) has no `uds` / `unix://` knob — a full
//!   grep for `uds`, `unix://`, `unix_socket` finds nothing in the config or
//!   CLI surface.
//! - The protocol-v2 endpoint DTO the runner accepts
//!   (`EndpointProfileConfigV2` in `rust/aiperf/src/runner_protocol/registry.rs`)
//!   has no `uds`/`uds_path` field, and the `ClientConfig` it builds
//!   (`registry.rs`, the `let client = ClientConfig { .. }` block) leaves
//!   `uds_path` at its `None` default.
//! - The only in-tree code that parses a `unix:` URL prefix into `uds_path` is
//!   `rust/aiperf/src/graph/transport_bench.rs`, a benchmark harness — not the
//!   product runner.
//!
//! So `aiperf profile` genuinely cannot target a Unix socket today: the missing
//! knob is a `uds`/`unix://` mapping into `EndpointProfileConfigV2` +
//! `registry.rs`'s `ClientConfig` builder (and a matching Python config field).
//! That wiring lives under `rust/aiperf/**` and the Python frontend — out of
//! scope for this mock-server feature. This test therefore proves the shipped
//! listener directly: it runs the exact serve loop the `--uds` binary path
//! spawns (`aiperf_mock_server::listener::serve_router_uds`) and drives it with
//! an HTTP/1.1 client over the Unix socket, asserting a correct chat-completion
//! response (status 200 + generated content + usage), which is the same router
//! and generation seam the TCP e2e suite exercises.

#![cfg(unix)]

use std::time::Duration;

use aiperf_mock_server::config::MockServerConfig;
use aiperf_mock_server::{app, build_router};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

/// A collision-free temp socket path under the OS temp dir.
fn temp_socket_path(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "aiperf-e2e-uds-{tag}-{}-{nanos}.sock",
        std::process::id()
    ))
}

/// Spawn the shipped UDS serve loop for a fast, fully-deterministic mock and
/// return the socket path once it is accepting connections.
async fn start_uds_mock() -> std::path::PathBuf {
    // Deterministic mock: fast (zero latency), analytic (scheduler off), no
    // jitter. Corpus is loaded so token generation has a real vocabulary.
    let cfg = MockServerConfig {
        fast: true,
        ..MockServerConfig::default()
    }
    .apply_flags();
    aiperf_mock_server::tokens::load_corpus();

    let path = temp_socket_path("chat");
    let path_str = path.to_str().unwrap().to_owned();

    let state = app::build_state(cfg);
    let router = build_router(state);

    let serve_path = path_str.clone();
    tokio::spawn(async move {
        let _ = aiperf_mock_server::listener::serve_router_uds(router, &serve_path).await;
    });

    // Wait for the socket to appear and accept.
    for _ in 0..100 {
        if path.exists() && UnixStream::connect(&path_str).await.is_ok() {
            return path;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("UDS mock never became reachable at {path_str}");
}

/// Send one raw HTTP/1.1 request over the Unix socket and return
/// `(status_line, headers_blob, body)`.
async fn uds_request(
    path: &std::path::Path,
    method: &str,
    target: &str,
    body: Option<&str>,
) -> (String, String, String) {
    let mut stream = UnixStream::connect(path).await.expect("connect UDS");

    let mut req = format!("{method} {target} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n");
    if let Some(b) = body {
        req.push_str("Content-Type: application/json\r\n");
        req.push_str(&format!("Content-Length: {}\r\n", b.len()));
        req.push_str("\r\n");
        req.push_str(b);
    } else {
        req.push_str("\r\n");
    }
    stream.write_all(req.as_bytes()).await.expect("write req");
    stream.flush().await.expect("flush");

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.expect("read resp");
    let text = String::from_utf8_lossy(&raw).into_owned();

    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((text.as_str(), ""));
    let status_line = head.lines().next().unwrap_or("").to_owned();
    (status_line, head.to_owned(), body.to_owned())
}

/// `GET /health` succeeds over the Unix socket (basic liveness over UDS).
#[tokio::test]
async fn uds_health_over_unix_socket() {
    let path = start_uds_mock().await;
    let (status, _headers, _body) = uds_request(&path, "GET", "/health", None).await;
    assert!(
        status.starts_with("HTTP/1.1 200"),
        "health status: {status}"
    );
    let _ = std::fs::remove_file(&path);
}

/// A non-streaming `POST /v1/chat/completions` over the Unix socket returns a
/// well-formed OpenAI chat completion: status 200, a non-empty assistant
/// message, and a completion-token count capped at the requested `max_tokens`.
#[tokio::test]
async fn uds_chat_completion_over_unix_socket() {
    let path = start_uds_mock().await;

    let max_tokens = 12;
    let body = json!({
        "model": "gpt-mock",
        "messages": [{"role": "user", "content": "Hello over a unix socket, please respond."}],
        "max_tokens": max_tokens,
        "stream": false,
    })
    .to_string();

    let (status, headers, body) =
        uds_request(&path, "POST", "/v1/chat/completions", Some(&body)).await;

    assert!(
        status.starts_with("HTTP/1.1 200"),
        "chat status: {status}\nheaders:\n{headers}\nbody:\n{body}"
    );

    let resp: Value = serde_json::from_str(&body)
        .unwrap_or_else(|e| panic!("chat body was not JSON ({e}):\n{body}"));

    // DATA: object shape + non-empty generated content.
    assert_eq!(
        resp.get("object").and_then(Value::as_str),
        Some("chat.completion"),
        "unexpected object field: {resp}"
    );
    let content = resp
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing choices[0].message.content: {resp}"));
    assert!(!content.is_empty(), "generated content should be non-empty");

    let role = resp
        .pointer("/choices/0/message/role")
        .and_then(Value::as_str);
    assert_eq!(role, Some("assistant"), "assistant role expected: {resp}");

    // DATA: OSL honored — completion tokens capped at the requested max_tokens.
    let completion_tokens = resp
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("missing usage.completion_tokens: {resp}"));
    assert!(
        completion_tokens > 0 && completion_tokens <= max_tokens,
        "completion_tokens {completion_tokens} outside (0, {max_tokens}]: {resp}"
    );

    let finish = resp
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str);
    assert!(
        matches!(finish, Some("stop") | Some("length")),
        "unexpected finish_reason {finish:?}: {resp}"
    );

    let _ = std::fs::remove_file(&path);
}
