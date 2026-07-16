// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for the mock server's Unix-domain-socket (UDS) HTTP/1.1
//! listener (`--uds`, env `MOCK_SERVER_UDS`), driven through the full product
//! path: `aiperf profile` targets the socket via `endpoint.udsPath`.
//!
//! # The product path
//!
//! The runner's HTTP transport connects a `tokio::net::UnixStream` (HTTP/1.1)
//! whenever `ClientConfig.uds_path` is set
//! (`rust/aiperf/src/transport::http/client/connection.rs`). That is now wired
//! end to end: the Python `endpoint.uds_path` field
//! (`src/aiperf/config/endpoint.py`) is projected by `rust_wire._authored_endpoint`
//! into the protocol-v2 `EndpointProfileConfigV2.uds_path`
//! (`rust/aiperf/src/engine/registry.rs`), which threads it into the
//! `ClientConfig` (forcing HTTP/1.1). The endpoint URL still supplies the
//! request path + `Host` header, so it stays a normal `http://…` value.
//!
//! [`uds_chat_via_aiperf_profile_raw_records`] proves it decisively: the run's
//! endpoint URL points at a **closed** TCP port, so a valid raw-record export
//! is only possible if every request was carried over the Unix socket.
//! [`uds_health_over_unix_socket`] / [`uds_chat_completion_over_unix_socket`]
//! add direct-client coverage of the shipped `serve_router_uds` loop.

#![cfg(unix)]

mod common;
use common::*;

use std::time::Duration;

use aiperf_mock_server::config::MockServerConfig;
use aiperf_mock_server::{app, build_router};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

/// Requests / OSL / concurrency for the profile-driven run.
const UDS_REQUESTS: u32 = 6;
const UDS_OSL: usize = 8;
const UDS_CONCURRENCY: u32 = 2;

/// A Config-v2 YAML that targets a Unix socket: the endpoint URL is a **dead**
/// TCP port (`127.0.0.1:1`) so only `udsPath` can serve the run, and readiness
/// probing is disabled (`waitForModelTimeout: 0.0`) so nothing dials the URL.
fn uds_config(socket: &str) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 model: gpt-4\n\
        \x20 endpoint:\n\
        \x20   url: http://127.0.0.1:1/v1/chat/completions\n\
        \x20   type: chat\n\
        \x20   streaming: true\n\
        \x20   udsPath: {socket}\n\
        \x20   waitForModelTimeout: 0.0\n\
        \x20 dataset:\n\
        \x20   type: synthetic\n\
        \x20   entries: {UDS_REQUESTS}\n\
        \x20   prompts:\n\
        \x20     isl: {{mean: 64, stddev: 0}}\n\
        \x20     osl: {{mean: {UDS_OSL}, stddev: 0}}\n\
        \x20 phases:\n\
        \x20   - name: profiling\n\
        \x20     type: concurrency\n\
        \x20     requests: {UDS_REQUESTS}\n\
        \x20     concurrency: {UDS_CONCURRENCY}\n\
        \x20 gpuTelemetry: {{enabled: false}}\n\
        \x20 serverMetrics: {{enabled: false}}\n\
        \x20 artifacts:\n\
        \x20   raw: true\n\
        \x20   records:\n\
        \x20     - jsonl\n\
        \x20 runtime:\n\
        \x20   ui: none\n"
    )
}

/// Reconstruct the streamed assistant content of one raw record from its SSE
/// `choices[0].delta.content` frames.
fn record_content(record: &Value) -> String {
    let mut out = String::new();
    if let Some(responses) = record.get("responses").and_then(Value::as_array) {
        for resp in responses {
            let Some(packets) = resp.get("packets").and_then(Value::as_array) else {
                continue;
            };
            for packet in packets {
                if packet.get("name").and_then(Value::as_str) != Some("data") {
                    continue;
                }
                let Some(raw) = packet.get("value").and_then(Value::as_str) else {
                    continue;
                };
                if raw.trim() == "[DONE]" {
                    continue;
                }
                if let Ok(obj) = serde_json::from_str::<Value>(raw.trim())
                    && let Some(c) = obj
                        .pointer("/choices/0/delta/content")
                        .and_then(Value::as_str)
                {
                    out.push_str(c);
                }
            }
        }
    }
    out
}

/// `aiperf profile` streams chat over the Unix socket end to end. The endpoint
/// URL is a closed TCP port, so a well-formed raw-record export proves every
/// request was carried over `udsPath`, not TCP.
#[tokio::test]
async fn uds_chat_via_aiperf_profile_raw_records() {
    if cfg!(target_os = "macos") {
        return;
    }
    let socket = start_uds_mock().await;
    let socket_str = socket.to_string_lossy().into_owned();

    let h = AIPerfHarness::new().await; // harness supplies venv + runner; its TCP mock is unused
    let cfg_file = h.artifact_path().join("uds.yaml");
    std::fs::write(&cfg_file, uds_config(&socket_str)).expect("write uds config");

    let r = h.run(&format!("--config {}", cfg_file.display()));
    assert!(
        r.success(),
        "uds run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );

    let records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        UDS_REQUESTS as usize,
        "one raw record per request over the Unix socket"
    );
    for (i, rec) in records.iter().enumerate() {
        let timing = extract_timing(rec);
        assert_eq!(
            timing.status,
            Some(200),
            "record {i}: status {:?}",
            timing.status
        );
        assert!(
            !record_content(rec).is_empty(),
            "record {i}: streamed content should be non-empty"
        );
    }
    let _ = std::fs::remove_file(&socket);
}

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

    // Serve the socket on its OWN dedicated runtime thread, NOT the caller's
    // `#[tokio::test]` runtime: the profile-driven test blocks that runtime on a
    // synchronous `aiperf` subprocess, which would otherwise starve the accept
    // loop and make the socket unreachable for the duration of the run.
    let serve_path = path_str.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("uds mock runtime");
        rt.block_on(async move {
            let state = app::build_state(cfg);
            let router = build_router(state);
            let _ = aiperf_mock_server::listener::serve_router_uds(router, &serve_path).await;
        });
    });

    // Wait for the socket to appear and accept.
    for _ in 0..200 {
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
