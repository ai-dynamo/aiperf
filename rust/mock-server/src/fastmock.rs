// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `--fastmock`: a minimal, fixed-response server that replaces the real mock
//! server entirely.
//!
//! Every request gets the same pre-built static payload — a single streamed
//! chat-completion chunk for anything other than a bare `GET`, and a static
//! model list for `GET`. There is no latency simulation, routing, token
//! rendering, or endpoint dispatch: this trades away behavioral fidelity for
//! raw throughput/latency headroom when the test only needs *a* fast HTTP
//! peer, not a faithful one. Ports the standalone `tools/fastmock.rs` binary
//! onto the crate's existing tokio runtime and listener plumbing so it shares
//! `--host`/`--port`/`--uds` with the real server instead of needing a
//! separate process.

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::MockServerConfig;
use crate::listener::build_listener;
#[cfg(unix)]
use crate::listener::bind_unix_listener;

struct Responses {
    chat: Arc<[u8]>,
    models: Arc<[u8]>,
}

fn build_responses() -> Responses {
    let body = b"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"x\"}}]}\n\ndata: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n",
        body.len()
    );
    let chat: Arc<[u8]> = [head.as_bytes(), body].concat().into();

    let models = b"{\"object\":\"list\",\"data\":[{\"id\":\"mock-model\",\"object\":\"model\"}]}";
    let mhead = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n",
        models.len()
    );
    let models_resp: Arc<[u8]> = [mhead.as_bytes(), models.as_ref()].concat().into();

    Responses {
        chat,
        models: models_resp,
    }
}

/// Byte-level, allocation-free header scan (mirrors `tools/fastmock.rs`).
fn content_length(head: &[u8]) -> usize {
    for line in head.split(|&b| b == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Some(colon) = line.iter().position(|&b| b == b':') else {
            continue;
        };
        let (name, val) = line.split_at(colon);
        if name.eq_ignore_ascii_case(b"content-length") {
            let val = std::str::from_utf8(&val[1..]).unwrap_or("").trim();
            return val.parse().unwrap_or(0);
        }
    }
    0
}

/// Answers every complete request found in `buf` starting at offset 0.
/// Returns the byte offset past the last complete request handled.
async fn drain_requests<S: AsyncWriteExt + Unpin>(
    buf: &[u8],
    stream: &mut S,
    resp: &Responses,
) -> std::io::Result<usize> {
    let mut off = 0usize;
    loop {
        let rest = &buf[off..];
        let Some(hpos) = rest.windows(4).position(|w| w == b"\r\n\r\n") else {
            break;
        };
        let head = &rest[..hpos];
        let cl = if head.starts_with(b"GET") {
            0
        } else {
            content_length(head)
        };
        let total = hpos + 4 + cl;
        if rest.len() < total {
            break;
        }
        let payload = if head.starts_with(b"GET") {
            &resp.models
        } else {
            &resp.chat
        };
        stream.write_all(payload).await?;
        off += total;
    }
    Ok(off)
}

async fn handle<S: AsyncReadExt + AsyncWriteExt + Unpin>(mut stream: S, resp: Arc<Responses>) {
    let mut buf = vec![0u8; 65536];
    let mut acc: Vec<u8> = Vec::new();
    loop {
        let n = match stream.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        acc.extend_from_slice(&buf[..n]);
        match drain_requests(&acc, &mut stream, &resp).await {
            Ok(off) => acc.drain(..off),
            Err(_) => break,
        };
    }
}

/// Bind `--host:--port` (and `--uds`, if set) and serve the static fastmock
/// payload until the process exits, ignoring every other configured
/// behavior.
pub async fn run(config: &MockServerConfig) -> anyhow::Result<()> {
    let resp = Arc::new(build_responses());

    let host: std::net::IpAddr = config
        .host
        .parse()
        .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
    let addr = SocketAddr::new(host, config.port);
    let listener = build_listener(addr)?;
    tracing::info!(%addr, "fastmock listening (static payload, no simulation)");

    #[cfg(unix)]
    if let Some(uds_path) = config.uds.clone() {
        let uds_listener = bind_unix_listener(&uds_path)?;
        let uds_resp = resp.clone();
        tracing::info!(uds_path = %uds_path, "fastmock listening (uds)");
        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = uds_listener.accept().await else {
                    continue;
                };
                let resp = uds_resp.clone();
                tokio::spawn(handle(stream, resp));
            }
        });
    }

    loop {
        let (stream, _peer) = listener.accept().await?;
        stream.set_nodelay(true).ok();
        let resp = resp.clone();
        tokio::spawn(handle(stream, resp));
    }
}
