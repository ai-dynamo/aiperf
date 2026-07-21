// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `--ludicrous-speed`: a minimal, fixed-response server that replaces the
//! real mock server entirely.
//!
//! This is the standalone `tools/fastmock.rs` binary's serving logic,
//! unchanged, wired to run from inside the `aiperf-mock-server` process
//! instead of a separate `rustc -O` build. It intentionally does NOT use
//! tokio/async: raw blocking `std::net` + thread-per-accept-loop is what
//! makes the standalone tool fast, so bringing it in as an async task on the
//! shared tokio runtime would reintroduce the scheduling overhead this
//! bypass exists to avoid. Every request gets the same pre-built static
//! payload — a single streamed chat-completion chunk for anything other than
//! a bare `GET`, and a static model list for `GET`. There is no latency
//! simulation, routing, token rendering, or endpoint dispatch.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::thread;

use crate::MockServerConfig;

fn find(h: &[u8], n: &[u8]) -> Option<usize> {
    h.windows(n.len()).position(|w| w == n)
}

/// Byte-level, allocation-free header scan. The naive version this replaced
/// (`String::from_utf8_lossy(head).to_lowercase()`) allocated two Strings on
/// every single request — at millions of req/s that's millions of heap
/// allocations/sec of pure overhead versus this zero-alloc version.
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

struct Responses {
    chat: Arc<Vec<u8>>,
    models: Arc<Vec<u8>>,
}

fn build_responses() -> Responses {
    let body = b"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"x\"}}]}\n\ndata: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    let head = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n", body.len());
    let chat: Arc<Vec<u8>> = Arc::new([head.as_bytes(), body].concat());

    let models = b"{\"object\":\"list\",\"data\":[{\"id\":\"mock-model\",\"object\":\"model\"}]}";
    let mhead = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n", models.len());
    let models_resp: Arc<Vec<u8>> = Arc::new([mhead.as_bytes(), models.as_ref()].concat());

    Responses {
        chat,
        models: models_resp,
    }
}

/// Parses and answers every complete request found in `chunk` starting at
/// offset 0. Returns the byte offset past the last complete request handled
/// (== chunk.len() when nothing is left over).
fn drain_requests<S: Write>(
    chunk: &[u8],
    stream: &mut S,
    chat: &Arc<Vec<u8>>,
    models: &Arc<Vec<u8>>,
) -> Result<usize, ()> {
    let mut off = 0usize;
    loop {
        let rest = &chunk[off..];
        let Some(hpos) = find(rest, b"\r\n\r\n") else {
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
        let resp = if head.starts_with(b"GET") { models } else { chat };
        if stream.write_all(resp).is_err() {
            return Err(());
        }
        off += total;
    }
    Ok(off)
}

fn handle<S: Read + Write>(mut stream: S, chat: Arc<Vec<u8>>, models: Arc<Vec<u8>>) {
    let mut buf = vec![0u8; 65536];
    // Only populated when a request spans more than one `read()` call (rare
    // at pipeline depth 1: the client waits for a response before sending
    // its next request, so a read almost always contains exactly one
    // complete request already). Keeping the fast path copy-free avoids a
    // Vec extend + drain per request in the overwhelmingly common case.
    let mut acc: Vec<u8> = Vec::new();
    loop {
        let n = match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        if acc.is_empty() {
            let off = match drain_requests(&buf[..n], &mut stream, &chat, &models) {
                Ok(off) => off,
                Err(()) => return,
            };
            if off < n {
                acc.extend_from_slice(&buf[off..n]);
            }
        } else {
            acc.extend_from_slice(&buf[..n]);
            let off = match drain_requests(&acc, &mut stream, &chat, &models) {
                Ok(off) => off,
                Err(()) => return,
            };
            acc.drain(..off);
        }
    }
}

/// The kernel serializes concurrent `accept` calls on the shared listener.
fn accept_loop_tcp(listener: Arc<TcpListener>, resp: &Responses) {
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        stream.set_nodelay(true).ok();
        let chat = resp.chat.clone();
        let models = resp.models.clone();
        thread::spawn(move || handle(stream, chat, models));
    }
}

fn serve_tcp(listener: TcpListener, threads: usize) {
    let listener = Arc::new(listener);
    let resp = Arc::new(build_responses());
    let threads = threads.max(1);
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let l = listener.clone();
        let r = resp.clone();
        handles.push(thread::spawn(move || accept_loop_tcp(l, &r)));
    }
    for h in handles {
        let _ = h.join();
    }
}

#[cfg(unix)]
fn accept_loop_unix(listener: Arc<std::os::unix::net::UnixListener>, resp: &Responses) {
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let chat = resp.chat.clone();
        let models = resp.models.clone();
        thread::spawn(move || handle(stream, chat, models));
    }
}

#[cfg(unix)]
fn serve_unix(listener: std::os::unix::net::UnixListener, threads: usize) {
    let listener = Arc::new(listener);
    let resp = Arc::new(build_responses());
    let threads = threads.max(1);
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let l = listener.clone();
        let r = resp.clone();
        handles.push(thread::spawn(move || accept_loop_unix(l, &r)));
    }
    for h in handles {
        let _ = h.join();
    }
}

fn auto_parallelism() -> usize {
    thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Bind `--host:--port` (and `--uds`, if set) and serve the static fastmock
/// payload until the process exits, ignoring every other configured
/// behavior. Blocking — call from `main` before any tokio runtime is built;
/// this never touches async.
pub fn run(config: &MockServerConfig) -> anyhow::Result<()> {
    let threads = auto_parallelism();

    #[cfg(unix)]
    if let Some(uds_path) = config.uds.clone() {
        use std::os::unix::fs::FileTypeExt;
        match std::fs::symlink_metadata(&uds_path) {
            Ok(meta) if meta.file_type().is_socket() => {
                std::fs::remove_file(&uds_path)?;
            }
            Ok(_) => {
                anyhow::bail!(
                    "--uds path {uds_path} exists and is not a socket; refusing to remove it"
                );
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        let listener = std::os::unix::net::UnixListener::bind(&uds_path)?;
        println!("fastmock listening on uds:{uds_path} ({threads} accept threads)");
        serve_unix(listener, threads);
        return Ok(());
    }

    let listener = TcpListener::bind(format!("{}:{}", config.host, config.port))?;
    println!(
        "fastmock listening on {}:{} ({threads} accept threads)",
        config.host, config.port
    );
    serve_tcp(listener, threads);
    Ok(())
}
