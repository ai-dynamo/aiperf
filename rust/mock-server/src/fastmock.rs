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
    completions: Arc<Vec<u8>>,
    models: Arc<Vec<u8>>,
}

fn build_responses() -> Responses {
    let body = b"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"x\"}}]}\n\ndata: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n",
        body.len()
    );
    let chat: Arc<Vec<u8>> = Arc::new([head.as_bytes(), body].concat());

    // Completions streams `choices[].text` under `object: text_completion`,
    // not `choices[].delta.content` -- the runtime's CompletionsEndpoint
    // refuses any other `object`, and an empty `text` parses to no response
    // data at all.
    let completions_body = b"data: {\"id\":\"x\",\"object\":\"text_completion\",\"created\":0,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"text\":\"x\",\"finish_reason\":null}]}\n\ndata: {\"id\":\"x\",\"object\":\"text_completion\",\"created\":0,\"model\":\"mock-model\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    let chead = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n",
        completions_body.len()
    );
    let completions: Arc<Vec<u8>> = Arc::new([chead.as_bytes(), completions_body].concat());

    let models = b"{\"object\":\"list\",\"data\":[{\"id\":\"mock-model\",\"object\":\"model\"}]}";
    let mhead = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n",
        models.len()
    );
    let models_resp: Arc<Vec<u8>> = Arc::new([mhead.as_bytes(), models.as_ref()].concat());

    Responses {
        chat,
        completions,
        models: models_resp,
    }
}

/// True when the request line targets the OpenAI completions path.
///
/// Scans the request line only: a header value (`Referer`, say) could otherwise
/// carry the path and mis-route the reply. `/v1/chat/completions` does not
/// contain `/v1/completions` as a substring, so the two paths never collide.
fn is_completions(head: &[u8]) -> bool {
    let line = head.split(|&b| b == b'\n').next().unwrap_or(head);
    find(line, b"/v1/completions").is_some()
}

/// Parses and answers every complete request found in `chunk` starting at
/// offset 0. Returns the byte offset past the last complete request handled
/// (== chunk.len() when nothing is left over).
fn drain_requests<S: Write>(
    chunk: &[u8],
    stream: &mut S,
    resp: &Responses,
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
        let reply = if head.starts_with(b"GET") {
            &resp.models
        } else if is_completions(head) {
            &resp.completions
        } else {
            &resp.chat
        };
        if stream.write_all(reply).is_err() {
            return Err(());
        }
        off += total;
    }
    Ok(off)
}

fn handle<S: Read + Write>(mut stream: S, resp: Arc<Responses>) {
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
            let off = match drain_requests(&buf[..n], &mut stream, &resp) {
                Ok(off) => off,
                Err(()) => return,
            };
            if off < n {
                acc.extend_from_slice(&buf[off..n]);
            }
        } else {
            acc.extend_from_slice(&buf[..n]);
            let off = match drain_requests(&acc, &mut stream, &resp) {
                Ok(off) => off,
                Err(()) => return,
            };
            acc.drain(..off);
        }
    }
}

/// A blocking stream listener the fastmock accept loop can run over, abstracting
/// `TcpListener`/`UnixListener` so the accept and serve logic is written once.
/// Each implementation applies its own transport tuning (TCP sets
/// `TCP_NODELAY`; the Unix socket has no such knob) inside [`accept_conn`].
///
/// [`accept_conn`]: FastListener::accept_conn
trait FastListener: Send + Sync + 'static {
    /// The accepted connection stream type (`TcpStream` / `UnixStream`).
    type Stream: Read + Write + Send + 'static;

    /// Accept the next connection, applying transport tuning. Errors are
    /// returned so the caller can skip a failed accept and keep looping, exactly
    /// as `Incoming` iteration did.
    fn accept_conn(&self) -> std::io::Result<Self::Stream>;
}

impl FastListener for TcpListener {
    type Stream = std::net::TcpStream;

    fn accept_conn(&self) -> std::io::Result<Self::Stream> {
        let (stream, _peer) = self.accept()?;
        stream.set_nodelay(true).ok();
        Ok(stream)
    }
}

#[cfg(unix)]
impl FastListener for std::os::unix::net::UnixListener {
    type Stream = std::os::unix::net::UnixStream;

    fn accept_conn(&self) -> std::io::Result<Self::Stream> {
        let (stream, _peer) = self.accept()?;
        Ok(stream)
    }
}

/// The kernel serializes concurrent `accept` calls on the shared listener.
fn accept_loop<L: FastListener>(listener: Arc<L>, resp: Arc<Responses>) {
    loop {
        let Ok(stream) = listener.accept_conn() else {
            continue;
        };
        let resp = resp.clone();
        thread::spawn(move || handle(stream, resp));
    }
}

fn serve<L: FastListener>(listener: L, threads: usize) {
    let listener = Arc::new(listener);
    let resp = Arc::new(build_responses());
    let threads = threads.max(1);
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let l = listener.clone();
        let r = resp.clone();
        handles.push(thread::spawn(move || accept_loop(l, r)));
    }
    for h in handles {
        let _ = h.join();
    }
}

fn auto_parallelism() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Bind `--host:--port` (and `--uds`, if set) and serve the static fastmock
/// payload until the process exits, ignoring every other configured
/// behavior. Blocking — call from `main` before any tokio runtime is built;
/// this never touches async.
/// Warn loudly on every startup path: this mode trades every behavioral
/// guarantee the real mock server makes for raw socket throughput, and a
/// quiet one-line log is easy to miss/forget once a session is scrolled past.
fn print_warning_banner() {
    eprintln!(
        "\n\
         \x1b[1;31m########################################################################\x1b[0m\n\
         \x1b[1;31m#\x1b[0m \x1b[1;33m⚠ WARNING: --ludicrous-speed / --plaid is NOT a realistic mock server\x1b[0m\n\
         \x1b[1;31m#\x1b[0m\n\
         \x1b[1;31m#\x1b[0m This is a RAW THROUGHPUT / EXTREME LOAD TEST ONLY.\n\
         \x1b[1;31m#\x1b[0m Every response is a single HARD-CODED payload:\n\
         \x1b[1;31m#\x1b[0m   - no latency simulation, no routing, no token rendering\n\
         \x1b[1;31m#\x1b[0m   - no endpoint dispatch, no per-request behavior of any kind\n\
         \x1b[1;31m#\x1b[0m   - request contents are ignored entirely\n\
         \x1b[1;31m#\x1b[0m\n\
         \x1b[1;31m#\x1b[0m Do NOT use this for behavioral, correctness, or benchmark-accuracy\n\
         \x1b[1;31m#\x1b[0m testing. Use it ONLY to saturate a transport/client path at the\n\
         \x1b[1;31m#\x1b[0m absolute ceiling of what the wire can carry.\n\
         \x1b[1;31m########################################################################\x1b[0m\n"
    );
}

pub fn run(config: &MockServerConfig) -> anyhow::Result<()> {
    print_warning_banner();
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
        serve(listener, threads);
        return Ok(());
    }

    let listener = TcpListener::bind(format!("{}:{}", config.host, config.port))?;
    println!(
        "fastmock listening on {}:{} ({threads} accept threads)",
        config.host, config.port
    );
    serve(listener, threads);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completions_path_routes_separately_from_chat() {
        assert!(is_completions(b"POST /v1/completions HTTP/1.1\r\n"));
        assert!(!is_completions(b"POST /v1/chat/completions HTTP/1.1\r\n"));
    }

    /// The scan is confined to the request line, so a header that happens to
    /// carry the completions path cannot steal a chat reply.
    #[test]
    fn a_header_carrying_the_path_does_not_misroute() {
        let head = b"POST /v1/chat/completions HTTP/1.1\r\nReferer: http://h/v1/completions\r\n";
        assert!(!is_completions(head));
    }

    /// Each payload must carry the `object` its runtime parser demands:
    /// CompletionsEndpoint accepts only `completion`/`text_completion` and
    /// reads `choices[].text`, while chat reads `choices[].delta.content`.
    /// A swapped discriminator parses to zero response data, which shows up as
    /// a run with no output tokens rather than as an error.
    #[test]
    fn each_payload_carries_the_object_its_parser_requires() {
        let responses = build_responses();
        let completions = String::from_utf8_lossy(&responses.completions).into_owned();
        assert!(completions.contains(r#""object":"text_completion""#));
        assert!(completions.contains(r#""text":"x""#));

        let chat = String::from_utf8_lossy(&responses.chat).into_owned();
        assert!(chat.contains(r#""object":"chat.completion.chunk""#));
    }
}
