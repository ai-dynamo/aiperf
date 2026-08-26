// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Passthrough logging HTTP proxy.
//!
//! Usage:
//!   UPSTREAM=http://vllm-server:8000 LISTEN=0.0.0.0:9000 logproxy
//!
//! Forwards every request byte-for-byte to `UPSTREAM` and streams the
//! response back without buffering (so SSE/chunked streaming responses pass
//! through live, preserving TTFT/ITL timing). Logs several JSONL records per
//! request: the request method/path/headers/body, the response status and
//! time-to-first-byte, the first data chunk, and a final frame count, response
//! byte count, and total duration. Gives multiple client tools (aiperf,
//! locust, ...) an identical, independently-logged view of the same upstream
//! server for apples-to-apples benchmark comparisons.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use bytes::Bytes;
use http_body_util::{BodyExt, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::service::service_fn;
use hyper::{Request, Response, Uri};
use hyper_util::client::legacy::Client;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use serde::Serialize;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

type ClientBody = http_body_util::combinators::BoxBody<Bytes, hyper::Error>;

static REQ_COUNTER: AtomicU64 = AtomicU64::new(0);

/// One JSONL record. `#[serde(tag = "event")]` gives every line an `"event"`
/// field so downstream tooling (jq, etc.) can filter by record type without
/// separate schemas per event.
#[derive(Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum LogEvent {
    Request {
        req_id: u64,
        method: String,
        path: String,
        peer: String,
        headers: Vec<(String, String)>,
        body: String,
    },
    ResponseHead {
        req_id: u64,
        status: u16,
        ttfb_ms: f64,
    },
    FirstDataChunk {
        req_id: u64,
        elapsed_ms: f64,
    },
    Done {
        req_id: u64,
        frames: u64,
        bytes: u64,
        total_ms: f64,
    },
    Error {
        req_id: u64,
        stage: &'static str,
        message: String,
    },
}

/// Spawns the single dedicated writer task that owns stderr for JSONL output.
/// All request-handling tasks only ever push events into this channel (a
/// cheap, non-blocking send), so serialization + the actual write/flush never
/// happen on the hot path.
fn spawn_log_writer() -> mpsc::UnboundedSender<LogEvent> {
    let (tx, mut rx) = mpsc::unbounded_channel::<LogEvent>();
    tokio::spawn(async move {
        use tokio::io::AsyncWriteExt;
        let mut stderr = tokio::io::stderr();
        while let Some(event) = rx.recv().await {
            match serde_json::to_string(&event) {
                Ok(mut line) => {
                    line.push('\n');
                    if stderr.write_all(line.as_bytes()).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("logproxy: failed to serialize log event: {e}");
                }
            }
        }
    });
    tx
}

#[tokio::main]
async fn main() {
    let upstream =
        std::env::var("UPSTREAM").unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());
    let listen = std::env::var("LISTEN").unwrap_or_else(|_| "0.0.0.0:9000".to_string());

    let upstream_uri: Uri = upstream.parse().expect("invalid UPSTREAM url");
    let upstream_authority = upstream_uri
        .authority()
        .expect("UPSTREAM must include host[:port]")
        .to_string();
    let upstream_scheme = upstream_uri.scheme_str().unwrap_or("http").to_string();

    eprintln!("logproxy: listening on {listen}, forwarding to {upstream}");

    let addr: SocketAddr = listen.parse().expect("invalid LISTEN addr");
    let listener = TcpListener::bind(addr).await.expect("bind failed");

    let client: Client<_, ClientBody> = Client::builder(TokioExecutor::new()).build_http();
    let log_tx = spawn_log_writer();

    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                eprintln!("accept error: {e}");
                continue;
            }
        };
        let io = TokioIo::new(stream);
        let client = client.clone();
        let upstream_authority = upstream_authority.clone();
        let upstream_scheme = upstream_scheme.clone();
        let log_tx = log_tx.clone();

        tokio::spawn(async move {
            let service = service_fn(move |req: Request<Incoming>| {
                proxy(
                    req,
                    client.clone(),
                    upstream_authority.clone(),
                    upstream_scheme.clone(),
                    peer,
                    log_tx.clone(),
                )
            });

            if let Err(e) = auto::Builder::new(TokioExecutor::new())
                .serve_connection(io, service)
                .await
            {
                eprintln!("conn error ({peer}): {e}");
            }
        });
    }
}

async fn proxy(
    req: Request<Incoming>,
    client: Client<hyper_util::client::legacy::connect::HttpConnector, ClientBody>,
    upstream_authority: String,
    upstream_scheme: String,
    peer: SocketAddr,
    log_tx: mpsc::UnboundedSender<LogEvent>,
) -> Result<Response<ClientBody>, Infallible> {
    let req_id = REQ_COUNTER.fetch_add(1, Ordering::Relaxed);
    let start = Instant::now();
    let method = req.method().clone();
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| "/".to_string());

    // Rebuild the URI pointing at the upstream host, keeping path+query as-is.
    let new_uri = format!("{upstream_scheme}://{upstream_authority}{path_and_query}");
    let new_uri: Uri = match new_uri.parse() {
        Ok(u) => u,
        Err(e) => {
            let _ = log_tx.send(LogEvent::Error {
                req_id,
                stage: "bad_uri",
                message: format!("{new_uri}: {e}"),
            });
            return Ok(error_response(400));
        }
    };

    let (mut parts, body) = req.into_parts();

    // Buffer the full request body (client payloads are small, single-shot
    // JSON, so buffering here is fine unlike the streamed response path
    // below) before logging, so a slow/blocked stderr writer never delays
    // forwarding the body upstream.
    let body_bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            let _ = log_tx.send(LogEvent::Error {
                req_id,
                stage: "request_body_read",
                message: e.to_string(),
            });
            return Ok(error_response(400));
        }
    };

    // Push the captured request (line + headers + body) into the log channel
    // and move on immediately — serialization and the actual write happen on
    // the dedicated writer task, never blocking the forwarding path below.
    let _ = log_tx.send(LogEvent::Request {
        req_id,
        method: method.to_string(),
        path: path_and_query.clone(),
        peer: peer.to_string(),
        headers: parts
            .headers
            .iter()
            .map(|(name, value)| {
                (
                    name.to_string(),
                    value.to_str().unwrap_or("<non-utf8>").to_string(),
                )
            })
            .collect(),
        body: String::from_utf8_lossy(&body_bytes).into_owned(),
    });

    parts.uri = new_uri;
    parts.headers.remove(hyper::header::HOST);
    let upstream_req = Request::from_parts(
        parts,
        http_body_util::Full::new(body_bytes)
            .map_err(|never: Infallible| match never {})
            .boxed(),
    );

    let resp = match client.request(upstream_req).await {
        Ok(r) => r,
        Err(e) => {
            let _ = log_tx.send(LogEvent::Error {
                req_id,
                stage: "upstream_request",
                message: format!("after {:?}: {e}", start.elapsed()),
            });
            return Ok(error_response(502));
        }
    };

    let ttfb = start.elapsed();
    let status = resp.status();
    let _ = log_tx.send(LogEvent::ResponseHead {
        req_id,
        status: status.as_u16(),
        ttfb_ms: ttfb.as_secs_f64() * 1000.0,
    });

    let (parts, mut incoming) = resp.into_parts();

    // Stream the body through chunk-by-chunk, logging first-data-frame and
    // final byte/frame counts as they occur, without buffering the whole thing.
    let (tx, rx) = mpsc::unbounded_channel::<Result<Frame<Bytes>, hyper::Error>>();
    let done_log_tx = log_tx.clone();

    tokio::spawn(async move {
        let mut first_data_at: Option<Instant> = None;
        let mut total_bytes: u64 = 0;
        let mut frames: u64 = 0;

        while let Some(frame_result) = incoming.frame().await {
            match frame_result {
                Ok(frame) => {
                    if let Some(data) = frame.data_ref() {
                        if first_data_at.is_none() {
                            first_data_at = Some(Instant::now());
                            let _ = done_log_tx.send(LogEvent::FirstDataChunk {
                                req_id,
                                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                            });
                        }
                        total_bytes += data.len() as u64;
                        frames += 1;
                    }
                    if tx.send(Ok(frame)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = done_log_tx.send(LogEvent::Error {
                        req_id,
                        stage: "body_stream",
                        message: e.to_string(),
                    });
                    let _ = tx.send(Err(e));
                    break;
                }
            }
        }

        let _ = done_log_tx.send(LogEvent::Done {
            req_id,
            frames,
            bytes: total_bytes,
            total_ms: start.elapsed().as_secs_f64() * 1000.0,
        });
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
    let body = StreamBody::new(stream).boxed();
    let response = Response::from_parts(parts, body);

    Ok(response)
}

fn error_response(code: u16) -> Response<ClientBody> {
    Response::builder()
        .status(code)
        .body(
            http_body_util::Full::new(Bytes::from_static(b"proxy error"))
                .map_err(|never| match never {})
                .boxed(),
        )
        .unwrap()
}
