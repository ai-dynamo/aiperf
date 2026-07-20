// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Passthrough logging HTTP proxy that also translates between the AWS
//! SageMaker Runtime wire protocol and a plain OpenAI-compatible chat
//! completions backend.
//!
//! Usage:
//!   UPSTREAM=http://vllm-server:8000 LISTEN=0.0.0.0:9000 aiperf-sagemaker-logproxy
//!
//! Requests to `/endpoints/<name>/invocations` or
//! `/endpoints/<name>/invocations-response-stream` (the paths boto3's
//! `invoke_endpoint`/`invoke_endpoint_with_response_stream` hit) are
//! rewritten to `UPSTREAM/v1/chat/completions` and forwarded unchanged
//! otherwise - the request body a `--transport sagemaker` client sends is
//! already an OpenAI-shaped chat completions JSON payload, so no body
//! conversion is needed on the way in.
//!
//! On the way back, non-streaming responses pass through verbatim (plain
//! JSON, matching `InvokeEndpoint`'s response contract). Streaming
//! responses are re-framed: real SageMaker containers emit
//! `application/vnd.amazon.eventstream`-framed binary messages (AWS's own
//! binary event framing) where each message's payload is one raw SSE
//! `data: ...\n\n` chunk from the underlying container - not plain SSE text
//! like a normal OpenAI-compatible server. `SageMakerTransport` in aiperf
//! decodes that framing and unwraps the SSE bytes back out, so to make a
//! plain vLLM/OpenAI-compatible server look like a real SageMaker endpoint,
//! this proxy re-segments vLLM's raw SSE byte stream on message boundaries
//! (`\n\n`) and wraps each complete `data: ...\n\n` chunk as one
//! `PayloadPart` event-stream message before writing it to the client.
//!
//! Any other path (e.g. `/metrics`) is forwarded byte-for-byte with no
//! translation, identical to `logproxy`.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use bytes::{Bytes, BytesMut};
use http_body_util::{BodyExt, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::header::{CONTENT_LENGTH, CONTENT_TYPE};
use hyper::service::service_fn;
use hyper::{HeaderMap, Request, Response, StatusCode, Uri};
use hyper_util::client::legacy::Client;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use serde::Serialize;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

type ClientBody = http_body_util::combinators::BoxBody<Bytes, hyper::Error>;

static REQ_COUNTER: AtomicU64 = AtomicU64::new(0);

const EVENTSTREAM_CONTENT_TYPE: &str = "application/vnd.amazon.eventstream";

/// Wall-clock milliseconds since the Unix epoch, for time-series analysis of
/// the log (the per-event `*_ms` fields are all relative to each request's
/// own start, which is enough for per-request timing but not for plotting
/// trends across the run).
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// One JSONL record, mirroring `logproxy`'s format with an added
/// `sagemaker` flag on the request event so translated vs. passthrough
/// requests are distinguishable in the log. Every variant carries `ts_ms`
/// (wall-clock time the event was recorded) so a log spanning a whole
/// benchmark run can be sliced into time buckets for trend analysis.
#[derive(Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum LogEvent {
    Request {
        req_id: u64,
        ts_ms: u64,
        method: String,
        path: String,
        upstream_path: String,
        sagemaker: bool,
        peer: String,
        headers: Vec<(String, String)>,
        body: String,
    },
    ResponseHead {
        req_id: u64,
        ts_ms: u64,
        status: u16,
        ttfb_ms: f64,
    },
    FirstDataChunk {
        req_id: u64,
        ts_ms: u64,
        elapsed_ms: f64,
    },
    Done {
        req_id: u64,
        ts_ms: u64,
        frames: u64,
        bytes: u64,
        total_ms: f64,
    },
    Error {
        req_id: u64,
        ts_ms: u64,
        stage: &'static str,
        message: String,
    },
}

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
                    eprintln!("sagemaker-logproxy: failed to serialize log event: {e}");
                }
            }
        }
    });
    tx
}

/// The two SageMaker Runtime invocation path shapes we translate.
/// `/endpoints/<name>/invocations` -> non-streaming `InvokeEndpoint`.
/// `/endpoints/<name>/invocations-response-stream` -> streaming
/// `InvokeEndpointWithResponseStream`. The endpoint name itself is not
/// used for routing (this proxy has exactly one upstream), only for
/// recognizing the shape.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RouteKind {
    Passthrough,
    SageMakerInvoke,
    SageMakerInvokeStream,
}

fn classify_path(path: &str) -> RouteKind {
    let Some(rest) = path.strip_prefix("/endpoints/") else {
        return RouteKind::Passthrough;
    };
    // rest is "<name>/invocations[-response-stream]", possibly with a
    // trailing query string already stripped by the caller (we classify
    // path-only, query is preserved separately when rewriting the URI).
    //
    // Split on the LAST '/', not the first: some clients (notably rust
    // aiperf's `--endpoint-type sagemaker`, which substitutes the
    // request's *model name* into the `{model_name}` path template rather
    // than a real SageMaker endpoint name) put a literal '/' inside the
    // "name" segment itself, e.g. `/endpoints/google/gemma-4-31B-it/
    // invocations-response-stream`. A first-'/' split would misparse that
    // as name="google", tail="gemma-4-31B-it/invocations-response-stream"
    // and fall through to Passthrough (404 against a plain chat backend).
    // Splitting on the last '/' always isolates the true tail regardless
    // of how many '/' the endpoint-name portion contains.
    let Some((_name, tail)) = rest.rsplit_once('/') else {
        return RouteKind::Passthrough;
    };
    match tail {
        "invocations" => RouteKind::SageMakerInvoke,
        "invocations-response-stream" => RouteKind::SageMakerInvokeStream,
        _ => RouteKind::Passthrough,
    }
}

#[tokio::main]
async fn main() {
    let upstream =
        std::env::var("UPSTREAM").unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());
    let listen = std::env::var("LISTEN").unwrap_or_else(|_| "0.0.0.0:9000".to_string());
    // The OpenAI-compatible path to rewrite SageMaker invocation paths to.
    let chat_path =
        std::env::var("UPSTREAM_CHAT_PATH").unwrap_or_else(|_| "/v1/chat/completions".to_string());

    let upstream_uri: Uri = upstream.parse().expect("invalid UPSTREAM url");
    let upstream_authority = upstream_uri
        .authority()
        .expect("UPSTREAM must include host[:port]")
        .to_string();
    let upstream_scheme = upstream_uri.scheme_str().unwrap_or("http").to_string();

    eprintln!(
        "sagemaker-logproxy: listening on {listen}, forwarding to {upstream} \
         (SageMaker invocation paths -> {chat_path})"
    );

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
        let chat_path = chat_path.clone();
        let log_tx = log_tx.clone();

        tokio::spawn(async move {
            let service = service_fn(move |req: Request<Incoming>| {
                proxy(
                    req,
                    client.clone(),
                    upstream_authority.clone(),
                    upstream_scheme.clone(),
                    chat_path.clone(),
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
    chat_path: String,
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
    let path_only = req.uri().path();
    let route = classify_path(path_only);

    // For SageMaker-shaped paths, forward to the OpenAI chat path instead of
    // the original path; everything else (query string included) passes
    // through untouched.
    let upstream_path_and_query = match route {
        RouteKind::Passthrough => path_and_query.clone(),
        RouteKind::SageMakerInvoke | RouteKind::SageMakerInvokeStream => {
            match req.uri().query() {
                Some(q) => format!("{chat_path}?{q}"),
                None => chat_path.clone(),
            }
        }
    };

    let new_uri = format!("{upstream_scheme}://{upstream_authority}{upstream_path_and_query}");
    let new_uri: Uri = match new_uri.parse() {
        Ok(u) => u,
        Err(e) => {
            let _ = log_tx.send(LogEvent::Error {
                ts_ms: now_ms(),
                req_id,
                stage: "bad_uri",
                message: format!("{new_uri}: {e}"),
            });
            return Ok(error_response(400));
        }
    };

    let (mut parts, body) = req.into_parts();

    let body_bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            let _ = log_tx.send(LogEvent::Error {
                ts_ms: now_ms(),
                req_id,
                stage: "request_body_read",
                message: e.to_string(),
            });
            return Ok(error_response(400));
        }
    };

    let _ = log_tx.send(LogEvent::Request {
        ts_ms: now_ms(),
        req_id,
        method: method.to_string(),
        path: path_and_query.clone(),
        upstream_path: upstream_path_and_query.clone(),
        sagemaker: route != RouteKind::Passthrough,
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
                ts_ms: now_ms(),
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
        ts_ms: now_ms(),
        req_id,
        status: status.as_u16(),
        ttfb_ms: ttfb.as_secs_f64() * 1000.0,
    });

    let (mut resp_parts, incoming) = resp.into_parts();

    // Only re-frame as an eventstream when the upstream actually succeeded
    // and streaming was requested. A non-2xx status means the real boto3
    // client would raise before ever touching `response['Body']` as an
    // EventStream, so error bodies pass through as plain JSON/text
    // regardless of route - translating them would just corrupt them.
    let wrap_as_eventstream =
        route == RouteKind::SageMakerInvokeStream && resp_parts.status.is_success();

    if wrap_as_eventstream {
        resp_parts.headers.remove(CONTENT_LENGTH);
        resp_parts
            .headers
            .insert(CONTENT_TYPE, EVENTSTREAM_CONTENT_TYPE.parse().unwrap());
    }

    let (tx, rx) = mpsc::unbounded_channel::<Result<Frame<Bytes>, hyper::Error>>();
    let done_log_tx = log_tx.clone();

    tokio::spawn(async move {
        if wrap_as_eventstream {
            pump_as_eventstream(incoming, tx, done_log_tx, req_id, start).await;
        } else {
            pump_passthrough(incoming, tx, done_log_tx, req_id, start).await;
        }
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
    let body = StreamBody::new(stream).boxed();
    let response = Response::from_parts(resp_parts, body);

    Ok(response)
}

/// Byte-for-byte relay, identical to `logproxy`'s streaming path.
async fn pump_passthrough(
    mut incoming: Incoming,
    tx: mpsc::UnboundedSender<Result<Frame<Bytes>, hyper::Error>>,
    log_tx: mpsc::UnboundedSender<LogEvent>,
    req_id: u64,
    start: Instant,
) {
    let mut first_data_at: Option<Instant> = None;
    let mut total_bytes: u64 = 0;
    let mut frames: u64 = 0;

    while let Some(frame_result) = incoming.frame().await {
        match frame_result {
            Ok(frame) => {
                if let Some(data) = frame.data_ref() {
                    if first_data_at.is_none() {
                        first_data_at = Some(Instant::now());
                        let _ = log_tx.send(LogEvent::FirstDataChunk {
                            ts_ms: now_ms(),
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
                let _ = log_tx.send(LogEvent::Error {
                    ts_ms: now_ms(),
                    req_id,
                    stage: "body_stream",
                    message: e.to_string(),
                });
                let _ = tx.send(Err(e));
                break;
            }
        }
    }

    let _ = log_tx.send(LogEvent::Done {
        ts_ms: now_ms(),
        req_id,
        frames,
        bytes: total_bytes,
        total_ms: start.elapsed().as_secs_f64() * 1000.0,
    });
}

/// Re-segments the upstream's raw SSE byte stream on `\n\n` message
/// boundaries and emits one AWS event-stream `PayloadPart` message per
/// complete `data: ...\n\n` chunk, matching how real SageMaker containers
/// frame their streaming output (one line per PayloadPart - see the
/// docstring on `SageMakerTransport._chunks_as_sse_frames` on the aiperf
/// side for the corresponding decode-side assumption).
#[allow(unused_assignments)] // `first_data_at`'s final write (trailing flush) is intentionally unread
async fn pump_as_eventstream(
    mut incoming: Incoming,
    tx: mpsc::UnboundedSender<Result<Frame<Bytes>, hyper::Error>>,
    log_tx: mpsc::UnboundedSender<LogEvent>,
    req_id: u64,
    start: Instant,
) {
    let mut first_data_at: Option<Instant> = None;
    let mut total_bytes: u64 = 0;
    let mut frames: u64 = 0;
    let mut buffer = BytesMut::new();

    macro_rules! emit_message {
        ($payload:expr) => {{
            let msg = encode_payload_part($payload);
            total_bytes += msg.len() as u64;
            frames += 1;
            if first_data_at.is_none() {
                first_data_at = Some(Instant::now());
                let _ = log_tx.send(LogEvent::FirstDataChunk {
                    ts_ms: now_ms(),
                    req_id,
                    elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                });
            }
            if tx.send(Ok(Frame::data(msg))).is_err() {
                return;
            }
        }};
    }

    while let Some(frame_result) = incoming.frame().await {
        match frame_result {
            Ok(frame) => {
                let Some(data) = frame.data_ref() else {
                    continue;
                };
                buffer.extend_from_slice(data);
                // Flush every complete "...\n\n"-terminated SSE message as
                // its own PayloadPart, keeping any trailing partial message
                // buffered for the next chunk.
                while let Some(boundary) = find_double_newline(&buffer) {
                    let message = buffer.split_to(boundary + 2);
                    emit_message!(message.freeze());
                }
            }
            Err(e) => {
                let _ = log_tx.send(LogEvent::Error {
                    ts_ms: now_ms(),
                    req_id,
                    stage: "body_stream",
                    message: e.to_string(),
                });
                let _ = tx.send(Err(e));
                return;
            }
        }
    }

    // Flush a trailing message that never got a closing blank line (e.g.
    // the upstream closed the connection right after the final `data:`
    // line without the usual terminator).
    if !buffer.is_empty() {
        emit_message!(buffer.freeze());
    }

    let _ = log_tx.send(LogEvent::Done {
        ts_ms: now_ms(),
        req_id,
        frames,
        bytes: total_bytes,
        total_ms: start.elapsed().as_secs_f64() * 1000.0,
    });
}

/// Finds the byte offset of the first `\n\n` in `buf`, if any.
fn find_double_newline(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}

/// AWS event-stream header value types (only the string type is used here).
const HEADER_TYPE_STRING: u8 = 7;

/// Appends one string-valued event-stream header (`name-len:1 name type:1
/// value-len:2 value`) to `buf`, per the `application/vnd.amazon.eventstream`
/// binary framing spec.
fn write_string_header(buf: &mut Vec<u8>, name: &str, value: &str) {
    let name_bytes = name.as_bytes();
    debug_assert!(name_bytes.len() <= u8::MAX as usize);
    buf.push(name_bytes.len() as u8);
    buf.extend_from_slice(name_bytes);
    buf.push(HEADER_TYPE_STRING);
    let value_bytes = value.as_bytes();
    debug_assert!(value_bytes.len() <= u16::MAX as usize);
    buf.extend_from_slice(&(value_bytes.len() as u16).to_be_bytes());
    buf.extend_from_slice(value_bytes);
}

/// Encodes one `PayloadPart` event-stream message wrapping `payload`
/// verbatim as its body. Wire format (big-endian throughout):
///   total_len:4  headers_len:4  prelude_crc:4  headers  payload  message_crc:4
/// `prelude_crc` covers only the first 8 bytes (total_len + headers_len);
/// `message_crc` covers everything from the start of the message through
/// the end of the payload. Both CRCs are CRC-32 (IEEE 802.3), matching
/// `crc32fast`'s default (and `zlib.crc32`/botocore's own decoder).
fn encode_payload_part(payload: Bytes) -> Bytes {
    let mut headers = Vec::new();
    write_string_header(&mut headers, ":message-type", "event");
    write_string_header(&mut headers, ":event-type", "PayloadPart");
    write_string_header(&mut headers, ":content-type", "application/octet-stream");

    let headers_len = headers.len() as u32;
    let payload_len = payload.len() as u32;
    // prelude(8) + prelude_crc(4) + headers + payload + message_crc(4)
    let total_len = 8 + 4 + headers_len + payload_len + 4;

    let mut msg = Vec::with_capacity(total_len as usize);
    msg.extend_from_slice(&total_len.to_be_bytes());
    msg.extend_from_slice(&headers_len.to_be_bytes());
    let prelude_crc = crc32fast::hash(&msg);
    msg.extend_from_slice(&prelude_crc.to_be_bytes());
    msg.extend_from_slice(&headers);
    msg.extend_from_slice(&payload);
    let message_crc = crc32fast::hash(&msg);
    msg.extend_from_slice(&message_crc.to_be_bytes());

    Bytes::from(msg)
}

fn error_response(code: u16) -> Response<ClientBody> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, "text/plain".parse().unwrap());
    let mut builder = Response::builder().status(
        StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
    );
    if let Some(h) = builder.headers_mut() {
        *h = headers;
    }
    builder
        .body(
            http_body_util::Full::new(Bytes::from_static(b"proxy error"))
                .map_err(|never| match never {})
                .boxed(),
        )
        .unwrap()
}
