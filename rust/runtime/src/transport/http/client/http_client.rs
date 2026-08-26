// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP request dispatch, response collection, and timing.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use http::{HeaderMap, HeaderName, HeaderValue, Method};
use http_body_util::BodyExt;
use url::Url;

use crate::clock::Clock;

use crate::transport::core::SseMessage;
use crate::transport::core::eventstream::{EventStreamDecodeError, EventStreamDecoder};
use crate::transport::core::{
    ErrorDetails, ErrorKind, RequestRecord, Response, TextResponse, TraceData,
};
use crate::transport::http::client::connection::{
    SendCompletion, Sender, TimedBody, establish, with_timeout,
};
use crate::transport::http::config::ClientConfig;
use crate::transport::http::sse::{
    SseMessageHandler, read_sse, read_sse_with_bounded_frames, read_sse_with_handler,
};

/// Re-frames an `application/vnd.amazon.eventstream` byte stream (AWS
/// SageMaker Runtime `InvokeEndpointWithResponseStream` framing) as
/// synthetic `"data: <json>\n\n"` SSE byte chunks, so the existing
/// [`read_sse_with_handler`] parser — with all of its TTFT/backpressure/
/// filter/recording logic — can consume it unmodified. AWS eventstream
/// responses have no terminal sentinel; they end at transport EOF, which
/// `read_sse_with_handler` already handles by flushing on stream close.
fn eventstream_to_sse<S>(stream: S) -> impl Stream<Item = Result<Bytes, ErrorDetails>>
where
    S: Stream<Item = Result<Bytes, ErrorDetails>> + 'static,
{
    struct State<S> {
        inner: std::pin::Pin<Box<S>>,
        decoder: EventStreamDecoder,
        pending: std::collections::VecDeque<Bytes>,
        is_terminal: bool,
    }

    let state = State {
        inner: Box::pin(stream),
        decoder: EventStreamDecoder::new(),
        pending: std::collections::VecDeque::new(),
        is_terminal: false,
    };

    futures::stream::unfold(state, |mut state| async move {
        loop {
            if state.is_terminal {
                return None;
            }
            if let Some(frame) = state.pending.pop_front() {
                return Some((Ok(frame), state));
            }
            match state.inner.next().await {
                Some(Ok(chunk)) => {
                    if let Err(error) = state.decoder.push(&chunk) {
                        state.is_terminal = true;
                        return Some((Err(eventstream_decode_error(error)), state));
                    }
                    let messages = match state.decoder.drain_messages() {
                        Ok(messages) => messages,
                        Err(error) => {
                            state.is_terminal = true;
                            return Some((Err(eventstream_decode_error(error)), state));
                        }
                    };
                    for message in messages {
                        // Real SageMaker containers (HF TGI/vLLM/LMI) put the
                        // full SSE-formatted `data: {...}` line inside
                        // PayloadPart.Bytes already; only bare-JSON payloads
                        // (no `data: ` prefix) need one synthesized here. This
                        // makes the decoder agree with AIPerf's own SageMaker
                        // transport and real AWS wire behavior either way.
                        let raw = message.payload.trim_ascii_start();
                        let inner_json = raw.strip_prefix(b"data: ").unwrap_or(raw);
                        let mut sse = bytes::BytesMut::with_capacity(inner_json.len() + 8);
                        sse.extend_from_slice(b"data: ");
                        sse.extend_from_slice(inner_json);
                        sse.extend_from_slice(b"\n\n");
                        state.pending.push_back(sse.freeze());
                    }
                }
                Some(Err(error)) => return Some((Err(error), state)),
                None if state.decoder.has_trailing_bytes() => {
                    state.is_terminal = true;
                    return Some((
                        Err(ErrorDetails::sse(
                            "truncated eventstream frame at response EOF",
                        )),
                        state,
                    ));
                }
                None => return None,
            }
        }
    })
}

fn eventstream_decode_error(error: EventStreamDecodeError) -> ErrorDetails {
    ErrorDetails::sse(format!("eventstream decode error: {error}"))
}

#[derive(Default)]
struct ChunkTiming {
    chunks: u32,
    bytes: u64,
    recv_start: Option<i64>,
    recv_end: Option<i64>,
    samples: Vec<(i64, u64)>,
}

impl ChunkTiming {
    fn observe(&mut self, timestamp_ns: i64, size: usize, collect_samples: bool) {
        self.chunks += 1;
        self.bytes += size as u64;
        if self.recv_start.is_none() {
            self.recv_start = Some(timestamp_ns);
        }
        self.recv_end = Some(timestamp_ns);
        if collect_samples {
            self.samples.push((timestamp_ns, size as u64));
        }
    }

    fn copy_to(&self, trace: &mut TraceData) {
        trace.response_receive_start_ns = self.recv_start;
        trace.response_receive_end_ns = self.recv_end;
        trace.response_chunks_count = self.chunks;
        trace.response_bytes_total = self.bytes;
        trace.response_chunks.clone_from(&self.samples);
    }
}

/// Map a response-body stream error into a transport [`ErrorDetails`].
fn body_err(e: impl std::fmt::Display) -> ErrorDetails {
    ErrorDetails::other(format!("body: {e}"))
}

fn body_limit_err(limit: u64, observed: u64) -> ErrorDetails {
    ErrorDetails::other(format!(
        "response body exceeded configured {limit}-byte limit after receiving {observed} bytes"
    ))
}

fn check_declared_body_length(
    headers: &hyper::HeaderMap,
    limit: Option<u64>,
) -> Result<(), ErrorDetails> {
    let Some(limit) = limit else {
        return Ok(());
    };
    let Some(length) = headers
        .get(hyper::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
    else {
        return Ok(());
    };
    if length > limit {
        return Err(body_limit_err(limit, length));
    }
    Ok(())
}

fn observe_body_chunk(
    observed: &mut u64,
    chunk_bytes: usize,
    limit: Option<u64>,
) -> Result<(), ErrorDetails> {
    let chunk_bytes = u64::try_from(chunk_bytes)
        .map_err(|_| ErrorDetails::other("response body chunk length exceeds u64"))?;
    *observed = observed
        .checked_add(chunk_bytes)
        .ok_or_else(|| ErrorDetails::other("response body byte count overflow"))?;
    if let Some(limit) = limit
        && *observed > limit
    {
        return Err(body_limit_err(limit, *observed));
    }
    Ok(())
}

async fn collect_body<S>(stream: S) -> Result<Bytes, ErrorDetails>
where
    S: Stream<Item = Result<Bytes, ErrorDetails>>,
{
    futures::pin_mut!(stream);
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        body.try_reserve(chunk.len())
            .map_err(|error| body_err(format!("allocating response body: {error}")))?;
        body.extend_from_slice(&chunk);
    }
    Ok(Bytes::from(body))
}

async fn drain_body<S>(stream: S) -> Result<(), ErrorDetails>
where
    S: Stream<Item = Result<Bytes, ErrorDetails>>,
{
    futures::pin_mut!(stream);
    while let Some(chunk) = stream.next().await {
        drop(chunk?);
    }
    Ok(())
}

/// Backpressured significance/observation hook for decoded SSE messages.
pub trait SseMessageFilter {
    /// Whether the reader must poll capacity between decoded frames.
    fn is_backpressured(&self) -> bool {
        true
    }

    /// Reserve downstream capacity for the next message.
    fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), ErrorDetails>>;

    /// Observe one ready message and report whether one-shot filtering is complete.
    ///
    /// Backpressured filters are still invoked for every frame; the completion
    /// signal only lets the synchronous fast path stop its first-token search.
    fn start_send(&mut self, ttft_ns: i64, message: &SseMessage) -> Result<bool, ErrorDetails>;
}

pub(crate) struct SynchronousSseMessageFilter<F>(F);

impl<F> SynchronousSseMessageFilter<F> {
    pub(crate) fn new(filter: F) -> Self {
        Self(filter)
    }
}

impl<F> SseMessageFilter for SynchronousSseMessageFilter<F>
where
    F: FnMut(i64, &SseMessage) -> bool,
{
    fn is_backpressured(&self) -> bool {
        false
    }

    fn poll_ready(&mut self, _context: &mut Context<'_>) -> Poll<Result<(), ErrorDetails>> {
        Poll::Ready(Ok(()))
    }

    fn start_send(&mut self, ttft_ns: i64, message: &SseMessage) -> Result<bool, ErrorDetails> {
        Ok((self.0)(ttft_ns, message))
    }
}

struct RecordingSseHandler<'a, F>
where
    F: SseMessageFilter + ?Sized,
{
    start_ns: i64,
    filter: &'a mut F,
    responses: &'a mut Vec<Response>,
}

impl<F> SseMessageHandler for RecordingSseHandler<'_, F>
where
    F: SseMessageFilter + ?Sized,
{
    fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), ErrorDetails>> {
        self.filter.poll_ready(context)
    }

    fn start_send(&mut self, message: SseMessage) -> Result<(), ErrorDetails> {
        let _ = self
            .filter
            .start_send(message.perf_ns - self.start_ns, &message)?;
        self.responses.push(Response::Sse(message));
        Ok(())
    }
}

/// Non-backpressured handler that records every message and stops filtering
/// after the first significant token.
///
/// The fallible handler preserves errors from external filters instead of
/// converting them into panics.
struct FirstTokenSseHandler<'a, F>
where
    F: SseMessageFilter + ?Sized,
{
    start_ns: i64,
    first_seen: bool,
    filter: &'a mut F,
    responses: &'a mut Vec<Response>,
}

/// No-record SSE handler used by bounded decision dispatch.
///
/// Unlike the ordinary streaming convenience callback, this handler can reject
/// a decoded frame. The SSE reader then stops before parsing another frame and
/// the caller drops the in-flight response rather than retaining terminal
/// response/raw-record state.
struct FallibleStreamingSseHandler<'a> {
    start_ns: i64,
    first_seen: bool,
    on_first_token: &'a mut dyn FnMut(i64),
    on_message: &'a mut dyn FnMut(&SseMessage) -> Result<bool, ErrorDetails>,
}

impl SseMessageHandler for FallibleStreamingSseHandler<'_> {
    fn poll_ready(&mut self, _context: &mut Context<'_>) -> Poll<Result<(), ErrorDetails>> {
        Poll::Ready(Ok(()))
    }

    fn start_send(&mut self, message: SseMessage) -> Result<(), ErrorDetails> {
        let is_meaningful = (self.on_message)(&message)?;
        if is_meaningful && !self.first_seen {
            self.first_seen = true;
            (self.on_first_token)(message.perf_ns - self.start_ns);
        }
        Ok(())
    }
}

impl<F> SseMessageHandler for FirstTokenSseHandler<'_, F>
where
    F: SseMessageFilter + ?Sized,
{
    fn poll_ready(&mut self, _context: &mut Context<'_>) -> Poll<Result<(), ErrorDetails>> {
        Poll::Ready(Ok(()))
    }

    fn start_send(&mut self, message: SseMessage) -> Result<(), ErrorDetails> {
        if !self.first_seen
            && self
                .filter
                .start_send(message.perf_ns - self.start_ns, &message)?
        {
            self.first_seen = true;
        }
        self.responses.push(Response::Sse(message));
        Ok(())
    }
}

pub struct HttpClient {
    clock: Rc<dyn Clock>,
    cfg: ClientConfig,
}

fn typed_headers(headers: &BTreeMap<String, String>) -> Result<HeaderMap, ErrorDetails> {
    headers
        .iter()
        .map(|(name, value)| {
            let name = HeaderName::try_from(name.as_str()).map_err(|error| {
                ErrorDetails::other(format!("invalid request header name {name:?}: {error}"))
            })?;
            let value = HeaderValue::try_from(value.as_str()).map_err(|error| {
                ErrorDetails::other(format!("invalid request header value for {name}: {error}"))
            })?;
            Ok((name, value))
        })
        .collect()
}

impl HttpClient {
    pub fn new(clock: Rc<dyn Clock>, cfg: ClientConfig) -> Self {
        Self { clock, cfg }
    }

    /// Send a POST request and record the response + timing.
    pub async fn request(
        &self,
        url: &Url,
        headers: &BTreeMap<String, String>,
        body: Bytes,
        streaming: bool,
        on_first_token: impl FnMut(i64),
    ) -> RequestRecord {
        self.request_with_method(Method::POST, url, headers, body, streaming, on_first_token)
            .await
    }

    /// Send a request with an explicit HTTP method and record response timing.
    pub async fn request_with_method(
        &self,
        method: Method,
        url: &Url,
        headers: &BTreeMap<String, String>,
        body: Bytes,
        streaming: bool,
        on_first_token: impl FnMut(i64),
    ) -> RequestRecord {
        let headers = match typed_headers(headers) {
            Ok(headers) => headers,
            Err(error) => {
                let start_ns = self.clock.now_ns();
                let mut record = RequestRecord::started(start_ns);
                record.error = Some(error);
                record.end_ns = Some(self.clock.now_ns());
                return record;
            }
        };
        let completion = Rc::new(SendCompletion::new());
        self.request_with_method_and_completion(
            method,
            url,
            &headers,
            body,
            streaming,
            on_first_token,
            completion,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn request_with_method_and_completion(
        &self,
        method: Method,
        url: &Url,
        headers: &HeaderMap,
        body: Bytes,
        streaming: bool,
        mut on_first_token: impl FnMut(i64),
        completion: Rc<SendCompletion>,
    ) -> RequestRecord {
        let start_ns = self.clock.now_ns();
        let mut record = RequestRecord::started(start_ns);
        let mut trace = TraceData {
            request_send_start_ns: Some(start_ns),
            ..TraceData::default()
        };
        let mut first_token_filter =
            SynchronousSseMessageFilter::new(|ttft_ns: i64, _message: &SseMessage| {
                on_first_token(ttft_ns);
                true
            });

        let body_len = body.len();
        let result = async {
            let (mut sender, _sock) =
                establish(url, &self.cfg, self.clock.clone(), &mut trace).await?;
            self.dispatch_with_method_and_completion(
                method,
                &mut sender,
                url,
                headers,
                body,
                streaming,
                &mut trace,
                &mut record,
                &mut first_token_filter,
                body_len,
                completion,
            )
            .await
        }
        .await;

        if let Err(e) = result {
            trace.error_timestamp_ns = Some(self.clock.now_ns());
            record.error = Some(e);
        }
        record.end_ns = Some(self.clock.now_ns());
        record.trace = Some(trace);
        record
    }

    /// Build the POST request shared by the lean no-record paths
    /// [`dispatch_streaming`](Self::dispatch_streaming) and
    /// [`dispatch_bounded_streaming_with_handler`](Self::dispatch_bounded_streaming_with_handler).
    /// Uses origin-form URI +
    /// explicit Host header so both HTTP/1.1 (Host required) and HTTP/2
    /// (`:authority` derived) work. `completion` is the shared signal that
    /// [`TimedBody`] stamps at end-of-stream (the real "send complete").
    fn build_request(
        &self,
        url: &Url,
        headers: &HeaderMap,
        body: Bytes,
        completion: Rc<SendCompletion>,
    ) -> Result<hyper::Request<TimedBody>, ErrorDetails> {
        self.build_request_with_method(Method::POST, url, headers, body, completion)
    }

    /// Build a request with an explicit method. Dataset/control-plane GETs use
    /// this path while benchmark inference keeps the POST-specialized wrappers.
    fn build_request_with_method(
        &self,
        method: Method,
        url: &Url,
        headers: &HeaderMap,
        body: Bytes,
        completion: Rc<SendCompletion>,
    ) -> Result<hyper::Request<TimedBody>, ErrorDetails> {
        let authority = url.authority();
        let path_and_query = match url.query() {
            Some(q) => format!("{}?{}", url.path(), q),
            None => url.path().to_string(),
        };
        let mut builder = hyper::Request::builder()
            .method(method)
            .uri(path_and_query.as_str());
        builder = builder.header(hyper::header::HOST, authority);
        builder
            .headers_mut()
            .ok_or_else(|| ErrorDetails::other("request builder rejected headers"))?
            .extend(headers.clone());
        builder
            .body(TimedBody::with_completion(
                body,
                self.clock.clone(),
                completion,
            ))
            .map_err(|e| ErrorDetails::other(format!("build request: {e}")))
    }

    /// Dispatch a request over an already-established (or pooled) `sender`,
    /// recording send/response timing into `trace`/`record`. Does not establish
    /// or close the connection, so the caller can return `sender` to a pool for
    /// reuse. Connect/DNS/reuse timings are expected to be pre-filled in `trace`.
    ///
    /// Enforces `cfg.request_timeout_ns` (when set to a positive value) around
    /// the whole send + response phase by racing it against a [`Clock`] timer. A
    /// `None`/non-positive timeout means "no deadline" — the un-raced hot path.
    #[allow(clippy::too_many_arguments)]
    pub async fn dispatch(
        &self,
        sender: &mut Sender,
        url: &Url,
        headers: &HeaderMap,
        body: Bytes,
        streaming: bool,
        trace: &mut TraceData,
        record: &mut RequestRecord,
        on_first_token: &mut impl FnMut(i64),
        body_len: usize,
    ) -> Result<(), ErrorDetails> {
        self.dispatch_with_method(
            Method::POST,
            sender,
            url,
            headers,
            body,
            streaming,
            trace,
            record,
            on_first_token,
            body_len,
        )
        .await
    }

    /// Dispatch with an explicit HTTP method over an established connection.
    #[allow(clippy::too_many_arguments)]
    pub async fn dispatch_with_method(
        &self,
        method: Method,
        sender: &mut Sender,
        url: &Url,
        headers: &HeaderMap,
        body: Bytes,
        streaming: bool,
        trace: &mut TraceData,
        record: &mut RequestRecord,
        on_first_token: &mut impl FnMut(i64),
        body_len: usize,
    ) -> Result<(), ErrorDetails> {
        let mut first_token_filter =
            SynchronousSseMessageFilter::new(|ttft_ns: i64, _message: &SseMessage| {
                on_first_token(ttft_ns);
                true
            });
        self.dispatch_with_method_and_completion(
            method,
            sender,
            url,
            headers,
            body,
            streaming,
            trace,
            record,
            &mut first_token_filter,
            body_len,
            Rc::new(SendCompletion::new()),
        )
        .await
    }

    /// Dispatch with a caller-owned send-completion signal. The transport
    /// facade uses this to arm cancellation only after the complete body is sent.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn dispatch_with_method_and_completion(
        &self,
        method: Method,
        sender: &mut Sender,
        url: &Url,
        headers: &HeaderMap,
        body: Bytes,
        streaming: bool,
        trace: &mut TraceData,
        record: &mut RequestRecord,
        first_token_filter: &mut impl SseMessageFilter,
        body_len: usize,
        completion: Rc<SendCompletion>,
    ) -> Result<(), ErrorDetails> {
        self.dispatch_with_method_and_completion_timeout(
            method,
            sender,
            url,
            headers,
            body,
            streaming,
            trace,
            record,
            first_token_filter,
            body_len,
            completion,
            self.cfg.request_timeout_ns,
        )
        .await
    }

    /// Dispatch with a caller-supplied remaining request budget.
    ///
    /// [`HttpTransport`](crate::transport::http::transport::http_transport::HttpTransport) uses
    /// this after connection acquisition so Config-v2's one absolute timeout
    /// cannot restart for the response phase. Other callers retain the
    /// client-wide `request_timeout_ns` through
    /// [`dispatch_with_method_and_completion`](Self::dispatch_with_method_and_completion).
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn dispatch_with_method_and_completion_timeout(
        &self,
        method: Method,
        sender: &mut Sender,
        url: &Url,
        headers: &HeaderMap,
        body: Bytes,
        streaming: bool,
        trace: &mut TraceData,
        record: &mut RequestRecord,
        first_token_filter: &mut impl SseMessageFilter,
        body_len: usize,
        completion: Rc<SendCompletion>,
        timeout_ns: Option<i64>,
    ) -> Result<(), ErrorDetails> {
        // A zero/None request timeout means "no deadline" — `with_timeout` takes
        // the un-raced path so the high-throughput dispatch stays overhead-free.
        with_timeout(
            self.clock.clone(),
            timeout_ns,
            self.dispatch_inner(
                method,
                sender,
                url,
                headers,
                body,
                streaming,
                trace,
                record,
                first_token_filter,
                body_len,
                completion,
            ),
            || ErrorDetails {
                kind: ErrorKind::Timeout,
                code: None,
                message: format!("request timeout after {}ns", timeout_ns.unwrap_or_default()),
            },
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn dispatch_inner(
        &self,
        method: Method,
        sender: &mut Sender,
        url: &Url,
        headers: &HeaderMap,
        body: Bytes,
        streaming: bool,
        trace: &mut TraceData,
        record: &mut RequestRecord,
        first_token_filter: &mut impl SseMessageFilter,
        body_len: usize,
        completion: Rc<SendCompletion>,
    ) -> Result<(), ErrorDetails> {
        // Time when the request body is fully written (end-of-stream), captured
        // by TimedBody — the real "send complete", distinct from response-headers.
        let req = self.build_request_with_method(method, url, headers, body, completion.clone())?;

        if trace.request_send_start_ns.is_none() {
            trace.request_send_start_ns = Some(self.clock.now_ns());
        }
        let resp = sender.send(req).await?;
        let hdr_ns = self.clock.now_ns();
        let send_end = completion.sent_ns().unwrap_or(hdr_ns);
        trace.request_send_end_ns = Some(send_end);
        trace.request_headers_sent_ns = completion.headers_ns().or(Some(send_end));
        trace.request_bytes_total = body_len as u64;
        trace.request_chunks_count = 1;
        if self.cfg.collect_trace_chunks {
            trace.request_chunks.push((send_end, body_len as u64));
        }
        trace.response_headers_received_ns = Some(hdr_ns);

        let status = resp.status();
        record.status = Some(status.as_u16());
        record.response_headers = resp
            .headers()
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.as_str().to_string(), value.to_string()))
            })
            .collect();
        trace.response_status_code = Some(status.as_u16());
        trace.response_reason = status.canonical_reason().map(str::to_string);

        record.recv_start_ns = Some(self.clock.now_ns());

        let content_type = resp
            .headers()
            .get(hyper::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let is_sse = streaming
            && content_type
                .as_deref()
                .map(|c| c.starts_with("text/event-stream"))
                .unwrap_or(false);
        let is_eventstream = streaming
            && content_type
                .as_deref()
                .map(|c| c.starts_with("application/vnd.amazon.eventstream"))
                .unwrap_or(false);

        check_declared_body_length(resp.headers(), self.cfg.max_response_body_bytes)?;

        let body_stream = resp.into_body().into_data_stream();
        let timing = Rc::new(RefCell::new(ChunkTiming::default()));
        let timing_map = timing.clone();
        let clock_map = self.clock.clone();
        let collect_chunks = self.cfg.collect_trace_chunks;
        let max_response_body_bytes = self.cfg.max_response_body_bytes;
        let mut observed_body_bytes = 0_u64;
        let timed = body_stream.map(move |item| match item {
            Ok(bytes) => {
                let timestamp_ns = clock_map.now_ns();
                timing_map
                    .borrow_mut()
                    .observe(timestamp_ns, bytes.len(), collect_chunks);
                observe_body_chunk(
                    &mut observed_body_bytes,
                    bytes.len(),
                    max_response_body_bytes,
                )?;
                Ok(bytes)
            }
            Err(error) => Err(body_err(error)),
        });

        if !status.is_success() {
            let collected = collect_body(timed).await;
            timing.borrow().copy_to(trace);
            let body = collected?;
            let ts = self.clock.now_ns();
            let text = String::from_utf8_lossy(&body).into_owned();
            record.responses.push(Response::Text(TextResponse {
                perf_ns: ts,
                text: text.clone(),
                body,
                content_type,
            }));
            // Body fully drained above, so the H1 connection is clean and can be
            // pooled even though we surface the HTTP failure to the caller.
            record.reusable_connection = true;
            return Err(ErrorDetails::http(status.as_u16(), text));
        }

        if is_sse || is_eventstream {
            let sse_stream: std::pin::Pin<Box<dyn Stream<Item = Result<Bytes, ErrorDetails>>>> =
                if is_eventstream {
                    Box::pin(eventstream_to_sse(timed))
                } else {
                    Box::pin(timed)
                };
            let sse_result = if first_token_filter.is_backpressured() {
                let mut handler = RecordingSseHandler {
                    start_ns: record.start_ns,
                    filter: first_token_filter,
                    responses: &mut record.responses,
                };
                read_sse_with_handler(sse_stream, self.clock.clone(), &mut handler).await
            } else {
                let mut handler = FirstTokenSseHandler {
                    start_ns: record.start_ns,
                    first_seen: false,
                    filter: first_token_filter,
                    responses: &mut record.responses,
                };
                read_sse_with_handler(sse_stream, self.clock.clone(), &mut handler).await
            };

            timing.borrow().copy_to(trace);
            sse_result?;
        } else {
            let collected = collect_body(timed).await;
            timing.borrow().copy_to(trace);
            let body = collected?;
            let ts = self.clock.now_ns();
            let text = String::from_utf8_lossy(&body).into_owned();
            if trace.response_receive_start_ns.is_none() {
                trace.response_receive_start_ns = Some(record.recv_start_ns.unwrap_or(ts));
                trace.response_receive_end_ns = Some(ts);
            }
            record.responses.push(Response::Text(TextResponse {
                perf_ns: ts,
                text,
                body,
                content_type,
            }));
        }

        Ok(())
    }

    /// A lean streaming dispatch for high-throughput callers: sends `body` on an
    /// established `sender`, then streams the SSE response, invoking
    /// `on_first_token` (with the TTFT delta in clock-ns) at the first message
    /// and `on_message` per parsed message — WITHOUT allocating a
    /// [`RequestRecord`]/[`TraceData`] or accumulating a `Vec` of responses.
    /// Returns the HTTP status code. The whole response body is consumed over
    /// the wire (each `SseMessage` is dropped right after `on_message`).
    pub async fn dispatch_streaming(
        &self,
        sender: &mut Sender,
        url: &Url,
        headers: &BTreeMap<String, String>,
        body: Bytes,
        on_first_token: &mut dyn FnMut(i64),
        on_message: &mut dyn FnMut(&SseMessage),
    ) -> Result<u16, ErrorDetails> {
        let headers = typed_headers(headers)?;
        self.dispatch_streaming_typed(sender, url, &headers, body, on_first_token, on_message)
            .await
    }

    async fn dispatch_streaming_typed(
        &self,
        sender: &mut Sender,
        url: &Url,
        headers: &HeaderMap,
        body: Bytes,
        on_first_token: &mut dyn FnMut(i64),
        on_message: &mut dyn FnMut(&SseMessage),
    ) -> Result<u16, ErrorDetails> {
        let start_ns = self.clock.now_ns();
        let req = self.build_request(url, headers, body, Rc::new(SendCompletion::new()))?;

        let resp = sender.send(req).await?;
        let code = resp.status().as_u16();
        let success = resp.status().is_success();
        check_declared_body_length(resp.headers(), self.cfg.max_response_body_bytes)?;
        let max_response_body_bytes = self.cfg.max_response_body_bytes;
        let mut observed_body_bytes = 0_u64;
        let body_stream = resp.into_body().into_data_stream();
        let limited = body_stream.map(move |item| match item {
            Ok(bytes) => {
                observe_body_chunk(
                    &mut observed_body_bytes,
                    bytes.len(),
                    max_response_body_bytes,
                )?;
                Ok(bytes)
            }
            Err(error) => Err(body_err(error)),
        });
        if !success {
            drain_body(limited).await?;
            return Ok(code);
        }

        let mut first = false;
        read_sse(limited, self.clock.clone(), |message: SseMessage| {
            if !first {
                first = true;
                on_first_token(message.perf_ns - start_ns);
            }
            on_message(&message);
        })
        .await?;
        Ok(code)
    }

    /// Lean streaming dispatch with a fallible decoded-frame consumer.
    ///
    /// The consumer runs before any terminal response/raw-record accumulation.
    /// Returning an error terminates response processing and leaves the caller
    /// to drop the connection lease, which is required for bounded decision
    /// admission failures.
    pub async fn dispatch_bounded_streaming_with_handler(
        &self,
        sender: &mut Sender,
        url: &Url,
        headers: &BTreeMap<String, String>,
        body: Bytes,
        max_sse_frame_bytes: usize,
        on_first_token: &mut dyn FnMut(i64),
        on_message: &mut dyn FnMut(&SseMessage) -> Result<bool, ErrorDetails>,
    ) -> Result<u16, ErrorDetails> {
        let headers = typed_headers(headers)?;
        self.dispatch_bounded_streaming_with_handler_typed(
            sender,
            url,
            &headers,
            body,
            max_sse_frame_bytes,
            on_first_token,
            on_message,
        )
        .await
    }

    pub(crate) async fn dispatch_bounded_streaming_with_handler_typed(
        &self,
        sender: &mut Sender,
        url: &Url,
        headers: &HeaderMap,
        body: Bytes,
        max_sse_frame_bytes: usize,
        on_first_token: &mut dyn FnMut(i64),
        on_message: &mut dyn FnMut(&SseMessage) -> Result<bool, ErrorDetails>,
    ) -> Result<u16, ErrorDetails> {
        let start_ns = self.clock.now_ns();

        // This lean path discards the send-complete timing, so the signal is a
        // throwaway retained only by the body.
        let req = self.build_request(url, headers, body, Rc::new(SendCompletion::new()))?;

        let resp = sender.send(req).await?;
        let code = resp.status().as_u16();
        let success = resp.status().is_success();
        let is_sse = resp
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("text/event-stream"));
        check_declared_body_length(resp.headers(), self.cfg.max_response_body_bytes)?;
        let max_response_body_bytes = self.cfg.max_response_body_bytes;
        let mut observed_body_bytes = 0_u64;
        let body_stream = resp.into_body().into_data_stream();
        let limited = body_stream.map(move |item| match item {
            Ok(bytes) => {
                observe_body_chunk(
                    &mut observed_body_bytes,
                    bytes.len(),
                    max_response_body_bytes,
                )?;
                Ok(bytes)
            }
            Err(error) => Err(body_err(error)),
        });
        if !success {
            drain_body(limited).await?;
            return Ok(code);
        }
        if !is_sse {
            drain_body(limited).await?;
            return Err(ErrorDetails::other(
                "bounded decision dispatch requires an SSE response",
            ));
        }

        let mut handler = FallibleStreamingSseHandler {
            start_ns,
            first_seen: false,
            on_first_token,
            on_message,
        };
        read_sse_with_bounded_frames(
            limited,
            self.clock.clone(),
            max_sse_frame_bytes,
            &mut handler,
        )
        .await?;
        Ok(code)
    }
}

#[cfg(test)]
mod eventstream_to_sse_tests {
    use super::*;
    use crate::transport::core::eventstream::EventStreamMessage;
    use bytes::BytesMut;
    use futures::stream;

    #[tokio::test]
    async fn reframes_payload_parts_as_sse_data_lines() {
        let m1 = EventStreamMessage::payload_part(Bytes::from_static(br#"{"i":1}"#));
        let m2 = EventStreamMessage::payload_part(Bytes::from_static(br#"{"i":2}"#));
        let wire = [
            m1.encode().expect("first frame encodes"),
            m2.encode().expect("second frame encodes"),
        ]
        .concat();

        let raw = stream::iter(vec![Ok::<Bytes, ErrorDetails>(Bytes::from(wire))]);
        let sse = eventstream_to_sse(raw);
        futures::pin_mut!(sse);

        let mut collected = Vec::new();
        while let Some(chunk) = sse.next().await {
            collected.push(chunk.unwrap());
        }
        let joined: Vec<u8> = collected.concat();
        let text = String::from_utf8(joined).unwrap();
        assert_eq!(text, "data: {\"i\":1}\n\ndata: {\"i\":2}\n\n");
    }

    #[tokio::test]
    async fn reframes_messages_split_across_chunk_boundaries() {
        let message = EventStreamMessage::payload_part(Bytes::from_static(br#"{"x":true}"#));
        let encoded = message.encode().expect("frame encodes");
        let (left, right) = encoded.split_at(encoded.len() / 2);

        let raw = stream::iter(vec![
            Ok::<Bytes, ErrorDetails>(Bytes::copy_from_slice(left)),
            Ok::<Bytes, ErrorDetails>(Bytes::copy_from_slice(right)),
        ]);
        let sse = eventstream_to_sse(raw);
        futures::pin_mut!(sse);

        let mut collected = Vec::new();
        while let Some(chunk) = sse.next().await {
            collected.push(chunk.unwrap());
        }
        let text = String::from_utf8(collected.concat()).unwrap();
        assert_eq!(text, "data: {\"x\":true}\n\n");
    }

    #[tokio::test]
    async fn emits_one_error_for_an_invalid_eventstream_prelude() {
        let valid = EventStreamMessage::payload_part(Bytes::from_static(br#"{"i":1}"#));
        let encoded = valid.encode().expect("frame encodes");
        let mut invalid = BytesMut::from(&encoded[..]);
        invalid[11] ^= 0xFF;
        let raw = stream::iter(vec![
            Ok::<Bytes, ErrorDetails>(invalid.freeze()),
            Ok(valid.encode().expect("frame encodes")),
        ]);
        let results: Vec<_> = eventstream_to_sse(raw).collect().await;

        assert_eq!(results.len(), 1);
        assert!(
            results[0]
                .as_ref()
                .is_err_and(|error| error.message.contains("prelude CRC"))
        );
    }

    #[tokio::test]
    async fn trailing_eventstream_bytes_emit_one_terminal_error() {
        let message = EventStreamMessage::payload_part(Bytes::from_static(br#"{"i":1}"#));
        let encoded = message.encode().expect("frame encodes");
        let truncated = Bytes::copy_from_slice(&encoded[..encoded.len() - 1]);
        let raw = stream::iter(vec![Ok::<Bytes, ErrorDetails>(truncated)]);
        let sse = eventstream_to_sse(raw);
        futures::pin_mut!(sse);

        let first = sse.next().await.expect("trailing bytes emit an error");
        assert!(first.as_ref().is_err_and(|error| {
            error.message == "truncated eventstream frame at response EOF"
        }));
        assert!(sse.next().await.is_none());
    }

    #[tokio::test]
    async fn clean_eventstream_boundary_ends_without_error() {
        let message = EventStreamMessage::payload_part(Bytes::from_static(br#"{"i":1}"#));
        let raw = stream::iter(vec![Ok::<Bytes, ErrorDetails>(
            message.encode().expect("frame encodes"),
        )]);
        let sse = eventstream_to_sse(raw);
        futures::pin_mut!(sse);

        assert_eq!(
            sse.next().await.expect("complete frame emits SSE").unwrap(),
            Bytes::from_static(b"data: {\"i\":1}\n\n")
        );
        assert!(sse.next().await.is_none());
    }
}
