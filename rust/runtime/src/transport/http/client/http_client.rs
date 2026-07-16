// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request path: send, then stream SSE or read a text body, recording all
//! timing into a RequestRecord. Port of `AioHttpClient._request`.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use http::Method;
use http_body_util::BodyExt;
use url::Url;

use crate::clock::Clock;

use crate::transport::core::{
    ErrorDetails, ErrorKind, RequestRecord, Response, TextResponse, TraceData,
};
use crate::transport::http::client::cancellation::{CancelOutcome, race_cancel_after_send};
use crate::transport::http::client::connection::{
    SendCompletion, Sender, TimedBody, establish, with_timeout,
};
use crate::transport::http::config::ClientConfig;
use crate::transport::http::models::SseMessage;
use crate::transport::http::sse::{SseMessageHandler, read_sse, read_sse_with_handler};

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

/// Non-backpressured fast-path SSE handler: records every message and drives the
/// filter's first-token search only until it reports the first significant
/// token, then stops invoking it while still draining the body.
///
/// This mirrors the former inline `read_sse` closure exactly — same
/// `perf_ns - start_ns` TTFT delta, same `first_seen` short-circuit, and an
/// always-ready `poll_ready` (a non-backpressured filter never blocks on
/// capacity, so it is never consulted here). The reason it exists as a handler
/// rather than an infallible closure is panic safety: `SseMessageFilter` is a
/// public trait, so an external filter may legitimately override
/// `is_backpressured()` to `false` yet return `Err` from `start_send`. Routing
/// through the fallible [`SseMessageHandler`] seam propagates that error to fail
/// the request instead of unwrapping it into a panic.
struct FirstTokenSseHandler<'a, F>
where
    F: SseMessageFilter + ?Sized,
{
    start_ns: i64,
    first_seen: bool,
    filter: &'a mut F,
    responses: &'a mut Vec<Response>,
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
        let completion = Rc::new(SendCompletion::new());
        self.request_with_method_and_completion(
            method,
            url,
            headers,
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
        headers: &BTreeMap<String, String>,
        body: Bytes,
        streaming: bool,
        mut on_first_token: impl FnMut(i64),
        completion: Rc<SendCompletion>,
    ) -> RequestRecord {
        let start_ns = self.clock.now_ns();
        let mut record = RequestRecord::started(start_ns);
        let mut trace = TraceData {
            // aiohttp emits on_request_start before connection acquisition.
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

    /// Like [`request`](Self::request) but cancels `cancel_after_ns` after send.
    pub async fn request_cancellable(
        &self,
        url: &Url,
        headers: &BTreeMap<String, String>,
        body: Bytes,
        streaming: bool,
        cancel_after_ns: i64,
        mut on_first_token: impl FnMut(i64),
    ) -> RequestRecord {
        let start_ns = self.clock.now_ns();
        let body_len = body.len();
        let mut record = RequestRecord::started(start_ns);
        let mut trace = TraceData {
            request_send_start_ns: Some(start_ns),
            ..TraceData::default()
        };
        let completion = Rc::new(SendCompletion::new());
        let completion_for_dispatch = completion.clone();
        let completion_for_record = completion.clone();
        let mut first_token_filter =
            SynchronousSseMessageFilter::new(|ttft_ns: i64, _message: &SseMessage| {
                on_first_token(ttft_ns);
                true
            });
        let request = async {
            let (mut sender, _socket) =
                establish(url, &self.cfg, self.clock.clone(), &mut trace).await?;
            self.dispatch_with_method_and_completion(
                Method::POST,
                &mut sender,
                url,
                headers,
                body,
                streaming,
                &mut trace,
                &mut record,
                &mut first_token_filter,
                body_len,
                completion_for_dispatch,
            )
            .await
        };
        let result = match race_cancel_after_send(
            self.clock.clone(),
            cancel_after_ns,
            completion,
            request,
        )
        .await
        {
            CancelOutcome::Completed(result) => result,
            CancelOutcome::Cancelled => {
                let now = self.clock.now_ns();
                record.cancellation_ns = Some(now);
                if let Some(sent_ns) = completion_for_record.sent_ns()
                    && trace.request_send_end_ns.is_none()
                {
                    trace.request_send_end_ns = Some(sent_ns);
                    trace.request_headers_sent_ns = completion_for_record.headers_ns();
                    trace.request_bytes_total = body_len as u64;
                    trace.request_chunks_count = 1;
                    if self.cfg.collect_trace_chunks {
                        trace.request_chunks.push((sent_ns, body_len as u64));
                    }
                }
                Err(ErrorDetails::cancelled(format!(
                    "RequestCancellationError: request cancelled {cancel_after_ns}ns after being sent"
                )))
            }
        };
        if let Err(error) = result {
            trace.error_timestamp_ns = Some(self.clock.now_ns());
            record.error = Some(error);
        }
        record.end_ns = Some(self.clock.now_ns());
        record.trace = Some(trace);
        record
    }

    /// Build the POST request shared by [`dispatch`](Self::dispatch) and
    /// [`dispatch_streaming`](Self::dispatch_streaming). Uses origin-form URI +
    /// explicit Host header so both HTTP/1.1 (Host required) and HTTP/2
    /// (`:authority` derived) work. `completion` is the shared signal that
    /// [`TimedBody`] stamps at end-of-stream (the real "send complete").
    fn build_request(
        &self,
        url: &Url,
        headers: &BTreeMap<String, String>,
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
        headers: &BTreeMap<String, String>,
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
        for (k, v) in headers {
            builder = builder.header(k.as_str(), v.as_str());
        }
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
        headers: &BTreeMap<String, String>,
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
        headers: &BTreeMap<String, String>,
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
        headers: &BTreeMap<String, String>,
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
        headers: &BTreeMap<String, String>,
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

    /// The un-timed send + response body raced by [`dispatch`](Self::dispatch).
    #[allow(clippy::too_many_arguments)]
    async fn dispatch_inner(
        &self,
        method: Method,
        sender: &mut Sender,
        url: &Url,
        headers: &BTreeMap<String, String>,
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
        // Response headers received; the body finished writing at `send_end`.
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

        if is_sse {
            let sse_result = if first_token_filter.is_backpressured() {
                let mut handler = RecordingSseHandler {
                    start_ns: record.start_ns,
                    filter: first_token_filter,
                    responses: &mut record.responses,
                };
                read_sse_with_handler(timed, self.clock.clone(), &mut handler).await
            } else {
                let mut handler = FirstTokenSseHandler {
                    start_ns: record.start_ns,
                    first_seen: false,
                    filter: first_token_filter,
                    responses: &mut record.responses,
                };
                read_sse_with_handler(timed, self.clock.clone(), &mut handler).await
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
        let start_ns = self.clock.now_ns();

        // This lean path discards the send-complete timing, so the signal is a
        // throwaway retained only by the body.
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
        read_sse(limited, self.clock.clone(), |m: SseMessage| {
            if !first {
                first = true;
                on_first_token(m.perf_ns - start_ns);
            }
            on_message(&m);
        })
        .await?;
        Ok(code)
    }
}
