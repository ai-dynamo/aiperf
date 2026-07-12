// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request path: send, then stream SSE or read a text body, recording all
//! timing into a RequestRecord. Port of `AioHttpClient._request`.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use bytes::Bytes;
use futures::StreamExt;
use http::Method;
use http_body_util::BodyExt;
use url::Url;

use aiperf_clock::Clock;

use crate::client::cancellation::{CancelOutcome, race_cancel_after_send};
use crate::client::connection::{SendCompletion, Sender, TimedBody, establish, with_timeout};
use crate::config::ClientConfig;
use crate::models::{
    ErrorDetails, ErrorKind, RequestRecord, Response, SseMessage, TextResponse, TraceData,
};
use crate::sse::read_sse;

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
        let mut trace = TraceData::default();
        let mut first_token_filter = |ttft_ns: i64, _message: &SseMessage| {
            on_first_token(ttft_ns);
            true
        };

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
        let mut trace = TraceData::default();
        let completion = Rc::new(SendCompletion::new());
        let completion_for_dispatch = completion.clone();
        let completion_for_record = completion.clone();
        let mut first_token_filter = |ttft_ns: i64, _message: &SseMessage| {
            on_first_token(ttft_ns);
            true
        };
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
                    trace.request_headers_sent_ns = Some(sent_ns);
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
        let mut first_token_filter = |ttft_ns: i64, _message: &SseMessage| {
            on_first_token(ttft_ns);
            true
        };
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
        first_token_filter: &mut impl FnMut(i64, &SseMessage) -> bool,
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
    /// [`HttpTransport`](crate::transport::http_transport::HttpTransport) uses
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
        first_token_filter: &mut impl FnMut(i64, &SseMessage) -> bool,
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
        first_token_filter: &mut impl FnMut(i64, &SseMessage) -> bool,
        body_len: usize,
        completion: Rc<SendCompletion>,
    ) -> Result<(), ErrorDetails> {
        // Time when the request body is fully written (end-of-stream), captured
        // by TimedBody — the real "send complete", distinct from response-headers.
        let req = self.build_request_with_method(method, url, headers, body, completion.clone())?;

        trace.request_send_start_ns = Some(self.clock.now_ns());
        let resp = sender.send(req).await?;
        // Response headers received; the body finished writing at `send_end`.
        let hdr_ns = self.clock.now_ns();
        let send_end = completion.sent_ns().unwrap_or(hdr_ns);
        trace.request_send_end_ns = Some(send_end);
        trace.request_headers_sent_ns = Some(send_end);
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

        let body_stream = resp.into_body().into_data_stream();
        let timing = Rc::new(RefCell::new(ChunkTiming::default()));
        let timing_map = timing.clone();
        let clock_map = self.clock.clone();
        let collect_chunks = self.cfg.collect_trace_chunks;
        let timed = body_stream.map(move |item| match item {
            Ok(bytes) => {
                let timestamp_ns = clock_map.now_ns();
                timing_map
                    .borrow_mut()
                    .observe(timestamp_ns, bytes.len(), collect_chunks);
                Ok(bytes)
            }
            Err(error) => Err(body_err(error)),
        });

        if !status.is_success() {
            let collected = timed
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<Bytes>, _>>();
            timing.borrow().copy_to(trace);
            let body = collected
                .map(|chunks| {
                    let total = chunks.iter().map(Bytes::len).sum();
                    let mut raw = Vec::with_capacity(total);
                    for chunk in chunks {
                        raw.extend_from_slice(&chunk);
                    }
                    String::from_utf8_lossy(&raw).into_owned()
                })
                .unwrap_or_default();
            return Err(ErrorDetails::http(status.as_u16(), body));
        }

        if is_sse {
            let start_ns = record.start_ns;
            let mut first_seen = false;
            let responses = &mut record.responses;
            let sse_result = read_sse(timed, self.clock.clone(), |m: SseMessage| {
                if !first_seen && first_token_filter(m.perf_ns - start_ns, &m) {
                    first_seen = true;
                }
                responses.push(Response::Sse(m));
            })
            .await;

            timing.borrow().copy_to(trace);
            sse_result?;
        } else {
            let collected = timed
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<Bytes>, _>>();
            timing.borrow().copy_to(trace);
            let collected = collected?;
            let ts = self.clock.now_ns();
            let total: usize = collected.iter().map(|b| b.len()).sum();
            let mut raw = Vec::with_capacity(total);
            for b in &collected {
                raw.extend_from_slice(b);
            }
            let body = Bytes::from(raw);
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
        if !resp.status().is_success() {
            let _ = resp.into_body().collect().await;
            return Ok(code);
        }

        let body_stream = resp.into_body().into_data_stream();
        let timed = body_stream.map(|item| item.map_err(body_err));
        let mut first = false;
        read_sse(timed, self.clock.clone(), |m: SseMessage| {
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
