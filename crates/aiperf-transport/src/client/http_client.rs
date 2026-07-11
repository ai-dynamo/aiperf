// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request path: send, then stream SSE or read a text body, recording all
//! timing into a RequestRecord. Port of `AioHttpClient._request`.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use bytes::Bytes;
use futures::StreamExt;
use http_body_util::BodyExt;
use url::Url;

use aiperf_clock::Clock;

use crate::client::cancellation::{CancelOutcome, race_cancel};
use crate::client::connection::{Sender, TimedBody, establish, with_timeout};
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
        mut on_first_token: impl FnMut(i64),
    ) -> RequestRecord {
        let start_ns = self.clock.now_ns();
        let mut record = RequestRecord::started(start_ns);
        let mut trace = TraceData::default();

        let body_len = body.len();
        let result = async {
            let (mut sender, _sock) =
                establish(url, &self.cfg, self.clock.clone(), &mut trace).await?;
            self.dispatch(
                &mut sender,
                url,
                headers,
                body,
                streaming,
                &mut trace,
                &mut record,
                &mut on_first_token,
                body_len,
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
        on_first_token: impl FnMut(i64),
    ) -> RequestRecord {
        let start_ns = self.clock.now_ns();
        let fut = self.request(url, headers, body, streaming, on_first_token);
        match race_cancel(self.clock.clone(), cancel_after_ns, fut).await {
            CancelOutcome::Completed(rec) => rec,
            CancelOutcome::Cancelled => {
                let now = self.clock.now_ns();
                let mut rec = RequestRecord::started(start_ns);
                rec.cancellation_ns = Some(now);
                rec.end_ns = Some(now);
                rec.error = Some(ErrorDetails::cancelled(format!(
                    "Request cancelled {cancel_after_ns}ns after being sent"
                )));
                rec
            }
        }
    }

    /// Build the POST request shared by [`dispatch`](Self::dispatch) and
    /// [`dispatch_streaming`](Self::dispatch_streaming). Uses origin-form URI +
    /// explicit Host header so both HTTP/1.1 (Host required) and HTTP/2
    /// (`:authority` derived) work. `sent_ns` is the shared cell that
    /// [`TimedBody`] stamps at end-of-stream (the real "send complete").
    fn build_request(
        &self,
        url: &Url,
        headers: &BTreeMap<String, String>,
        body: Bytes,
        sent_ns: Rc<std::cell::Cell<Option<i64>>>,
    ) -> Result<hyper::Request<TimedBody>, ErrorDetails> {
        let authority = url.authority();
        let path_and_query = match url.query() {
            Some(q) => format!("{}?{}", url.path(), q),
            None => url.path().to_string(),
        };
        let mut builder = hyper::Request::builder()
            .method("POST")
            .uri(path_and_query.as_str());
        builder = builder.header(hyper::header::HOST, authority);
        for (k, v) in headers {
            builder = builder.header(k.as_str(), v.as_str());
        }
        builder
            .body(TimedBody::new(body, self.clock.clone(), sent_ns))
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
        // A zero/None request timeout means "no deadline" — `with_timeout` takes
        // the un-raced path so the high-throughput dispatch stays overhead-free.
        let timeout_ns = self.cfg.request_timeout_ns;
        with_timeout(
            self.clock.clone(),
            timeout_ns,
            self.dispatch_inner(
                sender,
                url,
                headers,
                body,
                streaming,
                trace,
                record,
                on_first_token,
                body_len,
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
        // Time when the request body is fully written (end-of-stream), captured
        // by TimedBody — the real "send complete", distinct from response-headers.
        let sent_ns = std::rc::Rc::new(std::cell::Cell::new(None));
        let req = self.build_request(url, headers, body, sent_ns.clone())?;

        trace.request_send_start_ns = Some(self.clock.now_ns());
        let resp = sender.send(req).await?;
        // Response headers received; the body finished writing at `send_end`.
        let hdr_ns = self.clock.now_ns();
        let send_end = sent_ns.get().unwrap_or(hdr_ns);
        trace.request_send_end_ns = Some(send_end);
        trace.request_headers_sent_ns = Some(send_end);
        trace.request_bytes_total = body_len as u64;
        trace.request_chunks_count = 1;
        trace.response_headers_received_ns = Some(hdr_ns);

        let status = resp.status();
        record.status = Some(status.as_u16());
        trace.response_status_code = Some(status.as_u16());
        trace.response_reason = status.canonical_reason().map(str::to_string);

        if !status.is_success() {
            let body = resp
                .into_body()
                .collect()
                .await
                .map(|b| String::from_utf8_lossy(&b.to_bytes()).into_owned())
                .unwrap_or_default();
            return Err(ErrorDetails::http(status.as_u16(), body));
        }

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

        if is_sse {
            let timing = Rc::new(RefCell::new(ChunkTiming::default()));
            let timing_map = timing.clone();
            let clock_map = self.clock.clone();
            // Timestamp each transport chunk as it arrives, then hand the bytes
            // to the incremental SSE parser.
            let timed = body_stream.map(move |item| match item {
                Ok(b) => {
                    let ts = clock_map.now_ns();
                    let mut t = timing_map.borrow_mut();
                    t.chunks += 1;
                    t.bytes += b.len() as u64;
                    if t.recv_start.is_none() {
                        t.recv_start = Some(ts);
                    }
                    t.recv_end = Some(ts);
                    Ok(b)
                }
                Err(e) => Err(body_err(e)),
            });

            let start_ns = record.start_ns;
            let mut first_seen = false;
            let responses = &mut record.responses;
            let sse_result = read_sse(timed, self.clock.clone(), |m: SseMessage| {
                if !first_seen {
                    first_seen = true;
                    on_first_token(m.perf_ns - start_ns);
                }
                responses.push(Response::Sse(m));
            })
            .await;

            {
                let t = timing.borrow();
                trace.response_receive_start_ns = t.recv_start;
                trace.response_receive_end_ns = t.recv_end;
                trace.response_chunks_count = t.chunks;
                trace.response_bytes_total = t.bytes;
            }
            sse_result?;
        } else {
            let collected = body_stream
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<Bytes>, _>>()
                .map_err(body_err)?;
            let ts = self.clock.now_ns();
            let total: usize = collected.iter().map(|b| b.len()).sum();
            let mut text = String::new();
            for b in &collected {
                text.push_str(&String::from_utf8_lossy(b));
            }
            trace.response_receive_start_ns = Some(record.recv_start_ns.unwrap_or(ts));
            trace.response_receive_end_ns = Some(ts);
            trace.response_chunks_count = collected.len() as u32;
            trace.response_bytes_total = total as u64;
            record.responses.push(Response::Text(TextResponse {
                perf_ns: ts,
                text,
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

        // This lean path discards the send-complete timing, so the TimedBody
        // cell is write-only (never read back) — a throwaway.
        let req =
            self.build_request(url, headers, body, std::rc::Rc::new(std::cell::Cell::new(None)))?;

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
