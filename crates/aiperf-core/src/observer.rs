// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared measurement observer: a thread-safe [`RequestObserver`] backed by a
//! single `TraceCollector`, with one monotonic clock so all timestamps agree.

use std::borrow::Cow;
use std::sync::Mutex;
use std::time::Instant;

use uuid::Uuid;

use loadgen_core::collector::{
    PerRequestRecord, ReplayTerminalStatus, TraceCollector, TraceSimulationReport,
};
use loadgen_core::sink::RequestObserver;

use crate::http_sink::StreamOutcome;

/// Reconcile per-chunk token arrival times against the server's authoritative
/// output-token count (`usage.completion_tokens`).
///
/// When the counts agree (the common one-token-per-SSE-chunk case) the real
/// per-chunk times are used unchanged, preserving true inter-token jitter. When
/// they differ (a server that packs multiple tokens per chunk), the count is
/// made authoritative by stretching `first..last` evenly across it — this keeps
/// TTFT and end-to-end latency exact while making output-token throughput
/// correct. With no usage reported, the chunk times are used as-is.
fn reconcile_output_times(real: &[f64], authoritative: Option<u32>) -> Cow<'_, [f64]> {
    let Some(n) = authoritative.map(|n| n as usize) else {
        return Cow::Borrowed(real);
    };
    if n == real.len() || real.is_empty() || n == 0 {
        return Cow::Borrowed(real);
    }
    let first = real[0];
    let last = *real.last().unwrap();
    if n == 1 {
        return Cow::Owned(vec![first]);
    }
    let step = (last - first) / (n as f64 - 1.0);
    Cow::Owned((0..n).map(|i| first + step * i as f64).collect())
}

/// Collects measurement events from any sink into one `TraceCollector`.
pub struct CollectorObserver {
    inner: Mutex<TraceCollector>,
    start: Instant,
}

impl CollectorObserver {
    /// Create an observer. When `capture_records` is true, the collector
    /// retains per-request detail so [`finish_with_records`] can emit raw
    /// records.
    ///
    /// [`finish_with_records`]: CollectorObserver::finish_with_records
    pub fn new(start: Instant, capture_records: bool) -> Self {
        let mut collector = TraceCollector::default();
        collector.set_capture_per_request(capture_records);
        Self {
            inner: Mutex::new(collector),
            start,
        }
    }

    /// Milliseconds since the shared start instant.
    pub fn now_ms(&self) -> f64 {
        crate::elapsed_ms(self.start)
    }

    /// Produce the report, draining the accumulated collector.
    pub fn finish(&self, wall_ms: f64) -> TraceSimulationReport {
        let collector = std::mem::take(&mut *self.inner.lock().unwrap());
        collector.finish().with_wall_time_ms(wall_ms)
    }

    /// Submit one request's batched measurements under a single lock. This is
    /// the hot-path entry point: streaming accumulates token times lock-free and
    /// records them here once, so tokio's worker threads don't serialize on the
    /// collector mutex per token.
    pub fn submit(
        &self,
        uuid: Uuid,
        arrival_ms: f64,
        input_length: usize,
        requested_output: usize,
        outcome: &StreamOutcome,
    ) {
        let mut collector = self.inner.lock().unwrap();
        collector.on_arrival(uuid, arrival_ms, input_length, requested_output);
        collector.on_admit(uuid, outcome.admit_ms, 0);
        let times = reconcile_output_times(&outcome.token_times, outcome.completion_tokens);
        for at in times.iter() {
            collector.on_token(uuid, *at);
        }
        collector.on_terminal(uuid, outcome.terminal);
    }

    /// Produce the report along with the raw per-request records (empty unless
    /// the observer was created with `capture_records = true`).
    pub fn finish_with_records(
        &self,
        wall_ms: f64,
    ) -> (TraceSimulationReport, Vec<PerRequestRecord>) {
        let collector = std::mem::take(&mut *self.inner.lock().unwrap());
        let records = collector.per_request_records();
        let report = collector.finish().with_wall_time_ms(wall_ms);
        (report, records)
    }
}

impl RequestObserver for CollectorObserver {
    fn on_arrival(&self, u: Uuid, a: f64, i: usize, o: usize) {
        self.inner.lock().unwrap().on_arrival(u, a, i, o);
    }
    fn on_admit(&self, u: Uuid, a: f64, r: usize) {
        self.inner.lock().unwrap().on_admit(u, a, r);
    }
    fn on_token(&self, u: Uuid, at: f64) {
        self.inner.lock().unwrap().on_token(u, at);
    }
    fn on_terminal(&self, u: Uuid, s: ReplayTerminalStatus) {
        self.inner.lock().unwrap().on_terminal(u, s);
    }
}
