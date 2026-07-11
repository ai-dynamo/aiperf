// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared measurement observer: a thread-safe [`RequestObserver`] backed by a
//! single `TraceCollector`.
//!
//! The observer is a pure recorder — it does not own a clock. Callers supply
//! millisecond timestamps drawn from one shared time source (the live HTTP
//! path uses the `aiperf-transport` `Clock`), so arrival, admit, and token
//! events all sit on the same timeline.

use std::sync::Mutex;

use uuid::Uuid;

use loadgen_core::collector::{ReplayTerminalStatus, TraceCollector, TraceSimulationReport};
use loadgen_core::sink::RequestObserver;

/// Collects measurement events from any sink into one `TraceCollector`.
pub struct CollectorObserver {
    inner: Mutex<TraceCollector>,
}

impl CollectorObserver {
    /// Create an observer. When `capture_records` is true, the collector retains
    /// per-request detail for later record emission.
    pub fn new(capture_records: bool) -> Self {
        let mut collector = TraceCollector::default();
        collector.set_capture_per_request(capture_records);
        Self {
            inner: Mutex::new(collector),
        }
    }

    /// Produce the report, draining the accumulated collector. `wall_ms` is the
    /// total run duration in milliseconds on the shared timeline.
    pub fn finish(&self, wall_ms: f64) -> TraceSimulationReport {
        let collector = std::mem::take(&mut *self.inner.lock().unwrap());
        collector.finish().with_wall_time_ms(wall_ms)
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
