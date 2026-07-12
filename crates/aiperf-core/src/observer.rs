// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared measurement observer backed by one single-loop [`TraceCollector`].
//!
//! The observer is a pure recorder — it does not own a clock. Callers supply
//! millisecond timestamps drawn from one shared time source (the live HTTP
//! path uses the `aiperf-transport-http` `Clock`), so arrival, admit, and token
//! events all sit on the same timeline.

use std::cell::RefCell;

use uuid::Uuid;

use loadgen_core::collector::{ReplayTerminalStatus, TraceCollector, TraceSimulationReport};
use loadgen_core::sink::RequestObserver;

/// Collects measurement events from any sink into one `TraceCollector`.
pub struct CollectorObserver {
    inner: RefCell<TraceCollector>,
}

impl CollectorObserver {
    /// Create an observer. When `capture_records` is true, the collector retains
    /// per-request detail for later record emission.
    pub fn new(capture_records: bool) -> Self {
        let mut collector = TraceCollector::default();
        collector.set_capture_per_request(capture_records);
        Self {
            inner: RefCell::new(collector),
        }
    }

    /// Produce the report, draining the accumulated collector. `wall_ms` is the
    /// total run duration in milliseconds on the shared timeline.
    pub fn finish(&self, wall_ms: f64) -> TraceSimulationReport {
        self.take().finish().with_wall_time_ms(wall_ms)
    }

    /// Drain the owned collector without reducing it.
    ///
    /// Post-drain runtimes can move this plain data value to a reduction worker;
    /// no observer callback or clock access remains after this boundary.
    pub fn take(&self) -> TraceCollector {
        std::mem::take(&mut *self.inner.borrow_mut())
    }
}

impl RequestObserver for CollectorObserver {
    fn on_arrival(&self, u: Uuid, a: f64, i: usize, o: usize) {
        self.inner.borrow_mut().on_arrival(u, a, i, o);
    }
    fn on_admit(&self, u: Uuid, a: f64, r: usize) {
        self.inner.borrow_mut().on_admit(u, a, r);
    }
    fn on_token(&self, u: Uuid, at: f64) {
        self.inner.borrow_mut().on_token(u, at);
    }
    fn on_terminal(&self, u: Uuid, s: ReplayTerminalStatus) {
        self.inner.borrow_mut().on_terminal(u, s);
    }
}
