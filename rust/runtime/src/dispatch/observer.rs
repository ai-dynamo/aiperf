// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Measurement observer backed by a worker-local [`TraceCollector`].
//!
//! The observer is a pure recorder — it does not own a clock. Callers supply
//! millisecond timestamps from one shared time source so arrival, admission,
//! and token events use the same timeline.

use std::cell::RefCell;

use uuid::Uuid;

use super::collector::{ReplayTerminalStatus, TraceCollector, TraceSimulationReport};
use super::sink::RequestObserver;

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

    /// Drain the collector without reducing it.
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
    fn on_output_tokens(&self, u: Uuid, at_ms: &[f64]) {
        let mut collector = self.inner.borrow_mut();
        for &at in at_ms {
            collector.on_token(u, at);
        }
    }
    fn on_terminal(&self, u: Uuid, s: ReplayTerminalStatus) {
        self.inner.borrow_mut().on_terminal(u, s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_observer_emits_supported_fields_and_compatibility_defaults() {
        let observer = CollectorObserver::new(true);
        let request = Uuid::from_u128(1);
        let observer_api: &dyn RequestObserver = &observer;

        observer_api.on_arrival(request, 10.0, 8, 2);
        observer_api.on_admit(request, 12.0, 3);
        observer_api.on_token(request, 15.0);
        observer_api.on_token(request, 20.0);
        observer_api.on_terminal(request, ReplayTerminalStatus::Completed);

        let report = observer.finish(25.0);
        let summary = serde_json::to_value(&report).expect("report must serialize");
        let record = serde_json::to_value(&report.per_request[0]).expect("record must serialize");

        assert_eq!(summary["completed_requests"], 1);
        assert_eq!(summary["total_input_tokens"], 8);
        assert_eq!(summary["total_output_tokens"], 2);
        assert_eq!(summary["prefill_worker_seconds"], 0.0);
        assert_eq!(summary["decode_worker_seconds"], 0.0);
        assert_eq!(summary["prefill_gpus_per_worker"], 0);
        assert_eq!(summary["decode_gpus_per_worker"], 0);
        assert_eq!(summary["gpu_hours"], 0.0);
        assert_eq!(record["first_admit_ms"], 12.0);
        assert_eq!(record["reused_input_tokens"], 3);
        assert!(record["prefill_worker_idx"].is_null());
        assert!(record["decode_worker_idx"].is_null());
        assert!(record["prefill_admit_ms"].is_null());
        assert!(record["source_held_ms"].is_null());
        assert!(record["destination_reserved_ms"].is_null());
        assert!(record["destination_activated_ms"].is_null());
        assert!(record["decode_admit_ms"].is_null());
        assert!(record["source_released_ms"].is_null());
        assert!(record["decode_reused_input_tokens"].is_null());
        assert!(record["prefill_route_overlap_tokens"].is_null());
        assert!(record["decode_route_overlap_tokens"].is_null());
    }
}
