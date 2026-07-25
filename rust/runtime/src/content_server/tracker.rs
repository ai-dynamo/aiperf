// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded request tracking and injectable wall/monotonic time.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::content_server::model::{
    ContentRecordSender, ContentRequestRecord, RequestTrackerSnapshot,
};

/// Time source used by HTTP request lifecycle instrumentation.
///
/// Wall time supplies cross-service correlation while monotonic time supplies
/// drift-free intervals. Tests and alternate runtimes can inject both values.
pub trait ContentServerClock: fmt::Debug + Send + Sync {
    /// Nanoseconds since the Unix epoch.
    fn wall_time_ns(&self) -> u64;
    /// Nanoseconds on an arbitrary monotonic timeline.
    fn monotonic_ns(&self) -> u64;
}

/// Wall-clock nanoseconds since the Unix epoch for the content-server dispatch
/// tag (`td`).
///
/// Lives in this module — not the clock-injected HTTP transport — so the served
/// media latency in [`media_metrics`](crate::content_server) can subtract it
/// against arrivals stamped from the same `SystemTime` epoch.
pub fn dispatch_wall_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64
}

/// System wall time paired with an [`Instant`]-based monotonic origin.
#[derive(Debug)]
pub struct SystemContentServerClock {
    monotonic_origin: Instant,
}

impl Default for SystemContentServerClock {
    fn default() -> Self {
        Self {
            monotonic_origin: Instant::now(),
        }
    }
}

impl ContentServerClock for SystemContentServerClock {
    fn wall_time_ns(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64
    }

    fn monotonic_ns(&self) -> u64 {
        self.monotonic_origin
            .elapsed()
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64
    }
}

#[derive(Debug)]
struct TrackerState {
    records: VecDeque<ContentRequestRecord>,
    total_requests: u64,
    total_bytes_served: u64,
}

/// Thread-safe bounded request-record collector with lifetime counters.
#[derive(Debug)]
pub struct RequestTracker {
    max_records: usize,
    state: Mutex<TrackerState>,
    record_sink: Option<ContentRecordSender>,
}

impl RequestTracker {
    /// Construct a tracker retaining at most `max_records` recent completions.
    pub fn new(max_records: usize) -> Self {
        Self::new_with_sink(max_records, None)
    }

    /// Construct a tracker that additionally forwards every completed record to
    /// `record_sink` (for streaming aggregation), regardless of `max_records`.
    pub fn new_with_sink(max_records: usize, record_sink: Option<ContentRecordSender>) -> Self {
        Self {
            max_records,
            state: Mutex::new(TrackerState {
                records: VecDeque::new(),
                total_requests: 0,
                total_bytes_served: 0,
            }),
            record_sink,
        }
    }

    /// Append one completed request and update non-evicting lifetime counters.
    pub fn record(&self, record: ContentRequestRecord) {
        // Forward to the streaming consumer outside the lock; the retention
        // buffer below is independent, so the aggregator sees every record even
        // when retention is disabled.
        if let Some(sink) = &self.record_sink {
            let _ = sink.send(record.clone());
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.total_requests = state.total_requests.saturating_add(1);
        state.total_bytes_served = state.total_bytes_served.saturating_add(record.body_bytes);
        if self.max_records == 0 {
            return;
        }
        if state.records.len() == self.max_records {
            state.records.pop_front();
        }
        state.records.push_back(record);
    }

    /// Clone one internally consistent snapshot.
    pub fn snapshot(&self) -> RequestTrackerSnapshot {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        RequestTrackerSnapshot {
            total_requests: state.total_requests,
            total_bytes_served: state.total_bytes_served,
            records: state.records.iter().cloned().collect(),
        }
    }

    /// Lifetime request count.
    pub fn total_requests(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .total_requests
    }

    /// Lifetime body-byte count.
    pub fn total_bytes_served(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .total_bytes_served
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn record(index: u64) -> ContentRequestRecord {
        ContentRequestRecord {
            timestamp_ns: index,
            method: "GET".into(),
            path: format!("/content/file_{index}"),
            query_string: String::new(),
            http_version: "1.1".into(),
            client_host: String::new(),
            client_port: 0,
            request_headers: BTreeMap::new(),
            status_code: 200,
            content_type: "application/octet-stream".into(),
            response_headers: BTreeMap::new(),
            body_bytes: 10,
            body_chunk_count: 1,
            latency_ns: 1,
            time_to_first_byte_ns: 1,
            time_to_first_body_byte_ns: 1,
            transfer_duration_ns: 0,
            error: None,
        }
    }

    #[test]
    fn bounded_records_evict_but_lifetime_counters_survive() {
        let tracker = RequestTracker::new(3);
        for index in 0..5 {
            tracker.record(record(index));
        }
        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.total_requests, 5);
        assert_eq!(snapshot.total_bytes_served, 50);
        assert_eq!(snapshot.records.len(), 3);
        assert_eq!(snapshot.records[0].path, "/content/file_2");
        assert_eq!(snapshot.records[2].path, "/content/file_4");
    }

    #[test]
    fn sink_forwards_every_record_even_with_retention_disabled() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        // max_records = 0: nothing retained, but the sink must still see all.
        let tracker = RequestTracker::new_with_sink(0, Some(tx));
        for index in 0..3 {
            tracker.record(record(index));
        }
        let forwarded: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        assert_eq!(forwarded.len(), 3);
        assert_eq!(forwarded[0].path, "/content/file_0");
        assert_eq!(forwarded[2].path, "/content/file_2");
        assert_eq!(tracker.snapshot().records.len(), 0);
        assert_eq!(tracker.total_requests(), 3);
    }

    #[test]
    fn no_sink_behaves_as_before() {
        let tracker = RequestTracker::new_with_sink(2, None);
        tracker.record(record(1));
        assert_eq!(tracker.snapshot().records.len(), 1);
    }
}
