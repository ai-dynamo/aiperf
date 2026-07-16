// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request dispatch state machine + `DistributionMiss`
//! (`specs/2026-07-15-ultimate-cellular-velo-runtime-design.md` §4.5).
//!
//! When the control plane (the phaser §4) says "issue request R now", the data plane
//! (the dataset index §3) may or may not have R local yet. The dispatch is therefore a
//! **state machine per `request_id`**, `Unknown → Indexed → InFlight → Done`, that
//! handles both halves of the interlock:
//!
//! - `Indexed`  → issue, transition to `InFlight`.  (the happy path)
//! - `InFlight` / `Done` → **no-op** ([`DispatchDecision::Duplicate`]). A duplicate or
//!   retransmitted "issue R" must not double-issue — **exactly-once-issue** per id.
//! - `Unknown`  → the race: R was commanded before its chunk was pulled/indexed. This is
//!   made rare/impossible upstream (a bounded run distributes → barrier → dispatch; a
//!   streaming run keeps the phaser generation causally downstream of availability), so
//!   at this layer it is a **counted, surfaced [`DispatchDecision::Miss`]** — never a
//!   silent skip (the "no silent caps" rule). The caller may bounded-await the index and
//!   retry before accepting the miss; the tracker counts only the accepted misses.
//!
//! The state lives here; the [`DatasetIndex`](super::dataset_session::DatasetIndex)
//! supplies `Indexed` vs `Unknown` and the payload. A `ControlledIssuer` workload wires
//! this to the runner's `TurnLifecycleObserver::on_issue` + `SlotPool` admission — the
//! seams already exist; this is the decision core.

use std::collections::HashMap;

use super::dataset_session::DatasetIndex;

/// The dispatch state of one `request_id`. `Unknown` is the implicit default (absent
/// from the tracker's map); the tracker only records the terminal transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    /// Issued, awaiting its response. A second "issue R" is a duplicate.
    InFlight,
    /// Completed (response observed or terminal). A second "issue R" is a duplicate.
    Done,
}

/// What the caller should do with an "issue request R" command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchDecision<R> {
    /// Dispatch this payload; the request is now `InFlight`.
    Issue(R),
    /// Already in-flight or done — do nothing (exactly-once-issue dedup).
    Duplicate,
    /// The request is not indexed on this cell (a `DistributionMiss`); counted and
    /// surfaced, never silently skipped.
    Miss,
}

/// Tracks per-request dispatch state for one cell and counts distribution misses. Holds
/// only the non-default (`InFlight`/`Done`) states — `Unknown` is absence.
#[derive(Debug, Default)]
pub struct DispatchTracker {
    states: HashMap<u64, RequestState>,
    misses: u64,
    issued: u64,
    completed: u64,
}

impl DispatchTracker {
    /// A fresh tracker (every request `Unknown`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Handle an "issue request R" command against this cell's dataset `index`.
    ///
    /// - `InFlight`/`Done` → [`DispatchDecision::Duplicate`] (dedup; no state change).
    /// - indexed & not yet issued → [`DispatchDecision::Issue`], transition `InFlight`.
    /// - not indexed → [`DispatchDecision::Miss`], counted (the caller should have
    ///   bounded-awaited the index first; this is the defense-in-depth guard).
    pub fn on_issue<R: Clone>(
        &mut self,
        request_id: u64,
        index: &DatasetIndex<R>,
    ) -> DispatchDecision<R> {
        match self.states.get(&request_id) {
            Some(RequestState::InFlight | RequestState::Done) => DispatchDecision::Duplicate,
            None => match index.get(request_id) {
                Some(payload) => {
                    self.states.insert(request_id, RequestState::InFlight);
                    self.issued += 1;
                    DispatchDecision::Issue(payload.clone())
                }
                None => {
                    self.misses += 1;
                    DispatchDecision::Miss
                }
            },
        }
    }

    /// Mark a request complete (`InFlight` → `Done`). Idempotent; a completion for a
    /// never-issued request is recorded as `Done` (a defensive terminal).
    pub fn on_complete(&mut self, request_id: u64) {
        let was_inflight = matches!(self.states.get(&request_id), Some(RequestState::InFlight));
        self.states.insert(request_id, RequestState::Done);
        if was_inflight {
            self.completed += 1;
        }
    }

    /// The current state of a request (`None` = `Unknown`).
    pub fn state(&self, request_id: u64) -> Option<RequestState> {
        self.states.get(&request_id).copied()
    }

    /// Total accepted distribution misses (surfaced in the report as a distinct error
    /// class, never folded into server errors).
    pub fn distribution_misses(&self) -> u64 {
        self.misses
    }

    /// Number of requests issued (transitioned to `InFlight`).
    pub fn issued(&self) -> u64 {
        self.issued
    }

    /// Number of issued requests that completed.
    pub fn completed(&self) -> u64 {
        self.completed
    }

    /// Requests still in flight (issued, not yet completed).
    pub fn in_flight(&self) -> u64 {
        self.issued.saturating_sub(self.completed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cellular::dataset_session::{DatasetIndex, DatasetPublisher, DatasetRequest};

    async fn index_of(ids: &[u64]) -> DatasetIndex<String> {
        let publisher = DatasetPublisher::<String>::new();
        let sub = publisher.attach_raw();
        publisher.add(
            ids.iter()
                .map(|&request_id| DatasetRequest {
                    request_id,
                    payload: format!("req-{request_id}"),
                })
                .collect(),
        );
        publisher.finalize();
        DatasetIndex::build_owned(sub, |_| true).await
    }

    #[tokio::test]
    async fn indexed_request_issues_once_then_dedups() {
        let index = index_of(&[7]).await;
        let mut tracker = DispatchTracker::new();
        // First issue → Issue(payload), now InFlight.
        assert_eq!(
            tracker.on_issue(7, &index),
            DispatchDecision::Issue("req-7".to_string())
        );
        assert_eq!(tracker.state(7), Some(RequestState::InFlight));
        assert_eq!(tracker.issued(), 1);
        // Re-issue while in flight → Duplicate (exactly-once-issue).
        assert_eq!(tracker.on_issue(7, &index), DispatchDecision::Duplicate);
        // Complete, then re-issue → still Duplicate.
        tracker.on_complete(7);
        assert_eq!(tracker.state(7), Some(RequestState::Done));
        assert_eq!(tracker.completed(), 1);
        assert_eq!(tracker.on_issue(7, &index), DispatchDecision::Duplicate);
        assert_eq!(tracker.issued(), 1, "no double issue");
        assert_eq!(tracker.distribution_misses(), 0);
    }

    #[tokio::test]
    async fn unknown_request_is_a_counted_miss_not_a_silent_skip() {
        let index = index_of(&[0, 1, 2]).await;
        let mut tracker = DispatchTracker::new();
        // 99 is not indexed on this cell → Miss, counted.
        assert_eq!(tracker.on_issue(99, &index), DispatchDecision::Miss);
        assert_eq!(tracker.on_issue(42, &index), DispatchDecision::Miss);
        assert_eq!(tracker.distribution_misses(), 2);
        assert_eq!(tracker.issued(), 0);
        // A miss does not mark state, so a later successful index would still issue.
        assert_eq!(tracker.state(99), None);
    }

    #[tokio::test]
    async fn in_flight_accounting_tracks_issued_minus_completed() {
        let index = index_of(&[0, 1, 2, 3]).await;
        let mut tracker = DispatchTracker::new();
        for id in 0..4 {
            assert!(matches!(
                tracker.on_issue(id, &index),
                DispatchDecision::Issue(_)
            ));
        }
        assert_eq!(tracker.in_flight(), 4);
        tracker.on_complete(0);
        tracker.on_complete(1);
        assert_eq!(tracker.in_flight(), 2);
        assert_eq!(tracker.completed(), 2);
    }
}
