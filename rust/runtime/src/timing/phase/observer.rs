// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct phase lifecycle observation without a message bus.
//!
//! This trait retains the phase event content while deleting the ZMQ wire
//! envelopes. Calls are synchronous and local because phase state is already
//! serialized on one `LocalSet`.

use std::cell::RefCell;

use serde::{Deserialize, Serialize};

use super::{PhaseConfig, PhaseStats};

/// Optional DAG/dataflow counters attached to a phase-complete event.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseBranchStats {
    /// Branch work still pending at the snapshot instant.
    pub pending_work: u64,
    /// Branches or child tasks started during the phase.
    pub started: u64,
    /// Branches or child tasks completed during the phase.
    pub completed: u64,
    /// Branches suppressed by stop policy.
    pub suppressed: u64,
}

/// Local observer seam for phase lifecycle and progress events.
pub trait PhaseObserver {
    /// Observe the STARTED transition.
    fn on_phase_start(&self, config: &PhaseConfig, stats: PhaseStats);

    /// Observe a periodic or transition-adjacent progress snapshot.
    fn on_progress(&self, stats: PhaseStats);

    /// Observe the SENDING_COMPLETE transition.
    fn on_sending_complete(&self, stats: PhaseStats);

    /// Observe the COMPLETE transition and optional branch counters.
    fn on_phase_complete(&self, stats: PhaseStats, branch_stats: Option<PhaseBranchStats>);

    /// Observe completion of the ordered phase list.
    fn on_phases_complete(&self, _stats: Vec<PhaseStats>) {}
}

/// Observer that intentionally discards all phase events.
#[derive(Default)]
pub struct NoopPhaseObserver;

impl PhaseObserver for NoopPhaseObserver {
    fn on_phase_start(&self, _config: &PhaseConfig, _stats: PhaseStats) {}

    fn on_progress(&self, _stats: PhaseStats) {}

    fn on_sending_complete(&self, _stats: PhaseStats) {}

    fn on_phase_complete(&self, _stats: PhaseStats, _branch_stats: Option<PhaseBranchStats>) {}
}

/// Phase event classification retained by [`RecordingPhaseObserver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseEventKind {
    /// STARTED transition.
    Start,
    /// Periodic or transition-adjacent progress.
    Progress,
    /// SENDING_COMPLETE transition.
    SendingComplete,
    /// COMPLETE transition.
    Complete,
}

/// One recorded phase observer event.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseEvent {
    /// Event classification.
    pub kind: PhaseEventKind,
    /// Immutable phase snapshot.
    pub stats: PhaseStats,
    /// Optional branch snapshot on completion.
    pub branch_stats: Option<PhaseBranchStats>,
}

/// Report-oriented observer retaining lifecycle events in call order.
#[derive(Default)]
pub struct RecordingPhaseObserver {
    events: RefCell<Vec<PhaseEvent>>,
    completed_runs: RefCell<Vec<Vec<PhaseStats>>>,
}

impl RecordingPhaseObserver {
    /// Copy all observed phase events in delivery order.
    pub fn events(&self) -> Vec<PhaseEvent> {
        self.events.borrow().clone()
    }

    /// Copy all ordered run-completion snapshots.
    pub fn completed_runs(&self) -> Vec<Vec<PhaseStats>> {
        self.completed_runs.borrow().clone()
    }

    fn push(
        &self,
        kind: PhaseEventKind,
        stats: PhaseStats,
        branch_stats: Option<PhaseBranchStats>,
    ) {
        self.events.borrow_mut().push(PhaseEvent {
            kind,
            stats,
            branch_stats,
        });
    }
}

impl PhaseObserver for RecordingPhaseObserver {
    fn on_phase_start(&self, _config: &PhaseConfig, stats: PhaseStats) {
        self.push(PhaseEventKind::Start, stats, None);
    }

    fn on_progress(&self, stats: PhaseStats) {
        self.push(PhaseEventKind::Progress, stats, None);
    }

    fn on_sending_complete(&self, stats: PhaseStats) {
        self.push(PhaseEventKind::SendingComplete, stats, None);
    }

    fn on_phase_complete(&self, stats: PhaseStats, branch_stats: Option<PhaseBranchStats>) {
        self.push(PhaseEventKind::Complete, stats, branch_stats);
    }

    fn on_phases_complete(&self, stats: Vec<PhaseStats>) {
        self.completed_runs.borrow_mut().push(stats);
    }
}

/// Minimal console observer for phase-owned live progress.
#[derive(Default)]
pub struct ConsolePhaseObserver;

impl PhaseObserver for ConsolePhaseObserver {
    fn on_phase_start(&self, _config: &PhaseConfig, stats: PhaseStats) {
        eprintln!("phase {} started", stats.phase_id);
    }

    fn on_progress(&self, stats: PhaseStats) {
        eprintln!(
            "phase {} progress: sent={} completed={} cancelled={} in_flight={}",
            stats.phase_id,
            stats.requests_sent,
            stats.requests_completed,
            stats.requests_cancelled,
            stats.in_flight_requests
        );
    }

    fn on_sending_complete(&self, stats: PhaseStats) {
        eprintln!(
            "phase {} sending complete: sent={} in_flight={}",
            stats.phase_id, stats.requests_sent, stats.in_flight_requests
        );
    }

    fn on_phase_complete(&self, stats: PhaseStats, _branch_stats: Option<PhaseBranchStats>) {
        eprintln!(
            "phase {} complete: completed={} cancelled={} reason={:?}",
            stats.phase_id,
            stats.final_requests_completed.unwrap_or_default(),
            stats.final_requests_cancelled.unwrap_or_default(),
            stats.completion_reason
        );
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::clock::{Clock, sim_clock::SimClock};

    use super::*;
    use crate::timing::{PhaseKind, PhaseLifecycle, PhaseProgress, StopConfig};

    #[test]
    fn recording_observer_retains_typed_event_order() {
        let config = PhaseConfig::new("profiling", PhaseKind::Profiling, StopConfig::default());
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let mut lifecycle = PhaseLifecycle::new(clock, &config);
        let progress = PhaseProgress::new(config.stop);
        lifecycle.start().unwrap();
        let stats = PhaseStats::snapshot(&config, &lifecycle, &progress);
        let observer = RecordingPhaseObserver::default();
        observer.on_phase_start(&config, stats.clone());
        observer.on_progress(stats);

        assert_eq!(
            observer
                .events()
                .iter()
                .map(|event| event.kind)
                .collect::<Vec<_>>(),
            vec![PhaseEventKind::Start, PhaseEventKind::Progress]
        );
    }
}
