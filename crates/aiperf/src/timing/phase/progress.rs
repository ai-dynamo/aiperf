// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lock-free-by-serialization phase progress and one-shot notifications.
//!
//! Mutations are
//! synchronous on one local executor; only event waits are async.

use std::cell::{Cell, RefCell};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

use tokio::sync::Notify;

use crate::timing::StopConfig;

/// Facts needed when one wire request is issued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseSend {
    /// Whether this wire belongs to the root conversation rather than a DAG child.
    pub is_root: bool,
    /// Whether this is turn zero of a newly admitted root session.
    pub starts_session: bool,
    /// Planned root turns for a newly admitted session; ignored otherwise.
    pub planned_session_turns: u64,
}

impl PhaseSend {
    /// Construct a single-turn root session.
    pub fn single_turn_session() -> Self {
        Self {
            is_root: true,
            starts_session: true,
            planned_session_turns: 1,
        }
    }

    /// Construct a continuation in an existing root session.
    pub fn root_continuation() -> Self {
        Self {
            is_root: true,
            starts_session: false,
            planned_session_turns: 0,
        }
    }

    /// Construct a DAG-child wire that inherits its root's session.
    pub fn dag_child() -> Self {
        Self {
            is_root: false,
            starts_session: false,
            planned_session_turns: 0,
        }
    }
}

/// Result of atomically recording one issued request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseSendOutcome {
    /// Zero-based identifier within this phase.
    pub request_index: u64,
    /// Whether configured request/session bounds make this the final send.
    pub is_final_request: bool,
}

/// Facts needed when one wire request reaches terminal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseReturn {
    /// Whether the returned root turn closes its session.
    pub completes_session: bool,
    /// Whether transport classified the request as cancelled.
    pub cancelled: bool,
    /// Whether a non-cancelled return carried an error.
    pub errored: bool,
    /// Whether terminal must release prefill because TTFT never arrived.
    pub releases_prefill: bool,
}

/// Result of atomically recording one terminal request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseReturnOutcome {
    /// Counters advanced but other requests or branch work remain.
    Pending,
    /// Every frozen sent request is accounted for and branch work is drained.
    AllReturned,
    /// Completion counts were already frozen; the late callback was ignored.
    LateIgnored,
}

/// Snapshot of mutable and frozen phase counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseProgressCounters {
    /// Total wire requests issued, including DAG children.
    pub requests_sent: u64,
    /// Root wire requests issued.
    pub root_requests_sent: u64,
    /// Successfully returned requests.
    pub requests_completed: u64,
    /// Cancelled returned requests.
    pub requests_cancelled: u64,
    /// Non-cancelled requests that returned an error.
    pub request_errors: u64,
    /// Root sessions admitted.
    pub sent_sessions: u64,
    /// Root sessions completed.
    pub completed_sessions: u64,
    /// Root sessions cancelled.
    pub cancelled_sessions: u64,
    /// Planned turns across admitted root sessions.
    pub total_session_turns: u64,
    /// Prefill slots released by TTFT or terminal fallback.
    pub prefills_released: u64,
    /// Frozen sent count after issuance closes.
    pub final_requests_sent: Option<u64>,
    /// Frozen completed count after phase completion.
    pub final_requests_completed: Option<u64>,
    /// Frozen cancelled count after phase completion.
    pub final_requests_cancelled: Option<u64>,
    /// Frozen error count after phase completion.
    pub final_request_errors: Option<u64>,
    /// Frozen admitted-session count after issuance closes.
    pub final_sent_sessions: Option<u64>,
    /// Frozen completed-session count after phase completion.
    pub final_completed_sessions: Option<u64>,
    /// Frozen cancelled-session count after phase completion.
    pub final_cancelled_sessions: Option<u64>,
    /// Branch/fan-out work that must drain before completion.
    pub pending_branch_work: u64,
    /// Session slots explicitly released by force-completion cleanup.
    pub stuck_session_slots_released: u64,
    /// Prefill slots explicitly released by force-completion cleanup.
    pub stuck_prefill_slots_released: u64,
}

impl PhaseProgressCounters {
    /// Issued requests not yet returned or cancelled.
    pub fn in_flight_requests(&self) -> u64 {
        self.requests_sent
            .saturating_sub(self.requests_completed + self.requests_cancelled)
    }

    /// Started sessions whose terminal root turn has not returned.
    pub fn in_flight_sessions(&self) -> u64 {
        self.sent_sessions
            .saturating_sub(self.completed_sessions + self.cancelled_sessions)
    }

    /// Requests that have not released their prefill slot.
    pub fn in_flight_prefills(&self) -> u64 {
        self.requests_sent.saturating_sub(self.prefills_released)
    }
}

struct PhaseProgressInner {
    stop: StopConfig,
    counters: RefCell<PhaseProgressCounters>,
    sent_frozen: Cell<bool>,
    completed_frozen: Cell<bool>,
    all_sent: Cell<bool>,
    all_returned: Cell<bool>,
    sent_notify: Notify,
    returned_notify: Notify,
}

/// Cloneable local-loop progress handle shared by issuer and return callbacks.
#[derive(Clone)]
pub struct PhaseProgress {
    inner: Rc<PhaseProgressInner>,
}

impl PhaseProgress {
    /// Create zeroed progress for the configured stop bounds.
    pub fn new(stop: StopConfig) -> Self {
        Self {
            inner: Rc::new(PhaseProgressInner {
                stop,
                counters: RefCell::new(PhaseProgressCounters::default()),
                sent_frozen: Cell::new(false),
                completed_frozen: Cell::new(false),
                all_sent: Cell::new(false),
                all_returned: Cell::new(false),
                sent_notify: Notify::new(),
                returned_notify: Notify::new(),
            }),
        }
    }

    /// Atomically record one issued request.
    ///
    /// When this crosses the configured request/session bound, sent counts are
    /// frozen before the all-sent event is published.
    pub fn record_sent(&self, sent: PhaseSend) -> Result<PhaseSendOutcome, PhaseProgressError> {
        self.record_sent_batch(&[sent])
            .map(|mut outcomes| outcomes.pop().expect("one input send produces one outcome"))
    }

    /// Atomically record one admitted batch before evaluating stop bounds.
    ///
    /// A Graph-IR root and its statically admitted DAG children form one
    /// indivisible admission decision. Evaluating a session bound after only
    /// the root would freeze progress before those already-admitted children
    /// can be counted. Empty batches are accepted as a no-op.
    pub fn record_sent_batch(
        &self,
        sent: &[PhaseSend],
    ) -> Result<Vec<PhaseSendOutcome>, PhaseProgressError> {
        if sent.is_empty() {
            return Ok(Vec::new());
        }
        if self.inner.sent_frozen.get() {
            return Err(PhaseProgressError::SentAfterFreeze);
        }
        if sent
            .iter()
            .any(|sent| sent.starts_session && (!sent.is_root || sent.planned_session_turns == 0))
        {
            return Err(PhaseProgressError::InvalidSessionStart);
        }

        let (first_request_index, is_final_request) = {
            let mut counters = self.inner.counters.borrow_mut();
            let first_request_index = counters.requests_sent;
            for sent in sent {
                counters.requests_sent = counters.requests_sent.saturating_add(1);
                if sent.is_root {
                    counters.root_requests_sent = counters.root_requests_sent.saturating_add(1);
                }
                if sent.starts_session {
                    counters.sent_sessions = counters.sent_sessions.saturating_add(1);
                    counters.total_session_turns = counters
                        .total_session_turns
                        .saturating_add(sent.planned_session_turns);
                }
            }
            let request_bound = self
                .inner
                .stop
                .total_expected_requests
                .is_some_and(|expected| counters.requests_sent >= expected);
            let session_bound = self
                .inner
                .stop
                .expected_num_sessions
                .is_some_and(|expected| {
                    counters.sent_sessions >= expected
                        && counters.root_requests_sent >= counters.total_session_turns
                });
            (first_request_index, request_bound || session_bound)
        };

        if is_final_request {
            self.mark_sending_complete();
        }
        Ok((0..sent.len())
            .map(|offset| PhaseSendOutcome {
                request_index: first_request_index.saturating_add(offset as u64),
                is_final_request: is_final_request && offset + 1 == sent.len(),
            })
            .collect())
    }

    /// Freeze sent counters without publishing the all-sent event.
    ///
    /// The runner uses this split operation so it can preserve the required
    /// lifecycle → freeze → cancel-pending → signal ordering on timeout.
    pub fn freeze_sent_counts(&self) {
        if !self.inner.sent_frozen.replace(true) {
            let mut counters = self.inner.counters.borrow_mut();
            counters.final_requests_sent = Some(counters.requests_sent);
            counters.final_sent_sessions = Some(counters.sent_sessions);
        }
    }

    /// Publish the all-sent event after sent counts have been frozen.
    pub fn signal_all_sent(&self) {
        debug_assert!(self.inner.sent_frozen.get());
        if !self.inner.all_sent.replace(true) {
            self.inner.sent_notify.notify_waiters();
        }
        self.maybe_signal_all_returned();
    }

    /// Freeze sent counters, then publish the all-sent one-shot event.
    pub fn mark_sending_complete(&self) {
        self.freeze_sent_counts();
        self.signal_all_sent();
    }

    /// Atomically record one terminal callback.
    pub fn record_returned(&self, returned: PhaseReturn) -> PhaseReturnOutcome {
        if self.inner.completed_frozen.get() {
            return PhaseReturnOutcome::LateIgnored;
        }
        {
            let mut counters = self.inner.counters.borrow_mut();
            if returned.cancelled {
                counters.requests_cancelled += 1;
                if returned.completes_session {
                    counters.cancelled_sessions += 1;
                }
            } else {
                counters.requests_completed += 1;
                if returned.completes_session {
                    counters.completed_sessions += 1;
                }
                if returned.errored {
                    counters.request_errors += 1;
                }
            }
            if returned.releases_prefill {
                counters.prefills_released += 1;
            }
        }
        if self.maybe_signal_all_returned() {
            PhaseReturnOutcome::AllReturned
        } else {
            PhaseReturnOutcome::Pending
        }
    }

    /// Record the first-token release for one request.
    pub fn record_first_token(&self) {
        if !self.inner.completed_frozen.get() {
            self.inner.counters.borrow_mut().prefills_released += 1;
        }
    }

    /// Add branch work that must finish before all-returned can be signalled.
    pub fn begin_branch_work(&self) {
        let mut counters = self.inner.counters.borrow_mut();
        counters.pending_branch_work = counters.pending_branch_work.saturating_add(1);
    }

    /// Finish one unit of branch work and re-evaluate phase completion.
    pub fn finish_branch_work(&self) -> Result<bool, PhaseProgressError> {
        {
            let mut counters = self.inner.counters.borrow_mut();
            if counters.pending_branch_work == 0 {
                return Err(PhaseProgressError::BranchWorkUnderflow);
            }
            counters.pending_branch_work -= 1;
        }
        Ok(self.maybe_signal_all_returned())
    }

    /// True when every frozen send is accounted for and branch work is empty.
    pub fn check_all_returned_or_cancelled(&self) -> bool {
        let counters = self.inner.counters.borrow();
        counters.final_requests_sent.is_some_and(|sent| {
            counters.requests_completed + counters.requests_cancelled >= sent
                && counters.pending_branch_work == 0
        })
    }

    fn maybe_signal_all_returned(&self) -> bool {
        if self.check_all_returned_or_cancelled() {
            if !self.inner.all_returned.replace(true) {
                self.inner.returned_notify.notify_waiters();
            }
            true
        } else {
            false
        }
    }

    /// Force the all-returned event without falsifying completion counters.
    pub fn force_all_returned(&self) {
        if !self.inner.all_returned.replace(true) {
            self.inner.returned_notify.notify_waiters();
        }
    }

    /// Record slots recovered because cancelled work never returned.
    pub fn record_stuck_slots_released(&self, session: u64, prefill: u64) {
        let mut counters = self.inner.counters.borrow_mut();
        counters.stuck_session_slots_released = counters
            .stuck_session_slots_released
            .saturating_add(session);
        counters.stuck_prefill_slots_released = counters
            .stuck_prefill_slots_released
            .saturating_add(prefill);
    }

    /// Freeze terminal counters so late callbacks cannot alter final stats.
    pub fn freeze_completed_counts(&self) {
        if self.inner.completed_frozen.replace(true) {
            return;
        }
        let mut counters = self.inner.counters.borrow_mut();
        counters.final_requests_completed = Some(counters.requests_completed);
        counters.final_requests_cancelled = Some(counters.requests_cancelled);
        counters.final_request_errors = Some(counters.request_errors);
        counters.final_completed_sessions = Some(counters.completed_sessions);
        counters.final_cancelled_sessions = Some(counters.cancelled_sessions);
    }

    /// Wait until sent counts have been frozen and issuance has closed.
    pub async fn wait_all_sent(&self) {
        wait_flag(&self.inner.all_sent, &self.inner.sent_notify).await;
    }

    /// Wait until returns are complete or the runner forces the event.
    pub async fn wait_all_returned(&self) {
        wait_flag(&self.inner.all_returned, &self.inner.returned_notify).await;
    }

    /// Whether issuance has closed.
    pub fn all_sent(&self) -> bool {
        self.inner.all_sent.get()
    }

    /// Whether return handling has been signalled complete.
    pub fn all_returned(&self) -> bool {
        self.inner.all_returned.get()
    }

    /// Copy current and frozen counters.
    pub fn snapshot(&self) -> PhaseProgressCounters {
        *self.inner.counters.borrow()
    }
}

async fn wait_flag(flag: &Cell<bool>, notify: &Notify) {
    loop {
        let notified = notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if flag.get() {
            return;
        }
        notified.await;
    }
}

/// Invalid phase progress mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseProgressError {
    /// An issuer attempted to send after sent counts were frozen.
    SentAfterFreeze,
    /// Session-start metadata was inconsistent or had zero planned turns.
    InvalidSessionStart,
    /// Branch completion was recorded without matching pending work.
    BranchWorkUnderflow,
}

impl Display for PhaseProgressError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SentAfterFreeze => write!(f, "cannot record a send after sending complete"),
            Self::InvalidSessionStart => write!(
                f,
                "a new session must be a root request with at least one planned turn"
            ),
            Self::BranchWorkUnderflow => {
                write!(f, "cannot finish branch work when none is pending")
            }
        }
    }
}

impl Error for PhaseProgressError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_send_freezes_before_return_completion_check() {
        let progress = PhaseProgress::new(StopConfig {
            total_expected_requests: Some(1),
            ..StopConfig::default()
        });
        let returned = progress.record_returned(PhaseReturn {
            completes_session: true,
            ..PhaseReturn::default()
        });
        assert_eq!(returned, PhaseReturnOutcome::Pending);

        let sent = progress
            .record_sent(PhaseSend::single_turn_session())
            .unwrap();
        assert!(sent.is_final_request);
        assert!(progress.all_sent());
        assert!(progress.all_returned());
        assert_eq!(progress.snapshot().final_requests_sent, Some(1));
    }

    #[test]
    fn branch_work_conjunct_defers_return_completion() {
        let progress = PhaseProgress::new(StopConfig {
            total_expected_requests: Some(1),
            ..StopConfig::default()
        });
        progress.begin_branch_work();
        progress
            .record_sent(PhaseSend::single_turn_session())
            .unwrap();
        assert_eq!(
            progress.record_returned(PhaseReturn {
                completes_session: true,
                ..PhaseReturn::default()
            }),
            PhaseReturnOutcome::Pending
        );
        assert!(!progress.all_returned());
        assert!(progress.finish_branch_work().unwrap());
        assert!(progress.all_returned());
    }

    #[test]
    fn request_cap_counts_children_but_session_cap_uses_root_turns() {
        let progress = PhaseProgress::new(StopConfig {
            total_expected_requests: Some(4),
            expected_num_sessions: Some(1),
            expected_duration_ns: None,
        });
        let first = PhaseSend {
            is_root: true,
            starts_session: true,
            planned_session_turns: 2,
        };
        assert!(!progress.record_sent(first).unwrap().is_final_request);
        assert!(
            !progress
                .record_sent(PhaseSend::dag_child())
                .unwrap()
                .is_final_request
        );
        assert!(
            progress
                .record_sent(PhaseSend::root_continuation())
                .unwrap()
                .is_final_request
        );
        let snapshot = progress.snapshot();
        assert_eq!(snapshot.requests_sent, 3);
        assert_eq!(snapshot.root_requests_sent, 2);
        assert_eq!(snapshot.final_requests_sent, Some(3));
    }

    #[test]
    fn graph_admission_batch_freezes_only_after_every_child_is_counted() {
        let progress = PhaseProgress::new(StopConfig {
            expected_num_sessions: Some(1),
            ..StopConfig::default()
        });
        let outcomes = progress
            .record_sent_batch(&[
                PhaseSend::single_turn_session(),
                PhaseSend::dag_child(),
                PhaseSend::dag_child(),
            ])
            .unwrap();

        assert_eq!(outcomes.len(), 3);
        assert!(!outcomes[0].is_final_request);
        assert!(!outcomes[1].is_final_request);
        assert!(outcomes[2].is_final_request);
        let snapshot = progress.snapshot();
        assert_eq!(snapshot.requests_sent, 3);
        assert_eq!(snapshot.root_requests_sent, 1);
        assert_eq!(snapshot.sent_sessions, 1);
        assert_eq!(snapshot.final_requests_sent, Some(3));
        assert_eq!(
            progress.record_sent(PhaseSend::dag_child()),
            Err(PhaseProgressError::SentAfterFreeze)
        );
    }

    #[test]
    fn frozen_completion_ignores_late_returns() {
        let progress = PhaseProgress::new(StopConfig::default());
        progress
            .record_sent(PhaseSend::single_turn_session())
            .unwrap();
        progress.mark_sending_complete();
        progress.freeze_completed_counts();
        assert_eq!(
            progress.record_returned(PhaseReturn::default()),
            PhaseReturnOutcome::LateIgnored
        );
        let snapshot = progress.snapshot();
        assert_eq!(snapshot.requests_completed, 0);
        assert_eq!(snapshot.final_requests_completed, Some(0));
    }
}
