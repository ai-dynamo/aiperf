// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed phase progress snapshot shared by online and simulated runs.

use serde::{Deserialize, Serialize};

use super::{
    GracePeriod, PhaseCompletionReason, PhaseConfig, PhaseKind, PhaseLifecycle, PhaseProgress,
    PhaseState,
};

/// Immutable phase lifecycle and progress snapshot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseStats {
    /// Stable phase identifier.
    pub phase_id: String,
    /// Warmup or profiling role.
    pub kind: PhaseKind,
    /// Current lifecycle state.
    pub state: PhaseState,
    /// Clock timestamp at phase start.
    pub start_ns: Option<i64>,
    /// Clock timestamp when issuance closed.
    pub sent_end_ns: Option<i64>,
    /// Clock timestamp when return handling completed.
    pub requests_end_ns: Option<i64>,
    /// Configured request cap.
    pub total_expected_requests: Option<u64>,
    /// Configured session cap.
    pub expected_num_sessions: Option<u64>,
    /// Configured duration.
    pub expected_duration_ns: Option<i64>,
    /// Configured grace policy.
    pub grace_period: GracePeriod,
    /// Current requests issued.
    pub requests_sent: u64,
    /// Current requests completed.
    pub requests_completed: u64,
    /// Current requests cancelled.
    pub requests_cancelled: u64,
    /// Current errored requests.
    pub request_errors: u64,
    /// Current sessions admitted.
    pub sent_sessions: u64,
    /// Current sessions completed.
    pub completed_sessions: u64,
    /// Current sessions cancelled.
    pub cancelled_sessions: u64,
    /// Planned turns across every admitted root session.
    pub total_session_turns: u64,
    /// Current in-flight requests.
    pub in_flight_requests: u64,
    /// Current in-flight sessions.
    pub in_flight_sessions: u64,
    /// Current in-flight prefills.
    pub in_flight_prefills: u64,
    /// Pending branch work that gates completion.
    pub pending_branch_work: u64,
    /// Session slots recovered during force completion.
    pub stuck_session_slots_released: u64,
    /// Prefill slots recovered during force completion.
    pub stuck_prefill_slots_released: u64,
    /// Frozen requests issued at sending completion.
    pub final_requests_sent: Option<u64>,
    /// Frozen requests completed at phase completion.
    pub final_requests_completed: Option<u64>,
    /// Frozen requests cancelled at phase completion.
    pub final_requests_cancelled: Option<u64>,
    /// Frozen errored requests at phase completion.
    pub final_request_errors: Option<u64>,
    /// Frozen sessions admitted at sending completion.
    pub final_sent_sessions: Option<u64>,
    /// Frozen sessions completed at phase completion.
    pub final_completed_sessions: Option<u64>,
    /// Frozen sessions cancelled at phase completion.
    pub final_cancelled_sessions: Option<u64>,
    /// Sending stopped at the duration deadline.
    pub timeout_triggered: bool,
    /// Returns exceeded their grace deadline.
    pub grace_period_timeout_triggered: bool,
    /// Cancelled returns exceeded the drain deadline.
    pub cancel_drain_timeout_triggered: bool,
    /// Stuck-request cleanup forced phase completion.
    pub forced_completion: bool,
    /// External cancellation was requested.
    pub was_cancelled: bool,
    /// Terminal completion reason, absent before COMPLETE.
    pub completion_reason: Option<PhaseCompletionReason>,
}

impl PhaseStats {
    /// Snapshot one phase without sharing mutable counter state with observers.
    pub fn snapshot(
        config: &PhaseConfig,
        lifecycle: &PhaseLifecycle,
        progress: &PhaseProgress,
    ) -> Self {
        let lifecycle = lifecycle.snapshot();
        let counters = progress.snapshot();
        Self {
            phase_id: config.id.clone(),
            kind: config.kind,
            state: lifecycle.state,
            start_ns: lifecycle.started_at_ns,
            sent_end_ns: lifecycle.sending_complete_at_ns,
            requests_end_ns: lifecycle.complete_at_ns,
            total_expected_requests: config.stop.total_expected_requests,
            expected_num_sessions: config.stop.expected_num_sessions,
            expected_duration_ns: config.stop.expected_duration_ns,
            grace_period: config.grace_period,
            requests_sent: counters.requests_sent,
            requests_completed: counters.requests_completed,
            requests_cancelled: counters.requests_cancelled,
            request_errors: counters.request_errors,
            sent_sessions: counters.sent_sessions,
            completed_sessions: counters.completed_sessions,
            cancelled_sessions: counters.cancelled_sessions,
            total_session_turns: counters.total_session_turns,
            in_flight_requests: counters.in_flight_requests(),
            in_flight_sessions: counters.in_flight_sessions(),
            in_flight_prefills: counters.in_flight_prefills(),
            pending_branch_work: counters.pending_branch_work,
            stuck_session_slots_released: counters.stuck_session_slots_released,
            stuck_prefill_slots_released: counters.stuck_prefill_slots_released,
            final_requests_sent: counters.final_requests_sent,
            final_requests_completed: counters.final_requests_completed,
            final_requests_cancelled: counters.final_requests_cancelled,
            final_request_errors: counters.final_request_errors,
            final_sent_sessions: counters.final_sent_sessions,
            final_completed_sessions: counters.final_completed_sessions,
            final_cancelled_sessions: counters.final_cancelled_sessions,
            timeout_triggered: lifecycle.timeout_triggered,
            grace_period_timeout_triggered: lifecycle.grace_period_timeout_triggered,
            cancel_drain_timeout_triggered: lifecycle.cancel_drain_timeout_triggered,
            forced_completion: lifecycle.forced_completion,
            was_cancelled: lifecycle.was_cancelled,
            completion_reason: lifecycle.completion_reason,
        }
    }
}
