// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multi-turn request-rate workload model.
//!
//! This is the Rust counterpart of Python's `conversation_source.py`,
//! `credit/structs.py`, and `credit_counter.py` for the linear synthetic online
//! path: it samples runtime sessions, builds continuation turns, and keeps the
//! root-only counters the stop-condition chain evaluates.

use aiperf_timing::{RunState, StopConfig};
use uuid::Uuid;

use crate::workload::SkeletonWorkload;

/// A sampled runtime session for one synthetic conversation template.
#[derive(Clone, Debug)]
pub struct SampledSession {
    /// Template identifier. The synthetic source has exactly one template.
    pub conversation_id: String,
    /// Runtime session identifier for sticky routing and continuation matching.
    pub x_correlation_id: String,
    /// Total root turns to send for this session.
    pub num_turns: usize,
    input_length: usize,
    max_output_tokens: usize,
}

impl SampledSession {
    /// Build the first turn of the sampled session.
    pub fn build_first_turn(&self) -> TurnToSend {
        TurnToSend::synthetic(
            self.conversation_id.clone(),
            self.x_correlation_id.clone(),
            0,
            self.num_turns,
            self.input_length,
            self.max_output_tokens,
            None,
        )
    }
}

/// A turn awaiting credit issuance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnToSend {
    /// Template identifier.
    pub conversation_id: String,
    /// Runtime session identifier.
    pub x_correlation_id: String,
    /// Zero-based turn index in the session.
    pub turn_index: usize,
    /// Total number of root turns in this session.
    pub num_turns: usize,
    /// Prompt length for accounting.
    pub input_length: usize,
    /// Requested output tokens.
    pub max_output_tokens: usize,
    /// Prompt text sent on the wire.
    pub prompt_text: String,
    /// Delay before this continuation becomes issuable.
    pub delay_ms: Option<u64>,
}

impl TurnToSend {
    /// Build a synthetic turn over the fixed lorem segment pool placeholder.
    pub fn synthetic(
        conversation_id: String,
        x_correlation_id: String,
        turn_index: usize,
        num_turns: usize,
        input_length: usize,
        max_output_tokens: usize,
        delay_ms: Option<u64>,
    ) -> Self {
        let segment = vec!["lorem"; input_length].join(" ");
        Self {
            conversation_id,
            x_correlation_id,
            turn_index,
            num_turns: num_turns.max(1),
            input_length,
            max_output_tokens,
            prompt_text: format!("turn {turn_index}: {segment}"),
            delay_ms,
        }
    }

    /// Whether this is the session's final root turn.
    pub fn is_final_turn(&self) -> bool {
        self.turn_index + 1 >= self.num_turns
    }
}

/// Issued credit metadata retained until terminal return.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IssuedCredit {
    /// Monotonic credit id assigned before dispatch.
    pub id: u64,
    /// Template identifier.
    pub conversation_id: String,
    /// Runtime session identifier.
    pub x_correlation_id: String,
    /// Zero-based turn index in the session.
    pub turn_index: usize,
    /// Total number of root turns in this session.
    pub num_turns: usize,
    /// Prompt length for accounting.
    pub input_length: usize,
    /// Requested output tokens.
    pub max_output_tokens: usize,
}

impl IssuedCredit {
    /// Build issued-credit metadata from a turn and assigned id.
    pub fn from_turn(id: u64, turn: &TurnToSend) -> Self {
        Self {
            id,
            conversation_id: turn.conversation_id.clone(),
            x_correlation_id: turn.x_correlation_id.clone(),
            turn_index: turn.turn_index,
            num_turns: turn.num_turns,
            input_length: turn.input_length,
            max_output_tokens: turn.max_output_tokens,
        }
    }

    /// Whether this credit represents the final root turn for its session.
    pub fn is_final_turn(&self) -> bool {
        self.turn_index + 1 >= self.num_turns
    }
}

/// Source of runtime conversation sessions and continuation metadata.
pub trait ConversationSource {
    /// Sample the next runtime session.
    fn next(&mut self) -> SampledSession;

    /// Build the next turn after a returned credit, if one exists.
    fn next_turn(&self, credit: &IssuedCredit) -> Option<TurnToSend>;
}

/// Synthetic source used by the walking-skeleton online CLI.
pub struct SyntheticConversationSource {
    workload: SkeletonWorkload,
}

impl SyntheticConversationSource {
    /// Create a synthetic source from the current CLI workload settings.
    pub fn new(workload: SkeletonWorkload) -> Self {
        Self { workload }
    }
}

impl ConversationSource for SyntheticConversationSource {
    fn next(&mut self) -> SampledSession {
        SampledSession {
            conversation_id: "synthetic".to_string(),
            x_correlation_id: Uuid::new_v4().to_string(),
            num_turns: self.workload.turns.max(1),
            input_length: self.workload.input_tokens,
            max_output_tokens: self.workload.output_tokens,
        }
    }

    fn next_turn(&self, credit: &IssuedCredit) -> Option<TurnToSend> {
        let next_index = credit.turn_index + 1;
        if next_index >= credit.num_turns {
            return None;
        }
        Some(TurnToSend::synthetic(
            credit.conversation_id.clone(),
            credit.x_correlation_id.clone(),
            next_index,
            credit.num_turns,
            credit.input_length,
            credit.max_output_tokens,
            self.workload.think_time_ms,
        ))
    }
}

/// Lock-free-by-serialization counters for the single issuer loop.
#[derive(Default)]
pub struct CreditCounter {
    requests_sent: u64,
    root_requests_sent: u64,
    sent_sessions: u64,
    total_session_turns: u64,
}

impl CreditCounter {
    /// Increment sent counters and return `(credit_id, is_final_credit)`.
    pub fn increment_sent(&mut self, turn: &TurnToSend, stop: &StopConfig) -> (u64, bool) {
        let credit_id = self.requests_sent;
        let new_sent = self.requests_sent + 1;
        let new_root_sent = self.root_requests_sent + 1;
        let mut new_sessions = self.sent_sessions;
        let mut new_total_turns = self.total_session_turns;

        if turn.turn_index == 0 {
            new_sessions += 1;
            new_total_turns += turn.num_turns as u64;
        }

        let is_final_credit = stop
            .total_expected_requests
            .is_some_and(|total| new_sent >= total)
            || stop.expected_num_sessions.is_some_and(|expected| {
                new_sessions >= expected && new_root_sent >= new_total_turns
            });

        self.requests_sent = new_sent;
        self.root_requests_sent = new_root_sent;
        self.sent_sessions = new_sessions;
        self.total_session_turns = new_total_turns;

        (credit_id, is_final_credit)
    }

    /// Snapshot counters as a StopChecker [`RunState`].
    pub fn run_state(&self, started_at_ns: i64, sending_complete: bool) -> RunState {
        RunState {
            requests_sent: self.requests_sent,
            root_requests_sent: self.root_requests_sent,
            sent_sessions: self.sent_sessions,
            total_session_turns: self.total_session_turns,
            cancelled: false,
            sending_complete,
            started_at_ns,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workload::SkeletonWorkload;
    use aiperf_timing::StopConfig;

    #[test]
    fn synthetic_source_reuses_template_but_mints_session_ids() {
        let workload = SkeletonWorkload {
            num_requests: 0,
            input_tokens: 4,
            output_tokens: 2,
            turns: 3,
            think_time_ms: Some(7),
        };
        let mut source = SyntheticConversationSource::new(workload);
        let a = source.next().build_first_turn();
        let b = source.next().build_first_turn();
        assert_eq!(a.conversation_id, "synthetic");
        assert_eq!(b.conversation_id, "synthetic");
        assert_ne!(a.x_correlation_id, b.x_correlation_id);
        assert_eq!(a.turn_index, 0);
        assert_eq!(a.num_turns, 3);
        assert_eq!(a.delay_ms, None);
        assert_eq!(a.input_length, 4);
        assert_eq!(a.max_output_tokens, 2);
    }

    #[test]
    fn next_turn_carries_session_and_think_time() {
        let workload = SkeletonWorkload {
            num_requests: 0,
            input_tokens: 4,
            output_tokens: 2,
            turns: 3,
            think_time_ms: Some(7),
        };
        let mut source = SyntheticConversationSource::new(workload);
        let first = source.next().build_first_turn();
        let credit = IssuedCredit::from_turn(0, &first);
        let next = source.next_turn(&credit).unwrap();
        assert_eq!(next.x_correlation_id, first.x_correlation_id);
        assert_eq!(next.turn_index, 1);
        assert_eq!(next.num_turns, 3);
        assert_eq!(next.delay_ms, Some(7));
        assert!(source
            .next_turn(&IssuedCredit {
                turn_index: 2,
                ..credit
            })
            .is_none());
    }

    #[test]
    fn counter_matches_python_root_counting_rules() {
        let mut counter = CreditCounter::default();
        let stop = StopConfig {
            total_expected_requests: None,
            expected_num_sessions: Some(2),
            expected_duration_ns: None,
        };
        let t0 = TurnToSend::synthetic("c1".into(), "s1".into(), 0, 2, 4, 2, None);
        let (id0, final0) = counter.increment_sent(&t0, &stop);
        assert_eq!(id0, 0);
        assert!(!final0);
        let t1 = TurnToSend::synthetic("c1".into(), "s1".into(), 1, 2, 4, 2, None);
        let (_, final1) = counter.increment_sent(&t1, &stop);
        assert!(!final1);
        let t2 = TurnToSend::synthetic("c2".into(), "s2".into(), 0, 1, 4, 2, None);
        let (_, final2) = counter.increment_sent(&t2, &stop);
        assert!(final2);
        assert_eq!(counter.run_state(10, false).requests_sent, 3);
        assert_eq!(counter.run_state(10, false).root_requests_sent, 3);
        assert_eq!(counter.run_state(10, false).sent_sessions, 2);
        assert_eq!(counter.run_state(10, false).total_session_turns, 3);
    }
}
