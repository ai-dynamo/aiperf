// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Warmup-to-profile handoff observation for the accelerated cache-warmup
//! substage — a port of the return-observation half of Python's
//! `AgenticReplayStrategy.observe_credit_return`
//! (`src/aiperf/timing/strategies/agentic_replay.py`, lines 698-717).
//!
//! During accelerated cache-pressure warmup Python runs `observe_credit_return`
//! on *every* credit return: it advances the replay barrier gate
//! (`replay_gate.complete`), then — for non-final returns — records the live
//! credit and its return wall-time keyed by the credit's runtime correlation id,
//! and — for final returns — pops both records. Task 5 consumes these
//! `handoff_credits` / `return_wall_ns` maps to drive the compressed-turn
//! handoff; this module supplies the pure, deterministically-testable recorder.
//!
//! The [`HandoffCredit`] projection captures exactly the credit fields that
//! exist on the Rust [`IssuedCredit`]/[`TurnToSend`] seam
//! (`conversation_id`, `x_correlation_id`, `turn_index`, `num_turns`); the Python
//! `agent_depth` / `parent_correlation_id` / `root_correlation_id` / `branch_mode`
//! fields have no Rust `Credit` equivalent yet and so are intentionally absent.

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::agentx::replay_gate::ReplayGate;
use crate::multiturn::IssuedCredit;

/// Projection of the returned credit fields the warmup-to-profile handoff needs.
///
/// A pure value type (no `Rc`/lifetimes) so the recorder logic is unit-testable
/// without constructing a full [`IssuedCredit`]. Keyed into the recorder maps by
/// [`HandoffCredit::x_correlation_id`], mirroring Python's `credit.x_correlation_id`
/// dictionary key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandoffCredit {
    /// Template/conversation identity of the returned turn
    /// (`credit.conversation_id`).
    pub conversation_id: String,
    /// Runtime session correlation id — the recorder map key
    /// (`credit.x_correlation_id`).
    pub x_correlation_id: String,
    /// Zero-based index of the returned turn (`credit.turn_index`).
    pub turn_index: usize,
    /// Total turns this runtime session will send (`credit.num_turns`).
    pub num_turns: usize,
}

impl HandoffCredit {
    /// Whether the returned credit is its session's final turn (mirrors
    /// [`IssuedCredit::is_final_turn`] / Python `credit.is_final_turn`).
    pub fn is_final(&self) -> bool {
        self.turn_index + 1 >= self.num_turns
    }

    /// Project the fields the handoff needs off a returned [`IssuedCredit`].
    pub fn from_credit(credit: &IssuedCredit) -> Self {
        Self {
            conversation_id: credit.turn.conversation_id.clone(),
            x_correlation_id: credit.turn.x_correlation_id.clone(),
            turn_index: credit.turn.turn_index,
            num_turns: credit.turn.num_turns,
        }
    }
}

/// The warmup-to-profile handoff record: the last live (non-final) credit and
/// its return wall-time per correlation id.
///
/// Pure and deterministic — the caller injects the return wall through
/// [`HandoffRecorder::observe`] (routed via the runtime [`Clock`](crate::clock::Clock)
/// at the call site, never `Instant::now`). Ordered [`BTreeMap`]s make the map
/// contents and any snapshot deterministic.
#[derive(Debug, Default)]
pub struct HandoffRecorder {
    /// Live (non-final) returned credit per correlation id
    /// (Python `_handoff_credits`).
    handoff_credits: BTreeMap<String, HandoffCredit>,
    /// Return wall-clock nanoseconds per correlation id
    /// (Python `_handoff_returned_at_ns`).
    return_wall_ns: BTreeMap<String, i64>,
}

impl HandoffRecorder {
    /// Construct an empty recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe one credit return: pop both records on a final turn, otherwise
    /// record the live credit and its return wall (Python lines 710-717). The
    /// barrier-gate `complete` call is the caller's responsibility (it precedes
    /// this in `observe_credit_return`).
    pub fn observe(&mut self, credit: HandoffCredit, is_final: bool, wall_ns: i64) {
        let key = credit.x_correlation_id.clone();
        if is_final {
            self.handoff_credits.remove(&key);
            self.return_wall_ns.remove(&key);
        } else {
            self.return_wall_ns.insert(key.clone(), wall_ns);
            self.handoff_credits.insert(key, credit);
        }
    }

    /// The recorded live-credit map (Python `_handoff_credits`).
    pub fn handoff_credits(&self) -> &BTreeMap<String, HandoffCredit> {
        &self.handoff_credits
    }

    /// The recorded return-wall map (Python `_handoff_returned_at_ns`).
    pub fn return_wall_ns(&self) -> &BTreeMap<String, i64> {
        &self.return_wall_ns
    }
}

/// Bundle threaded into the accelerated-warmup return seam: the replay barrier
/// gate plus the handoff recorder, driven together on every credit return.
///
/// Held behind an `Option<Rc<AcceleratedObserver>>` at the scheduling call site;
/// `None` (every current caller) is the standard warmup/profiling path and
/// changes no runtime behavior. Interior [`RefCell`]s allow `&self` driving from
/// the shared, current-thread completion callback without `Arc<Mutex>`.
#[derive(Debug)]
pub struct AcceleratedObserver {
    /// Replay interval-barrier coordinator (Task 3).
    pub gate: RefCell<ReplayGate>,
    /// Warmup-to-profile handoff recorder.
    pub recorder: RefCell<HandoffRecorder>,
}

impl AcceleratedObserver {
    /// Bundle a barrier `gate` and handoff `recorder`.
    pub fn new(gate: ReplayGate, recorder: HandoffRecorder) -> Self {
        Self {
            gate: RefCell::new(gate),
            recorder: RefCell::new(recorder),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn credit(conv: &str, corr: &str, turn_index: usize, num_turns: usize) -> HandoffCredit {
        HandoffCredit {
            conversation_id: conv.into(),
            x_correlation_id: corr.into(),
            turn_index,
            num_turns,
        }
    }

    #[test]
    fn observe_records_non_final_and_pops_on_final() {
        let mut rec = HandoffRecorder::new();

        // Two lanes each return a non-final (turn 0 of 3) credit at a distinct
        // virtual wall — both are recorded under their correlation id.
        let a0 = credit("conv-a", "x-a", 0, 3);
        let b0 = credit("conv-b", "x-b", 0, 2);
        rec.observe(a0.clone(), a0.is_final(), 1_000);
        rec.observe(b0.clone(), b0.is_final(), 2_000);

        assert_eq!(rec.handoff_credits().get("x-a"), Some(&a0));
        assert_eq!(rec.handoff_credits().get("x-b"), Some(&b0));
        assert_eq!(rec.return_wall_ns().get("x-a"), Some(&1_000));
        assert_eq!(rec.return_wall_ns().get("x-b"), Some(&2_000));

        // A later non-final return on lane A overwrites the live credit + wall.
        let a1 = credit("conv-a", "x-a", 1, 3);
        rec.observe(a1.clone(), a1.is_final(), 3_500);
        assert_eq!(rec.handoff_credits().get("x-a"), Some(&a1));
        assert_eq!(rec.return_wall_ns().get("x-a"), Some(&3_500));

        // Lane A's final turn (2 of 3) pops BOTH maps for its correlation id;
        // lane B is untouched.
        let a2 = credit("conv-a", "x-a", 2, 3);
        assert!(a2.is_final());
        rec.observe(a2.clone(), a2.is_final(), 9_999);
        assert!(!rec.handoff_credits().contains_key("x-a"));
        assert!(!rec.return_wall_ns().contains_key("x-a"));
        assert_eq!(rec.handoff_credits().get("x-b"), Some(&b0));
        assert_eq!(rec.return_wall_ns().get("x-b"), Some(&2_000));
    }

    #[test]
    fn final_return_for_unrecorded_correlation_is_a_noop() {
        // A single-turn session (turn 0 of 1) is final on its first return: it
        // records nothing and the pop is a harmless no-op.
        let mut rec = HandoffRecorder::new();
        let only = credit("conv-c", "x-c", 0, 1);
        assert!(only.is_final());
        rec.observe(only.clone(), only.is_final(), 4_242);
        assert!(rec.handoff_credits().is_empty());
        assert!(rec.return_wall_ns().is_empty());
    }
}
