// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ordered stop-condition chain.
//!
//! Decides whether the load generator may send more requests. Each
//! [`StopCondition`] is a pure, read-only predicate over a [`RunState`] snapshot
//! and the current time (integer nanoseconds, the `Clock`'s native unit). A
//! condition [`applies`](StopCondition::applies) to a run only when the relevant
//! limit is configured (e.g. `Duration` applies iff a duration was requested), so
//! the [`StopChecker`] evaluates just the conditions that matter.
//!
//! Time never enters through a wall clock: the caller passes `now_ns`
//! (typically `clock.now_ns()`) into every check, keeping this module
//! clock-native yet free of any clock dependency and trivially testable by
//! injecting a value.
//!
//! Two decisions, both "first no wins":
//! - [`can_send_any`](StopChecker::can_send_any) — may ANY turn (first or
//!   subsequent) be sent? ALL applicable conditions must allow it.
//! - [`can_start_new_session`](StopChecker::can_start_new_session) — may a NEW
//!   session begin? More restrictive: `can_send_any` must pass first, then every
//!   condition's [`can_start_new_session`](StopCondition::can_start_new_session)
//!   must also allow it.

/// Configured stop thresholds. Each `Some` value activates the corresponding
/// [`StopCondition`]; `None` leaves it out of the chain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StopConfig {
    /// Cap on total wire requests (`--request-count`). Activates [`RequestCount`].
    pub total_expected_requests: Option<u64>,
    /// Cap on distinct sessions/conversations (`--num-conversations`). Activates
    /// [`SessionCount`].
    pub expected_num_sessions: Option<u64>,
    /// Benchmark duration in nanoseconds (`--benchmark-duration`). Activates
    /// [`Duration`].
    pub expected_duration_ns: Option<i64>,
}

/// Read-only snapshot of the counters and lifecycle flags the stop conditions
/// evaluate. Never mutated by this module.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunState {
    /// Total wire requests sent so far (across all sessions, including DAG
    /// children / background forks).
    pub requests_sent: u64,
    /// Wire requests sent for ROOT turns only (excludes reactively-dispatched
    /// DAG children). Compared against [`total_session_turns`](Self::total_session_turns)
    /// for session-completion — see [`SessionCount`].
    pub root_requests_sent: u64,
    /// Number of distinct sessions started so far.
    pub sent_sessions: u64,
    /// Total planned ROOT turns across all sessions in the sampler plan.
    pub total_session_turns: u64,
    /// The run was externally cancelled (e.g. Ctrl+C).
    pub cancelled: bool,
    /// The strategy loop marked all credits as sent.
    pub sending_complete: bool,
    /// Absolute clock time (ns) at which the phase started; the [`Duration`]
    /// condition measures elapsed time from here.
    pub started_at_ns: i64,
}

/// A single ordered stop condition: a pure predicate over a [`RunState`]
/// snapshot and `now_ns`.
pub trait StopCondition {
    /// Whether this condition is relevant for the given configuration (Python
    /// `should_use`). Conditions that don't apply are excluded from the chain.
    fn applies(cfg: &StopConfig) -> bool
    where
        Self: Sized;

    /// True if the phase may send ANY turn (first or subsequent) right now.
    fn can_send_any(&self, state: &RunState, now_ns: i64) -> bool;

    /// True if the phase may start a NEW session. Defaults to `true` (no extra
    /// restriction); overridden by [`SessionCount`] to gate new sessions while
    /// still allowing continuation turns of already-started sessions.
    fn can_start_new_session(&self, _state: &RunState, _now_ns: i64) -> bool {
        true
    }
}

/// Lifecycle gate: honored on every run. Blocks all sends once the run is
/// cancelled or the strategy loop has finished sending.
pub struct Lifecycle;

impl StopCondition for Lifecycle {
    fn applies(_cfg: &StopConfig) -> bool {
        true
    }

    fn can_send_any(&self, state: &RunState, _now_ns: i64) -> bool {
        !state.cancelled && !state.sending_complete
    }
}

/// Request-count cap: applies iff `total_expected_requests` is set. Stops once
/// `requests_sent` reaches the cap.
pub struct RequestCount {
    total: u64,
}

impl StopCondition for RequestCount {
    fn applies(cfg: &StopConfig) -> bool {
        cfg.total_expected_requests.is_some()
    }

    fn can_send_any(&self, state: &RunState, _now_ns: i64) -> bool {
        state.requests_sent < self.total
    }
}

/// Session-count cap: applies iff `expected_num_sessions` is set.
///
/// `can_send_any` stays true while EITHER the session limit is not yet reached
/// (a new session could still start) OR already-started sessions have unsent
/// turns remaining. The "unsent turns" check compares ROOT-only counters
/// (`root_requests_sent` vs `total_session_turns`) rather than the global
/// `requests_sent`: for background-fork parents, child wires arrive in parallel
/// with the parent's later turns and inflate the global counter past the root's
/// planned wire count, which would prematurely stop continuation turns.
///
/// `can_start_new_session` is the stricter gate — new sessions are blocked once
/// the session limit is reached, even though `can_send_any` still permits
/// finishing existing sessions' turns.
pub struct SessionCount {
    expected: u64,
}

impl StopCondition for SessionCount {
    fn applies(cfg: &StopConfig) -> bool {
        cfg.expected_num_sessions.is_some()
    }

    fn can_send_any(&self, state: &RunState, _now_ns: i64) -> bool {
        state.sent_sessions < self.expected || state.root_requests_sent < state.total_session_turns
    }

    fn can_start_new_session(&self, state: &RunState, _now_ns: i64) -> bool {
        state.sent_sessions < self.expected
    }
}

/// Duration cap: applies iff `expected_duration_ns` is set. Stops once the
/// elapsed time since `started_at_ns` reaches the configured duration.
pub struct Duration {
    expected_duration_ns: i64,
}

impl StopCondition for Duration {
    fn applies(cfg: &StopConfig) -> bool {
        cfg.expected_duration_ns.is_some()
    }

    fn can_send_any(&self, state: &RunState, now_ns: i64) -> bool {
        let time_left = self.expected_duration_ns - (now_ns - state.started_at_ns);
        time_left > 0
    }
}

/// Evaluates the ordered stop-condition chain for a run. Built from a
/// [`StopConfig`], it holds exactly the conditions that [`applies`](StopCondition::applies)
/// to that configuration (`Lifecycle` is always present and first).
pub struct StopChecker {
    conditions: Vec<Box<dyn StopCondition>>,
}

impl StopChecker {
    /// Build the checker, selecting the applicable conditions in canonical order:
    /// lifecycle, request-count, session-count, duration.
    pub fn new(cfg: &StopConfig) -> Self {
        let mut conditions: Vec<Box<dyn StopCondition>> = Vec::new();

        // Lifecycle is always used and evaluated first.
        conditions.push(Box::new(Lifecycle));
        if RequestCount::applies(cfg) {
            conditions.push(Box::new(RequestCount {
                total: cfg.total_expected_requests.expect("checked by applies"),
            }));
        }
        if SessionCount::applies(cfg) {
            conditions.push(Box::new(SessionCount {
                expected: cfg.expected_num_sessions.expect("checked by applies"),
            }));
        }
        if Duration::applies(cfg) {
            conditions.push(Box::new(Duration {
                expected_duration_ns: cfg.expected_duration_ns.expect("checked by applies"),
            }));
        }

        Self { conditions }
    }

    /// True if the phase may send ANY turn — every applicable condition must
    /// allow it (first "no" wins).
    pub fn can_send_any(&self, state: &RunState, now_ns: i64) -> bool {
        self.conditions
            .iter()
            .all(|c| c.can_send_any(state, now_ns))
    }

    /// True if the phase may start a NEW session. `can_send_any` must pass
    /// first, then every condition's `can_start_new_session` must also allow it.
    pub fn can_start_new_session(&self, state: &RunState, now_ns: i64) -> bool {
        if !self.can_send_any(state, now_ns) {
            return false;
        }
        self.conditions
            .iter()
            .all(|c| c.can_start_new_session(state, now_ns))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Base state with all sends allowed (nothing configured, no limits hit).
    fn ready_state() -> RunState {
        RunState::default()
    }

    #[test]
    fn lifecycle_blocks_when_cancelled() {
        let checker = StopChecker::new(&StopConfig::default());
        let mut state = ready_state();
        assert!(checker.can_send_any(&state, 0));
        state.cancelled = true;
        assert!(!checker.can_send_any(&state, 0));
    }

    #[test]
    fn lifecycle_blocks_when_sending_complete() {
        let checker = StopChecker::new(&StopConfig::default());
        let mut state = ready_state();
        state.sending_complete = true;
        assert!(!checker.can_send_any(&state, 0));
        // A new session is likewise blocked (can_send_any gates it).
        assert!(!checker.can_start_new_session(&state, 0));
    }

    #[test]
    fn request_count_stops_at_limit() {
        let cfg = StopConfig {
            total_expected_requests: Some(3),
            ..StopConfig::default()
        };
        let checker = StopChecker::new(&cfg);
        let mut state = ready_state();

        state.requests_sent = 2;
        assert!(checker.can_send_any(&state, 0));
        state.requests_sent = 3;
        assert!(!checker.can_send_any(&state, 0));
        state.requests_sent = 4;
        assert!(!checker.can_send_any(&state, 0));
    }

    #[test]
    fn duration_stops_once_now_passes_started_plus_expected() {
        let cfg = StopConfig {
            expected_duration_ns: Some(1_000),
            ..StopConfig::default()
        };
        let checker = StopChecker::new(&cfg);
        let mut state = ready_state();
        state.started_at_ns = 500;

        // now within window: 500 + 999 < 500 + 1000 -> time_left > 0.
        assert!(checker.can_send_any(&state, 1_499));
        // now exactly at the boundary: time_left == 0 -> stop.
        assert!(!checker.can_send_any(&state, 1_500));
        // now past the boundary.
        assert!(!checker.can_send_any(&state, 2_000));
    }

    #[test]
    fn session_count_gates_new_sessions_but_allows_continuation_turns() {
        let cfg = StopConfig {
            expected_num_sessions: Some(2),
            ..StopConfig::default()
        };
        let checker = StopChecker::new(&cfg);
        let mut state = ready_state();

        // Session limit reached, but planned root turns remain unsent:
        // can_send_any still true (continuation), can_start_new_session false.
        state.sent_sessions = 2;
        state.total_session_turns = 5;
        state.root_requests_sent = 3;
        assert!(checker.can_send_any(&state, 0));
        assert!(!checker.can_start_new_session(&state, 0));

        // All planned root turns sent AND limit reached -> fully stopped.
        state.root_requests_sent = 5;
        assert!(!checker.can_send_any(&state, 0));
        assert!(!checker.can_start_new_session(&state, 0));

        // Under the limit -> new sessions allowed again.
        state.sent_sessions = 1;
        assert!(checker.can_send_any(&state, 0));
        assert!(checker.can_start_new_session(&state, 0));
    }

    #[test]
    fn session_count_uses_root_counters_not_global_requests() {
        // Global requests_sent is irrelevant to SessionCount; only the
        // root-vs-total comparison drives the continuation branch.
        let cfg = StopConfig {
            expected_num_sessions: Some(1),
            ..StopConfig::default()
        };
        let checker = StopChecker::new(&cfg);
        let mut state = ready_state();
        state.sent_sessions = 1; // limit reached
        state.requests_sent = 1_000; // inflated by parallel child wires
        state.total_session_turns = 4;
        state.root_requests_sent = 2; // root still has turns to send

        // Despite the huge global requests_sent, continuation is allowed
        // because root_requests_sent < total_session_turns.
        assert!(checker.can_send_any(&state, 0));
    }

    #[test]
    fn combined_conditions_first_no_wins() {
        let cfg = StopConfig {
            total_expected_requests: Some(10),
            expected_num_sessions: Some(3),
            expected_duration_ns: Some(1_000),
        };
        let checker = StopChecker::new(&cfg);
        let mut state = ready_state();
        state.started_at_ns = 0;

        // All conditions satisfied.
        state.requests_sent = 5;
        state.sent_sessions = 1;
        state.total_session_turns = 6;
        state.root_requests_sent = 2;
        assert!(checker.can_send_any(&state, 500));
        assert!(checker.can_start_new_session(&state, 500));

        // Request-count exhausted alone blocks everything.
        state.requests_sent = 10;
        assert!(!checker.can_send_any(&state, 500));
        state.requests_sent = 5;

        // Duration elapsed alone blocks everything.
        assert!(!checker.can_send_any(&state, 1_000));

        // Session limit reached blocks new sessions but not continuation.
        state.sent_sessions = 3;
        assert!(checker.can_send_any(&state, 500));
        assert!(!checker.can_start_new_session(&state, 500));

        // Cancelled blocks everything regardless of other slack.
        state.cancelled = true;
        assert!(!checker.can_send_any(&state, 500));
    }

    #[test]
    fn only_configured_conditions_apply() {
        // No limits configured: only Lifecycle is active, so any counter value
        // is fine as long as the run is live.
        let checker = StopChecker::new(&StopConfig::default());
        let mut state = ready_state();
        state.requests_sent = u64::MAX;
        state.sent_sessions = u64::MAX;
        assert!(checker.can_send_any(&state, i64::MAX));
        assert!(checker.can_start_new_session(&state, i64::MAX));
    }
}
