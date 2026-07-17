// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transition-validated phase lifecycle over an injected clock.
//!
//! Every timestamp and deadline comes from the same
//! [`Clock`](crate::clock::Clock).

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

use crate::clock::Clock;
use serde::{Deserialize, Serialize};

use super::{GracePeriod, PhaseConfig};

/// Validated phase lifecycle state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseState {
    /// The phase has not started.
    #[default]
    Created,
    /// Issuance and progress reporting have started.
    Started,
    /// No additional requests may be issued.
    SendingComplete,
    /// Return handling and finalization are complete.
    Complete,
}

/// Why a phase reached [`PhaseState::Complete`].
///
/// Distinct reasons prevent reports from mislabeling user cancellation as a
/// grace timeout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseCompletionReason {
    /// Every issued request returned within the configured window.
    Completed,
    /// The return grace deadline elapsed, then cancellation drained cleanly.
    GraceTimeout,
    /// An external cancellation stopped the phase.
    Cancelled,
    /// Cancellation drain also timed out and cleanup forced completion.
    ForceCompleted,
    /// Execution failed and local lifecycle finalization ran.
    Failed,
}

/// Read-only lifecycle snapshot passed into stats construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseLifecycleSnapshot {
    /// Current validated state.
    pub state: PhaseState,
    /// Clock timestamp at phase start.
    pub started_at_ns: Option<i64>,
    /// Clock timestamp when issuance closed.
    pub sending_complete_at_ns: Option<i64>,
    /// Clock timestamp when return handling completed.
    pub complete_at_ns: Option<i64>,
    /// Sending ended because its duration deadline elapsed.
    pub timeout_triggered: bool,
    /// The grace deadline elapsed before all requests returned.
    pub grace_period_timeout_triggered: bool,
    /// The bounded cancellation-drain deadline elapsed.
    pub cancel_drain_timeout_triggered: bool,
    /// The stuck-request backstop forced completion.
    pub forced_completion: bool,
    /// Cancellation was requested at any lifecycle state.
    pub was_cancelled: bool,
    /// Terminal reason once complete.
    pub completion_reason: Option<PhaseCompletionReason>,
}

/// Explicit CREATED → STARTED → SENDING_COMPLETE → COMPLETE state machine.
pub struct PhaseLifecycle {
    clock: Rc<dyn Clock>,
    expected_duration_ns: Option<i64>,
    grace_period: GracePeriod,
    snapshot: PhaseLifecycleSnapshot,
}

impl PhaseLifecycle {
    pub fn new(clock: Rc<dyn Clock>, config: &PhaseConfig) -> Self {
        Self {
            clock,
            expected_duration_ns: config.stop.expected_duration_ns,
            grace_period: config.grace_period,
            snapshot: PhaseLifecycleSnapshot {
                state: PhaseState::Created,
                started_at_ns: None,
                sending_complete_at_ns: None,
                complete_at_ns: None,
                timeout_triggered: false,
                grace_period_timeout_triggered: false,
                cancel_drain_timeout_triggered: false,
                forced_completion: false,
                was_cancelled: false,
                completion_reason: None,
            },
        }
    }

    /// Transition CREATED → STARTED and stamp the injected clock.
    pub fn start(&mut self) -> Result<(), PhaseLifecycleError> {
        if self.snapshot.state != PhaseState::Created {
            return Err(PhaseLifecycleError::AlreadyStarted(self.snapshot.state));
        }
        self.snapshot.state = PhaseState::Started;
        self.snapshot.started_at_ns = Some(self.clock.now_ns());
        Ok(())
    }

    /// Transition STARTED → SENDING_COMPLETE.
    pub fn mark_sending_complete(
        &mut self,
        timeout_triggered: bool,
    ) -> Result<(), PhaseLifecycleError> {
        match self.snapshot.state {
            PhaseState::Created => return Err(PhaseLifecycleError::NotStarted),
            PhaseState::SendingComplete | PhaseState::Complete => {
                return Err(PhaseLifecycleError::SendingAlreadyComplete(
                    self.snapshot.state,
                ));
            }
            PhaseState::Started => {}
        }
        self.snapshot.state = PhaseState::SendingComplete;
        self.snapshot.sending_complete_at_ns = Some(self.clock.now_ns());
        self.snapshot.timeout_triggered |= timeout_triggered;
        Ok(())
    }

    /// Transition SENDING_COMPLETE → COMPLETE with an unambiguous reason.
    pub fn mark_complete(
        &mut self,
        reason: PhaseCompletionReason,
    ) -> Result<(), PhaseLifecycleError> {
        if self.snapshot.state != PhaseState::SendingComplete {
            return match self.snapshot.state {
                PhaseState::Complete => Err(PhaseLifecycleError::AlreadyComplete),
                state => Err(PhaseLifecycleError::SendingNotComplete(state)),
            };
        }
        self.snapshot.state = PhaseState::Complete;
        self.snapshot.complete_at_ns = Some(self.clock.now_ns());
        self.snapshot.grace_period_timeout_triggered |= matches!(
            reason,
            PhaseCompletionReason::GraceTimeout | PhaseCompletionReason::ForceCompleted
        );
        self.snapshot.forced_completion |= reason == PhaseCompletionReason::ForceCompleted;
        self.snapshot.completion_reason = Some(reason);
        Ok(())
    }

    /// Does not change lifecycle state.
    pub fn cancel(&mut self) {
        self.snapshot.was_cancelled = true;
    }

    pub fn mark_cancel_drain_timeout(&mut self) {
        self.snapshot.cancel_drain_timeout_triggered = true;
    }

    /// Remaining clock nanoseconds before the sending or return deadline.
    ///
    /// `None` means no deadline: either no phase duration was configured, the
    /// lifecycle has not started, or return handling uses infinite grace.
    pub fn time_left_ns(&self, include_grace_period: bool) -> Option<i64> {
        let duration = self.expected_duration_ns?;
        let started = self.snapshot.started_at_ns?;
        let allowance = if include_grace_period {
            match self.grace_period {
                GracePeriod::Disabled => duration,
                GracePeriod::Finite(grace) => duration.saturating_add(grace),
                GracePeriod::Infinite => return None,
            }
        } else {
            duration
        };
        Some(
            allowance
                .saturating_sub(self.clock.now_ns().saturating_sub(started))
                .max(0),
        )
    }

    /// Current immutable lifecycle fields.
    pub fn snapshot(&self) -> PhaseLifecycleSnapshot {
        self.snapshot
    }

    /// True after the STARTED transition.
    pub fn is_started(&self) -> bool {
        self.snapshot.state != PhaseState::Created
    }

    /// True after issuance has closed.
    pub fn is_sending_complete(&self) -> bool {
        matches!(
            self.snapshot.state,
            PhaseState::SendingComplete | PhaseState::Complete
        )
    }

    /// True after terminal finalization.
    pub fn is_complete(&self) -> bool {
        self.snapshot.state == PhaseState::Complete
    }
}

/// Invalid lifecycle transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseLifecycleError {
    /// `start` was called outside CREATED.
    AlreadyStarted(PhaseState),
    /// Sending completion was requested before start.
    NotStarted,
    /// Sending completion was requested more than once.
    SendingAlreadyComplete(PhaseState),
    /// Final completion was requested before sending completion.
    SendingNotComplete(PhaseState),
    /// Final completion was requested more than once.
    AlreadyComplete,
}

impl Display for PhaseLifecycleError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyStarted(state) => {
                write!(f, "phase already started from state {state:?}")
            }
            Self::NotStarted => write!(f, "phase not started; call start first"),
            Self::SendingAlreadyComplete(state) => {
                write!(f, "phase sending already complete in state {state:?}")
            }
            Self::SendingNotComplete(state) => write!(
                f,
                "phase has not completed sending; current state is {state:?}"
            ),
            Self::AlreadyComplete => write!(f, "phase already complete"),
        }
    }
}

impl Error for PhaseLifecycleError {}

#[cfg(test)]
mod tests {
    use crate::clock::sim_clock::SimClock;

    use super::*;
    use crate::timing::{PhaseKind, StopConfig};

    fn lifecycle(
        duration_ns: Option<i64>,
        grace_period: GracePeriod,
    ) -> (Rc<SimClock>, PhaseLifecycle) {
        let clock = Rc::new(SimClock::new());
        let config = PhaseConfig::new(
            "profiling",
            PhaseKind::Profiling,
            StopConfig {
                expected_duration_ns: duration_ns,
                ..StopConfig::default()
            },
        )
        .with_grace_period(grace_period);
        let clock_dyn: Rc<dyn Clock> = clock.clone();
        (clock, PhaseLifecycle::new(clock_dyn, &config))
    }

    #[test]
    fn full_lifecycle_uses_one_clock_timeline() {
        let (clock, mut lifecycle) = lifecycle(Some(100), GracePeriod::Finite(20));
        lifecycle.start().unwrap();
        clock.advance_to(40);
        lifecycle.mark_sending_complete(false).unwrap();
        clock.advance_to(75);
        lifecycle
            .mark_complete(PhaseCompletionReason::Completed)
            .unwrap();

        let snapshot = lifecycle.snapshot();
        assert_eq!(snapshot.state, PhaseState::Complete);
        assert_eq!(snapshot.started_at_ns, Some(0));
        assert_eq!(snapshot.sending_complete_at_ns, Some(40));
        assert_eq!(snapshot.complete_at_ns, Some(75));
    }

    #[test]
    fn invalid_transition_guards_enforce_lifecycle_ordering() {
        let (_, mut lifecycle) = lifecycle(None, GracePeriod::Disabled);
        assert_eq!(
            lifecycle.mark_sending_complete(false),
            Err(PhaseLifecycleError::NotStarted)
        );
        assert_eq!(
            lifecycle.mark_complete(PhaseCompletionReason::Completed),
            Err(PhaseLifecycleError::SendingNotComplete(PhaseState::Created))
        );
        lifecycle.start().unwrap();
        assert_eq!(
            lifecycle.start(),
            Err(PhaseLifecycleError::AlreadyStarted(PhaseState::Started))
        );
        lifecycle.mark_sending_complete(false).unwrap();
        assert_eq!(
            lifecycle.mark_sending_complete(false),
            Err(PhaseLifecycleError::SendingAlreadyComplete(
                PhaseState::SendingComplete
            ))
        );
        lifecycle
            .mark_complete(PhaseCompletionReason::Completed)
            .unwrap();
        assert_eq!(
            lifecycle.mark_complete(PhaseCompletionReason::Completed),
            Err(PhaseLifecycleError::AlreadyComplete)
        );
    }

    #[test]
    fn cancellation_is_orthogonal_and_reason_is_not_overloaded() {
        let (_, mut lifecycle) = lifecycle(None, GracePeriod::Disabled);
        lifecycle.cancel();
        lifecycle.start().unwrap();
        lifecycle.mark_sending_complete(false).unwrap();
        lifecycle
            .mark_complete(PhaseCompletionReason::Cancelled)
            .unwrap();

        let snapshot = lifecycle.snapshot();
        assert!(snapshot.was_cancelled);
        assert!(!snapshot.grace_period_timeout_triggered);
        assert_eq!(
            snapshot.completion_reason,
            Some(PhaseCompletionReason::Cancelled)
        );
    }

    #[test]
    fn time_left_includes_finite_grace_and_clamps_at_zero() {
        let (clock, mut lifecycle) = lifecycle(Some(100), GracePeriod::Finite(25));
        assert_eq!(lifecycle.time_left_ns(false), None);
        lifecycle.start().unwrap();
        assert_eq!(lifecycle.time_left_ns(false), Some(100));
        assert_eq!(lifecycle.time_left_ns(true), Some(125));
        clock.advance_to(80);
        assert_eq!(lifecycle.time_left_ns(false), Some(20));
        assert_eq!(lifecycle.time_left_ns(true), Some(45));
        clock.advance_to(200);
        assert_eq!(lifecycle.time_left_ns(false), Some(0));
        assert_eq!(lifecycle.time_left_ns(true), Some(0));
    }

    #[test]
    fn missing_duration_and_infinite_warmup_grace_have_no_deadline() {
        let (_, mut no_duration) = lifecycle(None, GracePeriod::Finite(25));
        no_duration.start().unwrap();
        assert_eq!(no_duration.time_left_ns(false), None);
        assert_eq!(no_duration.time_left_ns(true), None);

        let (_, mut infinite) = lifecycle(Some(100), GracePeriod::Infinite);
        infinite.start().unwrap();
        assert_eq!(infinite.time_left_ns(false), Some(100));
        assert_eq!(infinite.time_left_ns(true), None);
    }
}
