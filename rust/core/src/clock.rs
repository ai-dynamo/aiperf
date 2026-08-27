// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Boundary-owned clock abstraction for real and deterministic virtual execution.
//!
//! Every measurement and firing gate in the product routes its time through
//! [`Clock`]. The trait lives here rather than in the runtime so a plugin can
//! be authored, and its determinism tested, against the boundary alone.

use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

/// Immutable pairing of one instant on a run's monotonic timeline with the UTC
/// instant it represents.
///
/// Captured exactly once per run incarnation through
/// [`Clock::capture_utc_anchor`] and never re-read: a wall clock can step
/// backward under NTP or a VM resume, while `Clock::now_ns` cannot, so every
/// UTC-labelled value in the product is derived from this pairing rather than
/// from a fresh `SystemTime` read. `Copy` and serializable so a controller can
/// ship the UTC fact to a cell, and so a checkpoint can record which UTC
/// instant an incarnation was anchored to, without any peer reading a clock.
///
/// A restored anchor is valid only for the incarnation that captured it: its
/// `monotonic_ns` names a dead process's timeline origin. Event time is
/// durable; the monotonic pairing is per-incarnation and must be re-captured.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UtcMonotonicAnchor {
    /// Nanoseconds since the Unix epoch at the anchored instant.
    pub utc_ns: i64,
    /// The same instant on this run's monotonic timeline.
    pub monotonic_ns: i64,
    /// Half-width of the read bracket that produced the pairing; zero for a
    /// virtual clock, whose authored epoch is exact by construction.
    pub uncertainty_ns: i64,
}

impl UtcMonotonicAnchor {
    /// Construct a validated anchor.
    ///
    /// Rejects a UTC instant before the Unix epoch and a negative uncertainty;
    /// a negative `monotonic_ns` is permitted because a virtual timeline may be
    /// anchored at an arbitrary integer coordinate.
    pub fn new(
        utc_ns: i64,
        monotonic_ns: i64,
        uncertainty_ns: i64,
    ) -> Result<Self, ClockAnchorError> {
        if utc_ns < 0 {
            return Err(ClockAnchorError::BeforeUnixEpoch(utc_ns));
        }
        if uncertainty_ns < 0 {
            return Err(ClockAnchorError::InvalidUncertainty(uncertainty_ns));
        }
        Ok(Self {
            utc_ns,
            monotonic_ns,
            uncertainty_ns,
        })
    }

    /// Project a monotonic instant onto UTC.
    ///
    /// The single derivation site for "what wall-clock instant is this?". The
    /// result is non-decreasing in `monotonic_ns` because the function is
    /// affine and increasing, so a sequence of [`Clock::now_ns`] readings always
    /// projects to a non-decreasing UTC sequence no matter what the system wall
    /// clock does mid-run.
    pub fn utc_ns_at(&self, monotonic_ns: i64) -> Result<i64, ClockAnchorError> {
        let elapsed_ns = monotonic_ns
            .checked_sub(self.monotonic_ns)
            .ok_or(ClockAnchorError::AnchorArithmeticOverflow)?;
        let utc_ns = self
            .utc_ns
            .checked_add(elapsed_ns)
            .ok_or(ClockAnchorError::AnchorArithmeticOverflow)?;
        if utc_ns < 0 {
            return Err(ClockAnchorError::BeforeUnixEpoch(utc_ns));
        }
        Ok(utc_ns)
    }
}

/// Why a run could not be anchored to UTC.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClockAnchorError {
    /// The caller's uncertainty budget, or a computed uncertainty, is negative.
    InvalidUncertainty(i64),
    /// This clock can neither read a wall clock nor accept an authored epoch.
    Unsupported,
    /// A virtual clock has no wall clock, so an authored epoch is mandatory.
    AuthoredEpochRequired,
    /// A real clock reads its own UTC and refuses a conflicting authored epoch.
    AuthoredEpochUnsupported,
    /// The resolved UTC instant precedes the Unix epoch.
    BeforeUnixEpoch(i64),
    /// The resolved UTC nanosecond count does not fit a signed 64-bit integer.
    EpochOutOfRange(u128),
    /// The monotonic bracket around the wall-clock read was wider than allowed.
    UncertaintyExceeded {
        /// Half-width of the observed bracket.
        observed_ns: i64,
        /// Half-width the caller was willing to accept.
        max_ns: i64,
    },
    /// The two monotonic readings bracketing the wall-clock read regressed.
    ///
    /// Unreachable on `CLOCK_MONOTONIC`; checked because the alternative is a
    /// negative `uncertainty_ns` silently poisoning every later comparison.
    MonotonicRegression {
        /// Reading taken before the wall-clock read.
        first_ns: i64,
        /// Reading taken after the wall-clock read.
        second_ns: i64,
    },
    /// Checked arithmetic overflowed while anchoring or projecting.
    AnchorArithmeticOverflow,
}

impl std::fmt::Display for ClockAnchorError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidUncertainty(value) => {
                write!(formatter, "clock anchor uncertainty {value}ns is negative")
            }
            Self::Unsupported => formatter.write_str("this clock cannot anchor a run to UTC"),
            Self::AuthoredEpochRequired => {
                formatter.write_str("a virtual clock requires an authored UTC epoch")
            }
            Self::AuthoredEpochUnsupported => {
                formatter.write_str("a real clock reads UTC and rejects an authored epoch")
            }
            Self::BeforeUnixEpoch(value) => {
                write!(formatter, "anchored UTC {value}ns is before the Unix epoch")
            }
            Self::EpochOutOfRange(value) => {
                write!(
                    formatter,
                    "anchored UTC {value}ns does not fit i64 nanoseconds"
                )
            }
            Self::UncertaintyExceeded {
                observed_ns,
                max_ns,
            } => write!(
                formatter,
                "clock anchor uncertainty {observed_ns}ns exceeds the {max_ns}ns budget"
            ),
            Self::MonotonicRegression {
                first_ns,
                second_ns,
            } => write!(
                formatter,
                "monotonic clock regressed from {first_ns}ns to {second_ns}ns while anchoring"
            ),
            Self::AnchorArithmeticOverflow => {
                formatter.write_str("clock anchor arithmetic overflowed")
            }
        }
    }
}

impl std::error::Error for ClockAnchorError {}

/// The result of driving a graph to quiescence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunOutcome {
    /// Tasks remained parked with no future clock event to wake them.
    pub deadlocked: bool,
}

/// A sleepable time source.
pub trait Clock {
    /// Current time in nanoseconds (virtual for sim, monotonic for real).
    fn now_ns(&self) -> i64;

    /// A future that resolves after `duration_ns` of this clock's time.
    /// Non-positive durations resolve after a single task yield.
    fn sleep(self: Rc<Self>, duration_ns: i64) -> Pin<Box<dyn Future<Output = ()>>>;

    // Virtual-time control stays on the simulation clock because a real clock
    // cannot advance explicitly.

    /// Whether this clock requires an idle pump to advance virtual time.
    fn is_virtual(&self) -> bool {
        false
    }

    /// Anchor this run's monotonic timeline to a UTC instant.
    ///
    /// Called exactly once per run incarnation; the returned anchor is shared
    /// by value thereafter. Implementations must stay stateless — memoizing
    /// here would hide a double-capture bug instead of surfacing it, and
    /// `Clock` is shared as `Rc<dyn Clock>` across a `LocalSet`.
    ///
    /// The default body serves virtual clocks, which have no wall clock and so
    /// require `authored_utc_epoch_ns` and pair it with zero uncertainty; any
    /// other clock is refused with [`ClockAnchorError::Unsupported`] because it
    /// cannot honestly name a UTC instant. A real clock overrides this and is
    /// the only place in the product that reads `SystemTime`.
    fn capture_utc_anchor(
        &self,
        authored_utc_epoch_ns: Option<i64>,
        max_uncertainty_ns: i64,
    ) -> Result<UtcMonotonicAnchor, ClockAnchorError> {
        if max_uncertainty_ns < 0 {
            return Err(ClockAnchorError::InvalidUncertainty(max_uncertainty_ns));
        }
        if !self.is_virtual() {
            return Err(ClockAnchorError::Unsupported);
        }
        let utc_ns = authored_utc_epoch_ns.ok_or(ClockAnchorError::AuthoredEpochRequired)?;
        UtcMonotonicAnchor::new(utc_ns, self.now_ns(), 0)
    }

    /// Drive `body` to completion using this clock's reactor discipline.
    ///
    /// The default drives on a current-thread tokio runtime whose IO/timer
    /// reactor wakes real sleepers. A virtual clock overrides this with
    /// deterministic event-by-event advancement; a [`RunOutcome::deadlocked`]
    /// result means no virtual event can make progress.
    fn drive(self: Rc<Self>, body: Pin<Box<dyn Future<Output = ()> + '_>>) -> RunOutcome {
        // IO + time only: this driver needs no signal handling. See
        // turn_execution::run_worker_thread for why this does not remove
        // tokio's child-process orphan sweep from the park path.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .expect("current-thread runtime for real-clock run driver");
        tokio::task::LocalSet::new().block_on(&runtime, body);
        RunOutcome { deadlocked: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// Minimal virtual clock: the default `capture_utc_anchor` body serves it.
    struct CountingVirtualClock {
        now_ns: Cell<i64>,
    }

    impl Clock for CountingVirtualClock {
        fn now_ns(&self) -> i64 {
            self.now_ns.get()
        }

        fn sleep(self: Rc<Self>, _duration_ns: i64) -> Pin<Box<dyn Future<Output = ()>>> {
            Box::pin(std::future::ready(()))
        }

        fn is_virtual(&self) -> bool {
            true
        }
    }

    /// Neither virtual nor wall-clock capable: refused rather than allowed to
    /// invent a UTC instant.
    struct OpaqueClock;

    impl Clock for OpaqueClock {
        fn now_ns(&self) -> i64 {
            0
        }

        fn sleep(self: Rc<Self>, _duration_ns: i64) -> Pin<Box<dyn Future<Output = ()>>> {
            Box::pin(std::future::ready(()))
        }
    }

    #[test]
    fn virtual_clock_requires_an_authored_epoch_and_anchors_exactly() {
        let clock = CountingVirtualClock {
            now_ns: Cell::new(42),
        };
        assert_eq!(
            clock.capture_utc_anchor(None, 10),
            Err(ClockAnchorError::AuthoredEpochRequired)
        );

        let anchor = clock
            .capture_utc_anchor(Some(1_000_000_000), 0)
            .expect("an authored epoch anchors a virtual clock");
        assert_eq!(anchor.utc_ns, 1_000_000_000);
        assert_eq!(anchor.monotonic_ns, 42);
        assert_eq!(anchor.uncertainty_ns, 0);
    }

    /// The budget check precedes the virtual/real branch, so an ill-formed
    /// budget is rejected identically on every clock.
    #[test]
    fn a_negative_budget_is_rejected_before_the_clock_branch() {
        let virtual_clock = CountingVirtualClock {
            now_ns: Cell::new(0),
        };
        assert_eq!(
            virtual_clock.capture_utc_anchor(Some(0), -1),
            Err(ClockAnchorError::InvalidUncertainty(-1))
        );
        assert_eq!(
            OpaqueClock.capture_utc_anchor(None, -1),
            Err(ClockAnchorError::InvalidUncertainty(-1))
        );
        assert_eq!(
            OpaqueClock.capture_utc_anchor(None, 10),
            Err(ClockAnchorError::Unsupported)
        );
    }

    /// Projection is affine and increasing in the monotonic reading, so a
    /// derived UTC sequence is non-decreasing whatever the wall clock does.
    #[test]
    fn projection_is_a_pure_increasing_function_of_the_anchor() {
        let anchor = UtcMonotonicAnchor::new(1_000_000_000, 50, 2).expect("valid anchor");
        for monotonic_ns in [50, 51, 1_000, 1_000_000] {
            assert_eq!(
                anchor.utc_ns_at(monotonic_ns),
                Ok(1_000_000_000 + (monotonic_ns - 50))
            );
        }
        assert!(anchor.utc_ns_at(100) > anchor.utc_ns_at(99));
        assert_eq!(
            anchor.utc_ns_at(i64::MAX),
            Err(ClockAnchorError::AnchorArithmeticOverflow)
        );
    }

    /// The cellular path ships the anchor by value, so its encoding must round
    /// trip exactly.
    #[test]
    fn an_anchor_round_trips_through_serde() {
        let anchor = UtcMonotonicAnchor::new(1_764_000_000_000_000_000, -7, 3).expect("valid");
        let encoded = serde_json::to_string(&anchor).expect("anchor encodes");
        let decoded: UtcMonotonicAnchor = serde_json::from_str(&encoded).expect("anchor decodes");
        assert_eq!(decoded, anchor);
    }

    #[test]
    fn an_anchor_rejects_a_pre_epoch_instant_and_a_negative_uncertainty() {
        assert_eq!(
            UtcMonotonicAnchor::new(-1, 0, 0),
            Err(ClockAnchorError::BeforeUnixEpoch(-1))
        );
        assert_eq!(
            UtcMonotonicAnchor::new(0, 0, -1),
            Err(ClockAnchorError::InvalidUncertainty(-1))
        );
    }
}
