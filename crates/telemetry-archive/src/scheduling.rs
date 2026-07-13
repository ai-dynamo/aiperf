// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed-deadline telemetry cadence and single-flight issuance state.
//!
//! Completion-paced sleeps drift and hide missed observations. The types in
//! this module keep the original cadence anchor authoritative, compact skipped
//! ticks into exact inclusive ranges, and make the one-request-per-source rule
//! an explicit state transition.

use std::fmt::{self, Display, Formatter};

/// One cadence target derived from the immutable source anchor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CadenceDeadline {
    /// Zero-based cadence tick.
    pub tick: u64,
    /// Absolute injected-Clock deadline for this tick.
    pub scheduled_ns: i64,
}

/// One compact inclusive range of cadence deadlines that could not be issued.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MissedCadenceRange {
    /// First skipped tick.
    pub first_tick: u64,
    /// Last skipped tick.
    pub last_tick: u64,
    /// Absolute deadline of [`Self::first_tick`].
    pub first_deadline_ns: i64,
    /// Absolute deadline of [`Self::last_tick`].
    pub last_deadline_ns: i64,
    /// Number of skipped ticks in the inclusive range.
    pub count: u64,
}

/// Result of selecting the first cadence deadline strictly after an instant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CadenceAdvance {
    /// First future deadline after the supplied Clock instant.
    pub next: CadenceDeadline,
    /// Exact skipped range, when one or more unissued deadlines were due.
    pub missed: Option<MissedCadenceRange>,
}

/// Anchor-relative cadence that never repays timing debt with catch-up bursts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedDeadlineCadence {
    anchor_ns: i64,
    interval_ns: i64,
    next_tick: u64,
}

impl FixedDeadlineCadence {
    /// Creates a cadence whose tick zero occurs at `anchor_ns`.
    pub fn new(anchor_ns: i64, interval_ns: i64) -> Result<Self, SchedulingError> {
        if interval_ns <= 0 {
            return Err(SchedulingError::NonPositiveInterval(interval_ns));
        }
        Ok(Self {
            anchor_ns,
            interval_ns,
            next_tick: 0,
        })
    }

    /// Returns the immutable cadence anchor.
    #[must_use]
    pub const fn anchor_ns(&self) -> i64 {
        self.anchor_ns
    }

    /// Returns the positive cadence interval.
    #[must_use]
    pub const fn interval_ns(&self) -> i64 {
        self.interval_ns
    }

    /// Returns the next unissued tick index.
    #[must_use]
    pub const fn next_tick_index(&self) -> u64 {
        self.next_tick
    }

    /// Computes one tick deadline without changing cadence state.
    pub fn deadline_for(&self, tick: u64) -> Result<CadenceDeadline, SchedulingError> {
        let deadline = i128::from(self.anchor_ns)
            .checked_add(
                i128::from(self.interval_ns)
                    .checked_mul(i128::from(tick))
                    .ok_or(SchedulingError::ArithmeticOverflow)?,
            )
            .ok_or(SchedulingError::ArithmeticOverflow)?;
        Ok(CadenceDeadline {
            tick,
            scheduled_ns: i64::try_from(deadline)
                .map_err(|_| SchedulingError::ArithmeticOverflow)?,
        })
    }

    /// Returns the next unissued deadline without advancing it.
    pub fn next_deadline(&self) -> Result<CadenceDeadline, SchedulingError> {
        self.deadline_for(self.next_tick)
    }

    /// Marks the next deadline issued and returns its stable tick identity.
    pub fn issue_next(&mut self) -> Result<CadenceDeadline, SchedulingError> {
        let deadline = self.next_deadline()?;
        self.next_tick = self
            .next_tick
            .checked_add(1)
            .ok_or(SchedulingError::ArithmeticOverflow)?;
        Ok(deadline)
    }

    /// Skips every unissued deadline at or before `now_ns`.
    ///
    /// The returned deadline is strictly after `now_ns`. This is called after
    /// an attempt completes; a target equal to completion time is therefore a
    /// missed tick rather than an overlapping immediate retry.
    pub fn advance_after(&mut self, now_ns: i64) -> Result<CadenceAdvance, SchedulingError> {
        let first = self.next_deadline()?;
        if first.scheduled_ns > now_ns {
            return Ok(CadenceAdvance {
                next: first,
                missed: None,
            });
        }

        let elapsed = i128::from(now_ns) - i128::from(self.anchor_ns);
        let last_due = elapsed
            .checked_div(i128::from(self.interval_ns))
            .ok_or(SchedulingError::ArithmeticOverflow)?;
        let last_tick = u64::try_from(last_due).map_err(|_| SchedulingError::ArithmeticOverflow)?;
        if last_tick < self.next_tick {
            return Err(SchedulingError::ArithmeticOverflow);
        }
        let count = last_tick
            .checked_sub(self.next_tick)
            .and_then(|span| span.checked_add(1))
            .ok_or(SchedulingError::ArithmeticOverflow)?;
        let last = self.deadline_for(last_tick)?;
        self.next_tick = last_tick
            .checked_add(1)
            .ok_or(SchedulingError::ArithmeticOverflow)?;
        let next = self.next_deadline()?;
        Ok(CadenceAdvance {
            next,
            missed: Some(MissedCadenceRange {
                first_tick: first.tick,
                last_tick,
                first_deadline_ns: first.scheduled_ns,
                last_deadline_ns: last.scheduled_ns,
                count,
            }),
        })
    }
}

/// Effective absolute deadline for one control-plane call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AbsoluteCallDeadline {
    deadline_ns: i64,
}

impl AbsoluteCallDeadline {
    /// Takes the minimum of the request timeout and all active lifecycle caps.
    pub fn derive(
        request_start_ns: i64,
        request_timeout_ns: i64,
        boundary_deadline_ns: Option<i64>,
        run_deadline_ns: Option<i64>,
        shutdown_deadline_ns: Option<i64>,
    ) -> Result<Self, SchedulingError> {
        if request_timeout_ns <= 0 {
            return Err(SchedulingError::NonPositiveRequestTimeout(
                request_timeout_ns,
            ));
        }
        let mut deadline_ns = request_start_ns
            .checked_add(request_timeout_ns)
            .ok_or(SchedulingError::ArithmeticOverflow)?;
        for cap in [boundary_deadline_ns, run_deadline_ns, shutdown_deadline_ns]
            .into_iter()
            .flatten()
        {
            deadline_ns = deadline_ns.min(cap);
        }
        Ok(Self { deadline_ns })
    }

    /// Returns the authoritative absolute Clock deadline.
    #[must_use]
    pub const fn get(self) -> i64 {
        self.deadline_ns
    }

    /// Whether network IO must not begin or must already be cancelled.
    #[must_use]
    pub const fn is_expired_at(self, now_ns: i64) -> bool {
        now_ns >= self.deadline_ns
    }

    /// Lowers an active deadline without ever extending it.
    pub fn lower_to(&mut self, earlier_deadline_ns: i64) -> bool {
        let lowered = earlier_deadline_ns < self.deadline_ns;
        self.deadline_ns = self.deadline_ns.min(earlier_deadline_ns);
        lowered
    }
}

/// Stable reason one physical source attempt was issued.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceAttemptKind {
    /// A cadence-driven scrape.
    Continuous(CadenceDeadline),
    /// A forced source-local transition snapshot.
    Boundary {
        /// Stable transition identity from the sealed boundary plan.
        transition_id: String,
    },
}

/// Token proving that exactly one source attempt currently owns the gate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IssuedSourceAttempt {
    /// Monotone sequence assigned only when network work is actually issued.
    pub source_record_seq: u64,
    /// Cadence or boundary reason for issuance.
    pub kind: SourceAttemptKind,
    /// Current effective absolute deadline; shutdown may only lower it.
    pub deadline: AbsoluteCallDeadline,
}

/// Single-source state machine preventing overlapping control-plane requests.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SourceAttemptGate {
    next_source_record_seq: u64,
    active: Option<IssuedSourceAttempt>,
    issuance_closed: bool,
}

impl SourceAttemptGate {
    /// Creates an open gate whose first issued sequence is zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            next_source_record_seq: 0,
            active: None,
            issuance_closed: false,
        }
    }

    /// Issues one attempt or rejects overlap/stopped issuance.
    pub fn begin(
        &mut self,
        kind: SourceAttemptKind,
        deadline: AbsoluteCallDeadline,
    ) -> Result<IssuedSourceAttempt, SchedulingError> {
        if self.issuance_closed {
            return Err(SchedulingError::IssuanceClosed);
        }
        if self.active.is_some() {
            return Err(SchedulingError::AttemptAlreadyInFlight);
        }
        let source_record_seq = self.next_source_record_seq;
        self.next_source_record_seq = self
            .next_source_record_seq
            .checked_add(1)
            .ok_or(SchedulingError::ArithmeticOverflow)?;
        let issued = IssuedSourceAttempt {
            source_record_seq,
            kind,
            deadline,
        };
        self.active = Some(issued.clone());
        Ok(issued)
    }

    /// Completes exactly the currently active attempt.
    pub fn complete(
        &mut self,
        source_record_seq: u64,
    ) -> Result<IssuedSourceAttempt, SchedulingError> {
        let active = self
            .active
            .take()
            .ok_or(SchedulingError::NoAttemptInFlight)?;
        if active.source_record_seq != source_record_seq {
            let expected = active.source_record_seq;
            self.active = Some(active);
            return Err(SchedulingError::AttemptSequenceMismatch {
                expected,
                actual: source_record_seq,
            });
        }
        Ok(active)
    }

    /// Closes new issuance and lowers the active attempt's deadline.
    ///
    /// Repeated stop requests are idempotent except that an earlier deadline
    /// continues to tighten the active cancellation bound.
    pub fn stop(&mut self, shutdown_deadline_ns: i64) -> Option<AbsoluteCallDeadline> {
        self.issuance_closed = true;
        self.active.as_mut().map(|active| {
            active.deadline.lower_to(shutdown_deadline_ns);
            active.deadline
        })
    }

    /// Returns the active attempt, when one exists.
    #[must_use]
    pub fn active(&self) -> Option<&IssuedSourceAttempt> {
        self.active.as_ref()
    }

    /// Whether shutdown has closed all future issuance.
    #[must_use]
    pub const fn issuance_closed(&self) -> bool {
        self.issuance_closed
    }
}

/// Invalid scheduling input or illegal source-driver transition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchedulingError {
    /// Cadence intervals must be positive.
    NonPositiveInterval(i64),
    /// Per-source request timeouts must be positive.
    NonPositiveRequestTimeout(i64),
    /// Checked tick/deadline/sequence arithmetic overflowed.
    ArithmeticOverflow,
    /// A source attempted to launch a second overlapping request.
    AttemptAlreadyInFlight,
    /// A completion arrived without an active request.
    NoAttemptInFlight,
    /// A completion referred to a different active source sequence.
    AttemptSequenceMismatch {
        /// Active sequence retained by the gate.
        expected: u64,
        /// Sequence supplied by the completion.
        actual: u64,
    },
    /// Shutdown has permanently closed new source issuance.
    IssuanceClosed,
}

impl Display for SchedulingError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveInterval(value) => {
                write!(
                    formatter,
                    "telemetry cadence interval must be positive, got {value}"
                )
            }
            Self::NonPositiveRequestTimeout(value) => write!(
                formatter,
                "telemetry request timeout must be positive, got {value}"
            ),
            Self::ArithmeticOverflow => {
                formatter.write_str("telemetry scheduling arithmetic overflowed")
            }
            Self::AttemptAlreadyInFlight => {
                formatter.write_str("telemetry source already has an attempt in flight")
            }
            Self::NoAttemptInFlight => {
                formatter.write_str("telemetry source has no attempt in flight")
            }
            Self::AttemptSequenceMismatch { expected, actual } => write!(
                formatter,
                "telemetry completion sequence {actual} does not match active sequence {expected}"
            ),
            Self::IssuanceClosed => formatter.write_str("telemetry source issuance is closed"),
        }
    }
}

impl std::error::Error for SchedulingError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_duration_attempts_keep_exact_anchor_relative_deadlines() {
        let mut cadence = FixedDeadlineCadence::new(1_000, 10).unwrap();
        for expected_tick in 0_u64..5 {
            let issued = cadence.issue_next().unwrap();
            assert_eq!(issued.tick, expected_tick);
            assert_eq!(
                issued.scheduled_ns,
                1_000 + i64::try_from(expected_tick).unwrap() * 10
            );
            let advance = cadence.advance_after(issued.scheduled_ns).unwrap();
            assert_eq!(advance.missed, None);
            assert_eq!(advance.next.tick, expected_tick + 1);
        }
    }

    #[test]
    fn overrun_compacts_debt_without_drift_or_catch_up() {
        let mut cadence = FixedDeadlineCadence::new(0, 10).unwrap();
        assert_eq!(cadence.issue_next().unwrap().tick, 0);

        let advance = cadence.advance_after(35).unwrap();

        assert_eq!(
            advance.missed,
            Some(MissedCadenceRange {
                first_tick: 1,
                last_tick: 3,
                first_deadline_ns: 10,
                last_deadline_ns: 30,
                count: 3,
            })
        );
        assert_eq!(
            advance.next,
            CadenceDeadline {
                tick: 4,
                scheduled_ns: 40
            }
        );
        assert_eq!(cadence.issue_next().unwrap(), advance.next);
    }

    #[test]
    fn completion_exactly_on_a_future_target_marks_that_tick_missed() {
        let mut cadence = FixedDeadlineCadence::new(0, 10).unwrap();
        cadence.issue_next().unwrap();
        let advance = cadence.advance_after(10).unwrap();
        assert_eq!(advance.missed.unwrap().first_tick, 1);
        assert_eq!(advance.next.scheduled_ns, 20);
    }

    #[test]
    fn independent_sources_cannot_shift_each_others_anchors() {
        let mut slow = FixedDeadlineCadence::new(100, 10).unwrap();
        let mut fast = FixedDeadlineCadence::new(100, 10).unwrap();
        slow.issue_next().unwrap();
        fast.issue_next().unwrap();

        assert_eq!(slow.advance_after(137).unwrap().next.scheduled_ns, 140);
        assert_eq!(fast.advance_after(100).unwrap().next.scheduled_ns, 110);
    }

    #[test]
    fn deadline_uses_all_caps_and_shutdown_can_only_lower_it() {
        let mut deadline =
            AbsoluteCallDeadline::derive(100, 50, Some(140), Some(145), None).unwrap();
        assert_eq!(deadline.get(), 140);
        assert!(!deadline.lower_to(145));
        assert_eq!(deadline.get(), 140);
        assert!(deadline.lower_to(120));
        assert_eq!(deadline.get(), 120);
        assert!(deadline.is_expired_at(120));
    }

    #[test]
    fn single_flight_gate_rejects_overlap_and_late_completion() {
        let deadline = AbsoluteCallDeadline::derive(0, 100, None, None, None).unwrap();
        let mut gate = SourceAttemptGate::new();
        let issued = gate
            .begin(
                SourceAttemptKind::Continuous(CadenceDeadline {
                    tick: 0,
                    scheduled_ns: 0,
                }),
                deadline,
            )
            .unwrap();
        assert_eq!(
            gate.begin(
                SourceAttemptKind::Boundary {
                    transition_id: "t1".to_owned(),
                },
                deadline,
            ),
            Err(SchedulingError::AttemptAlreadyInFlight)
        );
        assert!(matches!(
            gate.complete(issued.source_record_seq + 1),
            Err(SchedulingError::AttemptSequenceMismatch { .. })
        ));
        assert_eq!(gate.complete(issued.source_record_seq).unwrap(), issued);
        assert_eq!(
            gate.complete(issued.source_record_seq),
            Err(SchedulingError::NoAttemptInFlight)
        );
    }

    #[test]
    fn stop_closes_issuance_and_tightens_the_active_deadline() {
        let deadline = AbsoluteCallDeadline::derive(0, 100, None, None, None).unwrap();
        let mut gate = SourceAttemptGate::new();
        let issued = gate
            .begin(
                SourceAttemptKind::Boundary {
                    transition_id: "t1".to_owned(),
                },
                deadline,
            )
            .unwrap();
        assert_eq!(gate.stop(25).unwrap().get(), 25);
        assert_eq!(gate.stop(50).unwrap().get(), 25);
        gate.complete(issued.source_record_seq).unwrap();
        assert_eq!(
            gate.begin(
                SourceAttemptKind::Boundary {
                    transition_id: "t2".to_owned(),
                },
                deadline,
            ),
            Err(SchedulingError::IssuanceClosed)
        );
    }

    #[test]
    fn checked_arithmetic_fails_closed() {
        assert!(matches!(
            FixedDeadlineCadence::new(0, 0),
            Err(SchedulingError::NonPositiveInterval(0))
        ));
        let cadence = FixedDeadlineCadence::new(i64::MAX, 1).unwrap();
        assert_eq!(
            cadence.deadline_for(1),
            Err(SchedulingError::ArithmeticOverflow)
        );
        assert!(matches!(
            AbsoluteCallDeadline::derive(i64::MAX, 1, None, None, None),
            Err(SchedulingError::ArithmeticOverflow)
        ));
    }
}
