// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Monotonic-to-Unix epoch anchors for cross-process placement.

use std::fmt::{self, Display, Formatter};
use std::time::{SystemTime, UNIX_EPOCH};

use aiperf_clock::Clock;

/// One bracketed monotonic/Unix epoch observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EpochAnchor {
    /// Midpoint of the monotonic bracket.
    pub clock_ns: i64,
    /// One wall-time observation in signed Unix nanoseconds.
    pub unix_epoch_ns: i128,
    /// Acquisition uncertainty, not oscillator-drift uncertainty.
    pub capture_uncertainty_ns: u64,
}

impl EpochAnchor {
    /// Derives approximate Unix placement from one later monotonic timestamp.
    pub fn unix_ns_at(self, clock_ns: i64) -> Result<i128, EpochAnchorError> {
        self.unix_epoch_ns
            .checked_add(i128::from(clock_ns) - i128::from(self.clock_ns))
            .ok_or(EpochAnchorError::ArithmeticOverflow)
    }
}

/// An injectable provider for the one allowed wall-time read bracket.
pub trait EpochAnchorProvider {
    /// Captures an anchor using the injected monotonic clock.
    fn anchor(&self, clock: &dyn Clock) -> Result<EpochAnchor, EpochAnchorError>;
}

/// System wall-time provider with an authored clock-resolution allowance.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemEpochAnchorProvider {
    resolution_allowance_ns: u64,
}

impl SystemEpochAnchorProvider {
    /// Constructs a provider whose uncertainty includes the given allowance.
    #[must_use]
    pub const fn new(resolution_allowance_ns: u64) -> Self {
        Self {
            resolution_allowance_ns,
        }
    }
}

impl EpochAnchorProvider for SystemEpochAnchorProvider {
    fn anchor(&self, clock: &dyn Clock) -> Result<EpochAnchor, EpochAnchorError> {
        let before = clock.now_ns();
        let unix_epoch_ns = system_unix_ns()?;
        let after = clock.now_ns();
        anchor_from_observations(before, unix_epoch_ns, after, self.resolution_allowance_ns)
    }
}

/// A reversed bracket, unavailable wall clock, or arithmetic overflow.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EpochAnchorError {
    /// The monotonic clock moved backwards across the wall read.
    ReversedBracket {
        /// First monotonic observation.
        before_ns: i64,
        /// Second monotonic observation.
        after_ns: i64,
    },
    /// Wall time cannot be represented as signed nanoseconds.
    WallTimeOutOfRange,
    /// Checked midpoint, uncertainty, or placement arithmetic overflowed.
    ArithmeticOverflow,
}

impl Display for EpochAnchorError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReversedBracket {
                before_ns,
                after_ns,
            } => write!(
                formatter,
                "epoch anchor monotonic bracket reversed: {before_ns} then {after_ns}"
            ),
            Self::WallTimeOutOfRange => {
                formatter.write_str("system wall time is outside signed nanosecond range")
            }
            Self::ArithmeticOverflow => formatter.write_str("epoch anchor arithmetic overflowed"),
        }
    }
}

impl std::error::Error for EpochAnchorError {}

fn anchor_from_observations(
    before_ns: i64,
    unix_epoch_ns: i128,
    after_ns: i64,
    resolution_allowance_ns: u64,
) -> Result<EpochAnchor, EpochAnchorError> {
    if after_ns < before_ns {
        return Err(EpochAnchorError::ReversedBracket {
            before_ns,
            after_ns,
        });
    }
    let span =
        u64::try_from(after_ns - before_ns).map_err(|_| EpochAnchorError::ArithmeticOverflow)?;
    let midpoint_delta =
        i64::try_from(span / 2).map_err(|_| EpochAnchorError::ArithmeticOverflow)?;
    let clock_ns = before_ns
        .checked_add(midpoint_delta)
        .ok_or(EpochAnchorError::ArithmeticOverflow)?;
    let half_span_rounded_up = span / 2 + span % 2;
    let capture_uncertainty_ns = half_span_rounded_up
        .checked_add(resolution_allowance_ns)
        .ok_or(EpochAnchorError::ArithmeticOverflow)?;
    Ok(EpochAnchor {
        clock_ns,
        unix_epoch_ns,
        capture_uncertainty_ns,
    })
}

fn system_unix_ns() -> Result<i128, EpochAnchorError> {
    let now = SystemTime::now();
    match now.duration_since(UNIX_EPOCH) {
        Ok(duration) => {
            i128::try_from(duration.as_nanos()).map_err(|_| EpochAnchorError::WallTimeOutOfRange)
        }
        Err(error) => i128::try_from(error.duration().as_nanos())
            .map(|duration| -duration)
            .map_err(|_| EpochAnchorError::WallTimeOutOfRange),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bracket_midpoint_and_uncertainty_are_checked() {
        assert_eq!(
            anchor_from_observations(10, 1_000, 15, 3).unwrap(),
            EpochAnchor {
                clock_ns: 12,
                unix_epoch_ns: 1_000,
                capture_uncertainty_ns: 6,
            }
        );
        assert!(matches!(
            anchor_from_observations(10, 1_000, 9, 0),
            Err(EpochAnchorError::ReversedBracket { .. })
        ));
    }

    #[test]
    fn later_placement_uses_only_monotonic_delta() {
        let anchor = EpochAnchor {
            clock_ns: 100,
            unix_epoch_ns: 1_000,
            capture_uncertainty_ns: 2,
        };
        assert_eq!(anchor.unix_ns_at(125), Ok(1_025));
        assert_eq!(anchor.unix_ns_at(75), Ok(975));
    }
}
