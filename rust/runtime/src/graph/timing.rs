// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared Graph-IR timing conversion bounds.

/// Why a graph timing value cannot reach the scheduler's signed-nanosecond clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimingRangeError {
    NonFinite,
    OutsideI64Nanoseconds,
}

const I64_NS_MIN: f64 = -9_223_372_036_854_775_808.0;
const I64_NS_MAX_EXCLUSIVE: f64 = 9_223_372_036_854_775_808.0;

fn validate_scaled(value: f64, nanoseconds_per_unit: f64) -> Result<(), TimingRangeError> {
    if !value.is_finite() {
        return Err(TimingRangeError::NonFinite);
    }
    let nanoseconds = value * nanoseconds_per_unit;
    if !nanoseconds.is_finite() || !(I64_NS_MIN..I64_NS_MAX_EXCLUSIVE).contains(&nanoseconds) {
        return Err(TimingRangeError::OutsideI64Nanoseconds);
    }
    Ok(())
}

pub(crate) fn validate_seconds(value: f64) -> Result<(), TimingRangeError> {
    validate_scaled(value, 1_000_000_000.0)
}

pub(crate) fn validate_milliseconds(value: f64) -> Result<(), TimingRangeError> {
    validate_scaled(value, 1_000_000.0)
}

pub(crate) fn validate_microseconds(value: f64) -> Result<(), TimingRangeError> {
    validate_scaled(value, 1_000.0)
}

pub(crate) fn milliseconds_to_microseconds(value: f64) -> Result<f64, TimingRangeError> {
    validate_milliseconds(value)?;
    let microseconds = value * 1_000.0;
    validate_microseconds(microseconds)?;
    Ok(microseconds)
}

pub(crate) fn checked_add_microseconds(left: f64, right: f64) -> Result<f64, TimingRangeError> {
    let total = left + right;
    validate_microseconds(total)?;
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_range_graph_timing_uses_signed_i64_nanosecond_domain() {
        assert_eq!(
            validate_microseconds(f64::NAN),
            Err(TimingRangeError::NonFinite)
        );
        assert_eq!(
            validate_microseconds(1.0e308),
            Err(TimingRangeError::OutsideI64Nanoseconds)
        );
        assert!(validate_microseconds(9.0e15).is_ok());
        assert!(validate_microseconds(-9.0e15).is_ok());
        assert_eq!(
            checked_add_microseconds(5.0e15, 5.0e15),
            Err(TimingRangeError::OutsideI64Nanoseconds)
        );
    }
}
