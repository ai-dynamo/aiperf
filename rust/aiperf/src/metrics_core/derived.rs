// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared derived-metric helpers.

/// Returns `end - start` when both values are finite and ordered.
pub fn delta_ms(start_ms: f64, end_ms: f64) -> Option<f64> {
    (start_ms.is_finite() && end_ms.is_finite() && end_ms >= start_ms).then_some(end_ms - start_ms)
}

/// Returns a latency value adjusted for errored records.
pub fn error_adjusted_result(value: Option<f64>, errored: bool) -> Option<f64> {
    if errored {
        Some(f64::INFINITY)
    } else {
        value.filter(|v| v.is_finite())
    }
}

/// Subtracts network RTT from a latency while saturating at zero.
pub fn network_adjusted_ms(latency_ms: f64, mean_rtt_ms: f64) -> Option<f64> {
    (latency_ms.is_finite() && mean_rtt_ms.is_finite())
        .then_some((latency_ms - mean_rtt_ms).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::{delta_ms, network_adjusted_ms};

    #[test]
    fn derived_helpers_reject_invalid_values() {
        assert_eq!(delta_ms(1.0, 3.5), Some(2.5));
        assert_eq!(delta_ms(3.5, 1.0), None);
        assert_eq!(network_adjusted_ms(10.0, 12.0), Some(0.0));
    }
}
