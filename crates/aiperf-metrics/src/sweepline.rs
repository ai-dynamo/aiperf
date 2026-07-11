// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sweep-line primitives for future time-weighted metric curves.

/// One weighted interval sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntervalSample {
    /// Inclusive start timestamp in nanoseconds.
    pub start_ns: i64,
    /// Exclusive end timestamp in nanoseconds.
    pub end_ns: i64,
    /// Sample value over the interval.
    pub value: f64,
}

/// Computes a time-weighted mean over interval samples.
pub fn time_weighted_mean(samples: &[IntervalSample]) -> Option<f64> {
    let mut weighted = 0.0;
    let mut duration = 0_i64;
    for sample in samples {
        if !sample.value.is_finite() || sample.end_ns <= sample.start_ns {
            continue;
        }
        let span = sample.end_ns - sample.start_ns;
        weighted += sample.value * span as f64;
        duration += span;
    }
    (duration > 0).then_some(weighted / duration as f64)
}

#[cfg(test)]
mod tests {
    use super::{IntervalSample, time_weighted_mean};

    #[test]
    fn time_weighted_mean_uses_interval_widths() {
        let samples = [
            IntervalSample {
                start_ns: 0,
                end_ns: 10,
                value: 1.0,
            },
            IntervalSample {
                start_ns: 10,
                end_ns: 30,
                value: 4.0,
            },
        ];
        assert_eq!(time_weighted_mean(&samples), Some(3.0));
    }
}
