// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Phase-boundary counter arithmetic shared by telemetry producers.
//!
//! AIPerf owns phase transitions, so cumulative GPU and server counters are
//! sampled synchronously at the start and end barriers and reset-clamped.

/// A validated phase-boundary counter delta.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CounterDelta {
    /// Counter value captured at the phase-start barrier.
    pub baseline: f64,
    /// Counter value captured at the phase-end barrier.
    pub final_value: f64,
    /// Non-negative phase delta after reset clamping.
    pub delta: f64,
}

/// Computes `max(final_value - baseline, 0)` for two finite snapshots.
///
/// A lower final value indicates that the producer restarted during the phase;
/// clamping prevents negative energy, throughput, or cache-hit results. Missing
/// and non-finite boundary values do not produce a delta.
pub fn boundary_counter_delta(
    baseline: Option<f64>,
    final_value: Option<f64>,
) -> Option<CounterDelta> {
    let baseline = baseline.filter(|value| value.is_finite())?;
    let final_value = final_value.filter(|value| value.is_finite())?;
    Some(CounterDelta {
        baseline,
        final_value,
        delta: (final_value - baseline).max(0.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_delta_preserves_growth_and_clamps_resets() {
        assert_eq!(
            boundary_counter_delta(Some(10.0), Some(13.5)),
            Some(CounterDelta {
                baseline: 10.0,
                final_value: 13.5,
                delta: 3.5,
            })
        );
        assert_eq!(
            boundary_counter_delta(Some(100.0), Some(4.0)).map(|result| result.delta),
            Some(0.0)
        );
    }

    #[test]
    fn boundary_delta_rejects_missing_or_non_finite_snapshots() {
        assert_eq!(boundary_counter_delta(None, Some(1.0)), None);
        assert_eq!(boundary_counter_delta(Some(1.0), None), None);
        assert_eq!(boundary_counter_delta(Some(f64::NAN), Some(1.0)), None);
        assert_eq!(boundary_counter_delta(Some(1.0), Some(f64::INFINITY)), None);
    }
}
