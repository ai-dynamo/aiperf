// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed-loop steady-state measurement window for concurrency-target runs.
//!
//! A concurrency-target run has an unavoidable ramp-up (the load generator fills
//! its in-flight slots) and a drain (the last in-flight requests complete after
//! new admission stops). Summarizing the whole run blends those transients into
//! the steady interval and biases throughput low and tail latency high.
//!
//! This module detects the steady-state window automatically: the interval
//! during which in-flight concurrency is at or above a configured fraction of
//! the target. The window opens at the first up-crossing of that threshold and
//! closes at the last down-crossing, so ramp-up and drain are excluded. The
//! steady summary is then computed by the ordinary [`MetricsAccumulator`]
//! machinery over a half-open `[start, end)` time range, reusing the exact same
//! record timestamps and summary computation as the whole-run summary — no
//! separate metrics engine is introduced.
//!
//! The capability is gated: it only produces an outcome when the feature is
//! enabled *and* a positive concurrency target is set. Otherwise callers see
//! `None` and behavior is unchanged.

use crate::metrics_core::accumulator::{AccumulatorSummary, MetricsAccumulator};
use crate::metrics_core::window::ExportContext;

/// Wall-clock floor for the steady window before a short-window warning fires.
const MIN_STEADY_WINDOW_NS: i64 = 10_000_000_000; // 10 seconds
/// Fraction of total run duration below which the steady window is warned short.
const MIN_STEADY_WINDOW_RUN_FRACTION: f64 = 0.10;
/// Default steady-state occupancy fraction of the concurrency target.
pub const DEFAULT_STEADY_STATE_FRACTION: f64 = 0.8;

/// Configuration for closed-loop steady-state windowing.
///
/// Co-located with the metrics-plane configuration ([`MetricsConfig`]) because
/// the window is computed from the same accumulated records as every other
/// summary. The concurrency *target* itself is owned by the workload/phase
/// configuration and is passed in at summary time.
///
/// [`MetricsConfig`]: crate::metrics_core::accumulator::MetricsConfig
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SteadyStateConfig {
    /// Enables steady-state detection and summarization for concurrency runs.
    pub enabled: bool,
    /// Occupancy fraction of the concurrency target that defines "steady".
    ///
    /// The detection threshold is `ceil(fraction * target_concurrency)`.
    /// Clamped into `(0, 1]` at use; the default is
    /// [`DEFAULT_STEADY_STATE_FRACTION`].
    pub fraction: f64,
}

impl Default for SteadyStateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            fraction: DEFAULT_STEADY_STATE_FRACTION,
        }
    }
}

impl SteadyStateConfig {
    /// Returns the sanitized occupancy fraction clamped into `(0, 1]`.
    fn effective_fraction(self) -> f64 {
        if self.fraction.is_finite() && self.fraction > 0.0 && self.fraction <= 1.0 {
            self.fraction
        } else {
            DEFAULT_STEADY_STATE_FRACTION
        }
    }
}

/// The detected steady-state interval and the concurrency profile that bounds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SteadyWindow {
    /// Inclusive lower bound in nanoseconds (first up-crossing of the threshold).
    pub start_ns: i64,
    /// Exclusive upper bound in nanoseconds (last down-crossing of the threshold).
    pub end_ns: i64,
    /// Concurrency threshold, `ceil(fraction * target_concurrency)`.
    pub threshold: usize,
    /// Peak in-flight concurrency observed over the run.
    pub peak_concurrency: usize,
}

impl SteadyWindow {
    /// Returns the window duration in nanoseconds (never negative).
    pub fn duration_ns(self) -> i64 {
        (self.end_ns - self.start_ns).max(0)
    }
}

/// The steady-state window, its summary, and the short-window warning state.
#[derive(Debug, Clone, PartialEq)]
pub struct SteadyStateOutcome {
    /// The detected steady window.
    pub window: SteadyWindow,
    /// Summary computed over records started within the half-open window.
    pub summary: AccumulatorSummary,
    /// Full run span start in nanoseconds (earliest record start).
    pub run_start_ns: i64,
    /// Full run span end in nanoseconds (latest record end).
    pub run_end_ns: i64,
    /// True when the window is shorter than `max(10s, 10% of run duration)`.
    pub short_window: bool,
}

impl SteadyStateOutcome {
    /// Returns a human-readable short-window warning when one applies.
    pub fn warning(&self) -> Option<String> {
        if !self.short_window {
            return None;
        }
        let run_ns = (self.run_end_ns - self.run_start_ns).max(0);
        Some(format!(
            "steady-state window is {:.2}s of a {:.2}s run \
             (below the max(10s, 10% of run) floor); \
             steady-state summary may be unreliable — \
             increase run duration or concurrency",
            self.window.duration_ns() as f64 / 1e9,
            run_ns as f64 / 1e9,
        ))
    }
}

/// Detects the steady-state window from per-record `[start, end)` intervals.
///
/// A two-pointer merge walks the sorted start and end events in timestamp order,
/// maintaining running in-flight concurrency. At tied timestamps starts are
/// processed before ends so a request that ends exactly as another starts does
/// not create a spurious dip. The window opens at the first start event that
/// raises concurrency to the threshold (up-crossing) and closes at the last end
/// event that lowers concurrency from the threshold back below it
/// (down-crossing). Returns `None` when no interval is supplied, the target is
/// zero, or concurrency never reaches the threshold.
pub fn detect_steady_window(
    intervals: &[(i64, i64)],
    target_concurrency: usize,
    fraction: f64,
) -> Option<SteadyWindow> {
    if intervals.is_empty() || target_concurrency == 0 {
        return None;
    }
    let fraction = if fraction.is_finite() && fraction > 0.0 && fraction <= 1.0 {
        fraction
    } else {
        DEFAULT_STEADY_STATE_FRACTION
    };
    // ceil(fraction * target) without float rounding surprises; at least 1.
    let threshold = ((fraction * target_concurrency as f64).ceil() as usize).max(1);

    let mut starts: Vec<i64> = intervals.iter().map(|&(start, _)| start).collect();
    // Clamp any inverted interval so an end never precedes its own start.
    let mut ends: Vec<i64> = intervals
        .iter()
        .map(|&(start, end)| end.max(start))
        .collect();
    starts.sort_unstable();
    ends.sort_unstable();

    let mut active: usize = 0;
    let mut peak: usize = 0;
    let mut window_start: Option<i64> = None;
    let mut window_end: Option<i64> = None;
    let (mut i, mut j) = (0usize, 0usize);

    while i < starts.len() || j < ends.len() {
        // Start-before-end on ties keeps simultaneous hand-offs from dipping.
        let take_start = match (starts.get(i), ends.get(j)) {
            (Some(&s), Some(&e)) => s <= e,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };

        if take_start {
            active += 1;
            peak = peak.max(active);
            // Up-crossing: first time we reach the threshold from below.
            if active == threshold && window_start.is_none() {
                window_start = Some(starts[i]);
            }
            i += 1;
        } else {
            let before = active;
            active = active.saturating_sub(1);
            // Down-crossing: dropping from exactly the threshold necessarily
            // lands below it. Keep the latest so the window closes at the *last*
            // sustained descent.
            if before == threshold && window_start.is_some() {
                window_end = Some(ends[j]);
            }
            j += 1;
        }
    }

    let start_ns = window_start?;
    // If concurrency never fell back below threshold after opening (e.g. the run
    // ended while still saturated), close the window at the last observed event.
    let end_ns = window_end.unwrap_or_else(|| {
        starts
            .last()
            .copied()
            .into_iter()
            .chain(ends.last().copied())
            .max()
            .unwrap_or(start_ns)
    });
    Some(SteadyWindow {
        start_ns,
        end_ns: end_ns.max(start_ns),
        threshold,
        peak_concurrency: peak,
    })
}

/// Computes the steady-state outcome for a populated accumulator.
///
/// Returns `None` when the feature is disabled, the target is not positive, the
/// accumulator holds no timestamped records, or concurrency never reaches the
/// threshold. When it returns `Some`, the [`SteadyStateOutcome::summary`] is
/// produced by the ordinary accumulator export over the detected half-open
/// window, so it carries every catalog metric (throughput, TTFT, ITL/TPOT, …)
/// with steady-state attribution.
pub fn steady_state_summary(
    accumulator: &MetricsAccumulator,
    config: &SteadyStateConfig,
    target_concurrency: usize,
) -> Option<SteadyStateOutcome> {
    if !config.enabled || target_concurrency == 0 {
        return None;
    }

    let store = accumulator.column_store();
    let starts = store.start_ns();
    let ends = store.end_ns();
    // Absent rows carry NaN; keep only fully timestamped, finite intervals.
    let intervals: Vec<(i64, i64)> = starts
        .iter()
        .zip(ends.iter())
        .filter(|(s, e)| s.is_finite() && e.is_finite())
        .map(|(&s, &e)| (s as i64, e as i64))
        .collect();
    if intervals.is_empty() {
        return None;
    }

    let window = detect_steady_window(&intervals, target_concurrency, config.effective_fraction())?;

    let run_start_ns = intervals.iter().map(|&(s, _)| s).min().unwrap_or(0);
    let run_end_ns = intervals.iter().map(|&(_, e)| e).max().unwrap_or(0);

    // Reuse the shared summary path over the detected half-open time range.
    let summary =
        accumulator.export_results(&ExportContext::time_range(window.start_ns, window.end_ns));

    let run_duration_ns = (run_end_ns - run_start_ns).max(0);
    let floor_ns =
        MIN_STEADY_WINDOW_NS.max((run_duration_ns as f64 * MIN_STEADY_WINDOW_RUN_FRACTION) as i64);
    let short_window = window.duration_ns() < floor_ns;

    let outcome = SteadyStateOutcome {
        window,
        summary,
        run_start_ns,
        run_end_ns,
        short_window,
    };
    if outcome.short_window {
        // Structured fields avoid allocating a message string on the warn path.
        tracing::warn!(
            target: "aiperf::metrics::steady_state",
            window_s = outcome.window.duration_ns() as f64 / 1e9,
            run_s = (run_end_ns - run_start_ns).max(0) as f64 / 1e9,
            threshold_concurrency = outcome.window.threshold,
            "steady-state window is below the max(10s, 10% of run) floor; \
             steady-state summary may be unreliable"
        );
    }
    Some(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_is_ceil_of_fraction_times_target() {
        // Ten fully-overlapping requests saturate any threshold <= 10.
        let intervals: Vec<(i64, i64)> = (0..10).map(|k| (k, 1000)).collect();
        // target 10, fraction 0.8 -> ceil(8.0) = 8
        assert_eq!(
            detect_steady_window(&intervals, 10, 0.8).unwrap().threshold,
            8
        );
        // target 10, fraction 0.75 -> ceil(7.5) = 8
        assert_eq!(
            detect_steady_window(&intervals, 10, 0.75)
                .unwrap()
                .threshold,
            8
        );
    }

    #[test]
    fn window_opens_at_up_crossing_and_closes_at_last_down_crossing() {
        // Concurrency profile with target 4, threshold ceil(0.8*4)=4.
        // Ramp: starts at 0,1,2,3 (reaches 4 in-flight at t=3 -> up-crossing).
        // Steady: all four overlap until staggered drain.
        // Drain: ends at 20,21,22,23. Concurrency drops below 4 at the first
        // end (t=20) but a fresh start at t=19 re-saturates so the *last*
        // down-crossing governs.
        let intervals = vec![
            (0, 20),  // A
            (1, 21),  // B
            (2, 22),  // C
            (3, 30),  // D  (reaches 4 in-flight at its start -> up-cross at t=3)
            (19, 31), // E  (re-saturates after A leaves)
            (40, 45), // late straggler, below threshold, must be excluded
        ];
        let window = detect_steady_window(&intervals, 4, 0.8).unwrap();
        assert_eq!(window.threshold, 4);
        assert_eq!(window.start_ns, 3, "window must open at the up-crossing");
        // E's start at 19 lifts in-flight to 5, so A ending at 20 only drops it
        // to 4 (still saturated). The last descent from 4 to 3 is B ending at
        // 21; C(22) and the late straggler are past the window.
        assert_eq!(
            window.end_ns, 21,
            "window must close at the last threshold down-crossing"
        );
        assert_eq!(window.peak_concurrency, 5);
    }

    #[test]
    fn returns_none_when_threshold_never_reached() {
        // Only ever two concurrent, target 10 -> threshold 8, never met.
        let intervals = vec![(0, 5), (1, 6), (10, 15), (11, 16)];
        assert!(detect_steady_window(&intervals, 10, 0.8).is_none());
    }

    #[test]
    fn gated_off_returns_none() {
        assert!(detect_steady_window(&[], 10, 0.8).is_none());
        assert!(detect_steady_window(&[(0, 1)], 0, 0.8).is_none());
    }

    #[test]
    fn simultaneous_handoff_does_not_dip_below_threshold() {
        // target 2, threshold 2. A ends exactly when C starts; concurrency must
        // be treated as never dropping below 2 across the handoff.
        let intervals = vec![
            (0, 10),  // A
            (1, 20),  // B
            (10, 30), // C starts exactly as A ends
        ];
        let window = detect_steady_window(&intervals, 2, 1.0).unwrap();
        assert_eq!(window.threshold, 2);
        assert_eq!(window.start_ns, 1);
        // C's start at 10 coincides with A's end, so the handoff holds at 2
        // (start-before-end). The last descent from 2 to 1 is B ending at 20;
        // C then runs alone below threshold.
        assert_eq!(window.end_ns, 20);
    }
}
