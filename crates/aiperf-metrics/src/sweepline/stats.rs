// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Duration-weighted statistics over sweep-line step functions.
//!
//! Clipping, active masks, and duration-CDF percentiles port
//! `src/aiperf/analysis/sweepline_stats.py:20-188`.

use super::{StepFn, lower_bound, upper_bound};

/// Time-weighted statistics for one step function.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SweepLineStats {
    /// Duration-weighted average.
    pub avg: f64,
    /// Minimum step value in the window.
    pub min: f64,
    /// Maximum step value in the window.
    pub max: f64,
    /// Duration-weighted median.
    pub p50: f64,
    /// Duration-weighted p90.
    pub p90: f64,
    /// Duration-weighted p95.
    pub p95: f64,
    /// Duration-weighted p99.
    pub p99: f64,
    /// Duration-weighted population standard deviation.
    pub std: f64,
}

impl SweepLineStats {
    /// All-zero result returned for an empty curve or invalid window.
    pub const ZERO: Self = Self {
        avg: 0.0,
        min: 0.0,
        max: 0.0,
        p50: 0.0,
        p90: 0.0,
        p95: 0.0,
        p99: 0.0,
        std: 0.0,
    };
}

/// One clipped constant-value segment of a step function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClippedSegment {
    /// Positive segment duration in nanoseconds.
    pub duration_ns: f64,
    /// Value held over the segment.
    pub value: f64,
}

/// Clips a step function to `[window_start_ns, window_end_ns]`.
///
/// The predecessor value is carried across the left boundary and zero is used
/// before the first event. This ports
/// `src/aiperf/analysis/sweepline_stats.py:20-52`.
pub fn build_clipped_segments(
    curve: &StepFn,
    window_start_ns: f64,
    window_end_ns: f64,
) -> Vec<ClippedSegment> {
    if curve.is_empty() {
        return Vec::new();
    }
    let timestamps = curve.timestamps_ns();
    let values = curve.values();
    let lo = upper_bound(timestamps, window_start_ns).saturating_sub(1);
    let hi = (lower_bound(timestamps, window_end_ns) + 1).min(timestamps.len());

    let mut starts = Vec::with_capacity(hi.saturating_sub(lo) + 1);
    let mut segment_values = Vec::with_capacity(hi.saturating_sub(lo) + 1);
    starts.push(window_start_ns);
    segment_values.push(if lo > 0 { values[lo - 1] } else { 0.0 });
    starts.extend_from_slice(&timestamps[lo..hi]);
    segment_values.extend_from_slice(&values[lo..hi]);

    let mut segments = Vec::with_capacity(starts.len());
    for index in 0..starts.len() {
        let start = starts[index].max(window_start_ns);
        let end = if index + 1 < starts.len() {
            starts[index + 1]
        } else {
            window_end_ns
        }
        .min(window_end_ns);
        let duration_ns = (end - start).max(0.0);
        if duration_ns > 0.0 {
            segments.push(ClippedSegment {
                duration_ns,
                value: segment_values[index],
            });
        }
    }
    segments
}

/// Computes duration-weighted statistics over a clipped step function.
///
/// Average and population variance use the full window span, including idle
/// zero-valued segments. Percentiles sort by value and locate the first
/// cumulative-duration fraction at or above 50/90/95/99 percent. This ports
/// `src/aiperf/analysis/sweepline_stats.py:55-98`.
pub fn compute_time_weighted_stats(
    curve: &StepFn,
    window_start_ns: f64,
    window_end_ns: f64,
) -> SweepLineStats {
    let total_duration = window_end_ns - window_start_ns;
    if curve.is_empty() || total_duration <= 0.0 {
        return SweepLineStats::ZERO;
    }
    let segments = build_clipped_segments(curve, window_start_ns, window_end_ns);
    if segments.is_empty() {
        return SweepLineStats::ZERO;
    }
    weighted_stats(&segments, total_duration)
}

/// Computes duration-weighted rate statistics only while `mask` is positive.
///
/// Idle segments do not enter the average, variance, or percentile CDF. The
/// merged grid includes only in-window rate/mask events plus exact window edges,
/// matching `src/aiperf/analysis/sweepline_stats.py:101-188`.
pub fn compute_active_weighted_stats(
    rate: &StepFn,
    mask: &StepFn,
    window_start_ns: f64,
    window_end_ns: f64,
) -> SweepLineStats {
    if window_end_ns <= window_start_ns || rate.is_empty() || mask.is_empty() {
        return SweepLineStats::ZERO;
    }

    let mut grid = vec![window_start_ns, window_end_ns];
    append_events_inside(
        &mut grid,
        rate.timestamps_ns(),
        window_start_ns,
        window_end_ns,
    );
    append_events_inside(
        &mut grid,
        mask.timestamps_ns(),
        window_start_ns,
        window_end_ns,
    );
    grid.sort_by(f64::total_cmp);
    grid.dedup_by(|left, right| *left == *right || (left.is_nan() && right.is_nan()));
    if grid.len() < 2 {
        return SweepLineStats::ZERO;
    }

    let mut segments = Vec::with_capacity(grid.len() - 1);
    for pair in grid.windows(2) {
        let duration_ns = pair[1] - pair[0];
        if duration_ns > 0.0 && mask.value_at(pair[0]) > 0.0 {
            segments.push(ClippedSegment {
                duration_ns,
                value: rate.value_at(pair[0]),
            });
        }
    }
    let active_duration = segments
        .iter()
        .map(|segment| segment.duration_ns)
        .sum::<f64>();
    if active_duration <= 0.0 {
        return SweepLineStats::ZERO;
    }
    weighted_stats(&segments, active_duration)
}

fn append_events_inside(grid: &mut Vec<f64>, events: &[f64], start: f64, end: f64) {
    let lo = upper_bound(events, start);
    let hi = lower_bound(events, end);
    grid.extend_from_slice(&events[lo..hi]);
}

fn weighted_stats(segments: &[ClippedSegment], denominator_duration: f64) -> SweepLineStats {
    let avg = segments
        .iter()
        .map(|segment| segment.value * segment.duration_ns)
        .sum::<f64>()
        / denominator_duration;
    let min = segments
        .iter()
        .map(|segment| segment.value)
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    let max = segments
        .iter()
        .map(|segment| segment.value)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    let variance = segments
        .iter()
        .map(|segment| {
            let delta = segment.value - avg;
            segment.duration_ns * delta * delta
        })
        .sum::<f64>()
        / denominator_duration;

    let mut by_value = segments.to_vec();
    by_value.sort_by(|left, right| left.value.total_cmp(&right.value));
    let percentile_duration = by_value
        .iter()
        .map(|segment| segment.duration_ns)
        .sum::<f64>();
    let percentile = |fraction: f64| {
        let mut cumulative = 0.0;
        for segment in &by_value {
            cumulative += segment.duration_ns;
            if cumulative / percentile_duration >= fraction {
                return segment.value;
            }
        }
        by_value.last().map(|segment| segment.value).unwrap_or(0.0)
    };

    SweepLineStats {
        avg,
        min,
        max,
        p50: percentile(0.50),
        p90: percentile(0.90),
        p95: percentile(0.95),
        p99: percentile(0.99),
        std: variance.sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_curve_has_constant_statistics() {
        let curve = StepFn::new(vec![0.0, 100.0], vec![5.0, 0.0]);
        assert_eq!(
            compute_time_weighted_stats(&curve, 0.0, 100.0),
            SweepLineStats {
                avg: 5.0,
                min: 5.0,
                max: 5.0,
                p50: 5.0,
                p90: 5.0,
                p95: 5.0,
                p99: 5.0,
                std: 0.0,
            }
        );
    }

    #[test]
    fn statistics_are_weighted_by_duration() {
        let curve = StepFn::new(vec![0.0, 80.0, 100.0], vec![2.0, 10.0, 0.0]);
        let stats = compute_time_weighted_stats(&curve, 0.0, 100.0);
        assert!((stats.avg - 3.6).abs() < 1e-12);
        assert!((stats.std - 3.2).abs() < 1e-12);
    }

    #[test]
    fn percentile_uses_duration_cdf() {
        let curve = StepFn::new(vec![0.0, 900.0, 1000.0], vec![1.0, 100.0, 0.0]);
        let stats = compute_time_weighted_stats(&curve, 0.0, 1000.0);
        assert_eq!(stats.p50, 1.0);
        assert_eq!(stats.p90, 1.0);
        assert_eq!(stats.p95, 100.0);
        assert_eq!(stats.p99, 100.0);
    }

    #[test]
    fn clipping_carries_predecessor_state() {
        let curve = StepFn::new(vec![0.0, 50.0, 100.0], vec![1.0, 5.0, 0.0]);
        let stats = compute_time_weighted_stats(&curve, 50.0, 100.0);
        assert_eq!(stats.avg, 5.0);
        assert_eq!(stats.min, 5.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn active_statistics_exclude_idle_tail() {
        let rate = StepFn::new(vec![0.0, 50.0], vec![100.0, 0.0]);
        let mask = StepFn::new(vec![0.0, 50.0], vec![1.0, 0.0]);
        let full = compute_time_weighted_stats(&rate, 0.0, 100.0);
        let active = compute_active_weighted_stats(&rate, &mask, 0.0, 100.0);
        assert_eq!(full.avg, 50.0);
        assert_eq!(active.avg, 100.0);
        assert_eq!(active.p99, 100.0);
    }

    #[test]
    fn empty_mask_is_inactive_everywhere() {
        let rate = StepFn::new(vec![0.0, 50.0], vec![100.0, 0.0]);
        assert_eq!(
            compute_active_weighted_stats(&rate, &StepFn::empty(), 0.0, 100.0),
            SweepLineStats::ZERO
        );
    }
}
