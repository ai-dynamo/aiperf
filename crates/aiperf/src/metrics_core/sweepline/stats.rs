// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Duration-weighted statistics over sweep-line step functions.
//!
//! Covers clipping, active masks, and duration-CDF percentiles.

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
/// before the first event.
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
    let mut segments = Vec::with_capacity(hi.saturating_sub(lo) + 1);
    let mut segment_start = window_start_ns;
    let mut segment_value = if lo > 0 { values[lo - 1] } else { 0.0 };
    for (&event_start, &event_value) in timestamps[lo..hi].iter().zip(&values[lo..hi]) {
        let start = segment_start.max(window_start_ns);
        let end = event_start.min(window_end_ns);
        let duration_ns = (end - start).max(0.0);
        if duration_ns > 0.0 {
            segments.push(ClippedSegment {
                duration_ns,
                value: segment_value,
            });
        }
        segment_start = event_start;
        segment_value = event_value;
    }
    let duration_ns = (window_end_ns - segment_start.max(window_start_ns)).max(0.0);
    if duration_ns > 0.0 {
        segments.push(ClippedSegment {
            duration_ns,
            value: segment_value,
        });
    }
    segments
}

/// Computes duration-weighted statistics over a clipped step function.
///
/// Average and population variance use the full window span, including idle
/// zero-valued segments. Percentiles sort by value and locate the first
/// cumulative-duration fraction at or above 50/90/95/99 percent.
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
    weighted_stats(segments, total_duration)
}

/// Computes duration-weighted rate statistics only while `mask` is positive.
///
/// Idle segments do not enter the average, variance, or percentile CDF. The
/// merged grid includes only in-window rate/mask events plus exact window edges.
pub fn compute_active_weighted_stats(
    rate: &StepFn,
    mask: &StepFn,
    window_start_ns: f64,
    window_end_ns: f64,
) -> SweepLineStats {
    if window_end_ns <= window_start_ns || rate.is_empty() || mask.is_empty() {
        return SweepLineStats::ZERO;
    }

    let grid = merge_events_inside(
        rate.timestamps_ns(),
        mask.timestamps_ns(),
        window_start_ns,
        window_end_ns,
    );
    if grid.len() < 2 {
        return SweepLineStats::ZERO;
    }

    let mut segments = Vec::with_capacity(grid.len() - 1);
    let mut rate_cursor = StepCursor::new(rate, window_start_ns);
    let mut mask_cursor = StepCursor::new(mask, window_start_ns);
    for pair in grid.windows(2) {
        let duration_ns = pair[1] - pair[0];
        let rate_value = rate_cursor.value_at(pair[0]);
        let mask_value = mask_cursor.value_at(pair[0]);
        if duration_ns > 0.0 && mask_value > 0.0 {
            segments.push(ClippedSegment {
                duration_ns,
                value: rate_value,
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
    weighted_stats(segments, active_duration)
}

/// Computes effective statistics for `numerator / denominator` without retaining
/// the materialized merged-grid step function.
///
/// The boundary walk is exactly [`StepFn::divide`] followed by
/// [`compute_time_weighted_stats`]: all same-timestamp updates are applied before
/// the following segment and non-positive denominators yield zero.
pub fn compute_divided_time_weighted_stats(
    numerator: &StepFn,
    denominator: &StepFn,
    window_start_ns: f64,
    window_end_ns: f64,
) -> SweepLineStats {
    compute_divided_weighted_stats(numerator, denominator, window_start_ns, window_end_ns).0
}

/// Computes active statistics for `numerator / denominator` without retaining
/// the materialized ratio curve.
pub fn compute_divided_active_weighted_stats(
    numerator: &StepFn,
    denominator: &StepFn,
    window_start_ns: f64,
    window_end_ns: f64,
) -> SweepLineStats {
    compute_divided_weighted_stats(numerator, denominator, window_start_ns, window_end_ns).1
}

/// Computes effective and active statistics for `numerator / denominator` in one
/// merged-boundary walk and one value sort.
///
/// The first result includes the entire window; the second includes only segments
/// with a positive denominator. Computing them together avoids retaining a ratio
/// [`StepFn`] and avoids sorting the same per-user values twice.
pub fn compute_divided_weighted_stats(
    numerator: &StepFn,
    denominator: &StepFn,
    window_start_ns: f64,
    window_end_ns: f64,
) -> (SweepLineStats, SweepLineStats) {
    let total_duration = window_end_ns - window_start_ns;
    if total_duration <= 0.0 || numerator.is_empty() || denominator.is_empty() {
        return (SweepLineStats::ZERO, SweepLineStats::ZERO);
    }
    let mut numerator_index = 0_usize;
    let mut denominator_index = 0_usize;
    let mut numerator_value = 0.0;
    let mut denominator_value = 0.0;
    advance_step_through(
        numerator,
        &mut numerator_index,
        &mut numerator_value,
        window_start_ns,
    );
    advance_step_through(
        denominator,
        &mut denominator_index,
        &mut denominator_value,
        window_start_ns,
    );

    let mut segments = Vec::with_capacity(numerator.len() + denominator.len() + 1);
    let mut segment_start = window_start_ns;
    loop {
        let next_numerator = numerator.timestamps_ns().get(numerator_index).copied();
        let next_denominator = denominator.timestamps_ns().get(denominator_index).copied();
        let next = match (next_numerator, next_denominator) {
            (Some(left), Some(right)) => {
                if left.total_cmp(&right) == std::cmp::Ordering::Greater {
                    right
                } else {
                    left
                }
            }
            (Some(value), None) | (None, Some(value)) => value,
            (None, None) => window_end_ns,
        };
        let segment_end = next.min(window_end_ns);
        let duration_ns = segment_end - segment_start;
        if duration_ns > 0.0 {
            segments.push(DividedSegment {
                // A signed duration records the active mask without expanding this
                // hot temporary from 16 to 24 bytes per event boundary.
                signed_duration_ns: if denominator_value > 0.0 {
                    duration_ns
                } else {
                    -duration_ns
                },
                value: if denominator_value > 0.0 {
                    numerator_value / denominator_value
                } else {
                    0.0
                },
            });
        }
        if next.total_cmp(&window_end_ns) != std::cmp::Ordering::Less {
            break;
        }
        segment_start = next;
        advance_step_through(numerator, &mut numerator_index, &mut numerator_value, next);
        advance_step_through(
            denominator,
            &mut denominator_index,
            &mut denominator_value,
            next,
        );
    }
    divided_weighted_stats(segments, total_duration)
}

#[derive(Debug, Clone, Copy)]
struct DividedSegment {
    signed_duration_ns: f64,
    value: f64,
}

fn divided_weighted_stats(
    mut segments: Vec<DividedSegment>,
    total_duration: f64,
) -> (SweepLineStats, SweepLineStats) {
    if segments.is_empty() {
        return (SweepLineStats::ZERO, SweepLineStats::ZERO);
    }

    let effective_avg = segments
        .iter()
        .map(|segment| segment.value * segment.signed_duration_ns.abs())
        .sum::<f64>()
        / total_duration;
    let active_duration = segments
        .iter()
        .filter(|segment| segment.signed_duration_ns > 0.0)
        .map(|segment| segment.signed_duration_ns)
        .sum::<f64>();
    let active_avg = if active_duration > 0.0 {
        segments
            .iter()
            .filter(|segment| segment.signed_duration_ns > 0.0)
            .map(|segment| segment.value * segment.signed_duration_ns)
            .sum::<f64>()
            / active_duration
    } else {
        0.0
    };
    let effective_min = segments
        .iter()
        .map(|segment| segment.value)
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    let effective_max = segments
        .iter()
        .map(|segment| segment.value)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    let active_min = segments
        .iter()
        .filter(|segment| segment.signed_duration_ns > 0.0)
        .map(|segment| segment.value)
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    let active_max = segments
        .iter()
        .filter(|segment| segment.signed_duration_ns > 0.0)
        .map(|segment| segment.value)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    let effective_variance = segments
        .iter()
        .map(|segment| {
            let delta = segment.value - effective_avg;
            segment.signed_duration_ns.abs() * delta * delta
        })
        .sum::<f64>()
        / total_duration;
    let active_variance = if active_duration > 0.0 {
        segments
            .iter()
            .filter(|segment| segment.signed_duration_ns > 0.0)
            .map(|segment| {
                let delta = segment.value - active_avg;
                segment.signed_duration_ns * delta * delta
            })
            .sum::<f64>()
            / active_duration
    } else {
        0.0
    };

    segments.sort_unstable_by(|left, right| left.value.total_cmp(&right.value));
    let percentile_duration = segments
        .iter()
        .map(|segment| segment.signed_duration_ns.abs())
        .sum::<f64>();
    let percentile = |fraction: f64, active_only: bool| {
        let denominator = if active_only {
            active_duration
        } else {
            percentile_duration
        };
        let mut cumulative = 0.0;
        for segment in &segments {
            if active_only && segment.signed_duration_ns <= 0.0 {
                continue;
            }
            cumulative += segment.signed_duration_ns.abs();
            if cumulative / denominator >= fraction {
                return segment.value;
            }
        }
        0.0
    };

    let effective = SweepLineStats {
        avg: effective_avg,
        min: effective_min,
        max: effective_max,
        p50: percentile(0.50, false),
        p90: percentile(0.90, false),
        p95: percentile(0.95, false),
        p99: percentile(0.99, false),
        std: effective_variance.sqrt(),
    };
    let active = if active_duration > 0.0 {
        SweepLineStats {
            avg: active_avg,
            min: active_min,
            max: active_max,
            p50: percentile(0.50, true),
            p90: percentile(0.90, true),
            p95: percentile(0.95, true),
            p99: percentile(0.99, true),
            std: active_variance.sqrt(),
        }
    } else {
        SweepLineStats::ZERO
    };
    (effective, active)
}

fn advance_step_through(curve: &StepFn, index: &mut usize, value: &mut f64, timestamp: f64) {
    while *index < curve.len()
        && curve.timestamps_ns()[*index].total_cmp(&timestamp) != std::cmp::Ordering::Greater
    {
        *value = curve.values()[*index];
        *index += 1;
    }
}

fn merge_events_inside(left: &[f64], right: &[f64], start: f64, end: f64) -> Vec<f64> {
    let left = &left[upper_bound(left, start)..lower_bound(left, end)];
    let right = &right[upper_bound(right, start)..lower_bound(right, end)];
    let mut grid = Vec::with_capacity(left.len() + right.len() + 2);
    grid.push(start);
    let mut left_index = 0;
    let mut right_index = 0;
    while left_index < left.len() || right_index < right.len() {
        let take_left = right_index == right.len()
            || (left_index < left.len()
                && left[left_index].total_cmp(&right[right_index]) != std::cmp::Ordering::Greater);
        let timestamp = if take_left {
            let timestamp = left[left_index];
            left_index += 1;
            timestamp
        } else {
            let timestamp = right[right_index];
            right_index += 1;
            timestamp
        };
        if grid.last().is_none_or(|previous| {
            *previous != timestamp && !(previous.is_nan() && timestamp.is_nan())
        }) {
            grid.push(timestamp);
        }
    }
    if grid
        .last()
        .is_none_or(|previous| *previous != end && !(previous.is_nan() && end.is_nan()))
    {
        grid.push(end);
    }
    grid
}

struct StepCursor<'a> {
    timestamps: &'a [f64],
    values: &'a [f64],
    next: usize,
    current: f64,
}

impl<'a> StepCursor<'a> {
    fn new(curve: &'a StepFn, start: f64) -> Self {
        let next = upper_bound(curve.timestamps_ns(), start);
        let current = next
            .checked_sub(1)
            .and_then(|index| curve.values().get(index))
            .copied()
            .unwrap_or(0.0);
        Self {
            timestamps: curve.timestamps_ns(),
            values: curve.values(),
            next,
            current,
        }
    }

    fn value_at(&mut self, timestamp: f64) -> f64 {
        while self.next < self.timestamps.len()
            && self.timestamps[self.next].total_cmp(&timestamp) != std::cmp::Ordering::Greater
        {
            self.current = self.values[self.next];
            self.next += 1;
        }
        self.current
    }
}

fn weighted_stats(mut segments: Vec<ClippedSegment>, denominator_duration: f64) -> SweepLineStats {
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

    segments.sort_unstable_by(|left, right| left.value.total_cmp(&right.value));
    let percentile_duration = segments
        .iter()
        .map(|segment| segment.duration_ns)
        .sum::<f64>();
    let percentile = |fraction: f64| {
        let mut cumulative = 0.0;
        for segment in &segments {
            cumulative += segment.duration_ns;
            if cumulative / percentile_duration >= fraction {
                return segment.value;
            }
        }
        segments.last().map(|segment| segment.value).unwrap_or(0.0)
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

    #[test]
    fn divided_statistics_match_the_materialized_ratio_exactly() {
        let numerator = StepFn::new(
            vec![-20.0, 0.0, 10.0, 10.0, 40.0, 75.0, 120.0],
            vec![2.0, 8.0, 10.0, 12.0, 3.0, 9.0, 0.0],
        );
        let denominator = StepFn::new(
            vec![-10.0, 0.0, 25.0, 40.0, 90.0, 120.0],
            vec![0.0, 2.0, 4.0, 0.0, 3.0, 0.0],
        );
        let ratio = numerator.divide(&denominator);

        for (window_start_ns, window_end_ns) in [
            (-30.0, -15.0),
            (-5.0, 130.0),
            (0.0, 120.0),
            (10.0, 40.0),
            (17.0, 83.0),
            (130.0, 140.0),
        ] {
            assert_eq!(
                compute_divided_time_weighted_stats(
                    &numerator,
                    &denominator,
                    window_start_ns,
                    window_end_ns,
                ),
                compute_time_weighted_stats(&ratio, window_start_ns, window_end_ns),
            );
            assert_eq!(
                compute_divided_active_weighted_stats(
                    &numerator,
                    &denominator,
                    window_start_ns,
                    window_end_ns,
                ),
                compute_active_weighted_stats(&ratio, &denominator, window_start_ns, window_end_ns,),
            );
        }
    }
}
