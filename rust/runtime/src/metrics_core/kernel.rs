// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distribution kernels.
//!
//! The report kernel intentionally uses manual linear interpolation over the fixed
//! genai-perf percentile band. Error-adjusted distributions use nearest-rank so a
//! finite-to-`+inf` boundary never computes `inf - inf`.

use crate::metrics_core::MetricValue;
use crate::metrics_core::store::TagSketch;
use rustc_hash::FxHashMap;
use serde::Serialize;
use std::collections::BTreeMap;

/// Percentile band used by AIPerf reports.
pub const PERCENTILES: [u32; 9] = [1, 5, 10, 25, 50, 75, 90, 95, 99];

// Trace replay commonly repeats one batch latency millions of times. Beyond
// this bound, probing cardinality costs less than one comparison-sort level.
const LOW_CARDINALITY_LIMIT: usize = 256;

/// Summary statistics for a metric distribution.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DistributionStats {
    /// Stable metric tag used by reports.
    pub tag: String,
    /// Arithmetic average.
    pub avg: MetricValue,
    /// Minimum value.
    pub min: MetricValue,
    /// Maximum value.
    pub max: MetricValue,
    /// Standard deviation; absent for error-adjusted `+inf` bands.
    pub std: Option<f64>,
    /// Sum of present values.
    pub sum: MetricValue,
    /// Number of present values.
    pub count: usize,
    /// Percentiles keyed by integer percentile.
    pub percentiles: BTreeMap<u32, MetricValue>,
}

impl DistributionStats {
    /// Builds report statistics from a bounded-memory [`TagSketch`] instead of the
    /// full value vector. Count, sum, average, min, and max stay exact; the standard
    /// deviation is the streaming Welford estimate; the percentiles are the
    /// t-digest's approximation of the same linear-interpolation band
    /// [`linear_distribution`] computes exactly. Returns `None` for an empty sketch,
    /// preserving the exact path.
    pub fn from_sketch(tag: impl Into<String>, sketch: &TagSketch, ddof: usize) -> Option<Self> {
        let count = sketch.count();
        if count == 0 {
            return None;
        }
        let count = count as usize;
        let sum = sketch.sum();
        let avg = sum / count as f64;
        let quantile_points = PERCENTILES
            .iter()
            .map(|percentile| *percentile as f64 / 100.0)
            .collect::<Vec<_>>();
        let mut percentiles = BTreeMap::new();
        for (percentile, value) in PERCENTILES.iter().zip(sketch.quantiles(&quantile_points)) {
            if let Some(value) = value {
                percentiles.insert(*percentile, MetricValue::from_f64(value, false));
            }
        }
        Some(Self {
            tag: tag.into(),
            avg: MetricValue::from_f64(avg, false),
            min: MetricValue::from_f64(sketch.min(), false),
            max: MetricValue::from_f64(sketch.max(), false),
            std: Some(sketch.std(ddof)),
            sum: MetricValue::from_f64(sum, false),
            count,
            percentiles,
        })
    }

    /// Builds an empty distribution for a tag.
    pub fn empty(tag: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            avg: MetricValue::Absent,
            min: MetricValue::Absent,
            max: MetricValue::Absent,
            std: None,
            sum: MetricValue::Absent,
            count: 0,
            percentiles: BTreeMap::new(),
        }
    }
}

/// Computes report statistics using manual linear interpolation.
pub fn linear_distribution(
    tag: impl Into<String>,
    mut values: Vec<f64>,
    running_sum: f64,
    ddof: usize,
) -> Option<DistributionStats> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    let tag = tag.into();
    if let Some(runs) = low_cardinality_runs(&values) {
        return Some(linear_distribution_from_runs(
            tag,
            &runs,
            values.len(),
            running_sum,
            ddof,
        ));
    }
    values.sort_unstable_by(f64::total_cmp);
    let count = values.len();
    let running_sum = if running_sum.is_finite() {
        running_sum
    } else {
        values.iter().sum()
    };
    let avg = running_sum / count as f64;
    let mut percentiles = BTreeMap::new();
    for percentile in PERCENTILES {
        let virtual_idx = percentile as f64 / 100.0 * (count - 1) as f64;
        let lo = virtual_idx.floor() as usize;
        let hi = (lo + 1).min(count - 1);
        let frac = virtual_idx - lo as f64;
        let value = values[lo] + frac * (values[hi] - values[lo]);
        percentiles.insert(percentile, MetricValue::from_f64(value, false));
    }
    let denom = count.saturating_sub(ddof);
    let std = if denom == 0 {
        0.0
    } else {
        let variance = values
            .iter()
            .map(|value| {
                let diff = *value - avg;
                diff * diff
            })
            .sum::<f64>()
            / denom as f64;
        variance.sqrt()
    };
    Some(DistributionStats {
        tag,
        avg: MetricValue::from_f64(avg, false),
        min: MetricValue::from_f64(values[0], false),
        max: MetricValue::from_f64(values[count - 1], false),
        std: Some(std),
        sum: MetricValue::from_f64(running_sum, false),
        count,
        percentiles,
    })
}

fn low_cardinality_runs(values: &[f64]) -> Option<Vec<(f64, usize)>> {
    let first = values[0];
    if values[1..]
        .iter()
        .all(|value| value.to_bits() == first.to_bits())
    {
        return Some(vec![(first, values.len())]);
    }

    let mut counts = FxHashMap::with_capacity_and_hasher(LOW_CARDINALITY_LIMIT, Default::default());
    for &value in values {
        let bits = value.to_bits();
        if let Some(count) = counts.get_mut(&bits) {
            *count += 1;
            continue;
        }
        if counts.len() == LOW_CARDINALITY_LIMIT {
            return None;
        }
        counts.insert(bits, 1);
    }
    let mut runs = counts
        .into_iter()
        .map(|(bits, count)| (f64::from_bits(bits), count))
        .collect::<Vec<_>>();
    runs.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));
    Some(runs)
}

fn linear_distribution_from_runs(
    tag: String,
    runs: &[(f64, usize)],
    count: usize,
    running_sum: f64,
    ddof: usize,
) -> DistributionStats {
    let running_sum = if running_sum.is_finite() {
        running_sum
    } else {
        let mut sum = 0.0;
        for &(value, repetitions) in runs {
            for _ in 0..repetitions {
                sum += value;
            }
        }
        sum
    };
    let avg = running_sum / count as f64;
    let mut percentiles = BTreeMap::new();
    for percentile in PERCENTILES {
        let virtual_idx = percentile as f64 / 100.0 * (count - 1) as f64;
        let lo = virtual_idx.floor() as usize;
        let hi = (lo + 1).min(count - 1);
        let frac = virtual_idx - lo as f64;
        let lo_value = run_value_at(runs, lo);
        let hi_value = run_value_at(runs, hi);
        let value = lo_value + frac * (hi_value - lo_value);
        percentiles.insert(percentile, MetricValue::from_f64(value, false));
    }
    let denom = count.saturating_sub(ddof);
    let std = if denom == 0 {
        0.0
    } else {
        let mut squared_deviations = 0.0;
        for &(value, repetitions) in runs {
            let diff = value - avg;
            let squared = diff * diff;
            // Replaying each occurrence preserves the old sorted-vector
            // addition order exactly; multiplying would change low ULPs.
            for _ in 0..repetitions {
                squared_deviations += squared;
            }
        }
        (squared_deviations / denom as f64).sqrt()
    };
    DistributionStats {
        tag,
        avg: MetricValue::from_f64(avg, false),
        min: MetricValue::from_f64(runs[0].0, false),
        max: MetricValue::from_f64(runs[runs.len() - 1].0, false),
        std: Some(std),
        sum: MetricValue::from_f64(running_sum, false),
        count,
        percentiles,
    }
}

fn run_value_at(runs: &[(f64, usize)], rank: usize) -> f64 {
    let mut seen = 0;
    for &(value, repetitions) in runs {
        seen += repetitions;
        if rank < seen {
            return value;
        }
    }
    runs[runs.len() - 1].0
}

/// Computes nearest-rank statistics, preserving positive infinity when requested.
pub fn nearest_distribution(
    tag: impl Into<String>,
    mut values: Vec<f64>,
    running_sum: f64,
    allow_pos_inf: bool,
) -> Option<DistributionStats> {
    values.retain(|value| {
        value.is_finite() || (allow_pos_inf && value.is_infinite() && value.is_sign_positive())
    });
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f64::total_cmp);
    let count = values.len();
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let avg = if allow_pos_inf && values.iter().any(|value| value.is_infinite()) {
        f64::INFINITY
    } else {
        running_sum / count as f64
    };
    let mut percentiles = BTreeMap::new();
    for percentile in PERCENTILES {
        let idx = ((count - 1) as f64 * percentile as f64 / 100.0).round_ties_even() as usize;
        percentiles.insert(
            percentile,
            MetricValue::from_f64(values[idx.min(count - 1)], allow_pos_inf),
        );
    }
    let std = if allow_pos_inf && values.iter().any(|value| value.is_infinite()) {
        None
    } else {
        let avg_for_std = avg;
        let variance = finite
            .iter()
            .map(|value| {
                let diff = *value - avg_for_std;
                diff * diff
            })
            .sum::<f64>()
            / finite.len() as f64;
        Some(variance.sqrt())
    };
    Some(DistributionStats {
        tag: tag.into(),
        avg: MetricValue::from_f64(avg, allow_pos_inf),
        min: MetricValue::from_f64(values[0], allow_pos_inf),
        max: MetricValue::from_f64(values[count - 1], allow_pos_inf),
        std,
        sum: MetricValue::from_f64(
            if allow_pos_inf && values.iter().any(|value| value.is_infinite()) {
                f64::INFINITY
            } else {
                running_sum
            },
            allow_pos_inf,
        ),
        count,
        percentiles,
    })
}

#[cfg(test)]
mod tests {
    use super::{linear_distribution, nearest_distribution};
    use crate::metrics_core::MetricValue;

    #[test]
    fn linear_kernel_uses_manual_percentile_band_and_population_std() {
        let stats = linear_distribution("latency", vec![1.0, 2.0, 3.0, 4.0], 10.0, 0).unwrap();
        assert_eq!(stats.count, 4);
        assert_eq!(stats.avg, MetricValue::Finite(2.5));
        assert_eq!(
            stats.percentiles.get(&50).copied(),
            Some(MetricValue::Finite(2.5))
        );
        assert!((stats.std.unwrap() - 1.118033988749895).abs() < 1e-12);
    }

    #[test]
    fn low_cardinality_kernel_replays_sorted_repetitions_exactly() {
        let values = vec![3.0, 1.0, 2.0, 1.0, 3.0, 1.0];
        let stats = linear_distribution("repeated", values.clone(), 11.0, 0).unwrap();
        let average = 11.0 / 6.0;
        let sorted = [1.0, 1.0, 1.0, 2.0, 3.0, 3.0];
        let expected_std = (sorted
            .iter()
            .map(|value| {
                let difference = *value - average;
                difference * difference
            })
            .sum::<f64>()
            / 6.0)
            .sqrt();

        assert_eq!(stats.std.unwrap().to_bits(), expected_std.to_bits());
        assert_eq!(stats.percentiles.get(&50), Some(&MetricValue::Finite(1.5)));

        let recomputed = linear_distribution("repeated", values, f64::NAN, 0).unwrap();
        assert_eq!(recomputed.sum, MetricValue::Finite(sorted.iter().sum()));
    }

    #[test]
    fn nearest_kernel_keeps_positive_infinity_for_adjusted_band() {
        let stats =
            nearest_distribution("adj_latency", vec![10.0, 20.0, f64::INFINITY], 30.0, true)
                .unwrap();
        assert_eq!(stats.std, None);
        assert_eq!(stats.max, MetricValue::PosInf);
        assert_eq!(
            stats.percentiles.get(&99).copied(),
            Some(MetricValue::PosInf)
        );
    }
}
