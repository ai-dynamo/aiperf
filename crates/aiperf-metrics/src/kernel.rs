// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distribution kernels.
//!
//! The report kernel intentionally uses manual linear interpolation over the fixed
//! genai-perf percentile band. Error-adjusted distributions use nearest-rank so a
//! finite-to-`+inf` boundary never computes `inf - inf`.

use crate::MetricValue;
use serde::Serialize;
use std::collections::BTreeMap;

/// Percentile band used by AIPerf reports.
pub const PERCENTILES: [u32; 9] = [1, 5, 10, 25, 50, 75, 90, 95, 99];

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
    values.sort_by(f64::total_cmp);
    let count = values.len();
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
        tag: tag.into(),
        avg: MetricValue::from_f64(avg, false),
        min: MetricValue::from_f64(values[0], false),
        max: MetricValue::from_f64(values[count - 1], false),
        std: Some(std),
        sum: MetricValue::from_f64(running_sum, false),
        count,
        percentiles,
    })
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
    values.sort_by(f64::total_cmp);
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
        let idx = ((count - 1) as f64 * percentile as f64 / 100.0).round() as usize;
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
    use crate::MetricValue;

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
